import time
import numpy as np
import mujoco
import mujoco.viewer
import json

SCALE = 1.0  # scale down the viewpoints

with open('optimized_viewpoints.json', 'r') as f:
    file = json.load(f)

WPS = file.get("waypoints", [])

if not WPS:
    raise RuntimeError("optimized_viewpoints.json has no 'waypoints' entries")

METRICS = file.get("metrics", {})
if not METRICS:
    raise RuntimeError("optimized_viewpoints.json has no 'metrics' entries")

DELTA_VS = np.asarray(METRICS["delta_v_list"], dtype=float)   # shape: (N_legs, 3)
T_LEGS   = np.asarray(METRICS["t_leg_list"], dtype=float)     # shape: (N_legs,)
NUM_LEGS = len(DELTA_VS)

# Use mass from optimizer if available, else fall back
M_ROBOT  = float(METRICS.get("mass0", 25.0))

VP_POS   = np.asarray([wp["pos"] for wp in WPS], dtype=float)
VP_VEL   = np.asarray([wp.get("vel",   [0.0, 0.0, 0.0]) for wp in WPS], dtype=float)
VP_QUAT  = np.asarray([wp.get("quat",  [1.0, 0.0, 0.0, 0.0]) for wp in WPS], dtype=float)
VP_OMEGA = np.asarray([wp.get("omega", [0.0, 0.0, 0.0]) for wp in WPS], dtype=float)

VIEWPOINTS_POS   = [p * SCALE for p in VP_POS]
VIEWPOINTS_VEL   = [v for v in VP_VEL]      # keep as-is for now
VIEWPOINTS_QUAT  = [q for q in VP_QUAT]
VIEWPOINTS_OMEGA = [w for w in VP_OMEGA]

# ----------------- Mujoco model setup -----------------

model = mujoco.MjModel.from_xml_path('robot.xml')
data  = mujoco.MjData(model)

# Thruster index mapping
def index_map(model):
    act = {}
    names = [
        "thruster_px","thruster_nx","thruster_py","thruster_ny","thruster_pz","thruster_nz",
        "rw_x","rw_y","rw_z"
    ]
    for i, nm in enumerate(names):

        act[nm] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)

    return act

ACT = index_map(model)


# ---- Thruster burn stats ----
THRUSTER_NAMES = ["thruster_px","thruster_nx","thruster_py","thruster_ny","thruster_pz","thruster_nz"]
THRUSTER_IDS   = [ACT[nm] for nm in THRUSTER_NAMES]

FIRE_THRESH = 0.1
burn_time_s = np.zeros(6, dtype=float)   # burn time per thruster
last_ctrl  = np.zeros(model.nu, dtype=float)


# ---------- init the ball pose & velocity ----------
iss_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "iss_free")
iss_qpos_adr = model.jnt_qposadr[iss_jid]
# iss_qvel_adr = model.jnt_dofadr[iss_jid] 


data.qpos[iss_qpos_adr:iss_qpos_adr+3]   = np.array([0.0, 0.0, 0.0])  
# data.qvel[iss_qvel_adr:iss_qvel_adr+3]   = np.array([0.0, 0.0, 0.0])


# ---- IVT leg / burn schedule state ----
leg_idx         = 0            # which Δv leg we’re on
leg_phase       = "burn"       # "burn" or "coast"
leg_phase_start = 0.0          # sim-time when current phase began
BURN_DURATION   = 1.0          # seconds to spread each impulse over (tunable)


mujoco.mj_forward(model, data)

# ---------- robot joint addresses (compute once) ----------
# Find the freejoint/qpos/qvel addresses for the controlled body named "robot"
robot_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
if robot_bid >= 0 and model.body_jntnum[robot_bid] > 0:
    # joint index for first joint on this body
    robot_jid = int(model.body_jntadr[robot_bid])
    robot_qpos_adr = int(model.jnt_qposadr[robot_jid])
    robot_qvel_adr = int(model.jnt_dofadr[robot_jid])

# ---------- simulation parameters ----------
DT                  = model.opt.timestep   # default 0.002
RUN_SPEED           = 1.0                  # 1.0 = real time (wall clock pacing)
CONTROL_HZ          = 500                  # low-level control loop frequency (Hz)
RENDER_HZ           = 60                   # viewer frame cap (Hz)
MAX_STEPS_PER_FRAME = 64                   # cap physics steps per UI frame
IDLE_SLEEP          = 0.0005               # short sleep to reduce CPU usage

def quat_to_R(q):
    """q=[w,x,y,z] -> R(world <- body)."""
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)]
    ])
    return R

# equivalent to q inverse
def quat_conj(q):
    w,x,y,z = q
    return np.array([w, -x, -y, -z])

# quaternion multiplicationS
def quat_mul(qa, qb):
    wa,xa,ya,za = qa
    wb,xb,yb,zb = qb
    return np.array([
        wa*wb - xa*xb - ya*yb - za*zb,
        wa*xb + xa*wb + ya*zb - za*yb,
        wa*yb - xa*zb + ya*wb + za*xb,
        wa*zb + xa*yb - ya*xb + za*wb
    ])

def attitude_error_vec(q_current, q_des):
    """ Compute attitude error vector from current to desired quaternion."""
    q_err = quat_mul(q_des, quat_conj(q_current))

    # Ensure shortest rotation
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


def step_until(target_sim_time):
    """Advance physics up to (but not beyond) target_sim_time, capped per frame."""
    steps = 0
    while data.time + 1e-12 < target_sim_time and steps < MAX_STEPS_PER_FRAME:
        mujoco.mj_step(model, data)
        steps += 1
    return steps


def ref(inp):

    arr = np.asarray(inp, dtype=float).reshape(-1)

    pW_des  = arr[0:3].copy()
    vW_des  = arr[3:6].copy()

    q = arr[6:10].copy()
    n = np.linalg.norm(q)

    q = q / n

    if q[0] < 0:
        q = -q
    
    qWB_des = q
    wB_des  = arr[10:13].copy()

    return pW_des, vW_des, qWB_des, wB_des


# allocate body-frame force to thrusters
def set_axis_with_min(u, pos_name, neg_name, F_axis, Fsat=10.0, min_fire=0.0):
    if abs(F_axis) < min_fire:
        return
    pos_id, neg_id = ACT[pos_name], ACT[neg_name]
    if F_axis >= 0.0:
        if pos_id is not None: u[pos_id] = min(F_axis, Fsat)
    else:
        if neg_id is not None: u[neg_id] = min(-F_axis, Fsat)


def controller(model, data, q_goal, w_goal, F_B_cmd):
    """
    Low-level controller:
      - Translation: apply body-frame force F_B_cmd using thrusters
      - Rotation: PD on attitude and angular rate using reaction wheels
    """
    u = np.zeros(model.nu)

    # ------------ read state ------------
    # robot pose
    qWB = np.array(data.qpos[robot_qpos_adr+3 : robot_qpos_adr+7])   # [w,x,y,z], world<-body

    # velocities
    wW  = np.array(data.qvel[robot_qvel_adr+3 : robot_qvel_adr+6])   # angular vel in world

    # rotation matrix
    R_WB = quat_to_R(qWB)             # world <- body
    wB   = R_WB.T @ wW                # angular vel in body frame

    # ------------ translation: use F_B_cmd directly ------------
    Fmax_axis = 20.0
    set_axis_with_min(u, "thruster_px","thruster_nx", F_B_cmd[0], Fsat=Fmax_axis)
    set_axis_with_min(u, "thruster_py","thruster_ny", F_B_cmd[1], Fsat=Fmax_axis)
    set_axis_with_min(u, "thruster_pz","thruster_nz", F_B_cmd[2], Fsat=Fmax_axis)

    # ------------ rotation: PD on attitude and rate ------------
    # attitude error from current to desired
    eR = attitude_error_vec(qWB, q_goal)   # body-frame error vector

    K_R = np.array([2.0, 2.0, 2.0])
    K_w = np.array([0.6, 0.6, 0.6])

    tauB = -K_R * eR - K_w * (wB - w_goal)
    tauB = np.clip(tauB, -5.0, 5.0)

    if ACT["rw_x"] is not None: u[ACT["rw_x"]] = tauB[0]
    if ACT["rw_y"] is not None: u[ACT["rw_y"]] = tauB[1]
    if ACT["rw_z"] is not None: u[ACT["rw_z"]] = tauB[2]

    return u


# ---- viewpoint following state ----
vp_idx        = 0                  # reached VP index
POS_DONE_TOL  = 0.2
VEL_DONE_TOL  = 0.2
STOP_TIME    = 0.0
stop_flag  = None

def reached(p, v, p_goal, v_goal):
    ep = p - p_goal
    ev = v - v_goal
    return (np.linalg.norm(ep) < POS_DONE_TOL) and (np.linalg.norm(ev) < VEL_DONE_TOL)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Camera setup
    viewer.cam.distance  = 100.0
    viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat    = [0, 0, 0]
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -45

    # Wall-clock anchors
    t0_wall = time.perf_counter()
    t0_sim  = data.time

    # Next trigger times (wall clock)
    next_ctrl   = t0_wall
    next_render = t0_wall

    while viewer.is_running():
        now = time.perf_counter()

        # (1) Control tick
        if now >= next_ctrl:
            t_sim = data.time

            # ------------ IVT burn / coast schedule for translation ------------
            F_B_cmd = np.zeros(3)

            if leg_idx < NUM_LEGS:
                if leg_phase == "burn":
                    # initialize start time on first entry
                    if leg_phase_start == 0.0:
                        leg_phase_start = t_sim

                    dt_phase = t_sim - leg_phase_start

                    if dt_phase < BURN_DURATION:
                        # target Δv in WORLD frame for this leg
                        delta_v_world = DELTA_VS[leg_idx]      # shape (3,)

                        # average acceleration over burn window
                        a_world = delta_v_world / BURN_DURATION
                        F_world = M_ROBOT * a_world

                        # convert to body frame using current attitude
                        qWB = np.array(data.qpos[robot_qpos_adr+3 : robot_qpos_adr+7])
                        R_WB = quat_to_R(qWB)                   # world <- body
                        F_B_cmd = R_WB.T @ F_world              # body-frame force
                    else:
                        # done burning this impulse, start coasting
                        leg_phase = "coast"
                        leg_phase_start = t_sim

                elif leg_phase == "coast":
                    dt_phase = t_sim - leg_phase_start
                    # coast for the scheduled leg time from the optimizer
                    if dt_phase >= T_LEGS[leg_idx]:
                        # advance to next leg, back to burn
                        leg_idx += 1
                        leg_phase = "burn"
                        leg_phase_start = 0.0

            # ------------ Attitude target from waypoints ------------
            # Map leg index to a waypoint index; waypoints are usually N_legs+1 long.
            att_idx = min(leg_idx, len(VIEWPOINTS_QUAT) - 1)
            q_goal = VIEWPOINTS_QUAT[att_idx]
            w_goal = VIEWPOINTS_OMEGA[att_idx]

            # Low-level control using reaction wheels + thrusters
            u = controller(model, data, q_goal, w_goal, F_B_cmd)

            data.ctrl[:] = u
            last_ctrl = u.copy()

            next_ctrl += 1.0 / CONTROL_HZ
            # Prevent excessive drift if the system lags
            if now > next_ctrl + 2.0 / CONTROL_HZ:
                next_ctrl = now + 1.0 / CONTROL_HZ

        # (2) Physics stepping: chase the wall-clock target without overshoot
        target_sim_time = t0_sim + (now - t0_wall) * RUN_SPEED
        steps = step_until(target_sim_time)

        if steps > 0:
            dt_sum = steps * DT
            # record thruster burn time
            for i, aid in enumerate(THRUSTER_IDS):
                if aid >= 0 and last_ctrl[aid] > FIRE_THRESH:
                    burn_time_s[i] += dt_sum

        print("\n=== Thruster usage summary ===")
        for i, nm in enumerate(THRUSTER_NAMES):
            print(f"{nm}  time = {burn_time_s[i]:8.3f} s")

        # (3) Rendering
        if now >= next_render:
            viewer.sync()
            next_render += 1.0 / RENDER_HZ
            if now > next_render + 2.0 / RENDER_HZ:
                next_render = now + 1.0 / RENDER_HZ

        time.sleep(IDLE_SLEEP)

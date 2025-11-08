import time
import numpy as np
import mujoco
import mujoco.viewer

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

# ---------- init the ball pose & velocity ----------
ball_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
ball_qpos_adr = model.jnt_qposadr[ball_jid]   
ball_qvel_adr = model.jnt_dofadr[ball_jid]    


data.qpos[ball_qpos_adr:ball_qpos_adr+3]   = np.array([1.8, 0.2, 0.2])  
data.qvel[ball_qvel_adr:ball_qvel_adr+3]   = np.array([0.0, 0.0, 0.0])  



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
    """
    Compute attitude error vector from current to desired quaternion.
    """
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

# --- reference: desired pose/velocity (world frame) ---
def ref(input):

    pW_des  = np.zeros(3)                 # expected position
    # pW_des  = np.array([1.0, 1.0, 1.0])
    vW_des  = np.zeros(3)                 # expected velocity
    qWB_des = np.array([0.70710678, 0.70710678, 0.0, 0.0])  # expected quaternion
    # qWB_des = np.array([1.0, 0.0, 0.0, 0.0])
    wB_des  = np.zeros(3)                 # expected angular velocity
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

def controller(model, data, input):
    u = np.zeros(model.nu)

    # ------------ read state ------------
    # Read robot pose/vel using its joint qpos/qvel addresses
    pW  = np.array(data.qpos[robot_qpos_adr:robot_qpos_adr+3])     # position in world
    qWB = np.array(data.qpos[robot_qpos_adr+3:robot_qpos_adr+7])     # quaternion [w,x,y,z] (world <- body)
    vW  = np.array(data.qvel[robot_qvel_adr+3:robot_qvel_adr+6])     # linear vel in world
    wW  = np.array(data.qvel[robot_qvel_adr:robot_qvel_adr+3])       # angular vel in world
   
    R_WB = quat_to_R(qWB)
    wB   = R_WB.T @ wW                 # angular vel in body frame

    # ------------ references ------------
    pW_des, vW_des, qWB_des, wB_des = ref(input)

   # ------------ translation: cascaded pos->vel->acc (with limits) ------------


    xW = (pW - pW_des)
    vW = (vW - vW_des)

    POS_TOL = 2e-3   # 2 mm
    VEL_TOL = 2e-3   # 2 mm/s
    xW = np.where(np.abs(xW) < POS_TOL, 0.0, xW)
    vW = np.where(np.abs(vW) < VEL_TOL, 0.0, vW)

    Fmax_axis = 20.0
    Kp_pos    = 100.0
    Kv_pos    = 100.0

    F_W = -Kp_pos * xW - Kv_pos * vW
    F_B = R_WB.T @ F_W
    


    set_axis_with_min(u, "thruster_px","thruster_nx", F_B[0], Fsat=Fmax_axis)
    set_axis_with_min(u, "thruster_py","thruster_ny", F_B[1], Fsat=Fmax_axis)
    set_axis_with_min(u, "thruster_pz","thruster_nz", F_B[2], Fsat=Fmax_axis)


    # ------------ attitude PD (body) -> reaction wheel torques ------------
    # reaction wheel torque control gains
    # Reduce P slightly and increase D for better damping of oscillations
    K_R = np.array([20.0, 20.0, 20.0])   # roll/pitch/yaw P
    K_w = np.array([20.0, 20.0, 20.0])   # roll/pitch/yaw D

    eR   = attitude_error_vec(qWB, qWB_des)              # attitude error vector
    tauB = -K_R*eR - K_w*(wB - wB_des)                   # desired body torque
    # tauB = np.array([0.0, 0.0, 0.0])

    # clip to motor control limits (as set in robot.xml: ±2.0)
    tauB = np.clip(tauB, -5.0, 5.0)

    if ACT["rw_x"] is not None: u[ACT["rw_x"]] = tauB[0]
    if ACT["rw_y"] is not None: u[ACT["rw_y"]] = tauB[1]
    if ACT["rw_z"] is not None: u[ACT["rw_z"]] = tauB[2]

    return u


with mujoco.viewer.launch_passive(model, data) as viewer:
    # Camera setup
    viewer.cam.distance  = 10.0
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
            # Use simulation time as the controller’s independent variable for stability
            t_sim = data.time
            u = controller(model, data, t_sim)
            data.ctrl[:] = u

            next_ctrl += 1.0 / CONTROL_HZ
            # Prevent excessive drift if the system lags
            if now > next_ctrl + 2.0 / CONTROL_HZ:
                next_ctrl = now + 1.0 / CONTROL_HZ

        # (2) Physics stepping: chase the wall-clock target without overshoot
        target_sim_time = t0_sim + (now - t0_wall) * RUN_SPEED
        step_until(target_sim_time)

        # (3) Rendering
        if now >= next_render:
            viewer.sync()
            next_render += 1.0 / RENDER_HZ
            if now > next_render + 2.0 / RENDER_HZ:
                next_render = now + 1.0 / RENDER_HZ

        time.sleep(IDLE_SLEEP)
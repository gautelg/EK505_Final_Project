import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('robot.xml')
data  = mujoco.MjData(model)

# Thruster index mapping
THRUST_IDX = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8}

DT                  = model.opt.timestep   # default 0.002
RUN_SPEED           = 1.0                  # 1.0 = real time (wall clock pacing)
CONTROL_HZ          = 200                  # low-level control loop frequency (Hz)
RENDER_HZ           = 60                   # viewer frame cap (Hz)
MAX_STEPS_PER_FRAME = 64                   # cap physics steps per UI frame
IDLE_SLEEP          = 0.0005               # short sleep to reduce CPU usage

def step_until(target_sim_time):
    """Advance physics up to (but not beyond) target_sim_time, capped per frame."""
    steps = 0
    while data.time + 1e-12 < target_sim_time and steps < MAX_STEPS_PER_FRAME:
        mujoco.mj_step(model, data)
        steps += 1
    return steps

#  Example input : time-based reference generator
def reference(t):

    # Example: 0–2s -> thruster 1 = +6; 2–5s -> thruster 1 = +4 and thruster 2 = +3; else 0.
    if 0.0 <= t < 2.0:
        return {1: 8.0}
    elif 2.0 <= t < 6.0:
        return {1: 0.0, 2: 8.0}
    elif 6.0 <= t < 8.0:
        return {1: 8.0, 2: 0.0}
    elif 8.0 <= t < 10.0:
        return {1: 0.0, 2: 8.0, 3:0.0, 4: 8.0, 5:0.0, 6: 8.0}
    elif 10.0 <= t < 14.0:
        return {1: 8.0, 2: 0.0, 3:8.0, 4: 0.0, 5:8.0, 6: 0.0}
    elif 14.0 <= t < 16.0:
        return {1: 0.0, 2: 8.0, 3:0.0, 4: 8.0, 5:0.0, 6: 8.0}
    elif 16.0 <= t < 20.0:
        return {7: 250.0}
    elif 18.0 <= t < 26.0:
        return {7: -250.0}
    elif 26.0 <= t < 30.0:
        return {7: 250.0}
    elif 30.0 <= t < 40.0:
        return {7: 0.0}
    # elif 24.0 <= t < 26.0:
    #     return {7: 5.0, 8: 5.0, 9:5.0}
    else:
        return {}

# Controller
def controller(model, data, t):

    u = np.zeros(model.nu)

    # (1) Feedforward / open-loop control
    ff = reference(t)
    for thr_id, force in ff.items():
        u[THRUST_IDX[thr_id]] = force

    # (2) Feedback (example placeholders — replace with real logic)
    #   err_roll, err_pitch, err_yaw = estimate_attitude_error(data)  
    #   derr_roll, derr_pitch, derr_yaw = estimate_rate(data)         
    #   Kp, Kd = 2.0, 0.5
    #   # assuming 7,8,9 are torque channels:
    #   u[THRUST_IDX[7]] += Kp*err_roll  + Kd*derr_roll
    #   u[THRUST_IDX[8]] += Kp*err_pitch + Kd*derr_pitch
    #   u[THRUST_IDX[9]] += Kp*err_yaw   + Kd*derr_yaw

    # (3) Saturation / rate limits 
    # SAT = 10.0
    # u = np.clip(u, -SAT, SAT)

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

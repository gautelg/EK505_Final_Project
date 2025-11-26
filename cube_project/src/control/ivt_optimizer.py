# src/control/ivt_optimizer.py

import numpy as np
import logging

import casadi as ca

def propagate_leg(
    r_i: np.ndarray,
    v_i: np.ndarray,
    delta_v_i: np.ndarray,
    delta_t_i: float,
    model: str = "simple",
    cw_params: dict = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate one leg of the trajectory.

    Parameters
    ----------
    r_i : (3,)
        Position at the start of the leg.
    v_i : (3,)
        Velocity at the start of the leg.
    delta_v_i : (3,)
        Impulsive velocity change applied at the START of the leg.
    delta_t_i : float
        Duration of the leg [s].
    model : str
        Dynamics model. Currently supports:
          - "simple": ṙ = v, v̇ = 0 between impulses
        "cw" can be added later without changing the OCP structure.
    cw_params : dict
        Reserved for future Clohessy-Wiltshire implementation
        (e.g., {"n": mean_motion}).

    Returns
    -------
    r_next : (3,)
        Position at the end of the leg.
    v_next : (3,)
        Velocity at the end of the leg.
    """
    r_i = np.asarray(r_i, dtype=float).reshape(3)
    v_i = np.asarray(v_i, dtype=float).reshape(3)
    delta_v_i = np.asarray(delta_v_i, dtype=float).reshape(3)

    if delta_t_i < 0.0:
        raise ValueError("delta_t_i must be non-negative")

    if model == "simple":
        # Apply impulse at the start of the leg
        v_plus = v_i + delta_v_i

        # Free drift with constant velocity
        r_next = r_i + v_plus * delta_t_i
        v_next = v_plus
        return r_next, v_next

    elif model == "cw":
        # Placeholder: future CW implementation can go here, e.g.:
        #   v_plus = v_i + delta_v_i
        #   r_next, v_next = propagate_cw(r_i, v_plus, delta_t_i, n=cw_params["n"])
        # For now, just fall back to simple dynamics and log it.
        logging.warning("[IVT Optimizer] 'cw' model requested but not implemented; falling back to 'simple'.")
        v_plus = v_i + delta_v_i
        r_next = r_i + v_plus * delta_t_i
        v_next = v_plus
        return r_next, v_next

    else:
        raise ValueError(f"Unknown dynamics model '{model}'. Supported: 'simple', 'cw'.")

def solve_ivt_ocp(
    waypoints: np.ndarray,
    x0: np.ndarray,
    ctrl_cfg: dict,
    collision_checker,
    station_center: np.ndarray,
):
    """
    Solve the impulsive IVT optimal control problem.

    Current implementation (Stage 0):
      - Uses simple free-drift dynamics (ṙ = v, v̇ = 0).
      - Applies ZERO impulses (delta_v_i = 0) on each leg.
      - Uses a fixed leg duration from ctrl_cfg["dt_leg"] (or 60 s default).
      - Returns a dynamically consistent trajectory (straight-line drift),
        but does NOT yet optimize fuel or time.

    Parameters
    ----------
    waypoints : (N,3)
        Safe collision-free path produced upstream.
    x0 : (6,)
        Initial relative state [r0, v0].
    ctrl_cfg : dict
        Control section from settings.yaml, including e.g.:
          - w_fuel
          - w_time
          - delta_v_max
          - viewpoint_tol
          - v_final_max
          - dt_leg        (optional, seconds)
          - dynamics_model (optional, "simple" or "cw")
    collision_checker : callable
        Function to test if positions are in collision. Currently unused here,
        but will be needed once we enforce collision constraints per leg.
    station_center : (3,)
        Used for drift sanity checks or fallback behaviors (reserved).

    Returns
    -------
    visit_positions : (N,3)
        Positions at each viewpoint visit (trajectory sample).
    visit_velocities : (N,3)
        Velocities at each viewpoint visit.
    metrics : dict
        Information on time, fuel, delta-vs, feasibility, etc.
    """

    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError("waypoints must be of shape (N, 3)")

    N = waypoints.shape[0]

    if N < 2:
        # Degenerate case: zero or one waypoint, nothing to optimize
        logging.warning("[IVT Optimizer] Fewer than 2 waypoints; returning trivial trajectory.")
        visit_positions = waypoints.copy()
        visit_velocities = np.zeros_like(visit_positions)
        metrics = {
            "fuel_used": 0.0,
            "time_total": 0.0,
            "delta_v_list": [],
            "delta_v_norms": [],
            "t_leg_list": [],
            "mass0": float(ctrl_cfg.get("mass0", 10.0)),
            "mass_final": float(ctrl_cfg.get("mass0", 10.0)),
            "mass_history": [float(ctrl_cfg.get("mass0", 10.0))],
            "v_final": [0.0, 0.0, 0.0],
            "status": "degenerate_N<2",
        }
        return visit_positions, visit_velocities, metrics

    logging.info("[IVT Optimizer] Global multi-leg OCP: optimizing Δv and Δt across all legs.")

    dynamics_model = ctrl_cfg.get("dynamics_model", "simple")

    # Bounds / tolerances from config
    tol = float(ctrl_cfg.get("viewpoint_tol", 0.5))           # meters
    delta_v_max = float(ctrl_cfg.get("delta_v_max", 0.10))    # m/s

    # Time bounds
    dt_min = float(ctrl_cfg.get("dt_min", 5.0))               # seconds
    dt_max = float(ctrl_cfg.get("dt_max", 200.0))             # seconds
    dt_guess = float(ctrl_cfg.get("dt_leg", 60.0))            # initial guess

    # Cost weights
    w_fuel = float(ctrl_cfg.get("w_fuel", 1.0))
    w_time = float(ctrl_cfg.get("w_time", 0.1))

    # Final velocity bound
    v_final_max = float(ctrl_cfg.get("v_final_max", 0.05))    # m/s

    # Initial state
    x0 = np.asarray(x0, dtype=float).reshape(6)
    r0 = x0[0:3].copy()
    v0 = x0[3:6].copy()

    # Number of legs
    n_legs = N - 1

    # -----------------------------
    # Global decision variables
    # -----------------------------
    # Δv stacked: [dv0_x, dv0_y, dv0_z, dv1_x, ...]
    dv_sym = ca.SX.sym("dv", 3 * n_legs)
    # Δt stacked: [dt0, dt1, ...]
    dt_sym = ca.SX.sym("dt", n_legs)

    # Decision vector x = [dv_sym; dt_sym]
    x_sym = ca.vertcat(dv_sym, dt_sym)

    # -----------------------------
    # Build cost and constraints
    # -----------------------------
    J = 0
    g_list = []

    # Start at initial state
    r = r0
    v = v0

    for k in range(n_legs):
        # Extract dv_k and dt_k from stacked variables
        dv_k = dv_sym[3 * k : 3 * (k + 1)]
        dt_k = dt_sym[k]

        # Dynamics: impulse at start, then drift
        v_plus = v + dv_k
        r_next = r + v_plus * dt_k
        v_next = v_plus

        # Cost contribution for this leg
        J = J + w_fuel * ca.dot(dv_k, dv_k) + w_time * dt_k

        # Constraint 1: proximity to waypoint k+1
        #   ||r_next - p_{k+1}|| <= tol
        diff = r_next - waypoints[k + 1]
        g_pos = ca.dot(diff, diff) - tol**2   # <= 0
        g_list.append(g_pos)

        # Constraint 2: delta-v magnitude bound
        #   ||dv_k|| <= delta_v_max
        g_dv = ca.dot(dv_k, dv_k) - (delta_v_max**2)  # <= 0
        g_list.append(g_dv)

        # On the final leg, enforce final velocity bound
        if k == n_legs - 1:
            g_vf = ca.dot(v_next, v_next) - (v_final_max**2)   # <= 0
            g_list.append(g_vf)

        # Update for next leg
        r = r_next
        v = v_next

    # Stack all constraints
    g_sym = ca.vertcat(*g_list)
    num_g = g_sym.shape[0]

    nlp = {
        "x": x_sym,
        "f": J,
        "g": g_sym,
    }

    solver = ca.nlpsol(
        "global_ivt_solver",
        "ipopt",
        nlp,
        {
            "ipopt.print_level": 0,
            "print_time": 0,
        },
    )

    # -----------------------------
    # Bounds and initial guess
    # -----------------------------
    # All g_j are inequalities: g_j <= 0
    lbg = [-ca.inf] * num_g
    ubg = [0.0] * num_g

    # Variable bounds:
    # dv entries are unbounded (norm bound via constraints),
    # dt entries are bounded between dt_min and dt_max.
    lbx = [-ca.inf] * (3 * n_legs) + [dt_min] * n_legs
    ubx = [ ca.inf] * (3 * n_legs) + [dt_max] * n_legs

    # Initial guess: dv = 0, dt = dt_guess
    x0_guess = np.zeros(3 * n_legs + n_legs, dtype=float)
    x0_guess[3 * n_legs :] = dt_guess

    sol = solver(
        x0=x0_guess,
        lbg=lbg,
        ubg=ubg,
        lbx=lbx,
        ubx=ubx,
    )
    stats = solver.stats()

    if not stats.get("success", False):
        logging.warning(
            "[IVT Optimizer] Global OCP solve failed, "
            "falling back to zero Δv and dt_guess per leg."
        )
        dv_opt = np.zeros((n_legs, 3), dtype=float)
        dt_opt = np.full(n_legs, dt_guess, dtype=float)
    else:
        x_opt = np.array(sol["x"]).reshape(-1)
        dv_flat = x_opt[0 : 3 * n_legs]
        dt_flat = x_opt[3 * n_legs :]

        dv_opt = dv_flat.reshape(n_legs, 3)
        dt_opt = dt_flat

    # -----------------------------
    # Propagate numerically using optimal Δv and Δt
    # -----------------------------
    visit_positions = np.zeros((N, 3), dtype=float)
    visit_velocities = np.zeros((N, 3), dtype=float)

    r = r0.copy()
    v = v0.copy()

    visit_positions[0] = r
    visit_velocities[0] = v

    delta_v_list = []
    t_leg_list = []

    for k in range(n_legs):
        dv_k = dv_opt[k]
        dt_k = float(dt_opt[k])

        r, v = propagate_leg(
            r_i=r,
            v_i=v,
            delta_v_i=dv_k,
            delta_t_i=dt_k,
            model=dynamics_model,
            cw_params=None,
        )

        visit_positions[k + 1] = r
        visit_velocities[k + 1] = v

        delta_v_list.append(dv_k.tolist())
        t_leg_list.append(dt_k)

    v_final = v.copy()

    # -----------------------------
    # Fuel and time metrics
    # -----------------------------
    if len(t_leg_list) > 0:
        time_total = float(sum(t_leg_list))
    else:
        time_total = 0.0

    # Convert delta_v_list (list of 3-vectors) to array
    if len(delta_v_list) > 0:
        dv_arr = np.asarray(delta_v_list, dtype=float)  # shape (L, 3)
        dv_norms = np.linalg.norm(dv_arr, axis=1)       # shape (L,)
    else:
        dv_arr = np.zeros((0, 3), dtype=float)
        dv_norms = np.zeros(0, dtype=float)

    # Rocket equation: m_{k+1} = m_k * exp(-|Δv_k| / (Isp * g0))
    mass0 = float(ctrl_cfg.get("mass0", 10.0))       # kg
    Isp = float(ctrl_cfg.get("Isp", 220.0))          # s
    g0 = float(ctrl_cfg.get("g0", 9.80665))          # m/s^2

    m = mass0
    mass_history = [m]
    for dv_mag in dv_norms:
        if Isp > 0 and g0 > 0:
            m = m * np.exp(-dv_mag / (Isp * g0))
        mass_history.append(m)

    fuel_used = mass0 - m

    metrics = {
        "fuel_used": float(fuel_used),
        "time_total": float(time_total),
        "delta_v_list": delta_v_list,
        "delta_v_norms": dv_norms.tolist(),
        "t_leg_list": t_leg_list,
        "mass0": mass0,
        "mass_final": float(m),
        "mass_history": mass_history,
        "v_final": v_final.tolist(),
        "status": "global_multi_leg_delta_v_and_time_optimization",
    }

    return visit_positions, visit_velocities, metrics




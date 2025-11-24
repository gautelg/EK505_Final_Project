# src/pipeline/run_collision_free_path.py

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np

from src.control.pointing import compute_pointing_vectors
from src.geometry.collision import build_collision_checker_from_geometry
from src.control.collision_avoidance import (
    compute_station_center,
    make_collision_free_path,
)

# NEW: import the waypoint-state tools (quaternions etc.)
from src.control.trajectory import (
    WaypointState,
    build_waypoint_states,
    pack_states_for_export,
)

# NEW: import the impulsive OCP solver (you will implement this next)
from src.control.ivt_optimizer import solve_ivt_ocp


def _get_output_dir(config: Dict[str, Any]) -> Path:
    """
    Resolve the output directory from config, defaulting to 'data/outputs'.
    """
    out_dir = config.get("output_folder", "data/outputs/")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def _build_ordered_waypoints(
    viewpoints: np.ndarray,
    path: np.ndarray,
) -> np.ndarray:
    """
    Build ordered waypoints from viewpoint positions and a path index sequence.

    Parameters
    ----------
    viewpoints : np.ndarray
        Array of shape (N_vp, 3) with viewpoint positions.
    path : np.ndarray
        Array of shape (N_path,) with indices into viewpoints.

    Returns
    -------
    np.ndarray
        Ordered waypoints of shape (N_path, 3).
    """
    viewpoints = np.asarray(viewpoints, dtype=float)
    path = np.asarray(path, dtype=int)

    if viewpoints.ndim != 2 or viewpoints.shape[1] != 3:
        raise ValueError("viewpoints must be of shape (N, 3)")
    if path.ndim != 1:
        raise ValueError("path must be a 1D array of indices")

    if np.any(path < 0) or np.any(path >= viewpoints.shape[0]):
        raise ValueError("path contains indices out of bounds for viewpoints")

    ordered = viewpoints[path]
    logging.info(
        f"[CollisionFreePath] Built ordered waypoints: {ordered.shape[0]} waypoints."
    )
    return ordered


def run_collision_free_path(
    config: Dict[str, Any],
    mesh,
    centroids: np.ndarray,
    normals: np.ndarray,
    viewpoints: np.ndarray,
    path: np.ndarray,
) -> Tuple[np.ndarray, List[WaypointState], np.ndarray]:
    """
    High-level pipeline:

      1. Build collision checker from mesh/centroids/normals.
      2. Build ordered waypoints from viewpoints and TSP path.
      3. Use collision_avoidance.make_collision_free_path to insert via points.
      4. Compute pointing vectors for each safe waypoint.
      5. Run impulsive fuel/time optimization (if enabled).
      6. Build per-waypoint states (pos, vel, quat, omega) and export JSON.

    Returns
    -------
    safe_waypoints : np.ndarray
        Collision-free waypoint list of shape (M, 3), M >= N_path.
    waypoint_states : list[WaypointState]
        Discrete per-visit states (position, velocity, quaternion, omega).
    pointing_vectors : np.ndarray
        Pointing vectors used to build orientations, shape (M, 3).
    """
    system_cfg = config.get("system", {})

    # 1) Collision checker
    collision_checker = build_collision_checker_from_geometry(
        mesh, centroids=centroids, normals=normals
    )
    station_center = compute_station_center(centroids)

    # 2) Ordered waypoints from TSP path
    ordered_waypoints = _build_ordered_waypoints(viewpoints, path)

    # 3) Collision avoidance configuration
    coll_cfg = config.get("collision", {})
    margin = float(coll_cfg.get("margin", 0.0))
    num_samples = int(coll_cfg.get("num_samples", 20))
    step = float(coll_cfg.get("step", 1.0))
    max_shift = float(coll_cfg.get("max_shift", 20.0))
    max_depth = int(coll_cfg.get("max_depth", 3))

    logging.info(
        "[CollisionFreePath] Collision avoidance params: "
        f"margin={margin}, num_samples={num_samples}, "
        f"step={step}, max_shift={max_shift}, max_depth={max_depth}"
    )

    safe_waypoints = make_collision_free_path(
        waypoints=ordered_waypoints,
        checker=collision_checker,
        center=station_center,
        margin=margin,
        num_samples=num_samples,
        step=step,
        max_shift=max_shift,
        max_depth=max_depth,
    )

    logging.info(
        f"[CollisionFreePath] Safe path has {safe_waypoints.shape[0]} waypoints "
        f"(from {ordered_waypoints.shape[0]} original)."
    )

    # 4) Pointing vectors (world-frame directions for camera)
    pointing_vectors = compute_pointing_vectors(
        waypoints=safe_waypoints,
        centroids=centroids,
        original_waypoints=ordered_waypoints,
        tol=1e-3,
    )

    out_dir = _get_output_dir(config)
    _export_safe_waypoints(out_dir, safe_waypoints)

    # 5) Impulsive optimization (fuel/time), if enabled
    ctrl_cfg = config.get("control", {})
    run_opt = system_cfg.get("run_optimization", False)

    if run_opt:
        logging.info("[CollisionFreePath] Running impulsive fuel/time optimization...")
        # Initial state: start at first safe waypoint with zero relative velocity
        x0 = np.zeros(6, dtype=float)
        x0[0:3] = safe_waypoints[0]

        # Solve the OCP. You will implement this in src/control/ivt_optimizer.py
        visit_positions, visit_velocities, metrics = solve_ivt_ocp(
            safe_waypoints,         # positions to visit (collision-free)
            x0,                     # initial state [r0, v0]
            ctrl_cfg,               # dict with w_fuel, w_time, delta_v_max, etc.
            collision_checker,      # collision checker
            station_center          # center of the station
        )
    else:
        logging.info(
            "[CollisionFreePath] run_optimization=False, "
            "using safe_waypoints with zero velocity as a trivial trajectory."
        )
        visit_positions = safe_waypoints.copy()
        visit_velocities = np.zeros_like(safe_waypoints)
        metrics = {
            "fuel_used": 0.0,
            "time_total": 0.0,
            "delta_v_list": [],
            "t_leg_list": [],
        }

    # 6) Build per-waypoint states (pos, vel, quat, omega) and export
    waypoint_states = build_waypoint_states(
        positions=visit_positions,
        velocities=visit_velocities,
        pointing_vectors=pointing_vectors,
        up_hint=np.array([0.0, 0.0, 1.0]),
    )

    _export_optimized_viewpoints(out_dir, waypoint_states, metrics)

    logging.info(
        f"[CollisionFreePath] Exported safe waypoints and optimized "
        f"viewpoints to {out_dir}"
    )

    return safe_waypoints, waypoint_states, pointing_vectors


def _export_safe_waypoints(out_dir: Path, waypoints: np.ndarray) -> None:
    """
    Export safe waypoints to JSON for inspection or reuse.
    """
    waypoints = np.asarray(waypoints, dtype=float)
    path = out_dir / "safe_waypoints.json"
    data = {
        "waypoints": waypoints.tolist(),
    }
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"[CollisionFreePath] Saved safe waypoints to {path}")


def _export_optimized_viewpoints(
    out_dir: Path,
    states: List[WaypointState],
    metrics: Dict[str, Any],
) -> None:
    """
    Export fully specified viewpoint states for the low-level controller.

    Each waypoint has:
      - pos:  [x, y, z]
      - vel:  [vx, vy, vz]
      - quat: [w, x, y, z] (world<-body)
      - omega:[wx, wy, wz] (here zero)
    """
    pos_arr, vel_arr, quat_arr, omega_arr = pack_states_for_export(states)
    N = pos_arr.shape[0]

    waypoints = []
    for i in range(N):
        waypoints.append(
            {
                "pos": pos_arr[i].tolist(),
                "vel": vel_arr[i].tolist(),
                "quat": quat_arr[i].tolist(),
                "omega": omega_arr[i].tolist(),
            }
        )

    data = {
        "waypoints": waypoints,
        "metrics": metrics,
    }

    path = out_dir / "optimized_viewpoints.json"
    with path.open("w") as f:
        json.dump(data, f, indent=2)

    logging.info(f"[CollisionFreePath] Saved optimized viewpoints to {path}")

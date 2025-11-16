# src/pipeline/run_collision_free_path.py

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from src.geometry.collision import build_collision_checker_from_geometry
from src.control.collision_avoidance import (
    compute_station_center,
    make_collision_free_path,
)

from src.control.trajectory import build_trajectory, Trajectory

def _get_output_dir(config: Dict[str, Any]) -> Path:
    """
    Resolve the output directory from config, defaulting to 'data/outputs'.
    """
    # OLD:
    # paths_cfg = config.get("paths", {})
    # out_dir = paths_cfg.get("output_dir", "data/outputs")

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
) -> Tuple[np.ndarray, Trajectory]:
    """
    High-level pipeline:

      1. Build collision checker from mesh/centroids/normals.
      2. Build ordered waypoints from viewpoints and TSP path.
      3. Use collision_avoidance.make_collision_free_path to insert via points.
      4. Time-parameterize the safe path to build a Trajectory.
      5. Export safe waypoints + trajectory meta-data for the controller.

    Parameters
    ----------
    config : dict
        Global configuration dictionary (parsed from settings.yaml).
    mesh : o3d.geometry.TriangleMesh
        Convex hull mesh from run_geometry.
    centroids : np.ndarray
        Face centroids from run_geometry, shape (N_face, 3).
    normals : np.ndarray
        Face normals from run_geometry, shape (N_face, 3).
    viewpoints : np.ndarray
        Viewpoint positions from run_viewpoints, shape (N_vp, 3).
    path : np.ndarray
        TSP path indices into viewpoints, shape (N_path,).

    Returns
    -------
    safe_waypoints : np.ndarray
        Collision-free waypoint list of shape (M, 3), M >= N_path.
    trajectory : Trajectory
        Time-parameterized Trajectory object.
    """
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

    # 4) Trajectory configuration
    traj_cfg = config.get("trajectory", {})
    cruise_speed = float(traj_cfg.get("cruise_speed", 0.05))  # m/s
    dwell_time = traj_cfg.get("dwell_time", 0.0)              # can be scalar or list

    logging.info(
        f"[CollisionFreePath] Trajectory params: cruise_speed={cruise_speed}, "
        f"dwell_time={dwell_time}"
    )

    trajectory = build_trajectory(
        waypoints=safe_waypoints,
        cruise_speed=cruise_speed,
        dwell_times=dwell_time,
        orientations=None,
    )

    # 5) Export to disk
    out_dir = _get_output_dir(config)
    _export_safe_waypoints(out_dir, safe_waypoints)
    _export_trajectory(out_dir, trajectory)

    logging.info(
        f"[CollisionFreePath] Exported collision-free path and trajectory to {out_dir}"
    )

    return safe_waypoints, trajectory


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


def _export_trajectory(out_dir: Path, traj: Trajectory) -> None:
    """
    Export trajectory meta-data to JSON so the controller can reconstruct
    or directly use it.
    """
    path = out_dir / "trajectory.json"

    data = {
        "waypoints": traj.waypoints.tolist(),
        "arrival_times": traj.arrival_times.tolist(),
        "dwell_times": traj.dwell_times.tolist(),
        "total_time": traj.total_time,
        # If you later add orientations per waypoint:
        "orientations": traj.orientations.tolist(),
    }

    with path.open("w") as f:
        json.dump(data, f, indent=2)

    logging.info(f"[CollisionFreePath] Saved trajectory to {path}")

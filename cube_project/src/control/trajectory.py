# src/control/trajectory.py

import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np


# -----------------------------
# Quaternion utilities
# -----------------------------

def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion [w, x, y, z].

    Returns identity [1,0,0,0] if the norm is too small.
    """
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix R (world <- body) to quaternion [w,x,y,z].

    Assumes R is a proper rotation (det ~ 1, orthonormal).
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3")

    trace = float(np.trace(R))

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0  # s = 4 * w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        # Find the major diagonal element
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0  # s = 4*x
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0  # s = 4*y
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0  # s = 4*z
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)
    return _normalize_quaternion(q)


# -----------------------------
# Waypoint orientation from pointing vectors
# -----------------------------

def compute_orientations_from_pointing(
    pointing_vectors: np.ndarray,
    up_hint: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Given a pointing direction in world frame for each waypoint, compute
    a world<-body quaternion that makes the body +X axis point along
    the desired direction, while keeping the body "up" reasonably aligned
    with a world up vector.

    Parameters
    ----------
    pointing_vectors : (N, 3) array
        Unit pointing directions in world frame for each waypoint.
        These should be the output of compute_pointing_vectors(...)
        in pointing.py.
    up_hint : (3,) array or None
        World-frame "up" direction used to resolve roll. If None,
        defaults to [0, 0, 1].

    Returns
    -------
    quats : (N, 4) array
        World<-body quaternions [w, x, y, z] for each waypoint.
    """
    dirs = np.asarray(pointing_vectors, dtype=float)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError("pointing_vectors must be shape (N, 3)")

    N = dirs.shape[0]
    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=float)
    up_hint = np.asarray(up_hint, dtype=float)
    up_hint_norm = np.linalg.norm(up_hint)
    if up_hint_norm < 1e-8:
        raise ValueError("up_hint must be nonzero")
    up_hint = up_hint / up_hint_norm

    quats = np.zeros((N, 4), dtype=float)

    for i in range(N):
        d = dirs[i]
        n = np.linalg.norm(d)
        if n < 1e-8:
            # Fallback: identity orientation
            quats[i] = np.array([1.0, 0.0, 0.0, 0.0])
            continue
        x_axis = d / n  # body +X in world frame

        # If pointing vector is nearly parallel to up_hint, pick another up
        if abs(np.dot(x_axis, up_hint)) > 0.99:
            tmp_up = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            tmp_up = up_hint

        # Compute an orthonormal basis: x = pointing, y, z
        y_axis = np.cross(tmp_up, x_axis)
        ny = np.linalg.norm(y_axis)
        if ny < 1e-8:
            # Degenerate, pick arbitrary perpendicular
            if abs(x_axis[0]) < 0.9:
                y_axis = np.cross(np.array([1.0, 0.0, 0.0]), x_axis)
            else:
                y_axis = np.cross(np.array([0.0, 1.0, 0.0]), x_axis)
            ny = np.linalg.norm(y_axis)
        y_axis = y_axis / ny

        z_axis = np.cross(x_axis, y_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])  # columns are body axes in world

        quats[i] = _rotmat_to_quat(R)

    logging.info(f"[Trajectory] Computed {N} orientation quaternions from pointing vectors.")
    return quats


# -----------------------------
# Discrete waypoint state struct
# -----------------------------

@dataclass
class WaypointState:
    """
    Discrete state at a viewpoint for export to low-level control.

    All quantities are in world frame:

      - pos : position, shape (3,)
      - vel : velocity, shape (3,)
      - quat: world<-body quaternion [w, x, y, z]
      - omega: angular velocity (here typically zero), shape (3,)
    """
    pos: np.ndarray
    vel: np.ndarray
    quat: np.ndarray
    omega: np.ndarray


def build_waypoint_states(
    positions: np.ndarray,
    velocities: Optional[np.ndarray],
    pointing_vectors: np.ndarray,
    up_hint: Optional[np.ndarray] = None,
) -> List[WaypointState]:
    """
    Given positions, velocities, and pointing vectors at each viewpoint,
    build a list of WaypointState objects suitable for export.

    Parameters
    ----------
    positions : (N, 3) array
        Viewpoint positions in world frame.
    velocities : (N, 3) array or None
        Desired velocities at each viewpoint. If None, zeros are used.
    pointing_vectors : (N, 3) array
        Unit pointing vectors (world frame), typically from compute_pointing_vectors.
    up_hint : (3,) array or None
        World up direction for orientation computation, default [0,0,1].

    Returns
    -------
    states : list of WaypointState
        One per viewpoint.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be shape (N, 3)")

    N = positions.shape[0]

    if velocities is None:
        velocities = np.zeros((N, 3), dtype=float)
    else:
        velocities = np.asarray(velocities, dtype=float)
        if velocities.shape != (N, 3):
            raise ValueError("velocities must be shape (N, 3)")

    pointing_vectors = np.asarray(pointing_vectors, dtype=float)
    if pointing_vectors.shape != (N, 3):
        raise ValueError("pointing_vectors must be shape (N, 3)")

    quats = compute_orientations_from_pointing(pointing_vectors, up_hint=up_hint)

    states: List[WaypointState] = []
    for i in range(N):
        pos = positions[i].copy()
        vel = velocities[i].copy()
        quat = quats[i].copy()
        omega = np.zeros(3, dtype=float)  # could later be nonzero if you want attitude profiles

        states.append(WaypointState(pos=pos, vel=vel, quat=quat, omega=omega))

    logging.info(f"[Trajectory] Built {N} waypoint states (pos, vel, quat, omega).")
    return states


# -----------------------------
# Convenience packing for export
# -----------------------------

def pack_states_for_export(states: List[WaypointState]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a list of WaypointState into plain numpy arrays for JSON export.

    Returns
    -------
    pos_arr : (N, 3)
    vel_arr : (N, 3)
    quat_arr: (N, 4)
    omega_arr: (N, 3)
    """
    N = len(states)
    pos_arr = np.zeros((N, 3), dtype=float)
    vel_arr = np.zeros((N, 3), dtype=float)
    quat_arr = np.zeros((N, 4), dtype=float)
    omega_arr = np.zeros((N, 3), dtype=float)

    for i, st in enumerate(states):
        pos_arr[i] = st.pos
        vel_arr[i] = st.vel
        quat_arr[i] = st.quat
        omega_arr[i] = st.omega

    return pos_arr, vel_arr, quat_arr, omega_arr

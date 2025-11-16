# src/control/attitude.py

import logging
from typing import Optional

import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z] (world <- body).
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        # Find the largest diagonal element and jump to case
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    return q


def _build_R_world_from_forward_and_up(
    forward_world: np.ndarray,
    up_world_hint: np.ndarray,
    forward_axis: str = "z",
) -> np.ndarray:
    """
    Construct a rotation matrix R_WB (world <- body) given:

      - forward_world: where the chosen body forward axis should point in world.
      - up_world_hint: approximate 'up' direction in world.
      - forward_axis: which body axis is considered forward: 'x', 'y', or 'z'.

    We build an orthonormal basis (x_w, y_w, z_w) that will be taken
    as the columns of R_WB, i.e. R_WB = [x_w, y_w, z_w].
    """
    f = _normalize(forward_world)
    u_hint = _normalize(up_world_hint)

    # Make up vector orthogonal to forward
    u = u_hint - np.dot(u_hint, f) * f
    if np.linalg.norm(u) < 1e-6:
        # Fallback: pick some arbitrary up that isn't parallel to forward
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(tmp, f)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        u = tmp - np.dot(tmp, f) * f
    u = _normalize(u)

    # Right = forward × up
    r = np.cross(f, u)
    if np.linalg.norm(r) < 1e-6:
        # Fallback orthogonal
        r = np.array([1.0, 0.0, 0.0])
    r = _normalize(r)

    # Recompute a perfectly orthonormal up
    u = np.cross(r, f)
    u = _normalize(u)

    # Now assign to body axes depending on which body axis is "forward"
    # We want R_WB columns = [x_w, y_w, z_w]
    if forward_axis == "z":
        x_w = r
        y_w = u
        z_w = f
    elif forward_axis == "x":
        x_w = f
        y_w = u
        z_w = r
    elif forward_axis == "y":
        x_w = r
        y_w = f
        z_w = u
    else:
        raise ValueError("forward_axis must be one of {'x', 'y', 'z'}")

    R = np.column_stack([x_w, y_w, z_w])
    return R


def compute_look_at_face_orientations(
    waypoints: np.ndarray,
    centroids: np.ndarray,
    forward_axis: str = "z",
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """
    For each waypoint p_i:
      - Find nearest face centroid o_j.
      - forward_world = normalize(o_j - p_i).
      - up_world_hint = world_up orthogonalized against forward_world.
      - Build R_WB so that body 'forward_axis' points along forward_world.
      - Convert R_WB to quaternion [w,x,y,z].

    Parameters
    ----------
    waypoints : (N,3) array
    centroids : (F,3) array
    forward_axis : 'x', 'y', or 'z'
        Which body axis is considered 'forward' (camera direction).
    world_up : (3,) array
        Global up direction in world frame.

    Returns
    -------
    quats : (N,4) array of [w,x,y,z]
    """
    waypoints = np.asarray(waypoints, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    world_up = _normalize(world_up)

    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError("waypoints must be shape (N, 3)")
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError("centroids must be shape (F, 3)")

    N = waypoints.shape[0]
    F = centroids.shape[0]
    if F == 0:
        raise ValueError("No centroids provided")

    # brute-force nearest centroid
    diffs = waypoints[:, None, :] - centroids[None, :, :]   # (N,F,3)
    dists2 = np.sum(diffs**2, axis=2)                       # (N,F)
    nearest_idx = np.argmin(dists2, axis=1)                 # (N,)

    quats = np.zeros((N, 4), dtype=float)

    for i in range(N):
        p = waypoints[i]
        o = centroids[int(nearest_idx[i])]

        # 1) Forward direction: towards the face
        f = o - p
        if np.linalg.norm(f) < 1e-6:
            # Degenerate: sit exactly on centroid; just use world_up
            f = world_up
        f = _normalize(f)

        # 2) Up hint: world_up made orthogonal to forward
        u = world_up - np.dot(world_up, f) * f
        if np.linalg.norm(u) < 1e-6:
            # If world_up is nearly parallel to f, choose another
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(tmp, f)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            u = tmp - np.dot(tmp, f) * f
        u = _normalize(u)

        # 3) Right = forward × up
        r = np.cross(f, u)
        r = _normalize(r)

        # Re-orthogonalize up for numerical stability
        u = np.cross(r, f)
        u = _normalize(u)

        # 4) Build R_WB with columns = body axes in world coords
        #    so that R_WB @ e_forward_body = f
        if forward_axis == "z":
            x_w = r
            y_w = u
            z_w = f
        elif forward_axis == "x":
            x_w = f
            y_w = u
            z_w = r
        elif forward_axis == "y":
            x_w = r
            y_w = f
            z_w = u
        else:
            raise ValueError("forward_axis must be one of {'x','y','z'}")

        R_WB = np.column_stack([x_w, y_w, z_w])

        # 5) Convert to quaternion
        quats[i] = _rotation_matrix_to_quaternion(R_WB)

    logging.info(
        f"[Attitude] Computed look-at-face orientations for {N} waypoints "
        f"(forward_axis='{forward_axis}')."
    )
    return quats
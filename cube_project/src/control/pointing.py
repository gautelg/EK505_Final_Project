# src/control/pointing.py

import numpy as np
import logging
from typing import Optional


def compute_pointing_vectors(
    waypoints: np.ndarray,
    centroids: np.ndarray,
    original_waypoints: Optional[np.ndarray] = None,
    tol: float = 1e-3,
) -> np.ndarray:
    """
    Compute a unit pointing vector for each waypoint that points towards
    the nearest face centroid.

    For "via" waypoints (points inserted by collision avoidance), we copy
    the pointing vector from the previous waypoint to avoid jitter.

    A waypoint is treated as "real" if it is within 'tol' of any original
    waypoint.

    Parameters
    ----------
    waypoints : (M, 3) array
        Collision-free waypoints.
    centroids : (F, 3) array
        Face centroids of the station mesh.
    original_waypoints : (N, 3) array or None
        Original ordered waypoints (before collision avoidance). If None,
        all waypoints are treated as "real".
    tol : float
        Distance tolerance to decide if a waypoint matches an original one.

    Returns
    -------
    pointing_vectors : (M, 3) array
        Unit vectors pointing from each waypoint to its target face.
    """
    waypoints = np.asarray(waypoints, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError("waypoints must be shape (M, 3)")
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError("centroids must be shape (F, 3)")

    M = waypoints.shape[0]
    F = centroids.shape[0]

    # Decide which waypoints are "real" (inspection) vs "via".
    if original_waypoints is None:
        is_real = np.ones(M, dtype=bool)
    else:
        original_waypoints = np.asarray(original_waypoints, dtype=float)
        if original_waypoints.ndim != 2 or original_waypoints.shape[1] != 3:
            raise ValueError("original_waypoints must be shape (N, 3)")

        is_real = np.zeros(M, dtype=bool)
        for i in range(M):
            diff = original_waypoints - waypoints[i]  # (N, 3)
            dists = np.linalg.norm(diff, axis=1)
            if np.any(dists < tol):
                is_real[i] = True

    pointing = np.zeros((M, 3), dtype=float)

    for i in range(M):
        if i > 0 and not is_real[i]:
            # Via point: copy previous pointing direction
            pointing[i] = pointing[i - 1]
            continue

        # Real waypoint (or first one): aim at nearest centroid
        wp = waypoints[i]
        diff = centroids - wp  # (F, 3)
        dists = np.linalg.norm(diff, axis=1)
        j_star = int(np.argmin(dists))
        d = diff[j_star]

        norm = np.linalg.norm(d)
        if norm < 1e-8:
            # Degenerate: waypoint is basically at the centroid
            # fall back to previous direction or some default
            if i > 0:
                pointing[i] = pointing[i - 1]
            else:
                pointing[i] = np.array([1.0, 0.0, 0.0])
        else:
            pointing[i] = d / norm

    logging.info(
        f"[Pointing] Computed pointing vectors for {M} waypoints "
        f"using {F} centroids."
    )
    return pointing

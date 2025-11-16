# src/control/collision_avoidance.py

import logging
from typing import List

import numpy as np

from src.geometry.collision import ConvexHullCollisionChecker


def compute_station_center(centroids: np.ndarray) -> np.ndarray:
    """
    Compute a simple station "center" from face centroids.

    Parameters
    ----------
    centroids : np.ndarray
        Array of shape (N, 3) with face centroids.

    Returns
    -------
    np.ndarray
        Center point of shape (3,).
    """
    centroids = np.asarray(centroids, dtype=float)
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError("centroids must be of shape (N, 3)")
    center = centroids.mean(axis=0)
    logging.info(f"[CollisionAvoidance] Station center estimated at {center}")
    return center


def _build_local_directions(midpoint: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Build four local tangent directions "up, down, left, right" around the station.

    Directions are constructed in a station-centric frame:

      - e_r: radial direction from center to midpoint
      - e_u: tangent "up" direction, orthogonal to e_r
      - e_l: tangent "left" = e_u × e_r

    Returns the four unit directions:
      [ e_u, -e_u, e_l, -e_l ]

    Parameters
    ----------
    midpoint : np.ndarray
        Midpoint of the segment, shape (3,).
    center : np.ndarray
        Station center, shape (3,).

    Returns
    -------
    np.ndarray
        Directions of shape (4, 3).
    """
    midpoint = np.asarray(midpoint, dtype=float)
    center = np.asarray(center, dtype=float)

    r = midpoint - center
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-8:
        # Degenerate case: midpoint is at the center.
        # Fall back to some arbitrary but fixed direction.
        e_r = np.array([1.0, 0.0, 0.0])
    else:
        e_r = r / r_norm

    # Choose an "up" axis that is not parallel to e_r
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(world_up, e_r)) > 0.9:
        world_up = np.array([0.0, 1.0, 0.0])

    # Make e_u tangent to station surface: orthogonal to e_r
    e_u = world_up - np.dot(world_up, e_r) * e_r
    e_u_norm = np.linalg.norm(e_u)
    if e_u_norm < 1e-8:
        # Fallback if something went wrong numerically
        e_u = np.array([0.0, 1.0, 0.0])
        e_u_norm = 1.0
    e_u = e_u / e_u_norm

    # Left direction is e_l = e_u × e_r
    e_l = np.cross(e_u, e_r)
    e_l_norm = np.linalg.norm(e_l)
    if e_l_norm < 1e-8:
        # Fallback to some orthogonal
        e_l = np.array([1.0, 0.0, 0.0])
        e_l_norm = 1.0
    e_l = e_l / e_l_norm

    directions = np.stack([e_u, -e_u, e_l, -e_l], axis=0)
    return directions


def _find_candidate_midpoints(
    pA: np.ndarray,
    pB: np.ndarray,
    center: np.ndarray,
    checker: ConvexHullCollisionChecker,
    margin: float,
    num_samples: int,
    step: float,
    max_shift: float,
) -> List[np.ndarray]:
    """
    For a colliding segment pA -> pB, try to find candidate midpoints by
    shifting the segment midpoint along four local directions until both
    subsegments are collision-free.

    Parameters
    ----------
    pA, pB : np.ndarray
        Segment endpoints, shape (3,).
    center : np.ndarray
        Station center, shape (3,).
    checker : ConvexHullCollisionChecker
        Collision checker instance.
    margin : float
        Safety margin in meters.
    num_samples : int
        Number of samples per segment for safety checks.
    step : float
        Step size for shifting midpoint in meters.
    max_shift : float
        Maximum shift distance in meters.

    Returns
    -------
    List[np.ndarray]
        List of candidate safe midpoints (possibly empty).
    """
    pA = np.asarray(pA, dtype=float)
    pB = np.asarray(pB, dtype=float)

    m = 0.5 * (pA + pB)
    directions = _build_local_directions(m, center)

    candidates: List[np.ndarray] = []

    # Try four tangent directions
    for d in directions:
        s = step
        while s <= max_shift:
            m_shift = m + s * d

            safe_left = checker.is_segment_safe(
                pA, m_shift, margin=margin, num_samples=num_samples
            )
            safe_right = checker.is_segment_safe(
                m_shift, pB, margin=margin, num_samples=num_samples
            )

            if safe_left and safe_right:
                candidates.append(m_shift)
                break

            s += step

    # Optional radial-out fallback if tangent directions all fail
    if not candidates:
        r = m - center
        r_norm = np.linalg.norm(r)
        if r_norm > 1e-8:
            e_r = r / r_norm
            s = step
            while s <= max_shift:
                m_shift = m + s * e_r

                safe_left = checker.is_segment_safe(
                    pA, m_shift, margin=margin, num_samples=num_samples
                )
                safe_right = checker.is_segment_safe(
                    m_shift, pB, margin=margin, num_samples=num_samples
                )

                if safe_left and safe_right:
                    candidates.append(m_shift)
                    break

                s += step

    return candidates


def repair_segment(
    pA: np.ndarray,
    pB: np.ndarray,
    checker: ConvexHullCollisionChecker,
    center: np.ndarray,
    margin: float = 0.0,
    num_samples: int = 20,
    step: float = 1.0,
    max_shift: float = 20.0,
    max_depth: int = 3,
    depth: int = 0,
) -> List[np.ndarray]:
    """
    Repair a potentially colliding segment pA -> pB by inserting intermediate
    waypoints using the "shifted midpoint" heuristic.

    Logic:
      1. If the original segment is safe, return [pA, pB].
      2. If max_depth reached, give up and return [pA, pB].
      3. Otherwise:
         a) compute midpoint m
         b) try shifting m along up/down/left/right and radial-out directions
         c) for each shifted midpoint m', require both pA->m' and m'->pB to be safe
         d) pick the candidate closest to pA
         e) recursively repair pA->m_best and m_best->pB
         f) return concatenated list [pA, ..., m_best, ..., pB]

    Parameters
    ----------
    pA, pB : np.ndarray
        Segment endpoints, shape (3,).
    checker : ConvexHullCollisionChecker
        Collision checker instance.
    center : np.ndarray
        Station center, shape (3,).
    margin : float
        Safety margin in meters.
    num_samples : int
        Number of samples per segment for safety checks.
    step : float
        Midpoint shift size in meters.
    max_shift : float
        Maximum shift distance in meters.
    max_depth : int
        Maximum recursion depth.
    depth : int
        Current recursion depth (internal).

    Returns
    -------
    List[np.ndarray]
        List of waypoints representing the repaired segment, from pA to pB inclusive.
    """
    pA = np.asarray(pA, dtype=float)
    pB = np.asarray(pB, dtype=float)

    # Base case 1: segment already safe
    if checker.is_segment_safe(pA, pB, margin=margin, num_samples=num_samples):
        return [pA, pB]

    # Base case 2: reached maximum recursion depth
    if depth >= max_depth:
        logging.warning(
            "[CollisionAvoidance] Max depth reached while repairing segment; "
            "returning potentially unsafe segment."
        )
        return [pA, pB]

    # Try to find candidate shifted midpoints
    candidates = _find_candidate_midpoints(
        pA=pA,
        pB=pB,
        center=center,
        checker=checker,
        margin=margin,
        num_samples=num_samples,
        step=step,
        max_shift=max_shift,
    )

    if not candidates:
        logging.warning(
            "[CollisionAvoidance] No safe midpoint found for segment at depth "
            f"{depth}; returning original segment."
        )
        return [pA, pB]

    # Choose the candidate closest to pA (your heuristic)
    candidates_arr = np.stack(candidates, axis=0)
    dists = np.linalg.norm(candidates_arr - pA[None, :], axis=1)
    best_idx = int(np.argmin(dists))
    m_best = candidates_arr[best_idx]

    # Recursively repair subsegments
    left_pts = repair_segment(
        pA,
        m_best,
        checker=checker,
        center=center,
        margin=margin,
        num_samples=num_samples,
        step=step,
        max_shift=max_shift,
        max_depth=max_depth,
        depth=depth + 1,
    )

    right_pts = repair_segment(
        m_best,
        pB,
        checker=checker,
        center=center,
        margin=margin,
        num_samples=num_samples,
        step=step,
        max_shift=max_shift,
        max_depth=max_depth,
        depth=depth + 1,
    )

    # Concatenate, avoiding duplicate of m_best
    repaired = left_pts[:-1] + right_pts
    return repaired


def make_collision_free_path(
    waypoints: np.ndarray,
    checker: ConvexHullCollisionChecker,
    center: np.ndarray,
    margin: float = 0.0,
    num_samples: int = 20,
    step: float = 1.0,
    max_shift: float = 20.0,
    max_depth: int = 3,
) -> np.ndarray:
    """
    Given an ordered list of waypoints, repair each segment to avoid collisions,
    returning a new list of collision-free waypoints (up to the resolution of
    the sampling and heuristic).

    Parameters
    ----------
    waypoints : np.ndarray
        Array of shape (N, 3) with ordered waypoints.
    checker : ConvexHullCollisionChecker
        Collision checker instance.
    center : np.ndarray
        Station center, shape (3,).
    margin : float
        Safety margin in meters.
    num_samples : int
        Number of samples per segment for safety checks.
    step : float
        Midpoint shift size in meters.
    max_shift : float
        Maximum shift distance in meters.
    max_depth : int
        Maximum recursion depth per segment.

    Returns
    -------
    np.ndarray
        New array of shape (M, 3) with collision-free waypoints (M >= N).
    """
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError("waypoints must be of shape (N, 3)")

    if waypoints.shape[0] < 2:
        return waypoints.copy()

    repaired_points: List[np.ndarray] = []
    repaired_points.append(waypoints[0])

    num_segments = waypoints.shape[0] - 1

    for i in range(num_segments):
        pA = waypoints[i]
        pB = waypoints[i + 1]

        segment_pts = repair_segment(
            pA=pA,
            pB=pB,
            checker=checker,
            center=center,
            margin=margin,
            num_samples=num_samples,
            step=step,
            max_shift=max_shift,
            max_depth=max_depth,
            depth=0,
        )

        # segment_pts includes both endpoints; we already have pA in repaired_points
        # so we skip the first one to avoid duplication.
        repaired_points.extend(segment_pts[1:])

        if i % 10 == 0:
            logging.info(
                f"[CollisionAvoidance] Processed segment {i+1}/{num_segments}, "
                f"current waypoint count: {len(repaired_points)}"
            )

    repaired_array = np.vstack(repaired_points)
    logging.info(
        f"[CollisionAvoidance] Collision-free path built with {repaired_array.shape[0]} waypoints "
        f"(from {waypoints.shape[0]} original)."
    )
    return repaired_array

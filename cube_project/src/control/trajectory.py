# src/control/trajectory.py

import logging
from typing import Optional, Tuple

import numpy as np


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion [w, x, y, z]."""
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        # Fallback to identity if something is badly wrong
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def _slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    """
    Spherical linear interpolation between two quaternions.

    Parameters
    ----------
    q0, q1 : np.ndarray
        Quaternions [w, x, y, z].
    s : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    np.ndarray
        Interpolated quaternion [w, x, y, z].
    """
    q0 = _normalize_quaternion(q0)
    q1 = _normalize_quaternion(q1)
    s = float(s)

    dot = float(np.dot(q0, q1))

    # Ensure shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # If very close, fall back to linear interpolation
    if dot > 0.9995:
        result = (1.0 - s) * q0 + s * q1
        return _normalize_quaternion(result)

    # Slerp
    theta_0 = np.arccos(dot)          # angle between
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * s
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = s0 * q0 + s1 * q1
    return _normalize_quaternion(result)


def compute_segment_times(
    waypoints: np.ndarray,
    cruise_speed: float,
    dwell_times: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute travel times between waypoints and arrival times at each waypoint.

    We assume:
      - Straight-line motion between waypoints at constant speed 'cruise_speed'.
      - Optional dwell time at each waypoint (e.g., for inspection).

    Time model:
      - arrival_times[0] = 0
      - At waypoint i:
          dwell for dwell_times[i]
          then travel to waypoint i+1 with travel_time[i]
      - arrival_times[i+1] = arrival_times[i] + dwell_times[i] + travel_time[i]
      - Total time includes dwell at the final waypoint:
          total_time = arrival_times[-1] + dwell_times[-1]

    Parameters
    ----------
    waypoints : np.ndarray
        Array of shape (N, 3) with ordered waypoint positions.
    cruise_speed : float
        Desired constant travel speed between waypoints (m/s).
    dwell_times : np.ndarray or None
        If None, no dwell time at any waypoint.
        If scalar-like, will be broadcast to all waypoints.
        If array of shape (N,), used per waypoint.

    Returns
    -------
    segment_times : np.ndarray
        Travel times per segment, shape (N-1,).
    arrival_times : np.ndarray
        Arrival times at each waypoint (start of dwell), shape (N,).
    dwell_times_arr : np.ndarray
        Dwell time per waypoint, shape (N,).
    total_time : float
        Total trajectory duration including final dwell.
    """
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError("waypoints must be of shape (N, 3)")

    N = waypoints.shape[0]
    if N < 1:
        raise ValueError("Need at least one waypoint")
    if cruise_speed <= 0.0:
        raise ValueError("cruise_speed must be > 0")

    # Distances and segment travel times
    if N == 1:
        segment_times = np.zeros(0, dtype=float)
    else:
        diffs = waypoints[1:] - waypoints[:-1]                # (N-1, 3)
        dists = np.linalg.norm(diffs, axis=1)                 # (N-1,)
        segment_times = dists / cruise_speed                  # (N-1,)

    # Dwell times: broadcast / default
    if dwell_times is None:
        dwell_times_arr = np.zeros(N, dtype=float)
    else:
        dwell_times_arr = np.asarray(dwell_times, dtype=float)
        if dwell_times_arr.ndim == 0:
            dwell_times_arr = np.full(N, float(dwell_times_arr), dtype=float)
        elif dwell_times_arr.shape != (N,):
            raise ValueError(
                f"dwell_times must be scalar or shape (N,), got shape {dwell_times_arr.shape}"
            )

    arrival_times = np.zeros(N, dtype=float)
    for i in range(N - 1):
        arrival_times[i + 1] = (
            arrival_times[i] + dwell_times_arr[i] + segment_times[i]
        )

    total_time = arrival_times[-1] + dwell_times_arr[-1]

    logging.info(
        f"[Trajectory] Computed segment times for {N} waypoints: "
        f"total_time = {total_time:.2f} s"
    )

    return segment_times, arrival_times, dwell_times_arr, float(total_time)


class Trajectory:
    """
    Time-parameterized trajectory over a sequence of waypoints.

    Provides evaluate(t) -> (p_des, v_des, q_des, w_des), where:

      - p_des: desired world position (3,)
      - v_des: desired world velocity (3,)
      - q_des: desired world<-body quaternion [w, x, y, z]
      - w_des: desired angular velocity in body or world frame (here: zero)

    Motion model:
      - At waypoint i:
          dwell from arrival_times[i] to arrival_times[i] + dwell_times[i]
          (holding position/orientation, zero velocity/angular velocity)
      - Then move linearly to waypoint i+1 with constant velocity over
        segment_times[i].
    """

    def __init__(
        self,
        waypoints: np.ndarray,
        orientations: Optional[np.ndarray],
        segment_times: np.ndarray,
        arrival_times: np.ndarray,
        dwell_times: np.ndarray,
        total_time: float,
    ):
        waypoints = np.asarray(waypoints, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3:
            raise ValueError("waypoints must be of shape (N, 3)")

        self.waypoints = waypoints
        self.N = waypoints.shape[0]

        # Orientation handling
        if orientations is None:
            # Default: identity quaternion at all waypoints
            self.orientations = np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float), (self.N, 1)
            )
        else:
            orientations = np.asarray(orientations, dtype=float)
            if orientations.shape != (self.N, 4):
                raise ValueError(
                    f"orientations must be shape (N, 4), got {orientations.shape}"
                )
            # Normalize all quaternions
            self.orientations = np.vstack(
                [_normalize_quaternion(q) for q in orientations]
            )

        self.segment_times = np.asarray(segment_times, dtype=float)
        self.arrival_times = np.asarray(arrival_times, dtype=float)
        self.dwell_times = np.asarray(dwell_times, dtype=float)
        self.total_time = float(total_time)

        if self.segment_times.shape[0] != self.N - 1 and self.N > 1:
            raise ValueError(
                "segment_times must have length N-1 where N is number of waypoints"
            )

        # Precompute segment start/end times (for interpolation)
        if self.N > 1:
            self.segment_starts = self.arrival_times[:-1] + self.dwell_times[:-1]
            self.segment_ends = self.arrival_times[1:]
        else:
            self.segment_starts = np.zeros(0, dtype=float)
            self.segment_ends = np.zeros(0, dtype=float)

        logging.info(
            f"[Trajectory] Trajectory initialized with {self.N} waypoints, "
            f"total_time = {self.total_time:.2f} s"
        )

    def evaluate(self, t: float):
        """
        Evaluate desired state at time t.

        Parameters
        ----------
        t : float
            Time in seconds since start. Values outside [0, total_time]
            are clamped.

        Returns
        -------
        p_des : np.ndarray
            Desired position (3,).
        v_des : np.ndarray
            Desired velocity (3,).
        q_des : np.ndarray
            Desired quaternion [w, x, y, z].
        w_des : np.ndarray
            Desired angular velocity (3,). Here set to zero.
        """
        if self.N == 1:
            # Single waypoint: hold position/orientation forever
            p = self.waypoints[0]
            q = self.orientations[0]
            v = np.zeros(3, dtype=float)
            w = np.zeros(3, dtype=float)
            return p, v, q, w

        # Clamp time to [0, total_time]
        t = float(np.clip(t, 0.0, self.total_time))

        # 1) Check if we're in a dwell interval
        for i in range(self.N):
            dwell_start = self.arrival_times[i]
            dwell_end = dwell_start + self.dwell_times[i]
            if self.dwell_times[i] > 0.0 and dwell_start <= t < dwell_end:
                p = self.waypoints[i]
                q = self.orientations[i]
                v = np.zeros(3, dtype=float)
                w = np.zeros(3, dtype=float)
                return p, v, q, w

        # 2) Otherwise, find the segment we are in
        # segment i: from waypoint i to i+1, between segment_starts[i] and segment_ends[i]
        # We can find the first segment_end >= t
        idx = int(np.searchsorted(self.segment_ends, t, side="right")) - 1
        idx = max(0, min(idx, self.N - 2))  # clamp to valid segment index

        t_start = self.segment_starts[idx]
        t_end = self.segment_ends[idx]
        if t_end <= t_start:
            # Degenerate case: zero-length segment; treat as dwell at waypoint idx+1
            p = self.waypoints[idx + 1]
            q = self.orientations[idx + 1]
            v = np.zeros(3, dtype=float)
            w = np.zeros(3, dtype=float)
            return p, v, q, w

        # Interpolation factor s in [0, 1]
        s = (t - t_start) / (t_end - t_start)
        s = float(np.clip(s, 0.0, 1.0))

        p0 = self.waypoints[idx]
        p1 = self.waypoints[idx + 1]
        q0 = self.orientations[idx]
        q1 = self.orientations[idx + 1]

        # Position and velocity
        p = (1.0 - s) * p0 + s * p1
        # Constant velocity along the segment
        segment_duration = t_end - t_start
        v = (p1 - p0) / segment_duration

        # Orientation via slerp; angular velocity set to zero for simplicity
        q = _slerp(q0, q1, s)
        w = np.zeros(3, dtype=float)

        return p, v, q, w


def build_trajectory(
    waypoints: np.ndarray,
    cruise_speed: float,
    dwell_times: Optional[np.ndarray] = None,
    orientations: Optional[np.ndarray] = None,
) -> Trajectory:
    """
    High-level helper: from waypoints (and optional orientations), build a
    time-parameterized Trajectory object.

    Parameters
    ----------
    waypoints : np.ndarray
        Array of shape (N, 3) with ordered waypoint positions.
    cruise_speed : float
        Desired constant speed between waypoints (m/s).
    dwell_times : np.ndarray or None
        Dwell times at each waypoint. If None, no dwell.
        If scalar-like, applied to all waypoints.
        If array, must be shape (N,).
    orientations : np.ndarray or None
        Optional quaternions per waypoint, shape (N, 4), [w, x, y, z].
        If None, identity quaternion is used everywhere.

    Returns
    -------
    Trajectory
        Trajectory instance with evaluate(t) method.
    """
    (
        segment_times,
        arrival_times,
        dwell_times_arr,
        total_time,
    ) = compute_segment_times(
        waypoints=waypoints,
        cruise_speed=cruise_speed,
        dwell_times=dwell_times,
    )

    traj = Trajectory(
        waypoints=waypoints,
        orientations=orientations,
        segment_times=segment_times,
        arrival_times=arrival_times,
        dwell_times=dwell_times_arr,
        total_time=total_time,
    )

    return traj

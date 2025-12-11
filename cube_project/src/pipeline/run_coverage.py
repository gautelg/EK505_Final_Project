# src/pipeline/run_coverage.py

import numpy as np
import logging


def compute_coverage(vis_matrix, visited_indices):
    """
    Compute mesh face coverage from a set of visited viewpoints.

    Parameters
    ----------
    vis_matrix : np.ndarray, shape (N_view, N_faces), dtype=bool or {0,1}
        vis_matrix[i, j] is True/1 if face j is visible from viewpoint i.
    visited_indices : array-like of int
        Indices of viewpoints that are actually visited by the path
        (e.g. the TSP path, or some subset of viewpoints touched by an optimized trajectory).

    Returns
    -------
    coverage : float
        Fraction of faces that are covered by at least one visited viewpoint.
        In [0, 1].
    covered_faces : np.ndarray, shape (N_faces,), dtype=bool
        Boolean mask: True where face j is covered by at least one visited viewpoint.
    """
    if vis_matrix is None:
        raise ValueError("vis_matrix is None; run visibility before computing coverage")

    vis_matrix = np.asarray(vis_matrix)
    if vis_matrix.ndim != 2:
        raise ValueError(f"vis_matrix must be 2D, got shape {vis_matrix.shape}")

    n_view, n_faces = vis_matrix.shape

    visited_indices = np.asarray(visited_indices, dtype=int)
    if visited_indices.size == 0:
        logging.warning("[COVERAGE] No visited indices given; coverage is 0.0")
        covered_faces = np.zeros(n_faces, dtype=bool)
        return 0.0, covered_faces

    # Guard against out-of-range indices
    if visited_indices.min() < 0 or visited_indices.max() >= n_view:
        raise IndexError(
            f"visited_indices out of range: min={visited_indices.min()}, "
            f"max={visited_indices.max()}, N_view={n_view}"
        )

    # Subset the visibility matrix to only the visited viewpoints
    sub_vis = vis_matrix[visited_indices, :]   # shape (len(visited_indices), N_faces)

    # A face is covered if ANY visited viewpoint sees it
    covered_faces = np.any(sub_vis.astype(bool), axis=0)  # shape (N_faces,)

    coverage = float(covered_faces.mean())
    n_cov = int(covered_faces.sum())

    logging.info(
        "[COVERAGE] Faces covered: %d / %d (%.2f %%)",
        n_cov,
        n_faces,
        coverage * 100.0,
    )

    return coverage, covered_faces

# src/geometry/collision.py

import logging
from typing import Optional

import numpy as np
import open3d as o3d


class ConvexHullCollisionChecker:
    """
    Collision checker for a convex, watertight mesh using a half-space representation.

    The mesh is represented by:
      - face centroids o_i ∈ R^3
      - outward-pointing face normals n_i ∈ R^3

    For a point x ∈ R^3:

      r_i(x) = x - o_i

    If there exists a face i such that
      <r_i(x), n_i> > margin,
    then x is considered OUTSIDE (collision-free with margin).

    If for all faces i:
      <r_i(x), n_i> <= margin,
    then x is considered INSIDE or too close (collision).

    This matches the description in the paper: we require the point to be on the
    "outside" of at least one plane of a convex obstacle.
    """

    def __init__(self, centroids: np.ndarray, normals: np.ndarray):
        """
        Parameters
        ----------
        centroids : np.ndarray
            Array of shape (N_face, 3) containing face centroids o_i.
        normals : np.ndarray
            Array of shape (N_face, 3) containing UNIT outward normals n_i.
        """
        if centroids.shape != normals.shape:
            raise ValueError(
                f"centroids and normals must have the same shape, "
                f"got {centroids.shape} and {normals.shape}"
            )
        if centroids.shape[1] != 3:
            raise ValueError(
                f"centroids/normals must be of shape (N, 3), got {centroids.shape}"
            )

        self.centroids = np.asarray(centroids, dtype=float)
        self.normals = np.asarray(normals, dtype=float)

        # Normalize normals defensively
        norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.normals = self.normals / norms

        logging.info(
            f"[Collision] ConvexHullCollisionChecker initialized with "
            f"{self.centroids.shape[0]} faces."
        )

    @classmethod
    def from_mesh(cls, mesh: o3d.geometry.TriangleMesh) -> "ConvexHullCollisionChecker":
        """
        Build collision checker from an Open3D TriangleMesh.

        Assumes the mesh is already a convex hull (as produced in run_geometry).

        Parameters
        ----------
        mesh : o3d.geometry.TriangleMesh
            Convex, watertight triangle mesh.

        Returns
        -------
        ConvexHullCollisionChecker
        """
        if not isinstance(mesh, o3d.geometry.TriangleMesh):
            raise TypeError("Expected an Open3D TriangleMesh")

        if not mesh.has_triangles():
            raise ValueError("Mesh has no triangles!")

        # Ensure normals exist and are consistent
        mesh.compute_triangle_normals()

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        centroids = vertices[triangles].mean(axis=1)
        normals = np.asarray(mesh.triangle_normals)

        logging.info(
            f"[Collision] Created checker from mesh with {len(triangles)} faces."
        )
        return cls(centroids=centroids, normals=normals)

    def is_point_safe(self, x: np.ndarray, margin: float = 0.0) -> bool:
        """
        Check if a single point is outside the convex hull by at least 'margin'.

        Parameters
        ----------
        x : np.ndarray
            Point of shape (3,).
        margin : float
            Safety margin in meters. Larger margin means "further away" from the hull.

        Returns
        -------
        bool
            True if x is outside / collision-free; False if inside or too close.
        """
        x = np.asarray(x, dtype=float).reshape(1, 3)  # shape (1,3)

        # r_i(x) = x - o_i, for all faces
        r = x - self.centroids  # broadcasting, shape (N_face, 3)

        # dot_i = <r_i, n_i>
        dots = np.einsum("ij,ij->i", r, self.normals)  # shape (N_face,)

        # Collision-free if we are "outside" at least one face by more than margin
        is_outside_any = np.any(dots > margin)

        return bool(is_outside_any)

    def are_points_safe(self, X: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """
        Vectorized version of is_point_safe for a batch of points.

        Parameters
        ----------
        X : np.ndarray
            Points of shape (N, 3).
        margin : float
            Safety margin in meters.

        Returns
        -------
        np.ndarray
            Boolean mask of shape (N,), True if corresponding point is safe.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("X must be of shape (N, 3)")

        # For each point x_k, compute dot products with all faces
        # We can do this by expanding: X -> (N, 1, 3), centroids/normals -> (1, F, 3)
        X_expanded = X[:, np.newaxis, :]  # (N, 1, 3)
        r = X_expanded - self.centroids[np.newaxis, :, :]  # (N, F, 3)
        dots = np.einsum("ijk,jk->ij", r, self.normals)   # (N, F)

        # safe if exists face with dot > margin
        is_outside_any = np.any(dots > margin, axis=1)    # (N,)

        return is_outside_any

    def is_segment_safe(
        self,
        pA: np.ndarray,
        pB: np.ndarray,
        margin: float = 0.0,
        num_samples: int = 20,
    ) -> bool:
        """
        Check if the straight segment from pA to pB is collision-free.

        The segment is sampled at 'num_samples + 1' points, including endpoints.

        Parameters
        ----------
        pA, pB : np.ndarray
            Endpoints of shape (3,).
        margin : float
            Safety margin in meters.
        num_samples : int
            Number of intervals to sample along the segment.
            (Total points = num_samples + 1)

        Returns
        -------
        bool
            True if all sampled points are safe, False if any is unsafe.
        """
        pA = np.asarray(pA, dtype=float)
        pB = np.asarray(pB, dtype=float)

        if pA.shape != (3,) or pB.shape != (3,):
            raise ValueError("pA and pB must be shape (3,)")

        # Sample lambda in [0, 1]
        lambdas = np.linspace(0.0, 1.0, num_samples + 1)
        points = (1.0 - lambdas)[:, None] * pA[None, :] + lambdas[:, None] * pB[None, :]

        safe_mask = self.are_points_safe(points, margin=margin)
        all_safe = np.all(safe_mask)

        return bool(all_safe)


def build_collision_checker_from_geometry(
    mesh: o3d.geometry.TriangleMesh,
    centroids: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
) -> ConvexHullCollisionChecker:
    """
    Convenience builder that uses centroids/normals if provided, otherwise
    computes them from the mesh.

    This fits nicely with run_geometry, which already returns (mesh, centroids, normals).

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        Convex hull mesh from run_geometry.
    centroids : np.ndarray, optional
        Precomputed centroids, shape (N, 3).
    normals : np.ndarray, optional
        Precomputed normals, shape (N, 3).

    Returns
    -------
    ConvexHullCollisionChecker
    """
    if centroids is not None and normals is not None:
        logging.info("[Collision] Using provided centroids/normals for collision checker.")
        return ConvexHullCollisionChecker(centroids=centroids, normals=normals)
    else:
        logging.info("[Collision] Computing centroids/normals from mesh for collision checker.")
        return ConvexHullCollisionChecker.from_mesh(mesh)

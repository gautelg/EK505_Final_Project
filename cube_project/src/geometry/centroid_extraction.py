# src/geometry/centroid_extraction.py

import numpy as np
import open3d as o3d
import logging

def extract_centroids(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """
    Computes centroids of all triangles in the mesh.

    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh.

    Returns:
        np.ndarray: Array of shape (num_triangles, 3) with triangle centroids.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(triangles) == 0:
        logging.warning("[CentroidExtraction] Mesh has no triangles!")
        return np.zeros((0, 3), dtype=np.float32)

    centroids = vertices[triangles].mean(axis=1)
    logging.info(f"[CentroidExtraction] Computed centroids: {len(centroids)}")
    return centroids

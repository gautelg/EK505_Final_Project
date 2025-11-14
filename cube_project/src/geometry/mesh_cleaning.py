# src/geometry/mesh_cleaning.py

import open3d as o3d
import logging

def clean_mesh(mesh: o3d.geometry.TriangleMesh,
               remove_degenerate: bool = True,
               smooth_iterations: int = 0) -> o3d.geometry.TriangleMesh:
    """
    Cleans a TriangleMesh by removing degenerate triangles and optionally
    applying Laplacian smoothing.

    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh to clean.
        remove_degenerate (bool): Remove degenerate triangles if True.
        smooth_iterations (int): Number of Laplacian smoothing iterations.

    Returns:
        o3d.geometry.TriangleMesh: Cleaned mesh.
    """
    if remove_degenerate:
        before = len(mesh.triangles)
        mesh.remove_degenerate_triangles()
        after = len(mesh.triangles)
        logging.info(f"[MeshCleaning] Removed {before - after} degenerate triangles.")

    if smooth_iterations > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=smooth_iterations)
        logging.info(f"[MeshCleaning] Applied {smooth_iterations} smoothing iterations.")

    mesh.compute_vertex_normals()  # recompute normals after cleaning
    logging.info(f"[MeshCleaning] Mesh cleaning complete. Triangles: {len(mesh.triangles)}")

    return mesh

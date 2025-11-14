# src/pipeline/run_geometry.py

import logging
import open3d as o3d
from src.geometry.mesh_loader import MeshLoader
from src.geometry.mesh_cleaning import clean_mesh
from src.geometry.centroid_extraction import extract_centroids

def run_geometry(config):
    """
    Full geometry pipeline:
    1. Load mesh
    2. Clean mesh (degenerate triangles, non-manifold edges)
    3. Optionally extract only outer surface (convex hull)
    4. Compute centroids and normals
    """
    mesh_path = config["geometry"]["mesh_path"]
    mesh_loader = MeshLoader(mesh_path)
    mesh = mesh_loader.load_mesh()
    logging.info(f"[Geometry] Loaded mesh: {mesh_path}")

    # -------------------------
    # Mesh Cleaning
    # -------------------------
    if config["geometry"].get("cleaning", {}).get("enable", True):
        mesh = clean_mesh(
            mesh,
            remove_degenerate=config["geometry"]["cleaning"].get("remove_degenerate", True),
            smooth_iterations=config["geometry"]["cleaning"].get("smooth_iterations", 0)
        )
        logging.info("[Geometry] Mesh cleaned")

    # -------------------------
    # Keep only outer surface (convex hull)
    # -------------------------
    if config["geometry"].get("clean_outer_surface", True):
        hull, _ = mesh.compute_convex_hull()
        hull.compute_vertex_normals()
        mesh = hull
        logging.info("[Geometry] Mesh outer surface extracted using convex hull")

    # -------------------------
    # Centroids and Normals
    # -------------------------
    centroids, normals = mesh_loader.compute_centroids(mesh)
    logging.info(f"[Geometry] Computed {len(centroids)} centroids and normals")

    return mesh, centroids, normals

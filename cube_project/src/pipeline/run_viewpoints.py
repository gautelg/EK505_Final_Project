# src/pipeline/run_viewpoints.py

from src.viewpoints.viewpoint_generator import generate_viewpoints
from src.viewpoints.visibility import VisibilityChecker
from src.viewpoints.clustering import cluster_viewpoints
from src.viewpoints.TSP_solver import tsp_nearest_neighbor
import logging
import numpy as np

def run_viewpoints(config, mesh, centroids, normals):
    """
    Full pipeline for generating viewpoints, filtering via visibility, clustering, and TSP.
    """

    # -----------------------------
    # Generate candidate viewpoints
    # -----------------------------
    vp_cfg = config["viewpoint"]
    viewpoints = generate_viewpoints(
        mesh,
        centroids=centroids,
        normals=normals,
        distance=vp_cfg["distance"],
        use_sphere=vp_cfg.get("use_sphere", False),
        num_samples=vp_cfg.get("num_sphere_samples", 500)
    )
    logging.info(f"[Viewpoints] Generated {len(viewpoints)} candidate viewpoints")

    # -----------------------------
    # Cluster / filter viewpoints
    # -----------------------------
    cluster_cfg = config.get("clustering", {})
    if cluster_cfg.get("enable", True):
        viewpoints = cluster_viewpoints(
            viewpoints,
            method=cluster_cfg.get("method", "kmeans"),
            n_clusters=cluster_cfg.get("n_clusters", 100)
        )
        logging.info(f"[Viewpoints] Clustered to {len(viewpoints)} viewpoints")

    # -----------------------------
    # Compute visibility
    # -----------------------------
    vis_cfg = config.get("visibility", {})
    n_faces = len(centroids)
    n_view = len(viewpoints)

    if vis_cfg.get("enable", True):
        vis_checker = VisibilityChecker(mesh)
        # Typically: list-of-lists, one list of visible face indices per viewpoint
        visibility_lists = vis_checker.check_visibility(viewpoints, centroids)
        logging.info("[Viewpoints] Visibility computed for clustered viewpoints")
    else:
        # Fallback: every viewpoint sees every face
        visibility_lists = [list(range(n_faces)) for _ in range(n_view)]
        logging.info("[Viewpoints] Visibility disabled; assuming full visibility")

    # Convert list-of-lists → boolean matrix (N_view, N_faces)
    vis_matrix = np.zeros((n_view, n_faces), dtype=bool)
    for i, face_ids in enumerate(visibility_lists):
        # face_ids is e.g. [0, 5, 17, ...] for viewpoint i
        vis_matrix[i, face_ids] = True

    # -----------------------------
    # Solve TSP for optimal path
    # -----------------------------
    path = tsp_nearest_neighbor(viewpoints)
    logging.info(f"[Viewpoints] TSP solved for path with {len(path)} waypoints")

    return viewpoints, path, vis_matrix

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
    # Compute visibility
    # -----------------------------
    if config.get("visibility", {}).get("enable", True):
        vis_checker = VisibilityChecker(mesh)
        visibility = vis_checker.check_visibility(viewpoints, centroids)
        logging.info("[Viewpoints] Visibility computed")
    else:
        visibility = [list(range(len(centroids))) for _ in range(len(viewpoints))]

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
    # Solve TSP for optimal path
    # -----------------------------
    path = tsp_nearest_neighbor(viewpoints)
    logging.info(f"[Viewpoints] TSP solved for path with {len(path)} waypoints")

    return viewpoints, path

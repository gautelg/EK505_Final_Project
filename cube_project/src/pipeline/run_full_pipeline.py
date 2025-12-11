# src/pipeline/run_full_pipeline.py

import numpy as np
import logging
import json  # NEW
from src.pipeline.run_geometry import run_geometry
from src.pipeline.run_viewpoints import run_viewpoints
from src.pipeline.run_coverage import compute_coverage  # NEW

def run_full_pipeline(config):
    system_cfg = config["system"]

    mesh = None
    centroids = None
    normals = None
    viewpoints = None
    path = None
    safe_waypoints = None
    trajectory = None
    pointing_vectors = None
    vis_matrix = None
    coverage = None          # NEW
    covered_faces = None     # NEW  (bool mask per face)

    # -----------------------------
    # GEOMETRY
    # -----------------------------
    if system_cfg.get("run_geometry", True):
        logging.info("[SYSTEM] Running geometry pipeline...")
        mesh, centroids, normals = run_geometry(config)
    else:
        logging.info("[SYSTEM] Geometry skipped")

    # -----------------------------
    # VIEWPOINTS
    # -----------------------------
    if system_cfg.get("run_viewpoints", True):
        logging.info("[SYSTEM] Running viewpoint generation...")
        viewpoints, path, vis_matrix = run_viewpoints(
            config, mesh, centroids, normals
        )

        # -------------------------
        # COVERAGE (discrete path)
        # -------------------------
        if vis_matrix is not None and path is not None:
            coverage, covered_faces = compute_coverage(vis_matrix, path)
            logging.info(
                "[SYSTEM] Coverage from TSP path: %.2f %%",
                coverage * 100.0,
            )

            # -------------------------
            # SAVE COVERAGE TO JSON
            # -------------------------
            vis_cfg = config.get("visualization", {})
            cov_json_path = vis_cfg.get("coverage_json_path", None)
            if cov_json_path is not None:
                n_faces = int(covered_faces.size)
                n_cov = int(covered_faces.sum())
                cov_payload = {
                    "coverage": coverage,
                    "faces_covered": n_cov,
                    "faces_total": n_faces,
                    "path_length": int(len(path)),
                    "num_viewpoints": int(len(viewpoints)),
                }
                # ensure parent dir exists if you want, or rely on user
                with open(cov_json_path, "w") as f:
                    json.dump(cov_payload, f, indent=2)
                logging.info(
                    "[SYSTEM] Coverage JSON written to %s", cov_json_path
                )

    else:
        logging.info("[SYSTEM] Viewpoint generation skipped")

    # -----------------------------
    # COLLISION-FREE PATH + TRAJECTORY
    # -----------------------------
    if system_cfg.get("run_collision_free_path", False):
        # only run if we have what we need
        if (
            mesh is not None
            and centroids is not None
            and normals is not None
            and viewpoints is not None
            and path is not None
        ):
            from src.pipeline.run_collision_free_path import run_collision_free_path
            logging.info("[SYSTEM] Running collision-free path planning...")
            safe_waypoints, waypoint_states, pointing_vectors = run_collision_free_path(
                config, mesh, centroids, normals, viewpoints, path
            )
        else:
            logging.warning(
                "[SYSTEM] Cannot run collision-free path planning: "
                "missing mesh/viewpoints/path."
            )
    else:
        logging.info("[SYSTEM] Collision-free path planning skipped")
    
    # -----------------------------
    # CONTROL
    # -----------------------------
    if system_cfg.get("run_control", False):
        from src.pipeline.run_control import run_control
        logging.info("[SYSTEM] Running control subsystem...")
        run_control(config, path)
    else:
        logging.info("[SYSTEM] Control skipped")

    # -----------------------------
    # DETECTION
    # -----------------------------
    if system_cfg.get("run_detection", False):
        from src.pipeline.run_detection import run_detection
        logging.info("[SYSTEM] Running detection subsystem...")
        run_detection(config, path)
    else:
        logging.info("[SYSTEM] Detection skipped")

    # -----------------------------
    # VISUALIZE
    # -----------------------------
    if system_cfg.get("visualize", True):
        from src.visualization.visualize import plot_path
        vis_cfg = config["visualization"]
        plot_pointing = vis_cfg.get("plot_pointing", False)
        pointing_scale = vis_cfg.get("pointing_scale", 2.0)
        plot_coverage = vis_cfg.get("plot_coverage", False)  # NEW

        # Case 1: We have optimized positions
        if system_cfg.get("run_optimization", False) and 'waypoint_states' in locals() and waypoint_states is not None:
            logging.info("[SYSTEM] Visualizing optimized trajectory...")

            # Extract optimized positions & velocities
            opt_positions = np.array([st.pos for st in waypoint_states])
            opt_pointing = pointing_vectors   # already computed earlier

            plot_path(
                mesh,
                opt_positions,
                None,   # sequentially connected
                vp_size=vis_cfg["viewpoint_size"],
                plot_normals=vis_cfg["plot_normals"],
                normal_length=vis_cfg["normal_length"],
                plot_projections=vis_cfg["plot_projections"],
                projection_subsample=vis_cfg["projection_subsample"],
                pointing_vectors=opt_pointing if plot_pointing else None,
                pointing_scale=pointing_scale,
                covered_faces=covered_faces if plot_coverage else None,  # NEW
            )

        # Case 2: No optimization but have safe collision-free path
        elif mesh is not None and safe_waypoints is not None:
            logging.info("[SYSTEM] Visualizing collision-free path...")
            vp_vis = safe_waypoints

            plot_path(
                mesh,
                vp_vis,
                None,
                vp_size=vis_cfg["viewpoint_size"],
                plot_normals=vis_cfg["plot_normals"],
                normal_length=vis_cfg["normal_length"],
                plot_projections=vis_cfg["plot_projections"],
                projection_subsample=vis_cfg["projection_subsample"],
                pointing_vectors=pointing_vectors if plot_pointing else None,
                pointing_scale=pointing_scale,
                covered_faces=covered_faces if plot_coverage else None,  # NEW
            )

        # Case 3: Legacy fallback
        elif mesh is not None and viewpoints is not None and path is not None:
            logging.info("[SYSTEM] Visualizing original viewpoint path...")
            plot_path(
                mesh,
                viewpoints,
                path,
                vp_size=vis_cfg["viewpoint_size"],
                plot_normals=vis_cfg["plot_normals"],
                normal_length=vis_cfg["normal_length"],
                plot_projections=vis_cfg["plot_projections"],
                projection_subsample=vis_cfg["projection_subsample"],
                pointing_vectors=None,
                pointing_scale=pointing_scale,
                covered_faces=covered_faces if plot_coverage else None,  # NEW
            )

    return {
        "mesh": mesh,
        "viewpoints": viewpoints,
        "path": path,
        "vis_matrix": vis_matrix,
        "coverage": coverage,
        "covered_faces": covered_faces,  # NEW
        "safe_waypoints": safe_waypoints,
        "trajectory": trajectory,
        "pointing_vectors": pointing_vectors,
    }

# src/pipeline/run_full_pipeline.py

import numpy as np
import logging
from src.pipeline.run_geometry import run_geometry
from src.pipeline.run_viewpoints import run_viewpoints
# import run_control, run_detection only if needed

def run_full_pipeline(config):
    system_cfg = config["system"]

    mesh = None
    centroids = None
    normals = None
    viewpoints = None
    path = None
    safe_waypoints = None
    trajectory = None

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
        viewpoints, path = run_viewpoints(config, mesh, centroids, normals)
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
            safe_waypoints, trajectory = run_collision_free_path(
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

        # Prefer collision-free path if we have it
        if mesh is not None and safe_waypoints is not None:
            logging.info("[SYSTEM] Visualizing collision-free path...")
            vp_vis = safe_waypoints
            path_vis = np.arange(vp_vis.shape[0])

            plot_path(
                mesh,
                vp_vis,
                path_vis,
                vp_size=config["visualization"]["viewpoint_size"],
                plot_normals=config["visualization"]["plot_normals"],
                normal_length=config["visualization"]["normal_length"],
                plot_projections=config["visualization"]["plot_projections"],
                projection_subsample=config["visualization"]["projection_subsample"],
                orientations=trajectory.orientations,
            )

        elif mesh is not None and viewpoints is not None and path is not None:
            logging.info("[SYSTEM] Visualizing original viewpoint path...")
            plot_path(
                mesh,
                viewpoints,
                path,
                vp_size=config["visualization"]["viewpoint_size"],
                plot_normals=config["visualization"]["plot_normals"],
                normal_length=config["visualization"]["normal_length"],
                plot_projections=config["visualization"]["plot_projections"],
                projection_subsample=config["visualization"]["projection_subsample"],
            )
        else:
            logging.warning("[SYSTEM] Not enough data to visualize.")

# Optional: allow running directly
if __name__ == "__main__":
    import yaml
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)

    logging.basicConfig(
        filename=config["logging_file"],
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info("Starting pipeline...")
    run_full_pipeline(config)
    logging.info("Pipeline finished.")
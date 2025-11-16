""" handles all visualization (spheres, lines, mesh display) """

import plotly.graph_objects as go
import numpy as np
from scipy.spatial import cKDTree

def _quat_to_rotmat(q):
    """
    Quaternion [w, x, y, z] -> 3x3 rotation matrix (world <- body).
    """
    q = np.asarray(q, dtype=float)
    w, x, y, z = q
    # standard formula
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)]
    ])
    return R

def plot_path(mesh,
            viewpoints,
            path,
            config=None,
            vp_size=None,
            plot_normals=None,
            normal_length=None,
            plot_projections=None,
            projection_subsample=None,
            orientations=None,
            plot_pointing=None,
            pointing_length=None,
            pointing_subsample=None,
):
    """
    Plots:
    - Mesh
    - Viewpoints
    - Shortest path (TSP)
    - Optional: Triangle normals as cones
    - Optional: Lines from centroids → nearest viewpoint (projection rays)
    - Optional: Attitude pointing direction at each waypoint
    """

    # -----------------------
    # Extract config if available
    # -----------------------
    if config is not None:
        viz_cfg = config.get("visualization", {})
        vp_size = vp_size if vp_size is not None else viz_cfg.get("viewpoint_size", 5)
        plot_normals = plot_normals if plot_normals is not None else viz_cfg.get("plot_normals", False)
        normal_length = normal_length if normal_length is not None else viz_cfg.get("normal_length", 1.0)
        plot_projections = plot_projections if plot_projections is not None else viz_cfg.get("plot_projections", True)
        projection_subsample = projection_subsample if projection_subsample is not None else viz_cfg.get("projection_subsample", 50)
        plot_pointing = plot_pointing if plot_pointing is not None else viz_cfg.get("plot_pointing", False)
        pointing_length = pointing_length if pointing_length is not None else viz_cfg.get("pointing_length", 2.0)
        pointing_subsample = pointing_subsample if pointing_subsample is not None else viz_cfg.get("pointing_subsample", 1)

        traj_cfg = config.get("trajectory", {})
        forward_axis = traj_cfg.get("forward_axis", "z")
        if forward_axis == "x":
            forward_body = np.array([1.0, 0.0, 0.0], dtype=float)
        elif forward_axis == "y":
            forward_body = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            forward_body = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        vp_size = vp_size or 5
        plot_normals = plot_normals or False
        normal_length = normal_length or 1.0
        plot_projections = plot_projections or True
        projection_subsample = projection_subsample or 50
        plot_pointing = True if plot_pointing is None else bool(plot_pointing)
        pointing_length = pointing_length or 2.0
        pointing_subsample = pointing_subsample or 1

        forward_body = np.array([0.0, 0.0, 1.0], dtype=float)

    # -----------------------
    # Mesh
    # -----------------------
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    x, y, z = vertices[:,0], vertices[:,1], vertices[:,2]
    i, j, k = triangles[:,0], triangles[:,1], triangles[:,2]

    fig = go.Figure(
        data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="lightgrey", opacity=0.5)]
    )

    # -----------------------
    # Viewpoints
    # -----------------------
    vp = np.array(viewpoints)
    fig.add_trace(
        go.Scatter3d(
            x=vp[:, 0],
            y=vp[:, 1],
            z=vp[:, 2],
            mode="markers",
            marker=dict(size=vp_size, color="red"),
            name="Viewpoints"
        )
    )

    # -----------------------
    # Path
    # -----------------------
    path_coords = vp[path]
    fig.add_trace(
        go.Scatter3d(
            x=path_coords[:, 0],
            y=path_coords[:, 1],
            z=path_coords[:, 2],
            mode="lines",
            line=dict(color="blue", width=4),
            name="TSP Path"
        )
    )

    # -----------------------
    # Normals (cones)
    # -----------------------
    if plot_normals:
        normals = np.asarray(mesh.triangle_normals)
        centroids = vertices[triangles].mean(axis=1)
        fig.add_trace(go.Cone(
            x=centroids[:,0],
            y=centroids[:,1],
            z=centroids[:,2],
            u=normals[:,0]*normal_length,
            v=normals[:,1]*normal_length,
            w=normals[:,2]*normal_length,
            colorscale='Blues',
            showscale=False,
            sizemode='absolute',
            sizeref=0.5,
            name="Normals"
        ))

    # -----------------------
    # Projection rays: centroid -> nearest viewpoint
    # -----------------------
    if plot_projections:
        centroids = vertices[triangles].mean(axis=1)
        tree = cKDTree(vp)  # KD-tree for fast nearest neighbor search
        num_centroids = len(centroids)
        for idx in range(0, num_centroids, projection_subsample):
            centroid = centroids[idx]
            dist, vp_idx = tree.query(centroid)  # closest viewpoint
            viewpoint = vp[vp_idx]
            fig.add_trace(
                go.Scatter3d(
                    x=[centroid[0], viewpoint[0]],
                    y=[centroid[1], viewpoint[1]],
                    z=[centroid[2], viewpoint[2]],
                    mode="lines",
                    line=dict(color="green", width=2),
                    name="Projection" if idx==0 else None
                )
            )

    # -----------------------
    # Pointing vectors (unit direction from attitude)
    # -----------------------
    if plot_pointing and orientations is not None:
        orientations = np.asarray(orientations, dtype=float)
        if orientations.shape[0] != vp.shape[0]:
            # If the sizes don't match, better to silently skip than plot wrong stuff
            pass
        else:
            xs, ys, zs = [], [], []

            N = vp.shape[0]
            for idx in range(0, N, pointing_subsample):
                p = vp[idx]
                q = orientations[idx]
                R = _quat_to_rotmat(q)
                f_world = R @ forward_body  # body-forward axis in world frame
                p_end = p + pointing_length * f_world

                xs.extend([p[0], p_end[0]])
                ys.extend([p[1], p_end[1]])
                zs.extend([p[2], p_end[2]])

            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color="orange", width=3),
                    name="Pointing direction",
                )
            )

    # -----------------------
    # Final layout
    # -----------------------
    fig.update_layout(
        scene=dict(aspectmode='data'),
        legend=dict(itemsizing='constant')
    )
    fig.show()

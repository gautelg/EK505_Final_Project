""" handles all visualization (spheres, lines, mesh display) """

import plotly.graph_objects as go
import numpy as np
from scipy.spatial import cKDTree

def plot_path(mesh, viewpoints, path, config=None,
              vp_size=None, plot_normals=None, normal_length=None,
              plot_projections=None, projection_subsample=None):
    """
    Plots:
    - Mesh
    - Viewpoints
    - Shortest path (TSP)
    - Optional: Triangle normals as cones
    - Optional: Lines from centroids → nearest viewpoint (projection rays)
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
    else:
        vp_size = vp_size or 5
        plot_normals = plot_normals or False
        normal_length = normal_length or 1.0
        plot_projections = plot_projections or True
        projection_subsample = projection_subsample or 50

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
    # Final layout
    # -----------------------
    fig.update_layout(
        scene=dict(aspectmode='data'),
        legend=dict(itemsizing='constant')
    )
    fig.show()

""" handles all visualization (mesh, viewpoints, path, normals, projection rays, pointing arrows) """

import plotly.graph_objects as go
import numpy as np
from scipy.spatial import cKDTree


def plot_path(
    mesh,
    viewpoints,
    path,
    vp_size,
    plot_normals,
    normal_length,
    plot_projections,
    projection_subsample,
    pointing_vectors=None,
    pointing_scale=1.0,
):
    """
    Plots:
    - Mesh
    - Viewpoints
    - Shortest path (TSP)
    - Optional: Triangle normals as cones
    - Optional: Lines from centroids → nearest viewpoint (projection rays)
    - Optional: Pointing vectors as arrows from each viewpoint
    """

    # -----------------------
    # Mesh
    # -----------------------
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color="lightgrey",
                opacity=0.5,
                name="Mesh",
            )
        ]
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
            name="Viewpoints",
        )
    )

    # -----------------------
    # Pointing arrows (viewpoint -> viewpoint + pointing_scale * d)
    # -----------------------
    if pointing_vectors is not None:
        viewpoints_arr = np.asarray(viewpoints, dtype=float)
        pointing_arr = np.asarray(pointing_vectors, dtype=float)

        if pointing_arr.shape != viewpoints_arr.shape:
            raise ValueError("pointing_vectors must have same shape as viewpoints (M, 3)")

        xs, ys, zs = [], [], []
        for i in range(viewpoints_arr.shape[0]):
            p = viewpoints_arr[i]
            d = pointing_arr[i]
            p_end = p + pointing_scale * d

            xs.extend([p[0], p_end[0], np.nan])
            ys.extend([p[1], p_end[1], np.nan])
            zs.extend([p[2], p_end[2], np.nan])

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="red", width=3),
                name="Pointing",
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
            name="TSP Path",
        )
    )

    # -----------------------
    # Normals (cones)
    # -----------------------
    if plot_normals:
        normals = np.asarray(mesh.triangle_normals)
        centroids = vertices[triangles].mean(axis=1)
        fig.add_trace(
            go.Cone(
                x=centroids[:, 0],
                y=centroids[:, 1],
                z=centroids[:, 2],
                u=normals[:, 0] * normal_length,
                v=normals[:, 1] * normal_length,
                w=normals[:, 2] * normal_length,
                colorscale="Blues",
                showscale=False,
                sizemode="absolute",
                sizeref=0.5,
                name="Normals",
            )
        )

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
                    name="Projection" if idx == 0 else None,
                )
            )

    # -----------------------
    # Final layout
    # -----------------------
    fig.update_layout(
        scene=dict(aspectmode="data"),
        legend=dict(itemsizing="constant"),
    )
    fig.show()

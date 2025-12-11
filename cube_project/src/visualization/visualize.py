""" handles all visualization (mesh, viewpoints, path, normals, projection rays, pointing arrows) """

import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from scipy.spatial import cKDTree

# Force Plotly to write a static HTML file and open it in your default browser
pio.renderers.default = "browser"

def plot_path(
    mesh,
    viewpoints,
    path=None,
    vp_size=3,
    plot_normals=False,
    normal_length=1.0,
    plot_projections=False,
    projection_subsample=10,
    pointing_vectors=None,
    pointing_scale=1.0,
    covered_faces=None,
):
    """
    Plots:
    - Mesh (uniform grey)
    - Viewpoints
    - Shortest path (TSP)
    - Optional: Triangle normals as cones
    - Optional: Lines from centroids → nearest viewpoint (projection rays)
    - Optional: Pointing vectors as arrows from each viewpoint
    - Optional: Coverage as red markers at centroids of uncovered faces
    """

    # -----------------------
    # Mesh (always uniform)
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
    # Coverage as centroids of uncovered faces
    # -----------------------
    if covered_faces is not None:
        covered_faces = np.asarray(covered_faces, dtype=bool)
        n_triangles = triangles.shape[0]
        if covered_faces.shape[0] != n_triangles:
            raise ValueError(
                f"covered_faces length {covered_faces.shape[0]} "
                f"does not match number of triangles {n_triangles}"
            )

        # Compute triangle centroids
        centroids = vertices[triangles].mean(axis=1)

        # Focus on uncovered faces
        uncovered_mask = ~covered_faces
        uncovered_centroids = centroids[uncovered_mask]

        # Subsample to avoid murdering the browser on huge meshes
        max_points = 10000  # tweak as needed
        if uncovered_centroids.shape[0] > max_points:
            idx = np.linspace(
                0, uncovered_centroids.shape[0] - 1, max_points
            ).astype(int)
            uncovered_centroids = uncovered_centroids[idx]

        if uncovered_centroids.size > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=uncovered_centroids[:, 0],
                    y=uncovered_centroids[:, 1],
                    z=uncovered_centroids[:, 2],
                    mode="markers",
                    marker=dict(size=2, color="red"),
                    name="Uncovered faces",
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
        for idx in range(viewpoints_arr.shape[0]):
            p = viewpoints_arr[idx]
            d = pointing_arr[idx]
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
    if path is None:
        # Draw simple connected polyline through viewpoints
        fig.add_trace(
            go.Scatter3d(
                x=viewpoints[:, 0],
                y=viewpoints[:, 1],
                z=viewpoints[:, 2],
                mode="lines",
                line=dict(color="blue", width=4),
                name="Path",
            )
        )
    else:
        # Legacy mode: path is an index list into viewpoints
        path_coords = viewpoints[path]
        fig.add_trace(
            go.Scatter3d(
                x=path_coords[:, 0],
                y=path_coords[:, 1],
                z=path_coords[:, 2],
                mode="lines",
                line=dict(color="blue", width=4),
                name="Path",
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
            _, vp_idx = tree.query(centroid)  # closest viewpoint
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

    # Write a self-contained HTML file and open it
    out_path = "output/coverage_vis.html"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path, auto_open=True)

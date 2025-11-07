import open3d as o3d
import numpy as np
from scipy.spatial.distance import cdist
import json
import time
import os

# ================================================================
# 1️⃣ LOAD MESH AND COMPUTE TRIANGLE CENTROIDS
# ================================================================
stl_path = r"E:\EK505_introToRobotics\finalProject\EK505_Final_Project\Station_Model\iss_wt_simplified.obj"
print("[INFO] Mesh exists:", os.path.exists(stl_path))

mesh = o3d.io.read_triangle_mesh(stl_path)
mesh.compute_vertex_normals()
print("[INFO] Mesh loaded. Triangles:", len(mesh.triangles))

triangles = np.asarray(mesh.triangles)
vertices = np.asarray(mesh.vertices)
centroids = vertices[triangles].mean(axis=1)
print("[INFO] Number of centroids:", len(centroids))

# ================================================================
# 2️⃣ GENERATE CANDIDATE VIEWPOINTS (FIBONACCI SPHERE)
# ================================================================
def fibonacci_sphere(samples, radius):
    points = []
    offset = 2.0 / samples
    increment = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y*y)
        phi = ((i % samples) * increment)
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([radius*x, radius*y, radius*z])
    return np.array(points)

viewpoints = fibonacci_sphere(samples=500, radius=120)
print("[INFO] Generated candidate viewpoints:", len(viewpoints))

# ================================================================
# 3️⃣ CREATE TENSOR-BASED RAYCASTING SCENE
# ================================================================
t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(t_mesh)
print("[INFO] Tensor RaycastingScene created.")

centroid_tensor = o3d.core.Tensor(centroids, dtype=o3d.core.Dtype.Float32)
num_centroids = len(centroids)

# ================================================================
# 4️⃣ VISIBILITY CHECK USING BATCH cast_rays
# ================================================================
visibility = []
start_time = time.time()
print("[INFO] Starting visibility computation...")

for j, vp in enumerate(viewpoints):
    origins = np.tile(vp, (num_centroids, 1)).astype(np.float32)
    directions = centroids - vp
    dists = np.linalg.norm(directions, axis=1)
    directions /= dists[:, np.newaxis]

    # Open3D cast_rays expects a single Tensor of shape (N, 6): [origin_x,y,z, dir_x,y,z]
    rays_tensor = o3d.core.Tensor(np.hstack((origins, directions)), dtype=o3d.core.Dtype.Float32)
    results = scene.cast_rays(rays_tensor)

    t_hit = results['t_hit'].numpy()
    prim_ids = results['primitive_ids'].numpy()

    # Centroid is visible if the hit distance is smaller than true distance and primitive matches
    visible_idx = np.where((t_hit < dists) & (prim_ids == np.arange(num_centroids)))[0]
    visibility.append(visible_idx.tolist())

print(f"[INFO] Visibility computed in {time.time() - start_time:.2f} sec.")

# ================================================================
# 5️⃣ GREEDY SET COVER — REMOVE REDUNDANT VIEWPOINTS
# ================================================================
needed = set(range(num_centroids))
selected_viewpoints = []

while needed:
    best_vp = max(range(len(viewpoints)),
                  key=lambda j: len(needed.intersection(visibility[j])))
    selected_viewpoints.append(best_vp)
    needed -= set(visibility[best_vp])

optimized_vps = viewpoints[selected_viewpoints]
print(f"[INFO] Reduced {len(viewpoints)} candidates -> {len(optimized_vps)} essential viewpoints.")

# ================================================================
# 6️⃣ SOLVE TSP USING NEAREST NEIGHBOR
# ================================================================
dist_matrix = cdist(optimized_vps, optimized_vps)

def tsp_nearest_neighbor(dist_matrix):
    n = len(dist_matrix)
    unvisited = set(range(n))
    path = [0]
    unvisited.remove(0)
    while unvisited:
        last = path[-1]
        next_node = min(unvisited, key=lambda x: dist_matrix[last, x])
        path.append(next_node)
        unvisited.remove(next_node)
    return path

path = tsp_nearest_neighbor(dist_matrix)
print("[INFO] TSP solved. Path length:", len(path))

# ================================================================
# 7️⃣ EXPORT OPTIMIZED VIEWPOINTS AND PATH TO JSON
# ================================================================
output_data = {
    "viewpoints": optimized_vps.tolist(),
    "tsp_path": path
}

output_filename = "optimized_viewpoints.json"
with open(output_filename, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"[INFO] JSON file saved: {output_filename}")

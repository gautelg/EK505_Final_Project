""" only computes visibility (logic, tensors, rays) """

import open3d as o3d
import numpy as np
import logging

class VisibilityChecker:
    def __init__(self, mesh):
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(t_mesh)
        logging.info("Raycasting scene created.")

    def check_visibility(self, viewpoints, centroids):
        visibility = []
        num_centroids = len(centroids)
        for idx, vp in enumerate(viewpoints):
            origins = np.tile(vp, (num_centroids, 1)).astype(np.float32)
            directions = centroids - vp
            dists = np.linalg.norm(directions, axis=1)
            directions /= dists[:, np.newaxis]

            rays = o3d.core.Tensor(np.hstack([origins, directions]).astype(np.float32))
            results = self.scene.cast_rays(rays)
            t_hit = results['t_hit'].numpy()
            prim_ids = results['primitive_ids'].numpy()
            visible_idx = np.where((t_hit < dists) & (prim_ids == np.arange(num_centroids)))[0]
            visibility.append(visible_idx.tolist())
            if idx % 50 == 0:
                logging.info(f"Processed visibility for viewpoint {idx}/{len(viewpoints)}")
        return visibility

import open3d as o3d
import numpy as np
import logging

class MeshLoader:
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path

    def load_mesh(self):
        mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        if not mesh.has_triangles():
            raise ValueError("Mesh has no triangles!")
        mesh.compute_vertex_normals()
        logging.info(f"Mesh loaded. Triangles: {len(mesh.triangles)}")
        return mesh

    @staticmethod
    def compute_centroids(mesh):
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        centroids = vertices[triangles].mean(axis=1)
        normals = np.asarray(mesh.triangle_normals)
        logging.info(f"Computed centroids: {len(centroids)}")
        return centroids, normals

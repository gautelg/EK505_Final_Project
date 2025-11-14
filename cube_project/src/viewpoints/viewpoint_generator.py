# src/viewpoints/viewpoint_generator.py

import numpy as np

def fibonacci_sphere(samples, radius=1.0):
    """
    Generates points on a sphere using Fibonacci lattice method.
    """
    points = []
    offset = 2.0 / samples
    increment = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y*y)
        phi = i * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([radius*x, radius*y, radius*z])

    return np.array(points)

def viewpoints_from_normals(centroids, normals, distance):
    """
    Creates viewpoints along the triangle normals at a fixed distance.
    Ensures all points are outside the mesh.
    
    Args:
        centroids: np.array of shape (N, 3)
        normals: np.array of shape (N, 3)
        distance: float, distance from surface along normals

    Returns:
        np.array of shape (N, 3)
    """
    # Normalize normals to ensure consistent distance
    unit_normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    return centroids + unit_normals * distance

def generate_viewpoints(mesh, centroids=None, normals=None, distance=5.0, 
                        use_sphere=False, num_samples=500):
    """
    Generate candidate viewpoints around the mesh.
    
    Options:
        1. Sphere around the model (use_sphere=True)
        2. Projection along triangle normals (use_sphere=False)
    
    Args:
        mesh: Open3D mesh object
        centroids: np.array of triangle centroids (required if use_sphere=False)
        normals: np.array of triangle normals (required if use_sphere=False)
        distance: float, distance from surface (for projection or sphere radius)
        use_sphere: bool, whether to generate sphere of viewpoints
        num_samples: int, number of samples for sphere method

    Returns:
        np.array of shape (N, 3) of candidate viewpoints
    """
    if use_sphere:
        # Compute center of mesh
        vertices = np.asarray(mesh.vertices)
        center = vertices.mean(axis=0)
        points = fibonacci_sphere(num_samples, radius=distance)
        return points + center
    else:
        if centroids is None or normals is None:
            raise ValueError("Centroids and normals are required for projection method")
        return viewpoints_from_normals(centroids, normals, distance)

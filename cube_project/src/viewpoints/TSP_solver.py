import numpy as np
import logging
from scipy.spatial.distance import cdist
from itertools import permutations


def solve_tsp(points, method="nearest_neighbor"):
    """
    Entry point for solving the Traveling Salesman Problem (TSP) for a set of 3D points.

    Parameters
    ----------
    points : np.ndarray
        Nx3 array of viewpoint coordinates.
    method : str
        "nearest_neighbor" (fast heuristic) or "brute_force" (exact, small N only).

    Returns
    -------
    path : list[int]
        Ordered list of indices representing the optimized visit order.
    """
    if points is None or len(points) == 0:
        logging.warning("[TSP] No points provided, returning empty path.")
        return []

    if method == "nearest_neighbor":
        return tsp_nearest_neighbor(points)
    elif method == "brute_force":
        return tsp_brute_force(points)
    else:
        logging.warning(f"[TSP] Unknown method '{method}', defaulting to nearest_neighbor.")
        return tsp_nearest_neighbor(points)


def tsp_nearest_neighbor(points):
    """
    Greedy nearest-neighbor heuristic for TSP.
    """
    points = np.asarray(points)
    n = len(points)
    if n == 1:
        return [0]

    dist_matrix = cdist(points, points)
    unvisited = set(range(n))
    path = [0]
    unvisited.remove(0)

    while unvisited:
        last = path[-1]
        next_node = min(unvisited, key=lambda x: dist_matrix[last, x])
        path.append(next_node)
        unvisited.remove(next_node)

    logging.info(f"[TSP] Nearest-neighbor path computed with {len(path)} waypoints.")
    return path


def tsp_brute_force(points):
    """
    Brute-force TSP for very small sets (<=10 points).
    """
    points = np.asarray(points)
    n = len(points)
    if n > 10:
        logging.warning("[TSP] Too many points for brute-force; using nearest_neighbor instead.")
        return tsp_nearest_neighbor(points)

    best_path = None
    best_cost = np.inf

    for perm in permutations(range(n)):
        cost = np.sum(np.linalg.norm(points[list(perm)[1:]] - points[list(perm)[:-1]], axis=1))
        if cost < best_cost:
            best_cost = cost
            best_path = perm

    logging.info(f"[TSP] Brute-force path computed (cost={best_cost:.3f}).")
    return list(best_path)

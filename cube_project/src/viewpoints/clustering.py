import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import logging

def cluster_viewpoints(viewpoints, method="dbscan", eps=1.0, min_samples=1, n_clusters=10):
    """
    Reduce number of viewpoints by clustering nearby viewpoints.
    Returns cluster centroids.
    """
    if method == "dbscan":
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(viewpoints)
        labels = clustering.labels_
        unique_labels = set(labels)
        cluster_centers = []
        for lbl in unique_labels:
            if lbl == -1:
                # Noise, keep as-is
                cluster_centers.extend(viewpoints[labels == lbl])
            else:
                cluster_centers.append(viewpoints[labels == lbl].mean(axis=0))
        cluster_centers = np.array(cluster_centers)
        logging.info(f"DBSCAN reduced {len(viewpoints)} -> {len(cluster_centers)} viewpoints")
        return cluster_centers

    elif method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(viewpoints)
        cluster_centers = kmeans.cluster_centers_
        logging.info(f"KMeans reduced {len(viewpoints)} -> {len(cluster_centers)} viewpoints")
        return cluster_centers

    else:
        logging.warning(f"Unknown clustering method {method}, returning original viewpoints")
        return viewpoints

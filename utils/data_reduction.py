# utils/data_reduction.py
"""
data_reduction.py
-----------------
Implements the K-means + Spectral Angle Mapper (SAM) data reduction
strategy described in the benchmark paper to reduce redundant pixels
in the training set and speed up model training.
"""

import numpy as np
from sklearn.cluster import KMeans

def spectral_angle_mapper(x, y):
    """Compute the spectral angle (in radians) between two spectra."""
    num = np.dot(x, y)
    denom = np.linalg.norm(x) * np.linalg.norm(y) + 1e-8
    return np.arccos(np.clip(num / denom, -1, 1))

def reduce_training_data(X, y, cfg):
    """
    Perform K-means reduction independently per class:
      - K clusters per class (cfg.reduction.clusters_per_class)
      - select n pixels nearest to each centroid by SAM distance
      - total ≈ K * n pixels per class

    Returns reduced X_red, y_red arrays.
    """
    if not getattr(cfg.reduction, "enabled", False):
        return X, y

    K = cfg.reduction.clusters_per_class
    n_per_centroid = cfg.reduction.pixels_per_centroid
    unique_classes = np.unique(y)
    rng = np.random.default_rng(cfg.partition.random_seed)

    X_reduced, y_reduced = [], []

    for c in unique_classes:
        X_c = X[y == c]
        if len(X_c) < K:
            X_reduced.append(X_c)
            y_reduced.append(np.full(len(X_c), c))
            continue

        # --- 1. K-means clustering on this class ---
        km = KMeans(n_clusters=K, n_init=5, random_state=cfg.partition.random_seed)
        labels = km.fit_predict(X_c)
        centroids = km.cluster_centers_

        # --- 2. For each centroid, select n closest by SAM ---
        for i in range(K):
            cluster_points = X_c[labels == i]
            if len(cluster_points) == 0:
                continue
            angles = np.array([spectral_angle_mapper(centroids[i], p) for p in cluster_points])
            nearest_idx = np.argsort(angles)[:n_per_centroid]
            selected = cluster_points[nearest_idx]
            X_reduced.append(selected)
            y_reduced.append(np.full(len(selected), c))

    X_red = np.vstack(X_reduced)
    y_red = np.concatenate(y_reduced)
    print(f"[Data Reduction] Reduced training set to {len(y_red)} pixels "
          f"({K} clusters × {n_per_centroid} per class).")
    
    classes, counts = np.unique(y_red, return_counts=True)
    for c, n in zip(classes, counts):
        print(f"[Class {c}] {n} pixels after reduction")
    return X_red, y_red

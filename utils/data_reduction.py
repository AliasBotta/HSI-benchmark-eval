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


# ============================================================
# Spectral Angle Mapper
# ============================================================

def spectral_angle_mapper(x, y):
    """Compute the spectral angle (in radians) between two spectra."""
    num = np.dot(x, y)
    denom = np.linalg.norm(x) * np.linalg.norm(y) + 1e-8
    return np.arccos(np.clip(num / denom, -1, 1))


# ============================================================
# Data Reduction (Config-free)
# ============================================================

def reduce_training_data(
    X,
    y,
    enabled=True,
    clusters_per_class=100,
    voxels_per_centroid=20,
    random_seed=42,
):
    """
    Perform K-means reduction independently per class:
      - K clusters per class (clusters_per_class)
      - select n pixels nearest to each centroid by SAM distance
      - total ≈ K * n pixels per class

    Returns reduced X_red, y_red arrays.

    Parameters
    ----------
    X : np.ndarray
        Spectral features (N × D)
    y : np.ndarray
        Integer class labels (N,)
    enabled : bool
        If False, returns X, y unchanged.
    clusters_per_class : int
        Number of K-means clusters per class.
    voxels_per_centroid : int
        Number of samples to keep per cluster (closest by SAM).
    random_seed : int
        Random seed for reproducibility.
    """
    if not enabled:
        print("[Data Reduction] Disabled. Returning original dataset.")
        return X, y

    rng = np.random.default_rng(random_seed)
    unique_classes = np.unique(y)

    X_reduced, y_reduced = [], []

    for c in unique_classes:
        X_c = X[y == c]
        if len(X_c) < clusters_per_class:
            # If class too small, keep all pixels
            X_reduced.append(X_c)
            y_reduced.append(np.full(len(X_c), c))
            continue

        # --- 1. K-means clustering on this class ---
        km = KMeans(n_clusters=clusters_per_class, n_init=5, random_state=random_seed)
        labels = km.fit_predict(X_c)
        centroids = km.cluster_centers_

        # --- 2. For each centroid, select n closest by SAM ---
        for i in range(clusters_per_class):
            cluster_points = X_c[labels == i]
            if len(cluster_points) == 0:
                continue
            angles = np.array([spectral_angle_mapper(centroids[i], p) for p in cluster_points])
            nearest_idx = np.argsort(angles)[:voxels_per_centroid]
            selected = cluster_points[nearest_idx]
            X_reduced.append(selected)
            y_reduced.append(np.full(len(selected), c))

    X_red = np.vstack(X_reduced)
    y_red = np.concatenate(y_reduced)

    print(f"[Data Reduction] Reduced training set to {len(y_red)} pixels "
          f"({clusters_per_class} clusters × {voxels_per_centroid} per class).")

    classes, counts = np.unique(y_red, return_counts=True)
    for c, n in zip(classes, counts):
        print(f"[Class {c}] {n} pixels after reduction")

    return X_red, y_red

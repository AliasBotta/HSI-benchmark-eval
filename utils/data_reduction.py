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

def _sam_distances_to_centroid(centroid, points):
    """
    Vectorized SAM distance between a centroid and a set of points.
    Returns angles in radians shape: (N_points,)
    """
    # Normalize to avoid repeated norms in the loop
    c = centroid / (np.linalg.norm(centroid) + 1e-8)
    p = points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-8)
    cosang = np.clip(np.sum(p * c[None, :], axis=1), -1.0, 1.0)
    return np.arccos(cosang)

def reduce_training_data(
    X,
    y,
    enabled=True,
    clusters_per_class=100,
    target_per_class=1000,
    random_seed=42,
):
    """
    Perform K-means reduction independently per class following the paper:
      - K clusters per class (clusters_per_class, default 100)
      - From each cluster, pick n = floor(target_per_class / K) pixels
        closest to the cluster centroid using SAM.
      - Ensures per-class cap around target_per_class (may be slightly less
        depending on empty/small clusters).

    Returns reduced X_red, y_red arrays.

    Parameters
    ----------
    X : np.ndarray
        Spectral features (N × D)
    y : np.ndarray
        Integer class labels (N,) in {0:NT, 1:TT, 2:BV, 3:BG}
    enabled : bool
        If False, returns X, y unchanged.
    clusters_per_class : int
        Number of K-means clusters per class (paper uses 100).
    target_per_class : int
        Target number of samples per class after reduction
        (paper investigates 1000, 2000, 4000; recommend 1000).
    random_seed : int
        Random seed for reproducibility.
    """
    if not enabled:
        print("[Data Reduction] Disabled. Returning original dataset.")
        return X, y

    if target_per_class <= 0:
        raise ValueError("target_per_class must be > 0")

    per_cluster = max(1, target_per_class // clusters_per_class)

    rng = np.random.default_rng(random_seed)
    unique_classes = np.unique(y)

    X_reduced, y_reduced = [], []

    for c in unique_classes:
        X_c = X[y == c]
        n_c = len(X_c)

        if n_c == 0:
            continue

        if n_c <= target_per_class:
            # Not enough pixels to reduce; keep all
            X_reduced.append(X_c)
            y_reduced.append(np.full(n_c, c))
            print(f"[Data Reduction] Class {c}: kept all {n_c} (<= target {target_per_class}).")
            continue

        # --- 1) K-means per class ---
        k = min(clusters_per_class, n_c)  # guard for very small classes
        km = KMeans(n_clusters=k, n_init=10, random_state=random_seed)
        labels = km.fit_predict(X_c)
        centroids = km.cluster_centers_

        # --- 2) Select nearest-by-SAM per centroid ---
        selected_idxs = []
        for i in range(k):
            cluster_idx = np.where(labels == i)[0]
            if cluster_idx.size == 0:
                continue
            pts = X_c[cluster_idx]
            angles = _sam_distances_to_centroid(centroids[i], pts)
            take = min(per_cluster, cluster_idx.size)
            local_sel = cluster_idx[np.argpartition(angles, take - 1)[:take]]
            selected_idxs.append(local_sel)

        if selected_idxs:
            sel = np.concatenate(selected_idxs, axis=0)
            # Cap to target_per_class if we got more due to rounding
            if sel.size > target_per_class:
                sel = rng.choice(sel, size=target_per_class, replace=False)
            X_reduced.append(X_c[sel])
            y_reduced.append(np.full(sel.size, c))
            print(f"[Data Reduction] Class {c}: {sel.size} / target {target_per_class} selected "
                  f"({k} clusters × {per_cluster}).")
        else:
            # Fallback: random subset if k-means produced empty clusters only
            take = min(target_per_class, n_c)
            sel = rng.choice(n_c, size=take, replace=False)
            X_reduced.append(X_c[sel])
            y_reduced.append(np.full(take, c))
            print(f"[Data Reduction] Class {c}: fallback random {take}.")

    X_red = np.vstack(X_reduced) if X_reduced else np.empty((0, X.shape[1]))
    y_red = np.concatenate(y_reduced) if y_reduced else np.empty((0,), dtype=int)

    print(f"[Data Reduction] Reduced training set to {len(y_red)} pixels "
          f"(target {target_per_class} per class).")
    classes, counts = np.unique(y_red, return_counts=True)
    for c, n in zip(classes, counts):
        print(f"[Class {c}] {n} pixels after reduction")

    return X_red, y_red
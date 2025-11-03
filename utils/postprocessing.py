"""
postprocessing.py
-----------------
Implements the Hierarchical segmentation + Majority Voting fusion step
from the HSI-benchmark paper.

This step smooths pixel-level predictions spatially using segmentation
on the PCA(1) (or full cube), followed by majority voting within each segment.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from .hkm_nmf import h2nmf_segmentation


def hierarchical_kmeans(X, target_clusters=24, random_state=0):
    """
    Perform Hierarchical K-Means clustering recursively until reaching
    approximately 'target_clusters' clusters.

    Parameters
    ----------
    X : np.ndarray
        (N, D) feature matrix (e.g., reduced spectral data)
    target_clusters : int
        Desired number of total clusters.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    labels : np.ndarray
        (N,) cluster labels (0..target_clusters-1)
    """
    print(f"[Postprocessing] Running Hierarchical K-Means with target={target_clusters}...")

    clusters = [np.arange(len(X))]
    labels = np.full(len(X), -1, dtype=int)
    rng = np.random.default_rng(random_state)

    while len(clusters) < target_clusters and len(clusters) > 0:
        sizes = [len(c) for c in clusters]
        idx_max = int(np.argmax(sizes))
        current = clusters.pop(idx_max)
        if len(current) < 2:
            continue

        km = KMeans(n_clusters=2, n_init=5, random_state=rng.integers(0, 1e6))
        sub_labels = km.fit_predict(X[current])

        sub_idx_0 = current[sub_labels == 0]
        sub_idx_1 = current[sub_labels == 1]

        clusters.append(sub_idx_0)
        clusters.append(sub_idx_1)

    for i, c in enumerate(clusters):
        labels[c] = i

    print(f"[Postprocessing] ✅ HKM segmentation complete ({len(clusters)} clusters).")
    return labels


from sklearn.cluster import KMeans
import numpy as np

def majority_voting(knn_class_map, pc1=None, cube=None, n_clusters=24, use_h2nmf=False):
    """
    Majority voting post-processing after KNN filtering.

    Now uses Hierarchical K-means segmentation in [PC1, λ·x, λ·y] space
    to create spatially compact regions (as in the paper).
    """

    H, W = knn_class_map.shape
    if pc1 is None:
        raise ValueError("majority_voting requires pc1 (PCA(1)) image for clustering.")

    # --- Spatially-aware HKM segmentation ---
    yy, xx = np.mgrid[0:H, 0:W]
    lambda_spatial = 2.0  # spatial weighting factor (1–3 recommended)
    feats = np.stack([pc1, lambda_spatial * xx, lambda_spatial * yy], axis=-1).reshape(-1, 3)

    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = km.fit_predict(feats).reshape(H, W)

    # --- Majority vote within each cluster ---
    class_mv = np.zeros_like(knn_class_map)
    for i in range(n_clusters):
        mask = (labels == i)
        if not np.any(mask):
            continue
        vals, counts = np.unique(knn_class_map[mask], return_counts=True)
        maj = vals[np.argmax(counts)]
        class_mv[mask] = maj

    print("[Postprocessing] ✅ Majority voting fusion complete.")
    return class_mv

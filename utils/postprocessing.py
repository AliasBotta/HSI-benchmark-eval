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


def majority_voting(knn_class_map, pc1=None, cube=None, n_clusters=24, use_h2nmf=True):
    """
    Apply HKM or H2NMF segmentation + majority voting to smooth predictions.

    Parameters
    ----------
    knn_class_map : np.ndarray
        (H, W) class labels after KNN spatial filtering.
    pc1 : np.ndarray, optional
        (H, W) precomputed PCA(1) image, if available.
    cube : np.ndarray, optional
        (bands, H, W) preprocessed hyperspectral cube.
    n_clusters : int
        Number of clusters for segmentation (default = 24).
    use_h2nmf : bool
        If True, use hierarchical rank-2 NMF segmentation (closer to MATLAB).
        If False, fall back to hierarchical KMeans on PCA features.

    Returns
    -------
    mv_class_map : np.ndarray
        (H, W) array of smoothed predictions.
    """
    print("[Postprocessing] Applying Majority Voting (HKM/H2NMF)...")

    if cube is None and pc1 is None:
        print("[Postprocessing] ⚠ No cube or PC1 provided — fallback to identity (no segmentation).")
        return knn_class_map.copy()

    if use_h2nmf:
        print(f"[Postprocessing] Using hierarchical rank-2 NMF segmentation ({n_clusters} clusters).")
        cluster_map = h2nmf_segmentation(cube, target_clusters=n_clusters)
    else:
        if pc1 is not None:
            X_red = pc1.reshape(-1, 1)
        else:
            bands, H, W = cube.shape
            X = cube.reshape(bands, -1).T
            pca = PCA(n_components=min(5, bands))
            X_red = pca.fit_transform(X)
        seg_labels = hierarchical_kmeans(X_red, target_clusters=n_clusters)
        H, W = knn_class_map.shape
        cluster_map = seg_labels.reshape(H, W)

    # --- Majority voting within clusters ---
    H, W = knn_class_map.shape
    mv_class_map = np.zeros((H, W), dtype=knn_class_map.dtype)
    for cl in np.unique(cluster_map):
        idx = np.where(cluster_map == cl)
        if len(idx[0]) == 0:
            continue
        labels, counts = np.unique(knn_class_map[idx], return_counts=True)
        mv_class_map[idx] = labels[np.argmax(counts)]

    print("[Postprocessing] ✅ Majority voting fusion complete.")
    return mv_class_map

"""
spatial_filtering.py
--------------------
Implements the spatial–spectral KNN smoothing step guided by PCA(1),
as described in the HSI-benchmark paper.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import uniform_filter1d


def apply_knn_filter(prob_map, pc1=None, cube=None, K=40, lambda_=1.0,
                     window_size=8, distance="euclidean"):
    """
    Apply KNN smoothing to model probability maps guided by PCA(1) image.
    Faithful to the HSI-benchmark (2.5D spatial–spectral smoothing guided by PC1).

    Parameters
    ----------
    prob_map : np.ndarray
        (N_pixels, num_classes) probability or score map (flattened).
    pc1 : np.ndarray, optional
        (H, W) precomputed first principal component image (preferred, raw scale).
    cube : np.ndarray, optional
        (bands, H, W) preprocessed HSI cube. Used only if pc1 is not provided.
    K : int
        Number of neighbors for spatial KNN.
    lambda_ : float
        Spatial scaling factor for coordinates.
    window_size : int
        Window size for the vertical sliding window (14 by default, legacy).
    distance : str
        Distance metric for KNN ('euclidean' recommended).

    Returns
    -------
    smoothed_prob_map : np.ndarray
        Same shape as prob_map.
    """
    print("[Spatial Filtering] Applying KNN filter guided by PCA(1)...")

    # --- fallback: 1D smoothing if no cube/pc1 provided ---
    if cube is None and pc1 is None:
        print("[Spatial Filtering] ⚠ No cube or PC1 provided — applying 1D fallback smoothing.")
        return uniform_filter1d(prob_map, size=window_size, axis=0)

    # --- compute PC1 if not provided ---
    if pc1 is None:
        from sklearn.decomposition import PCA
        bands, H, W = cube.shape
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(cube.reshape(bands, -1).T).reshape(H, W)
    else:
        H, W = pc1.shape

    n_pixels = H * W
    n_classes = prob_map.shape[1]
    n_probs = prob_map.shape[0]

    # --- feature space (PC1 + spatial coordinates, 2.5D) ---
    yy, xx = np.mgrid[0:H, 0:W]
    features = np.stack([pc1.ravel(), lambda_ * xx.ravel(), lambda_ * yy.ravel()], axis=1)

    # --- adjust prob_map size to match cube (pad or crop) ---
    if n_probs < n_pixels:
        print(f"[Spatial Filtering] Padding prob_map ({n_probs} → {n_pixels}) for full-image smoothing")
        pad = np.repeat(prob_map.mean(axis=0, keepdims=True), n_pixels - n_probs, axis=0)
        prob_map_full = np.concatenate([prob_map, pad], axis=0)
    elif n_probs > n_pixels:
        print(f"[Spatial Filtering] Cropping prob_map ({n_probs} → {n_pixels}) to match cube size")
        prob_map_full = prob_map[:n_pixels]
    else:
        prob_map_full = prob_map

    # --- full-image 2D KNN smoothing (paper-compliant) ---
    print("[Spatial Filtering] Using 2D KNN smoothing in [PC1, x, y] space...")
    nn = NearestNeighbors(n_neighbors=min(K, features.shape[0]), metric=distance)
    nn.fit(features)
    neigh_idx = nn.kneighbors(features, return_distance=False)

    # Average neighbor probabilities
    smoothed_full = prob_map_full[neigh_idx].mean(axis=1)

    # --- crop back to labeled pixels only ---
    if n_probs < n_pixels:
        print(f"[Spatial Filtering] Cropped smoothed map ({n_pixels} → {n_probs})")
        smoothed = smoothed_full[:n_probs]
    else:
        smoothed = smoothed_full

    print("[Spatial Filtering] ✅ KNN smoothing complete.")
    print(f"[Spatial Filtering] Output shape: {smoothed.shape}")
    return smoothed

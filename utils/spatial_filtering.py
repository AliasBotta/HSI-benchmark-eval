
"""
spatial_filtering.py
--------------------
Implements the spatial–spectral LOCAL KNN smoothing step guided by PCA(1),
as described in the HSI-benchmark paper.

This version uses a sliding vertical window ("8 rows") 
instead of a global search.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

def apply_knn_filter(prob_map, pc1=None, cube=None, K=40, lambda_=1.0,
                       window_size=8, distance="euclidean"):
    """
    Apply LOCAL KNN smoothing to model probability maps guided by PCA(1).
    Faithful to the HSI-benchmark (2.5D spatial–spectral smoothing
    guided by PC1, using a local vertical window).

    Parameters
    ----------
    prob_map : np.ndarray
        (N_pixels, num_classes) probability map (flattened).
    pc1 : np.ndarray
        (H, W) precomputed first principal component image.
    cube : np.ndarray
        (bands, H, W) preprocessed HSI cube. Used only if pc1 is not provided.
    K : int
        Number of neighbors (K=40 in paper).
    lambda_ : float
        Spatial scaling factor (lambda=1 in paper).
    window_size : int
        Vertical window size (8 rows in paper).
    distance : str
        Distance metric ('euclidean' in paper).

    Returns
    -------
    smoothed_prob_map : np.ndarray
        Same shape as prob_map.
    """
    print(f"[Spatial Filtering] Applying LOCAL KNN filter (Window={window_size}, K={K})...")

    if pc1 is None:
        if cube is None:
            raise ValueError("KNN filter needs either 'pc1' or 'cube' input.")
        bands, H, W = cube.shape
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(cube.reshape(bands, -1).T).reshape(H, W)
    else:
        H, W = pc1.shape

    n_pixels = H * W
    n_classes = prob_map.shape[1]
    
    if prob_map.shape[0] != n_pixels:
        raise ValueError(f"Prob map shape ({prob_map.shape[0]}) does not match cube shape ({n_pixels}).")

    yy, xx = np.mgrid[0:H, 0:W]
    lambda_spatial = lambda_ * H 
    features = np.stack([
        pc1.ravel(), 
        lambda_spatial * (xx.ravel() / W), 
        lambda_spatial * (yy.ravel() / H)
    ], axis=1)

    pad = window_size // 2
    smoothed_full = np.zeros_like(prob_map)
    
    for r in range(H):
        r_start = max(0, r - pad)
        r_end = min(H, r + pad + 1) 
        
        idx_start = r_start * W
        idx_end = r_end * W
        
        features_local = features[idx_start:idx_end]
        probs_local = prob_map[idx_start:idx_end]
        
        n_neighbors_local = min(K + 1, features_local.shape[0]) 
        nn = NearestNeighbors(n_neighbors=n_neighbors_local, metric=distance, n_jobs=-1)
        nn.fit(features_local)
        
        features_current_row = features[r*W : (r+1)*W]
        
        neigh_idx = nn.kneighbors(features_current_row, return_distance=False)
        
        smoothed_probs_row = probs_local[neigh_idx].mean(axis=1)
        
        smoothed_full[r*W : (r+1)*W] = smoothed_probs_row

    print("[Spatial Filtering] ✅ LOCAL KNN smoothing complete.")
    return smoothed_full
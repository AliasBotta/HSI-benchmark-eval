import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import uniform_filter1d


def apply_knn_filter(prob_map, cube, cfg):
    """
    Apply KNN smoothing to model probability maps guided by PCA(1) image.
    Faithful to HSI-benchmark (2D spatial–spectral smoothing guided by PC1).
    Args:
        prob_map: (N_labeled, num_classes)
        cube: (bands, H, W)
        cfg: config with spatial_spectral.knn_filter params
    Returns:
        smoothed_prob_map: same shape as prob_map
    """
    print("[Spatial Filtering] Applying KNN filter guided by PCA(1)...")

    # --- fallback: 1D smoothing if no cube ---
    if cube is None:
        print("[Spatial Filtering] ⚠ No cube provided — applying 1D fallback smoothing.")
        return uniform_filter1d(prob_map, size=3, axis=0)

    bands, H, W = cube.shape
    n_pixels = H * W
    n_classes = prob_map.shape[1]
    n_probs = prob_map.shape[0]

    # --- PCA(1) image as guide ---
    pca = PCA(n_components=1)
    pca_img = pca.fit_transform(cube.reshape(bands, -1).T).reshape(H, W)

    # --- build feature space (PC1 + spatial coords) ---
    lam = getattr(cfg.spatial_spectral.knn_filter, "lambda", 1.0)
    K = getattr(cfg.spatial_spectral.knn_filter, "K", 25)

    yy, xx = np.mgrid[0:H, 0:W]
    features = np.stack([pca_img.ravel(), lam * yy.ravel(), lam * xx.ravel()], axis=1)

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

    # --- fit KNN and smooth ---
    nn = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(features)
    _, indices = nn.kneighbors(features)

    smoothed_full = np.zeros_like(prob_map_full)
    for c in range(n_classes):
        smoothed_full[:, c] = np.mean(prob_map_full[indices, c], axis=1)

    # --- crop back to labeled pixels only ---
    if n_probs < n_pixels:
        print(f"[Spatial Filtering] Cropped smoothed map ({n_pixels} → {n_probs})")
        smoothed = smoothed_full[:n_probs]
    else:
        smoothed = smoothed_full

    print("[Spatial Filtering] ✅ KNN smoothing complete.")
    print(f"[Spatial Filtering] Output shape: {smoothed.shape}")
    return smoothed

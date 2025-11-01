import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def apply_knn_filter(prob_map, cube=None, cfg=None):
    """
    Apply KNN smoothing to model probability maps guided by PCA(1) image.

    Args:
        prob_map: (N_pixels, num_classes)
        cube: (bands, H, W) or None
        cfg: config with spatial_spectral.knn_filter params
    Returns:
        smoothed_prob_map: same shape as prob_map
    """
    print("[Spatial Filtering] Applying KNN filter guided by PCA(1)...")

    # --- Config parameters ---
    K = getattr(cfg.spatial_spectral.knn_filter, "K", 40)
    lam = getattr(cfg.spatial_spectral.knn_filter, "lambda", 1.0)

    # === CASE 1: full spatial filtering (cube available) ===
    if cube is not None:
        bands, H, W = cube.shape
        cube_flat = cube.reshape(bands, -1).T

        # PCA(1) projection to get intensity map
        pca = PCA(n_components=1)
        pca_img = pca.fit_transform(cube_flat).reshape(H, W)

        # Build guided features (PCA intensity + spatial coords)
        coords = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), -1).reshape(-1, 2)
        features = np.concatenate([pca_img.flatten()[:, None], lam * coords], axis=1)

        # Fit KNN and smooth probabilities
        nn = NearestNeighbors(n_neighbors=K, metric='euclidean').fit(features)
        _, indices = nn.kneighbors(features)

        smoothed = np.zeros_like(prob_map)
        for i in range(prob_map.shape[1]):
            smoothed[:, i] = np.mean(prob_map[indices, i], axis=1)

        print(f"[Spatial Filtering] Done (K={K}, λ={lam}, cube shape={cube.shape})")
        return smoothed

    # === CASE 2: fallback (no cube provided) ===
    print("[Spatial Filtering] ⚠️ No cube provided — applying 1D fallback smoothing.")
    smoothed = prob_map.copy()
    for i in range(prob_map.shape[1]):  # per class
        smoothed[:, i] = np.convolve(prob_map[:, i], np.ones(5)/5, mode="same")
    smoothed = (1 - lam) * prob_map + lam * smoothed
    return smoothed

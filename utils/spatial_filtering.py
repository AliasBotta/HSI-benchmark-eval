import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def apply_knn_filter(prob_map, cube, cfg, mask=None):
    """
    Apply KNN smoothing to model probability maps guided by PCA(1) image.
    Args:
        prob_map: (N_labeled, num_classes)
        cube: (bands, H, W)
        cfg: config with spatial_spectral.knn_filter params
        mask: (H, W) boolean array for labeled pixels (optional)
    Returns:
        smoothed_prob_map: same shape as prob_map
    """
    print("[Spatial Filtering] Applying KNN filter guided by PCA(1)...")

    if cube is None:
        print("[Spatial Filtering] ⚠ No cube provided — applying 1D fallback smoothing.")
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(prob_map, size=3, axis=0)
        return smoothed

    # --- PCA(1) projection ---
    pca = PCA(n_components=1)
    bands, H, W = cube.shape
    cube_flat = cube.reshape(bands, -1).T
    pca_img = pca.fit_transform(cube_flat).reshape(H, W)

    # --- Coordinates + guided features ---
    lam = getattr(cfg.spatial_spectral.knn_filter, "lambda", 1)
    coords = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), -1).reshape(-1, 2)
    features = np.concatenate([pca_img.flatten()[:, None], lam * coords], axis=1)

    # --- Mask: only use labeled pixels ---
    if mask is not None:
        mask_flat = mask.flatten()
        features = features[mask_flat]
    elif prob_map.shape[0] < features.shape[0]:
        # Auto-infer approximate mask size from prob_map length
        features = features[:prob_map.shape[0]]

    # --- Fit KNN + smooth probabilities ---
    K = cfg.spatial_spectral.knn_filter.K
    nn = NearestNeighbors(n_neighbors=K, metric='euclidean').fit(features)
    distances, indices = nn.kneighbors(features)

    smoothed = np.zeros_like(prob_map)
    for i in range(prob_map.shape[1]):
        smoothed[:, i] = np.mean(prob_map[indices, i], axis=1)

    return smoothed

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def apply_knn_filter(prob_map, cube, cfg):
    """
    Apply KNN smoothing to model probability maps guided by PCA(1) image.
    Approximates MATLAB knnFilter_window().
    Args:
        prob_map: (N_pixels, num_classes)
        cube: (bands, H, W)
        cfg: config with spatial_spectral.knn_filter params
    Returns:
        smoothed_prob_map: same shape as prob_map
    """
    print("[Spatial Filtering] Applying KNN filter guided by PCA(1)...")

    # --- PCA(1) projection ---
    pca = PCA(n_components=1)
    bands, H, W = cube.shape
    cube_flat = cube.reshape(bands, -1).T
    pca_img = pca.fit_transform(cube_flat).reshape(H, W)

    # --- Build guided coordinates ---
    lam = cfg.spatial_spectral.knn_filter.lambda
    coords = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), -1).reshape(-1, 2)
    features = np.concatenate([pca_img.flatten()[:, None], lam * coords], axis=1)

    # --- Fit KNN and smooth probabilities ---
    K = cfg.spatial_spectral.knn_filter.K
    nn = NearestNeighbors(n_neighbors=K, metric='euclidean').fit(features)
    distances, indices = nn.kneighbors(features)

    smoothed = np.zeros_like(prob_map)
    for i in range(prob_map.shape[1]):  # class dimension
        smoothed[:, i] = np.mean(prob_map[indices, i], axis=1)

    return smoothed

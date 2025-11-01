import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import uniform_filter1d

def apply_knn_filter(prob_map, cube, cfg):
    """
    Apply KNN smoothing to model probability maps guided by PCA(1) image.
    Args:
        prob_map: (N_labeled, num_classes)
        cube: (bands, H, W)
        cfg: config with spatial_spectral.knn_filter params
    Returns:
        smoothed_prob_map: same shape as prob_map
    """
    print("[Spatial Filtering] Applying KNN filter guided by PCA(1)...")

    # ---- fallback if no cube ----
    if cube is None:
        print("[Spatial Filtering] ⚠ No cube provided — applying 1D fallback smoothing.")
        return uniform_filter1d(prob_map, size=3, axis=0)

    bands, H, W = cube.shape

    # --- create PCA(1) map ---
    pca = PCA(n_components=1)
    cube_flat = cube.reshape(bands, -1).T
    pca_img = pca.fit_transform(cube_flat).reshape(H, W)

    # --- flatten guided features ---
    lam = getattr(cfg.spatial_spectral.knn_filter, "lambda", 1)
    coords = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), -1).reshape(-1, 2)
    features = np.concatenate([pca_img.flatten()[:, None], lam * coords], axis=1)

    # --- if cube has more pixels than prob_map, auto-match shapes ---
    n_probs = prob_map.shape[0]
    n_feats = features.shape[0]

    if n_probs != n_feats:
        print(f"[Spatial Filtering] Adjusting features to match prob_map size ({n_feats} → {n_probs})")
        if n_probs < n_feats:
            # keep only the first n_probs pixels (approximate match)
            features = features[:n_probs]
        else:
            # pad features if needed (rare)
            pad = np.repeat(features[-1][None, :], n_probs - n_feats, axis=0)
            features = np.concatenate([features, pad], axis=0)

    # --- fit KNN + smooth ---
    K = cfg.spatial_spectral.knn_filter.K
    nn = NearestNeighbors(n_neighbors=K, metric='euclidean').fit(features)
    distances, indices = nn.kneighbors(features)

    smoothed = np.zeros_like(prob_map)
    for i in range(prob_map.shape[1]):
        smoothed[:, i] = np.mean(prob_map[indices, i], axis=1)

    return smoothed

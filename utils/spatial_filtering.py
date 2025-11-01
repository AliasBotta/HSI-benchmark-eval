import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import uniform_filter1d

def apply_knn_filter(prob_map, cube, cfg):
    """
    Apply KNN smoothing to model probability maps guided by PCA(1) image.
    faithful to the HSI-benchmark implementation.
    """
    print("[Spatial Filtering] Applying KNN filter guided by PCA(1)...")

    if cube is None:
        print("[Spatial Filtering] ⚠ No cube provided — applying 1D fallback smoothing.")
        return uniform_filter1d(prob_map, size=3, axis=0)

    bands, H, W = cube.shape
    pca = PCA(n_components=1)
    pca_img = pca.fit_transform(cube.reshape(bands, -1).T).reshape(H, W)
    lam = getattr(cfg.spatial_spectral.knn_filter, "lambda", 1.0)
    K = getattr(cfg.spatial_spectral.knn_filter, "K", 25)

    # --- coordinate grid ---
    yy, xx = np.mgrid[0:H, 0:W]
    features = np.stack([pca_img.ravel(), lam * yy.ravel(), lam * xx.ravel()], axis=1)

    # --- compute neighbors ---
    nbrs = NearestNeighbors(n_neighbors=K, metric='euclidean').fit(features)
    _, indices = nbrs.kneighbors(features)

    # --- smooth each class probability ---
    smoothed = np.zeros_like(prob_map)
    n_classes = prob_map.shape[1]

    # ensure prob_map covers all pixels (pad if necessary)
    n_pixels = H * W
    if len(prob_map) < n_pixels:
        pad = np.repeat(prob_map[-1][None, :], n_pixels - len(prob_map), axis=0)
        prob_map = np.concatenate([prob_map, pad])

    for c in range(n_classes):
        smoothed[:, c] = np.mean(prob_map[indices, c], axis=1)

    smoothed = smoothed[:len(prob_map)]  # back to labeled pixel count if needed
    return smoothed

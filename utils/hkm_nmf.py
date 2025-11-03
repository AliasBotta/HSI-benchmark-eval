# utils/hkm_nmf.py
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds

def _binary_split_indices(X_cols: np.ndarray, random_state: int = 0):
    """
    Binary split of a set of columns using rank-2 projection + k-means(2).
    X_cols: (bands, n_points) subset of the cube as feature matrix.
    Returns two index masks (relative to the local subset).
    """
    # Rank-2 projection (SVD as fast proxy for rank-2 NMF split)
    # NOTE: we use |U| to mimic non-negativity's effect on directionality.
    U, S, Vt = svds(X_cols, k=min(2, min(X_cols.shape)-1))
    Z = (U @ np.diag(S))  # (bands, 2)
    # Project points
    P = X_cols.T @ Z      # (n_points, 2)
    # KMeans with 2 clusters
    km = KMeans(n_clusters=2, n_init=5, random_state=random_state)
    labels = km.fit_predict(P)
    mask0 = (labels == 0)
    mask1 = ~mask0
    # Ensure non-empty splits
    if mask0.sum() == 0 or mask1.sum() == 0:
        # Fallback: random balanced split
        idx = np.arange(P.shape[0])
        np.random.default_rng(random_state).shuffle(idx)
        mid = len(idx)//2
        mask0 = np.zeros(len(idx), bool); mask0[idx[:mid]] = True
        mask1 = ~mask0
    return mask0, mask1

def h2nmf_segmentation(cube: np.ndarray, target_clusters: int = 24, random_state: int = 0):
    """
    Hierarchical rank-2 style segmentation on spectral cube.
    cube: (bands, H, W)
    Returns cluster_map: (H, W) with labels 0..(target_clusters-1)
    """
    bands, H, W = cube.shape
    X = cube.reshape(bands, -1)  # (bands, N)
    N = X.shape[1]
    clusters = [np.arange(N)]
    rng = np.random.default_rng(random_state)

    while len(clusters) < target_clusters:
        # pick the largest cluster to split
        sizes = [len(c) for c in clusters]
        i_big = int(np.argmax(sizes))
        idx = clusters.pop(i_big)
        if len(idx) < 2:
            # cannot split further; put it back and try another
            clusters.append(idx)
            # if all clusters are size<2, break
            if all(len(c) < 2 for c in clusters): break
            continue
        mask0, mask1 = _binary_split_indices(X[:, idx], random_state=rng.integers(0, 1e9))
        clusters.append(idx[mask0])
        clusters.append(idx[mask1])

    # Build label map
    labels = np.full(N, -1, int)
    for lab, idx in enumerate(clusters):
        labels[idx] = lab
    return labels.reshape(H, W)

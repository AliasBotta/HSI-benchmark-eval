"""
postprocessing.py
-----------------
Implements the Hierarchical segmentation + Majority Voting fusion step
from the HSI-benchmark paper.

This version uses a 1:1 port of the paper's H2NMF/SVD clustering.
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds 
from sklearn.utils.extmath import svd_flip


def _fast_separable_nmf(M, r):
    """
    Python port of the FastSepNMF algorithm.
    Finds 'r' endmembers from the input matrix M.

    Parameters
    ----------
    M : np.ndarray
        Input matrix, shape (features, samples)
    r : int
        Number of endmembers (clusters) to find.

    Returns
    -------
    J : np.ndarray
        Array of indices (shape r) corresponding to the endmember columns in M.
    U : np.ndarray
        Orthogonal basis (not strictly needed for splitclust).
    """
    m, n = M.shape
    J = np.zeros(r, dtype=int)
    U = np.zeros((m, r))

    normM = np.sum(M**2, axis=0)
    normM1 = normM.copy()
    nM = np.max(normM)

    for i in range(r):
        current_max_norm = np.max(normM)

        b_indices = np.where(
            np.abs(current_max_norm - normM) / (current_max_norm + 1e-9) <= 1e-6
        )[0]

        if len(b_indices) > 1:
            b = b_indices[np.argmax(normM1[b_indices])]
        else:
            b = b_indices[0]

        J[i] = b 
        U[:, i] = M[:, b] 

        for j in range(i):
            U[:, i] -= U[:, j] * (U[:, j].T @ U[:, i])

        norm_Ui = np.linalg.norm(U[:, i])
        if norm_Ui > 1e-9:
            U[:, i] /= norm_Ui
        else:
            U[:, i] = 0.0 

        v = U[:, i]

        normM -= (v.T @ M)**2
        normM[normM < 0] = 0.0 

        if np.max(normM) / nM <= 1e-9:
            break

    return J, U


def _splitclust(M):
    """
    Port of the researchers' splitclust.m function.
    Uses SVD+NMF-initialized K-Means, not Spectral Clustering.

    Input M is (N_pixels, N_bands).
    """
    if M.shape[0] <= 2:
        return None, None

    try:

        u_m, s_m, vh_m = svds(M.T, k=2)

        u_m, vh_m = svd_flip(u_m, vh_m)

        M_for_nmf = np.diag(s_m) @ vh_m 

        Kf, _ = _fast_separable_nmf(M_for_nmf, 2)

        v_m = vh_m.T 
        v_k_rows = v_m[Kf, :] 

        U0 = u_m @ np.diag(s_m) @ v_k_rows.T


        init_centroids = U0.T 

        km = KMeans(
            n_clusters=2,
            init=init_centroids,
            n_init=1, 
            random_state=42,
            max_iter=300
        )
        labels = km.fit_predict(M) 

        K1 = np.where(labels == 0)[0]
        K2 = np.where(labels == 1)[0]

        if K1.size == 0 or K2.size == 0:
            raise ValueError("Empty cluster from SVD/NMF init")

        return K1, K2

    except Exception as e:
        try:
            print(f"[Postprocessing] WARNING: SVD/NMF init failed ({e}). Falling back to k-means++.")
            km = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=42).fit(M)
            labels = km.labels_
            K1 = np.where(labels == 0)[0]
            K2 = np.where(labels == 1)[0]
            if K1.size == 0 or K2.size == 0:
                return None, None
            return K1, K2
        except Exception:
            return None, None 


def _h2nmf_segmentation(cube_h_w_b, r_clusters):
    """
    Minimalist port of hierclust2nmfMulti.m
    Input cube shape is (H, W, bands).

    (This function is unchanged, but it now calls the *new* _splitclust)
    """
    H, W, bands = cube_h_w_b.shape
    n_pixels = H * W

    M_pixels_bands = cube_h_w_b.reshape(n_pixels, bands)

    sol_K = {0: np.arange(n_pixels)}
    sol_leafnodes = [0]
    sol_maxnode = 0
    count = 1

    print(f"[Postprocessing] Running Hierarchical (H2NMF/SVD) segmentation...")

    while count < r_clusters:
        if not sol_leafnodes:
            break

        leaf_sizes = [sol_K[key].size for key in sol_leafnodes]
        split_node_idx = np.argmax(leaf_sizes)
        split_node_key = sol_leafnodes.pop(split_node_idx)

        pixel_indices_to_split = sol_K[split_node_key]
        M_cluster = M_pixels_bands[pixel_indices_to_split]

        K1_local, K2_local = _splitclust(M_cluster)

        if K1_local is None or K1_local.size == 0 or K2_local is None or K2_local.size == 0:
            if len(sol_leafnodes) == 0: 
                break
            continue

        key1 = sol_maxnode + 1
        key2 = sol_maxnode + 2

        sol_K[key1] = pixel_indices_to_split[K1_local]
        sol_K[key2] = pixel_indices_to_split[K2_local]

        sol_leafnodes.extend([key1, key2])
        del sol_K[split_node_key] 

        sol_maxnode += 2
        count += 1

    labels = np.zeros(n_pixels, dtype=int)
    final_clusters = list(sol_K.keys())
    for i, key in enumerate(final_clusters):
        labels[sol_K[key]] = i

    final_n_clusters = len(final_clusters)
    print(f"[Postprocessing] ✅ HKM segmentation complete ({final_n_clusters} clusters).")
    return labels.reshape(H, W), final_n_clusters 


def majority_voting(knn_class_map, pc1=None, cube=None, n_clusters=24):
    """
    Majority voting post-processing using the H2NMF/SVD hierarchical clustering.

    (This function is unchanged)
    """
    H, W = knn_class_map.shape

    if cube is None:
         raise ValueError("H2NMF clustering requires the full 'cube' (not just PC1).")

    cube_h_w_b = np.transpose(cube, (1, 2, 0))
    labels, actual_n_clusters = _h2nmf_segmentation(cube_h_w_b, n_clusters) 

    class_mv = np.zeros_like(knn_class_map)
    for i in range(actual_n_clusters):
        mask = (labels == i)
        if not np.any(mask):
            continue
        vals, counts = np.unique(knn_class_map[mask], return_counts=True)
        maj = vals[np.argmax(counts)]
        class_mv[mask] = maj

    print("[Postprocessing] ✅ Majority voting fusion complete.")
    return class_mv

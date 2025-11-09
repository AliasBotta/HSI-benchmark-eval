# /home/ale/repos/HSI-benchmark-eval/utils/postprocessing.py
"""
postprocessing.py
-----------------
Implements the Hierarchical segmentation + Majority Voting fusion step
from the HSI-benchmark paper.

This version uses a 1:1 port of the paper's H2NMF/SVD clustering.
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds # <-- CHANGED: Using svds for (bands, pixels) matrix
from sklearn.utils.extmath import svd_flip

# ===================================================================
# NEW HELPER FUNCTION (Port of MATLAB FastSepNMF.m)
# ===================================================================

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

    # L2-norm squared of each column (sample)
    normM = np.sum(M**2, axis=0)
    # Save initial norms for tie-breaking
    normM1 = normM.copy()
    # Max norm in the set
    nM = np.max(normM)

    for i in range(r):
        # Find column with max L2-norm
        current_max_norm = np.max(normM)

        # Find ties (indices)
        b_indices = np.where(
            np.abs(current_max_norm - normM) / (current_max_norm + 1e-9) <= 1e-6
        )[0]

        if len(b_indices) > 1:
            # Break tie using original max norm
            b = b_indices[np.argmax(normM1[b_indices])]
        else:
            b = b_indices[0]

        J[i] = b # Store index of the endmember
        U[:, i] = M[:, b] # Get the column vector

        # Project onto orthogonal complement of U_1...U_{i-1} (Gram-Schmidt)
        for j in range(i):
            U[:, i] -= U[:, j] * (U[:, j].T @ U[:, i])

        # Normalize the new orthogonal vector
        norm_Ui = np.linalg.norm(U[:, i])
        if norm_Ui > 1e-9:
            U[:, i] /= norm_Ui
        else:
            U[:, i] = 0.0 # Vector is already in the span

        # v is the normalized orthogonal component
        v = U[:, i]

        # Update norms: ||r_k||^2 = ||r_{k-1}||^2 - (v^T * m_k)^2
        # This projects all points onto the new component v and subtracts
        # that energy from their L2-norm.
        normM -= (v.T @ M)**2
        normM[normM < 0] = 0.0 # Fix numerical precision issues

        if np.max(normM) / nM <= 1e-9:
            # Stop if residual is tiny
            break

    return J, U

# ===================================================================
# REPLACED FUNCTION (Port of MATLAB splitclust.m)
# ===================================================================

def _splitclust(M):
    """
    Port of the researchers' splitclust.m function.
    Uses SVD+NMF-initialized K-Means, not Spectral Clustering.

    Input M is (N_pixels, N_bands).
    """
    if M.shape[0] <= 2:
        # Not enough samples to split
        return None, None

    try:
        # 1. [u,s,v] = fastsvds(M,2)
        # The MATLAB code runs SVD on M=(bands, pixels).
        # Our M is (pixels, bands), so we run SVD on M.T
        # We want k=2 components.

        # M.T is (bands, pixels) -> (128, N)
        u_m, s_m, vh_m = svds(M.T, k=2)
        # u_m: (bands, 2)
        # s_m: (2,)
        # vh_m: (2, pixels)

        # Ensure deterministic output (flip signs)
        u_m, vh_m = svd_flip(u_m, vh_m)

        # 2. Kf = FastSepNMF(s*v',2,0)
        # `s*v'` in MATLAB is (2,2) @ (2, pixels) = (2, pixels)
        M_for_nmf = np.diag(s_m) @ vh_m # (2, pixels)

        # Kf holds the indices of the 2 endmember *pixels*
        Kf, _ = _fast_separable_nmf(M_for_nmf, 2)

        # 3. U0 = u*s*v(Kf,:)'
        # MATLAB `v` is (pixels, 2). `v(Kf,:)` selects 2 rows -> (2, 2)
        v_m = vh_m.T # (pixels, 2)
        v_k_rows = v_m[Kf, :] # (2, 2)

        # U0 = (bands, 2) @ (2, 2) @ (2, 2) = (bands, 2)
        # These are the 2 initial centroids (in 'bands' space)
        U0 = u_m @ np.diag(s_m) @ v_k_rows.T

        # 4. [IDX,U] = kmeans(M', 2, 'Start', U0');
        # We run K-Means on the *original data* M (pixels, bands)
        # using the centroids U0.T (2, bands) as init.

        init_centroids = U0.T # (2, bands)

        km = KMeans(
            n_clusters=2,
            init=init_centroids,
            n_init=1, # MATLAB uses n_init=1 when 'Start' is provided
            random_state=42,
            max_iter=300
        )
        labels = km.fit_predict(M) # M is (pixels, bands)

        K1 = np.where(labels == 0)[0]
        K2 = np.where(labels == 1)[0]

        if K1.size == 0 or K2.size == 0:
            raise ValueError("Empty cluster from SVD/NMF init")

        return K1, K2

    except Exception as e:
        # Fallback to standard K-Means on raw data if SVD/NMF fails
        # (This was also in your original code and is a good safety net)
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
            return None, None # Final failure

# ===================================================================
# UNCHANGED HIERARCHICAL LOOP
# ===================================================================

def _h2nmf_segmentation(cube_h_w_b, r_clusters):
    """
    Minimalist port of hierclust2nmfMulti.m
    Input cube shape is (H, W, bands).

    (This function is unchanged, but it now calls the *new* _splitclust)
    """
    H, W, bands = cube_h_w_b.shape
    n_pixels = H * W

    # Transpose to (N_pixels, N_bands)
    M_pixels_bands = cube_h_w_b.reshape(n_pixels, bands)

    # Store cluster data {key: pixel_indices}
    sol_K = {0: np.arange(n_pixels)}
    # Store keys of leaves
    sol_leafnodes = [0]
    sol_maxnode = 0
    count = 1

    print(f"[Postprocessing] Running Hierarchical (H2NMF/SVD) segmentation...")

    while count < r_clusters:
        if not sol_leafnodes:
            break

        # Find biggest leaf node to split
        leaf_sizes = [sol_K[key].size for key in sol_leafnodes]
        split_node_idx = np.argmax(leaf_sizes)
        split_node_key = sol_leafnodes.pop(split_node_idx)

        pixel_indices_to_split = sol_K[split_node_key]
        M_cluster = M_pixels_bands[pixel_indices_to_split]

        # --- THIS CALL NOW USES THE NEW SVD/NMF-init KMEANS ---
        K1_local, K2_local = _splitclust(M_cluster)
        # -----------------------------------------------------

        if K1_local is None or K1_local.size == 0 or K2_local is None or K2_local.size == 0:
            # Failed to split (e.g., cluster was too small or uniform)
            if len(sol_leafnodes) == 0: # Only node left and it failed
                break
            continue

        key1 = sol_maxnode + 1
        key2 = sol_maxnode + 2

        sol_K[key1] = pixel_indices_to_split[K1_local]
        sol_K[key2] = pixel_indices_to_split[K2_local]

        sol_leafnodes.extend([key1, key2])
        del sol_K[split_node_key] # Free memory

        sol_maxnode += 2
        count += 1

    # Create the final label map from the leaf nodes
    labels = np.zeros(n_pixels, dtype=int)
    # Re-label final clusters contiguously
    final_clusters = list(sol_K.keys())
    for i, key in enumerate(final_clusters):
        labels[sol_K[key]] = i

    # Ensure we have the correct number of clusters in the map
    final_n_clusters = len(final_clusters)
    print(f"[Postprocessing] ✅ HKM segmentation complete ({final_n_clusters} clusters).")
    return labels.reshape(H, W), final_n_clusters # Return map and actual cluster count

# ===================================================================
# UNCHANGED MAJORITY VOTE FUNCTION
# ===================================================================

def majority_voting(knn_class_map, pc1=None, cube=None, n_clusters=24):
    """
    Majority voting post-processing using the H2NMF/SVD hierarchical clustering.

    (This function is unchanged)
    """
    H, W = knn_class_map.shape

    if cube is None:
         raise ValueError("H2NMF clustering requires the full 'cube' (not just PC1).")

    # --- 1) Paper-aligned HKM-NMF/SVD clustering ---
    # _h2nmf_segmentation expects (H, W, bands)
    cube_h_w_b = np.transpose(cube, (1, 2, 0))
    labels, actual_n_clusters = _h2nmf_segmentation(cube_h_w_b, n_clusters) # (H, W)

    # --- 2) Majority vote within each cluster ---
    class_mv = np.zeros_like(knn_class_map)
    for i in range(actual_n_clusters):
        mask = (labels == i)
        if not np.any(mask):
            continue
        # Get all classes predicted by KNN in this segment
        vals, counts = np.unique(knn_class_map[mask], return_counts=True)
        # Find the one with the most votes
        maj = vals[np.argmax(counts)]
        # Assign that class to the whole segment
        class_mv[mask] = maj

    print("[Postprocessing] ✅ Majority voting fusion complete.")
    return class_mv

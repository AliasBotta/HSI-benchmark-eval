# /home/ale/repos/HSI-benchmark-eval/utils/postprocessing.py

"""
postprocessing.py
-----------------
Implements the Hierarchical segmentation + Majority Voting fusion step
from the HSI-benchmark paper.

This version uses a minimal port of the paper's H2NMF/SVD clustering.
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import svds
from sklearn.utils.extmath import svd_flip

# --- Start of HKM Helper Functions (Ported from MATLAB) ---

def _splitclust(M):
    """
    Minimalist port of splitclust(M) using SVD-initialized K-Means=2.
    This replaces the complex SVD+NMF+KMeans init from the paper
    with a robust SVD+KMeans init (a common spectral clustering step).
    
    Input M is (N_pixels, N_bands).
    """
    if M.shape[0] <= 2:
        # Not enough samples to split
        return None, None
        
    try:
        # Get the 2 largest principal components/vectors
        # M (pixels, bands) -> U(pixels, 2)
        # Using 'arpack' for sparse SVD, similar to MATLAB's svds
        U, s, Vh = svds(M, k=2, tol=1e-6, solver='arpack')
        # Ensure deterministic output
        U, Vh = svd_flip(U, Vh)
        
        # Cluster pixels based on their 2-component embedding (U)
        km = KMeans(n_clusters=2, init='k-means++', n_init=5, random_state=42, max_iter=300)
        labels = km.fit_predict(U)
        
        K1 = np.where(labels == 0)[0]
        K2 = np.where(labels == 1)[0]
        
        if K1.size == 0 or K2.size == 0:
            raise ValueError("Empty cluster from SVD init")

        return K1, K2

    except Exception:
        # Fallback to standard K-Means on raw data if SVD fails
        try:
            km = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=42).fit(M)
            labels = km.labels_
            K1 = np.where(labels == 0)[0]
            K2 = np.where(labels == 1)[0]
            if K1.size == 0 or K2.size == 0:
                return None, None
            return K1, K2
        except Exception:
            return None, None # Final failure

def _h2nmf_segmentation(cube_h_w_b, r_clusters):
    """
    Minimalist port of hierclust2nmfMulti.m
    Input cube shape is (H, W, bands).
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
        
        K1_local, K2_local = _splitclust(M_cluster)
        
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

# --- End of HKM Helper Functions ---


def majority_voting(knn_class_map, pc1=None, cube=None, n_clusters=24):
    """
    Majority voting post-processing using the H2NMF/SVD hierarchical clustering.
    REMOVED: use_h2nmf flag and K-Means 2.5D fallback.
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
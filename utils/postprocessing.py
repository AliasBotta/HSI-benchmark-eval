import numpy as np
from sklearn.cluster import KMeans

def majority_voting(prob_map, cube, cfg):
    """
    Perform majority voting based on HKM segmentation.
    Approximates MATLAB hierclust2nmfMulti + majorityVoting().
    """
    print("[Postprocessing] Applying Majority Voting (HKM)...")

    bands, H, W = cube.shape
    num_clusters = cfg.spatial_spectral.hkm.clusters
    cube_flat = cube.reshape(bands, -1).T

    # --- Segment the cube (HKM approx via KMeans) ---
    km = KMeans(n_clusters=num_clusters, n_init=3, random_state=42)
    cluster_ids = km.fit_predict(cube_flat)

    # --- Majority voting per cluster ---
    preds = np.argmax(prob_map, axis=1)
    final_preds = np.copy(preds)
    for cid in range(num_clusters):
        mask = cluster_ids == cid
        if np.any(mask):
            votes = preds[mask]
            if len(votes) > 0:
                majority_class = np.bincount(votes).argmax()
                final_preds[mask] = majority_class

    return final_preds

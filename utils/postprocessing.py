import numpy as np
from sklearn.cluster import KMeans

def majority_voting(prob_map, cube=None, cfg=None):
    """
    Apply Majority Voting (MV) post-processing combining:
      - Supervised probability maps (prob_map)
      - Unsupervised segmentation (HKM clustering on PCA(1) or spectra)
    """
    print("[Postprocessing] Applying Majority Voting (HKM)...")

    if cube is None:
        print("[Postprocessing] ⚠ No cube provided — fallback to argmax fusion.")
        return np.argmax(prob_map, axis=1)

    bands, H, W = cube.shape
    cube_flat = cube.reshape(bands, -1).T

    # --- Hierarchical K-Means segmentation (HKM) approximation ---
    n_clusters = getattr(cfg.spatial_spectral.hkm, "clusters", 24)
    print(f"[Postprocessing] Running HKM segmentation with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    seg_labels = kmeans.fit_predict(cube_flat)

    preds_supervised = np.argmax(prob_map, axis=1)
    preds_mv = preds_supervised.copy()

    # --- Apply majority voting per cluster ---
    for seg in np.unique(seg_labels):
        idx = np.where(seg_labels == seg)[0]
        if len(idx) == 0:
            continue
        counts = np.bincount(preds_supervised[idx], minlength=prob_map.shape[1])
        preds_mv[idx] = np.argmax(counts)

    print("[Postprocessing] ✅ Majority voting fusion complete.")
    return preds_mv

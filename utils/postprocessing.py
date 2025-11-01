import numpy as np
from sklearn.cluster import KMeans

def majority_voting(prob_map, cube=None, cfg=None):
    """
    Apply Hierarchical K-Means (HKM) majority voting to smooth predictions.
    Args:
        prob_map: (N_pixels, num_classes)
        cube: (bands, H, W)
        cfg: configuration object
    Returns:
        preds_mv: smoothed predictions (N_pixels,)
    """
    print("[Postprocessing] Applying Majority Voting (HKM)...")

    if cube is None:
        print("[Postprocessing] ⚠ No cube provided — fallback to argmax fusion.")
        return np.argmax(prob_map, axis=1)

    bands, H, W = cube.shape
    n_pixels = H * W
    n_classes = prob_map.shape[1]

    # --- flatten cube and reduce with PCA(1) or mean intensity ---
    mean_img = cube.mean(axis=0).reshape(-1, 1)

    # --- KMeans segmentation ---
    n_clusters = getattr(cfg.postprocessing.hkm, "num_clusters", 24)
    print(f"[Postprocessing] Running HKM segmentation with {n_clusters} clusters...")

    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    seg_labels = km.fit_predict(mean_img)  # shape (H*W,)

    # --- Align lengths if mismatch (common case: labeled pixels only in prob_map) ---
    if len(prob_map) < n_pixels:
        print(f"[Postprocessing] Adjusting HKM labels to match prob_map size ({n_pixels} → {len(prob_map)})")
        seg_labels = seg_labels[:len(prob_map)]
    elif len(prob_map) > n_pixels:
        print(f"[Postprocessing] Padding HKM labels ({n_pixels} → {len(prob_map)})")
        pad = np.repeat(seg_labels[-1], len(prob_map) - n_pixels)
        seg_labels = np.concatenate([seg_labels, pad])

    # --- Majority voting within each cluster ---
    preds_supervised = np.argmax(prob_map, axis=1)
    preds_mv = preds_supervised.copy()

    for k in np.unique(seg_labels):
        idx = np.where(seg_labels == k)[0]
        if len(idx) == 0:
            continue
        labels, counts = np.unique(preds_supervised[idx], return_counts=True)
        preds_mv[idx] = labels[np.argmax(counts)]

    print("[Postprocessing] ✅ Majority voting fusion complete.")
    return preds_mv

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def majority_voting(prob_map, cube=None, cfg=None):
    """
    Apply Hierarchical K-Means (HKM) majority voting to smooth predictions.
    Faithful to the HSI-benchmark reference implementation.
    Args:
        prob_map: (N_labeled, num_classes)
        cube: (bands, H, W)
        cfg: config object
    Returns:
        preds_mv: (N_labeled,) array of smoothed predictions
    """
    print("[Postprocessing] Applying Majority Voting (HKM)...")

    if cube is None:
        print("[Postprocessing] ⚠ No cube provided — fallback to argmax fusion.")
        return np.argmax(prob_map, axis=1)

    bands, H, W = cube.shape
    n_pixels = H * W
    n_classes = prob_map.shape[1]

    # --- PCA(1) image as segmentation guide ---
    pca = PCA(n_components=1)
    pca_img = pca.fit_transform(cube.reshape(bands, -1).T).reshape(H, W)

    # --- KMeans clustering on PCA(1) image ---
    n_clusters = getattr(cfg.postprocessing.hkm, "num_clusters", 20)
    print(f"[Postprocessing] Running HKM segmentation with {n_clusters} clusters...")

    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    seg_labels = km.fit_predict(pca_img.reshape(-1, 1))  # (H*W,)

    # --- handle mismatched sizes ---
    n_probs = prob_map.shape[0]
    if n_probs < n_pixels:
        print(f"[Postprocessing] Padding HKM labels ({n_pixels} → {n_probs})")
        seg_labels = seg_labels[:n_probs]
    elif n_probs > n_pixels:
        print(f"[Postprocessing] Adjusting HKM labels to match prob_map size ({n_pixels} → {n_probs})")
        pad = np.repeat(seg_labels[-1], n_probs - n_pixels)
        seg_labels = np.concatenate([seg_labels, pad])

    # --- supervised predictions (argmax per pixel) ---
    preds = np.argmax(prob_map, axis=1)
    preds_mv = preds.copy()

    # --- majority voting within clusters ---
    for k in np.unique(seg_labels):
        idx = np.where(seg_labels == k)[0]
        if len(idx) == 0:
            continue
        labels, counts = np.unique(preds[idx], return_counts=True)
        preds_mv[idx] = labels[np.argmax(counts)]

    print("[Postprocessing] ✅ Majority voting fusion complete.")
    return preds_mv

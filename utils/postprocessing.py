import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def majority_voting(prob_map, cube=None, cfg=None):
    """
    Apply Hierarchical K-Means (HKM) majority voting identical to the benchmark.
    """
    print("[Postprocessing] Applying Majority Voting (HKM)...")

    if cube is None:
        print("[Postprocessing] ⚠ No cube provided — fallback to argmax fusion.")
        return np.argmax(prob_map, axis=1)

    bands, H, W = cube.shape
    n_pixels = H * W
    n_classes = prob_map.shape[1]
    n_clusters = getattr(cfg.postprocessing.hkm, "num_clusters", 20)

    # --- reduce cube to PCA(1) image ---
    pca = PCA(n_components=1)
    pca_img = pca.fit_transform(cube.reshape(bands, -1).T)
    guide_img = pca_img.reshape(H, W)

    print(f"[Postprocessing] Running HKM segmentation with {n_clusters} clusters...")

    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    seg_labels = km.fit_predict(guide_img.reshape(-1, 1))

    # --- adjust lengths if needed ---
    if len(prob_map) < n_pixels:
        seg_labels = seg_labels[:len(prob_map)]
    elif len(prob_map) > n_pixels:
        pad = np.repeat(seg_labels[-1], len(prob_map) - n_pixels)
        seg_labels = np.concatenate([seg_labels, pad])

    preds = np.argmax(prob_map, axis=1)
    preds_mv = preds.copy()

    # --- majority vote within clusters ---
    for k in np.unique(seg_labels):
        idx = np.where(seg_labels == k)[0]
        if len(idx) == 0:
            continue
        labels, counts = np.unique(preds[idx], return_counts=True)
        preds_mv[idx] = labels[np.argmax(counts)]

    print("[Postprocessing] ✅ Majority voting fusion complete.")
    return preds_mv

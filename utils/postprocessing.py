import numpy as np
from sklearn.cluster import KMeans

def majority_voting(prob_map, cube=None, cfg=None):
    """
    Apply Majority Voting post-processing (as in the benchmark paper).
    Combines supervised prob. maps with unsupervised segmentation (HKM).
    If cube=None, performs a safe fallback averaging (1D mode).
    """
    print("[Postprocessing] Applying Majority Voting (HKM)...")

    mv_cfg = getattr(cfg.spatial_spectral, "mv", None)
    hkm_cfg = getattr(cfg.spatial_spectral, "hkm", None)

    # === CASE 1: full MV fusion (requires cube) ===
    if cube is not None:
        bands, H, W = cube.shape
        cube_flat = cube.reshape(bands, -1).T

        # --- Unsupervised segmentation (HKM) ---
        n_clusters = getattr(hkm_cfg, "clusters", 24)
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        seg_labels = km.fit_predict(cube_flat)

        # --- Majority voting per segment ---
        preds = np.argmax(prob_map, axis=1)
        preds_mv = preds.copy()
        for seg in np.unique(seg_labels):
            idx = np.where(seg_labels == seg)[0]
            if len(idx) == 0:
                continue
            counts = np.bincount(preds[idx], minlength=prob_map.shape[1])
            preds_mv[idx] = np.argmax(counts)

        print(f"[Postprocessing] Done (HKM clusters={n_clusters})")
        return preds_mv

    # === CASE 2: fallback (no cube) ===
    print("[Postprocessing] ⚠ No cube provided — using direct argmax fusion.")
    preds_mv = np.argmax(prob_map, axis=1)
    return preds_mv

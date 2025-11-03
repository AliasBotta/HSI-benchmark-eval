# scripts/train.py
"""
Complete pipeline (paper-aligned):
RAW → Preprocess (already done) → PCA(1) → Supervised(ClassMap, probAllImage)
→ KNN filter (knn.classMap, knn.mapProb) → HKM + Majority Voting (mv.classMap)
Evaluations: (1) Spectral, (2) Spatial/KNN, (3) MV.

Modular by model: --model {dnn, svm-l, svm-rbf, knn-e, rf, ebeae, nebeae}
Each model is implemented in models/<model>.py and must return:
  - class_map: (H, W)  with labels 0..3 (TT, NT, BV, BG)
  - prob_all : (H, W, 4) class probabilities (order: [TT, NT, BV, BG])
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from utils.helpers import setup_logger, set_seed, ensure_dir
from utils.data_loading import list_all_instances
from utils.metrics import compute_all_metrics
from utils.spatial_filtering import apply_knn_filter
from utils.postprocessing import majority_voting
from models import get_runner


# ============================================================
# GLOBAL PIPELINE PARAMETERS (replacing YAML)
# ============================================================

SEED = 42
DEVICE = "cuda"
OUTPUT_DIR = Path("outputs/default")
PROCESSED_DIR = Path("data/processed")

# Partition
N_FOLDS = 5
SPLIT = (0.6, 0.2, 0.2)  # train, val, test

# KNN filter params
KNN_ENABLED = True
KNN_K = 40
KNN_WINDOW = 14
KNN_LAMBDA = 1
KNN_DISTANCE = "euclidean"

# HKM + Majority Voting
HKM_CLUSTERS = 24

# Classes
NUM_CLASSES = 4
BG_INDEX = 3


# ============================================================
# Utility functions
# ============================================================

def _list_processed_instances(processed_dir: Path):
    processed_dir = Path(processed_dir)
    return sorted([
        d for d in processed_dir.iterdir()
        if d.is_dir() and (d / "preprocessed_cube.npy").exists() and (d / "gtMap.npy").exists()
    ])


def _pid_from_name(dirname: str) -> str:
    """'004-02' -> '004'"""
    return dirname.split("-")[0]


def _make_patient_folds(instances, n_folds=N_FOLDS, split=SPLIT, random_seed=SEED):
    """Create patient-level folds (same rotation logic as paper)."""
    pids = sorted(list({_pid_from_name(d.name) for d in instances}))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(pids)

    n_total = len(pids)
    n_train = int(split[0] * n_total)
    n_val = int(split[1] * n_total)
    n_test = max(1, n_total - n_train - n_val)

    folds = []
    for k in range(n_folds):
        start = (k * n_test) % n_total
        end = start + n_test
        test_p = pids[start:end]
        remain = [p for p in pids if p not in test_p]
        n_val_local = int(split[1] / (split[0] + split[1]) * len(remain))
        val_p = remain[:n_val_local]
        train_p = remain[n_val_local:]
        folds.append((train_p, val_p, test_p))
    return folds


def _gather_training_pixels(processed_dir: Path, instances, train_pids):
    """Flatten all labeled pixels from training patients."""
    X_list, y_list = [], []
    for d in instances:
        pid = _pid_from_name(d.name)
        if pid not in train_pids:
            continue
        cube = np.load(d / "preprocessed_cube.npy")      # (bands,H,W)
        gt = np.load(d / "gtMap.npy")                    # (H,W) with labels 1..4 (MATLAB-style)
        bands, H, W = cube.shape
        flat = cube.reshape(bands, -1).T                 # (H*W, bands)
        gt_f = gt.reshape(-1)
        m = gt_f > 0
        if m.any():
            X_list.append(flat[m])
            # Map to zero-based fixed order {0:NT, 1:TT, 2:BV, 3:BG}
            y_list.append(gt_f[m] - 1)
    if not X_list:
        return np.empty((0,)), np.empty((0,), int)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)



def _evaluate_map(gt_map, pred_map, bg_index=BG_INDEX):
    """Compute metrics on labeled pixels (gt>0) excluding BG from evaluation.
    Predictions that are BG where GT is non-BG are counted as errors."""
    gt = gt_map.flatten()
    pr = pred_map.flatten()

    mask_labeled = gt > 0
    gt = gt[mask_labeled] - 1
    pr = pr[mask_labeled]

    # Drop BG from GT, but keep predictions (BG pred = error)
    keep = gt != bg_index
    gt = gt[keep]              # values in {0,1,2}
    pr = pr[keep]              # values in {0,1,2,3}

    # Cap predictions to valid evaluation labels (avoid index issues)
    pr_eval = np.where(pr > 2, -1, pr)
    return compute_all_metrics(gt, pr_eval, num_classes=3, labels=[0, 1, 2])


# ============================================================
# Main Pipeline
# ============================================================

def main(model_name: str):
    set_seed(SEED)
    logger = setup_logger("outputs/logs", name=f"train_{model_name}")
    ensure_dir(OUTPUT_DIR)

    processed_dir = PROCESSED_DIR
    instances = _list_processed_instances(processed_dir)
    if not instances:
        raise FileNotFoundError(
            "No preprocessed instances found in data/processed. Run preprocess.py first."
        )

    folds = _make_patient_folds(instances)
    logger.info(f"Found {len(instances)} instances "
                f"({len(set(_pid_from_name(d.name) for d in instances))} patients).")
    logger.info(f"Starting {N_FOLDS}-fold CV by patient. Model: {model_name}")

    # Get the model runner
    runner = get_runner(model_name)

    all_rows = []
    for k, (train_p, val_p, test_p) in enumerate(folds, start=1):
        logger.info(f"Fold {k}: train={len(train_p)}, val={len(val_p)}, test={len(test_p)}")

        # 1) Training pixels
        Xtr_raw, ytr_raw = _gather_training_pixels(processed_dir, instances, set(train_p))
        logger.info(f"[Fold {k}] Training pixels (raw): {len(ytr_raw)}")

        if Xtr_raw.size == 0:
            logger.warning(f"[Fold {k}] Empty training set. Skipping this fold.")
            continue

        # --- Class distribution (raw) ---
        cls, cnt = np.unique(ytr_raw, return_counts=True)
        logger.info(f"[Fold {k}] Raw class distribution (0:NT,1:TT,2:BV,3:BG): {dict(zip(cls, cnt))}")

        # 2) Paper-aligned training-set reduction (KMeans+SAM)
        Xtr, ytr = reduce_training_data(
            Xtr_raw, ytr_raw,
            enabled=REDUCTION_ENABLED,
            clusters_per_class=REDUCTION_CLUSTERS,
            target_per_class=REDUCTION_TARGET_PER_CLASS,
            random_seed=SEED
        )
        logger.info(f"[Fold {k}] Reduced training pixels: {len(ytr)}")

        # --- distribution after reduction ---
        cls_r, cnt_r = np.unique(ytr, return_counts=True)
        logger.info(f"[Fold {k}] Reduced class distribution: {dict(zip(cls_r, cnt_r))}")

        # 3) Train model
        runner.fit(Xtr, ytr)
        logger.info(f"[Fold {k}] Model training complete")

        # 4) Inference + post-processing
        fold_rows = []
        for d in instances:
            pid = _pid_from_name(d.name)
            if pid not in test_p:
                continue

            cube = np.load(d / "preprocessed_cube.npy")  # (bands,H,W)
            gt = np.load(d / "gtMap.npy")                # (H,W)
            bands, H, W = cube.shape

            # --- PCA(1) once per image ---
            pca = PCA(n_components=1)
            pca_img = pca.fit_transform(cube.reshape(bands, -1).T).reshape(H, W)

            # (1) Supervised classification
            class_map, prob_all = runner.predict_full(cube)  # (H,W), (H,W,4)
            # prob_all must follow [NT, TT, BV, BG]; ensure shape consistency
            if prob_all.shape[-1] != 4:
                raise ValueError("Model must output probabilities with 4 channels [NT, TT, BV, BG].")

            # (2) KNN filtering guided by PCA(1)
            prob_flat = prob_all.reshape(-1, prob_all.shape[-1])
            prob_knn_flat = apply_knn_filter(
                prob_flat, pc1=pca_img, cube=cube,
                K=KNN_K, window_size=KNN_WINDOW,
                lambda_=KNN_LAMBDA, distance=KNN_DISTANCE
            )
            class_knn = np.argmax(prob_knn_flat, axis=1).reshape(H, W)

            # (3) HKM + Majority Voting (H2NMF-based)
            class_mv = majority_voting(
                class_knn, pc1=pca_img, cube=cube,
                n_clusters=HKM_CLUSTERS, use_h2nmf=True
            )

            # ---- Metrics (excluding BG) ----
            m_spec = _evaluate_map(gt, class_map)
            m_knn = _evaluate_map(gt, class_knn)
            m_mv = _evaluate_map(gt, class_mv)

            row = {
                "fold": k, "patient": pid,
                "spectral_f1": m_spec["f1_macro"], "spectral_oa": m_spec["oa"],
                "spatial_f1": m_knn["f1_macro"], "spatial_oa": m_knn["oa"],
                "mv_f1": m_mv["f1_macro"], "mv_oa": m_mv["oa"],
            }
            fold_rows.append(row)
            logger.info(f"[Fold {k} | PID {pid}] F1: Spec={row['spectral_f1']:.3f} "
                        f"→ KNN={row['spatial_f1']:.3f} → MV={row['mv_f1']:.3f}")

        all_rows.extend(fold_rows)

    # Results summary
    df = pd.DataFrame(all_rows)
    out_dir = OUTPUT_DIR
    df.to_csv(out_dir / "metrics_per_patient.csv", index=False)

    if not df.empty:
        summary = df.groupby("fold")[[
            "spectral_f1", "spatial_f1", "mv_f1",
            "spectral_oa", "spatial_oa", "mv_oa"
        ]].mean()
        summary.loc["mean"] = summary.mean()
        summary.loc["std"] = summary.std()
        summary.to_csv(out_dir / "metrics_summary.csv")
        logger.info(f"📊 Saved → {out_dir/'metrics_summary.csv'}")

    logger.info("✅ Pipeline completed.")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Model name (dnn, svm-l, svm-rbf, knn-e, rf, ebeae, nebeae)")
    args = ap.parse_args()
    main(args.model)

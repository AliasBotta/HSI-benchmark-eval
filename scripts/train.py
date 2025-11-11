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


Usage:
    python3 -m scripts.train --model model_name
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import torch

from sklearn.decomposition import PCA

from utils.helpers import setup_logger, set_seed, ensure_dir
from utils.data_loading import list_all_instances
from utils.metrics import compute_all_metrics
from utils.spatial_filtering import apply_knn_filter
from utils.postprocessing import majority_voting
from models import get_runner
from utils.data_reduction import reduce_training_data


# ============================================================
# GLOBAL PIPELINE PARAMETERS 
# ============================================================

SEED = 42
DEVICE = "cuda"
GLOBAL_OUTPUT_DIR = Path("outputs")
PROCESSED_DIR = Path("data/processed")

# Partition
N_FOLDS = 5
SPLIT = (0.6, 0.2, 0.2)  # train, val, test

# Training-set reduction (paper presets)
REDUCTION_ENABLED = True
REDUCTION_CLUSTERS = 100          # paper
REDUCTION_TARGET_PER_CLASS = 1000 # {1000, 2000, 4000}; paper uses 1000 for speed

# KNN filter params (paper)
KNN_ENABLED = True
KNN_K = 40
KNN_WINDOW = 8
KNN_LAMBDA = 1
KNN_DISTANCE = "euclidean"

# HKM + Majority Voting
HKM_CLUSTERS = 24

# Classes
NUM_CLASSES = 4
# Mapping is fixed: {0:NT, 1:TT, 2:BV, 3:BG}


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
    """Create patient-level folds with circular test rotation (fixed-size test set per fold)."""
    pids = sorted(list({_pid_from_name(d.name) for d in instances}))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(pids)

    n_total = len(pids)
    n_train = int(split[0] * n_total)
    n_val = int(split[1] * n_total)
    n_test = max(1, n_total - n_train - n_val)  # fixed test size per fold

    folds = []
    for k in range(n_folds):
        start = (k * n_test) % n_total
        end = start + n_test

        # circular slice to guarantee fixed-size test set
        if end <= n_total:
            test_p = pids[start:end]
        else:
            wrap = end % n_total
            test_p = pids[start:] + pids[:wrap]

        remain = [p for p in pids if p not in test_p]
        # keep val proportion relative to (train+val)
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



def _evaluate_map(gt_map, pred_map):
    """Compute metrics on labeled pixels (gt>0) excluding BG from evaluation.
    Predictions that are BG where GT is non-BG are counted as errors."""
    gt = gt_map.flatten()
    pr = pred_map.flatten()

    mask_labeled = gt > 0
    gt = gt[mask_labeled] - 1                      # now in {0:NT,1:TT,2:BV,3:BG}
    pr = pr[mask_labeled]                          # predictions in {0..3}

    # Drop BG from GT, keep BG predictions as errors
    keep = gt != 3
    gt = gt[keep]                                  # values in {0,1,2}
    pr = pr[keep]                                  # values in {0,1,2,3}

    # Map predictions >2 to an invalid label (-1) to be counted as errors
    pr_eval = np.where(pr > 2, -1, pr)
    return compute_all_metrics(gt, pr_eval, num_classes=3, labels=[0, 1, 2])


# ============================================================
# Main Pipeline
# ============================================================

def main(model_name: str):
    set_seed(SEED)

    # <--- MODIFICA: Crea una cartella di output unica per questa run ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = GLOBAL_OUTPUT_DIR / f"{model_name}_{timestamp}"
    ensure_dir(run_output_dir)

    # Salva i log *dentro* la cartella della run
    logger = setup_logger(run_output_dir, name=f"train_{model_name}")
    logger.info(f"Run output directory: {run_output_dir}")
    # <--- FINE MODIFICA ---

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

    # <--- MODIFICA: Liste per conservare i runner e i risultati di ogni fold
    all_fold_runners = []
    all_fold_results = []
    # <--- FINE MODIFICA ---

    for k, (train_p, val_p, test_p) in enumerate(folds, start=1):
        logger.info(f"Fold {k}: train={len(train_p)}, val={len(val_p)}, test={len(test_p)}")

        # <--- MODIFICA: Istanzia un nuovo runner per ogni fold
        runner = get_runner(model_name)
        # <--- FINE MODIFICA ---

        # 1) Training & Validation pixels
        Xtr_raw, ytr_raw = _gather_training_pixels(processed_dir, instances, set(train_p))
        Xval_raw, yval_raw = _gather_training_pixels(processed_dir, instances, set(val_p))

        logger.info(f"[Fold {k}] Training pixels (raw): {len(ytr_raw)}")
        logger.info(f"[Fold {k}] Validation pixels (raw): {len(yval_raw)}")

        if Xtr_raw.size == 0 or Xval_raw.size == 0:
            logger.warning(f"[Fold {k}] Empty training or validation set. Skipping this fold.")
            all_fold_runners.append(None) # Aggiungi un placeholder
            all_fold_results.append(pd.DataFrame()) # Aggiungi df vuoto
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
        # Use the un-reduced validation set for tuning
        Xval, yval = Xval_raw, yval_raw

        logger.info(f"[Fold {k}] Reduced training pixels: {len(ytr)}")

        # --- distribution after reduction ---
        cls_r, cnt_r = np.unique(ytr, return_counts=True)
        logger.info(f"[Fold {k}] Reduced class distribution: {dict(zip(cls_r, cnt_r))}")

        # 3) Train model (now includes optimization on val set)
        runner.fit(Xtr, ytr, Xval, yval) # <-- MODIFIED SIGNATURE
        logger.info(f"[Fold {k}] Model training complete")

        # <--- MODIFICA: Salva il runner addestrato di questo fold
        all_fold_runners.append(runner)
        # <--- FINE MODIFICA ---

        # 4) Inference + post-processing
        fold_rows = []
        for d in instances:
            pid = _pid_from_name(d.name)
            if pid not in test_p:
                continue

            cube = np.load(d / "preprocessed_cube.npy")  # (bands,H,W)
            gt = np.load(d / "gtMap.npy")              # (H,W)
            bands, H, W = cube.shape

            # Check if this Ground Truth map contains any tumor pixels (Label 2)
            # gtMap labels are 1=NT, 2=TT, 3=BV, 4=BG
            if not np.any(gt == 2):
                logger.warning(f"[Fold {k} | PID {pid} | {d.name}] Skipping image: "
                               f"No tumor (TT) pixels found in Ground Truth.")
                continue

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
                class_knn,
                pc1=pca_img, # Ignored by new HKM, but cube is used
                cube=cube,
                n_clusters=HKM_CLUSTERS
            )

            # ---- Metrics (excluding BG) ----
            m_spec = _evaluate_map(gt, class_map)
            m_knn = _evaluate_map(gt, class_knn)
            m_mv = _evaluate_map(gt, class_mv)

            # <--- MODIFICA: Salva TUTTE le metriche, non solo f1 e oa
            row = {"fold": k, "patient": pid}
            row.update(_flatten_metrics_dict(m_spec, "spectral"))
            row.update(_flatten_metrics_dict(m_knn, "spatial"))
            row.update(_flatten_metrics_dict(m_mv, "mv"))
            # <--- FINE MODIFICA

            fold_rows.append(row)

            # <--- MODIFICA: Aggiornata la chiave per il log
            logger.info(f"[Fold {k} | PID {pid}] F1: Spec={row['spectral_f1_macro']:.3f} "
                        f"→ KNN={row['spatial_f1_macro']:.3f} → MV={row['mv_f1_macro']:.3f}")
            # <--- FINE MODIFICA ---

        all_fold_results.append(pd.DataFrame(fold_rows))

    # <--- MODIFICA: Logica di salvataggio e aggregazione a fine pipeline ---

    if not all_fold_results:
        logger.warning("No results were generated. Exiting.")
        return

    # 1. Salva i dati raw (metriche per paziente/immagine)
    df_all_images = pd.concat(all_fold_results)
    raw_metrics_path = run_output_dir / "metrics_per_patient.csv"
    df_all_images.to_csv(raw_metrics_path, index=False)
    logger.info(f"📊 Saved raw per-image metrics → {raw_metrics_path}")

    # 2. Calcola e salva il sommario
    if not df_all_images.empty:
        # <--- MODIFICA: Aggiorna l'elenco delle colonne per il summary
        # Ottieni tutte le colonne tranne 'fold' e 'patient'
        metric_cols = [col for col in df_all_images.columns if col not in ['fold', 'patient']]
        summary = df_all_images.groupby("fold")[metric_cols].mean()
        # <--- FINE MODIFICA ---

        summary.loc["mean"] = summary.mean()
        summary.loc["std"] = summary.std()

        summary_path = run_output_dir / "metrics_summary.csv"
        summary.to_csv(summary_path)
        logger.info(f"📊 Saved fold summary → {summary_path}")

        # 3. Identifica e salva il modello mediano
        try:
            # <--- MODIFICA: Usa la chiave corretta
            valid_folds = [f for f in range(1, N_FOLDS + 1) if f in summary.index]
            fold_scores = summary.loc[valid_folds, "spatial_f1_macro"] # <-- key updated
            # <--- FINE MODIFICA ---

            if not fold_scores.empty:
                median_score = fold_scores.median()

                # Trova l'indice (basato su 1) del fold più vicino alla mediana
                median_fold_num = (fold_scores - median_score).abs().idxmin() # e.g., 3
                median_fold_idx = median_fold_num - 1 # e.g., 2 (per lista 0-based)

                median_runner = all_fold_runners[median_fold_idx]

                if median_runner is not None:
                    logger.info(f"Median fold identified: Fold {median_fold_num} (Score: {fold_scores.loc[median_fold_num]:.4f}, Median: {median_score:.4f})")

                    # Salva il modello
                    if model_name == "dnn":
                        model_path = run_output_dir / "median_model.pth"
                        torch.save(median_runner.net.state_dict(), model_path)
                    else:
                        model_path = run_output_dir / "median_model.joblib"
                        joblib.dump(median_runner, model_path)

                    logger.info(f"✅ Saved median model → {model_path}")
                else:
                    logger.warning(f"Median fold ({median_fold_num}) was skipped, cannot save model.")
            else:
                 logger.warning("No valid fold scores found, cannot determine median model.")

        except Exception as e:
            logger.error(f"❌ Failed to save median model: {e}")

    logger.info(f"✅ Pipeline completed. Outputs in {run_output_dir}")
    # <--- FINE MODIFICA ---


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Model name (dnn, svm-l, svm-rbf, knn-e, rf, ebeae, nebeae)")
    args = ap.parse_args()
    main(args.model)

""""
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
    python3 -m scripts.train --model model_name --only-classifier
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



SEED = 42
DEVICE = "cuda"
GLOBAL_OUTPUT_DIR = Path("outputs")
PROCESSED_DIR = Path("data/processed")

N_FOLDS = 5
SPLIT = (0.6, 0.2, 0.2)

REDUCTION_ENABLED = True
REDUCTION_CLUSTERS = 100
REDUCTION_TARGET_PER_CLASS = 1000

KNN_ENABLED = True
KNN_K = 40
KNN_WINDOW = 8
KNN_LAMBDA = 1
KNN_DISTANCE = "euclidean"

HKM_CLUSTERS = 24

NUM_CLASSES = 4



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
    n_test = max(1, n_total - n_train - n_val)

    folds = []
    for k in range(n_folds):
        start = (k * n_test) % n_total
        end = start + n_test

        if end <= n_total:
            test_p = pids[start:end]
        else:
            wrap = end % n_total
            test_p = pids[start:] + pids[:wrap]

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
        cube = np.load(d / "preprocessed_cube.npy")
        gt = np.load(d / "gtMap.npy")
        bands, H, W = cube.shape
        flat = cube.reshape(bands, -1).T
        gt_f = gt.reshape(-1)
        m = gt_f > 0
        if m.any():
            X_list.append(flat[m])
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
    gt = gt[mask_labeled] - 1
    pr = pr[mask_labeled]

    keep = gt != 3
    gt = gt[keep]
    pr = pr[keep]

    pr_eval = np.where(pr > 2, -1, pr)
    return compute_all_metrics(gt, pr_eval, num_classes=3, labels=[0, 1, 2])

def _flatten_metrics_dict(m_dict, prefix):
    """Flatten metrics dictionary and add a prefix."""
    row = {}
    for k, v in m_dict.items():
        new_key = f"{prefix}_{k}"
        row[new_key] = v
    return row


def main(model_name: str, only_classifier: bool = False):
    set_seed(SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = GLOBAL_OUTPUT_DIR / f"{model_name}_{timestamp}"
    ensure_dir(run_output_dir)

    logger = setup_logger(run_output_dir, name=f"train_{model_name}")
    logger.info(f"Run output directory: {run_output_dir}")

    if only_classifier:
        logger.warning("Modalità --only-classifier ATTIVA. "
                       "Gli step KNN Filter e Majority Voting verranno saltati.")

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

    all_fold_runners = []
    all_fold_results = []

    for k, (train_p, val_p, test_p) in enumerate(folds, start=1):
        logger.info(f"Fold {k}: train={len(train_p)}, val={len(val_p)}, test={len(test_p)}")

        runner = get_runner(model_name)

        Xtr_raw, ytr_raw = _gather_training_pixels(processed_dir, instances, set(train_p))
        Xval_raw, yval_raw = _gather_training_pixels(processed_dir, instances, set(val_p))

        logger.info(f"[Fold {k}] Training pixels (raw): {len(ytr_raw)}")
        logger.info(f"[Fold {k}] Validation pixels (raw): {len(yval_raw)}")

        if Xtr_raw.size == 0 or Xval_raw.size == 0:
            logger.warning(f"[Fold {k}] Empty training or validation set. Skipping this fold.")
            all_fold_runners.append(None)
            all_fold_results.append(pd.DataFrame())
            continue

        cls, cnt = np.unique(ytr_raw, return_counts=True)
        logger.info(f"[Fold {k}] Raw class distribution (0:NT,1:TT,2:BV,3:BG): {dict(zip(cls, cnt))}")

        Xtr, ytr = reduce_training_data(
            Xtr_raw, ytr_raw,
            enabled=REDUCTION_ENABLED,
            clusters_per_class=REDUCTION_CLUSTERS,
            target_per_class=REDUCTION_TARGET_PER_CLASS,
            random_seed=SEED
        )
        Xval, yval = Xval_raw, yval_raw

        logger.info(f"[Fold {k}] Reduced training pixels: {len(ytr)}")

        cls_r, cnt_r = np.unique(ytr, return_counts=True)
        logger.info(f"[Fold {k}] Reduced class distribution: {dict(zip(cls_r, cnt_r))}")

        runner.fit(Xtr, ytr, Xval, yval)
        logger.info(f"[Fold {k}] Model training complete")

        all_fold_runners.append(runner)

        fold_rows = []
        for d in instances:
            pid = _pid_from_name(d.name)
            if pid not in test_p:
                continue

            cube = np.load(d / "preprocessed_cube.npy")
            gt = np.load(d / "gtMap.npy")
            bands, H, W = cube.shape

            if not np.any(gt == 2): # Il tuo check sul tumore (classe 2 in GT 1-indexed)
                logger.warning(f"[Fold {k} | PID {pid} | {d.name}] Skipping image: "
                               f"No tumor (TT) pixels found in Ground Truth.")
                continue

            # --- 1. GENERA LA PREDIZIONE "SPECTRAL" ---
            class_map, prob_all = runner.predict_full(cube)  # <-- Questa è la mappa che vuoi
            if prob_all.shape[-1] != 4:
                raise ValueError("Model must output probabilities with 4 channels [NT, TT, BV, BG].")

            # --- 2. SALVA I FILE .NPY (LA TUA RICHIESTA) ---
            # (Questo blocco è invariato e verrà eseguito sempre)
            pred_output_dir = run_output_dir / f"fold_{k}_predictions"
            ensure_dir(pred_output_dir)
            img_name = d.name
            pred_path = pred_output_dir / f"{img_name}_spectral_pred.npy"
            np.save(pred_path, class_map)
            gt_path = pred_output_dir / f"{img_name}_gt.npy"
            if not gt_path.exists():
                np.save(gt_path, gt)

            # --- 3. IL RESTO DELLA PIPELINE DIVENTA CONDIZIONALE ---

            # Calcola sempre le metriche spectral
            m_spec = _evaluate_map(gt, class_map)

            # Inizializza la riga del CSV solo con i dati spectral
            row = {"fold": k, "patient": pid, "image_name": img_name}
            row.update(_flatten_metrics_dict(m_spec, "spectral"))

            # Prepara il messaggio di log (per ora parziale)
            log_msg_f1 = f"[Fold {k} | PID {pid}] F1: Spec={row['spectral_f1_macro']:.3f}"

            if not only_classifier:
                # Esegui PCA (necessario per KNN e MV)
                pca = PCA(n_components=1)
                pca_img = pca.fit_transform(cube.reshape(bands, -1).T).reshape(H, W)

                # Esegui KNN Filter
                prob_flat = prob_all.reshape(-1, prob_all.shape[-1])
                prob_knn_flat = apply_knn_filter(
                    prob_flat, pc1=pca_img, cube=cube,
                    K=KNN_K, window_size=KNN_WINDOW,
                    lambda_=KNN_LAMBDA, distance=KNN_DISTANCE
                )
                class_knn = np.argmax(prob_knn_flat, axis=1).reshape(H, W)

                # Esegui Majority Voting
                class_mv = majority_voting(
                    class_knn,
                    pc1=pca_img,
                    cube=cube,
                    n_clusters=HKM_CLUSTERS
                )

                # Calcola metriche per KNN e MV
                m_knn = _evaluate_map(gt, class_knn)
                m_mv = _evaluate_map(gt, class_mv)

                # Aggiorna la riga del CSV
                row.update(_flatten_metrics_dict(m_knn, "spatial"))
                row.update(_flatten_metrics_dict(m_mv, "mv"))

                # Completa il messaggio di log
                log_msg_f1 += (f" → KNN={row['spatial_f1_macro']:.3f} "
                               f"→ MV={row['mv_f1_macro']:.3f}")

            # Salva la riga (completa o parziale) e stampa il log
            fold_rows.append(row)
            logger.info(log_msg_f1)

        all_fold_results.append(pd.DataFrame(fold_rows))


    if not all_fold_results:
        logger.warning("No results were generated. Exiting.")
        return

    df_all_images = pd.concat(all_fold_results)
    raw_metrics_path = run_output_dir / "metrics_per_patient.csv"
    df_all_images.to_csv(raw_metrics_path, index=False)
    logger.info(f"📊 Saved raw per-image metrics → {raw_metrics_path}")

    if not df_all_images.empty:
        # Calcola la media (questo funziona anche se le colonne 'spatial' e 'mv' mancano)
        metric_cols = [col for col in df_all_images.columns if col not in ['fold', 'patient', 'image_name']]
        summary = df_all_images.groupby("fold")[metric_cols].mean()

        summary.loc["mean"] = summary.mean()
        summary.loc["std"] = summary.std()

        summary_path = run_output_dir / "metrics_summary.csv"
        summary.to_csv(summary_path)
        logger.info(f"📊 Saved fold summary → {summary_path}")

        try:
            valid_folds = [f for f in range(1, N_FOLDS + 1) if f in summary.index]

            if only_classifier or "spatial_f1_macro" not in summary.columns:
                median_metric_key = "spectral_f1_macro"
                logger.info("Identifying median model based on 'spectral_f1_macro'")
            else:
                median_metric_key = "spatial_f1_macro"
                logger.info("Identifying median model based on 'spatial_f1_macro'")

            fold_scores = summary.loc[valid_folds, median_metric_key]

            if not fold_scores.empty:
                median_score = fold_scores.median()

                median_fold_num = (fold_scores - median_score).abs().idxmin()
                median_fold_idx = median_fold_num - 1

                median_runner = all_fold_runners[median_fold_idx]

                if median_runner is not None:
                    logger.info(f"Median fold identified: Fold {median_fold_num} "
                                f"(Score: {fold_scores.loc[median_fold_num]:.4f}, Median: {median_score:.4f})")

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



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Model name (dnn, svm-l, svm-rbf, knn-e, rf, ebeae, nebeae)")
    ap.add_argument("--only-classifier",
                    action="store_true",
                    help="Only run the base classifier (Spectral) and skip spatial/MV steps.")
    args = ap.parse_args()
    # Passa il nuovo argomento alla funzione main
    main(args.model, only_classifier=args.only_classifier)

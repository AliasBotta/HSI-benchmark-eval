"""
train.py
---------
Supervised training pipeline for the HSI DNN baseline, extended to include
the three configurations described in the benchmark paper:
1. Spectral (pure DNN)
2. Spatial–Spectral (KNN smoothing)
3. Majority Voting (HKM + voting)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import OmegaConf
import os

from utils.helpers import setup_logger, set_seed, get_device, ensure_dir
from utils.dataset import load_all_processed, make_kfold_splits
from utils.metrics import compute_all_metrics
from utils.data_reduction import reduce_training_data
from models.dnn_1d import DNN1D
from utils.spatial_filtering import apply_knn_filter
from utils.postprocessing import majority_voting


# ============================================================
# Evaluation Function
# ============================================================

def evaluate(model, loader, device, num_classes, return_probs=False):
    """Run inference and compute metrics (optionally return probabilities)."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = softmax(logits).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.append(preds)
            all_labels.append(yb.numpy())
            all_probs.append(probs)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)

    metrics = compute_all_metrics(y_true, y_pred, num_classes)
    if return_probs:
        return metrics, y_true, y_probs
    return metrics


# ============================================================
# Main Training Routine
# ============================================================

def main(cfg_path):
    # --- Load config ---
    cfg = OmegaConf.load(cfg_path)
    default_cfg_path = os.path.join("configs", "default.yaml")
    if os.path.exists(default_cfg_path):
        base_cfg = OmegaConf.load(default_cfg_path)
        cfg = OmegaConf.merge(base_cfg, cfg)

    set_seed(cfg.experiment.seed)
    device = get_device(cfg.experiment.device)
    logger = setup_logger("outputs/logs", name=cfg.experiment.name)
    ensure_dir(cfg.experiment.output_dir)

    # --- Load dataset ---
    X, y, pids = load_all_processed(cfg.data.processed_dir)
    mask = (y > 0) & (y < 4)
    X, y, pids = X[mask], y[mask] - 1, pids[mask]
    logger.info(f"Loaded {len(y)} samples from processed dataset.")

    # =======================================================
    # Train once (Spectral DNN), evaluate all 3 configurations
    # =======================================================
    all_results = []

    for fold_idx, train_set, val_set, test_set in make_kfold_splits(X, y, pids, cfg):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_set, val_set, test_set
        logger.info(f"\n🟩 Fold {fold_idx+1}: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

        # --- Data reduction ---
        if getattr(cfg.reduction, "enabled", False):
            X_train, y_train = reduce_training_data(X_train, y_train, cfg)

        # --- Prepare tensors ---
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                                  batch_size=cfg.training.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                                batch_size=cfg.training.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test_t, y_test_t),
                                 batch_size=cfg.training.batch_size, shuffle=False)

        # --- Define model ---
        model = DNN1D(cfg.model).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.learning_rate,
            momentum=getattr(cfg.training, "momentum", 0.9)
        )

        # --- Train ---
        for epoch in range(cfg.training.epochs):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                val_metrics = evaluate(model, val_loader, device, cfg.model.num_classes)
                logger.info(f"[Fold {fold_idx+1}] Epoch {epoch+1:03d} | "
                            f"Loss={total_loss/len(train_loader):.4f} | "
                            f"Val F1={val_metrics['f1_macro']:.3f} | OA={val_metrics['oa']:.3f}")

        logger.info(f"[Fold {fold_idx+1}] ✅ Training completed")

        # =======================================================
        # 1. Spectral Evaluation
        # =======================================================
        metrics_spectral, y_true, prob_map = evaluate(
            model, test_loader, device, cfg.model.num_classes, return_probs=True
        )
        logger.info(f"[Spectral] F1={metrics_spectral['f1_macro']:.3f}, OA={metrics_spectral['oa']:.3f}")

               # =======================================================
        # 2. Spatial–Spectral (KNN filter)
        # =======================================================
        # Attempt to load cube from processed data
        patient_id = str(pids[0])
        cube_path = Path(cfg.data.processed_dir) / patient_id / "preprocessed_cube.npy"
        logger.info(f"[DEBUG] patient_id example: {patient_id}")

        if cube_path.exists():
            cube = np.load(cube_path)
            logger.info(f"[Spatial–Spectral] Loaded cube for KNN filtering: {cube_path}")
        else:
            cube = None
            logger.warning("[Spatial–Spectral] ⚠ Cube not found, fallback to 1D smoothing")

        prob_map_knn = apply_knn_filter(prob_map, cube=cube, cfg=cfg)
        preds_knn = np.argmax(prob_map_knn, axis=1)
        metrics_spatial = compute_all_metrics(y_true, preds_knn, cfg.model.num_classes)
        logger.info(f"[Spatial–Spectral] F1={metrics_spatial['f1_macro']:.3f}, OA={metrics_spatial['oa']:.3f}")

        # =======================================================
        # 3. Majority Voting (HKM + voting)
        # =======================================================
        preds_mv = majority_voting(prob_map_knn, cube=cube, cfg=cfg)
        metrics_mv = compute_all_metrics(y_true, preds_mv, cfg.model.num_classes)
        logger.info(f"[Majority Voting] F1={metrics_mv['f1_macro']:.3f}, OA={metrics_mv['oa']:.3f}")

        # --- Save results for this fold ---
        all_results.append({
            "fold": fold_idx + 1,
            "spectral_f1": metrics_spectral["f1_macro"],
            "spatial_f1": metrics_spatial["f1_macro"],
            "mv_f1": metrics_mv["f1_macro"],
            "spectral_oa": metrics_spectral["oa"],
            "spatial_oa": metrics_spatial["oa"],
            "mv_oa": metrics_mv["oa"]
        })

    # =======================================================
    # Save cross-fold summary
    # =======================================================
    df = pd.DataFrame(all_results)
    df.loc["mean"] = df.mean(numeric_only=True)
    df.loc["std"] = df.std(numeric_only=True)
    out_path = Path(cfg.experiment.output_dir) / "metrics_summary.csv"
    df.to_csv(out_path, index=True)
    logger.info(f"\n📊 Saved summary results → {out_path}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML file")
    args = parser.parse_args()
    main(args.config)

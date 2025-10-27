"""
train.py
---------
Supervised training pipeline for HSI DNN baseline with evaluation.
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
from utils.dataset import load_all_processed, split_by_patient
from utils.metrics import compute_all_metrics
from models.dnn_1d import DNN1D


def evaluate(model, loader, device, num_classes):
    """Run inference and compute metrics."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    return compute_all_metrics(y_true, y_pred, num_classes)


def main(cfg_path):
    # --- Load config, merging with default if specified ---
    cfg = OmegaConf.load(cfg_path)
    default_cfg_path = os.path.join("configs", "default.yaml")
    if "defaults" in cfg and os.path.exists(default_cfg_path):
        base_cfg = OmegaConf.load(default_cfg_path)
        cfg = OmegaConf.merge(base_cfg, cfg)
    set_seed(cfg.experiment.seed)
    device = get_device(cfg.experiment.device)
    logger = setup_logger("outputs/logs", name=cfg.experiment.name)

    # Load dataset
    X, y, pids = load_all_processed(cfg.data.processed_dir)
    # --- Filter valid labels: keep TT, NT, BV (exclude 0=unlabeled, 4=BG)
    mask = (y > 0) & (y < 4)
    X, y, pids = X[mask], y[mask] - 1, pids[mask]
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_by_patient(X, y, pids, cfg)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=cfg.training.batch_size, shuffle=False)

    # Model setup
    model = DNN1D(cfg.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.learning_rate)

    # Training loop
    logger.info(f"Training DNN for {cfg.training.epochs} epochs...")
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

        # Optional: validation loss
        if (epoch + 1) % 10 == 0:
            metrics_val = evaluate(model, val_loader, device, cfg.model.num_classes)
            logger.info(f"Epoch {epoch+1}: TrainLoss={total_loss/len(train_loader):.4f}, "
                        f"Val F1={metrics_val['f1_macro']:.3f}, OA={metrics_val['oa']:.3f}")

    # Save model
    ensure_dir(cfg.experiment.output_dir)
    model_path = Path(cfg.experiment.output_dir) / "model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"✅ Training complete, model saved → {model_path}")

    # --- Evaluation on test set ---
    logger.info("Evaluating on test set...")
    metrics_test = evaluate(model, test_loader, device, cfg.model.num_classes)
    logger.info(f"Test results: F1={metrics_test['f1_macro']:.3f}, "
                f"OA={metrics_test['oa']:.3f}, "
                f"Sens={metrics_test['sensitivity_mean']:.3f}, "
                f"Spec={metrics_test['specificity_mean']:.3f}")

    # Save results to CSV
    df = pd.DataFrame([metrics_test])
    df.to_csv(Path(cfg.experiment.output_dir) / "metrics.csv", index=False)
    logger.info("📊 Saved metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML file")
    args = parser.parse_args()
    main(args.config)


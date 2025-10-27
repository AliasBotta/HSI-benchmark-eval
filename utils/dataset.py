"""
dataset.py
-----------
Utility functions for loading preprocessed HSI cubes and GT maps
into flat pixel-wise arrays for training and evaluation.
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


def load_all_processed(data_dir):
    """
    Load all preprocessed cubes and GT maps from data/processed/.
    Returns:
        X: np.ndarray (N_pixels, bands)
        y: np.ndarray (N_pixels,)
        patient_ids: list of str (same length as y)
    """
    X_list, y_list, pid_list = [], [], []
    root = Path(data_dir)
    instances = sorted([p for p in root.iterdir() if p.is_dir()])

    for inst in instances:
        cube_path = inst / "preprocessed_cube.npy"
        gt_path = inst / "gtMap.npy"
        if not cube_path.exists() or not gt_path.exists():
            print(f"[WARNING] Missing data in {inst}")
            continue

        cube = np.load(cube_path)       # (bands, H, W)
        gt = np.load(gt_path)           # (H, W)

        # flatten spatial dims
        if gt.ndim == 3 and gt.shape[-1] == 1:
            gt = gt.squeeze(-1)
        H, W = gt.shape
        cube_flat = cube.reshape(cube.shape[0], -1).T   # (H*W, bands)
        gt_flat = gt.flatten()

        # keep only labeled pixels
        mask = gt_flat > 0
        X_list.append(cube_flat[mask])
        y_list.append(gt_flat[mask])
        pid_list.extend([inst.name] * mask.sum())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y, np.array(pid_list)


def split_by_patient(X, y, patient_ids, cfg):
    """
    Split data into train/val/test ensuring no patient overlap.
    Uses stratified split on patient-level if possible.
    """
    unique_pids = np.unique(patient_ids)
    rng = np.random.default_rng(cfg.partition.random_seed)
    rng.shuffle(unique_pids)

    n_total = len(unique_pids)
    n_train = int(cfg.partition.split[0] * n_total)
    n_val = int(cfg.partition.split[1] * n_total)

    train_pids = unique_pids[:n_train]
    val_pids = unique_pids[n_train:n_train+n_val]
    test_pids = unique_pids[n_train+n_val:]

    def select(pids):
        mask = np.isin(patient_ids, pids)
        return X[mask], y[mask]

    X_train, y_train = select(train_pids)
    X_val, y_val = select(val_pids)
    X_test, y_test = select(test_pids)

    print(f"Patients: train={len(train_pids)}, val={len(val_pids)}, test={len(test_pids)}")
    print(f"Samples: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


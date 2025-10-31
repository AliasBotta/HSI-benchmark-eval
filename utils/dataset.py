"""
dataset.py
-----------
Utility functions for loading preprocessed HSI cubes and GT maps
into flat pixel-wise arrays for training and evaluation.
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def make_kfold_splits(X, y, patient_ids, cfg):
    """
    Perform k-fold cross-validation at patient level according to config split ratios.

    cfg.partition.split = [train_ratio, val_ratio, test_ratio]
    ensures approximately that each fold uses the specified proportions.
    """

    unique_pids = np.unique(patient_ids)
    rng = np.random.default_rng(cfg.partition.random_seed)
    rng.shuffle(unique_pids)

    n_total = len(unique_pids)
    n_train = int(cfg.partition.split[0] * n_total)
    n_val   = int(cfg.partition.split[1] * n_total)
    n_test  = n_total - n_train - n_val

    # 5 folds => rotate test set position across splits
    fold_size = max(1, n_test)
    folds = []
    for fold_idx in range(cfg.partition.folds):
        start = fold_idx * fold_size % n_total
        end   = (start + fold_size) % n_total

        if end > start:
            test_pids = unique_pids[start:end]
            remain_pids = np.concatenate([unique_pids[:start], unique_pids[end:]])
        else:
            test_pids = np.concatenate([unique_pids[start:], unique_pids[:end]])
            remain_pids = np.setdiff1d(unique_pids, test_pids)

        # Within remaining, take val and train respecting ratios
        n_remain = len(remain_pids)
        n_val_local = int(cfg.partition.split[1] / (cfg.partition.split[0] + cfg.partition.split[1]) * n_remain)
        rng.shuffle(remain_pids)
        val_pids = remain_pids[:n_val_local]
        train_pids = remain_pids[n_val_local:]

        def select(pids):
            mask = np.isin(patient_ids, pids)
            return X[mask], y[mask]

        yield fold_idx, select(train_pids), select(val_pids), select(test_pids)

def extract_patches(X_cube, gt, patch_size=5):
    """
    Extract (spectral, spatial) patches centered on each labeled pixel.
    Returns:
        X_patches: (N, bands, H, W)
        y: (N,)
    """
    pad = patch_size // 2
    bands, H, W = X_cube.shape
    X_pad = np.pad(X_cube, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')

    coords = np.argwhere(gt > 0)
    X_patches, y = [], []

    for (yy, xx) in coords:
        patch = X_pad[:, yy:yy+patch_size, xx:xx+patch_size]
        X_patches.append(patch)
        y.append(gt[yy, xx])

    return np.stack(X_patches), np.array(y)



def load_all_processed(data_dir, cfg=None):
    """
    Load all preprocessed cubes and GT maps from data/processed/.
    If cfg.data.spatial_enabled=True, extracts (spectral, spatial) patches.

    Returns:
        X: np.ndarray (N_pixels, bands) or (N_patches, bands, H, W)
        y: np.ndarray (N,)
        patient_ids: np.ndarray (N,)
    """
    from utils.dataset import extract_patches  # ensure helper available
    X_list, y_list, pid_list = [], [], []
    root = Path(data_dir)
    instances = sorted([p for p in root.iterdir() if p.is_dir()])

    spatial_mode = False
    patch_size = 5
    if cfg is not None and getattr(cfg.data, "spatial_enabled", False):
        spatial_mode = True
        patch_size = getattr(cfg.data, "patch_size", 5)

    for inst in instances:
        cube_path = inst / "preprocessed_cube.npy"
        gt_path = inst / "gtMap.npy"
        if not cube_path.exists() or not gt_path.exists():
            print(f"[WARNING] Missing data in {inst}")
            continue

        cube = np.load(cube_path)  # (bands, H, W)
        gt = np.load(gt_path)      # (H, W)
        if gt.ndim == 3 and gt.shape[-1] == 1:
            gt = gt.squeeze(-1)

        pid = inst.name.split('-')[0]

        if spatial_mode:
            # --- Spatial–Spectral patch extraction ---
            X_patches, y_patches = extract_patches(cube, gt, patch_size)
            X_list.append(X_patches)
            y_list.append(y_patches)
            pid_list.extend([pid] * len(y_patches))
        else:
            # --- Spectral-only mode (flat pixels) ---
            H, W = gt.shape
            cube_flat = cube.reshape(cube.shape[0], -1).T   # (H*W, bands)
            gt_flat = gt.flatten()
            mask = gt_flat > 0
            X_list.append(cube_flat[mask])
            y_list.append(gt_flat[mask])
            pid_list.extend([pid] * mask.sum())

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


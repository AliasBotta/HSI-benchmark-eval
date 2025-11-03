"""
data_loading.py
---------------
Functions for reading hyperspectral cubes and ground-truth maps
from ENVI .hdr/.raw files in the HSI benchmark dataset.
"""

import os
from pathlib import Path
import numpy as np
import spectral


# ============================================================
# ENVI Readers
# ============================================================

def load_envi_image(base_path):
    """
    Load an ENVI image given its base path (without .hdr extension).
    Returns a NumPy array with shape (bands, height, width).
    """
    hdr_path = str(base_path) + ".hdr"
    if not os.path.exists(hdr_path):
        raise FileNotFoundError(f"Missing header file for {base_path}")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Missing image data for {base_path}")

    img = spectral.envi.open(hdr_path, str(base_path))
    arr = np.asarray(img.load()).astype(np.float32)

    # Ensure correct shape order (bands, H, W)
    if arr.shape[0] != img.nbands:
        arr = np.transpose(arr, (2, 0, 1))
    return arr


def load_gt_map(base_path):
    """
    Load the ground-truth map (pixel-level labels).
    Non-labeled pixels are expected to be 0.
    """
    hdr_path = str(base_path) + ".hdr"
    if not os.path.exists(hdr_path):
        raise FileNotFoundError(f"Missing header file for {base_path}")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Missing ground truth data for {base_path}")

    gt = spectral.envi.open(hdr_path, str(base_path))
    gt_map = np.asarray(gt.load()).astype(np.int32)

    # replace NaNs and negatives with 0
    gt_map = np.nan_to_num(gt_map, nan=0).clip(min=0)
    return gt_map


def load_dataset_instance(instance_dir,
                          raw_name="raw",
                          white_ref_name="whiteReference",
                          dark_ref_name="darkReference",
                          gt_map_name="gtMap"):
    """
    Load all components of a single dataset instance folder:
      - raw hyperspectral cube
      - white reference
      - dark reference
      - ground truth map

    Returns
    -------
    data : dict[str, np.ndarray]
        Keys: 'raw', 'white_ref', 'dark_ref', 'gt'
    """
    d = Path(instance_dir)
    data = {}

    def safe_load_image(base_name):
        # Try lowercase / capitalized variants
        for name in [base_name, base_name.capitalize()]:
            path = d / name
            if path.exists() or (path.with_suffix(".hdr").exists()):
                return load_envi_image(path)
        raise FileNotFoundError(f"Missing file for {base_name} in {d}")

    try:
        data["raw"] = safe_load_image(raw_name)
        data["white_ref"] = safe_load_image(white_ref_name)
        data["dark_ref"] = safe_load_image(dark_ref_name)
        data["gt"] = load_gt_map(d / gt_map_name)
    except Exception as e:
        print(f"[WARNING] Could not load {instance_dir}: {e}")
        return None

    # --- Normalize GT shape ---
    gt = data["gt"]
    if gt.ndim == 3 and gt.shape[-1] == 1:
        gt = gt.squeeze(-1)  # (H, W, 1) → (H, W)

    raw_h, raw_w = data["raw"].shape[1:]
    gt_h, gt_w = gt.shape[:2]

    # --- Handle small mismatches silently ---
    if (raw_h, raw_w) != (gt_h, gt_w):
        # if difference only by 1 px (rounding, sensor cropping), crop to overlap
        min_h = min(raw_h, gt_h)
        min_w = min(raw_w, gt_w)
        gt = gt[:min_h, :min_w]
        data["raw"] = data["raw"][:, :min_h, :min_w]
        # optional debug print
        print(f"[INFO] Auto-aligned shapes in {instance_dir}: "
              f"raw→{data['raw'].shape[1:]}, gt→{gt.shape}")

    data["gt"] = gt
    return data



# ============================================================
# Directory Utilities
# ============================================================

def list_all_instances(root_dir):
    """
    Recursively list all valid dataset instances under multiple campaigns.
    Works with structure like:
        FirstCampaign/004-02/
        SecondCampaign/037-01/
        ThirdCampaign/050-01/

    Returns a list of Paths, each pointing to the folder containing
    raw/whiteReference/darkReference/gtMap.
    """
    instances = []
    root_dir = Path(root_dir)
    for campaign_dir in root_dir.iterdir():
        if not campaign_dir.is_dir():
            continue
        for instance_dir in campaign_dir.iterdir():
            if not instance_dir.is_dir():
                continue
            if (instance_dir / "raw.hdr").exists() or (instance_dir / "raw").exists():
                instances.append(instance_dir)
    return sorted(instances)


def load_instance_paths(instance_dir):
    """
    Return a dictionary of all expected file paths inside an instance folder.
    Helps debugging dataset integrity.
    """
    d = Path(instance_dir)
    paths = {
        "raw": d / "raw",
        "raw_hdr": d / "raw.hdr",
        "white_ref": d / "whiteReference",
        "white_hdr": d / "whiteReference.hdr",
        "dark_ref": d / "darkReference",
        "dark_hdr": d / "darkReference.hdr",
        "gt": d / "gtMap",
        "gt_hdr": d / "gtMap.hdr",
    }
    return paths


# ============================================================
# Saving Utilities
# ============================================================

def save_numpy_arrays(out_dir, cube, gt_map):
    """
    Save preprocessed cube and GT map to .npy files for fast reload later.
    Each instance will have:
        preprocessed_cube.npy
        gtMap.npy
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "preprocessed_cube.npy", cube)
    np.save(out_dir / "gtMap.npy", gt_map)

    print(f"[INFO] Saved → {out_dir/'preprocessed_cube.npy'} and gtMap.npy")

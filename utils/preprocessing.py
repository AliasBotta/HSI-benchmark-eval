"""
preprocessing.py
----------------
Implements the full pre-processing pipeline for hyperspectral cubes:
calibration, smoothing, trimming, downsampling, normalization,
and absorbance conversion, as described in the HIRIS-Lab benchmark.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


# ============================================================
# Calibration and Noise Correction
# ============================================================

def calibrate_hsi(raw, white, dark):
    """
    Apply radiometric calibration: CI = (RI - DI) / (WI - DI).
    Clips the result to [0, 1].
    """
    eps = 1e-8
    calibrated = (raw - dark) / (white - dark + eps)
    calibrated = np.clip(calibrated, 0, 1)
    return calibrated


def smooth_spectral(data, window_size=5):
    """
    Apply a moving average filter along spectral dimension.
    Reduces sensor noise between adjacent wavelengths.
    """
    if window_size < 2:
        return data
    return uniform_filter1d(data, size=window_size, axis=0, mode="nearest")


def remove_noisy_channels(data, start, end):
    """
    Remove extreme noisy bands as per the original benchmark.
    Args:
        data: (bands, H, W)
        start: number of bands to remove at beginning
        end: number of bands to remove at end
    """
    n_bands = data.shape[0]
    start = min(start, n_bands - 1)
    end = min(end, n_bands - start - 1)
    return data[start:n_bands - end]


# ============================================================
# Spectral Processing
# ============================================================

def downsample_spectrum(data, step=3.61, final_channels=128):
    """
    Uniformly sample the spectral dimension to reduce band count.
    Approximates 3.61 nm spectral step reduction.
    """
    n_bands = data.shape[0]
    if final_channels >= n_bands:
        return data
    idx = np.linspace(0, n_bands - 1, final_channels).astype(int)
    return data[idx]


def normalize_minmax(data, axis=(1,2)):
    dmin = data.min(axis=axis, keepdims=True)
    dmax = data.max(axis=axis, keepdims=True)
    return (data - dmin) / (dmax - dmin + 1e-8)


def convert_to_absorbance(reflectance):
    """
    Convert reflectance to absorbance: A = -log(R).
    Clips reflectance values to avoid log(0).
    """
    reflectance = np.clip(reflectance, 1e-6, 1.0)
    return -np.log(reflectance)


# ============================================================
# Full Preprocessing Chain
# ============================================================

def preprocess_hsi_cube(raw, white, dark, cfg):
    """
    Execute the full preprocessing pipeline in the correct order:
      1. Radiometric calibration
      2. (Optional) Smoothing
      3. Remove noisy bands
      4. (Optional) Convert to absorbance
      5. (Optional) Spectral downsampling
      6. (Optional) Normalization
    Returns:
        cube: np.ndarray of shape (bands, H, W)
    """
    # --- 1. Calibration ---
    cube = calibrate_hsi(raw, white, dark)

    # --- 2. Smoothing ---
    if getattr(cfg.data.smoothing, "enabled", False):
        cube = smooth_spectral(cube, cfg.data.smoothing.window)

    # --- 3. Remove noisy channels ---
    cube = remove_noisy_channels(
        cube,
        cfg.data.remove_bands.start,
        cfg.data.remove_bands.end
    )

    # --- 4. Convert to absorbance (optional) ---
    if getattr(cfg.data, "absorbance_conversion", False):
        cube = convert_to_absorbance(cube)

    # --- 5. Downsampling (optional) ---
    if getattr(cfg.data.downsampling, "enabled", False):
        cube = downsample_spectrum(
            cube,
            cfg.data.downsampling.step_nm,
            cfg.data.downsampling.final_channels
        )

    # --- 6. Normalization ---
    norm_method = getattr(cfg.data, "normalization", "minmax")
    if norm_method == "minmax":
        cube = normalize_minmax(cube)

    return cube

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
    Apply a moving average filter along the spectral dimension
    to remove spikes in the signal.
    """
    if window_size < 2:
        return data
    return uniform_filter1d(data, size=window_size, axis=0, mode="nearest") #1d convolution


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

def downsample_spectrum(data, final_channels=128):
    """
    Uniformly sample the spectral dimension to reduce band count.
    Approximates 3.61 nm spectral step reduction.
    """
    n_bands = data.shape[0]
    if final_channels >= n_bands:
        return data
    idx = np.linspace(0, n_bands - 1, final_channels).astype(int)
    return data[idx]


def normalize_minmax(data):
    """
    Normalize cube values PIXEL-WISE to [0, 1] to focus on spectral shape.
    This is the method (per-pixel normalization) described in the paper.
    """
    # Input shape: (bands, H, W)
    bands, H, W = data.shape
    
    # Reshape to (H*W, bands)
    flat = data.reshape(bands, -1).T
    
    # Calculate min/max per pixel (row-wise)
    min_vals = flat.min(axis=1, keepdims=True)
    max_vals = flat.max(axis=1, keepdims=True)
    
    # Calculate denominator, adding epsilon for flat spectra (max=min)
    denom = max_vals - min_vals + 1e-8
    
    # Apply normalization
    norm_flat = (flat - min_vals) / denom
    
    # Reshape back to (bands, H, W)
    return norm_flat.T.reshape(bands, H, W)


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

def preprocess_hsi_cube(
    raw,
    white,
    dark,
    remove_bands=(56, 126),
    smoothing_enabled=True,
    smoothing_window=5,
    absorbance_conversion=False,
    normalization="minmax",
    downsampling_enabled=True,
    final_channels=128,
):
    """
    Execute the full preprocessing pipeline in the correct order:
      1. Radiometric calibration
      2. Smoothing (optional)
      3. Remove noisy bands
      4. Convert to absorbance (optional) (not used)
      5. Spectral downsampling (optional)
      6. Normalization
    Returns:
        cube: np.ndarray of shape (bands, H, W)
    """
    # 1. Radiometric calibration
    cube = calibrate_hsi(raw, white, dark)

    # 2. Smoothing
    if smoothing_enabled:
        cube = smooth_spectral(cube, window_size=smoothing_window)

    # 3. Remove noisy bands
    cube = remove_noisy_channels(cube, *remove_bands)

    # 4. Absorbance conversion
    if absorbance_conversion:
        cube = convert_to_absorbance(cube)

    # 5. Spectral downsampling
    if downsampling_enabled:
        cube = downsample_spectrum(cube, final_channels=final_channels)

    # 6. Normalization
    if normalization == "minmax":
        cube = normalize_minmax(cube)

    return cube

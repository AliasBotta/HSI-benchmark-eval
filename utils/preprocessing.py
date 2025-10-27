"""
preprocessing.py
----------------
Implements the pre-processing pipeline for hyperspectral cubes:
calibration, smoothing, trimming, downsampling, normalization,
and absorbance conversion.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def calibrate_hsi(raw, white, dark):
    """Apply radiometric calibration: CI = (RI - DI) / (WI - DI)."""
    eps = 1e-8
    calibrated = (raw - dark) / (white - dark + eps)
    calibrated = np.clip(calibrated, 0, 1)
    return calibrated


def smooth_spectral(data, window_size=5):
    """Apply a moving average filter along spectral dimension."""
    return uniform_filter1d(data, size=window_size, axis=0, mode="nearest")


def remove_noisy_channels(data, start, end):
    """Remove extreme noisy bands as per paper."""
    return data[start : -end]


def downsample_spectrum(data, step=3.61, final_channels=128):
    """
    Decimate the spectral channels by uniform sampling.
    This approximates the 3.61 nm interval reduction.
    """
    n_bands = data.shape[0]
    idx = np.linspace(0, n_bands - 1, final_channels).astype(int)
    return data[idx]


def normalize_minmax(data):
    """Normalize spectral values to [0, 1]."""
    dmin, dmax = data.min(), data.max()
    return (data - dmin) / (dmax - dmin + 1e-8)


def convert_to_absorbance(reflectance):
    """Convert reflectance to absorbance: A = -log(R)."""
    reflectance = np.clip(reflectance, 1e-6, 1.0)
    return -np.log(reflectance)


def preprocess_hsi_cube(raw, white, dark, cfg):
    """
    Full pre-processing chain as per the paper.
    Returns a 3D calibrated and preprocessed HSI cube.
    """
    cube = calibrate_hsi(raw, white, dark)

    if cfg.data.smoothing.enabled:
        cube = smooth_spectral(cube, cfg.data.smoothing.window)

    cube = remove_noisy_channels(
        cube, cfg.data.remove_bands.start, cfg.data.remove_bands.end
    )

    if cfg.data.absorbance_conversion:
        cube = convert_to_absorbance(cube)

    if cfg.data.downsampling.enabled:
        cube = downsample_spectrum(
            cube, cfg.data.downsampling.step_nm, cfg.data.downsampling.final_channels
        )

    if cfg.data.normalization == "minmax":
        cube = normalize_minmax(cube)

    return cube


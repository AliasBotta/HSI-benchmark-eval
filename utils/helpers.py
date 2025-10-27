"""
helpers.py
-----------

General-purpose helper utilities for configuration, logging, reproducibility,
and simple system functions used across the HSI benchmark pipeline.
"""

import os
import sys
import time
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf


# ============================================================
# Configuration Utilities
# ============================================================

def load_config(cfg_path: str):
    """Load an OmegaConf YAML config file."""
    cfg = OmegaConf.load(cfg_path)
    return OmegaConf.to_container(cfg, resolve=True)


def print_config(cfg):
    """Pretty print the configuration tree."""
    print(OmegaConf.to_yaml(cfg))


# ============================================================
# Experiment Setup
# ============================================================

def set_seed(seed: int = 42):
    """Ensure reproducibility across random, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda"):
    """Return available device."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Logging Utilities
# ============================================================

def setup_logger(log_dir: str, name: str = "train", level=logging.INFO):
    """
    Create a console and file logger.
    Args:
        log_dir: Directory where to store the log file.
        name: Name of the logger (train/eval/etc.).
        level: Logging level.
    Returns:
        logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate handlers

    # Stream handler (console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    # Avoid re-adding handlers if already set
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    logger.info(f"Logger initialized → {log_file}")
    return logger


# ============================================================
# File and Directory Helpers
# ============================================================

def ensure_dir(path: str):
    """Create a directory if it doesn’t exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_timestamp():
    """Return a formatted timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_output_dir(cfg, subfolder=None):
    """Return the experiment’s output directory path."""
    base_dir = Path(cfg.experiment.output_dir)
    if subfolder:
        return base_dir / subfolder
    return base_dir


def list_files(directory: str, extension: str = None):
    """List all files in a directory with optional extension filter."""
    files = sorted(Path(directory).glob(f"*{extension or ''}"))
    return [str(f) for f in files]


# ============================================================
# Timer Context Manager
# ============================================================

class Timer:
    """Simple context manager for timing code blocks."""
    def __init__(self, name="Block"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        print(f"[TIMER] Starting {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"[TIMER] {self.name} finished in {elapsed:.2f}s")


# ============================================================
# Miscellaneous
# ============================================================

def flatten_dict(d, parent_key="", sep="."):
    """Flatten nested dictionaries (useful for logging configs)."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


"""
preprocess.py
-------------
Script to preprocess all HSI dataset instances according to config.
Usage:
    python scripts/preprocess.py --config configs/default.yaml
"""

import os
import argparse
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

from utils.data_loading import list_all_instances, load_dataset_instance
from utils.preprocessing import preprocess_hsi_cube
from utils.helpers import ensure_dir, setup_logger


def save_preprocessed_cube(out_dir, cube, gt=None):
    """Save the preprocessed cube (and optionally GT) as NumPy .npy files."""
    ensure_dir(out_dir)
    np.save(Path(out_dir) / "preprocessed_cube.npy", cube)
    if gt is not None:
        np.save(Path(out_dir) / "gtMap.npy", gt)


def main(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    logger = setup_logger("outputs/logs", name="preprocess")

    root_dir = Path(cfg.data.root_dir)
    processed_dir = Path(cfg.data.processed_dir)
    ensure_dir(processed_dir)

    instances = list_all_instances(root_dir)
    logger.info(f"Found {len(instances)} dataset instances.")

    for instance in instances:
        logger.info(f"Processing {instance.name}...")
        data = load_dataset_instance(instance, cfg)
        if data is None:
            logger.warning(f"Skipping {instance}")
            continue

        cube = preprocess_hsi_cube(
            data["raw"], data["white_ref"], data["dark_ref"], cfg
        )

        out_dir = processed_dir / instance.name
        save_preprocessed_cube(out_dir, cube, data["gt"])

        logger.info(f"Saved preprocessed cube to {out_dir}")

    logger.info("Preprocessing completed for all instances.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)


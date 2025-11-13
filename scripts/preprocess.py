"""
preprocess.py
-------------
Preprocess all HSI dataset instances according to the benchmark pipeline.

Usage:
    python3 -m scripts.preprocess
"""

from pathlib import Path
import numpy as np

from utils.data_loading import list_all_instances, load_dataset_instance
from utils.preprocessing import preprocess_hsi_cube
from utils.helpers import ensure_dir, setup_logger

ROOT_DIR = Path("data/hsi_dataset")       
PROCESSED_DIR = Path("data/processed")    

RAW_CUBE_NAME = "raw"
WHITE_REF_NAME = "whiteReference"
DARK_REF_NAME = "darkReference"
GT_MAP_NAME = "gtMap"

REMOVE_BANDS_START = 56
REMOVE_BANDS_END = 126
SMOOTHING_ENABLED = True
SMOOTHING_WINDOW = 5 
ABSORBANCE_CONVERSION = False 
NORMALIZATION = "minmax"
DOWNSAMPLING_ENABLED = True
FINAL_CHANNELS = 128
STEP_NM = 3.61 


def save_preprocessed_cube(out_dir, cube, gt=None):
    """Save the preprocessed cube (and optionally GT) as NumPy .npy files."""
    ensure_dir(out_dir)
    np.save(Path(out_dir) / "preprocessed_cube.npy", cube)
    if gt is not None:
        np.save(Path(out_dir) / "gtMap.npy", gt)
    print(f"[INFO] Saved preprocessed cube → {out_dir}")

def main():
    logger = setup_logger("outputs/logs", name="preprocess")
    ensure_dir(PROCESSED_DIR)

    instances = list_all_instances(ROOT_DIR)
    logger.info(f"Found {len(instances)} dataset instances.")

    for instance in instances:
        logger.info(f"Processing {instance.name}...")
        data = load_dataset_instance(
            instance,
            raw_name=RAW_CUBE_NAME,
            white_ref_name=WHITE_REF_NAME,
            dark_ref_name=DARK_REF_NAME,
            gt_map_name=GT_MAP_NAME,
        )

        if data is None:
            logger.warning(f"Skipping {instance}")
            continue

        cube = preprocess_hsi_cube(
            data["raw"],
            data["white_ref"],
            data["dark_ref"],
            remove_bands=(REMOVE_BANDS_START, REMOVE_BANDS_END),
            smoothing_enabled=SMOOTHING_ENABLED,
            smoothing_window=SMOOTHING_WINDOW,
            absorbance_conversion=ABSORBANCE_CONVERSION,
            normalization=NORMALIZATION,
            downsampling_enabled=DOWNSAMPLING_ENABLED,
            final_channels=FINAL_CHANNELS,
        )

        out_dir = PROCESSED_DIR / instance.name
        save_preprocessed_cube(out_dir, cube, data["gt"])

        logger.info(f"Saved preprocessed cube to {out_dir}")

    logger.info("✅ Preprocessing completed for all instances.")

if __name__ == "__main__":
    main()

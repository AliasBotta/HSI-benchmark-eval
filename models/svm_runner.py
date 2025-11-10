# models/svm_runner_gpu.py
"""
SVMRunner (GPU-Accelerated)
---------
Implements the SVM classifier using RAPIDS cuML to run on NVIDIA GPUs.
This is intended for SPEED, not for 1:1 paper replication, although
it follows the same methodological steps (Grid Search, 'ovo').

It assumes all input data (X_train, X_val, cube) is ALREADY
pre-normalized to the [0, 1] range.

It performs a grid search for hyperparameters on the validation set
as required by the paper's methodology.
"""

import numpy as np
import cupy as cp
from cuml.svm import SVC
from cuml.metrics import f1_score
from . import BaseRunner # Assumes BaseRunner is in __init__.py

# --- Hyperparameter Grid ---
# Define the "coarse search" grid as per the paper's methodology
# You can expand these ranges if needed
PARAM_GRID_LINEAR = {
    'C': [0.01, 0.1, 1, 10, 100]
}
PARAM_GRID_RBF = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

# --- Metric Configuration ---
# The paper excludes BG class from Macro F1-Score
# Assuming 0=NT, 1=TT, 2=BV, 3=BG
# We must use cupy arrays for cuML metrics
METRIC_LABELS = cp.array([0, 1, 2])


class SVMRunner(BaseRunner):
    """
    GPU-Accelerated (cuML) SVM classifier runner.

    Implements:
        - fit(X_train, y_train, X_val, y_val)
        - predict_full(cube)
    """

    def __init__(self, kernel="linear", num_classes=4):
        self.name = f"svm-{kernel}-gpu"
        self.kernel = kernel
        self.num_classes = num_classes
        # The final, best-performing classifier
        self.clf = None

        if self.kernel not in ["linear", "rbf"]:
            raise ValueError(f"Unsupported kernel for SVMRunner: {kernel}")

    # ------------------------------------------------------------
    # Training (with Hyperparameter Optimization on GPU)
    # ------------------------------------------------------------
    def fit(self, X_train, y_train, X_val, y_val):
        """
        Fit the cuML SVM model.
        Performs a grid search on the validation set to find the best
        hyperparameters, as required by the paper.

        Assumes X_train, y_train, X_val, y_val are numpy arrays.
        """
        if X_train.size == 0:
            print(f"[{self.name}] ⚠ Empty training set, skipping training.")
            return

        if X_val is None or y_val is None:
            print(f"[{self.name}] ⚠ No validation set provided for hyperparameter tuning.")
            print(f"[{self.name}] Training with default parameters (C=1.0).")
            # Fallback: train with default C=1.0 if no val set
            param_grid = [{"C": 1.0, "gamma": "auto"}]
        else:
            print(f"[{self.name}] Starting hyperparameter search ({self.kernel} kernel)...")
            # Select the correct grid based on the kernel
            if self.kernel == "linear":
                param_grid = [{"C": c} for c in PARAM_GRID_LINEAR['C']]
            else: # rbf
                param_grid = [
                    {"C": c, "gamma": g}
                    for c in PARAM_GRID_RBF['C']
                    for g in PARAM_GRID_RBF['gamma']
                ]

        # --- Move data to GPU ---
        print(f"[{self.name}] Moving data to GPU...")
        try:
            X_train_gpu = cp.asarray(X_train)
            y_train_gpu = cp.asarray(y_train)
            X_val_gpu = cp.asarray(X_val) if X_val is not None else None
            y_val_gpu = cp.asarray(y_val) if y_val is not None else None
        except Exception as e:
            print(f"[{self.name}] ❌ ERROR: Failed to move data to GPU. Is cuML/cupy installed correctly?")
            print(e)
            return

        best_score = -1.0
        best_params = {}
        best_model = None

        # --- Grid Search Loop ---
        for params in param_grid:
            # Note: No pipeline, as data is already scaled
            model = SVC(
                kernel=self.kernel,
                C=params.get('C', 1.0),
                gamma=params.get('gamma', 'auto'),
                probability=True,
                multiclass_strategy="ovo", # 'ovo' is cuML's default, matches LIBSVM
                random_state=42,
                # Increase tolerance for faster convergence on large, noisy data
                tol=1e-3
            )

            # --- Train on the training set (on GPU) ---
            model.fit(X_train_gpu, y_train_gpu)

            # --- Evaluate on the validation set (on GPU) ---
            if X_val_gpu is not None:
                preds_val = model.predict(X_val_gpu)

                # Calculate Macro F1-Score *excluding BG class* using cuML
                score = f1_score(
                    y_val_gpu,
                    preds_val,
                    labels=METRIC_LABELS,
                    average="macro",
                    zero_division=0.0
                )

                # cupy .item() converts scalar to Python native
                score = score.item()

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
            else:
                # If no validation set, just use the first (or only) model
                best_model = model
                best_params = params
                break

        self.clf = best_model
        print(f"[{self.name}] ✅ Training complete.")
        if X_val is not None:
            print(f"    -> Best Score (Val Macro F1): {best_score:.4f}")
            print(f"    -> Best Params: {best_params}")


    # ------------------------------------------------------------
    # Prediction (on GPU)
    # ------------------------------------------------------------
    def predict_full(self, cube):
        """
        Predict pixel-wise classes and probabilities for a full HSI cube.
        Performs all computation on the GPU.

        Parameters
        ----------
        cube : np.ndarray
            (bands, H, W) preprocessed HSI cube (as numpy array).

        Returns
        -------
        class_map : np.ndarray
            (H, W) predicted labels (0..num_classes-1) (as numpy array)
        prob_all : np.ndarray
            (H, W, num_classes) probability scores (as numpy array)
        """
        if self.clf is None:
            raise RuntimeError(f"[{self.name}] ❌ Model is not trained. Call fit() first.")

        bands, H, W = cube.shape

        # Reshape and move input cube data to GPU
        # (bands, H*W) -> (H*W, bands)
        flat = cube.reshape(bands, -1).T
        try:
            flat_gpu = cp.asarray(flat)
        except Exception as e:
            print(f"[{self.name}] ❌ ERROR: Failed to move prediction data to GPU.")
            print(e)
            return None, None

        print(f"[{self.name}] Predicting on cube ({bands} bands, {H}×{W} spatial).")

        # --- Run prediction and probability calculation on GPU ---
        proba_gpu = self.clf.predict_proba(flat_gpu)  # (H*W, num_classes)
        class_map_gpu = cp.argmax(proba_gpu, axis=1)

        # --- Transfer results from GPU back to CPU (numpy) ---
        try:
            # .get() is an alternative to cp.asnumpy()
            class_map = class_map_gpu.get().reshape(H, W)
            prob_all = proba_gpu.get().reshape(H, W, -1)
        except Exception as e:
            print(f"[{self.name}] ❌ ERROR: Failed to move results from GPU to CPU.")
            print(e)
            return None, None

        print(f"[{self.name}] ✅ Prediction complete.")
        return class_map, prob_all

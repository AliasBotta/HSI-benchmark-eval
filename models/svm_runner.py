# models/svm_runner.py
"""
SVMRunner
---------
Implements a Support Vector Machine (SVM) classifier for hyperspectral
pixel-wise classification. Supports both linear and RBF kernels, with
probability outputs enabled for soft decisions.
"""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from . import BaseRunner


class SVMRunner(BaseRunner):
    """
    SVM classifier runner.

    Methods:
        - fit(X, y)
        - predict_full(cube)
    """

    def __init__(self,
                 kernel="linear",
                 C=1.0,
                 tolerance=1e-3,
                 max_iter=100000,
                 num_classes=4):
        self.name = f"svm-{kernel}"
        self.kernel = kernel
        self.C = C
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.num_classes = num_classes

        # Pipeline: standardization + SVM classifier
        self.clf = make_pipeline(
            StandardScaler(),
            SVC(
                kernel=kernel,
                C=C,
                gamma="scale",                      # 'scale' ~ KernelScale 'auto' in MATLAB
                probability=True,                   # required for soft outputs
                decision_function_shape="ovr",      # one-vs-rest strategy
                tol=tolerance,
                max_iter=max_iter,
                random_state=42
            )
        )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    def fit(self, X, y):
        """Fit the SVM model on spectral data."""
        if X.size == 0:
            print("[SVMRunner] ⚠ Empty training set, skipping training.")
            return

        print(f"[SVMRunner] Training ({self.kernel} kernel, C={self.C}, tol={self.tolerance}) "
              f"with {len(y)} samples.")
        self.clf.fit(X, y)

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------
    def predict_full(self, cube):
        """
        Predict pixel-wise classes and probabilities for a full HSI cube.

        Parameters
        ----------
        cube : np.ndarray
            (bands, H, W) preprocessed HSI cube.

        Returns
        -------
        class_map : np.ndarray
            (H, W) predicted labels (0..num_classes-1)
        prob_all : np.ndarray
            (H, W, num_classes) probability scores
        """
        bands, H, W = cube.shape
        flat = cube.reshape(bands, -1).T

        print(f"[SVMRunner] Predicting on cube ({bands} bands, {H}×{W} spatial).")
        proba = self.clf.predict_proba(flat)  # (H*W, num_classes)
        class_map = np.argmax(proba, axis=1).reshape(H, W)
        prob_all = proba.reshape(H, W, -1)

        print("[SVMRunner] ✅ Prediction complete.")
        return class_map, prob_all

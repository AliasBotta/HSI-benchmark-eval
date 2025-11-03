# models/rf_runner.py
"""
RFRunner
--------
Implements a Random Forest classifier for hyperspectral
pixel-wise classification. Provides probability outputs and
a class map compatible with the benchmark pipeline.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from . import BaseRunner


class RFRunner(BaseRunner):
    """
    Random Forest classifier runner.

    Methods:
        - fit(X, y)
        - predict_full(cube)
    """

    def __init__(self,
                 n_trees=200,
                 max_depth=None,
                 num_classes=4,
                 random_state=42):
        self.name = "rf"
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.num_classes = num_classes
        self.random_state = random_state
        self.clf = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    def fit(self, X, y):
        """Fit the Random Forest model on spectral data."""
        if X.size == 0:
            print("[RFRunner] ⚠ Empty training set, skipping training.")
            return

        print(f"[RFRunner] Training with {len(y)} samples, trees={self.n_trees}, depth={self.max_depth}.")
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
            (bands, H, W) preprocessed hyperspectral cube.

        Returns
        -------
        class_map : np.ndarray
            (H, W) predicted labels (0..num_classes-1)
        prob_all : np.ndarray
            (H, W, num_classes) probability scores
        """
        bands, H, W = cube.shape
        flat = cube.reshape(bands, -1).T

        print(f"[RFRunner] Predicting on cube ({bands} bands, {H}×{W} spatial).")
        proba = self.clf.predict_proba(flat)  # (H*W, num_classes)
        class_map = np.argmax(proba, axis=1).reshape(H, W)
        prob_all = proba.reshape(H, W, -1)

        print("[RFRunner] ✅ Prediction complete.")
        return class_map, prob_all

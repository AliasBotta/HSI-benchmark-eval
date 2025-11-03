# models/knn_runner.py
"""
KNNRunner
---------
Implements a pixel-wise K-Nearest Neighbors classifier for
hyperspectral image classification.

Supports probability outputs and can be used as a baseline
supervised model in the benchmark pipeline.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from . import BaseRunner


class KNNRunner(BaseRunner):
    """
    KNN classifier runner.

    Methods:
        - fit(X, y)
        - predict_full(cube)
    """

    def __init__(self,
                 n_neighbors=5,
                 metric="euclidean",
                 num_classes=4):
        self.name = f"knn-{ 'e' if metric == 'euclidean' else 'c' }"
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.num_classes = num_classes
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    def fit(self, X, y):
        """Fit the KNN model on spectral data."""
        if X.size == 0:
            print("[KNNRunner] ⚠ Empty training data, skipping training.")
            return
        print(f"[KNNRunner] Training with {len(y)} samples, K={self.n_neighbors}, metric={self.metric}.")
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

        print(f"[KNNRunner] Predicting on cube ({bands} bands, {H}×{W} spatial).")
        proba = self.clf.predict_proba(flat)  # (H*W, num_classes)
        class_map = np.argmax(proba, axis=1).reshape(H, W)
        prob_all = proba.reshape(H, W, -1)

        print("[KNNRunner] ✅ Prediction complete.")
        return class_map, prob_all

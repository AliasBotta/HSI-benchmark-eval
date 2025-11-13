"""
RFRunner (Paper-Compliant)
--------
Implements a Random Forest classifier for hyperspectral
pixel-wise classification.

[cite_start]This version is compliant with the benchmark paper[cite: 6].
It implements hyperparameter optimization in the .fit() method
[cite_start]to find the optimal number of trees (T)[cite: 925].
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from . import BaseRunner


class RFRunner(BaseRunner):
    """
    [cite_start]Random Forest classifier runner, compliant with[cite: 6].

    Implements hyperparameter optimization for n_estimators (T) using
    [cite_start]the validation set, as required by the paper's methodology[cite: 945, 949].
    """

    def __init__(self,
                 max_depth=None,
                 n_trees_search_space=[50, 100, 150, 200, 300],
                 random_state=42):
        """
        Initializes the RFRunner.

        Parameters
        ----------
        max_depth : int, optional
            Fixed max_depth for all trees. Default is None.
        n_trees_search_space : list of int
            [cite_start]List of T (n_estimators) values to test during optimization[cite: 925].
        random_state : int
            Random state for reproducibility.
        """
        self.name = "rf"
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_trees_search_space = n_trees_search_space

        self.metric_labels = [1, 2, 3]

        self.clf = None  
        self.best_T_ = 0 

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit the Random Forest model.

        If validation data (X_val, y_val) is provided, this method performs
        a hyperparameter search to find the optimal number of trees (T)
        [cite_start]by evaluating the Macro F1-Score on the validation set[cite: 925, 949].
        """
        if X_train.size == 0:
            print("[RFRunner] ⚠ Empty training set, skipping training.")
            return

        if X_val is None or y_val is None:
            print("[RFRunner] ⚠ No validation set provided. Skipping optimization.")
            print("[RFRunner] Training with default T=100.")
            self.best_T_ = 100
            self.clf = RandomForestClassifier(
                n_estimators=self.best_T_,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.clf.fit(X_train, y_train)
            return

        print(f"[RFRunner] Optimizing T (n_estimators) on validation set.")
        print(f"         Search space: {self.n_trees_search_space}")

        best_model = None
        best_score = -1.0

        for T in self.n_trees_search_space:
            model = RandomForestClassifier(
                n_estimators=T,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            y_pred_val = model.predict(X_val)

            score = f1_score(
                y_val,
                y_pred_val,
                labels=self.metric_labels,
                average="macro",
                zero_division=0.0
            )

            if score > best_score:
                best_score = score
                self.best_T_ = T
                best_model = model

        self.clf = best_model
        print(f"[RFRunner] ✅ Optimization complete. Best T={self.best_T_} (Val F1={best_score:.4f}).")

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
            (H, W) predicted labels
        prob_all : np.ndarray
            (H, W, num_classes) probability scores
        """
        if self.clf is None:
            raise RuntimeError("[RFRunner] ❌ Model not trained. Call .fit() first.")

        bands, H, W = cube.shape

        flat_pixels = cube.reshape(bands, -1).T

        print(f"[RFRunner] Predicting on cube ({bands} bands, {H}×{W} spatial).")

        proba = self.clf.predict_proba(flat_pixels)

        class_preds_flat = self.clf.classes_[np.argmax(proba, axis=1)]

        num_classes = proba.shape[1]
        class_map = class_preds_flat.reshape(H, W)
        prob_all = proba.reshape(H, W, num_classes)

        print("[RFRunner] ✅ Prediction complete.")
        return class_map, prob_all

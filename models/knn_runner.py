"""
KNNRunner (Paper Compliant)
---------
Implements KNN classifier with a grid search for hyperparameter
optimization on the validation set, as required by the paper.

The paper specifies 'N' (n_neighbors) is the hyperparameter
to be optimized.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from . import BaseRunner

PARAM_GRID_KNN = {
    'n_neighbors': [1, 3, 5, 7, 11, 15]
}

METRIC_LABELS = [0, 1, 2]


class KNNRunner(BaseRunner):
    """
    KNN classifier runner with hyperparameter optimization.
    """
    def __init__(self, metric="euclidean", num_classes=4):
        self.name = f"knn-{'e' if metric == 'euclidean' else 'c'}"
        self.metric = metric
        self.num_classes = num_classes
        self.clf = None  

    def fit(self, X_train, y_train, X_val, y_val): 
        """
        Fit the KNN model.
        Performs a grid search on the validation set to find the best
        n_neighbors, as required by the paper[cite: 927, 945].
        """
        if X_train.size == 0:
            print("[KNNRunner] ⚠ Empty training set, skipping training.")
            return

        print(f"[KNNRunner] Starting hyperparameter search (metric={self.metric})...")

        param_grid = [{"n_neighbors": n} for n in PARAM_GRID_KNN['n_neighbors']]

        best_score = -1.0
        best_params = {}
        best_model = None

        for params in param_grid:
            model = KNeighborsClassifier(
                n_neighbors=params['n_neighbors'],
                metric=self.metric,
                n_jobs=-1  
            )

            model.fit(X_train, y_train)

            if X_val is not None and y_val is not None:
                preds_val = model.predict(X_val)

                score = f1_score(
                    y_val,
                    preds_val,
                    labels=METRIC_LABELS,
                    average="macro",
                    zero_division=0.0
                )

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
            else:
                best_model = model
                best_params = params
                print("[KNNRunner] ⚠ No validation set provided. Using default params.")
                break

        self.clf = best_model
        print(f"[KNNRunner] ✅ Training complete.")
        print(f"  -> Best Score (Val Macro F1): {best_score:.4f}")
        print(f"  -> Best Params: {best_params}")

    def predict_full(self, cube):
        """
        Predict pixel-wise classes and probabilities for a full HSI cube.
        """
        if self.clf is None:
            raise RuntimeError("[KNNRunner] ❌ Model is not trained. Call fit() first.")

        bands, H, W = cube.shape
        flat = cube.reshape(bands, -1).T

        print(f"[KNNRunner] Predicting on cube ({bands} bands, {H}×{W} spatial).")
        proba = self.clf.predict_proba(flat)  
        class_map = np.argmax(proba, axis=1).reshape(H, W)
        prob_all = proba.reshape(H, W, -1)

        print("[KNNRunner] ✅ Prediction complete.")
        return class_map, prob_all

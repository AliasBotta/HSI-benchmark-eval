"""
SVMRunner (Paper Compliant)
---------
Implements SVM classifier with a grid search for hyperparameter
optimization on the validation set, as required by the paper.

Uses:
- MinMaxScaler
- 'ovo' multiclass strategy (to match LIBSVM default)
- Grid search to find C (for linear) and C/gamma (for RBF)
"""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler  
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from . import BaseRunner

PARAM_GRID_LINEAR = {
    'C': [0.01, 0.1, 1, 10, 100]
}
PARAM_GRID_RBF = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}
METRIC_LABELS = [0, 1, 2]


class SVMRunner(BaseRunner):
    """
    SVM classifier runner with hyperparameter optimization.
    """
    def __init__(self, kernel="linear", num_classes=4):
        self.name = f"svm-{kernel}"
        self.kernel = kernel
        self.num_classes = num_classes
        self.clf = None 

    def fit(self, X_train, y_train, X_val, y_val): 
        """
        Fit the SVM model.
        Performs a grid search on the validation set to find the best
        hyperparameters, as required by the paper.
        """
        if X_train.size == 0:
            print("[SVMRunner] ⚠ Empty training set, skipping training.")
            return

        print(f"[SVMRunner] Starting hyperparameter search ({self.kernel} kernel)...")

        if self.kernel == "linear":
            param_grid = [{"C": c} for c in PARAM_GRID_LINEAR['C']]
        else: 
            param_grid = [
                {"C": c, "gamma": g}
                for c in PARAM_GRID_RBF['C']
                for g in PARAM_GRID_RBF['gamma']
            ]

        best_score = -1.0
        best_params = {}
        best_model = None

        for params in param_grid:
            model = SVC(
                kernel=self.kernel,
                C=params.get('C', 1.0),
                gamma=params.get('gamma', 'auto'),
                probability=True,
                decision_function_shape="ovo",
                random_state=42
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
                print("[SVMRunner] ⚠ No validation set provided. Using default params.")
                break

        self.clf = best_model
        print(f"[SVMRunner] ✅ Training complete.")
        print(f"    -> Best Score (Val Macro F1): {best_score:.4f}")
        print(f"    -> Best Params: {best_params}")


    def predict_full(self, cube):
        """
        Predict pixel-wise classes and probabilities for a full HSI cube.
        """
        if self.clf is None:
            raise RuntimeError("[SVMRunner] ❌ Model is not trained. Call fit() first.")

        bands, H, W = cube.shape
        flat = cube.reshape(bands, -1).T

        print(f"[SVMRunner] Predicting on cube ({bands} bands, {H}×{W} spatial).")
        proba = self.clf.predict_proba(flat)  
        class_map = np.argmax(proba, axis=1).reshape(H, W)
        prob_all = proba.reshape(H, W, -1)

        print("[SVMRunner] ✅ Prediction complete.")
        return class_map, prob_all

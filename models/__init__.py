# models/__init__.py
"""
Model registry and base interface for HSI benchmark runners.

Each model implements a subclass of BaseRunner providing:
    - fit(X, y)
    - predict_full(cube) → (class_map, prob_all)
"""

import numpy as np
from abc import ABC, abstractmethod

# ============================================================
# Base Interface
# ============================================================

class BaseRunner(ABC):
    """
    Base interface for all supervised models.

    Each subclass must implement:
        - fit(X_train, y_train, X_val, y_val)
        - predict_full(cube) -> (class_map, prob_all)
    """
    name: str = "base"

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None): # <-- MODIFIED
        """
        Train model on flattened pixel-level spectra.
        X_val and y_val are optionally used for hyperparameter optimization.
        """
        pass

    @abstractmethod
    def predict_full(self, cube):
        """Predict class map and probability map for a full HSI cube."""
        pass


# ============================================================
# Factory / Model Switch
# ============================================================

def get_runner(model_name: str) -> BaseRunner:
    """
    Factory function returning the appropriate model runner
    given the model name.

    Parameters
    ----------
    model_name : str
        One of:
        ['dnn', 'svm-l', 'svm-rbf', 'knn-e', 'rf', 'ebeae', 'nebeae']

    Returns
    -------
    runner : BaseRunner
        Instantiated runner implementing fit() and predict_full().
    """
    name = str(model_name).lower()

    if name in {"dnn", "dnn_1d"}:
        from .dnn_runner import DNNRunner
        return DNNRunner()
    if name in {"svm-l", "svm_l", "svm-linear"}:
        from .svm_runner import SVMRunner
        return SVMRunner(kernel="linear")
    if name in {"svm-rbf", "svm_rbf"}:
        from .svm_runner import SVMRunner
        return SVMRunner(kernel="rbf")
    if name in {"knn-e", "knn_e"}:
        from .knn_runner import KNNRunner
        return KNNRunner(metric="euclidean")
    if name in {"knn-c", "knn_c"}:
        from .knn_runner import KNNRunner
        return KNNRunner(metric="cosine")
    if name in {"rf", "random_forest"}:
        from .rf_runner import RFRunner
        return RFRunner()
    if name in {"ebeae"}:
        from .unmixing_runner import UnmixingRunner
        return UnmixingRunner(name="ebeae")
    if name in {"nebeae"}:
        from .unmixing_runner import UnmixingRunner
        return UnmixingRunner(name="nebae")

    raise ValueError(f"[Model Registry] ❌ Unknown model name: {model_name}")

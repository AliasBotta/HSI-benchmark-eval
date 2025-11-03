"""
metrics.py
-----------
Implements evaluation metrics used in the HSI benchmark paper:
- F1_macro
- Overall Accuracy (OA)
- Sensitivity (per class)
- Specificity (per class)
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


# ============================================================
# Basic Metrics
# ============================================================

def overall_accuracy(y_true, y_pred):
    """Compute overall accuracy (OA)."""
    return np.mean(y_true == y_pred)


# ============================================================
# Full Benchmark Metrics
# ============================================================

def compute_all_metrics(y_true, y_pred, num_classes, labels=None):
    """
    Compute global and per-class metrics consistent with HSI-benchmark.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    num_classes : int
        Total number of classes (e.g., 4 for TT, NT, BV, BG).
    labels : list[int], optional
        Specific labels to include in the evaluation
        (e.g., [0,1,2] to exclude background class).

    Returns
    -------
    metrics : dict
        Dictionary containing macro F1, OA, per-class sensitivity/specificity,
        and their means.
    """
    metrics = {}
    if labels is None:
        labels = list(range(num_classes))
    labels = list(labels)

    # --- Macro F1 on the eval labels only ---
    metrics["f1_macro"] = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    metrics["oa"] = accuracy_score(y_true, y_pred)

    # --- Per-class metrics (mask-based, robust to invalid preds) ---
    sensitivity = []
    specificity = []

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for c in labels:
        tpos = np.sum((y_true == c) & (y_pred == c))
        fneg = np.sum((y_true == c) & (y_pred != c))
        fpos = np.sum((y_true != c) & (y_pred == c))
        tneg = np.sum((y_true != c) & (y_pred != c))

        sens = tpos / (tpos + fneg + 1e-8)
        spec = tneg / (tneg + fpos + 1e-8)

        metrics[f"sens_class_{c}"] = sens
        metrics[f"spec_class_{c}"] = spec
        sensitivity.append(sens)
        specificity.append(spec)

    metrics["sensitivity_mean"] = float(np.mean(sensitivity)) if sensitivity else 0.0
    metrics["specificity_mean"] = float(np.mean(specificity)) if specificity else 0.0
    return metrics


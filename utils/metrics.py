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
from sklearn.metrics import f1_score, recall_score, confusion_matrix


def overall_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def sensitivity_specificity(y_true, y_pred, num_classes):
    """
    Compute per-class sensitivity (TPR) and specificity (TNR)
    Returns dicts of arrays.
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    sens = np.zeros(num_classes)
    spec = np.zeros(num_classes)
    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)
        sens[i] = TP / (TP + FN + 1e-8)
        spec[i] = TN / (TN + FP + 1e-8)
    return sens, spec


def compute_all_metrics(y_true, y_pred, num_classes):
    """Compute all evaluation metrics and return as dict."""
    f1 = f1_score(y_true, y_pred, average="macro")
    oa = overall_accuracy(y_true, y_pred)
    sens, spec = sensitivity_specificity(y_true, y_pred, num_classes)
    return {
        "f1_macro": f1,
        "oa": oa,
        "sensitivity_mean": sens.mean(),
        "specificity_mean": spec.mean(),
    }


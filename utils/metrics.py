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
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix


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
    metrics = {}

    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    metrics["oa"] = accuracy_score(y_true, y_pred)

    # Sensitivity (recall) and specificity mean
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    sensitivity = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
    specificity = (np.sum(cm) - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))) / (np.sum(cm) - cm.sum(axis=1) + 1e-8)

    metrics["sensitivity_mean"] = np.mean(sensitivity)
    metrics["specificity_mean"] = np.mean(specificity)

    # 👉 aggiungi qui metriche per classe
    for i, (se, sp) in enumerate(zip(sensitivity, specificity)):
        metrics[f"sens_class_{i}"] = se
        metrics[f"spec_class_{i}"] = sp

    return metrics


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


def overall_accuracy(y_true, y_pred):
    """Compute overall accuracy (OA)."""
    return np.mean(y_true == y_pred)


def compute_all_metrics(y_true, y_pred, num_classes):
    """Compute global and per-class metrics consistent with HSI-benchmark."""
    metrics = {}

    # --- Macro F1 and OA ---
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    metrics["oa"] = accuracy_score(y_true, y_pred)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    # --- Per-class metrics ---
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)

    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)

        sensitivity[i] = TP / (TP + FN + 1e-8)  # True Positive Rate
        specificity[i] = TN / (TN + FP + 1e-8)  # True Negative Rate

        # Save per-class results
        metrics[f"sens_class_{i}"] = sensitivity[i]
        metrics[f"spec_class_{i}"] = specificity[i]

    # --- Mean values (global sensitivity/specificity) ---
    metrics["sensitivity_mean"] = np.mean(sensitivity)
    metrics["specificity_mean"] = np.mean(specificity)

    return metrics

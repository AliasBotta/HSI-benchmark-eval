""""
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


def compute_all_metrics(y_true, y_pred, num_classes, labels=None):
    """
    Compute global and per-class metrics consistent with HSI-benchmark.

    This implementation computes macro F1 manually over the provided 'labels'
    so that predictions outside 'labels' (e.g., -1 standing for BG errors)
    are treated as negatives for all evaluated classes (i.e., they count as errors),
    instead of being silently ignored by sklearn's label filtering.
    """
    metrics = {}
    if labels is None: 
        labels = list(range(num_classes))
    labels = list(labels)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics["oa"] = accuracy_score(y_true, y_pred)


    f1s = []
    sensitivity = []
    specificity = []


    for c in labels:
        y_t_pos = (y_true == c)
        y_p_pos = (y_pred == c)

        tp = np.sum(y_t_pos & y_p_pos)
        fn = np.sum(y_t_pos & (~y_p_pos))
        fp = np.sum((~y_t_pos) & y_p_pos)
        tn = np.sum((~y_t_pos) & (~y_p_pos))

        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)


        metrics[f"sens_class_{c}"] = rec
        metrics[f"spec_class_{c}"] = tn / (tn + fp + 1e-8)
        sensitivity.append(rec)
        specificity.append(metrics[f"spec_class_{c}"])
        f1s.append(f1)



    metrics["f1_macro"] = float(np.mean(f1s)) if f1s else 0.0
    metrics["sensitivity_mean"] = float(np.mean(sensitivity)) if sensitivity else 0.0
    metrics["specificity_mean"] = float(np.mean(specificity)) if specificity else 0.0
    return metrics

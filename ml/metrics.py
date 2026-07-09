# ml/metrics.py
import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # [[TN, FP], [FN, TP]]
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """AUC via méthode trapézoïdale, implémentation from scratch."""
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()

    for t in thresholds:
        pred = (y_scores >= t).astype(int)
        cm = confusion_matrix(y_true, pred)
        tprs.append(cm[1, 1] / pos if pos else 0.0)
        fprs.append(cm[0, 1] / neg if neg else 0.0)

    tprs.append(1.0)
    fprs.append(1.0)
    return float(np.trapz(tprs, fprs))

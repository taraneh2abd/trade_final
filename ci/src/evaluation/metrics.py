import numpy as np


def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float((y_true == y_pred).mean())


def precision(y_true, y_pred) -> float:
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall(y_true, y_pred) -> float:
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(y_true, y_pred) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    denom = p + r
    return float(2 * p * r / denom) if denom > 0 else 0.0

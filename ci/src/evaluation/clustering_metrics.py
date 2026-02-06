from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def _inertia(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Sum of squared distances to cluster centroids (like KMeans inertia).
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    inertia = 0.0
    for k in np.unique(labels):
        pts = X[labels == k]
        if len(pts) == 0:
            continue
        c = pts.mean(axis=0)
        inertia += float(((pts - c) ** 2).sum())
    return float(inertia)


def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Returns:
      - silhouette (higher better)
      - davies_bouldin (lower better)
      - calinski_harabasz (higher better)
      - inertia (lower better)
      - ARI/NMI if y_true available (higher better)
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)

    # if all labels same, many metrics break; handle gracefully
    if len(np.unique(labels)) < 2:
        out = {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
            "inertia": _inertia(X, labels),
        }
        if y_true is not None:
            y_true = np.asarray(y_true, dtype=int)
            out["ari"] = float("nan")
            out["nmi"] = float("nan")
        return out

    out = {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "inertia": _inertia(X, labels),
    }

    if y_true is not None:
        y_true = np.asarray(y_true, dtype=int)
        out["ari"] = float(adjusted_rand_score(y_true, labels))
        out["nmi"] = float(normalized_mutual_info_score(y_true, labels))

    return out

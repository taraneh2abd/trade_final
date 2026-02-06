# ci\src\problems\clustering.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseProblem, ProblemInfo


@dataclass
class ClusteringProblem(BaseProblem):
    """
    Clustering problem wrapper for Iris / Synthetic (make_blobs).
    Provides X and (optional) y_true for external metrics.
    """
    dataset: str = "iris"        # "iris" or "blobs"
    n_clusters: int = 3          # expected k
    seed: int = 42

    # blobs options
    n_samples: int = 500
    n_features: int = 2
    cluster_std: float = 1.0

    def __post_init__(self):
        self.dataset = str(self.dataset).lower().strip()
        if self.n_clusters <= 1:
            raise ValueError("n_clusters must be >= 2")

        if self.dataset == "iris":
            from sklearn.datasets import load_iris
            data = load_iris()
            self.X = data.data.astype(float)
            self.y_true = data.target.astype(int)
            # iris has 3 classes
            self.n_clusters = 3
            self.feature_names = list(getattr(data, "feature_names", [f"f{i}" for i in range(self.X.shape[1])]))
        elif self.dataset == "blobs":
            from sklearn.datasets import make_blobs
            X, y = make_blobs(
                n_samples=int(self.n_samples),
                n_features=int(self.n_features),
                centers=int(self.n_clusters),
                cluster_std=float(self.cluster_std),
                random_state=int(self.seed),
            )
            self.X = X.astype(float)
            self.y_true = y.astype(int)  # has ground-truth (synthetic)
            self.feature_names = [f"f{i}" for i in range(self.X.shape[1])]
        else:
            raise ValueError("dataset must be 'iris' or 'blobs'")

        self.n_samples_ = int(self.X.shape[0])
        self.n_features_ = int(self.X.shape[1])

        # standardize X (very common for clustering)
        mu = self.X.mean(axis=0)
        sigma = self.X.std(axis=0)
        sigma[sigma == 0] = 1.0
        self.X = (self.X - mu) / sigma

    def info(self) -> ProblemInfo:
        return ProblemInfo(
            name=f"clustering_{self.dataset}",
            problem_type="clustering",
            objective="maximize",  # we often maximize silhouette, etc.
            dimension=self.n_features_,
            extra={"dataset": self.dataset, "n_clusters": self.n_clusters, "n_samples": self.n_samples_},
        )

    def get_data(self) -> Dict[str, Any]:
        return {
            "X": self.X,
            "y_true": self.y_true,
            "n_clusters": self.n_clusters,
            "feature_names": self.feature_names,
        }

    def evaluate(self, solution: Any) -> float:
        """
        Not used directly; clustering methods output labels, then metrics are computed externally.
        """
        raise NotImplementedError("Use clustering methods + clustering_metrics to evaluate.")

    def get_llm_description(self) -> Dict[str, Any]:
        return {
            "problem_type": "clustering",
            "task": "unsupervised clustering",
            "dataset": self.dataset,
            "n_samples": self.n_samples_,
            "n_features": self.n_features_,
            "n_clusters": self.n_clusters,
            "notes": "Return cluster labels and clustering metrics (silhouette/DB/CH; ARI/NMI if y_true exists).",
        }

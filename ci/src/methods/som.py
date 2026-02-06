from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from .base import BaseMethod, ProgressCallback
from .result import MethodResult
from src.utils.seeding import set_global_seed
from src.evaluation.clustering_metrics import compute_clustering_metrics


class SOM(BaseMethod):
    name = "SOM"

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {
            "grid_h": 10,
            "grid_w": 10,
            "epochs": 50,
            "learning_rate": 0.5,
            "sigma": 2.0,              # neighborhood radius
            "decay": "linear",         # "linear" or "exp"
            "cluster_mode": "kmeans",  # "bmu" or "kmeans"
        }

    @classmethod
    def param_schema(cls) -> Dict[str, Any]:
        return {
            "grid_h": {"type": int, "min": 2, "max": 50},
            "grid_w": {"type": int, "min": 2, "max": 50},
            "epochs": {"type": int, "min": 5, "max": 2000},
            "learning_rate": {"type": float, "min": 1e-4, "max": 2.0},
            "sigma": {"type": float, "min": 0.1, "max": 50.0},
            "decay": {"type": str},
            "cluster_mode": {"type": str},
        }

    def _grid_coords(self, gh: int, gw: int) -> np.ndarray:
        coords = []
        for i in range(gh):
            for j in range(gw):
                coords.append([i, j])
        return np.asarray(coords, dtype=float)  # (m,2)

    def _decay(self, x0: float, t: int, T: int, mode: str) -> float:
        if mode == "exp":
            # exp decay to ~1% at end
            return float(x0 * np.exp(-5.0 * (t / max(T, 1))))
        # linear
        return float(x0 * (1.0 - (t / max(T, 1))) + 0.01 * x0 * (t / max(T, 1)))

    def solve(
        self,
        problem: Any,
        params: Dict[str, Any],
        progress_cb: Optional[ProgressCallback] = None,
        seed: Optional[int] = None,
    ) -> MethodResult:
        set_global_seed(seed)
        rng = np.random.default_rng(seed)

        data = problem.get_data()
        X = np.asarray(data["X"], dtype=float)
        y_true = data.get("y_true", None)
        k = int(data.get("n_clusters", 3))

        gh = int(params["grid_h"])
        gw = int(params["grid_w"])
        epochs = int(params["epochs"])
        lr0 = float(params["learning_rate"])
        sigma0 = float(params["sigma"])
        decay_mode = str(params["decay"]).lower().strip()
        cluster_mode = str(params["cluster_mode"]).lower().strip()

        n, d = X.shape
        m = gh * gw

        # init weights by sampling data
        W = X[rng.integers(0, n, size=m)].copy()  # (m,d)

        coords = self._grid_coords(gh, gw)  # (m,2)

        history = []

        # training (online)
        for ep in range(1, epochs + 1):
            lr = self._decay(lr0, ep - 1, epochs - 1, decay_mode)
            sigma = self._decay(sigma0, ep - 1, epochs - 1, decay_mode)
            sig2 = (sigma ** 2) + 1e-12

            idx = rng.permutation(n)
            for ii in idx:
                x = X[ii]  # (d,)
                # BMU
                dist2 = np.sum((W - x) ** 2, axis=1)  # (m,)
                bmu = int(np.argmin(dist2))

                # neighborhood influence
                dc2 = np.sum((coords - coords[bmu]) ** 2, axis=1)  # (m,)
                h = np.exp(-dc2 / (2.0 * sig2))  # (m,)
                # update
                W += (lr * h)[:, None] * (x[None, :] - W)

            # quick monitoring: compute labels on ep end and silhouette
            labels_ep = self._labels_from_weights(X, W, gh, gw, k, cluster_mode)
            met = compute_clustering_metrics(X, labels_ep, y_true=y_true)
            history.append(met.get("silhouette", float("nan")))

            if progress_cb and (ep == 1 or ep % 10 == 0):
                progress_cb(ep, history[-1], {"silhouette": history[-1]})

        labels = self._labels_from_weights(X, W, gh, gw, k, cluster_mode)
        metrics = compute_clustering_metrics(X, labels, y_true=y_true)

        # best_fitness: silhouette (bigger better)
        best_fit = float(metrics.get("silhouette", float("nan")))

        return MethodResult(
            method_name=self.name,
            best_solution={"labels": labels.tolist()},
            best_fitness=best_fit,
            history=history,
            iterations=epochs,
            status="ok",
            metrics=metrics,
        )

    def _labels_from_weights(
        self,
        X: np.ndarray,
        W: np.ndarray,
        gh: int,
        gw: int,
        k: int,
        cluster_mode: str,
    ) -> np.ndarray:
        """
        cluster_mode:
          - "bmu": label = bmu index (0..gh*gw-1)  [many clusters]
          - "kmeans": kmeans on neuron weights -> map bmu to cluster id (0..k-1)
        """
        # BMU for each sample
        dist2 = ((X[:, None, :] - W[None, :, :]) ** 2).sum(axis=2)  # (n,m)
        bmu = np.argmin(dist2, axis=1).astype(int)

        if cluster_mode == "bmu":
            return bmu

        # kmeans on weights
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=int(k), n_init=10, random_state=0)
        neuron_cluster = km.fit_predict(W)  # (m,)
        labels = neuron_cluster[bmu]
        return labels.astype(int)

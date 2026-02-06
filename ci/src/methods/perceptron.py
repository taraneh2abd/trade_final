from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from .base import BaseMethod, ProgressCallback
from .result import MethodResult
from src.utils.seeding import set_global_seed
from src.evaluation.metrics import accuracy, precision, recall, f1_score


class Perceptron(BaseMethod):
    name = "Perceptron"

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {
            "learning_rate": 0.01,
            "epochs": 200,
            "l2": 0.0,
            "threshold": 0.5,
        }

    @classmethod
    def param_schema(cls) -> Dict[str, Any]:
        return {
            "learning_rate": {"type": float, "min": 1e-5, "max": 1.0},
            "epochs": {"type": int, "min": 10, "max": 5000},
            "l2": {"type": float, "min": 0.0, "max": 10.0},
            "threshold": {"type": float, "min": 0.1, "max": 0.9},
        }

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
        Xtr, ytr = data["X_train"], data["y_train"]
        Xva, yva = data["X_val"], data["y_val"]
        Xte, yte = data["X_test"], data["y_test"]

        lr = float(params["learning_rate"])
        epochs = int(params["epochs"])
        l2 = float(params["l2"])
        thr = float(params["threshold"])

        n, d = Xtr.shape
        w = rng.normal(0, 0.01, size=(d,))
        b = 0.0

        history = []
        best_f1 = -1.0
        best_state = (w.copy(), b)

        for ep in range(1, epochs + 1):
            # shuffle
            idx = rng.permutation(n)
            X = Xtr[idx]
            y = ytr[idx]

            # Perceptron update (online)
            for i in range(n):
                xi = X[i]
                yi = int(y[i])
                score = float(np.dot(w, xi) + b)
                pred = 1 if score >= 0 else 0
                err = yi - pred

                if err != 0:
                    w = w + lr * err * xi
                    b = b + lr * err

                if l2 > 0:
                    w = w * (1.0 - lr * l2)

            # evaluate on val
            val_scores = Xva @ w + b
            val_pred = (val_scores >= 0).astype(int)  # perceptron threshold at 0
            f1v = f1_score(yva, val_pred)
            history.append(f1v)

            if f1v > best_f1:
                best_f1 = f1v
                best_state = (w.copy(), b)

            if progress_cb and (ep == 1 or ep % 20 == 0):
                progress_cb(ep, best_f1, {"val_f1": f1v})

        # load best
        w, b = best_state

        # test metrics
        test_scores = Xte @ w + b
        y_pred = (test_scores >= 0).astype(int)

        metrics = {
            "test_accuracy": accuracy(yte, y_pred),
            "test_precision": precision(yte, y_pred),
            "test_recall": recall(yte, y_pred),
            "test_f1": f1_score(yte, y_pred),
            "val_best_f1": float(best_f1),
        }

        # best_fitness: برای classification بهتره "F1" رو maximize کنیم
        return MethodResult(
            method_name=self.name,
            best_solution={"w": w.tolist(), "b": float(b)},
            best_fitness=float(metrics["test_f1"]),
            history=history,
            iterations=epochs,
            status="ok",
            metrics=metrics,
        )

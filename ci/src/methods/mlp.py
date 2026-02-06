from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from .base import BaseMethod, ProgressCallback
from .result import MethodResult
from src.utils.seeding import set_global_seed
from src.evaluation.metrics import accuracy, precision, recall, f1_score


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(0.0, x)


def relu_grad(x):
    return (x > 0).astype(float)


class MLP(BaseMethod):
    name = "MLP"

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {
            "hidden_size": 16,
            "learning_rate": 0.01,
            "epochs": 300,
            "batch_size": 64,
            "l2": 0.0,
            "patience": 40,
            "threshold": 0.5,
        }

    @classmethod
    def param_schema(cls) -> Dict[str, Any]:
        return {
            "hidden_size": {"type": int, "min": 4, "max": 256},
            "learning_rate": {"type": float, "min": 1e-5, "max": 1.0},
            "epochs": {"type": int, "min": 10, "max": 5000},
            "batch_size": {"type": int, "min": 8, "max": 1024},
            "l2": {"type": float, "min": 0.0, "max": 10.0},
            "patience": {"type": int, "min": 5, "max": 500},
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
        Xtr, ytr = data["X_train"], data["y_train"].astype(float)
        Xva, yva = data["X_val"], data["y_val"].astype(float)
        Xte, yte = data["X_test"], data["y_test"].astype(int)

        n, d = Xtr.shape

        h = int(params["hidden_size"])
        lr = float(params["learning_rate"])
        epochs = int(params["epochs"])
        bs = int(params["batch_size"])
        l2 = float(params["l2"])
        patience = int(params["patience"])
        thr = float(params["threshold"])

        # Xavier init
        W1 = rng.normal(0, 1.0 / np.sqrt(d), size=(d, h))
        b1 = np.zeros((h,))
        W2 = rng.normal(0, 1.0 / np.sqrt(h), size=(h, 1))
        b2 = np.zeros((1,))

        def forward(X):
            z1 = X @ W1 + b1
            a1 = relu(z1)
            z2 = a1 @ W2 + b2
            p = sigmoid(z2).reshape(-1)
            return z1, a1, p

        def bce(y, p):
            p = np.clip(p, 1e-8, 1 - 1e-8)
            return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

        best_val_f1 = -1.0
        best_state = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        wait = 0
        history = []

        for ep in range(1, epochs + 1):
            idx = rng.permutation(n)
            X = Xtr[idx]
            y = ytr[idx]

            for start in range(0, n, bs):
                xb = X[start:start + bs]
                yb = y[start:start + bs]

                z1, a1, p = forward(xb)

                # gradients (BCE + sigmoid)
                # dL/dz2 = p - y
                dz2 = (p - yb).reshape(-1, 1) / len(yb)
                dW2 = a1.T @ dz2 + l2 * W2
                db2g = dz2.sum(axis=0)

                da1 = dz2 @ W2.T
                dz1 = da1 * relu_grad(z1)
                dW1 = xb.T @ dz1 + l2 * W1
                db1g = dz1.sum(axis=0)

                W2 -= lr * dW2
                b2 -= lr * db2g
                W1 -= lr * dW1
                b1 -= lr * db1g

            # val
            _, _, pva = forward(Xva)
            yva_pred = (pva >= thr).astype(int)
            f1v = f1_score(yva.astype(int), yva_pred)
            history.append(f1v)

            if f1v > best_val_f1:
                best_val_f1 = f1v
                best_state = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
                wait = 0
            else:
                wait += 1

            if progress_cb and (ep == 1 or ep % 25 == 0):
                progress_cb(ep, best_val_f1, {"val_f1": f1v})

            if wait >= patience:
                break

        W1, b1, W2, b2 = best_state

        # test
        z1 = Xte @ W1 + b1
        a1 = relu(z1)
        pte = sigmoid(a1 @ W2 + b2).reshape(-1)
        y_pred = (pte >= thr).astype(int)

        metrics = {
            "test_accuracy": accuracy(yte, y_pred),
            "test_precision": precision(yte, y_pred),
            "test_recall": recall(yte, y_pred),
            "test_f1": f1_score(yte, y_pred),
            "val_best_f1": float(best_val_f1),
            "stopped_epoch": int(len(history)),
        }

        return MethodResult(
            method_name=self.name,
            best_solution={
                "W1": W1.tolist(),
                "b1": b1.tolist(),
                "W2": W2.reshape(-1).tolist(),
                "b2": float(b2.reshape(-1)[0]),
            },
            best_fitness=float(metrics["test_f1"]),
            history=history,
            iterations=int(len(history)),
            status="ok",
            metrics=metrics,
        )

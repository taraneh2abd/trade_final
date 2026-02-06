# ci\src\problems\titanic.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseProblem, ProblemInfo


@dataclass
class TitanicProblem(BaseProblem):
    csv_path: str = "ci/data/train.csv"
    split: Tuple[float, float, float] = (0.70, 0.15, 0.15)
    seed: int = 42

    def __post_init__(self):
        a, b, c = self.split
        if not np.isclose(a + b + c, 1.0):
            raise ValueError("split must sum to 1.0, e.g. (0.70,0.15,0.15)")

        df = pd.read_csv(self.csv_path)

        required = {"Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        # ====== Target ======
        y = df["Survived"].astype(int).to_numpy()

        # ====== Features selection (safe + standard) ======
        X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].copy()

        # ====== Missing handling ======
        # Age: fill median
        X["Age"] = X["Age"].fillna(X["Age"].median())

        # Embarked: fill mode
        X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode(dropna=True)[0])

        # ====== Encoding categorical ======
        # Sex: male/female -> 0/1
        X["Sex"] = X["Sex"].map({"male": 0, "female": 1}).astype(int)

        # Embarked: one-hot (C/Q/S)
        embarked_oh = pd.get_dummies(X["Embarked"], prefix="Embarked")
        X = X.drop(columns=["Embarked"])
        X = pd.concat([X, embarked_oh], axis=1)

        # Ensure fixed columns (in case any category missing)
        for col in ["Embarked_C", "Embarked_Q", "Embarked_S"]:
            if col not in X.columns:
                X[col] = 0

        # ====== Convert to numpy ======
        X_np = X.to_numpy(dtype=float)

        # ====== Split indices ======
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(X_np))
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(self.split[0] * n)
        n_val = int(self.split[1] * n)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        X_train_raw, y_train = X_np[train_idx], y[train_idx]
        X_val_raw, y_val = X_np[val_idx], y[val_idx]
        X_test_raw, y_test = X_np[test_idx], y[test_idx]

        # ====== Normalization (fit on train only) ======
        mu = X_train_raw.mean(axis=0)
        sigma = X_train_raw.std(axis=0)
        sigma[sigma == 0] = 1.0

        self.X_train = (X_train_raw - mu) / sigma
        self.y_train = y_train
        self.X_val = (X_val_raw - mu) / sigma
        self.y_val = y_val
        self.X_test = (X_test_raw - mu) / sigma
        self.y_test = y_test

        self.feature_names = list(X.columns)
        self.n_features = self.X_train.shape[1]

    def info(self) -> ProblemInfo:
        return ProblemInfo(
            name="titanic",
            problem_type="classification",
            objective="maximize",
            dimension=self.n_features,
            extra={"n_features": self.n_features, "split": list(self.split)},
        )

    def get_data(self) -> Dict[str, Any]:
        return {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_val": self.X_val,
            "y_val": self.y_val,
            "X_test": self.X_test,
            "y_test": self.y_test,
            "feature_names": self.feature_names,
        }

    def evaluate(self, solution: Any) -> float:
        """
        این evaluate برای Titanic مستقیم استفاده نمی‌شه مثل مسائل opt/TSP.
        ما ارزیابی را داخل روش‌های Perceptron/MLP انجام می‌دهیم.
        """
        raise NotImplementedError("Use Perceptron/MLP methods to train and evaluate on Titanic.")

    def get_llm_description(self) -> Dict[str, Any]:
        return {
            "problem_type": "classification",
            "task": "Titanic survival prediction",
            "target": "Survived (0/1)",
            "features": self.feature_names,
            "preprocessing": [
                "Age missing -> median",
                "Embarked missing -> mode",
                "Sex -> binary encoding",
                "Embarked -> one-hot",
                "Standardization using train mean/std",
                "Split train/val/test = 70/15/15",
            ],
            "metrics": ["accuracy", "precision", "recall", "f1"],
        }

# ci\src\problems\tsp.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseProblem, ProblemInfo


@dataclass
class TSPProblem(BaseProblem):
    """
    TSP with Euclidean distances.
    Representation: permutation of [0..n-1]
    Objective: minimize total tour length (round trip).
    """
    coords: np.ndarray  # shape (n, 2)
    name_: str = "tsp"

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("coords must be shape (n, 2)")
        self.n = int(self.coords.shape[0])
        if self.n < 3:
            raise ValueError("TSP needs at least 3 cities")

        # distance matrix
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self.D = np.sqrt(np.sum(diff ** 2, axis=2))

    @staticmethod
    def random(n_cities: int = 30, seed: Optional[int] = None, scale: float = 100.0) -> "TSPProblem":
        rng = np.random.default_rng(seed)
        coords = rng.random((n_cities, 2)) * scale
        return TSPProblem(coords=coords, name_=f"tsp_{n_cities}")

    def info(self) -> ProblemInfo:
        return ProblemInfo(
            name=self.name_,
            problem_type="tsp",  # ✅ اصلاح شد (قبلاً optimization بود)
            objective="minimize",
            dimension=self.n,
            extra={"representation": "permutation", "n_cities": self.n},
        )

    def evaluate(self, solution: Any) -> float:
        tour = np.asarray(solution, dtype=int).reshape(-1)
        if tour.size != self.n:
            raise ValueError(f"Tour length mismatch: expected {self.n}, got {tour.size}")

        # check permutation validity
        if set(tour.tolist()) != set(range(self.n)):
            raise ValueError("Tour must be a permutation of 0..n-1")

        total = 0.0
        for i in range(self.n - 1):
            total += self.D[tour[i], tour[i + 1]]
        total += self.D[tour[-1], tour[0]]  # return
        return float(total)

    def get_llm_description(self) -> Dict[str, Any]:
        return {
            "problem_type": "tsp",  # ✅ اصلاح شد (قبلاً optimization بود)
            "task": "traveling salesman problem (TSP)",
            "objective": "minimize",
            "representation": "permutation",
            "n_cities": self.n,
            "notes": "Solution is a permutation of cities [0..n-1]. Fitness is total tour length (round trip).",
        }

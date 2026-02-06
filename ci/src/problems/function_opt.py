# ci\src\problems\function_opt.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .base import BaseProblem, ProblemInfo


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    n = x.size
    return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    n = x.size
    a, b, c = 20.0, 0.2, 2 * np.pi
    s1 = np.sum(x ** 2)
    s2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(s1 / n))
    term2 = -np.exp(s2 / n)
    return float(term1 + term2 + a + np.e)


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


FUNC_MAP = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "rosenbrock": rosenbrock,
}

DEFAULT_BOUNDS = {
    "sphere": (-5.12, 5.12),
    "rastrigin": (-5.12, 5.12),
    "ackley": (-32.768, 32.768),
    "rosenbrock": (-5.0, 10.0),
}


@dataclass
class FunctionOptimizationProblem(BaseProblem):
    func_name: str = "rastrigin"
    dim: int = 10

    def __post_init__(self):
        if self.func_name not in FUNC_MAP:
            raise ValueError(f"Unknown function: {self.func_name}. Choose from {list(FUNC_MAP.keys())}")
        if self.dim <= 0:
            raise ValueError("dim must be positive")

        self.func = FUNC_MAP[self.func_name]
        self.lower, self.upper = DEFAULT_BOUNDS[self.func_name]

    def info(self) -> ProblemInfo:
        return ProblemInfo(
            name=f"{self.func_name}_{self.dim}d",
            problem_type="optimization",
            objective="minimize",
            dimension=self.dim,
            extra={"bounds": [self.lower, self.upper], "representation": "real"},
        )

    def sample_random(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower, self.upper, size=(self.dim,))

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lower, self.upper)

    def evaluate(self, solution: Any) -> float:
        x = np.asarray(solution, dtype=float).reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"Solution dim mismatch: expected {self.dim}, got {x.size}")
        x = self.clip(x)
        return self.func(x)

    def get_llm_description(self) -> Dict[str, Any]:
        return {
            "problem_type": "optimization",
            "task": "continuous function optimization",
            "objective": "minimize",
            "representation": "real",
            "function": self.func_name,
            "dimension": self.dim,
            "bounds": {"lower": self.lower, "upper": self.upper},
            "known_optimum": 0.0,
            "notes": "Search in continuous space; return best fitness and convergence history.",
        }

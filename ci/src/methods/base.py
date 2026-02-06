from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from .result import MethodResult
from src.utils.logging import get_logger
from src.utils.seeding import set_global_seed

ProgressCallback = Callable[[int, float, Dict[str, Any]], None]


class BaseMethod(ABC):
    """
    قرارداد مشترک برای همه روش‌ها.
    طبق پروژه باید:
    - API استاندارد (run/solve)
    - Validate پارامترها
    - Logging
    - Progress callback
    """

    name: str = "BaseMethod"

    def __init__(self, logger_name: str = "project_ci"):
        self.logger = get_logger(logger_name)

    @classmethod
    @abstractmethod
    def default_params(cls) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def param_schema(cls) -> Dict[str, Any]:
        """
        schema نمونه:
        {
          "max_iterations": {"type": int, "min": 1, "max": 100000},
          "learning_rate": {"type": float, "min": 1e-6, "max": 1.0},
          "activation": {"type": str, "choices": ["relu", "sigmoid", "tanh"]},
        }
        """
        raise NotImplementedError

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        schema = cls.param_schema()
        out = dict(cls.default_params())
        out.update(params or {})

        for k, rules in schema.items():
            if k not in out:
                raise ValueError(f"Missing parameter: {k}")

            v = out[k]
            t = rules.get("type")

            if t is not None and not isinstance(v, t):
                raise TypeError(f"Param '{k}' must be {t.__name__}, got {type(v).__name__}")

            if "min" in rules and v < rules["min"]:
                raise ValueError(f"Param '{k}' must be >= {rules['min']}, got {v}")

            if "max" in rules and v > rules["max"]:
                raise ValueError(f"Param '{k}' must be <= {rules['max']}, got {v}")

            if "choices" in rules and v not in rules["choices"]:
                raise ValueError(f"Param '{k}' must be one of {rules['choices']}, got {v}")

        # اجازه بده پارامتر اضافه هم باشد (برای انعطاف)، اما schema را حتما چک کردیم
        return out

    def run(
        self,
        problem: Any,
        params: Optional[Dict[str, Any]] = None,
        progress_cb: Optional[ProgressCallback] = None,
        seed: Optional[int] = None,
    ) -> MethodResult:
        # seed برای تکرارپذیری
        set_global_seed(seed)

        # validate
        validated = self.validate_params(params or {})

        self.logger.info(f"START {self.name} | params={validated}")

        t0 = time.time()
        try:
            res = self.solve(problem, validated, progress_cb=progress_cb, seed=seed)
            res.status = res.status or "ok"
        except Exception as e:
            self.logger.exception(f"FAILED {self.name} | error={e}")
            return MethodResult(
                method_name=self.name,
                best_solution=None,
                best_fitness=float("inf"),
                history=[],
                metrics={},
                time_sec=time.time() - t0,
                iterations=0,
                status="failed",
                params_used=validated,
                message=str(e),
            )

        res.time_sec = time.time() - t0
        res.params_used = validated
        res.method_name = self.name

        # اگر iterations را خود solve نداد، از history حدس بزن
        if res.iterations == 0 and res.history:
            res.iterations = len(res.history)

        self.logger.info(f"END {self.name} | best={res.best_fitness} | iters={res.iterations} | time={res.time_sec:.3f}s")
        return res

    @abstractmethod
    def solve(
        self,
        problem: Any,
        params: Dict[str, Any],
        progress_cb: Optional[ProgressCallback] = None,
        seed: Optional[int] = None,
    ) -> MethodResult:
        raise NotImplementedError

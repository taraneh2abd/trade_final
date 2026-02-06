# ci\src\problems\base.py   

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ProblemInfo:
    name: str
    problem_type: str  # "optimization" | "classification" | "clustering"
    objective: str     # "minimize" | "maximize"
    dimension: Optional[int] = None
    extra: Dict[str, Any] = None


class BaseProblem(ABC):
    @abstractmethod
    def info(self) -> ProblemInfo:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, solution: Any) -> float:
        """
        برای optimization: یک عدد fitness برگردون
        (کمتر بهتر اگر objective=minimize)
        """
        raise NotImplementedError

    @abstractmethod
    def get_llm_description(self) -> Dict[str, Any]:
        """
        خلاصه‌ی استاندارد برای LLM/Orchestrator
        """
        raise NotImplementedError

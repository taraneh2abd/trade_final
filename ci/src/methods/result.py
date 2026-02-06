from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MethodResult:
    method_name: str
    best_solution: Any
    best_fitness: float

    # برای نمودار همگرایی (بهترین مقدار در هر iteration/epoch)
    history: List[float] = field(default_factory=list)

    # متریک‌های اضافی (accuracy, gap, silhouette, ...)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # اطلاعات اجرایی
    time_sec: float = 0.0
    iterations: int = 0
    status: str = "ok"  # ok / early_stop / failed
    params_used: Dict[str, Any] = field(default_factory=dict)

    # پیام اختیاری برای دیباگ
    message: Optional[str] = None

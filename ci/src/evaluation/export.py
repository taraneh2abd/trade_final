# ci\src\evaluation\export.py

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.methods.result import MethodResult


def save_result_json(path: str, result: MethodResult, extra: Dict[str, Any] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "method_name": result.method_name,
        "best_fitness": float(result.best_fitness),
        "best_solution": np.asarray(result.best_solution).tolist() if result.best_solution is not None else None,
        "history": [float(x) for x in result.history],
        "metrics": result.metrics,
        "time_sec": float(result.time_sec),
        "iterations": int(result.iterations),
        "status": result.status,
        "params_used": result.params_used,
        "message": result.message,
    }
    if extra:
        payload["extra"] = extra

    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

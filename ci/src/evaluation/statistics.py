# ci\src\evaluation\statistics.py
from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
from scipy.stats import wilcoxon


def wilcoxon_compare_csv(
    csv_path: str,
    metric: str,
    method_a: str,
    method_b: str,
    alternative: str = "less",   # برای minimize معمولاً "less" یعنی A < B
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test روی نتایج paired (همان seedها) از یک CSV بنچمارک.

    CSV باید ستون‌های زیر را داشته باشد:
      - method
      - seed
      - <metric>  (مثلاً best_fitness)

    alternative:
      - "less":     method_a < method_b
      - "greater":  method_a > method_b
      - "two-sided": تفاوت دوطرفه
    """
    df = pd.read_csv(csv_path)

    required = {"method", "seed", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    a = df[df["method"] == method_a].sort_values("seed")
    b = df[df["method"] == method_b].sort_values("seed")

    if len(a) == 0 or len(b) == 0:
        raise ValueError(f"Methods not found in CSV. Have: {df['method'].unique().tolist()}")

    if len(a) != len(b):
        raise ValueError(f"Sample size mismatch: {method_a}={len(a)} vs {method_b}={len(b)}")

    # ensure same seeds
    if not (a["seed"].to_numpy() == b["seed"].to_numpy()).all():
        raise ValueError("Seeds are not aligned between methods. Ensure paired runs by same seeds.")

    da = a[metric].to_numpy(dtype=float)
    db = b[metric].to_numpy(dtype=float)

    # Wilcoxon expects paired differences; handle zeros robustly
    # zero_method="zsplit" is safer when many ties/zeros exist.
    stat, p = wilcoxon(da, db, alternative=alternative, zero_method="zsplit")

    return {
        "csv_path": csv_path,
        "metric": metric,
        "method_a": method_a,
        "method_b": method_b,
        "alternative": alternative,
        "n": int(len(da)),
        "a_values": da.tolist(),
        "b_values": db.tolist(),
        "statistic": float(stat),
        "p_value": float(p),
    }

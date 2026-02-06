from __future__ import annotations
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseMethod, ProgressCallback
from .result import MethodResult
from src.utils.seeding import set_global_seed


class PSO(BaseMethod):
    name = "PSO"

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        # مطابق PDF
        return {
            "n_particles": 50,          # 20-200
            "max_iterations": 500,      # 100-2000
            "w": 0.7,                   # 0.4-0.9
            "c1": 1.5,                  # 1.0-2.5
            "c2": 1.5,                  # 1.0-2.5
            "w_decay": True,            # linearly decrease w
            "velocity_clamp": 0.5,      # 0.1-1.0 (fraction of search range)
        }

    @classmethod
    def param_schema(cls) -> Dict[str, Any]:
        return {
            "n_particles": {"type": int, "min": 20, "max": 200},
            "max_iterations": {"type": int, "min": 100, "max": 2000},
            "w": {"type": float, "min": 0.4, "max": 0.9},
            "c1": {"type": float, "min": 1.0, "max": 2.5},
            "c2": {"type": float, "min": 1.0, "max": 2.5},
            "w_decay": {"type": bool},
            "velocity_clamp": {"type": float, "min": 0.1, "max": 1.0},
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

        # ---- dimension & bounds
        rep = "real"
        if callable(getattr(problem, "info", None)):
            pinfo = problem.info()
            dim = int(pinfo.dimension) if pinfo.dimension is not None else None
            bounds = pinfo.extra.get("bounds") if pinfo.extra else None
            rep = (pinfo.extra.get("representation") if pinfo.extra else "real") or "real"
        else:
            dim = getattr(problem, "dim", None)
            bounds = None

        if dim is None:
            raise ValueError("Problem must provide dimension (dim or info().dimension).")

        if rep != "real":
            raise ValueError("This PSO implementation is for real-coded (continuous) problems.")

        if bounds is None:
            lower = getattr(problem, "lower", None)
            upper = getattr(problem, "upper", None)
            if lower is None or upper is None:
                raise ValueError("Problem must provide bounds (lower/upper or info().extra['bounds']).")
            lower, upper = float(lower), float(upper)
        else:
            lower, upper = float(bounds[0]), float(bounds[1])

        n = params["n_particles"]
        iters = params["max_iterations"]
        w0 = params["w"]
        c1 = params["c1"]
        c2 = params["c2"]
        w_decay = params["w_decay"]
        v_clamp_frac = params["velocity_clamp"]

        # velocity clamp (fraction of range)
        vmax = v_clamp_frac * (upper - lower)

        # init
        X = rng.uniform(lower, upper, size=(n, dim))
        V = rng.uniform(-vmax, vmax, size=(n, dim))

        pbest_X = X.copy()
        pbest_F = np.array([problem.evaluate(x) for x in X], dtype=float)

        g_idx = int(np.argmin(pbest_F))
        gbest_X = pbest_X[g_idx].copy()
        gbest_F = float(pbest_F[g_idx])

        history = [gbest_F]

        # decay target (ثابت و ساده)
        w_end = 0.4  # مطابق range حداقل PDF

        for t in range(1, iters + 1):
            if w_decay:
                # linear decay from w0 to w_end
                w = w0 + (w_end - w0) * (t / iters)
            else:
                w = w0

            r1 = rng.random((n, dim))
            r2 = rng.random((n, dim))

            V = w * V + c1 * r1 * (pbest_X - X) + c2 * r2 * (gbest_X - X)
            V = np.clip(V, -vmax, vmax)

            X = X + V
            X = np.clip(X, lower, upper)

            F = np.array([problem.evaluate(x) for x in X], dtype=float)

            improved = F < pbest_F
            if np.any(improved):
                pbest_F[improved] = F[improved]
                pbest_X[improved] = X[improved]

            g_idx = int(np.argmin(pbest_F))
            g_new = float(pbest_F[g_idx])
            if g_new < gbest_F:
                gbest_F = g_new
                gbest_X = pbest_X[g_idx].copy()

            history.append(gbest_F)

            if progress_cb:
                progress_cb(t, gbest_F, {"w": w, "gbest": gbest_F})

        return MethodResult(
            method_name=self.name,
            best_solution=gbest_X,
            best_fitness=gbest_F,
            history=history,
            iterations=iters,
            status="ok",
            metrics={},
        )

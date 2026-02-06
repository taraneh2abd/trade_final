# ci\src\methods\ga.py

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from .base import BaseMethod, ProgressCallback
from .result import MethodResult
from src.utils.seeding import set_global_seed


class GA(BaseMethod):
    name = "GA"

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        # مطابق PDF
        return {
            "population_size": 100,   # 50-500
            "generations": 500,       # 100-2000
            "crossover_rate": 0.8,    # 0.6-0.95
            "mutation_rate": 0.1,     # 0.01-0.3
            "selection": "tournament",  # tournament/roulette/rank
            "tournament_size": 3,     # 2-10
            "elitism": 2,             # 0-10
            "crossover_type": "pmx",  # pmx/ox/cx for permutation; real uses "sbx"
        }

    @classmethod
    def param_schema(cls) -> Dict[str, Any]:
        return {
            "population_size": {"type": int, "min": 50, "max": 500},
            "generations": {"type": int, "min": 100, "max": 2000},
            "crossover_rate": {"type": float, "min": 0.6, "max": 0.95},
            "mutation_rate": {"type": float, "min": 0.01, "max": 0.3},
            "selection": {"type": str, "choices": ["tournament", "roulette", "rank"]},
            "tournament_size": {"type": int, "min": 2, "max": 10},
            "elitism": {"type": int, "min": 0, "max": 10},
            # PDF: permutation crossover options. We also accept "sbx" for real-coded.
            "crossover_type": {"type": str, "choices": ["pmx", "ox", "cx", "sbx"]},
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

        # ---- problem metadata
        rep = "real"
        bounds = None
        dim = None

        if callable(getattr(problem, "info", None)):
            pinfo = problem.info()
            dim = int(pinfo.dimension) if pinfo.dimension is not None else None
            if pinfo.extra:
                bounds = pinfo.extra.get("bounds")
                rep = (pinfo.extra.get("representation") or "real")
        else:
            dim = getattr(problem, "dim", None)

        if dim is None:
            raise ValueError("Problem must provide dimension (dim or info().dimension).")

        pop_size = params["population_size"]
        generations = params["generations"]
        cx_rate = params["crossover_rate"]
        mut_rate = params["mutation_rate"]
        selection = params["selection"]
        tsize = params["tournament_size"]
        elitism = min(params["elitism"], pop_size)
        cx_type = params["crossover_type"]

        # ---- helpers: evaluation
        def eval_real(P: np.ndarray) -> np.ndarray:
            return np.array([problem.evaluate(ind) for ind in P], dtype=float)

        # ---- selection
        def select_indices_tournament(F: np.ndarray) -> int:
            idx = rng.integers(0, pop_size, size=(tsize,))
            return int(idx[np.argmin(F[idx])])

        def select_indices_roulette(F: np.ndarray) -> int:
            # minimization -> convert to weights (higher weight = better)
            # add epsilon for stability
            eps = 1e-12
            inv = 1.0 / (F + eps)
            prob = inv / np.sum(inv)
            return int(rng.choice(pop_size, p=prob))

        def select_indices_rank(F: np.ndarray) -> int:
            order = np.argsort(F)  # best first
            ranks = np.empty(pop_size, dtype=int)
            ranks[order] = np.arange(pop_size)  # 0 best
            # rank-based probability: best gets highest
            weights = (pop_size - ranks).astype(float)
            prob = weights / np.sum(weights)
            return int(rng.choice(pop_size, p=prob))

        def pick_parent(F: np.ndarray) -> int:
            if selection == "tournament":
                return select_indices_tournament(F)
            if selection == "roulette":
                return select_indices_roulette(F)
            return select_indices_rank(F)

        # =========================================================
        # REAL-CODED GA (Function Optimization)
        # =========================================================
        if rep == "real":
            if bounds is None:
                lower = getattr(problem, "lower", None)
                upper = getattr(problem, "upper", None)
                if lower is None or upper is None:
                    raise ValueError("Real-coded GA needs bounds (lower/upper or info().extra['bounds']).")
                lower, upper = float(lower), float(upper)
            else:
                lower, upper = float(bounds[0]), float(bounds[1])

            # اگر کاربر crossover_type های permutation گذاشت، خودکار sbx کن (ولی پارامتر PDF حفظ می‌شود)
            if cx_type in ["pmx", "ox", "cx"]:
                cx_type = "sbx"

            sbx_eta = 15.0   # ثابت (اگر خواستی بعداً به PDF اضافه می‌کنیم)
            mut_eta = 20.0   # ثابت

            def sbx(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                if np.allclose(p1, p2):
                    return p1.copy(), p2.copy()
                u = rng.random(size=p1.shape)
                beta = np.empty_like(p1)
                mask = u <= 0.5
                beta[mask] = (2.0 * u[mask]) ** (1.0 / (sbx_eta + 1.0))
                beta[~mask] = (1.0 / (2.0 * (1.0 - u[~mask]))) ** (1.0 / (sbx_eta + 1.0))
                c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                return c1, c2

            def poly_mut(x: np.ndarray, p_gene: float) -> np.ndarray:
                y = x.copy()
                m = rng.random(size=y.shape) < p_gene
                if not np.any(m):
                    return y
                u = rng.random(size=y.shape)
                delta = np.zeros_like(y)
                mask = u < 0.5
                delta[mask] = (2.0 * u[mask]) ** (1.0 / (mut_eta + 1.0)) - 1.0
                delta[~mask] = 1.0 - (2.0 * (1.0 - u[~mask])) ** (1.0 / (mut_eta + 1.0))
                span = (upper - lower) if (upper - lower) != 0 else 1.0
                y[m] = y[m] + delta[m] * span
                return np.clip(y, lower, upper)

            # init
            P = rng.uniform(lower, upper, size=(pop_size, dim))
            F = eval_real(P)

            best_i = int(np.argmin(F))
            best_x = P[best_i].copy()
            best_f = float(F[best_i])
            history = [best_f]

            for gen in range(1, generations + 1):
                elite_idx = np.argsort(F)[:elitism] if elitism > 0 else np.array([], dtype=int)
                elites = P[elite_idx].copy() if elitism > 0 else None

                newP = []
                if elites is not None:
                    newP.extend([e.copy() for e in elites])

                while len(newP) < pop_size:
                    i1 = pick_parent(F)
                    i2 = pick_parent(F)
                    p1, p2 = P[i1].copy(), P[i2].copy()

                    if rng.random() < cx_rate:
                        c1, c2 = sbx(p1, p2)
                    else:
                        c1, c2 = p1, p2

                    # mutation_rate = per-individual -> تبدیل به per-gene
                    p_gene = min(1.0, mut_rate / max(1, dim))
                    c1 = poly_mut(c1, p_gene)
                    c2 = poly_mut(c2, p_gene)

                    newP.append(c1)
                    if len(newP) < pop_size:
                        newP.append(c2)

                P = np.asarray(newP, dtype=float)
                F = eval_real(P)

                bi = int(np.argmin(F))
                bf = float(F[bi])
                if bf < best_f:
                    best_f = bf
                    best_x = P[bi].copy()

                history.append(best_f)
                if progress_cb:
                    progress_cb(gen, best_f, {"gen": gen, "best": best_f})

            return MethodResult(
                method_name=self.name,
                best_solution=best_x,
                best_fitness=best_f,
                history=history,
                iterations=generations,
                status="ok",
                metrics={"representation": "real"},
            )

        # =========================================================
        # PERMUTATION GA (برای TSP بعداً)
        # =========================================================
        if rep == "permutation":
            # این بخش الان برای FunctionOpt استفاده نمی‌شه، اما برای TSP لازم می‌شه.
            # فرض: هر فرد یک permutation از 0..dim-1 هست و problem.evaluate(permutation) برمی‌گردونه.
            def eval_perm(P: np.ndarray) -> np.ndarray:
                return np.array([problem.evaluate(ind.tolist()) for ind in P], dtype=float)

            def pmx(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                n = a.size
                c1, c2 = a.copy(), b.copy()
                i, j = sorted(rng.integers(0, n, size=2))
                if i == j:
                    return c1, c2
                c1[i:j], c2[i:j] = b[i:j], a[i:j]

                def fix(child, parent_seg, donor_seg):
                    mapping = {donor_seg[k]: parent_seg[k] for k in range(len(parent_seg))}
                    for idx in list(range(0, i)) + list(range(j, n)):
                        val = child[idx]
                        while val in mapping:
                            val = mapping[val]
                        child[idx] = val

                fix(c1, a[i:j], b[i:j])
                fix(c2, b[i:j], a[i:j])
                return c1, c2

            def ox(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                n = a.size
                i, j = sorted(rng.integers(0, n, size=2))
                c1 = -np.ones(n, dtype=int)
                c2 = -np.ones(n, dtype=int)
                c1[i:j] = a[i:j]
                c2[i:j] = b[i:j]

                def fill(child, donor):
                    pos = j % n
                    for x in donor:
                        if x not in child:
                            child[pos] = x
                            pos = (pos + 1) % n

                fill(c1, b)
                fill(c2, a)
                return c1, c2

            def cx(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                n = a.size
                c1 = -np.ones(n, dtype=int)
                c2 = -np.ones(n, dtype=int)
                start = rng.integers(0, n)
                idx = start
                while c1[idx] == -1:
                    c1[idx] = a[idx]
                    c2[idx] = b[idx]
                    idx = int(np.where(a == b[idx])[0][0])
                # fill remaining
                for t in range(n):
                    if c1[t] == -1:
                        c1[t] = b[t]
                    if c2[t] == -1:
                        c2[t] = a[t]
                return c1, c2

            def swap_mut(x: np.ndarray) -> np.ndarray:
                y = x.copy()
                i, j = rng.integers(0, y.size, size=2)
                y[i], y[j] = y[j], y[i]
                return y

            # init permutations
            P = np.array([rng.permutation(dim) for _ in range(pop_size)], dtype=int)
            F = eval_perm(P)

            best_i = int(np.argmin(F))
            best_x = P[best_i].copy()
            best_f = float(F[best_i])
            history = [best_f]

            for gen in range(1, generations + 1):
                elite_idx = np.argsort(F)[:elitism] if elitism > 0 else np.array([], dtype=int)
                elites = P[elite_idx].copy() if elitism > 0 else None

                newP = []
                if elites is not None:
                    newP.extend([e.copy() for e in elites])

                while len(newP) < pop_size:
                    i1 = pick_parent(F)
                    i2 = pick_parent(F)
                    p1, p2 = P[i1].copy(), P[i2].copy()

                    if rng.random() < cx_rate:
                        if cx_type == "pmx":
                            c1, c2 = pmx(p1, p2)
                        elif cx_type == "ox":
                            c1, c2 = ox(p1, p2)
                        else:
                            c1, c2 = cx(p1, p2)
                    else:
                        c1, c2 = p1, p2

                    if rng.random() < mut_rate:
                        c1 = swap_mut(c1)
                    if rng.random() < mut_rate:
                        c2 = swap_mut(c2)

                    newP.append(c1)
                    if len(newP) < pop_size:
                        newP.append(c2)

                P = np.asarray(newP, dtype=int)
                F = eval_perm(P)

                bi = int(np.argmin(F))
                bf = float(F[bi])
                if bf < best_f:
                    best_f = bf
                    best_x = P[bi].copy()

                history.append(best_f)
                if progress_cb:
                    progress_cb(gen, best_f, {"gen": gen, "best": best_f})

            return MethodResult(
                method_name=self.name,
                best_solution=best_x.tolist(),
                best_fitness=best_f,
                history=history,
                iterations=generations,
                status="ok",
                metrics={"representation": "permutation"},
            )

        raise ValueError(f"Unknown representation: {rep}")

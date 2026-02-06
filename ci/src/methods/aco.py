from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from .base import BaseMethod, ProgressCallback
from .result import MethodResult
from src.utils.seeding import set_global_seed


class ACO(BaseMethod):
    name = "ACO"

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        # مطابق PDF
        return {
            "n_ants": 50,              # 10-100
            "max_iterations": 500,     # 100-2000
            "alpha": 1.0,              # 0.5-2.0
            "beta": 2.0,               # 1.0-5.0
            "evaporation_rate": 0.5,   # 0.1-0.9
            "q": 1.0,
            "initial_pheromone": 0.1,
            "local_search": True,      # 2-opt
        }

    @classmethod
    def param_schema(cls) -> Dict[str, Any]:
        return {
            "n_ants": {"type": int, "min": 10, "max": 100},
            "max_iterations": {"type": int, "min": 100, "max": 2000},
            "alpha": {"type": float, "min": 0.5, "max": 2.0},
            "beta": {"type": float, "min": 1.0, "max": 5.0},
            "evaporation_rate": {"type": float, "min": 0.1, "max": 0.9},
            "q": {"type": float, "min": 0.000001, "max": 1e9},
            "initial_pheromone": {"type": float, "min": 0.0, "max": 1e9},
            "local_search": {"type": bool},
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

        # Expect TSP-like problem:
        # - problem.n (cities)
        # - problem.D distance matrix
        # - evaluate(tour) -> length
        if not hasattr(problem, "D") or not hasattr(problem, "n"):
            raise ValueError("ACO expects TSPProblem with attributes D (distance matrix) and n (#cities).")

        n = int(problem.n)
        D = np.asarray(problem.D, dtype=float)

        alpha = params["alpha"]
        beta = params["beta"]
        rho = params["evaporation_rate"]
        Q = params["q"]
        n_ants = params["n_ants"]
        iters = params["max_iterations"]
        tau0 = params["initial_pheromone"]
        do_ls = params["local_search"]

        # heuristic: eta = 1 / distance (avoid div0)
        eps = 1e-12
        eta = 1.0 / (D + eps)
        np.fill_diagonal(eta, 0.0)

        # pheromone matrix
        tau = np.full((n, n), float(tau0), dtype=float)

        def route_length(tour: np.ndarray) -> float:
            total = 0.0
            for i in range(n - 1):
                total += D[tour[i], tour[i + 1]]
            total += D[tour[-1], tour[0]]
            return float(total)

        def two_opt(tour: np.ndarray) -> np.ndarray:
            best = tour.copy()
            best_len = route_length(best)
            improved = True
            while improved:
                improved = False
                for i in range(1, n - 2):
                    for k in range(i + 1, n - 1):
                        new = best.copy()
                        new[i:k+1] = best[i:k+1][::-1]
                        new_len = route_length(new)
                        if new_len + 1e-12 < best_len:
                            best = new
                            best_len = new_len
                            improved = True
                # loop until no improvement
            return best

        best_tour = None
        best_len = float("inf")
        history = []

        for t in range(1, iters + 1):
            all_tours = []
            all_lens = []

            for _ in range(n_ants):
                start = int(rng.integers(0, n))
                tour = [start]
                visited = np.zeros(n, dtype=bool)
                visited[start] = True

                for _step in range(n - 1):
                    i = tour[-1]
                    candidates = np.where(~visited)[0]

                    # probabilities proportional to (tau^alpha)*(eta^beta)
                    num = (tau[i, candidates] ** alpha) * (eta[i, candidates] ** beta)
                    s = np.sum(num)
                    if s <= 0:
                        j = int(rng.choice(candidates))
                    else:
                        prob = num / s
                        j = int(rng.choice(candidates, p=prob))

                    tour.append(j)
                    visited[j] = True

                tour = np.array(tour, dtype=int)

                if do_ls:
                    tour = two_opt(tour)

                L = route_length(tour)

                all_tours.append(tour)
                all_lens.append(L)

                if L < best_len:
                    best_len = L
                    best_tour = tour.copy()

            history.append(best_len)

            # evaporate
            tau *= (1.0 - rho)

            # deposit (best of iteration)
            idx_best = int(np.argmin(all_lens))
            tb = all_tours[idx_best]
            Lb = float(all_lens[idx_best])

            deposit = Q / max(Lb, eps)
            for i in range(n - 1):
                a, b = tb[i], tb[i + 1]
                tau[a, b] += deposit
                tau[b, a] += deposit
            tau[tb[-1], tb[0]] += deposit
            tau[tb[0], tb[-1]] += deposit

            if progress_cb:
                progress_cb(t, best_len, {"best_len": best_len})

        return MethodResult(
            method_name=self.name,
            best_solution=best_tour.tolist() if best_tour is not None else None,
            best_fitness=best_len,
            history=history,
            iterations=iters,
            status="ok",
            metrics={},
        )

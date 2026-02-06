import os
import numpy as np
import pandas as pd

from src.orchestrator.llm_orchestrator import LLMOrchestrator

from src.problems.function_opt import FunctionOptimizationProblem
from src.problems.tsp import TSPProblem
from src.problems.titanic import TitanicProblem
from src.problems.clustering import ClusteringProblem


def make_tsp_coords(n: int = 30, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)   
    return rng.uniform(0.0, 100.0, size=(n, 2))


def main():
    # IMPORTANT: do NOT hardcode keys in code. Use env.
    api_key = os.getenv("CHATANYWHERE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set env var CHATANYWHERE_API_KEY first.")

    os.makedirs("results/processed", exist_ok=True)
    os.makedirs("results/orchestrator_logs", exist_ok=True)
    os.makedirs("results/raw", exist_ok=True)

    orch = LLMOrchestrator(api_key=api_key, logs_dir="results/orchestrator_logs")

    coords = make_tsp_coords(n=30, seed=123)
    problems = [
        ("funcopt_rastrigin10d", FunctionOptimizationProblem(func_name="rastrigin", dim=10)),
        ("tsp30", TSPProblem(coords=coords, name_="tsp30")),
        ("titanic", TitanicProblem(seed=42)),
        ("clustering_iris", ClusteringProblem(dataset="iris", seed=42)),
    ]

    seeds = [1, 2, 3, 4, 5]

    rows = []
    for pname, p in problems:
        for s in seeds:
            tag = f"orch_{pname}"
            log = orch.run(problem=p, seed=s, tag=tag)

            rows.append(
                {
                    "problem": pname,
                    "seed": s,
                    "selected_method": log["decision"]["method_name"],
                    "backup_method": log["decision"].get("backup_method"),
                    "best_fitness": log["result"]["best_fitness"],
                    "time_sec": log["result"]["time_sec"],
                    "status": log["result"]["status"],
                    "performance": (log.get("analysis_json") or {}).get("performance"),
                    "backup_ran": bool(log.get("backup_step7") and isinstance(log["backup_step7"], dict) and "backup_result" in log["backup_step7"]),
                }
            )

            print(
                pname,
                "seed=", s,
                "method=", log["decision"]["method_name"],
                "perf=", (log.get("analysis_json") or {}).get("performance"),
                "best=", log["result"]["best_fitness"],
                "backup_ran=", rows[-1]["backup_ran"],
            )

    df = pd.DataFrame(rows)
    out_csv = "results/processed/orchestrator_eval_4problems_5runs.csv"
    df.to_csv(out_csv, index=False)

    print("\nSaved ->", out_csv)
    print(df.groupby(["problem", "selected_method"])[["best_fitness", "time_sec"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()

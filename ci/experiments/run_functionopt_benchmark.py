import os
import pandas as pd

from src.problems.function_opt import FunctionOptimizationProblem
from src.methods.pso import PSO
from src.methods.ga import GA
from src.evaluation.export import save_result_json


def run_benchmark():
    problem = FunctionOptimizationProblem(func_name="rastrigin", dim=10)
    desc = problem.get_llm_description()

    methods = [
        ("PSO", PSO(), {
            "n_particles": 50,
            "max_iterations": 500,
            "w": 0.7,
            "c1": 1.5,
            "c2": 1.5,
            "w_decay": True,
            "velocity_clamp": 0.5,
        }),
        ("GA", GA(), {
            "population_size": 160,
            "generations": 300,
            "crossover_rate": 0.9,
            "mutation_rate": 0.2,
            "selection": "tournament",
            "tournament_size": 3,
            "elitism": 2,
            "crossover_type": "pmx",  # برای real خودکار sbx می‌شود
        }),
    ]

    seeds = [1, 2, 3, 4, 5]

    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/processed", exist_ok=True)

    rows = []
    for mname, mobj, params in methods:
        for seed in seeds:
            res = mobj.run(problem=problem, params=params, seed=seed)
            out_json = f"results/raw/{mname.lower()}_pdf_rastrigin10d_seed{seed}.json"
            save_result_json(out_json, res, extra=desc)

            rows.append({
                "method": mname,
                "problem": "rastrigin_10d",
                "seed": seed,
                "best_fitness": res.best_fitness,
                "time_sec": res.time_sec,
                "iterations": res.iterations,
                "status": res.status,
            })

            print(f"{mname} seed={seed} best={res.best_fitness:.6e} time={res.time_sec:.3f}s status={res.status}")

    df = pd.DataFrame(rows)
    csv_path = "results/processed/benchmark_rastrigin10d_5runs_pdf.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary -> {csv_path}")
    print(df.groupby("method")[["best_fitness", "time_sec"]].agg(["mean", "std"]))


if __name__ == "__main__":
    run_benchmark()

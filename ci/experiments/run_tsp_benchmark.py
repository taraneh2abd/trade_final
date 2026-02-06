import os
import pandas as pd

from src.problems.tsp import TSPProblem
from src.methods.aco import ACO
from src.methods.ga import GA
from src.evaluation.export import save_result_json


def run_benchmark():
    # یک نمونه ثابت TSP برای همه روش‌ها (منصفانه)
    p = TSPProblem.random(n_cities=30, seed=42)
    desc = p.get_llm_description()

    methods = [
        ("ACO", ACO(), {
            "n_ants": 50,
            "max_iterations": 300,   # برای اینکه زمان معقول بمونه (PDF 100-2000)
            "alpha": 1.0,
            "beta": 2.0,
            "evaporation_rate": 0.5,
            "q": 1.0,
            "initial_pheromone": 0.1,
            "local_search": True,    # PDF
        }),
        ("GA", GA(), {
            "population_size": 120,
            "generations": 300,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
            "selection": "tournament",
            "tournament_size": 3,
            "elitism": 2,
            "crossover_type": "ox",  # pmx/ox/cx
        }),
    ]

    seeds = [1, 2, 3, 4, 5]

    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/processed", exist_ok=True)

    rows = []
    for mname, mobj, params in methods:
        for seed in seeds:
            res = mobj.run(problem=p, params=params, seed=seed)

            out_json = f"results/raw/{mname.lower()}_tsp30_seed{seed}.json"
            save_result_json(out_json, res, extra=desc)

            rows.append({
                "method": mname,
                "problem": "tsp_30",
                "seed": seed,
                "best_fitness": res.best_fitness,  # طول مسیر (کمتر بهتر)
                "time_sec": res.time_sec,
                "iterations": res.iterations,
                "status": res.status,
            })

            print(f"{mname} seed={seed} best_len={res.best_fitness:.3f} time={res.time_sec:.3f}s status={res.status}")

    df = pd.DataFrame(rows)
    csv_path = "results/processed/benchmark_tsp30_5runs.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary -> {csv_path}")
    print(df.groupby("method")[["best_fitness", "time_sec"]].agg(["mean", "std"]))


if __name__ == "__main__":
    run_benchmark()

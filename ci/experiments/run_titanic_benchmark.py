import os
import pandas as pd

from src.problems.titanic import TitanicProblem
from src.methods.perceptron import Perceptron
from src.methods.mlp import MLP
from src.evaluation.export import save_result_json


def run_benchmark():
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/processed", exist_ok=True)

    seeds = [1, 2, 3, 4, 5]

    p = TitanicProblem(seed=42)
    desc = p.get_llm_description()

    methods = [
        ("Perceptron", Perceptron(), {"learning_rate": 0.01, "epochs": 200, "l2": 0.0, "threshold": 0.5}),
        ("MLP", MLP(), {"hidden_size": 16, "learning_rate": 0.01, "epochs": 300, "batch_size": 64, "l2": 0.0, "patience": 40, "threshold": 0.5}),
    ]

    rows = []
    for name, method, params in methods:
        for seed in seeds:
            res = method.run(problem=p, params=params, seed=seed)

            out_json = f"results/raw/{name.lower()}_titanic_seed{seed}.json"
            save_result_json(out_json, res, extra=desc)

            rows.append({
                "method": name,
                "seed": seed,
                "best_fitness": res.best_fitness,   # اینجا = test_f1
                "test_accuracy": res.metrics.get("test_accuracy"),
                "test_precision": res.metrics.get("test_precision"),
                "test_recall": res.metrics.get("test_recall"),
                "test_f1": res.metrics.get("test_f1"),
                "time_sec": res.time_sec,
                "iterations": res.iterations,
                "status": res.status,
            })

            print(f"{name} seed={seed} test_f1={res.metrics.get('test_f1'):.4f} acc={res.metrics.get('test_accuracy'):.4f} time={res.time_sec:.3f}s")

    df = pd.DataFrame(rows)
    csv_path = "results/processed/benchmark_titanic_5runs.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary -> {csv_path}")
    print(df.groupby("method")[["test_f1", "test_accuracy", "time_sec"]].agg(["mean", "std"]))


if __name__ == "__main__":
    run_benchmark()

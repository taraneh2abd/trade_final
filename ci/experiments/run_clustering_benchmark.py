import os
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

from src.problems.clustering import ClusteringProblem
from src.methods.som import SOM
from src.evaluation.clustering_metrics import compute_clustering_metrics
from src.evaluation.export import save_result_json
from src.methods.result import MethodResult


def run_one_kmeans(X, y_true, k, seed):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    metrics = compute_clustering_metrics(X, labels, y_true=y_true)
    # best_fitness = silhouette
    return MethodResult(
        method_name="KMeans",
        best_solution={"labels": labels.tolist()},
        best_fitness=float(metrics.get("silhouette", float("nan"))),
        history=[],
        iterations=1,
        status="ok",
        metrics=metrics,
    )


def main():
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/processed", exist_ok=True)

    # dataset (سریع و استاندارد)
    p = ClusteringProblem(dataset="iris", seed=42)
    desc = p.get_llm_description()
    data = p.get_data()
    X = data["X"]
    y_true = data.get("y_true", None)
    k = data["n_clusters"]

    seeds = [1, 2, 3, 4, 5]

    som = SOM()
    som_params = {
        "grid_h": 10,
        "grid_w": 10,
        "epochs": 50,
        "learning_rate": 0.5,
        "sigma": 2.0,
        "decay": "linear",
        "cluster_mode": "kmeans",
    }

    rows = []
    for seed in seeds:
        # SOM
        res_som = som.run(problem=p, params=som_params, seed=seed)
        save_result_json(f"results/raw/som_iris_seed{seed}.json", res_som, extra=desc)
        rows.append({
            "method": "SOM",
            "seed": seed,
            **res_som.metrics,
            "time_sec": res_som.time_sec,
            "status": res_som.status,
        })
        print(f"SOM seed={seed} silhouette={res_som.metrics.get('silhouette'):.4f} time={res_som.time_sec:.3f}s")

        # KMeans baseline
        res_km = run_one_kmeans(X, y_true, k, seed)
        save_result_json(f"results/raw/kmeans_iris_seed{seed}.json", res_km, extra=desc)
        rows.append({
            "method": "KMeans",
            "seed": seed,
            **res_km.metrics,
            "time_sec": res_km.time_sec,
            "status": res_km.status,
        })
        print(f"KMeans seed={seed} silhouette={res_km.metrics.get('silhouette'):.4f} time={res_km.time_sec:.3f}s")

    df = pd.DataFrame(rows)
    csv_path = "results/processed/benchmark_clustering_iris_5runs.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary -> {csv_path}")

    # خلاصه mean/std روی چند متریک مهم
    cols = ["silhouette", "davies_bouldin", "calinski_harabasz"]
    if "ari" in df.columns:
        cols += ["ari", "nmi"]
    cols += ["time_sec"]

    print(df.groupby("method")[cols].agg(["mean", "std"]))


if __name__ == "__main__":
    main()

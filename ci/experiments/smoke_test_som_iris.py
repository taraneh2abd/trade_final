from src.problems.clustering import ClusteringProblem
from src.methods.som import SOM
from src.evaluation.export import save_result_json

def cb(it, best, extra):
    if it == 1 or it % 10 == 0:
        print(f"epoch={it} silhouette={extra['silhouette']:.4f}")

if __name__ == "__main__":
    p = ClusteringProblem(dataset="iris", seed=42)  # k=3
    m = SOM()

    res = m.run(
        problem=p,
        params={
            "grid_h": 10,
            "grid_w": 10,
            "epochs": 50,
            "learning_rate": 0.5,
            "sigma": 2.0,
            "decay": "linear",
            "cluster_mode": "kmeans",  # خروجی k=3 مثل PDF
        },
        progress_cb=cb,
        seed=42,
    )

    print("\nFINAL silhouette:", res.best_fitness)
    print("metrics:", res.metrics)

    save_result_json("results/raw/som_iris_seed42.json", res, extra=p.get_llm_description())
    print("Saved -> results/raw/som_iris_seed42.json")

from src.problems.tsp import TSPProblem
from src.methods.aco import ACO
from src.evaluation.export import save_result_json

def cb(it, best, extra):
    if it % 50 == 0 or it == 1:
        print(f"iter={it} best_len={best:.3f}")

if __name__ == "__main__":
    p = TSPProblem.random(n_cities=30, seed=42)

    m = ACO()
    res = m.run(
        problem=p,
        params={
            "n_ants": 50,
            "max_iterations": 300,   # برای تست سریع کمتر از 500
            "alpha": 1.0,
            "beta": 2.0,
            "evaporation_rate": 0.5,
            "q": 1.0,
            "initial_pheromone": 0.1,
            "local_search": True,
        },
        progress_cb=cb,
        seed=42
    )

    print("\nFINAL:", res.best_fitness, "status=", res.status, "iters=", res.iterations)

    save_result_json("results/raw/aco_tsp30_seed42.json", res, extra=p.get_llm_description())
    print("Saved -> results/raw/aco_tsp30_seed42.json")

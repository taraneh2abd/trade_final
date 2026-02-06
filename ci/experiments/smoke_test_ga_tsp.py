from src.problems.tsp import TSPProblem
from src.methods.ga import GA
from src.evaluation.export import save_result_json

def cb(it, best, extra):
    if it % 50 == 0 or it == 1:
        print(f"gen={it} best_len={best:.3f}")

if __name__ == "__main__":
    # همون دیتای تصادفی ثابت برای تکرارپذیری
    p = TSPProblem.random(n_cities=30, seed=42)

    m = GA()
    res = m.run(
        problem=p,
        params={
            "population_size": 120,     # داخل رنج PDF (50-500)
            "generations": 300,         # داخل رنج PDF (100-2000)
            "crossover_rate": 0.9,      # داخل رنج PDF
            "mutation_rate": 0.1,       # داخل رنج PDF
            "selection": "tournament",  # PDF options
            "tournament_size": 3,       # PDF range
            "elitism": 2,               # PDF range
            "crossover_type": "ox",     # یکی از pmx/ox/cx (برای TSP معمولاً ox خوبه)
        },
        progress_cb=cb,
        seed=42,
    )

    print("\nFINAL:", res.best_fitness, "status=", res.status, "iters=", res.iterations)

    save_result_json("results/raw/ga_tsp30_seed42.json", res, extra=p.get_llm_description())
    print("Saved -> results/raw/ga_tsp30_seed42.json")

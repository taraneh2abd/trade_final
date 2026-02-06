from src.problems.function_opt import FunctionOptimizationProblem
from src.methods.ga import GA

def cb(it, best, extra):
    if it % 50 == 0 or it == 1:
        print(f"gen={it} best={best:.6e}")

if __name__ == "__main__":
    p = FunctionOptimizationProblem(func_name="rastrigin", dim=10)

    m = GA()
    res = m.run(
        problem=p,
        params={
            "population_size": 160,
            "generations": 300,
            "crossover_rate": 0.9,
            "mutation_rate": 0.2,
            "selection": "tournament",
            "tournament_size": 3,
            "elitism": 2,
            "crossover_type": "pmx",  # برای real خودکار می‌شه sbx
        },
        progress_cb=cb,
        seed=42,
    )

    print("\nFINAL:", res.best_fitness, "status=", res.status, "iters=", res.iterations)

from src.problems.function_opt import FunctionOptimizationProblem
from src.methods.pso import PSO
from src.evaluation.export import save_result_json

def cb(it, best, extra):
    if it % 20 == 0 or it == 1:
        print(f"iter={it} best={best:.6f} w={extra.get('w'):.3f}")

if __name__ == "__main__":
    p = FunctionOptimizationProblem(func_name="rastrigin", dim=10)

    m = PSO()
    res = m.run(
        problem=p,
        params={
            "n_particles": 50,        # PDF default
            "max_iterations": 500,    # PDF default
            "w": 0.7,                 # PDF default
            "c1": 1.5,                # PDF default
            "c2": 1.5,                # PDF default
            "w_decay": True,          # PDF default
            "velocity_clamp": 0.5,    # PDF default
        },
        progress_cb=cb,
        seed=42,
    )

    print("\nFINAL:", res.best_fitness, "status=", res.status, "iters=", res.iterations)

    save_result_json(
        "results/raw/pso_pdf_rastrigin_10d_seed42.json",
        res,
        extra=p.get_llm_description(),
    )
    print("Saved -> results/raw/pso_pdf_rastrigin_10d_seed42.json")

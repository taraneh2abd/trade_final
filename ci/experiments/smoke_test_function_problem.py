import numpy as np
from src.problems.function_opt import FunctionOptimizationProblem

if __name__ == "__main__":
    p = FunctionOptimizationProblem(func_name="rastrigin", dim=10)
    print(p.info())
    print(p.get_llm_description())

    rng = np.random.default_rng(42)
    x = p.sample_random(rng)
    f = p.evaluate(x)
    print("sample fitness:", f)

    x0 = np.zeros(10)
    print("fitness at zero (should be 0):", p.evaluate(x0))

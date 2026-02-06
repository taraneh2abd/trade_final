import numpy as np
from src.problems.function_opt import FunctionOptimizationProblem

if __name__ == "__main__":
    for fname in ["sphere", "rastrigin", "ackley", "rosenbrock"]:
        p = FunctionOptimizationProblem(func_name=fname, dim=10)
        z = np.zeros(10)
        print(fname, "fitness at zero =", p.evaluate(z))

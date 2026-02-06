from src.methods.base import BaseMethod
from src.methods.result import MethodResult


class DummyMethod(BaseMethod):
    name = "DummyMethod"

    @classmethod
    def default_params(cls):
        return {"max_iterations": 5}

    @classmethod
    def param_schema(cls):
        return {"max_iterations": {"type": int, "min": 1, "max": 1000}}

    def solve(self, problem, params, progress_cb=None, seed=None):
        best = 100.0
        history = []
        for it in range(params["max_iterations"]):
            best *= 0.9
            history.append(best)
            if progress_cb:
                progress_cb(it + 1, best, {"note": "running"})
        return MethodResult(
            method_name=self.name,
            best_solution=None,
            best_fitness=best,
            history=history,
            iterations=len(history),
            status="ok",
        )


def cb(it, best, extra):
    print(f"iter={it} best={best:.4f} extra={extra}")


if __name__ == "__main__":
    m = DummyMethod()
    res = m.run(problem=None, params={"max_iterations": 5}, progress_cb=cb, seed=42)
    print(res)

from src.problems.titanic import TitanicProblem
from src.methods.perceptron import Perceptron

def cb(it, best, extra):
    if it == 1 or it % 20 == 0:
        print(f"epoch={it} best_val_f1={best:.4f} val_f1={extra['val_f1']:.4f}")

if __name__ == "__main__":
    p = TitanicProblem(seed=42)
    m = Perceptron()
    res = m.run(problem=p, params={"learning_rate": 0.01, "epochs": 200, "l2": 0.0}, progress_cb=cb, seed=42)
    print(res.metrics)

from src.problems.titanic import TitanicProblem
from src.methods.mlp import MLP

def cb(it, best, extra):
    if it == 1 or it % 25 == 0:
        print(f"epoch={it} best_val_f1={best:.4f} val_f1={extra['val_f1']:.4f}")

if __name__ == "__main__":
    p = TitanicProblem(seed=42)
    m = MLP()
    res = m.run(problem=p, params={"hidden_size": 16, "learning_rate": 0.01, "epochs": 300, "batch_size": 64, "patience": 40}, progress_cb=cb, seed=42)
    print(res.metrics)

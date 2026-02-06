# تست سریع همه متدها
# python ci/experiments/test_all_methods.py

# رسم نمودارها
from src.evaluation.plots import Plotter

plotter = Plotter()

# مثال: رسم همگرایی
histories = {
    "GA": [100, 80, 60, 50, 45, 42, 41, 40.5, 40.2, 40.1],
    "PSO": [100, 85, 70, 58, 49, 43, 41, 40.3, 40.2, 40.1],
    "ACO": [100, 90, 82, 75, 68, 62, 57, 53, 50, 48],
}

plotter.plot_convergence(
    histories,
    title="TSP Optimization Convergence",
    filename="tsp_convergence.png"
)
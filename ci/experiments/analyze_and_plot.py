# ci/experiments/analyze_and_plot.py
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from src.evaluation.plots import Plotter

def load_results():
    """Load results from JSON files"""
    results = []
    for f in glob.glob("results/raw/*.json"):
        with open(f) as file:
            data = json.load(file)
            data["filename"] = f
            results.append(data)
    return results

def create_all_plots():
    """Create all required plots"""
    results = load_results()
    if not results:
        print("⚠️  No results found for analysis!")
        return
    
    plotter = Plotter()
    
    # 1. Group by problem
    problems = {}
    for r in results:
        prob_name = r.get("extra", {}).get("name", "unknown")
        if prob_name not in problems:
            problems[prob_name] = []
        problems[prob_name].append(r)
    
    # 2. Create plots for each problem
    for prob_name, prob_results in problems.items():
        print(f"📈 Creating plots for: {prob_name}")
        
        # Convergence plot
        histories = {}
        for i, r in enumerate(prob_results):
            method = r.get("method_name", f"Run_{i+1}")
            history = r.get("history", [])
            if history:
                histories[method] = history
        
        if histories:
            plotter.plot_convergence(
                histories,
                title=f"Convergence - {prob_name}",
                filename=f"{prob_name}_convergence.png"
            )
        
        # Fitness comparison plot
        fitness_data = {}
        for r in prob_results:
            method = r.get("method_name", "unknown")
            if method not in fitness_data:
                fitness_data[method] = []
            fitness_data[method].append(r.get("best_fitness", 0))
        
        if len(fitness_data) > 1:
            plotter.plot_box_comparison(
                fitness_data,
                title=f"Method Comparison - {prob_name}",
                filename=f"{prob_name}_comparison.png"
            )
    
    # 3. Summary plot
    create_summary_plot(results)

def create_summary_plot(results):
    """Create overall summary plot"""
    df = pd.DataFrame([
        {
            "problem": r.get("extra", {}).get("name", "unknown"),
            "method": r.get("method_name", "unknown"),
            "fitness": r.get("best_fitness", 0),
            "time": r.get("time_sec", 0)
        }
        for r in results
    ])
    
    plt.figure(figsize=(12, 6))
    
    # Average fitness per problem
    avg_fitness = df.groupby(["problem", "method"])["fitness"].mean().unstack()
    avg_fitness.plot(kind="bar", ax=plt.gca())
    plt.title("Average Fitness by Problem and Method")
    plt.xlabel("Problem")
    plt.ylabel("Fitness")
    plt.legend(title="Method")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/figures/summary_fitness.png", dpi=300)
    
    print("✅ Plots created in results/figures/")

if __name__ == "__main__":
    create_all_plots()
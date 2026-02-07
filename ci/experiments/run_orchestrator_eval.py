# ci/experiments/run_orchestrator_eval.py
import os
import numpy as np
import pandas as pd
from colorama import init, Fore, Back, Style
from tabulate import tabulate

# Initialize colorama
init(autoreset=True)

from src.orchestrator.llm_orchestrator import LLMOrchestrator
from src.problems.function_opt import FunctionOptimizationProblem
from src.problems.tsp import TSPProblem
from src.problems.titanic import TitanicProblem
from src.problems.clustering import ClusteringProblem

def make_tsp_coords(n: int = 30, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)   
    return rng.uniform(0.0, 100.0, size=(n, 2))

def print_summary_table(df):
    """Print final summary as beautiful table"""
    summary = df.groupby(["problem", "selected_method"])[["best_fitness", "time_sec"]].agg(["mean", "std"])
    
    # Prepare table data
    table_data = []
    for (problem, method), row in summary.iterrows():
        fitness_mean = row[('best_fitness', 'mean')]
        fitness_std = row[('best_fitness', 'std')]
        time_mean = row[('time_sec', 'mean')]
        time_std = row[('time_sec', 'std')]
        
        table_data.append([
            problem,
            method,
            f"{fitness_mean:.6f} ± {fitness_std:.6f}",
            f"{time_mean:.2f}s ± {time_std:.2f}s"
        ])
    
    # Print table
    print(f"\n{Fore.CYAN}{'═' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'FINAL SUMMARY RESULTS':^80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'═' * 80}{Style.RESET_ALL}")
    
    headers = ["Problem", "Method", "Fitness (mean ± std)", "Time (mean ± std)"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    
    # Also print to file
    with open("results/processed/summary_table.txt", "w") as f:
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    # Header
    print(f"{Back.CYAN}{Fore.BLACK}{'='*60}{Style.RESET_ALL}")
    print(f"{Back.CYAN}{Fore.BLACK}{'META-MIND CI ORCHESTRATOR':^60}{Style.RESET_ALL}")
    print(f"{Back.CYAN}{Fore.BLACK}{'='*60}{Style.RESET_ALL}")
    
    # API Key check
    api_key = os.getenv("CHATANYWHERE_API_KEY", "").strip()
    if not api_key:
        print(f"{Fore.RED}❌ ERROR: Set env var CHATANYWHERE_API_KEY first.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Command: export CHATANYWHERE_API_KEY='your-key-here'{Style.RESET_ALL}")
        raise RuntimeError("Set env var CHATANYWHERE_API_KEY first.")

    # Setup directories
    os.makedirs("results/processed", exist_ok=True)
    os.makedirs("results/orchestrator_logs", exist_ok=True)
    os.makedirs("results/raw", exist_ok=True)

    # Initialize orchestrator
    print(f"\n{Fore.GREEN}🚀 Initializing LLM Orchestrator...{Style.RESET_ALL}")
    orch = LLMOrchestrator(api_key=api_key, logs_dir="results/orchestrator_logs")

    # Create problems
    coords = make_tsp_coords(n=30, seed=123)
    problems = [
        ("Rastrigin", FunctionOptimizationProblem(func_name="rastrigin", dim=10)),
        ("TSP", TSPProblem(coords=coords, name_="tsp30")),
        # ("Titanic", TitanicProblem(seed=42)),
        # ("Iris", ClusteringProblem(dataset="iris", seed=42)),
    ]

    # Run 5 times each
    seeds = [1,2,3]
    rows = []
    
    print(f"\n{Fore.YELLOW}📊 Starting 4 problems × 5 runs = 20 experiments{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}⏱️  Estimated time: ~5-10 minutes{Style.RESET_ALL}")
    
    for pname, p in problems:
        print(f"\n{Fore.MAGENTA}{'─' * 60}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}🎯 PROBLEM: {pname}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'─' * 60}{Style.RESET_ALL}")
        
        for s in seeds:
            tag = f"orch_{pname}"
            
            try:
                log = orch.run(problem=p, seed=s, tag=tag, verbose=True)  # verbose=True shows pretty table
                
                # Collect data for CSV
                rows.append({
                    "problem": pname,
                    "seed": s,
                    "selected_method": log["decision"]["method_name"],
                    "backup_method": log.get("backup_method"),
                    "best_fitness": log["result"]["best_fitness"],
                    "time_sec": log["result"]["time_sec"],
                    "status": log["result"]["status"],
                    "performance": log["step6"]["performance_assessment"],
                    "backup_ran": log["backup_ran"],
                })

            except Exception as e:
                print(f"{Fore.RED}❌ Error in {pname} seed {s}: {e}{Style.RESET_ALL}")
                rows.append({
                    "problem": pname,
                    "seed": s,
                    "selected_method": "ERROR",
                    "backup_method": None,
                    "best_fitness": float('nan'),
                    "time_sec": 0,
                    "status": "failed",
                    "performance": "error",
                    "backup_ran": False,
                })
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    out_csv = "results/processed/orchestrator_eval_4problems_5runs.csv"
    df.to_csv(out_csv, index=False)
    
    print(f"\n{Fore.GREEN}✅ CSV saved: {out_csv}{Style.RESET_ALL}")
    
    # Print final summary
    print_summary_table(df)
    
    # Final message
    print(f"\n{Back.GREEN}{Fore.BLACK}{'='*60}{Style.RESET_ALL}")
    print(f"{Back.GREEN}{Fore.BLACK}{'EXPERIMENT COMPLETE!':^60}{Style.RESET_ALL}")
    print(f"{Back.GREEN}{Fore.BLACK}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📁 Logs: results/orchestrator_logs/{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📊 Data: results/processed/{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📈 Raw: results/raw/{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
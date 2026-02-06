import os
from src.orchestrator.llm_orchestrator import LLMOrchestrator
from src.problems.function_opt import FunctionOptimizationProblem

def main():
    api_key = os.getenv("CHATANYWHERE_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set CHATANYWHERE_API_KEY first.")

    orch = LLMOrchestrator(api_key=api_key, logs_dir="results/orchestrator_logs")
    p = FunctionOptimizationProblem(func_name="rastrigin", dim=10)

    out = orch.run(problem=p, seed=1, tag="orch_smoke_funcopt")
    print("DECISION:", out["decision"])
    print("RESULT:", out["result"])
    print("\nANALYSIS:\n", out["analysis_text"])
    print("\nLOG FILE:", out["log_path"])

if __name__ == "__main__":
    main()

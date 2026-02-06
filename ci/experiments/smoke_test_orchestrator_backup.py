import os
from src.orchestrator.llm_orchestrator import LLMOrchestrator
from src.problems.function_opt import FunctionOptimizationProblem

def main():
    api_key = os.getenv("CHATANYWHERE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set env var CHATANYWHERE_API_KEY first.")

    orch = LLMOrchestrator(api_key=api_key)

    # Function optimization که احتمالاً بعضی بارها poor میشه و backup می‌پره
    p = FunctionOptimizationProblem(func_name="rastrigin", dim=10)

    log = orch.run(problem=p, seed=1, tag="orch_smoke_backup_funcopt")
    print("decision:", log["decision"])
    print("analysis:", log["analysis_json"])
    print("backup_step7:", log["backup_step7"])
    print("log_path:", log["log_path"])

if __name__ == "__main__":
    main()

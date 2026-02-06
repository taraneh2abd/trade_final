from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.evaluation.export import save_result_json

from src.methods.pso import PSO
from src.methods.ga import GA
from src.methods.aco import ACO
from src.methods.som import SOM
from src.methods.perceptron import Perceptron
from src.methods.mlp import MLP

from src.orchestrator.chatanywhere_client import ChatAnywhereClient


@dataclass
class OrchestratorDecision:
    method_name: str
    params: Dict[str, Any]
    reasoning: str


class LLMOrchestrator:
    """
    Flow مطابق PDF:
    1) Parse Problem
    2) LLM Select Method + Params (JSON)
    3) Run Method
    4) Evaluate + Summarize Results
    5) Feedback Results to LLM
    6) LLM Interpretation + Recommendations (JSON: good/acceptable/poor + ...)
    7) Log All Interactions (JSON)

    + (اختیاری ولی خیلی مفید): اگر Step6 گفت poor → backup method را واقعاً اجرا می‌کنیم.
    """

    def __init__(
        self,
        api_key: str,
        logs_dir: str = "results/orchestrator_logs",
        model: str = "gpt-4o-mini-2024-07-18",
        enable_backup: bool = True,
    ):
        self.logs_dir = logs_dir
        self.enable_backup = enable_backup
        os.makedirs(self.logs_dir, exist_ok=True)
        self.llm = ChatAnywhereClient(api_key=api_key, model=model)

    # ---------- Step 1: Problem Parser ----------
    def parse_problem(self, problem: Any) -> Dict[str, Any]:
        if hasattr(problem, "get_llm_description"):
            return problem.get_llm_description()
        info = problem.info()
        return {
            "problem_type": info.problem_type,
            "name": info.name,
            "objective": info.objective,
            "dimension": info.dimension,
            "extra": info.extra,
        }

    # ---------- Methods factory ----------
    def _build_method(self, method_name: str):
        m = method_name.strip().upper()
        if m == "PSO":
            return PSO()
        if m == "GA":
            return GA()
        if m == "ACO":
            return ACO()
        if m == "SOM":
            return SOM()
        if m == "PERCEPTRON":
            return Perceptron()
        if m == "MLP":
            return MLP()
        raise ValueError(f"Unknown method: {method_name}")

    # ---------- helpers ----------
    def _history_summary(self, history: Optional[list]) -> Dict[str, Any]:
        if not history:
            return {"has_history": False}
        return {
            "has_history": True,
            "len": int(len(history)),
            "first": float(history[0]),
            "last": float(history[-1]),
            "best": float(min(history)),
        }

    def _sanitize_params(self, method: Any, llm_params: Dict[str, Any]) -> Tuple[Dict[str, Any], list]:
        schema = {}
        if hasattr(method, "param_schema"):
            try:
                schema = method.param_schema() or {}
            except Exception:
                schema = {}

        defaults = {}
        if hasattr(method, "default_params"):
            try:
                defaults = method.default_params() or {}
            except Exception:
                defaults = {}

        allowed = set(schema.keys()) if schema else set(defaults.keys())

        clean: Dict[str, Any] = {}
        llm_params = llm_params or {}
        for k, v in llm_params.items():
            if k in allowed:
                clean[k] = v

        for k, v in defaults.items():
            clean.setdefault(k, v)

        dropped = sorted(list(set(llm_params.keys()) - set(clean.keys())))
        return clean, dropped

    def _coerce_perf(self, x: Any) -> str:
        s = str(x or "").strip().lower()
        if s in {"good", "acceptable", "poor"}:
            return s
        return "acceptable"

    def _pick_backup(self, desc: Dict[str, Any], primary_method: str) -> str:
        """
        یک backup ثابت و منطقی (Rule-based) — تصمیم اجرا با orchestrator است.
        """
        ptype = str(desc.get("problem_type", "")).lower()

        if ptype == "optimization":
            return "PSO" if primary_method.upper() == "GA" else "GA"

        if ptype == "classification":
            return "PERCEPTRON" if primary_method.upper() == "MLP" else "MLP"

        if ptype == "tsp":
            return "ACO" if primary_method.upper() == "GA" else "GA"

        if ptype == "clustering":
            # اگر KMeans به عنوان method داری می‌تونی اینجا بذاری، فعلاً SOM backup ندارد
            return primary_method

        return primary_method

    # ---------- Step 2: LLM Decision (JSON) ----------
    def llm_decide_method_and_params(self, desc: Dict[str, Any]) -> Dict[str, Any]:
        system = (
            "You are a CI orchestrator.\n"
            "Return ONLY valid JSON. No extra text.\n"
            "Schema:\n"
            "{"
            '"method_name": "PSO|GA|ACO|SOM|Perceptron|MLP",'
            '"params": { ... },'
            '"reasoning": "..."'
            "}"
        )

        user = (
            "Choose one method and parameters based on the problem.\n"
            "Rules:\n"
            "- TSP -> GA or ACO\n"
            "- Continuous function optimization -> GA or PSO\n"
            "- Titanic classification -> Perceptron or MLP\n"
            "- Clustering -> SOM\n\n"
            "IMPORTANT CONSTRAINTS:\n"
            "- For TSP: if you select GA, you MUST set crossover_type='ox' (not pmx).\n"
            "- For TSP: mutation_rate should be around 0.05-0.2, crossover_rate around 0.8-0.95.\n\n"
            f"Problem description:\n{json.dumps(desc, ensure_ascii=False)}"
        )

        return self.llm.generate_json(system=system, user=user, temperature=0.0)

    # ---------- Step 6: LLM Interpretation + Recommendations (STRICT JSON) ----------
    def llm_interpret_and_recommend(self, desc: Dict[str, Any], decision: OrchestratorDecision, res: Any) -> Dict[str, Any]:
        expected = {
            "known_optimum": desc.get("known_optimum", None),
            "objective": desc.get("objective", None),
            "problem_type": desc.get("problem_type", None),
        }

        result_summary = {
            "method_name": res.method_name,
            "best_fitness": res.best_fitness,
            "iterations": res.iterations,
            "time_sec": res.time_sec,
            "status": res.status,
            "metrics": res.metrics,
            "history_summary": self._history_summary(getattr(res, "history", None)),
        }

        system = (
            "You are an expert CI analyst.\n"
            "Return ONLY valid JSON. No extra text.\n"
            "You MUST follow this exact schema and fill all fields:\n"
            "{\n"
            '  "performance_assessment": "good|acceptable|poor",\n'
            '  "comparison_to_expected": "string",\n'
            '  "natural_language_explanation": "string",\n'
            '  "recommendations": {\n'
            '    "parameter_tuning": ["..."],\n'
            '    "alternative_methods": ["..."],\n'
            '    "hybrid_approaches": ["..."]\n'
            "  }\n"
            "}\n"
            "Rules:\n"
            "- Be concrete and reference provided numbers/metrics.\n"
            "- If known_optimum is provided, compare against it.\n"
            "- Keep each recommendation short and actionable.\n"
        )

        user = (
            f"Problem:\n{json.dumps(desc, ensure_ascii=False)}\n\n"
            f"Expected/Optimum info:\n{json.dumps(expected, ensure_ascii=False)}\n\n"
            f"Decision:\n{json.dumps({'method_name': decision.method_name, 'params': decision.params, 'reasoning': decision.reasoning}, ensure_ascii=False)}\n\n"
            f"Run result summary:\n{json.dumps(result_summary, ensure_ascii=False)}\n\n"
            "Now produce the Step-6 JSON."
        )

        out = self.llm.generate_json(system=system, user=user, temperature=0.2)

        j = out.get("json") or {}
        fixed = {
            "performance_assessment": self._coerce_perf(j.get("performance_assessment")),
            "comparison_to_expected": str(j.get("comparison_to_expected", "")).strip(),
            "natural_language_explanation": str(j.get("natural_language_explanation", "")).strip(),
            "recommendations": {
                "parameter_tuning": list(j.get("recommendations", {}).get("parameter_tuning", []) or []),
                "alternative_methods": list(j.get("recommendations", {}).get("alternative_methods", []) or []),
                "hybrid_approaches": list(j.get("recommendations", {}).get("hybrid_approaches", []) or []),
            },
        }

        return {"text": out.get("text", ""), "raw": out.get("raw", None), "json": fixed}

    # ---------- Main runner ----------
    def run(self, problem: Any, seed: int = 42, tag: str = "orch_run", verbose: bool = True) -> Dict[str, Any]:
        os.makedirs("results/raw", exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Step 1
        desc = self.parse_problem(problem)

        # Step 2
        dec = self.llm_decide_method_and_params(desc)
        dec_json = dec["json"]

        method_name = str(dec_json.get("method_name", "")).strip()
        llm_params = dec_json.get("params", {}) or {}
        reasoning = str(dec_json.get("reasoning", "")).strip()

        # Build + sanitize
        method = self._build_method(method_name)
        params, dropped = self._sanitize_params(method, llm_params)

        decision = OrchestratorDecision(method_name=method_name, params=params, reasoning=reasoning)

        # Step 3 run
        t0 = time.time()
        res = method.run(problem=problem, params=params, seed=seed)
        wall = time.time() - t0

        # Step 6 (LLM output)
        analysis = self.llm_interpret_and_recommend(desc, decision, res)
        step6 = analysis["json"]
        perf = step6["performance_assessment"]

        # Backup logic (Orchestrator decides; LLM فقط پیشنهاد می‌دهد)
        backup_ran = False
        backup_result = None
        backup_method_name = None

        if self.enable_backup and perf == "poor":
            backup_method_name = self._pick_backup(desc, decision.method_name)
            if backup_method_name.upper() != decision.method_name.upper():
                backup_method = self._build_method(backup_method_name)
                backup_params, _ = self._sanitize_params(backup_method, {})  # defaults
                t1 = time.time()
                backup_result = backup_method.run(problem=problem, params=backup_params, seed=seed)
                _ = time.time() - t1
                backup_ran = True

        # Step 7 log
        log = {
            "tag": tag,
            "seed": seed,
            "problem_desc": desc,

            "llm_decision": {
                "response_text": dec["text"],
                "response_json": dec_json,
                "raw": dec["raw"],
            },

            "decision": {
                "method_name": decision.method_name,
                "params": decision.params,
                "reasoning": decision.reasoning,
                "dropped_param_keys": dropped,
            },

            "run_result": {
                "method_name": res.method_name,
                "best_fitness": res.best_fitness,
                "iterations": res.iterations,
                "time_sec": res.time_sec,
                "status": res.status,
                "metrics": res.metrics,
                "history_len": len(res.history) if res.history else 0,
            },

            "llm_analysis_step6": {
                "json": step6,
                "response_text": analysis.get("text", ""),
                "raw": analysis.get("raw", None),
            },

            "backup": {
                "enabled": self.enable_backup,
                "backup_ran": backup_ran,
                "backup_method": backup_method_name,
                "backup_result": None if not backup_result else {
                    "method_name": backup_result.method_name,
                    "best_fitness": backup_result.best_fitness,
                    "iterations": backup_result.iterations,
                    "time_sec": backup_result.time_sec,
                    "status": backup_result.status,
                    "metrics": backup_result.metrics,
                },
            },

            "wall_time_sec": wall,
        }

        log_path = os.path.join(self.logs_dir, f"{tag}_seed{seed}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

        save_result_json(f"results/raw/{tag}_seed{seed}.json", res, extra=desc)

        # چاپ مثل قبل
        if verbose:
            print("DECISION:", log["decision"])
            print("RESULT:", log["run_result"])
            print("STEP6(perf):", step6["performance_assessment"])
            if backup_ran:
                print("BACKUP_RAN:", True, "| backup_method:", backup_method_name, "| backup_best:", backup_result.best_fitness)
            else:
                print("BACKUP_RAN:", False)

        return {
            "decision": log["decision"],
            "result": log["run_result"],
            "step6": step6,
            "backup_ran": backup_ran,
            "backup_method": backup_method_name,
            "backup_result": None if not backup_result else {
                "method_name": backup_result.method_name,
                "best_fitness": backup_result.best_fitness,
                "time_sec": backup_result.time_sec,
                "status": backup_result.status,
            },
            "log_path": log_path,
        }

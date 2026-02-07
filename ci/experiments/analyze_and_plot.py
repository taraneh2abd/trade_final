# analyze_and_plot.py

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

from src.evaluation.plots import Plotter


RAW_DIR = Path("results/raw")
FIG_ROOT = Path("results/figures")


# -------------------------------------------------
# Loading
# -------------------------------------------------
def load_all_results() -> List[dict]:
    results = []

    if not RAW_DIR.exists():
        print("❌ results/raw not found")
        return results

    for fp in RAW_DIR.glob("*.json"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            # اگر seed داخل فایل نبود از اسم دربیار
            if "seed" not in data:
                name = fp.stem
                if "seed" in name:
                    try:
                        data["seed"] = int(name.split("seed")[-1].split("_")[0])
                    except Exception:
                        data["seed"] = 0

            # بکاپ بودن
            if "is_backup" not in data:
                data["is_backup"] = "backup" in fp.stem.lower()

            results.append(data)

        except Exception as e:
            print(f"⚠️ Failed loading {fp}: {e}")

    print(f"✅ Loaded {len(results)} runs")
    return results


# -------------------------------------------------
# Group by problem
# -------------------------------------------------
def group_by_problem(results: List[dict]) -> Dict[str, List[dict]]:
    problems = defaultdict(list)

    for r in results:
        name = r.get("extra", {}).get("name", "unknown")
        problems[name].append(r)

    return problems


# -------------------------------------------------
# Main plotting routine
# -------------------------------------------------
def create_problem_level_outputs():
    all_results = load_all_results()
    if not all_results:
        print("❌ No results")
        return

    problems = group_by_problem(all_results)

    print(f"\n📦 Problems detected: {list(problems.keys())}")

    for prob_name, prob_results in problems.items():
        print(f"\n==============================")
        print(f"📊 Processing problem: {prob_name}")
        print(f"   runs: {len(prob_results)}")

        # پوشه مخصوص این مسئله
        problem_dir = FIG_ROOT / prob_name
        problem_dir.mkdir(parents=True, exist_ok=True)

        plotter = Plotter(output_dir=str(problem_dir))

        # فقط آنهایی که history دارند
        valid_results = [r for r in prob_results if r.get("history")]

        # ---------------------------------
        # 1) Convergence
        # ---------------------------------
        if valid_results:
            print("   → convergence")
            plotter.plot_all_seeds_convergence(
                results_data=valid_results,
                problem_name=prob_name,
                filename="all_seeds_convergence.png",
            )
        else:
            print("   ⚠️ no convergence data")

        # ---------------------------------
        # 2) Statistical comparison
        # ---------------------------------
        print("   → statistical comparison")
        plotter.plot_statistical_comparison(
            all_results=prob_results,
            filename="statistical_comparison.png",
        )

        # ---------------------------------
        # 3) Text report
        # ---------------------------------
        print("   → text report")
        plotter._create_text_report(prob_results)

    print("\n🎉 Done.")


# -------------------------------------------------
if __name__ == "__main__":
    create_problem_level_outputs()

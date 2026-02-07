# ci\experiments\analyze_and_plot.py

import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
from src.evaluation.plots import Plotter


def load_all_results(results_dir: str = "results/raw") -> List[Dict[str, Any]]:
    """
    بارگذاری تمام نتایج از فایل‌های JSON
    """
    all_results = []
    json_files = glob.glob(f"{results_dir}/*.json")
    
    if not json_files:
        print(f"⚠️ No JSON files found in {results_dir}")
        return []
    
    print(f"📂 Loading {len(json_files)} result files...")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['filename'] = Path(file_path).name
                
                # استخراج seed از نام فایل یا داده‌ها
                if 'seed' not in data:
                    # تلاش برای استخراج seed از نام فایل
                    import re
                    seed_match = re.search(r'seed_?(\d+)', file_path.lower())
                    if seed_match:
                        data['seed'] = int(seed_match.group(1))
                    else:
                        data['seed'] = 0
                
                # تشخیص backup methods
                if 'is_backup' not in data:
                    data['is_backup'] = 'backup' in file_path.lower() or 'reserve' in file_path.lower()
                
                all_results.append(data)
                
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    print(f"✅ Loaded {len(all_results)} results")
    return all_results


def create_targeted_plots():
    """
    ایجاد نمودارهای هدفمند درخواستی:
    1. نمودار همگرایی با تمام seedها و نوار اطمینان
    2. تحلیل آماری جامع برای تمام متدها
    """
    # بارگذاری نتایج
    all_results = load_all_results()
    
    if not all_results:
        print("❌ No results to analyze!")
        return
    
    # ایجاد دایرکتوری خروجی
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ایجاد تحلیل‌گر
    plotter = Plotter()
    
    print("\n" + "="*60)
    print("CREATING TARGETED PLOTS")
    print("="*60)
    
    # گروه‌بندی بر اساس مسئله
    problems = {}
    for result in all_results:
        prob_name = result.get("extra", {}).get("name", "unknown")
        if prob_name not in problems:
            problems[prob_name] = []
        problems[prob_name].append(result)
    
    print(f"📊 Found {len(problems)} distinct problems")
    
    # 1. ایجاد نمودار همگرایی برای هر مسئله
    print("\n📈 1. Creating convergence plots with confidence bands...")
    for prob_name, prob_results in problems.items():
        print(f"   Processing: {prob_name}")
        
        # فیلتر کردن نتایجی که تاریخچه دارند
        valid_results = [r for r in prob_results if r.get("history")]
        
        if len(valid_results) > 0:
            plotter.plot_all_seeds_convergence(
                results_data=valid_results,
                problem_name=prob_name,
                filename=f"all_seeds_convergence_{prob_name}.png"
            )
        else:
            print(f"   ⚠️ No convergence data for {prob_name}")
    
    # 2. ایجاد تحلیل آماری جامع
    print("\n📊 2. Creating comprehensive statistical comparison...")
    # plotter.plot_statistical_comparison(
    #     all_results=all_results,
    #     filename="statistical_comparison_all_methods.png"
    # )
    
    for prob_name, prob_results in problems.items():
        print(f"   Statistical analysis for: {prob_name}")
        
        plotter.plot_statistical_comparison(
            all_results=prob_results,
            filename=f"statistical_comparison_{prob_name}.png"
        )

    # 3. ایجاد گزارش متنی
    print("\n📝 3. Generating detailed report...")
    plotter._create_text_report(all_results)
    
    # 4. نمایش خلاصه نتایج
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    df = pd.DataFrame([
        {
            'Method': r.get("method_name", "unknown"),
            'Seed': r.get("seed", 0),
            'Fitness': float(r.get("best_fitness", 0)),
            'Time': float(r.get("time_sec", 0)),
            'Problem': r.get("extra", {}).get("name", "unknown"),
            'IsBackup': r.get("is_backup", False)
        }
        for r in all_results
    ])
    
    # محاسبه آمار خلاصه
    summary_stats = df.groupby('Method').agg({
        'Fitness': ['mean', 'std', 'min', 'max'],
        'Time': ['mean', 'std'],
        'Seed': 'count'
    }).round(4)
    
    print("\nSummary Statistics for All Methods:")
    print("-" * 80)
    print(summary_stats.to_string())
    
    # پیدا کردن بهترین روش
    if not df.empty:
        best_method_by_avg = df.groupby('Method')['Fitness'].mean().idxmin()
        best_method_by_median = df.groupby('Method')['Fitness'].median().idxmin()
        
        print(f"\n🎯 Best method by average fitness: {best_method_by_avg}")
        print(f"🎯 Best method by median fitness: {best_method_by_median}")
        
        # محاسبه درصد بهبود
        methods = df['Method'].unique()
        if len(methods) > 1:
            avg_values = df.groupby('Method')['Fitness'].mean()
            best_avg = avg_values.min()
            second_best = avg_values.nsmallest(2).iloc[-1]
            
            improvement = ((second_best - best_avg) / second_best) * 100
            print(f"📈 Improvement over second best: {improvement:.2f}%")
    
    print("\n✅ All plots and analysis completed!")
    print(f"📁 Output saved in: {output_dir.absolute()}")


def quick_analysis():
    """تحلیل سریع برای کاربرانی که می‌خواهند سریع نتایج را ببینند"""
    all_results = load_all_results()
    
    if not all_results:
        return
    
    # خلاصه سریع
    print("\n📋 QUICK ANALYSIS")
    print("-" * 40)
    
    methods = {}
    for result in all_results:
        method = result.get("method_name", "unknown")
        fitness = result.get("best_fitness", 0)
        time = result.get("time_sec", 0)
        
        if method not in methods:
            methods[method] = {'fitness': [], 'time': []}
        
        methods[method]['fitness'].append(fitness)
        methods[method]['time'].append(time)
    
    # نمایش خلاصه
    for method, data in methods.items():
        avg_fit = np.mean(data['fitness'])
        std_fit = np.std(data['fitness'])
        avg_time = np.mean(data['time'])
        
        print(f"\n{method}:")
        print(f"  Fitness: {avg_fit:.6f} ± {std_fit:.6f}")
        print(f"  Time: {avg_time:.2f}s")
        print(f"  Runs: {len(data['fitness'])}")


if __name__ == "__main__":
    print("🔬 Experimental Results Analysis")
    print("=" * 50)
    
    # ایجاد نمودارهای هدفمند
    create_targeted_plots()
    
    # نمایش تحلیل سریع
    quick_analysis()
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import os
import scipy.stats as stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


class Plotter:
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # تنظیمات عمومی
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10

    def save_fig(self, filename: str, dpi: int = 300, tight: bool = True):
        """ذخیره نمودار در فایل"""
        path = self.output_dir / filename
        if tight:
            plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f"📊 Plot saved: {path}")
        plt.close()

    def plot_all_seeds_convergence(
        self,
        results_data: List[Dict[str, Any]],
        problem_name: str,
        confidence_level: float = 0.95,
        filename: Optional[str] = None,
    ):
        """
        رسم نمودار همگرایی برای تمام seedها با نوار اطمینان (confidence bands)
        
        Args:
            results_data: لیست نتایج هر اجرا (seed)
            problem_name: نام مسئله
            confidence_level: سطح اطمینان برای نوارها (پیش‌فرض: 0.95)
            filename: نام فایل خروجی
        """
        plt.figure(figsize=(14, 8))
        
        # گروه‌بندی داده‌ها بر اساس متد
        methods_data = {}
        for result in results_data:
            method = result.get("method_name", "unknown")
            seed = result.get("seed", 0)
            history = result.get("history", [])
            
            if method not in methods_data:
                methods_data[method] = {}
            methods_data[method][seed] = history
        
        if not methods_data:
            print("⚠️ No convergence data found!")
            return
        
        # رنگ‌های متفاوت برای هر متد
        colors = plt.cm.tab20(np.linspace(0, 1, len(methods_data)))
        
        # تنظیم subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # رسم نمودار همگرایی با نوار اطمینان
        for idx, (method, seeds_data) in enumerate(methods_data.items()):
            color = colors[idx]
            
            # آماده‌سازی داده‌ها برای تمام seedها
            all_histories = []
            seed_colors = plt.cm.Set3(np.linspace(0, 1, len(seeds_data)))
            
            # رسم هر seed به صورت جداگانه
            for seed_idx, (seed, history) in enumerate(seeds_data.items()):
                if history:
                    seed_color = seed_colors[seed_idx]
                    ax1.plot(history, alpha=0.4, linewidth=0.8, 
                            color=seed_color, label=f'{method} (Seed {seed})' if idx == 0 else "")
            
            # محاسبه میانگین و نوار اطمینان برای هر متد
            histories_list = list(seeds_data.values())
            if histories_list and all(histories_list):
                # هم‌طول کردن تاریخچه‌ها
                max_len = max(len(h) for h in histories_list)
                padded_histories = []
                for h in histories_list:
                    if len(h) < max_len:
                        padded = h + [h[-1]] * (max_len - len(h))
                    else:
                        padded = h[:max_len]
                    padded_histories.append(padded)
                
                padded_histories = np.array(padded_histories)
                
                # محاسبه میانگین و انحراف معیار
                mean_history = np.mean(padded_histories, axis=0)
                std_history = np.std(padded_histories, axis=0)
                
                # محاسبه نوار اطمینان
                n_seeds = len(padded_histories)
                if n_seeds > 1:
                    # استفاده از توزیع t برای نمونه‌های کوچک
                    t_value = stats.t.ppf((1 + confidence_level) / 2, n_seeds - 1)
                    ci = t_value * std_history / np.sqrt(n_seeds)
                else:
                    ci = std_history
                
                # رسم میانگین و نوار اطمینان
                x = np.arange(len(mean_history))
                ax1.plot(x, mean_history, label=f'{method} (Mean)', 
                        color=color, linewidth=3, linestyle='-')
                
                # نوار اطمینان
                ax1.fill_between(x, mean_history - ci, mean_history + ci, 
                               alpha=0.2, color=color, label=f'{method} (95% CI)')
        
        # تنظیمات نمودار همگرایی
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Best Fitness', fontsize=12)
        ax1.set_title(f'All Seeds Convergence with Confidence Bands - {problem_name}', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', ncol=2, fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # نمودار دوم: مقایسه seedها در آخرین iteration
        last_values = []
        method_labels = []
        seed_labels = []
        
        for method, seeds_data in methods_data.items():
            for seed, history in seeds_data.items():
                if history:
                    last_values.append(history[-1])
                    method_labels.append(method)
                    seed_labels.append(f'Seed {seed}')
        
        # ایجاد DataFrame برای seaborn
        df_last = pd.DataFrame({
            'Method': method_labels,
            'Seed': seed_labels,
            'Final Fitness': last_values
        })
        
        # رسم boxplot برای آخرین مقادیر
        sns.boxplot(x='Method', y='Final Fitness', data=df_last, ax=ax2, palette='Set3')
        sns.stripplot(x='Method', y='Final Fitness', data=df_last, ax=ax2, 
                     color='black', alpha=0.5, size=6, jitter=True)
        
        ax2.set_xlabel('Method', fontsize=12)
        ax2.set_ylabel('Final Fitness', fontsize=12)
        ax2.set_title('Final Fitness Distribution Across Seeds', 
                     fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            self.save_fig(filename)
        else:
            self.save_fig(f"all_seeds_convergence_{problem_name}.png")

    def plot_statistical_comparison(
        self,
        all_results: List[Dict[str, Any]],
        filename: Optional[str] = None,
    ):
        """
        تحلیل و مقایسه آماری جامع برای تمام متدها و seedها
        
        این نمودار شامل:
        1. میانگین و انحراف معیار fitness
        2. میانگین زمان اجرا
        3. بهترین و بدترین نتایج
        4. آزمون‌های آماری
        5. رتبه‌بندی نهایی
        """
        # استخراج و سازماندهی داده‌ها
        data_records = []
        
        for result in all_results:
            method = result.get("method_name", "unknown")
            seed = result.get("seed", 0)
            fitness = result.get("best_fitness", 0)
            time_sec = result.get("time_sec", 0)
            problem = result.get("extra", {}).get("name", "unknown")
            is_backup = result.get("is_backup", False)
            
            data_records.append({
                'Method': method,
                'Seed': seed,
                'Fitness': float(fitness),
                'Time': float(time_sec),
                'Problem': problem,
                'IsBackup': is_backup,
                'History': result.get("history", [])
            })
        
        df = pd.DataFrame(data_records)
        
        if df.empty:
            print("⚠️ No data available for statistical comparison!")
            return
        
        # ایجاد نمودار جامع
        fig = plt.figure(figsize=(16, 14))
        
        # 1. میانگین fitness برای هر متد (شامل backupها)
        ax1 = plt.subplot(3, 3, 1)
        mean_fitness = df.groupby(['Method', 'IsBackup'])['Fitness'].mean().reset_index()
        colors = ['green' if not b else 'red' for b in mean_fitness['IsBackup']]
        bars = ax1.bar(range(len(mean_fitness)), mean_fitness['Fitness'], 
                      color=colors, edgecolor='black')
        
        # افزودن مقدار روی میله‌ها
        for i, (bar, val) in enumerate(zip(bars, mean_fitness['Fitness'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Average Fitness')
        ax1.set_title('Average Fitness by Method')
        ax1.set_xticks(range(len(mean_fitness)))
        ax1.set_xticklabels(mean_fitness['Method'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # ایجاد legend برای backupها
        backup_patch = Patch(color='red', label='Backup Methods')
        main_patch = Patch(color='green', label='Main Methods')
        ax1.legend(handles=[main_patch, backup_patch])
        
        # 2. میانگین زمان اجرا
        ax2 = plt.subplot(3, 3, 2)
        mean_time = df.groupby('Method')['Time'].mean().sort_values()
        bars2 = ax2.bar(range(len(mean_time)), mean_time.values, 
                       color='skyblue', edgecolor='black')
        
        for i, (bar, val) in enumerate(zip(bars2, mean_time.values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}s', ha='center', va='bottom', fontsize=8)
        
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Average Time (s)')
        ax2.set_title('Average Execution Time')
        ax2.set_xticks(range(len(mean_time)))
        ax2.set_xticklabels(mean_time.index, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. توزیع fitness برای هر متد (violin plot)
        ax3 = plt.subplot(3, 3, 3)
        methods_order = df.groupby('Method')['Fitness'].mean().sort_values().index
        violin_parts = ax3.violinplot([df[df['Method']==m]['Fitness'] 
                                     for m in methods_order], 
                                     showmeans=True, showmedians=True)
        
        # رنگ‌آمیزی violin plot
        for pc, color in zip(violin_parts['bodies'], plt.cm.Set3(np.linspace(0, 1, len(methods_order)))):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Fitness')
        ax3.set_title('Fitness Distribution (Violin Plot)')
        ax3.set_xticks(range(1, len(methods_order) + 1))
        ax3.set_xticklabels(methods_order, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. بهترین نتایج برای هر seed
        ax4 = plt.subplot(3, 3, 4)
        best_per_seed = df.loc[df.groupby('Seed')['Fitness'].idxmin()]
        best_counts = best_per_seed['Method'].value_counts()
        
        colors4 = plt.cm.Pastel1(np.linspace(0, 1, len(best_counts)))
        wedges, texts, autotexts = ax4.pie(best_counts.values, labels=best_counts.index,
                                          autopct='%1.1f%%', colors=colors4,
                                          startangle=90, textprops={'fontsize': 9})
        
        ax4.set_title('Best Method per Seed')
        
        # 5. مقایسه fitness و زمان (scatter plot)
        ax5 = plt.subplot(3, 3, 5)
        scatter_data = df.groupby('Method').agg({'Fitness': 'mean', 'Time': 'mean'}).reset_index()
        
        scatter = ax5.scatter(scatter_data['Time'], scatter_data['Fitness'], 
                             s=200, c=range(len(scatter_data)), 
                             cmap='viridis', edgecolor='black', alpha=0.7)
        
        # افزودن label برای هر نقطه
        for i, row in scatter_data.iterrows():
            ax5.annotate(row['Method'], (row['Time'], row['Fitness']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        ax5.set_xlabel('Average Time (s)')
        ax5.set_ylabel('Average Fitness')
        ax5.set_title('Fitness vs Time Trade-off')
        ax5.grid(True, alpha=0.3)
        
        # 6. رتبه‌بندی روش‌ها
        ax6 = plt.subplot(3, 3, 6)
        
        # محاسبه امتیاز برای هر متد
        method_scores = {}
        for method in df['Method'].unique():
            method_df = df[df['Method'] == method]
            
            # میانگین fitness (وزن بیشتر)
            avg_fitness = method_df['Fitness'].mean()
            
            # میانگین زمان (وزن کمتر)
            avg_time = method_df['Time'].mean()
            
            # ثبات (انحراف معیار کمتر بهتر است)
            fitness_std = method_df['Fitness'].std()
            
            # امتیاز ترکیبی
            score = (0.6 * (1 / avg_fitness if avg_fitness > 0 else 0) + 
                    0.2 * (1 / avg_time if avg_time > 0 else 0) + 
                    0.2 * (1 / (fitness_std + 1e-10)))
            
            method_scores[method] = score
        
        # مرتب‌سازی بر اساس امتیاز
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
        methods_ranked = [m[0] for m in sorted_methods]
        scores_ranked = [m[1] for m in sorted_methods]
        
        bars6 = ax6.barh(range(len(methods_ranked)), scores_ranked, 
                        color=plt.cm.RdYlGn(np.linspace(0, 1, len(methods_ranked))))
        
        ax6.set_yticks(range(len(methods_ranked)))
        ax6.set_yticklabels(methods_ranked)
        ax6.set_xlabel('Composite Score')
        ax6.set_title('Method Ranking (Higher is Better)')
        ax6.grid(True, alpha=0.3, axis='x')
        
        # 7. تحلیل آماری - آزمون t-test بین بهترین و سایر روش‌ها
        ax7 = plt.subplot(3, 3, (7, 9))
        
        if len(df['Method'].unique()) >= 2:
            best_method = methods_ranked[0]
            other_methods = methods_ranked[1:4] if len(methods_ranked) > 4 else methods_ranked[1:]
            
            comparisons = []
            p_values = []
            
            for other_method in other_methods:
                best_fitness = df[df['Method'] == best_method]['Fitness'].values
                other_fitness = df[df['Method'] == other_method]['Fitness'].values
                
                if len(best_fitness) > 1 and len(other_fitness) > 1:
                    # آزمون t-test مستقل
                    t_stat, p_value = stats.ttest_ind(best_fitness, other_fitness)
                    comparisons.append(f"{best_method} vs {other_method}")
                    p_values.append(p_value)
            
            if comparisons:
                # رسم نتایج آزمون
                x_pos = np.arange(len(comparisons))
                bars7 = ax7.bar(x_pos, p_values, color=['green' if p > 0.05 else 'red' 
                                                      for p in p_values])
                
                # خط آستانه معناداری
                ax7.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, 
                           label='Significance level (α=0.05)')
                
                ax7.set_xlabel('Method Comparison')
                ax7.set_ylabel('p-value')
                ax7.set_title('Statistical Significance (t-test)')
                ax7.set_xticks(x_pos)
                ax7.set_xticklabels(comparisons, rotation=45, ha='right')
                ax7.legend()
                ax7.grid(True, alpha=0.3, axis='y')
                
                # افزودن مقدار p-value روی میله‌ها
                for bar, p_val in zip(bars7, p_values):
                    height = bar.get_height()
                    ax7.text(bar.get_x() + bar.get_width()/2, height,
                            f'p={p_val:.4f}', ha='center', va='bottom', fontsize=8)
            else:
                ax7.text(0.5, 0.5, 'Insufficient data\nfor statistical tests',
                        ha='center', va='center', transform=ax7.transAxes,
                        fontsize=12, alpha=0.5)
        else:
            ax7.text(0.5, 0.5, 'Need at least 2 methods\nfor statistical comparison',
                    ha='center', va='center', transform=ax7.transAxes,
                    fontsize=12, alpha=0.5)
        
        # افزودن جدول خلاصه نتایج
        plt.figtext(0.02, 0.02, 
                   self._generate_summary_text(df, methods_ranked), 
                   fontsize=9, family='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", 
                           facecolor="lightgray", alpha=0.5))
        
        plt.suptitle('COMPREHENSIVE STATISTICAL ANALYSIS OF ALL METHODS', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if filename:
            self.save_fig(filename)
        else:
            self.save_fig("statistical_comparison_all_methods.png")

    def _generate_summary_text(self, df: pd.DataFrame, ranked_methods: List[str]) -> str:
        """تولید متن خلاصه نتایج"""
        summary_lines = ["=" * 60]
        summary_lines.append("FINAL ANALYSIS AND RECOMMENDATION")
        summary_lines.append("=" * 60)
        
        # جمع‌آوری آمار برای هر متد
        for i, method in enumerate(ranked_methods, 1):
            method_df = df[df['Method'] == method]
            
            avg_fitness = method_df['Fitness'].mean()
            std_fitness = method_df['Fitness'].std()
            avg_time = method_df['Time'].mean()
            is_backup = method_df['IsBackup'].iloc[0] if not method_df.empty else False
            
            summary_lines.append(f"\n{i}. {method} {'(BACKUP)' if is_backup else ''}")
            summary_lines.append(f"   Average Fitness: {avg_fitness:.6f}")
            summary_lines.append(f"   Std Dev: {std_fitness:.6f}")
            summary_lines.append(f"   Average Time: {avg_time:.2f}s")
            
            if i == 1:
                summary_lines.append(f"   → RECOMMENDED AS BEST METHOD")
        
        # نتیجه‌گیری کلی
        summary_lines.append("\n" + "=" * 60)
        summary_lines.append("KEY FINDINGS:")
        
        best_method = ranked_methods[0] if ranked_methods else "N/A"
        best_df = df[df['Method'] == best_method]
        
        if not best_df.empty:
            improvement = self._calculate_improvement(df, best_method)
            summary_lines.append(f"1. Best overall method: {best_method}")
            summary_lines.append(f"2. Average improvement: {improvement:.1f}% over other methods")
            
            # بررسی backup methods
            backup_methods = df[df['IsBackup']]['Method'].unique()
            if len(backup_methods) > 0:
                best_backup = None
                best_backup_score = float('-inf')
                
                for backup in backup_methods:
                    backup_df = df[df['Method'] == backup]
                    if not backup_df.empty:
                        score = backup_df['Fitness'].mean()
                        if score > best_backup_score:
                            best_backup_score = score
                            best_backup = backup
                
                if best_backup:
                    summary_lines.append(f"3. Best backup method: {best_backup}")
        
        summary_lines.append("=" * 60)
        
        return "\n".join(summary_lines)

    def _calculate_improvement(self, df: pd.DataFrame, best_method: str) -> float:
        """محاسبه درصد بهبود بهترین روش نسبت به میانگین سایر روش‌ها"""
        other_methods = [m for m in df['Method'].unique() if m != best_method]
        
        if not other_methods:
            return 0.0
        
        best_avg = df[df['Method'] == best_method]['Fitness'].mean()
        other_avg = df[df['Method'].isin(other_methods)]['Fitness'].mean()
        
        if other_avg == 0:
            return 0.0
        
        # فرض: fitness کمتر بهتر است (minimization)
        improvement = ((other_avg - best_avg) / other_avg) * 100
        return improvement

    def create_comprehensive_report(self, all_results: List[Dict[str, Any]]):
        """
        ایجاد گزارش جامع با تمام نمودارهای درخواستی
        """
        print("📊 Creating comprehensive analysis report...")
        
        # گروه‌بندی بر اساس مسئله
        problems = {}
        for result in all_results:
            prob_name = result.get("extra", {}).get("name", "unknown")
            if prob_name not in problems:
                problems[prob_name] = []
            problems[prob_name].append(result)
        
        # 1. برای هر مسئله، نمودار همگرایی با تمام seedها
        for prob_name, prob_results in problems.items():
            print(f"  📈 Creating convergence plot for: {prob_name}")
            self.plot_all_seeds_convergence(
                prob_results,
                prob_name,
                filename=f"all_seeds_convergence_{prob_name}.png"
            )
        
        # 2. تحلیل آماری جامع برای تمام متدها
        print("  📊 Creating statistical comparison plot...")
        self.plot_statistical_comparison(
            all_results,
            filename="statistical_comparison_all_methods.png"
        )
        
        # 3. ایجاد گزارش متنی
        self._create_text_report(all_results)
        
        print("✅ Comprehensive report created successfully!")

    def _create_text_report(self, all_results: List[Dict[str, Any]]):
        """ایجاد گزارش متنی از نتایج"""
        report_path = self.output_dir / "analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("EXPERIMENTAL RESULTS ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # جمع‌آوری داده‌ها
            df = pd.DataFrame([
                {
                    'Method': r.get("method_name", "unknown"),
                    'Seed': r.get("seed", 0),
                    'Fitness': r.get("best_fitness", 0),
                    'Time': r.get("time_sec", 0),
                    'Problem': r.get("extra", {}).get("name", "unknown"),
                    'IsBackup': r.get("is_backup", False)
                }
                for r in all_results
            ])
            
            # خلاصه کلی
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 40 + "\n")
            
            for method in df['Method'].unique():
                method_df = df[df['Method'] == method]
                f.write(f"\n{method}:\n")
                f.write(f"  Average Fitness: {method_df['Fitness'].mean():.6f}\n")
                f.write(f"  Std Deviation: {method_df['Fitness'].std():.6f}\n")
                f.write(f"  Average Time: {method_df['Time'].mean():.2f}s\n")
                f.write(f"  Number of runs: {len(method_df)}\n")
                if method_df['IsBackup'].iloc[0]:
                    f.write(f"  Type: Backup Method\n")
            
            # رتبه‌بندی
            f.write("\n\nMETHOD RANKING:\n")
            f.write("-" * 40 + "\n")
            
            ranking = df.groupby('Method')['Fitness'].mean().sort_values()
            for i, (method, score) in enumerate(ranking.items(), 1):
                f.write(f"{i}. {method}: {score:.6f}\n")
            
            # توصیه نهایی
            f.write("\n\nRECOMMENDATION:\n")
            f.write("-" * 40 + "\n")
            
            best_method = ranking.index[0]
            best_score = ranking.iloc[0]
            f.write(f"Best method: {best_method}\n")
            f.write(f"Average fitness: {best_score:.6f}\n")
            
            if len(ranking) > 1:
                second_best = ranking.iloc[1]
                improvement = ((second_best - best_score) / second_best) * 100
                f.write(f"Improvement over second best: {improvement:.1f}%\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"📝 Text report saved: {report_path}")


# تابع کمکی برای استفاده سریع
def create_comprehensive_analysis(results_dir: str = "results/raw"):
    """تابع سریع برای ایجاد تحلیل جامع"""
    import glob
    import json
    
    # بارگذاری تمام نتایج
    all_results = []
    json_files = glob.glob(f"{results_dir}/*.json")
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                all_results.append(data)
            except:
                print(f"⚠️ Error loading: {file_path}")
    
    if not all_results:
        print("⚠️ No results found!")
        return
    
    # ایجاد گزارش
    plotter = Plotter()
    plotter.create_comprehensive_report(all_results)
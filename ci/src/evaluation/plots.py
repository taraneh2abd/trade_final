# ci/src/evaluation/plot.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import os


class Plotter:
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # تنظیمات عمومی
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

    def save_fig(self, filename: str, dpi: int = 300, tight: bool = True):
        """ذخیره نمودار در فایل"""
        path = self.output_dir / filename
        if tight:
            plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f"📊 Plot saved: {path}")
        plt.close()

    def plot_convergence(
        self,
        histories: Dict[str, Union[List[float], List[List[float]]]],
        title: str = "Convergence Curves",
        xlabel: str = "Iteration",
        ylabel: str = "Best Fitness",
        log_y: bool = False,
        filename: Optional[str] = None,
        show_std: bool = True,
        show_legend: bool = True,
    ):
        """رسم نمودار همگرایی (میانگین + نوار خطا برای چندین اجرا)"""
        plt.figure(figsize=(10, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
        
        for (method, history_data), color in zip(histories.items(), colors):
            if isinstance(history_data[0], list):  # Multiple runs
                # محاسبه میانگین و انحراف معیار
                max_len = max(len(h) for h in history_data)
                padded = []
                for h in history_data:
                    if h:
                        padded.append(h + [h[-1]] * (max_len - len(h)))
                    else:
                        padded.append([0] * max_len)
                
                if padded:
                    padded = np.array(padded)
                    mean_hist = np.mean(padded, axis=0)
                    std_hist = np.std(padded, axis=0)
                    
                    x = np.arange(len(mean_hist))
                    plt.plot(x, mean_hist, label=method, color=color, linewidth=2)
                    
                    if show_std and len(history_data) > 1:
                        plt.fill_between(
                            x, 
                            mean_hist - std_hist, 
                            mean_hist + std_hist, 
                            alpha=0.2, 
                            color=color
                        )
            else:  # Single run
                plt.plot(history_data, label=method, color=color, linewidth=2)
        
        if log_y:
            plt.yscale('log')
            plt.ylabel(f"{ylabel} (log scale)")
        else:
            plt.ylabel(ylabel)
        
        plt.xlabel(xlabel)
        plt.title(title)
        
        if show_legend:
            plt.legend(loc='upper right')
        
        plt.grid(True, alpha=0.3)
        
        if filename:
            self.save_fig(filename)
        else:
            plt.show()

    def plot_box_comparison(
        self,
        results_dict: Dict[str, List[float]],
        title: str = "Method Comparison",
        ylabel: str = "Fitness",
        rotate_labels: bool = True,
        filename: Optional[str] = None,
        show_values: bool = True,
    ):
        """رسم boxplot برای مقایسه متدها"""
        plt.figure(figsize=(10, 6))
        
        methods = list(results_dict.keys())
        data = [results_dict[method] for method in methods]
        
        box = plt.boxplot(data, labels=methods, patch_artist=True)
        
        # رنگ‌آمیزی
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # تنظیمات محورها
        plt.ylabel(ylabel)
        plt.title(title)
        
        if rotate_labels:
            plt.xticks(rotation=45, ha='right')
        
        # نمایش مقادیر median
        if show_values:
            for i, line in enumerate(box['medians']):
                x, y = line.get_xydata()[1]  # نقطه وسط خط median
                plt.text(x, y, f'{y:.3f}', 
                        ha='center', va='bottom', 
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', 
                                facecolor='yellow', alpha=0.5))
        
        plt.grid(True, alpha=0.3, axis='y')
        
        if filename:
            self.save_fig(filename)
        else:
            plt.show()

    def plot_bar_metrics(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        metric_name: str = "best_fitness",
        title: str = "Performance Comparison",
        ylabel: Optional[str] = None,
        filename: Optional[str] = None,
        horizontal: bool = False,
    ):
        """رسم نمودار میله‌ای برای معیارهای مختلف"""
        plt.figure(figsize=(10, 6))
        
        methods = list(metrics_dict.keys())
        values = [metrics_dict[m].get(metric_name, 0) for m in methods]
        
        if horizontal:
            bars = plt.barh(methods, values, color='skyblue', edgecolor='black')
            plt.xlabel(ylabel or metric_name)
            plt.ylabel('Method')
            
            # اضافه کردن مقدار روی هر میله
            for bar in bars:
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2,
                        f'{width:.4f}', ha='left', va='center', fontsize=9)
        else:
            bars = plt.bar(methods, values, color='skyblue', edgecolor='black')
            plt.ylabel(ylabel or metric_name)
            plt.xlabel('Method')
            plt.xticks(rotation=45, ha='right')
            
            # اضافه کردن مقدار روی هر میله
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.title(title)
        plt.grid(True, alpha=0.3, axis='y' if not horizontal else 'x')
        
        if filename:
            self.save_fig(filename)
        else:
            plt.show()

    def plot_scatter_comparison(
        self,
        method1_results: List[float],
        method2_results: List[float],
        method1_name: str = "Method 1",
        method2_name: str = "Method 2",
        title: str = "Scatter Comparison",
        filename: Optional[str] = None,
    ):
        """رسم نمودار پراکندگی برای مقایسه دو متد"""
        plt.figure(figsize=(8, 8))
        
        min_val = min(min(method1_results), min(method2_results))
        max_val = max(max(method1_results), max(method2_results))
        
        plt.scatter(method1_results, method2_results, alpha=0.6, 
                   edgecolors='black', linewidth=0.5)
        
        # خط y=x برای مقایسه
        plt.plot([min_val, max_val], [min_val, max_val], 
                'r--', alpha=0.5, label='y=x')
        
        plt.xlabel(f"{method1_name} Fitness")
        plt.ylabel(f"{method2_name} Fitness")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if filename:
            self.save_fig(filename)
        else:
            plt.show()

    def plot_multiple_runs(
        self,
        all_histories: List[List[float]],
        labels: Optional[List[str]] = None,
        title: str = "Multiple Runs",
        xlabel: str = "Iteration",
        ylabel: str = "Fitness",
        filename: Optional[str] = None,
    ):
        """رسم چندین اجرا روی یک نمودار"""
        plt.figure(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_histories)))
        
        for i, history in enumerate(all_histories):
            label = labels[i] if labels and i < len(labels) else f"Run {i+1}"
            plt.plot(history, label=label, color=colors[i], alpha=0.7, linewidth=1.5)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if filename:
            self.save_fig(filename)
        else:
            plt.show()

    def load_and_plot_from_json(
        self,
        json_pattern: str = "results/raw/*.json",
        plot_type: str = "convergence",
        group_by: str = "method",
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        """بارگذاری نتایج از JSON و رسم نمودار"""
        import glob
        
        json_files = glob.glob(json_pattern)
        if not json_files:
            print(f"⚠️ No JSON files found matching: {json_pattern}")
            return
        
        data_by_group = {}
        
        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # گروه‌بندی بر اساس method_name یا problem
            if group_by == "method":
                key = data.get("method_name", "unknown")
            elif group_by == "problem":
                key = data.get("extra", {}).get("name", "unknown")
            else:
                key = "all"
            
            if key not in data_by_group:
                data_by_group[key] = []
            
            history = data.get("history", [])
            if history:
                data_by_group[key].append(history)
        
        if plot_type == "convergence":
            # برای هر گروه، میانگین بگیر
            avg_histories = {}
            for key, hist_list in data_by_group.items():
                if hist_list:
                    max_len = max(len(h) for h in hist_list)
                    padded = []
                    for h in hist_list:
                        if h:
                            padded.append(h + [h[-1]] * (max_len - len(h)))
                    
                    if padded:
                        avg_histories[key] = np.mean(padded, axis=0).tolist()
            
            if avg_histories:
                self.plot_convergence(
                    avg_histories,
                    title=title or f"Convergence ({group_by})",
                    filename=filename or f"convergence_{group_by}.png"
                )
        
        elif plot_type == "box":
            # مقادیر نهایی
            final_values = {}
            for key, hist_list in data_by_group.items():
                final_values[key] = [h[-1] if h else 0 for h in hist_list]
            
            if final_values:
                self.plot_box_comparison(
                    final_values,
                    title=title or f"Final Fitness ({group_by})",
                    filename=filename or f"boxplot_{group_by}.png"
                )


# تابع کمکی برای استفاده سریع
def quick_plot_convergence(histories: Dict[str, List[float]], 
                          save_path: Optional[str] = None, **kwargs):
    """تابع سریع برای رسم همگرایی"""
    plotter = Plotter()
    plotter.plot_convergence(histories, **kwargs)
    
    if save_path:
        plotter.save_fig(save_path)
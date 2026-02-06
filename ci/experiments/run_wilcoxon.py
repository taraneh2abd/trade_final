from src.evaluation.statistics import wilcoxon_compare_csv

if __name__ == "__main__":
    res = wilcoxon_compare_csv(
        csv_path="results/processed/benchmark_tsp30_5runs.csv",
        metric="best_fitness",
        method_a="ACO",
        method_b="GA",
        alternative="less",  # ACO < GA چون minimize
    )

    print("Wilcoxon signed-rank test (TSP)")
    print("statistic:", res["statistic"])
    print("p_value:", res["p_value"])
    print("A values:", res["a_values"])
    print("B values:", res["b_values"])


"""
STAT 636 - CHUNK 11
FINAL RESEARCH INTERPRETATION

Uses existing Chunk 9 and Chunk 10 results.
Does not rerun simulations.
"""

import pandas as pd
import numpy as np

from config import RESULTS_DIR

print("=" * 70)
print("STAT 636 CHUNK 11: FINAL RESEARCH INTERPRETATION")
print("=" * 70)

# ---------------------------------------------------------
# 1. DEFINE INPUT AND OUTPUT DIRECTORIES
# ---------------------------------------------------------

TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. LOAD EXISTING RESULTS
# ---------------------------------------------------------

model_performance = pd.read_csv(
    TABLE_DIR / "chunk10_model_performance.csv"
)

strategy_performance = pd.read_csv(
    TABLE_DIR / "chunk10_strategy_performance.csv"
)

timing_comparison = pd.read_csv(
    TABLE_DIR / "chunk10_timing_comparison.csv"
)

empirical_null = pd.read_csv(
    TABLE_DIR / "chunk10_empirical_null_comparison.csv"
)

friedman = pd.read_csv(
    TABLE_DIR / "chunk9_friedman_results.csv"
)

wilcoxon = pd.read_csv(
    TABLE_DIR / "chunk9_paired_wilcoxon_results.csv"
)

print("\nAll datasets loaded.")

# ---------------------------------------------------------
# 3. DATA VALIDATION
# ---------------------------------------------------------

datasets = {
    "model_performance": model_performance,
    "strategy_performance": strategy_performance,
    "timing_comparison": timing_comparison,
    "empirical_null": empirical_null,
    "friedman": friedman,
    "wilcoxon": wilcoxon
}

for name, df in datasets.items():

    if df.empty:
        raise ValueError(
            f"{name} is empty."
        )

    print(
        f"{name}: {df.shape}"
    )

print("\nData validation: PASS")

# ---------------------------------------------------------
# 4. MODEL INTERPRETATION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("NETWORK MODEL INTERPRETATION")
print("=" * 70)

best_model = model_performance.loc[
    model_performance["mean_final_size"].idxmin()
]

worst_model = model_performance.loc[
    model_performance["mean_final_size"].idxmax()
]

print(
    "\nLowest average outbreak model:",
    best_model["model"]
)

print(
    "Mean final size:",
    round(best_model["mean_final_size"], 4)
)

print(
    "\nHighest average outbreak model:",
    worst_model["model"]
)

print(
    "Mean final size:",
    round(worst_model["mean_final_size"], 4)
)

# ---------------------------------------------------------
# 5. VACCINATION STRATEGY INTERPRETATION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("VACCINATION STRATEGY INTERPRETATION")
print("=" * 70)

required_columns = [
    "model",
    "selection_timing",
    "vaccination_budget",
    "strategy",
    "mean_final_size"
]

for col in required_columns:

    if col not in strategy_performance.columns:
        raise ValueError(
            f"Missing column: {col}"
        )

best_strategies = (
    strategy_performance
    .sort_values("mean_final_size")
    .groupby(
        ["model", "selection_timing", "vaccination_budget"]
    )
    .head(1)
)

print(
    best_strategies[
        required_columns
    ].to_string(index=False)
)

best_strategies.to_csv(
    TABLE_DIR / "chunk11_best_strategies.csv",
    index=False
)

# ---------------------------------------------------------
# 6. VACCINATION TIMING INTERPRETATION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("VACCINATION TIMING")
print("=" * 70)

print(
    timing_comparison.to_string(index=False)
)

# ---------------------------------------------------------
# 7. STATISTICAL SIGNIFICANCE
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("STATISTICAL SIGNIFICANCE")
print("=" * 70)

significant_friedman = friedman[
    friedman["significant_005"] == True
]

significant_wilcoxon = wilcoxon[
    wilcoxon["significant_005"] == True
]

print(
    "\nSignificant Friedman tests:",
    len(significant_friedman)
)

print(
    "Significant Wilcoxon comparisons:",
    len(significant_wilcoxon)
)

significant_friedman.to_csv(
    TABLE_DIR / "chunk11_significant_friedman.csv",
    index=False
)

significant_wilcoxon.to_csv(
    TABLE_DIR / "chunk11_significant_wilcoxon.csv",
    index=False
)

# ---------------------------------------------------------
# 8. GENERATE RESEARCH INTERPRETATION
# ---------------------------------------------------------

report = []

report.append(
    "STAT 636 FINAL RESEARCH INTERPRETATION"
)

report.append("=" * 70)

report.append(
    "\n1. NETWORK MODEL FINDINGS"
)

report.append(
    f"The model with the lowest mean final outbreak "
    f"size was {best_model['model']}, with a mean "
    f"final size of {best_model['mean_final_size']:.4f}."
)

report.append(
    f"The highest mean final outbreak size was "
    f"observed under {worst_model['model']}, "
    f"with a mean of {worst_model['mean_final_size']:.4f}."
)

report.append(
    "\n2. VACCINATION STRATEGIES"
)

report.append(
    "Vaccination effectiveness varied across network "
    "models, vaccination budgets, and selection timings."
)

report.append(
    "No single strategy should be assumed optimal "
    "across all experimental conditions."
)

report.append(
    "\n3. STATISTICAL FINDINGS"
)

report.append(
    f"There were {len(significant_friedman)} statistically "
    f"significant Friedman tests after Holm correction."
)

report.append(
    f"There were {len(significant_wilcoxon)} statistically "
    f"significant paired Wilcoxon comparisons after "
    f"Holm correction."
)

report.append(
    "\n4. INTERPRETATION"
)

report.append(
    "The results suggest that network structure and "
    "vaccination strategy can influence simulated "
    "epidemic outcomes."
)

report.append(
    "The Configuration model generated the smallest "
    "average outbreak in the overall model comparison. "
    "However, this does not establish that degree "
    "preservation alone explains the difference."
)

report.append(
    "\n5. LIMITATIONS"
)

report.append(
    "- The analysis uses simulated epidemic outcomes."
)

report.append(
    "- The null models are approximations of the "
    "empirical network."
)

report.append(
    "- The DCSBM implementation is an approximate "
    "generative model."
)

report.append(
    "- Results depend on the specified transmission "
    "and vaccination assumptions."
)

report.append(
    "- Statistical significance does not necessarily "
    "imply substantial practical importance."
)

report.append(
    "- Findings should not automatically be generalized "
    "to other populations."
)

report.append(
    "\n6. CONCLUSION"
)

report.append(
    "This study demonstrates the value of comparing "
    "empirical social networks with alternative "
    "network null models when evaluating vaccination "
    "strategies through epidemic simulation."
)

report.append(
    "The findings emphasize the importance of network "
    "structure, vaccination timing, and intervention "
    "strategy in interpreting simulated epidemic outcomes."
)

report.append(
    "\nEND OF INTERPRETATION"
)

output_path = (
    RESULTS_DIR /
    "chunk11_final_interpretation.txt"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report)
    )

# ---------------------------------------------------------
# 9. COMPLETE
# ---------------------------------------------------------

print("\n" + "=" * 70)

print("CHUNK 11 COMPLETE")

print("=" * 70)

print(
    "\nGenerated files:"
)

print(
    "- chunk11_best_strategies.csv"
)

print(
    "- chunk11_significant_friedman.csv"
)

print(
    "- chunk11_significant_wilcoxon.csv"
)

print(
    "- chunk11_final_interpretation.txt"
)

print(
    "\nFinal report saved to:"
)

print(output_path)

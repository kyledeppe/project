
"""
STAT 636
CHUNK 10: FINAL RESULTS INTEGRATION

Integrates:
- Empirical network summaries
- Null model validation
- SIR simulation results
- Vaccination strategy comparisons
- Advanced statistical tests

Outputs:
- Integrated CSV tables
- Publication-quality figures
- Final research interpretation report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR

TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("STAT 636 FINAL RESULTS INTEGRATION")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_csv(filename):

    path = TABLE_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return pd.read_csv(path)


sir = load_csv(
    "null_model_sir_individual_results.csv"
)

summary = load_csv(
    "null_model_sir_summary.csv"
)

best = load_csv(
    "null_model_best_strategies.csv"
)

null_validation = load_csv(
    "null_model_validation.csv"
)

friedman = load_csv(
    "chunk9_friedman_results.csv"
)

wilcoxon = load_csv(
    "chunk9_paired_wilcoxon_results.csv"
)

print("\nAll datasets loaded successfully.")

print("SIR records:", len(sir))
print("Summary records:", len(summary))
print("Best strategy records:", len(best))
print("Friedman tests:", len(friedman))
print("Wilcoxon tests:", len(wilcoxon))


# ============================================================
# 2. VALIDATE DATA
# ============================================================

required_columns = [
    "model",
    "selection_timing",
    "strategy",
    "vaccination_budget",
    "replicate",
    "final_size",
    "peak_infected"
]

missing = [
    c for c in required_columns
    if c not in sir.columns
]

if missing:

    raise ValueError(
        f"Missing SIR columns: {missing}"
    )

if sir[required_columns].isnull().any().any():

    raise ValueError(
        "Missing values detected in required SIR columns."
    )

print("\nData validation: PASS")


# ============================================================
# 3. OVERALL MODEL PERFORMANCE
# ============================================================

model_performance = (

    sir.groupby("model")

    .agg(

        simulations=("final_size", "count"),

        mean_final_size=("final_size", "mean"),

        sd_final_size=("final_size", "std"),

        median_final_size=("final_size", "median"),

        mean_peak_infected=("peak_infected", "mean"),

        sd_peak_infected=("peak_infected", "std"),

        mean_generations=("generations", "mean")

    )

    .reset_index()

)

model_performance.to_csv(
    TABLE_DIR / "chunk10_model_performance.csv",
    index=False
)

print("\nOVERALL MODEL PERFORMANCE")
print(model_performance.to_string(index=False))


# ============================================================
# 4. VACCINATION STRATEGY PERFORMANCE
# ============================================================

strategy_performance = (

    sir.groupby(
        [
            "model",
            "selection_timing",
            "strategy",
            "vaccination_budget"
        ]
    )

    .agg(

        simulations=("final_size", "count"),

        mean_final_size=("final_size", "mean"),

        median_final_size=("final_size", "median"),

        mean_peak_infected=("peak_infected", "mean"),

        mean_final_proportion=(
            "final_size_proportion",
            "mean"
        ),

        mean_peak_proportion=(
            "peak_proportion",
            "mean"
        )

    )

    .reset_index()

)

strategy_performance.to_csv(
    TABLE_DIR / "chunk10_strategy_performance.csv",
    index=False
)


# ============================================================
# 5. COMPARE EMPIRICAL NETWORK AGAINST NULL MODELS
# ============================================================

empirical = strategy_performance[
    strategy_performance["model"] == "Empirical"
].copy()

nulls = strategy_performance[
    strategy_performance["model"] != "Empirical"
].copy()

comparison = nulls.merge(

    empirical,

    on=[
        "selection_timing",
        "strategy",
        "vaccination_budget"
    ],

    suffixes=("_null", "_empirical")

)

comparison["final_size_difference"] = (

    comparison["mean_final_size_empirical"]

    - comparison["mean_final_size_null"]

)

comparison["peak_difference"] = (

    comparison["mean_peak_infected_empirical"]

    - comparison["mean_peak_infected_null"]

)

comparison.to_csv(
    TABLE_DIR / "chunk10_empirical_null_comparison.csv",
    index=False
)

print("\nEMPIRICAL VS NULL MODEL COMPARISON")

print(
    comparison[
        [
            "selection_timing",
            "strategy",
            "vaccination_budget",
            "model_null",
            "final_size_difference",
            "peak_difference"
        ]
    ].head(15).to_string(index=False)
)


# ============================================================
# 6. PREVIOUS DAY VS SAME DAY
# ============================================================

timing = (

    sir.groupby(

        [
            "model",
            "strategy",
            "vaccination_budget",
            "selection_timing"
        ]

    )

    .agg(

        mean_final_size=("final_size", "mean"),

        mean_peak_infected=("peak_infected", "mean")

    )

    .reset_index()

)

timing_wide = timing.pivot(

    index=[
        "model",
        "strategy",
        "vaccination_budget"
    ],

    columns="selection_timing",

    values=[
        "mean_final_size",
        "mean_peak_infected"
    ]

).reset_index()

timing_wide.columns = [

    "_".join(
        str(x) for x in col if str(x)
    )

    if isinstance(col, tuple)

    else col

    for col in timing_wide.columns

]

if (
    "mean_final_size_Previous_Day" in timing_wide.columns
    and "mean_final_size_Same_Day" in timing_wide.columns
):

    timing_wide["timing_final_size_difference"] = (

        timing_wide["mean_final_size_Same_Day"]

        - timing_wide["mean_final_size_Previous_Day"]

    )

timing_wide.to_csv(
    TABLE_DIR / "chunk10_timing_comparison.csv",
    index=False
)


# ============================================================
# 7. IDENTIFY BEST VACCINATION STRATEGIES
# ============================================================

best_overall = (

    strategy_performance

    .sort_values("mean_final_size")

    .groupby(
        [
            "model",
            "selection_timing",
            "vaccination_budget"
        ]
    )

    .head(1)

    .sort_values(
        [
            "model",
            "selection_timing",
            "vaccination_budget"
        ]
    )

)

best_overall.to_csv(
    TABLE_DIR / "chunk10_best_strategies.csv",
    index=False
)

print("\nBEST VACCINATION STRATEGIES")

print(
    best_overall.to_string(index=False)
)


# ============================================================
# 8. STATISTICAL SIGNIFICANCE SUMMARY
# ============================================================

significant_friedman = friedman[
    friedman["holm_adjusted_p"] < 0.05
].copy()

significant_wilcoxon = wilcoxon[
    wilcoxon["holm_adjusted_p"] < 0.05
].copy()

print("\nSTATISTICAL SIGNIFICANCE")

print(
    "Significant Friedman tests:",
    len(significant_friedman)
)

print(
    "Significant Wilcoxon tests:",
    len(significant_wilcoxon)
)


# ============================================================
# 9. FIGURE 1: OVERALL NETWORK MODEL PERFORMANCE
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(

    model_performance["model"],

    model_performance["mean_final_size"],

    color=[
        "#264653",
        "#2A9D8F",
        "#E9C46A",
        "#E76F51"
    ]

)

ax.set_title(
    "Mean Final Outbreak Size by Network Model"
)

ax.set_xlabel("Network Model")

ax.set_ylabel("Mean Final Outbreak Size")

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "chunk10_model_performance.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. FIGURE 2: EMPIRICAL VS NULL MODELS
# ============================================================

plot_data = (

    strategy_performance

    .groupby(
        ["model", "vaccination_budget"]
    )

    ["mean_final_size"]

    .mean()

    .reset_index()

)

fig, ax = plt.subplots(figsize=(10, 6))

for model, group in plot_data.groupby("model"):

    group = group.sort_values("vaccination_budget")

    ax.plot(

        group["vaccination_budget"] * 100,

        group["mean_final_size"],

        marker="o",

        label=model

    )

ax.set_title(
    "Outbreak Size Across Vaccination Budgets"
)

ax.set_xlabel("Vaccination Budget (%)")

ax.set_ylabel("Mean Final Outbreak Size")

ax.legend()

ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "chunk10_vaccination_budget_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. FIGURE 3: VACCINATION TIMING
# ============================================================

timing_plot = (

    timing.groupby(
        ["selection_timing", "strategy"]
    )

    ["mean_final_size"]

    .mean()

    .reset_index()

)

fig, ax = plt.subplots(figsize=(11, 6))

for strategy, group in timing_plot.groupby("strategy"):

    ax.plot(

        group["selection_timing"],

        group["mean_final_size"],

        marker="o",

        label=strategy

    )

ax.set_title(
    "Previous-Day vs Same-Day Vaccination"
)

ax.set_xlabel("Selection Timing")

ax.set_ylabel("Mean Final Outbreak Size")

ax.legend()

ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "chunk10_vaccination_timing.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. FINAL REPORT
# ============================================================

report_path = (
    RESULTS_DIR / "chunk10_final_research_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STAT 636 FINAL RESEARCH RESULTS\n"
    )

    f.write("=" * 65 + "\n\n")

    f.write(
        "1. DATASET\n"
    )

    f.write(
        f"Total simulations: {len(sir)}\n"
    )

    f.write(
        f"Network models: {sir['model'].nunique()}\n"
    )

    f.write(
        f"Vaccination strategies: "
        f"{sir['strategy'].nunique()}\n\n"
    )

    f.write(
        "2. OVERALL MODEL PERFORMANCE\n"
    )

    f.write(
        model_performance.to_string(index=False)
    )

    f.write("\n\n")

    f.write(
        "3. STATISTICAL TESTING\n"
    )

    f.write(
        f"Friedman tests: {len(friedman)}\n"
    )

    f.write(
        f"Significant Friedman tests: "
        f"{len(significant_friedman)}\n"
    )

    f.write(
        f"Wilcoxon tests: {len(wilcoxon)}\n"
    )

    f.write(
        f"Significant Wilcoxon tests: "
        f"{len(significant_wilcoxon)}\n\n"
    )

    f.write(
        "4. BEST VACCINATION STRATEGIES\n"
    )

    f.write(
        best_overall.to_string(index=False)
    )

    f.write("\n\n")

    f.write(
        "5. INTERPRETATION\n"
    )

    f.write(
        "Results describe the simulated outcomes under "
        "the specified network and SIR assumptions. "
        "Statistical significance does not establish "
        "causality or guarantee real-world effectiveness. "
        "Effect sizes, uncertainty intervals, and "
        "network-model assumptions should be considered "
        "when interpreting findings.\n"
    )


# ============================================================
# 13. COMPLETION
# ============================================================

print("\n" + "=" * 70)

print("CHUNK 10 COMPLETE")

print("=" * 70)

print("\nGenerated tables:")

for filename in [

    "chunk10_model_performance.csv",

    "chunk10_strategy_performance.csv",

    "chunk10_empirical_null_comparison.csv",

    "chunk10_timing_comparison.csv",

    "chunk10_best_strategies.csv"

]:

    print(filename)


print("\nGenerated figures:")

for filename in [

    "chunk10_model_performance.png",

    "chunk10_vaccination_budget_comparison.png",

    "chunk10_vaccination_timing.png"

]:

    print(filename)


print("\nFinal report:")

print(report_path)

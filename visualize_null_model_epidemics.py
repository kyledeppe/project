# ============================================================
# STAT 636 PROJECT
# CHUNK 8D: NULL MODEL VISUALIZATION
# ============================================================

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from pathlib import Path

import config


# ------------------------------------------------------------
# DIRECTORIES
# ------------------------------------------------------------

TABLE_DIR = config.RESULTS_DIR / "tables"

FIGURE_DIR = config.RESULTS_DIR / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# PLOTTING CONFIGURATION
# ------------------------------------------------------------

sns.set_theme(
    style="whitegrid",
    context="notebook"
)

plt.rcParams["figure.dpi"] = 120

plt.rcParams["savefig.dpi"] = 300

plt.rcParams["font.family"] = "DejaVu Sans"


# ------------------------------------------------------------
# LOAD RESULTS
# ------------------------------------------------------------

summary_path = (
    TABLE_DIR / "null_model_sir_summary.csv"
)

individual_path = (
    TABLE_DIR / "null_model_sir_individual_results.csv"
)

summary = pd.read_csv(summary_path)

individual = pd.read_csv(individual_path)


# ------------------------------------------------------------
# VALIDATE INPUT DATA
# ------------------------------------------------------------

required_summary_columns = [
    "model",
    "selection_timing",
    "strategy",
    "vaccination_budget",
    "simulations",
    "mean_final_size",
    "final_ci_lower",
    "final_ci_upper",
    "mean_peak_infected",
    "peak_ci_lower",
    "peak_ci_upper"
]

required_individual_columns = [
    "model",
    "selection_timing",
    "strategy",
    "vaccination_budget",
    "replicate",
    "final_size",
    "peak_infected"
]

for column in required_summary_columns:

    if column not in summary.columns:
        raise ValueError(
            f"Missing summary column: {column}"
        )

for column in required_individual_columns:

    if column not in individual.columns:
        raise ValueError(
            f"Missing individual column: {column}"
        )


print("=" * 65)
print("STAT 636 NULL MODEL VISUALIZATION")
print("=" * 65)

print("Summary records:", len(summary))

print("Individual records:", len(individual))

print("Models:", summary["model"].unique())

print("Strategies:", summary["strategy"].unique())

print(
    "Vaccination budgets:",
    sorted(summary["vaccination_budget"].unique())
)

# ============================================================
# FIGURE GENERATION
# ============================================================

MODEL_ORDER = [
    "Empirical",
    "ER",
    "Configuration",
    "DCSBM"
]

STRATEGY_ORDER = [
    "Random",
    "Degree",
    "Weighted_Degree",
    "Betweenness",
    "Community_Bridge"
]

TIMING_ORDER = [
    "Previous_Day",
    "Same_Day"
]


def create_comparison_figures(
    data,
    mean_column,
    lower_column,
    upper_column,
    ylabel,
    filename_prefix,
    title
):

    for timing in TIMING_ORDER:

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(16, 11),
            sharey=True
        )

        axes = axes.flatten()

        for index, model in enumerate(MODEL_ORDER):

            ax = axes[index]

            model_data = data[
                (data["model"] == model) &
                (data["selection_timing"] == timing)
            ]

            for strategy in STRATEGY_ORDER:

                subset = model_data[
                    model_data["strategy"] == strategy
                ].sort_values(
                    "vaccination_budget"
                )

                if subset.empty:
                    continue

                x = subset[
                    "vaccination_budget"
                ].to_numpy()

                y = subset[
                    mean_column
                ].to_numpy()

                lower = subset[
                    lower_column
                ].to_numpy()

                upper = subset[
                    upper_column
                ].to_numpy()

                line = ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2,
                    markersize=5,
                    label=strategy
                )[0]

                ax.fill_between(
                    x,
                    lower,
                    upper,
                    color=line.get_color(),
                    alpha=0.12
                )

            ax.set_title(
                model,
                fontsize=13,
                fontweight="bold"
            )

            ax.set_xlabel(
                "Vaccination budget (%)"
            )

            ax.set_ylabel(ylabel)

            ax.legend(
                fontsize=8,
                loc="best"
            )

            ax.grid(
                alpha=0.25
            )

        fig.suptitle(
            f"{title}\n{timing.replace('_', ' ')}",
            fontsize=16,
            fontweight="bold"
        )

        plt.tight_layout()

        output_path = (
            FIGURE_DIR /
            f"{filename_prefix}_{timing.lower()}.png"
        )

        plt.savefig(
            output_path,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved figure:", output_path)


# ------------------------------------------------------------
# FINAL OUTBREAK SIZE
# ------------------------------------------------------------

create_comparison_figures(
    data=summary,
    mean_column="mean_final_size",
    lower_column="final_ci_lower",
    upper_column="final_ci_upper",
    ylabel="Mean final outbreak size",
    filename_prefix="null_model_final_outbreak",
    title="Vaccination Effectiveness Across Network Models"
)


# ------------------------------------------------------------
# PEAK INFECTIONS
# ------------------------------------------------------------

create_comparison_figures(
    data=summary,
    mean_column="mean_peak_infected",
    lower_column="peak_ci_lower",
    upper_column="peak_ci_upper",
    ylabel="Mean peak infected individuals",
    filename_prefix="null_model_peak_infections",
    title="Peak Infection Comparison Across Network Models"
)


print("\nFIGURE GENERATION COMPLETE")

# ============================================================
# STATISTICAL COMPARISONS
# ============================================================

global_results = []

pairwise_results = []


def holm_adjust(pvalues):

    pvalues = np.asarray(
        pvalues,
        dtype=float
    )

    n = len(pvalues)

    order = np.argsort(pvalues)

    adjusted = np.empty(n)

    running_max = 0.0

    for rank, index in enumerate(order):

        value = (
            (n - rank) *
            pvalues[index]
        )

        running_max = max(
            running_max,
            value
        )

        adjusted[index] = min(
            running_max,
            1.0
        )

    return adjusted


# ------------------------------------------------------------
# GROUP EXPERIMENTS
# ------------------------------------------------------------

group_columns = [
    "selection_timing",
    "strategy",
    "vaccination_budget"
]

grouped = individual.groupby(
    group_columns
)


for group_values, group_data in grouped:

    timing, strategy, budget = group_values

    model_samples = {}

    for model in MODEL_ORDER:

        values = group_data.loc[
            group_data["model"] == model,
            "final_size"
        ].dropna().to_numpy()

        if len(values) > 0:

            model_samples[model] = values

    # --------------------------------------------------------
    # KRUSKAL-WALLIS TEST
    # --------------------------------------------------------

    if len(model_samples) >= 2:

        test = stats.kruskal(
            *model_samples.values()
        )

        global_results.append({

            "selection_timing": timing,

            "strategy": strategy,

            "vaccination_budget": budget,

            "test": "Kruskal-Wallis",

            "statistic": test.statistic,

            "p_value": test.pvalue,

            "number_of_models": len(model_samples)

        })

    # --------------------------------------------------------
    # EMPIRICAL VS NULL MODELS
    # --------------------------------------------------------

    if "Empirical" not in model_samples:
        continue

    empirical = model_samples["Empirical"]

    comparisons = []

    for model in [
        "ER",
        "Configuration",
        "DCSBM"
    ]:

        if model not in model_samples:
            continue

        null_values = model_samples[model]

        test = stats.mannwhitneyu(
            empirical,
            null_values,
            alternative="two-sided"
        )

        comparisons.append({

            "model": model,

            "empirical_mean": np.mean(
                empirical
            ),

            "null_mean": np.mean(
                null_values
            ),

            "mean_difference": (
                np.mean(null_values) -
                np.mean(empirical)
            ),

            "empirical_median": np.median(
                empirical
            ),

            "null_median": np.median(
                null_values
            ),

            "statistic": test.statistic,

            "p_value": test.pvalue

        })

    if comparisons:

        adjusted = holm_adjust([
            item["p_value"]
            for item in comparisons
        ])

        for item, adjusted_p in zip(
            comparisons,
            adjusted
        ):

            pairwise_results.append({

                "selection_timing": timing,

                "strategy": strategy,

                "vaccination_budget": budget,

                **item,

                "holm_adjusted_p": adjusted_p,

                "significant_005": (
                    adjusted_p < 0.05
                )

            })


# ------------------------------------------------------------
# SAVE STATISTICAL RESULTS
# ------------------------------------------------------------

global_df = pd.DataFrame(
    global_results
)

pairwise_df = pd.DataFrame(
    pairwise_results
)

global_path = (
    TABLE_DIR /
    "null_model_kruskal_tests.csv"
)

pairwise_path = (
    TABLE_DIR /
    "null_model_pairwise_tests.csv"
)

global_df.to_csv(
    global_path,
    index=False
)

pairwise_df.to_csv(
    pairwise_path,
    index=False
)


print("\nSTATISTICAL TESTING COMPLETE")

print("\nKruskal-Wallis results:")
print(global_df.head(10).to_string(index=False))

print("\nPairwise comparison results:")
print(pairwise_df.head(10).to_string(index=False))

print("\nSignificant pairwise comparisons:")

print(
    pairwise_df[
        pairwise_df["significant_005"]
    ].shape[0]
)

print("\nSaved:")
print(global_path)
print(pairwise_path)
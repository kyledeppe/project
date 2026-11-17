
"""
STAT 636 - CHUNK 12
FINAL RESEARCH REPORT GENERATION

Modular implementation.

Section 1:
Imports, configuration, and output directories.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

from config import PROJECT_DIR, RESULTS_DIR

PROJECT_DIR = Path(PROJECT_DIR)
RESULTS_DIR = Path(RESULTS_DIR)

TABLE_DIR = RESULTS_DIR / "tables"

CHUNK12_DIR = RESULTS_DIR / "chunk12_final_report"

CHUNK12_TABLES = CHUNK12_DIR / "tables"

CHUNK12_FIGURES = CHUNK12_DIR / "figures"

# Create output directories.

for directory in [
    CHUNK12_DIR,
    CHUNK12_TABLES,
    CHUNK12_FIGURES
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

# ============================================================
# 2. VISUALIZATION SETTINGS
# ============================================================

sns.set_theme(
    style="whitegrid",
    context="paper"
)

plt.rcParams.update({

    "figure.dpi": 150,

    "savefig.dpi": 300,

    "font.size": 10,

    "axes.titlesize": 12,

    "axes.labelsize": 10

})

# ============================================================
# 3. INITIALIZATION
# ============================================================

print("=" * 70)

print(
    "STAT 636: CHUNK 12 FINAL RESEARCH REPORT"
)

print("=" * 70)

print("\nProject directory:")
print(PROJECT_DIR)

print("\nResults directory:")
print(RESULTS_DIR)

print("\nChunk 12 output directory:")
print(CHUNK12_DIR)

print("\nInitialization complete.")


# ============================================================
# SECTION 2: LOAD EXISTING DATASETS
# ============================================================

print("\n" + "=" * 70)

print("SECTION 2: LOADING RESEARCH DATASETS")

print("=" * 70)


def load_csv(filename):

    path = TABLE_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"Required dataset not found: {path}"
        )

    df = pd.read_csv(path)

    print(
        f"Loaded {filename}: {df.shape}"
    )

    return df


# Load Chunk 10 datasets.

model_performance = load_csv(
    "chunk10_model_performance.csv"
)

strategy_performance = load_csv(
    "chunk10_strategy_performance.csv"
)

timing_comparison = load_csv(
    "chunk10_timing_comparison.csv"
)

empirical_null = load_csv(
    "chunk10_empirical_null_comparison.csv"
)


# Load Chunk 11 datasets.

best_strategies = load_csv(
    "chunk11_best_strategies.csv"
)


# Load Chunk 9 statistical results.

friedman = load_csv(
    "chunk9_friedman_results.csv"
)

wilcoxon = load_csv(
    "chunk9_paired_wilcoxon_results.csv"
)


print("\nAll research datasets loaded successfully.")

print("SIR-related strategy records:", len(strategy_performance))

print("Friedman tests:", len(friedman))

print("Wilcoxon tests:", len(wilcoxon))



# ============================================================
# SECTION 3: DATA VALIDATION
# ============================================================

print("\n" + "=" * 70)

print("SECTION 3: DATA VALIDATION")

print("=" * 70)


datasets = {

    "model_performance": model_performance,

    "strategy_performance": strategy_performance,

    "timing_comparison": timing_comparison,

    "empirical_null": empirical_null,

    "best_strategies": best_strategies,

    "friedman": friedman,

    "wilcoxon": wilcoxon

}


for name, df in datasets.items():

    if df.empty:

        raise ValueError(
            f"{name} contains no observations."
        )

    print(
        f"{name}: "
        f"{len(df)} records | "
        f"Missing: {df.isna().sum().sum()}"
    )


# Verify expected network models.

expected_models = {

    "Empirical",

    "ER",

    "Configuration",

    "DCSBM"

}


observed_models = set(
    model_performance["model"].astype(str)
)


if not expected_models.issubset(observed_models):

    raise ValueError(
        "Expected network models are missing."
    )


print("\nData validation: PASS")


# ============================================================
# NETWORK MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("OVERALL NETWORK MODEL PERFORMANCE")

print("=" * 70)


model_performance = model_performance.sort_values(
    "mean_final_size"
)


print(
    model_performance.to_string(index=False)
)


lowest_model = model_performance.iloc[0]

highest_model = model_performance.iloc[-1]


print("\nLowest outbreak model:")

print(lowest_model["model"])

print(
    "Mean final size:",
    round(lowest_model["mean_final_size"], 4)
)


print("\nHighest outbreak model:")

print(highest_model["model"])

print(
    "Mean final size:",
    round(highest_model["mean_final_size"], 4)
)


# ============================================================
# VACCINATION STRATEGY SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("BEST VACCINATION STRATEGIES")

print("=" * 70)


best_strategies = best_strategies.sort_values(
    "mean_final_size"
)


print(
    best_strategies.to_string(index=False)
)



# ============================================================
# SECTION 4: STATISTICAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)

print("SECTION 4: STATISTICAL ANALYSIS")

print("=" * 70)


# ============================================================
# FRIEDMAN TESTS
# ============================================================

significant_friedman = friedman[
    friedman["significant_005"] == True
].copy()


print("\nFriedman tests:")

print("Total:", len(friedman))

print(
    "Significant after Holm adjustment:",
    len(significant_friedman)
)


if len(significant_friedman) > 0:

    print(
        significant_friedman.to_string(index=False)
    )


# ============================================================
# PAIRED WILCOXON TESTS
# ============================================================

significant_wilcoxon = wilcoxon[
    wilcoxon["significant_005"] == True
].copy()


print("\nPaired Wilcoxon tests:")

print("Total:", len(wilcoxon))

print(
    "Significant after Holm adjustment:",
    len(significant_wilcoxon)
)


# ============================================================
# VACCINATION TIMING
# ============================================================

timing_comparison = timing_comparison.copy()


timing_comparison["timing_final_size_difference"] = (

    timing_comparison[
        "mean_final_size_Same_Day"
    ]

    -

    timing_comparison[
        "mean_final_size_Previous_Day"
    ]

)


mean_timing_difference = (

    timing_comparison[
        "timing_final_size_difference"
    ].mean()

)


print("\nAverage Same-Day minus Previous-Day difference:")

print(
    round(mean_timing_difference, 4)
)


print("\nInterpretation:")

if mean_timing_difference < 0:

    print(
        "Same-Day selection had a lower average "
        "final outbreak size across the analyzed conditions."
    )

elif mean_timing_difference > 0:

    print(
        "Previous-Day selection had a lower average "
        "final outbreak size across the analyzed conditions."
    )

else:

    print(
        "Both timings had the same average final outbreak size."
    )



# ============================================================
# SECTION 5: EXPORT FINAL TABLES
# ============================================================

print("\n" + "=" * 70)

print("SECTION 5: EXPORTING RESEARCH TABLES")

print("=" * 70)


tables_to_save = {

    "final_model_performance.csv":
        model_performance,

    "final_strategy_performance.csv":
        strategy_performance,

    "final_timing_comparison.csv":
        timing_comparison,

    "final_empirical_null_comparison.csv":
        empirical_null,

    "final_friedman_tests.csv":
        friedman,

    "final_wilcoxon_tests.csv":
        wilcoxon,

    "significant_friedman_results.csv":
        significant_friedman,

    "significant_wilcoxon_results.csv":
        significant_wilcoxon,

}


# Determine best strategy within each condition.

best_by_model = (

    strategy_performance

    .sort_values("mean_final_size")

    .groupby(
        [
            "model",
            "selection_timing",
            "vaccination_budget"
        ]
    )

    .first()

    .reset_index()

)


tables_to_save[
    "best_strategy_by_condition.csv"
] = best_by_model


for filename, df in tables_to_save.items():

    output_path = CHUNK12_TABLES / filename

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {filename}"
    )


print("\nAll final research tables exported.")



# ============================================================
# SECTION 6: GENERATE FINAL FIGURES
# ============================================================

print("\n" + "=" * 70)

print("SECTION 6: GENERATING FINAL FIGURES")

print("=" * 70)


# ------------------------------------------------------------
# FIGURE 1: NETWORK MODEL COMPARISON
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))


sns.barplot(
    data=model_performance,
    x="model",
    y="mean_final_size",
    hue="model",
    legend=False,
    palette="viridis",
    ax=ax
)


ax.set_title(
    "Average Epidemic Final Size by Network Model"
)

ax.set_xlabel("Network Model")

ax.set_ylabel("Mean Final Outbreak Size")


plt.tight_layout()


plt.savefig(
    CHUNK12_FIGURES / "figure1_model_comparison.png"
)

plt.close()


# ------------------------------------------------------------
# FIGURE 2: VACCINATION BUDGET
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))


sns.lineplot(
    data=strategy_performance,
    x="vaccination_budget",
    y="mean_final_size",
    hue="strategy",
    style="selection_timing",
    markers=True,
    dashes=True,
    errorbar=None,
    ax=ax
)


ax.set_title(
    "Vaccination Budget and Epidemic Final Size"
)

ax.set_xlabel("Vaccination Budget")

ax.set_ylabel("Mean Final Outbreak Size")


plt.tight_layout()


plt.savefig(
    CHUNK12_FIGURES / "figure2_vaccination_budget.png"
)

plt.close()


# ------------------------------------------------------------
# FIGURE 3: VACCINATION TIMING
# ------------------------------------------------------------

timing_plot = timing_comparison.melt(

    id_vars=[
        "model",
        "strategy",
        "vaccination_budget"
    ],

    value_vars=[
        "mean_final_size_Previous_Day",
        "mean_final_size_Same_Day"
    ],

    var_name="timing",

    value_name="mean_final_size"

)


timing_plot["timing"] = (

    timing_plot["timing"]

    .str.replace(
        "mean_final_size_",
        "",
        regex=False
    )

)


fig, ax = plt.subplots(figsize=(10, 6))


sns.barplot(
    data=timing_plot,
    x="timing",
    y="mean_final_size",
    hue="model",
    errorbar=None,
    ax=ax
)


ax.set_title(
    "Previous-Day versus Same-Day Vaccination"
)

ax.set_xlabel("Vaccination Selection Timing")

ax.set_ylabel("Mean Final Outbreak Size")


plt.tight_layout()


plt.savefig(
    CHUNK12_FIGURES / "figure3_vaccination_timing.png"
)

plt.close()


# ------------------------------------------------------------
# FIGURE 4: STATISTICAL SIGNIFICANCE
# ------------------------------------------------------------

significance_counts = pd.DataFrame({

    "Test": [
        "Friedman",
        "Paired Wilcoxon"
    ],

    "Significant": [
        len(significant_friedman),
        len(significant_wilcoxon)
    ],

    "Not Significant": [
        len(friedman) - len(significant_friedman),
        len(wilcoxon) - len(significant_wilcoxon)
    ]

})


fig, ax = plt.subplots(figsize=(9, 5))


significance_counts.set_index("Test").plot(

    kind="bar",

    stacked=True,

    color=[
        "#2E8B57",
        "#D3D3D3"
    ],

    ax=ax

)


ax.set_title(
    "Statistical Significance After Holm Adjustment"
)

ax.set_xlabel("Statistical Test")

ax.set_ylabel("Number of Tests")


ax.legend(
    ["Significant", "Not Significant"]
)


plt.xticks(rotation=0)

plt.tight_layout()


plt.savefig(
    CHUNK12_FIGURES / "figure4_statistical_significance.png"
)

plt.close()


print("\nFour research figures generated successfully.")



# ============================================================
# SECTION 7: GENERATE RESEARCH DOCUMENTS
# ============================================================

print("\n" + "=" * 70)

print("SECTION 7: GENERATING RESEARCH DOCUMENTS")

print("=" * 70)


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

executive_summary = f"""
STAT 636
EXECUTIVE SUMMARY

Research Topic:
Network Structure and Vaccination Strategies in Epidemic
Spread on Weighted Social Networks.

NETWORK MODEL FINDINGS

The Configuration model demonstrated the lowest average
epidemic final size.

Mean final outbreak size:
{lowest_model['mean_final_size']:.4f}

The {highest_model['model']} model demonstrated the highest
average epidemic final size.

Mean final outbreak size:
{highest_model['mean_final_size']:.4f}

VACCINATION STRATEGIES

Five vaccination strategies were investigated:

1. Random
2. Degree
3. Weighted Degree
4. Betweenness
5. Community Bridge

Vaccination budgets ranged from 5% to 20%.

VACCINATION TIMING

Average Same-Day minus Previous-Day final-size difference:

{mean_timing_difference:.4f}

STATISTICAL FINDINGS

Significant Friedman tests after Holm adjustment:
{len(significant_friedman)}

Significant paired Wilcoxon comparisons after Holm adjustment:
{len(significant_wilcoxon)}

CONCLUSION

The findings suggest that network structure, vaccination
strategy, vaccination budget, and selection timing all
deserve consideration when evaluating epidemic interventions.

END OF EXECUTIVE SUMMARY
"""


(CHUNK12_DIR / "executive_summary.txt").write_text(
    executive_summary,
    encoding="utf-8"
)


# ============================================================
# METHODS AND RESULTS
# ============================================================

methods_results = f"""
STAT 636
METHODS AND RESULTS

1. RESEARCH DESIGN

The research examined epidemic propagation across
weighted social networks and evaluated alternative
vaccination strategies.

2. NETWORK MODELS

Four network structures were investigated:

- Empirical
- Erdős-Rényi
- Configuration
- Degree-Corrected Stochastic Block Model

3. EPIDEMIC SIMULATION

An SIR framework was used to simulate epidemic spread.

The outcomes included:

- Final outbreak size
- Final outbreak proportion
- Peak infected individuals
- Peak infection proportion
- Number of generations

4. VACCINATION STRATEGIES

Five vaccination strategies were examined:

- Random
- Degree
- Weighted Degree
- Betweenness
- Community Bridge

Vaccination budgets were 5%, 10%, 15%, and 20%.

5. VACCINATION TIMING

Previous-Day and Same-Day selection timings were compared.

6. NETWORK MODEL RESULTS

Lowest mean final outbreak size:

{lowest_model['model']}: {lowest_model['mean_final_size']:.4f}

Highest mean final outbreak size:

{highest_model['model']}: {highest_model['mean_final_size']:.4f}

7. STATISTICAL ANALYSIS

Friedman tests evaluated differences across paired
network-model conditions.

Paired Wilcoxon signed-rank tests evaluated empirical
versus null-model comparisons.

Holm adjustment was applied for multiple comparisons.

Significant Friedman tests:
{len(significant_friedman)}

Significant Wilcoxon tests:
{len(significant_wilcoxon)}

END OF METHODS AND RESULTS
"""


(CHUNK12_DIR / "methods_and_results.txt").write_text(
    methods_results,
    encoding="utf-8"
)


# ============================================================
# FINAL RESEARCH CONCLUSIONS
# ============================================================

conclusions = f"""
STAT 636
FINAL RESEARCH CONCLUSIONS

1. NETWORK STRUCTURE

The empirical network produced the highest average
final outbreak size, while the Configuration model
produced the lowest.

This suggests that preserving degree structure alone
may not fully reproduce epidemic behavior observed
in the empirical network.

2. VACCINATION EFFECTIVENESS

Vaccination effectiveness varied across network models,
vaccination strategies, budgets, and selection timings.

No single strategy should be considered universally
optimal based on these results.

3. VACCINATION TIMING

Same-Day vaccination sometimes produced lower average
final outbreak sizes, although exceptions were observed.

4. STATISTICAL EVIDENCE

Significant Friedman tests:
{len(significant_friedman)}

Significant paired Wilcoxon comparisons:
{len(significant_wilcoxon)}

5. LIMITATIONS

The conclusions are conditional on the simulated
network structures and SIR assumptions.

Future research should investigate:

- Alternative infection and recovery probabilities.
- Additional epidemic models.
- Larger and more diverse network datasets.
- Alternative community detection methods.
- Robustness across additional random seeds.
- Sensitivity to vaccination eligibility and seed selection.

6. FINAL CONCLUSION

Network topology and vaccination design both matter
when evaluating epidemic interventions.

The empirical network and its null models produced
different epidemic outcomes, demonstrating the
importance of network structure in epidemic modeling.

END OF RESEARCH CONCLUSIONS
"""


(CHUNK12_DIR / "research_conclusions.txt").write_text(
    conclusions,
    encoding="utf-8"
)


print("\nResearch documents generated successfully.")



# ============================================================
# SECTION 8: FINAL COMBINED REPORT
# ============================================================

print("\n" + "=" * 70)

print("SECTION 8: FINAL REPORT INTEGRATION")

print("=" * 70)


combined_report = (

    "=" * 70

    + "\nSTAT 636 FINAL RESEARCH REPORT\n"

    + "=" * 70

    + "\n\n"

    + executive_summary

    + "\n\n"

    + methods_results

    + "\n\n"

    + conclusions

)


report_path = (
    CHUNK12_DIR / "STAT636_FINAL_RESEARCH_REPORT.txt"
)


report_path.write_text(
    combined_report,
    encoding="utf-8"
)


print("Final research report saved.")


# ============================================================
# FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("STAT 636 CHUNK 12 COMPLETE")

print("=" * 70)


print("\nResearch findings:")

print(
    f"Lowest outbreak model: "
    f"{lowest_model['model']} "
    f"({lowest_model['mean_final_size']:.4f})"
)


print(
    f"Highest outbreak model: "
    f"{highest_model['model']} "
    f"({highest_model['mean_final_size']:.4f})"
)


print(
    "Significant Friedman tests:",
    len(significant_friedman)
)


print(
    "Significant Wilcoxon tests:",
    len(significant_wilcoxon)
)


print("\nFinal report:")

print(report_path)


print("\nGenerated figures:")

for path in sorted(CHUNK12_FIGURES.glob("*.png")):

    print(path.name)


print("\nGenerated tables:")

for path in sorted(CHUNK12_TABLES.glob("*.csv")):

    print(path.name)


print("\n" + "=" * 70)

print("STAT 636 CHUNK 12 COMPLETE")

print("=" * 70)

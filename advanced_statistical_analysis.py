
"""
STAT 636 PROJECT
CHUNK 9A: ADVANCED STATISTICAL ANALYSIS

Uses existing epidemic simulation results.
Does not rerun simulations.

Analyses:
    1. Experimental pairing diagnostics
    2. Friedman tests
    3. Paired Wilcoxon tests
    4. Holm corrections
    5. Effect sizes
    6. Bootstrap confidence intervals
"""

import numpy as np
import pandas as pd

from scipy.stats import (
    friedmanchisquare,
    wilcoxon
)

from statsmodels.stats.multitest import multipletests

import config


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    config.RESULTS_DIR
    / "tables"
    / "null_model_sir_individual_results.csv"
)

OUTPUT_DIR = (
    config.RESULTS_DIR
    / "tables"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELS = [
    "Empirical",
    "ER",
    "Configuration",
    "DCSBM"
]

OUTCOMES = [
    "final_size",
    "peak_infected"
]

BOOTSTRAP_REPLICATES = 5000

RANDOM_SEED = 636

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("STAT 636 ADVANCED STATISTICAL ANALYSIS")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

print("\nLoaded individual simulation results.")

print("Records:", len(df))

print("Models:", df["model"].unique())

print("Strategies:", df["strategy"].unique())

print(
    "Selection timings:",
    df["selection_timing"].unique()
)


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = [
    "model",
    "selection_timing",
    "strategy",
    "vaccination_budget",
    "replicate",
    "initial_infected",
    "final_size",
    "peak_infected"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values detected in required columns."
    )

if not set(MODELS).issubset(set(df["model"].unique())):
    raise ValueError(
        "One or more expected network models are missing."
    )

print("\nData validation: PASS")


# ============================================================
# DEFINE EXPERIMENTAL CONDITION
# ============================================================

CONDITION_COLUMNS = [
    "selection_timing",
    "strategy",
    "vaccination_budget"
]

# Replicate numbers alone do not establish pairing.
# The initial infection seed must also match.

PAIRING_COLUMNS = [
    "replicate",
    "initial_infected"
]


# ============================================================
# PAIRING DIAGNOSTICS
# ============================================================

print("\n" + "=" * 65)
print("PAIRING DIAGNOSTICS")
print("=" * 65)

pairing_results = []

for condition, group in df.groupby(
    CONDITION_COLUMNS,
    observed=True
):

    pivot = group.pivot_table(
        index=PAIRING_COLUMNS,
        columns="model",
        values="final_size",
        aggfunc="size"
    )

    if all(model in pivot.columns for model in MODELS):

        complete = pivot[MODELS].notna().all(axis=1)

        paired_count = int(complete.sum())

    else:

        paired_count = 0

    pairing_results.append({

        "selection_timing": condition[0],

        "strategy": condition[1],

        "vaccination_budget": condition[2],

        "paired_replicates": paired_count,

        "pairing_complete": (
            paired_count == len(pivot)
        )

    })

pairing_df = pd.DataFrame(pairing_results)

pairing_df.to_csv(
    OUTPUT_DIR / "chunk9_pairing_diagnostics.csv",
    index=False
)

print(pairing_df.head(10).to_string(index=False))

print(
    "\nSaved:",
    "chunk9_pairing_diagnostics.csv"
)


# ============================================================
# HELPER: HOLM CORRECTION
# ============================================================

def apply_holm(results, p_column="p_value"):

    results = results.copy()

    results["holm_adjusted_p"] = np.nan

    results["significant_005"] = False

    valid = results[p_column].notna()

    if valid.sum() > 0:

        adjusted = multipletests(
            results.loc[valid, p_column],
            method="holm"
        )

        results.loc[
            valid,
            "holm_adjusted_p"
        ] = adjusted[1]

        results.loc[
            valid,
            "significant_005"
        ] = adjusted[0]

    return results


# ============================================================
# HELPER: BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_mean_difference(
    differences,
    n_boot=BOOTSTRAP_REPLICATES
):

    differences = np.asarray(
        differences,
        dtype=float
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) == 0:

        return np.nan, np.nan

    boot_means = np.empty(n_boot)

    for b in range(n_boot):

        sample = rng.choice(
            differences,
            size=len(differences),
            replace=True
        )

        boot_means[b] = sample.mean()

    lower, upper = np.percentile(
        boot_means,
        [2.5, 97.5]
    )

    return float(lower), float(upper)


# ============================================================
# HELPER: PAIRED EFFECT SIZE
# ============================================================

def paired_effect_size(differences):

    differences = np.asarray(
        differences,
        dtype=float
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) < 2:

        return np.nan

    sd = np.std(
        differences,
        ddof=1
    )

    if sd == 0:

        return np.nan

    return float(
        np.mean(differences) / sd
    )


print("\nChunk 9A: Helpers and diagnostics complete.")



# ============================================================
# CHUNK 9B: FRIEDMAN TESTS
# ============================================================

print("\n" + "=" * 65)
print("FRIEDMAN TESTS")
print("=" * 65)

friedman_results = []

for condition, group in df.groupby(
    CONDITION_COLUMNS,
    observed=True
):

    for outcome in OUTCOMES:

        pivot = group.pivot(
            index=PAIRING_COLUMNS,
            columns="model",
            values=outcome
        )

        pivot = pivot.reindex(
            columns=MODELS
        ).dropna()

        if len(pivot) < 2:
            continue

        if pivot.nunique().max() == 1:
            continue

        statistic, p_value = friedmanchisquare(
            *[
                pivot[model].values
                for model in MODELS
            ]
        )

        n = len(pivot)

        k = len(MODELS)

        # Kendall's W effect size.
        kendalls_w = (
            statistic / (n * (k - 1))
        )

        friedman_results.append({

            "outcome": outcome,

            "selection_timing": condition[0],

            "strategy": condition[1],

            "vaccination_budget": condition[2],

            "paired_replicates": n,

            "friedman_statistic": statistic,

            "p_value": p_value,

            "kendalls_w": kendalls_w

        })


friedman_df = pd.DataFrame(
    friedman_results
)

# Correct across all experimental conditions
# separately for each outcome.

friedman_df["holm_adjusted_p"] = np.nan

friedman_df["significant_005"] = False

for outcome in OUTCOMES:

    mask = (
        friedman_df["outcome"] == outcome
    )

    corrected = apply_holm(
        friedman_df.loc[mask],
        "p_value"
    )

    friedman_df.loc[
        mask,
        "holm_adjusted_p"
    ] = corrected["holm_adjusted_p"].values

    friedman_df.loc[
        mask,
        "significant_005"
    ] = corrected["significant_005"].values


friedman_path = (
    OUTPUT_DIR
    / "chunk9_friedman_results.csv"
)

friedman_df.to_csv(
    friedman_path,
    index=False
)

print("\nFriedman results:")

print(
    friedman_df.head(12).to_string(
        index=False
    )
)

print("\nSaved:", friedman_path)


# ============================================================
# CHUNK 9B: PAIRED WILCOXON TESTS
# ============================================================

print("\n" + "=" * 65)
print("PAIRED WILCOXON TESTS")
print("=" * 65)

pairwise_results = []

NULL_MODELS = [
    "ER",
    "Configuration",
    "DCSBM"
]

for condition, group in df.groupby(
    CONDITION_COLUMNS,
    observed=True
):

    for outcome in OUTCOMES:

        pivot = group.pivot(
            index=PAIRING_COLUMNS,
            columns="model",
            values=outcome
        )

        pivot = pivot.reindex(
            columns=MODELS
        ).dropna()

        if len(pivot) < 2:
            continue

        empirical = pivot["Empirical"].values

        for null_model in NULL_MODELS:

            null_values = pivot[
                null_model
            ].values

            differences = (
                empirical - null_values
            )

            try:

                if np.allclose(
                    differences,
                    0
                ):

                    statistic = 0.0
                    p_value = 1.0

                else:

                    test = wilcoxon(
                        empirical,
                        null_values,
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto"
                    )

                    statistic = float(
                        test.statistic
                    )

                    p_value = float(
                        test.pvalue
                    )

            except ValueError:

                statistic = np.nan
                p_value = np.nan

            # Effect size.
            effect_size = paired_effect_size(
                differences
            )

            # Bootstrap confidence interval.
            ci_lower, ci_upper = (
                bootstrap_mean_difference(
                    differences
                )
            )

            pairwise_results.append({

                "outcome": outcome,

                "selection_timing": condition[0],

                "strategy": condition[1],

                "vaccination_budget": condition[2],

                "null_model": null_model,

                "paired_replicates": len(pivot),

                "empirical_mean": float(
                    empirical.mean()
                ),

                "null_mean": float(
                    null_values.mean()
                ),

                "mean_difference": float(
                    differences.mean()
                ),

                "effect_size_dz": effect_size,

                "ci_lower": ci_lower,

                "ci_upper": ci_upper,

                "wilcoxon_statistic": statistic,

                "p_value": p_value

            })


pairwise_df = pd.DataFrame(
    pairwise_results
)


# ============================================================
# HOLM MULTIPLE-TESTING CORRECTION
# ============================================================

pairwise_df["holm_adjusted_p"] = np.nan

pairwise_df["significant_005"] = False

# Define each outcome and experimental condition
# as a separate family of three comparisons.

FAMILY_COLUMNS = [
    "outcome",
    "selection_timing",
    "strategy",
    "vaccination_budget"
]

for _, indices in pairwise_df.groupby(
    FAMILY_COLUMNS,
    observed=True
).groups.items():

    family = pairwise_df.loc[indices]

    corrected = apply_holm(
        family,
        "p_value"
    )

    pairwise_df.loc[
        indices,
        "holm_adjusted_p"
    ] = corrected["holm_adjusted_p"].values

    pairwise_df.loc[
        indices,
        "significant_005"
    ] = corrected["significant_005"].values


# ============================================================
# SAVE PAIRWISE RESULTS
# ============================================================

pairwise_path = (
    OUTPUT_DIR
    / "chunk9_paired_wilcoxon_results.csv"
)

pairwise_df.to_csv(
    pairwise_path,
    index=False
)

print("\nPaired Wilcoxon results:")

print(
    pairwise_df.head(15).to_string(
        index=False
    )
)

print("\nSaved:", pairwise_path)


# ============================================================
# SIGNIFICANT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("STATISTICALLY SIGNIFICANT RESULTS")
print("=" * 65)

significant_friedman = friedman_df[
    friedman_df["significant_005"]
]

significant_pairwise = pairwise_df[
    pairwise_df["significant_005"]
]

print("\nSignificant Friedman tests:")

if len(significant_friedman) > 0:

    print(
        significant_friedman.to_string(
            index=False
        )
    )

else:

    print("No significant Friedman tests after Holm correction.")


print("\nSignificant paired comparisons:")

if len(significant_pairwise) > 0:

    print(
        significant_pairwise.to_string(
            index=False
        )
    )

else:

    print("No significant paired comparisons after Holm correction.")


# ============================================================
# SAVE SIGNIFICANT RESULTS
# ============================================================

significant_friedman.to_csv(
    OUTPUT_DIR / "chunk9_significant_friedman.csv",
    index=False
)

significant_pairwise.to_csv(
    OUTPUT_DIR / "chunk9_significant_pairwise.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("CHUNK 9B COMPLETE")
print("=" * 65)

print(
    "Friedman tests:",
    len(friedman_df)
)

print(
    "Paired Wilcoxon tests:",
    len(pairwise_df)
)

print(
    "Significant Friedman tests:",
    len(significant_friedman)
)

print(
    "Significant paired comparisons:",
    len(significant_pairwise)
)

print("\nOutput files:")

print("chunk9_friedman_results.csv")

print("chunk9_paired_wilcoxon_results.csv")

print("chunk9_significant_friedman.csv")

print("chunk9_significant_pairwise.csv")

print("\nSTAT 636 CHUNK 9B COMPLETE")

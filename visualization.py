
"""
STAT 636 PROJECT
Visualization Module

Chunk 7A of 7C:
- Imports
- Directory setup
- Simulation summary loading
- Validation
- Plotting helper functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

import config


# ============================================================
# 1. DIRECTORIES
# ============================================================

TABLE_DIR = config.TABLE_DIR

FIGURE_DIR = config.RESULTS_DIR / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_FILE = (
    TABLE_DIR / "sir_simulation_summary.csv"
)

RESULTS_FILE = (
    TABLE_DIR / "sir_simulation_results.csv"
)


# ============================================================
# 2. PLOT SETTINGS
# ============================================================

plt.rcParams.update({

    "figure.dpi": 120,

    "savefig.dpi": 300,

    "font.size": 10,

    "axes.titlesize": 11,

    "axes.labelsize": 10,

    "legend.fontsize": 8,

    "figure.titlesize": 15,

    "axes.spines.top": False,

    "axes.spines.right": False

})


# ============================================================
# 3. STRATEGY COLORS
# ============================================================

STRATEGY_COLORS = {

    "Random": "#808080",

    "Degree": "#D55E00",

    "Weighted Degree": "#0072B2",

    "Betweenness": "#009E73",

    "Community Bridges": "#CC79A7",

    "Within-Community Hubs": "#E69F00"

}


SELECTION_DAYS = [
    "Day 1",
    "Day 2"
]


# ============================================================
# 4. LOAD SIMULATION SUMMARY
# ============================================================

def load_simulation_summary():

    if not SUMMARY_FILE.exists():

        raise FileNotFoundError(
            f"Could not find {SUMMARY_FILE}. "
            "Run sir_simulation.py first."
        )

    summary = pd.read_csv(
        SUMMARY_FILE
    )

    required_columns = [

        "selection_day",

        "strategy",

        "budget",

        "beta",

        "mean_final_outbreak_size",

        "ci_lower_final_outbreak_size",

        "ci_upper_final_outbreak_size",

        "mean_peak_infected",

        "ci_lower_peak_infected",

        "ci_upper_peak_infected"

    ]

    missing = [

        column

        for column in required_columns

        if column not in summary.columns

    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    summary["budget"] = pd.to_numeric(
        summary["budget"],
        errors="coerce"
    )

    summary["beta"] = pd.to_numeric(
        summary["beta"],
        errors="coerce"
    )

    if summary[
        ["budget", "beta"]
    ].isna().any().any():

        raise ValueError(
            "Budget or beta contains missing values."
        )

    summary["budget_percent"] = (
        summary["budget"] * 100
    )

    print(
        "Loaded simulation conditions:",
        len(summary)
    )

    return summary


# ============================================================
# 5. STRATEGY HELPER FUNCTIONS
# ============================================================

def get_strategy_color(strategy):

    return STRATEGY_COLORS.get(
        strategy,
        "#333333"
    )


def get_sorted_strategies(data):

    preferred = list(
        STRATEGY_COLORS.keys()
    )

    available = list(
        data["strategy"].dropna().unique()
    )

    return [

        strategy

        for strategy in preferred

        if strategy in available

    ] + [

        strategy

        for strategy in available

        if strategy not in preferred

    ]


# ============================================================
# 6. PLOT CONFIDENCE INTERVALS
# ============================================================

def add_confidence_interval_plot(
    ax,
    data,
    outcome,
    lower_column,
    upper_column
):
    """
    Draw one line per vaccination strategy.

    Shaded regions represent the supplied
    lower and upper confidence limits.
    """

    strategies = get_sorted_strategies(
        data
    )

    for strategy in strategies:

        subset = data[
            data["strategy"] == strategy
        ].sort_values(
            "budget"
        )

        if subset.empty:
            continue

        x = subset[
            "budget_percent"
        ].to_numpy()

        y = subset[
            outcome
        ].to_numpy()

        lower = subset[
            lower_column
        ].to_numpy()

        upper = subset[
            upper_column
        ].to_numpy()

        color = get_strategy_color(
            strategy
        )

        ax.plot(
            x,
            y,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=strategy
        )

        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.12
        )

    ax.set_xlabel(
        "Vaccination budget (%)"
    )

    ax.set_xticks(
        sorted(
            data["budget_percent"].unique()
        )
    )

    ax.grid(
        alpha=0.25
    )


# ============================================================
# 7. SAVE FIGURES
# ============================================================

def save_figure(fig, filename):

    output = FIGURE_DIR / filename

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Saved figure:",
        output
    )


# ============================================================
# 8. FIGURE 1: FINAL OUTBREAK SIZE
# ============================================================

def plot_final_outbreak_size(summary):
    """
    Primary research figure.

    Rows:
        Day 1 vaccination selection
        Day 2 vaccination selection

    Columns:
        Different transmission rates

    Lines:
        Vaccination strategies

    Shading:
        95% confidence intervals
    """

    transmission_rates = sorted(
        summary["beta"].unique()
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(transmission_rates),
        figsize=(
            5 * len(transmission_rates),
            9
        ),
        squeeze=False,
        sharex=True
    )

    for row, day in enumerate(SELECTION_DAYS):

        for col, beta in enumerate(
            transmission_rates
        ):

            ax = axes[row, col]

            subset = summary[
                (summary["selection_day"] == day)
                &
                (summary["beta"] == beta)
            ]

            if subset.empty:

                ax.set_visible(False)

                continue

            add_confidence_interval_plot(
                ax=ax,
                data=subset,
                outcome="mean_final_outbreak_size",
                lower_column="ci_lower_final_outbreak_size",
                upper_column="ci_upper_final_outbreak_size"
            )

            ax.set_title(
                f"Transmission rate β = {beta:g}"
            )

            ax.set_ylabel(
                f"{day} selection\n"
                "Mean final outbreak size"
            )

            ax.set_ylim(
                bottom=0
            )

    # Create a shared legend.
    handles, labels = (
        axes[0, 0].get_legend_handles_labels()
    )

    if handles:

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0)
        )

    fig.suptitle(
        "Final Outbreak Size by Vaccination Strategy",
        y=1.02
    )

    fig.tight_layout(
        rect=[0, 0.07, 1, 0.98]
    )

    save_figure(
        fig,
        "final_outbreak_size.png"
    )


# ============================================================
# 9. FIGURE 2: PEAK INFECTED POPULATION
# ============================================================

def plot_peak_infected(summary):
    """
    Compare maximum simultaneously infected
    participants across vaccination strategies.

    This measures epidemic intensity rather
    than total epidemic size.
    """

    transmission_rates = sorted(
        summary["beta"].unique()
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(transmission_rates),
        figsize=(
            5 * len(transmission_rates),
            9
        ),
        squeeze=False,
        sharex=True
    )

    for row, day in enumerate(SELECTION_DAYS):

        for col, beta in enumerate(
            transmission_rates
        ):

            ax = axes[row, col]

            subset = summary[
                (summary["selection_day"] == day)
                &
                (summary["beta"] == beta)
            ]

            if subset.empty:

                ax.set_visible(False)

                continue

            add_confidence_interval_plot(
                ax=ax,
                data=subset,
                outcome="mean_peak_infected",
                lower_column="ci_lower_peak_infected",
                upper_column="ci_upper_peak_infected"
            )

            ax.set_title(
                f"Transmission rate β = {beta:g}"
            )

            ax.set_ylabel(
                f"{day} selection\n"
                "Mean peak infected"
            )

            ax.set_ylim(
                bottom=0
            )

    handles, labels = (
        axes[0, 0].get_legend_handles_labels()
    )

    if handles:

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0)
        )

    fig.suptitle(
        "Peak Epidemic Size by Vaccination Strategy",
        y=1.02
    )

    fig.tight_layout(
        rect=[0, 0.07, 1, 0.98]
    )

    save_figure(
        fig,
        "peak_infected.png"
    )


# ============================================================
# 10. FIGURE 3: DAY 1 VERSUS DAY 2 SELECTION
# ============================================================

def plot_selection_timing(summary):
    """
    Compare previous-day and same-day targeting.

    Each panel represents one vaccination strategy.

    Colors:
        Different transmission rates.

    Solid lines:
        Day 1 selection.

    Dashed lines:
        Day 2 selection.
    """

    strategies = get_sorted_strategies(
        summary
    )

    transmission_rates = sorted(
        summary["beta"].unique()
    )

    if len(strategies) == 0:

        raise ValueError(
            "No vaccination strategies found."
        )

    ncols = 3

    nrows = int(
        np.ceil(
            len(strategies) / ncols
        )
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            6 * ncols,
            4 * nrows
        ),
        squeeze=False
    )

    for i, strategy in enumerate(strategies):

        ax = axes.flat[i]

        subset = summary[
            summary["strategy"] == strategy
        ]

        for beta_index, beta in enumerate(
            transmission_rates
        ):

            color = plt.cm.viridis(
                beta_index / max(
                    1,
                    len(transmission_rates) - 1
                )
            )

            for day, linestyle, marker in [

                ("Day 1", "-", "o"),

                ("Day 2", "--", "s")

            ]:

                line_data = subset[
                    (subset["beta"] == beta)
                    &
                    (subset["selection_day"] == day)
                ].sort_values(
                    "budget"
                )

                if line_data.empty:
                    continue

                ax.plot(
                    line_data["budget_percent"],
                    line_data["mean_final_outbreak_size"],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.6,
                    markersize=4,
                    label=f"β={beta:g}, {day}"
                )

        ax.set_title(
            strategy
        )

        ax.set_xlabel(
            "Vaccination budget (%)"
        )

        ax.set_ylabel(
            "Mean final outbreak size"
        )

        ax.grid(
            alpha=0.25
        )

    # Hide unused panels.
    for i in range(
        len(strategies),
        len(axes.flat)
    ):

        axes.flat[i].set_visible(
            False
        )

    handles, labels = (
        axes.flat[0].get_legend_handles_labels()
    )

    if handles:

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            fontsize=7,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02)
        )

    fig.suptitle(
        "Previous-Day versus Same-Day Vaccination Selection",
        y=1.02
    )

    fig.tight_layout(
        rect=[0, 0.09, 1, 0.97]
    )

    save_figure(
        fig,
        "selection_timing_comparison.png"
    )


# ============================================================
# 11. FIGURE 4: TRANSMISSION-RATE SENSITIVITY
# ============================================================

def plot_transmission_sensitivity(summary):
    """
    Examine how vaccination effectiveness changes
    with transmission rate.

    Separate panels represent vaccination budgets.

    Results average across selection days.
    """

    budgets = sorted(
        summary["budget"].unique()
    )

    strategies = get_sorted_strategies(
        summary
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(budgets),
        figsize=(
            5 * len(budgets),
            4
        ),
        squeeze=False,
        sharey=True
    )

    for i, budget in enumerate(budgets):

        ax = axes[0, i]

        subset = summary[
            summary["budget"] == budget
        ]

        for strategy in strategies:

            strategy_data = subset[
                subset["strategy"] == strategy
            ].groupby(
                "beta",
                as_index=False
            )[
                "mean_final_outbreak_size"
            ].mean()

            if strategy_data.empty:
                continue

            strategy_data = strategy_data.sort_values(
                "beta"
            )

            ax.plot(
                strategy_data["beta"],
                strategy_data["mean_final_outbreak_size"],
                marker="o",
                linewidth=1.7,
                color=get_strategy_color(strategy),
                label=strategy
            )

        ax.set_title(
            f"{budget:.0%} vaccinated"
        )

        ax.set_xlabel(
            "Transmission rate (β)"
        )

        ax.set_xscale(
            "log"
        )

        ax.grid(
            alpha=0.25
        )

    axes[0, 0].set_ylabel(
        "Mean final outbreak size"
    )

    handles, labels = (
        axes[0, 0].get_legend_handles_labels()
    )

    if handles:

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, -0.12)
        )

    fig.suptitle(
        "Sensitivity to Transmission Rate",
        y=1.04
    )

    fig.tight_layout(
        rect=[0, 0.13, 1, 0.95]
    )

    save_figure(
        fig,
        "transmission_sensitivity.png"
    )


# ============================================================
# 12. TABLE 1: BEST VACCINATION STRATEGY
# ============================================================

def create_best_strategy_table(summary):
    """
    Identify the strategy with the smallest
    estimated mean final outbreak size.

    Comparisons are performed separately for each:

        Selection day
        Vaccination budget
        Transmission rate

    This is a descriptive comparison, not
    a statistical significance test.
    """

    grouping = [
        "selection_day",
        "budget",
        "beta"
    ]

    # Remove rows with missing outcome values.
    valid = summary.dropna(
        subset=[
            "mean_final_outbreak_size"
        ]
    ).copy()

    best_indices = (
        valid.groupby(
            grouping
        )["mean_final_outbreak_size"].idxmin()
    )

    best = valid.loc[
        best_indices
    ].copy()

    best = best.sort_values(
        grouping
    )

    columns = [
        "selection_day",
        "strategy",
        "budget",
        "beta",
        "mean_final_outbreak_size",
        "ci_lower_final_outbreak_size",
        "ci_upper_final_outbreak_size",
        "mean_peak_infected"
    ]

    best = best[columns].copy()

    best["budget_percent"] = (
        best["budget"] * 100
    )

    output = (
        TABLE_DIR /
        "best_vaccination_strategies.csv"
    )

    best.to_csv(
        output,
        index=False
    )

    print(
        "Saved table:",
        output
    )

    return best


# ============================================================
# 13. TABLE 2: VACCINATION EFFECTIVENESS
# ============================================================

def create_effectiveness_table(summary):
    """
    Calculate percentage reduction in mean final
    outbreak size relative to random vaccination.

    Comparisons are matched by:

        Selection day
        Vaccination budget
        Transmission rate

    Positive reduction:
        Better than random vaccination.

    Negative reduction:
        Worse than random vaccination.
    """

    grouping = [
        "selection_day",
        "budget",
        "beta"
    ]

    random_baseline = summary[
        summary["strategy"] == "Random"
    ][
        grouping + [
            "mean_final_outbreak_size"
        ]
    ].rename(
        columns={
            "mean_final_outbreak_size":
                "random_mean_outbreak"
        }
    )

    comparison = summary.merge(
        random_baseline,
        on=grouping,
        how="left",
        validate="many_to_one"
    )

    comparison["reduction_percent"] = np.where(

        comparison["random_mean_outbreak"] > 0,

        100 * (
            comparison["random_mean_outbreak"]
            - comparison["mean_final_outbreak_size"]
        ) / comparison["random_mean_outbreak"],

        np.nan
    )

    columns = [
        "selection_day",
        "strategy",
        "budget",
        "beta",
        "mean_final_outbreak_size",
        "random_mean_outbreak",
        "reduction_percent"
    ]

    effectiveness = comparison[
        columns
    ].sort_values(
        [
            "selection_day",
            "beta",
            "budget",
            "reduction_percent"
        ],
        ascending=[
            True,
            True,
            True,
            False
        ]
    )

    output = (
        TABLE_DIR /
        "vaccination_effectiveness.csv"
    )

    effectiveness.to_csv(
        output,
        index=False
    )

    print(
        "Saved table:",
        output
    )

    return effectiveness


# ============================================================
# 14. EXPERIMENTAL OVERVIEW
# ============================================================

def print_experimental_overview(summary):

    print("\n====================================")

    print("EXPERIMENTAL OVERVIEW")

    print("====================================")

    print(
        "\nSelection days:"
    )

    print(
        sorted(
            summary["selection_day"].unique()
        )
    )

    print(
        "\nVaccination strategies:"
    )

    print(
        sorted(
            summary["strategy"].unique()
        )
    )

    print(
        "\nVaccination budgets:"
    )

    print(
        sorted(
            summary["budget"].unique()
        )
    )

    print(
        "\nTransmission rates:"
    )

    print(
        sorted(
            summary["beta"].unique()
        )
    )

    if "n_simulations" in summary.columns:

        print(
            "\nSimulation replicates:"
        )

        print(
            sorted(
                summary["n_simulations"].unique()
            )
        )


# ============================================================
# 15. MAIN VISUALIZATION ANALYSIS
# ============================================================

def run_visualization_analysis():
    """
    Run the complete visualization workflow.

    Generates four figures and two result tables.
    """

    print("\n====================================")

    print("STAT 636 VISUALIZATION ANALYSIS")

    print("====================================")

    # Load simulation results.
    summary = load_simulation_summary()

    # Print experiment details.
    print_experimental_overview(
        summary
    )

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------

    print("\nGenerating figures...")

    plot_final_outbreak_size(
        summary
    )

    plot_peak_infected(
        summary
    )

    plot_selection_timing(
        summary
    )

    plot_transmission_sensitivity(
        summary
    )

    # --------------------------------------------------------
    # TABLES
    # --------------------------------------------------------

    print("\nGenerating summary tables...")

    best = create_best_strategy_table(
        summary
    )

    effectiveness = create_effectiveness_table(
        summary
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\nBEST STRATEGY PREVIEW")

    print(
        best.head(10).to_string(
            index=False
        )
    )

    print("\nEFFECTIVENESS PREVIEW")

    print(
        effectiveness.head(10).to_string(
            index=False
        )
    )

    print("\n====================================")

    print("VISUALIZATION ANALYSIS COMPLETED")

    print("====================================")

    return {

        "summary": summary,

        "best_strategies": best,

        "effectiveness": effectiveness

    }


# ============================================================
# 16. EXECUTE SCRIPT
# ============================================================

if __name__ == "__main__":

    run_visualization_analysis()

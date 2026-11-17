
"""
STAT 636 Project
Chunk 5A: Vaccination Strategy Functions

Implements:
    1. Random vaccination
    2. Degree vaccination
    3. Weighted-degree vaccination
    4. Betweenness vaccination
    5. Community-bridge vaccination
    6. Within-community hub vaccination
"""

import numpy as np
import pandas as pd
import networkx as nx

from pathlib import Path

import config

from spectral_analysis import (
    calculate_community_bridge_entropy,
    calculate_degree_correction
)


# ============================================================
# 1. VACCINATION BUDGETS
# ============================================================

VACCINATION_BUDGETS = [0.05, 0.10, 0.15, 0.20]


def calculate_vaccination_count(
    population_size,
    budget
):
    """
    Calculate the number of participants to vaccinate.

    Uses ceiling to ensure the requested percentage
    is met or exceeded.
    """

    if not 0 <= budget <= 1:
        raise ValueError("Budget must be between 0 and 1.")

    return int(
        np.ceil(population_size * budget)
    )


# ============================================================
# 2. DEGREE SCORES
# ============================================================

def calculate_degree_scores(G, node_order):
    """
    Calculate the number of distinct contacts per participant.
    """

    return np.array([
        G.degree(node)
        for node in node_order
    ], dtype=float)


# ============================================================
# 3. WEIGHTED-DEGREE SCORES
# ============================================================

def calculate_weighted_degree_scores(
    G,
    node_order
):
    """
    Calculate total contact duration per participant.

    Strength_i = sum_j w_ij
    """

    strengths = dict(
        G.degree(weight="duration")
    )

    return np.array([
        strengths[node]
        for node in node_order
    ], dtype=float)


# ============================================================
# 4. WEIGHTED BETWEENNESS CENTRALITY
# ============================================================

def calculate_betweenness_scores(
    G,
    node_order
):
    """
    Calculate weighted shortest-path betweenness.

    Contact duration is converted into distance:

        distance_ij = 1 / duration_ij

    Therefore, longer contacts correspond to shorter
    effective distances.

    Scores are normalized by NetworkX.
    """

    H = G.copy()

    for u, v, data in H.edges(data=True):

        duration = float(
            data.get("duration", 0)
        )

        if duration <= 0:
            raise ValueError(
                "Contact durations must be positive."
            )

        H[u][v]["distance"] = 1.0 / duration

    betweenness = nx.betweenness_centrality(
        H,
        normalized=True,
        weight="distance"
    )

    return np.array([
        betweenness[node]
        for node in node_order
    ], dtype=float)


# ============================================================
# 5. COMMUNITY BRIDGE SCORES
# ============================================================

def calculate_bridge_scores(
    G,
    node_order,
    community_labels
):
    """
    Calculate normalized contact entropy.

    Higher entropy indicates that a participant's
    contact duration is distributed across more
    detected communities.
    """

    result = calculate_community_bridge_entropy(
        G,
        node_order,
        community_labels
    )

    return result["normalized_entropy"]


# ============================================================
# 6. WITHIN-COMMUNITY HUB SCORES
# ============================================================

def calculate_within_community_hub_scores(
    G,
    node_order,
    community_labels
):
    """
    Calculate relative contact strength within
    each detected community.

    theta_i = strength_i / mean_strength_community_i
    """

    result = calculate_degree_correction(
        G,
        node_order,
        community_labels
    )

    return result["theta"]


# ============================================================
# 7. CREATE ALL DETERMINISTIC STRATEGY SCORES
# ============================================================

def calculate_all_strategy_scores(
    G,
    node_order,
    community_labels
):
    """
    Calculate participant scores for all five
    deterministic targeting strategies.

    Random vaccination is generated separately
    for each simulation replicate.
    """

    scores = pd.DataFrame({

        "id": node_order,

        "degree": calculate_degree_scores(
            G,
            node_order
        ),

        "weighted_degree": calculate_weighted_degree_scores(
            G,
            node_order
        ),

        "betweenness": calculate_betweenness_scores(
            G,
            node_order
        ),

        "bridge_entropy": calculate_bridge_scores(
            G,
            node_order,
            community_labels
        ),

        "within_community_hub": (
            calculate_within_community_hub_scores(
                G,
                node_order,
                community_labels
            )
        )

    })

    return scores


# ============================================================
# 8. RANK PARTICIPANTS
# ============================================================

def rank_participants(
    scores,
    strategy
):
    """
    Rank participants from highest to lowest score.

    Ties are resolved by participant ID to ensure
    reproducibility.

    Random vaccination is intentionally excluded.
    """

    strategy_columns = {

        "Degree": "degree",

        "Weighted Degree": "weighted_degree",

        "Betweenness": "betweenness",

        "Community Bridges": "bridge_entropy",

        "Within-Community Hubs":
            "within_community_hub"

    }

    if strategy not in strategy_columns:

        raise ValueError(
            f"Unknown deterministic strategy: {strategy}"
        )

    column = strategy_columns[strategy]

    ranking = scores.sort_values(
        by=[column, "id"],
        ascending=[False, True],
        kind="mergesort"
    ).copy()

    ranking["rank"] = np.arange(
        1,
        len(ranking) + 1
    )

    ranking["strategy"] = strategy

    return ranking


# ============================================================
# 9. SELECT TOP PARTICIPANTS
# ============================================================

def select_top_participants(
    ranking,
    budget,
    population_size
):
    """
    Select the top-ranked participants for a
    specified vaccination budget.
    """

    n_vaccinated = calculate_vaccination_count(
        population_size,
        budget
    )

    selected = ranking.head(
        n_vaccinated
    )["id"].tolist()

    return selected


# ============================================================
# 10. RANDOM VACCINATION
# ============================================================

def select_random_participants(
    node_order,
    budget,
    random_state=None
):
    """
    Select participants uniformly at random.

    A separate random sample is generated for each
    simulation replicate.

    Use the same random_state to reproduce a sample.
    """

    rng = np.random.default_rng(
        random_state
    )

    n_vaccinated = calculate_vaccination_count(
        len(node_order),
        budget
    )

    selected = rng.choice(
        np.asarray(node_order),
        size=n_vaccinated,
        replace=False
    )

    return selected.tolist()


# ============================================================
# 11. COMPLETE VACCINATION STRATEGY SELECTOR
# ============================================================

def select_vaccinated(
    scores,
    node_order,
    strategy,
    budget,
    random_state=None
):
    """
    Select vaccinated participants using one of
    the six strategies.
    """

    if strategy == "Random":

        return select_random_participants(
            node_order,
            budget,
            random_state
        )

    ranking = rank_participants(
        scores,
        strategy
    )

    return select_top_participants(
        ranking,
        budget,
        len(node_order)
    )


# ============================================================
# 12. AVAILABLE STRATEGIES
# ============================================================

STRATEGIES = [

    "Random",

    "Degree",

    "Weighted Degree",

    "Betweenness",

    "Community Bridges",

    "Within-Community Hubs"

]



# ============================================================
# 13. GENERATE DETERMINISTIC VACCINATION RANKINGS
# ============================================================

def generate_rankings(scores):
    """
    Generate rankings for all five deterministic strategies.
    """

    rankings = {}

    for strategy in STRATEGIES:

        if strategy == "Random":
            continue

        rankings[strategy] = rank_participants(
            scores,
            strategy
        )

    return rankings


# ============================================================
# 14. SAVE VACCINATION RANKINGS
# ============================================================

def save_vaccination_rankings(
    rankings,
    day
):
    """
    Save all deterministic rankings in one CSV file.
    """

    all_rankings = []

    for strategy, ranking in rankings.items():

        temp = ranking.copy()

        temp["selection_day"] = day

        all_rankings.append(temp)

    combined = pd.concat(
        all_rankings,
        ignore_index=True
    )

    prefix = day.lower().replace(" ", "")

    combined.to_csv(
        config.TABLE_DIR /
        f"vaccination_rankings_{prefix}.csv",
        index=False
    )

    return combined


# ============================================================
# 15. GENERATE VACCINATION SELECTIONS
# ============================================================

def generate_vaccination_selections(
    rankings,
    node_order,
    day,
    random_seed=636
):
    """
    Generate vaccination selections for every strategy
    and vaccination budget.

    Each selection is labeled with its selection day
    and intended outbreak day.

    Random selections here are reproducible examples.
    The epidemic simulation should regenerate random
    selections separately for every replicate.
    """

    results = []

    population_size = len(node_order)

    for budget in VACCINATION_BUDGETS:

        n_vaccinated = calculate_vaccination_count(
            population_size,
            budget
        )

        # ------------------------------------
        # RANDOM
        # ------------------------------------

        random_selected = select_random_participants(
            node_order,
            budget,
            random_state=(
                random_seed
                + int(budget * 1000)
            )
        )

        for participant in random_selected:

            results.append({

                "selection_day": day,

                "target_day": "Day 2",

                "strategy": "Random",

                "budget": budget,

                "n_vaccinated": n_vaccinated,

                "id": participant

            })

        # ------------------------------------
        # DETERMINISTIC STRATEGIES
        # ------------------------------------

        for strategy, ranking in rankings.items():

            selected = select_top_participants(
                ranking,
                budget,
                population_size
            )

            for participant in selected:

                results.append({

                    "selection_day": day,

                    "target_day": "Day 2",

                    "strategy": strategy,

                    "budget": budget,

                    "n_vaccinated": n_vaccinated,

                    "id": participant

                })

    selection_table = pd.DataFrame(results)

    prefix = day.lower().replace(" ", "")

    selection_table.to_csv(
        config.TABLE_DIR /
        f"vaccination_selections_{prefix}.csv",
        index=False
    )

    return selection_table


# ============================================================
# 16. ANALYZE ONE DAY'S VACCINATION STRATEGIES
# ============================================================

def analyze_vaccination_strategies(
    G,
    node_order,
    community_labels,
    day
):
    """
    Calculate all vaccination scores and rankings
    for one daily network.
    """

    print("\n====================================")
    print(f"VACCINATION ANALYSIS: {day}")
    print("====================================")

    # Calculate participant scores.
    scores = calculate_all_strategy_scores(
        G,
        node_order,
        community_labels
    )

    # Add selection-day identifier.
    scores["selection_day"] = day

    # Save participant-level scores.
    prefix = day.lower().replace(" ", "")

    scores.to_csv(
        config.TABLE_DIR /
        f"vaccination_scores_{prefix}.csv",
        index=False
    )

    # Generate rankings.
    rankings = generate_rankings(
        scores
    )

    ranking_table = save_vaccination_rankings(
        rankings,
        day
    )

    # Generate vaccination selections.
    selections = generate_vaccination_selections(
        rankings,
        node_order,
        day
    )

    print("\nVACCINATION STRATEGIES")

    print(STRATEGIES)

    print("\nNUMBER OF PARTICIPANTS:", len(node_order))

    print("\nVACCINATION BUDGETS")

    for budget in VACCINATION_BUDGETS:

        print(
            f"{budget:.0%}: "
            f"{calculate_vaccination_count(len(node_order), budget)}"
        )

    print("\nVACCINATION ANALYSIS COMPLETED.")

    return {
        "scores": scores,
        "rankings": rankings,
        "ranking_table": ranking_table,
        "selections": selections
    }


# ============================================================
# 17. COMPLETE VACCINATION ANALYSIS PIPELINE
# ============================================================

def run_vaccination_analysis():
    """
    Run all vaccination strategies on both daily networks.

    Day 1 rankings:
        Used for previous-day selection.

    Day 2 rankings:
        Used for same-day selection.

    Both sets of selections target a Day 2 epidemic.
    """

    from data_loader import load_school_networks
    from spectral_analysis import run_spectral_analysis

    print("\n====================================")
    print("STAT 636 VACCINATION ANALYSIS")
    print("====================================")

    # ------------------------------------
    # Load daily networks.
    # ------------------------------------

    data = load_school_networks()

    G1 = data.G1
    G2 = data.G2

    node_order = data.node_order

    # ------------------------------------
    # Obtain spectral community assignments.
    # ------------------------------------

    print("\nLoading spectral community assignments...")

    community_file = (
            config.TABLE_DIR /
            "community_assignments_both_days.csv"
    )

    # Run spectral analysis if the file does not exist.
    if not community_file.exists():
        print("Community file not found.")

        from spectral_analysis import run_spectral_analysis

        run_spectral_analysis()

    if not community_file.exists():
        raise FileNotFoundError(
            "Spectral analysis did not create "
            "community_assignments_both_days.csv"
        )

    communities = pd.read_csv(community_file)

    # Standardize participant IDs.
    communities["id"] = (
        communities["id"].astype(str).str.strip()
    )

    node_order_clean = [
        str(node).strip()
        for node in node_order
    ]

    # Check duplicate IDs.
    if communities["id"].duplicated().any():
        raise ValueError(
            "Duplicate participant IDs in community file."
        )

    # Index by participant ID.
    communities = communities.set_index("id")

    # Reorder to match the network.
    communities = communities.reindex(
        node_order_clean
    )

    # Verify assignments.
    required_columns = [
        "community_day1",
        "community_day2"
    ]

    for column in required_columns:

        if column not in communities.columns:
            raise ValueError(
                f"Missing column: {column}"
            )

        if communities[column].isna().any():
            missing = communities[
                communities[column].isna()
            ].index.tolist()

            raise ValueError(
                f"Missing {column} for participants: "
                f"{missing[:20]}"
            )

    # Extract assignments in network order.
    labels1 = communities[
        "community_day1"
    ].to_numpy()

    labels2 = communities[
        "community_day2"
    ].to_numpy()

    print("Community assignments successfully loaded.")

    print("Day 1 communities:", len(set(labels1)))
    print("Day 2 communities:", len(set(labels2)))

    # ------------------------------------
    # Analyze Day 1.
    # ------------------------------------

    day1_results = analyze_vaccination_strategies(
        G1,
        node_order,
        labels1,
        "Day 1"
    )

    # ------------------------------------
    # Analyze Day 2.
    # ------------------------------------

    day2_results = analyze_vaccination_strategies(
        G2,
        node_order,
        labels2,
        "Day 2"
    )

    # ------------------------------------
    # Combine experimental selections.
    # ------------------------------------

    combined_selections = pd.concat(
        [
            day1_results["selections"],
            day2_results["selections"]
        ],
        ignore_index=True
    )

    combined_selections.to_csv(
        config.TABLE_DIR /
        "vaccination_selections_all.csv",
        index=False
    )

    # ------------------------------------
    # Save summary.
    # ------------------------------------

    summary = pd.DataFrame({

        "strategy": STRATEGIES,

        "day1_available": [True] * len(STRATEGIES),

        "day2_available": [True] * len(STRATEGIES)

    })

    summary.to_csv(
        config.TABLE_DIR /
        "vaccination_strategy_summary.csv",
        index=False
    )

    print("\n====================================")
    print("VACCINATION ANALYSIS COMPLETED")
    print("====================================")

    print(
        "\nResults saved to:",
        config.RESULTS_DIR
    )

    return {
        "day1": day1_results,
        "day2": day2_results,
        "combined_selections": combined_selections,
        "summary": summary
    }


# ============================================================
# 18. EXECUTE DIRECTLY
# ============================================================

if __name__ == "__main__":

    results = run_vaccination_analysis()

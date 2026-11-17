
"""
STAT 636
Chunk 5: Vaccination Strategy Tests
"""

import numpy as np

from data_loader import load_school_networks

from spectral_analysis import (
    spectral_preprocessing,
    compare_clustering_methods
)

from vaccination import (
    STRATEGIES,
    VACCINATION_BUDGETS,
    calculate_vaccination_count,
    calculate_all_strategy_scores,
    generate_rankings,
    select_vaccinated,
    select_random_participants
)


# ============================================================
# LOAD NETWORK
# ============================================================

data = load_school_networks()

G = data.G1
A = data.A1

node_order = data.node_order
metadata = data.metadata


# ============================================================
# OBTAIN COMMUNITY ASSIGNMENTS
# ============================================================

spectral = spectral_preprocessing(
    A,
    dimensions=3
)

Y = spectral["normalized_embedding"]

_, assignments, _ = compare_clustering_methods(
    Y,
    metadata,
    node_order
)

community_labels = assignments[
    ("K-means", 10)
]


# ============================================================
# TEST 1: VACCINATION COUNTS
# ============================================================

expected_counts = [13, 25, 37, 49]

actual_counts = [
    calculate_vaccination_count(
        242,
        budget
    )
    for budget in VACCINATION_BUDGETS
]

assert actual_counts == expected_counts

print("TEST 1 PASSED: Vaccination counts.")


# ============================================================
# TEST 2: ALL STRATEGY SCORES
# ============================================================

scores = calculate_all_strategy_scores(
    G,
    node_order,
    community_labels
)

assert len(scores) == 242

assert scores.shape[1] == 6

assert np.isfinite(
    scores.select_dtypes(
        include=[np.number]
    ).to_numpy()
).all()

print("TEST 2 PASSED: Strategy scores.")


# ============================================================
# TEST 3: DETERMINISTIC RANKINGS
# ============================================================

rankings = generate_rankings(
    scores
)

assert len(rankings) == 5

for strategy, ranking in rankings.items():

    assert len(ranking) == 242

    assert ranking["rank"].is_unique

    assert ranking["rank"].min() == 1

    assert ranking["rank"].max() == 242

print("TEST 3 PASSED: Deterministic rankings.")


# ============================================================
# TEST 4: DEGREE RANKING ORDER
# ============================================================

degree_ranking = rankings["Degree"]

assert (
    degree_ranking["degree"].is_monotonic_decreasing
)

print("TEST 4 PASSED: Degree ordering.")


# ============================================================
# TEST 5: RANDOM VACCINATION
# ============================================================

random1 = select_random_participants(
    node_order,
    0.10,
    random_state=636
)

random2 = select_random_participants(
    node_order,
    0.10,
    random_state=636
)

assert random1 == random2

assert len(random1) == 25

assert len(set(random1)) == 25

print("TEST 5 PASSED: Random vaccination.")


# ============================================================
# TEST 6: ALL SIX STRATEGIES
# ============================================================

for strategy in STRATEGIES:

    selected = select_vaccinated(
        scores,
        node_order,
        strategy,
        0.10,
        random_state=636
    )

    assert len(selected) == 25

    assert len(set(selected)) == 25

    assert set(selected).issubset(
        set(node_order)
    )

print("TEST 6 PASSED: All six strategies.")


# ============================================================
# TEST 7: VACCINATION BUDGETS
# ============================================================

for budget, expected in zip(
    VACCINATION_BUDGETS,
    expected_counts
):

    selected = select_vaccinated(
        scores,
        node_order,
        "Degree",
        budget
    )

    assert len(selected) == expected

print("TEST 7 PASSED: All vaccination budgets.")


# ============================================================
# FINAL
# ============================================================

print("\nAVAILABLE STRATEGIES:")

for strategy in STRATEGIES:

    print("-", strategy)

print("\nALL CHUNK 5 TESTS COMPLETED.")

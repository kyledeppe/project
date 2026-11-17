
"""
STAT 636
Chunk 6: SIR Simulation Tests
"""

import numpy as np
import networkx as nx

from sir_simulation import (
    prepare_simulation_network,
    simulate_sir,
    choose_initial_infection,
    summarize_simulations
)


# ============================================================
# TEST 1: CREATE A SMALL CONTACT NETWORK
# ============================================================

G = nx.Graph()

G.add_nodes_from([
    "A",
    "B",
    "C",
    "D",
    "E"
])

G.add_edge(
    "A", "B",
    duration=1000
)

G.add_edge(
    "B", "C",
    duration=1000
)

G.add_edge(
    "C", "D",
    duration=1000
)

# E is an isolate.

node_order = [
    "A",
    "B",
    "C",
    "D",
    "E"
]

adjacency, node_to_index = (
    prepare_simulation_network(
        G,
        node_order
    )
)

assert len(adjacency) == 5

assert len(adjacency[4]) == 0

print("TEST 1 PASSED: Network preparation.")


# ============================================================
# TEST 2: INITIAL INFECTION SELECTION
# ============================================================

seed_order = [
    "A",
    "B",
    "C",
    "D",
    "E"
]

vaccinated = [
    "A",
    "B"
]

initial_case = choose_initial_infection(
    node_order,
    vaccinated,
    seed_order
)

assert initial_case == "C"

print("TEST 2 PASSED: Initial infection selection.")


# ============================================================
# TEST 3: VACCINATED PARTICIPANTS ARE PROTECTED
# ============================================================

result = simulate_sir(
    adjacency=adjacency,
    node_order=node_order,
    vaccinated=["B", "D"],
    beta=0.001,
    infectious_period=3,
    seed_order=["A", "B", "C", "D", "E"],
    random_state=636
)

assert result["initial_case"] not in ["B", "D"]

assert result["final_outbreak_size"] <= 3

assert result["attack_rate"] <= 1

print("TEST 3 PASSED: Vaccination protection.")


# ============================================================
# TEST 4: OUTBREAK SIZE IS VALID
# ============================================================

assert (
    1 <= result["final_outbreak_size"] <= 3
)

assert (
    1 <= result["peak_infected"] <= 3
)

assert result["outbreak_duration"] >= 1

print("TEST 4 PASSED: Epidemic statistics.")


# ============================================================
# TEST 5: RANDOM SEED REPRODUCIBILITY
# ============================================================

result1 = simulate_sir(
    adjacency,
    node_order,
    vaccinated=[],
    beta=0.0001,
    infectious_period=3,
    seed_order=seed_order,
    random_state=123
)

result2 = simulate_sir(
    adjacency,
    node_order,
    vaccinated=[],
    beta=0.0001,
    infectious_period=3,
    seed_order=seed_order,
    random_state=123
)

assert (
    result1["final_outbreak_size"]
    == result2["final_outbreak_size"]
)

assert (
    result1["peak_infected"]
    == result2["peak_infected"]
)

print("TEST 5 PASSED: Reproducibility.")


# ============================================================
# TEST 6: SUMMARY STATISTICS
# ============================================================

import pandas as pd

example_results = pd.DataFrame({

    "selection_day": ["Day 1"] * 5,

    "outbreak_day": ["Day 2"] * 5,

    "strategy": ["Degree"] * 5,

    "budget": [0.10] * 5,

    "beta": [0.0001] * 5,

    "infectious_period": [3] * 5,

    "final_outbreak_size": [10, 12, 11, 9, 13],

    "attack_rate": [0.05, 0.06, 0.055, 0.045, 0.065],

    "peak_infected": [4, 5, 4, 3, 6],

    "peak_fraction": [0.02, 0.025, 0.02, 0.015, 0.03],

    "outbreak_duration": [5, 6, 5, 4, 7]

})

summary = summarize_simulations(
    example_results
)

assert len(summary) == 1

assert summary["n_simulations"].iloc[0] == 5

assert (
    summary["ci_lower_final_outbreak_size"].iloc[0]
    <= summary["mean_final_outbreak_size"].iloc[0]
)

assert (
    summary["ci_upper_final_outbreak_size"].iloc[0]
    >= summary["mean_final_outbreak_size"].iloc[0]
)

print("TEST 6 PASSED: Confidence interval calculations.")


# ============================================================
# FINAL
# ============================================================

print("\nALL CHUNK 6 TESTS COMPLETED.")

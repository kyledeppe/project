
"""
STAT 636
Chunk 8C: Paired SIR Experiments on Null Models

Compares vaccination strategies across:
    Empirical
    Erdős-Rényi
    Configuration
    DCSBM

Selection timing:
    Previous-day selection (day 1 -> day 2)
    Same-day selection (day 2 -> day 2)

Outputs:
    Individual simulation results
    Summary statistics
    Confidence intervals
"""

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from scipy.stats import sem

import config


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

SEED = 636

N_SIMULATIONS = 200

VACCINATION_BUDGETS = [
    0.05,
    0.10,
    0.15,
    0.20
]

STRATEGIES = [
    "Random",
    "Degree",
    "Weighted_Degree",
    "Betweenness",
    "Community_Bridge"
]

NETWORK_MODELS = [
    "Empirical",
    "ER",
    "Configuration",
    "DCSBM"
]

SELECTION_TIMINGS = [
    "Previous_Day",
    "Same_Day"
]

# Transmission rate per second of contact.
# DELETE? BETA = 0.00005
BETA = 0.0002

# One infectious generation per individual.
# This is a simplified Reed-Frost SIR model.
INFECTIOUS_PERIOD_STEPS = 1


# ============================================================
# DIRECTORIES
# ============================================================

RESULTS_DIR = Path(config.RESULTS_DIR)

NULL_DIR = RESULTS_DIR / "null_models"

TABLE_DIR = RESULTS_DIR / "tables"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RANDOM NUMBER GENERATOR
# ============================================================

rng = np.random.default_rng(SEED)


# ============================================================
# LOAD ORIGINAL SPARSE ADJACENCY ARCHIVES
# ============================================================

def load_empirical_adjacency(day):

    path = (
        RESULTS_DIR /
        f"adjacency_{day}.npz"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Cannot find {path}"
        )

    archive = np.load(
        path,
        allow_pickle=True
    )

    required = [
        "data",
        "indices",
        "indptr",
        "shape",
        "node_order"
    ]

    for key in required:

        if key not in archive:

            raise ValueError(
                f"Missing archive key: {key}"
            )

    from scipy.sparse import csr_matrix

    A = csr_matrix(
        (
            archive["data"],
            archive["indices"],
            archive["indptr"]
        ),
        shape=tuple(archive["shape"])
    ).toarray()

    node_order = archive["node_order"].astype(str)

    archive.close()

    return A, node_order


# ============================================================
# LOAD GENERATED NULL NETWORKS
# ============================================================

def load_null_adjacency(model, day):

    filename = (
        f"{model.lower()}_{day.lower()}.csv"
    )

    path = NULL_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"Cannot find null network: {path}"
        )

    A = np.loadtxt(
        path,
        delimiter=","
    )

    return np.asarray(
        A,
        dtype=float
    )


# ============================================================
# LOAD COMMUNITY ASSIGNMENTS
# ============================================================

def load_community_assignments(
    node_order,
    day
):

    path = (
        TABLE_DIR /
        "community_assignments_both_days.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Cannot find community assignments: {path}"
        )

    df = pd.read_csv(path)

    community_column = (
        f"community_{day}"
    )

    required = [
        "id",
        community_column
    ]

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

    df["id"] = df["id"].astype(str)

    lookup = dict(
        zip(
            df["id"],
            df[community_column]
        )
    )

    communities = np.array([
        lookup.get(str(node), np.nan)
        for node in node_order
    ])

    if pd.isna(communities).any():

        raise ValueError(
            f"Missing community assignments for {day}"
        )

    return communities


# ============================================================
# VALIDATE NETWORKS
# ============================================================

def validate_network(A, name):

    A = np.asarray(
        A,
        dtype=float
    )

    if A.ndim != 2:

        raise ValueError(
            f"{name}: adjacency must be two-dimensional."
        )

    if A.shape[0] != A.shape[1]:

        raise ValueError(
            f"{name}: adjacency must be square."
        )

    if not np.isfinite(A).all():

        raise ValueError(
            f"{name}: non-finite values detected."
        )

    if np.any(A < 0):

        raise ValueError(
            f"{name}: negative weights detected."
        )

    if not np.allclose(A, A.T):

        raise ValueError(
            f"{name}: network is not symmetric."
        )

    if not np.allclose(np.diag(A), 0):

        raise ValueError(
            f"{name}: self-loops detected."
        )

    return A


print(
    "STAT 636 configuration loaded."
)

print(
    "Results directory:",
    RESULTS_DIR
)


# ============================================================
# LOAD ALL NETWORKS
# ============================================================

def load_all_networks():

    networks = {}

    A1, node_order1 = load_empirical_adjacency(
        "day1"
    )

    A2, node_order2 = load_empirical_adjacency(
        "day2"
    )

    if not np.array_equal(
        node_order1,
        node_order2
    ):

        raise ValueError(
            "Day 1 and day 2 node orders differ."
        )

    node_order = node_order1

    if A1.shape != A2.shape:

        raise ValueError(
            "Empirical network dimensions differ."
        )

    communities_day1 = load_community_assignments(
        node_order,
        "day1"
    )

    communities_day2 = load_community_assignments(
        node_order,
        "day2"
    )

    networks["Empirical"] = {

        "day1": validate_network(
            A1,
            "Empirical day 1"
        ),

        "day2": validate_network(
            A2,
            "Empirical day 2"
        ),

        "communities_day1": communities_day1,

        "communities_day2": communities_day2

    }

    # --------------------------------------------------------
    # LOAD NULL MODELS
    # --------------------------------------------------------

    for model in [
        "ER",
        "Configuration",
        "DCSBM"
    ]:

        A1_null = load_null_adjacency(
            model,
            "day1"
        )

        A2_null = load_null_adjacency(
            model,
            "day2"
        )

        if (
            A1_null.shape != A1.shape or
            A2_null.shape != A2.shape
        ):

            raise ValueError(
                f"{model}: incorrect network dimensions."
            )

        networks[model] = {

            "day1": validate_network(
                A1_null,
                f"{model} day 1"
            ),

            "day2": validate_network(
                A2_null,
                f"{model} day 2"
            ),

            # Use the empirical spectral community
            # assignments as a common reference.
            "communities_day1": communities_day1,

            "communities_day2": communities_day2

        }

    return networks, node_order


# ============================================================
# INITIALIZE NETWORKS
# ============================================================

print("\n")
print("=" * 60)
print("LOADING PAIRED NETWORKS")
print("=" * 60)

NETWORKS, NODE_ORDER = load_all_networks()

N = len(NODE_ORDER)

print(
    "Participants:",
    N
)

for model in NETWORK_MODELS:

    print(
        f"{model}: "
        f"Day 1 = {NETWORKS[model]['day1'].shape}, "
        f"Day 2 = {NETWORKS[model]['day2'].shape}"
    )

print(
    "\nAll eight network matrices loaded successfully."
)



# ============================================================
# VACCINATION RANKING FUNCTIONS
# ============================================================

def calculate_vaccination_scores(
    A,
    communities
):

    A = validate_network(
        A,
        "Vaccination network"
    )

    n = A.shape[0]

    communities = np.asarray(
        communities
    )

    if len(communities) != n:

        raise ValueError(
            "Community assignments must match network size."
        )

    # --------------------------------------------------------
    # BUILD NETWORKX GRAPH
    # --------------------------------------------------------

    G = nx.from_numpy_array(
        A,
        create_using=nx.Graph
    )

    # --------------------------------------------------------
    # DEGREE
    # --------------------------------------------------------

    degree = np.array([
        G.degree(i)
        for i in range(n)
    ], dtype=float)

    # --------------------------------------------------------
    # WEIGHTED DEGREE / STRENGTH
    # --------------------------------------------------------

    strength = A.sum(axis=1)

    # --------------------------------------------------------
    # WEIGHTED BETWEENNESS
    # --------------------------------------------------------

    for u, v, data in G.edges(data=True):

        weight = data["weight"]

        data["distance"] = (
            1.0 / weight
            if weight > 0
            else np.inf
        )

    betweenness_dict = nx.betweenness_centrality(
        G,
        weight="distance",
        normalized=True
    )

    betweenness = np.array([
        betweenness_dict[i]
        for i in range(n)
    ])

    # --------------------------------------------------------
    # COMMUNITY-BRIDGE ENTROPY
    # --------------------------------------------------------

    unique_communities = np.unique(
        communities
    )

    K = len(unique_communities)

    bridge_entropy = np.zeros(n)

    for i in range(n):

        total_strength = strength[i]

        if total_strength <= 0:
            continue

        proportions = []

        for community in unique_communities:

            members = np.flatnonzero(
                communities == community
            )

            community_weight = A[
                i,
                members
            ].sum()

            proportions.append(
                community_weight / total_strength
            )

        proportions = np.asarray(
            proportions
        )

        positive = proportions[
            proportions > 0
        ]

        entropy = -np.sum(
            positive * np.log(positive)
        )

        if K > 1:

            bridge_entropy[i] = (
                entropy / np.log(K)
            )

    # --------------------------------------------------------
    # RETURN SCORES
    # --------------------------------------------------------

    return {

        "Degree": degree,

        "Weighted_Degree": strength,

        "Betweenness": betweenness,

        "Community_Bridge": bridge_entropy

    }


# ============================================================
# CREATE RANKINGS
# ============================================================

def create_rankings(
    scores
):

    rankings = {}

    for strategy, values in scores.items():

        rankings[strategy] = np.argsort(
            -np.asarray(values),
            kind="stable"
        )

    return rankings


# ============================================================
# PRECOMPUTE RANKINGS FOR ALL NETWORKS
# ============================================================

print("\n")
print("=" * 60)
print("CALCULATING VACCINATION RANKINGS")
print("=" * 60)

ALL_RANKINGS = {}

for model in NETWORK_MODELS:

    print(
        f"\nProcessing {model}"
    )

    for timing in SELECTION_TIMINGS:

        if timing == "Previous_Day":

            selection_day = "day1"

            community_key = "communities_day1"

        else:

            selection_day = "day2"

            community_key = "communities_day2"

        A = NETWORKS[model][selection_day]

        communities = NETWORKS[model][community_key]

        scores = calculate_vaccination_scores(
            A,
            communities
        )

        rankings = create_rankings(
            scores
        )

        for strategy in STRATEGIES:

            if strategy == "Random":
                continue

            ALL_RANKINGS[
                (model, timing, strategy)
            ] = rankings[strategy]

            print(
                f"{timing}: {strategy} complete"
            )


print(
    "\nAll vaccination rankings calculated."
)



# ============================================================
# STOCHASTIC SIR SIMULATION
# ============================================================

def run_sir(
    A,
    vaccinated,
    initial_infected,
    rng
):
    # If the initial infected individual is vaccinated,
    # the infection is blocked.

    if vaccinated[initial_infected]:

        return {

            "final_size": 0,

            "final_size_proportion": 0.0,

            "peak_infected": 0,

            "peak_proportion": 0.0,

            "generations": 0

        }

    else:

        A = np.asarray(A, dtype=float)
        n = A.shape[0]

        vaccinated = np.asarray(
            vaccinated,
            dtype=bool
        )

        if len(vaccinated) != n:

            raise ValueError(
                "Vaccination vector has incorrect length."
            )

        if vaccinated[initial_infected]:
            return {
                "final_size": 0,
                "final_size_proportion": 0.0,
                "peak_infected": 0,
                "peak_proportion": 0.0,
                "generations": 0
            }

        # --------------------------------------------------------
        # INITIAL CONDITIONS
        # --------------------------------------------------------

        ever_infected = np.zeros(
            n,
            dtype=bool
        )

        ever_infected[
            initial_infected
        ] = True

        infectious = [
            initial_infected
        ]

        final_size = 1

        peak_infected = 1

        generation = 0

        # --------------------------------------------------------
        # EPIDEMIC GENERATIONS
        # --------------------------------------------------------

        while len(infectious) > 0:

            generation += 1

            # Identify susceptible individuals.
            susceptible = np.flatnonzero(
                (~vaccinated) &
                (~ever_infected)
            )

            if len(susceptible) == 0:
                break

            # Combined contact duration from all
            # currently infectious individuals.
            exposure = A[
                np.ix_(
                    infectious,
                    susceptible
                )
            ].sum(axis=0)

            # Combined transmission probability.
            probability = (
                1.0 -
                np.exp(-BETA * exposure)
            )

            # Stochastic transmission.
            random_values = rng.random(
                len(susceptible)
            )

            newly_infected = susceptible[
                random_values < probability
            ]

            if len(newly_infected) == 0:
                break

            # Newly infected become infectious
            # in the next generation.
            ever_infected[
                newly_infected
            ] = True

            final_size += len(
                newly_infected
            )

            infectious = newly_infected.tolist()

            peak_infected = max(
                peak_infected,
                len(infectious)
            )

        # --------------------------------------------------------
        # RETURN OUTCOMES
        # --------------------------------------------------------

        return {

            "final_size": int(final_size),

            "final_size_proportion": (
                final_size / n
            ),

            "peak_infected": int(peak_infected),

            "peak_proportion": (
                peak_infected / n
            ),

            "generations": generation

        }



# ============================================================
# GENERATE VACCINATION SELECTIONS
# ============================================================

def select_vaccinated(
    model,
    timing,
    strategy,
    budget,
    rng
):

    number_vaccinated = int(
        np.ceil(budget * N)
    )

    if strategy == "Random":

        selected = rng.choice(
            N,
            size=number_vaccinated,
            replace=False
        )

    else:

        ranking = ALL_RANKINGS[
            (model, timing, strategy)
        ]

        selected = ranking[
            :number_vaccinated
        ]

    return np.asarray(
        selected,
        dtype=int
    )


# ============================================================
# RUN COMPLETE PAIRED EXPERIMENT
# ============================================================

def run_paired_experiment():

    results = []

    total_experiments = (
        N_SIMULATIONS *
        len(VACCINATION_BUDGETS)
    )

    experiment_counter = 0

    print("\n")
    print("=" * 65)
    print("STARTING PAIRED SIR EXPERIMENT")
    print("=" * 65)

    print(
        "Simulations per setting:",
        N_SIMULATIONS
    )

    print(
        "Transmission rate:",
        BETA
    )

    # --------------------------------------------------------
    # LOOP OVER VACCINATION BUDGETS
    # --------------------------------------------------------

    for budget in VACCINATION_BUDGETS:

        number_vaccinated = int(
            np.ceil(budget * N)
        )

        print("\n")
        print("-" * 65)

        print(
            f"Vaccination budget: {budget:.0%}"
        )

        print(
            "Participants vaccinated:",
            number_vaccinated
        )

        # ----------------------------------------------------
        # LOOP OVER SIMULATION REPLICATES
        # ----------------------------------------------------

        for replicate in range(
            1,
            N_SIMULATIONS + 1
        ):

            # Store selections for every combination.
            selections = {}

            # ------------------------------------------------
            # GENERATE ALL VACCINATION SETS
            # ------------------------------------------------

            for model in NETWORK_MODELS:

                for timing in SELECTION_TIMINGS:

                    for strategy in STRATEGIES:

                        selected = select_vaccinated(
                            model,
                            timing,
                            strategy,
                            budget,
                            rng
                        )

                        selections[
                            (model, timing, strategy)
                        ] = selected

            # ------------------------------------------------
            # COMMON INITIAL INFECTION SEED
            # ------------------------------------------------

            initial_infected = int(
                rng.integers(0, N)
            )

            # ------------------------------------------------
            # RUN EVERY EXPERIMENT USING THE SAME SEED
            # ------------------------------------------------

            for model in NETWORK_MODELS:

                # Every experiment uses its corresponding
                # day-2 outbreak network.
                A_outbreak = NETWORKS[
                    model
                ]["day2"]

                for timing in SELECTION_TIMINGS:

                    for strategy in STRATEGIES:

                        selected = selections[
                            (model, timing, strategy)
                        ]

                        vaccinated = np.zeros(
                            N,
                            dtype=bool
                        )

                        vaccinated[
                            selected
                        ] = True

                        # Each simulation receives its
                        # own stochastic transmission draws.
                        simulation_rng = np.random.default_rng(
                            rng.integers(0, 2**32)
                        )

                        outcome = run_sir(
                            A=A_outbreak,
                            vaccinated=vaccinated,
                            initial_infected=initial_infected,
                            rng=simulation_rng
                        )

                        # ------------------------------------
                        # SAVE RESULT
                        # ------------------------------------

                        results.append({

                            "model": model,

                            "selection_timing": timing,

                            "strategy": strategy,

                            "vaccination_budget": budget,

                            "number_vaccinated": len(
                                selected
                            ),

                            "replicate": replicate,

                            "initial_infected": NODE_ORDER[
                                initial_infected
                            ],

                            "final_size": outcome[
                                "final_size"
                            ],

                            "final_size_proportion": outcome[
                                "final_size_proportion"
                            ],

                            "peak_infected": outcome[
                                "peak_infected"
                            ],

                            "peak_proportion": outcome[
                                "peak_proportion"
                            ],

                            "generations": outcome[
                                "generations"
                            ]

                        })

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if replicate % 25 == 0:

                print(
                    f"Budget {budget:.0%}: "
                    f"completed replicate "
                    f"{replicate}/{N_SIMULATIONS}"
                )

        experiment_counter += 1

    print("\n")
    print("=" * 65)
    print("PAIRED SIR EXPERIMENT COMPLETE")
    print("=" * 65)

    return pd.DataFrame(results)


# ============================================================
# EXECUTE EXPERIMENT
# ============================================================

simulation_results = run_paired_experiment()


# ============================================================
# SAVE INDIVIDUAL SIMULATION RESULTS
# ============================================================

individual_path = (
    TABLE_DIR /
    "null_model_sir_individual_results.csv"
)

simulation_results.to_csv(
    individual_path,
    index=False
)

print(
    "\nIndividual simulation results saved:"
)

print(individual_path)

print(
    "Number of simulation records:",
    len(simulation_results)
)



# ============================================================
# SUMMARY STATISTICS
# ============================================================

def summarize_simulations(df):

    grouping = [
        "model",
        "selection_timing",
        "strategy",
        "vaccination_budget"
    ]

    summary = (

        df.groupby(grouping)

        .agg(

            simulations=("final_size", "count"),

            mean_final_size=(
                "final_size",
                "mean"
            ),

            sd_final_size=(
                "final_size",
                "std"
            ),

            median_final_size=(
                "final_size",
                "median"
            ),

            mean_peak_infected=(
                "peak_infected",
                "mean"
            ),

            sd_peak_infected=(
                "peak_infected",
                "std"
            ),

            median_peak_infected=(
                "peak_infected",
                "median"
            ),

            mean_generations=(
                "generations",
                "mean"
            )

        )

        .reset_index()

    )

    # --------------------------------------------------------
    # 95% CONFIDENCE INTERVALS
    # --------------------------------------------------------

    summary["se_final_size"] = (
        summary["sd_final_size"] /
        np.sqrt(summary["simulations"])
    )

    summary["se_peak"] = (
        summary["sd_peak_infected"] /
        np.sqrt(summary["simulations"])
    )

    summary["final_ci_lower"] = (
        summary["mean_final_size"] -
        1.96 * summary["se_final_size"]
    )

    summary["final_ci_upper"] = (
        summary["mean_final_size"] +
        1.96 * summary["se_final_size"]
    )

    summary["peak_ci_lower"] = (
        summary["mean_peak_infected"] -
        1.96 * summary["se_peak"]
    )

    summary["peak_ci_upper"] = (
        summary["mean_peak_infected"] +
        1.96 * summary["se_peak"]
    )

    # --------------------------------------------------------
    # PROPORTIONAL OUTBREAK SIZE
    # --------------------------------------------------------

    summary["mean_final_proportion"] = (
        summary["mean_final_size"] / N
    )

    summary["mean_peak_proportion"] = (
        summary["mean_peak_infected"] / N
    )

    return summary


# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

simulation_summary = summarize_simulations(
    simulation_results
)


# ============================================================
# SAVE SUMMARY TABLE
# ============================================================

summary_path = (
    TABLE_DIR /
    "null_model_sir_summary.csv"
)

simulation_summary.to_csv(
    summary_path,
    index=False
)

print("\n")
print("=" * 65)
print("SIR SUMMARY STATISTICS")
print("=" * 65)

print(
    simulation_summary.head(20).to_string(
        index=False
    )
)

print(
    "\nSummary saved:"
)

print(summary_path)


# ============================================================
# SAVE BEST STRATEGIES
# ============================================================

best_strategies = (

    simulation_summary

    .sort_values(
        [
            "model",
            "selection_timing",
            "vaccination_budget",
            "mean_final_size"
        ]
    )

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

best_path = (
    TABLE_DIR /
    "null_model_best_strategies.csv"
)

best_strategies.to_csv(
    best_path,
    index=False
)

print(
    "\nBest strategy table saved:"
)

print(best_path)


# ============================================================
# FINAL DIAGNOSTICS
# ============================================================

print("\n")
print("=" * 65)
print("CHUNK 8C COMPLETE")
print("=" * 65)

print(
    "Total simulation records:",
    len(simulation_results)
)

print(
    "Network models:",
    len(NETWORK_MODELS)
)

print(
    "Vaccination strategies:",
    len(STRATEGIES)
)

print(
    "Vaccination budgets:",
    len(VACCINATION_BUDGETS)
)

print(
    "Selection timings:",
    len(SELECTION_TIMINGS)
)

print(
    "\nOutput files:"
)

print(individual_path)
print(summary_path)
print(best_path)

print(
    "\nAll experiments finished."
)

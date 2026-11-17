
"""
STAT 636 Project
Chunk 6A: Stochastic SIR Epidemic Simulation

Implements a discrete-time stochastic SIR model
on the Day 2 weighted contact network.

States:
    0 = Susceptible
    1 = Infected
    2 = Recovered
    3 = Vaccinated
"""

import numpy as np
import pandas as pd
import networkx as nx

from pathlib import Path

import config


# ============================================================
# 1. SIMULATION PARAMETERS
# ============================================================

RANDOM_SEED = 636

N_SIMULATIONS = 500

INFECTIOUS_PERIOD = 3

TRANSMISSION_RATES = [
    0.00005,
    0.0001,
    0.0002,
    0.0005,
]

VACCINATION_BUDGETS = [
    0.05,
    0.10,
    0.15,
    0.20
]

# Ensure output directories exist.
config.RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

config.TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. PREPARE NETWORK FOR SIMULATION
# ============================================================

def prepare_simulation_network(
    G,
    node_order
):
    """
    Convert the NetworkX graph into an efficient
    adjacency-list representation.

    All nodes must be represented, including isolates.

    Returns:
        adjacency: List of (neighbor_index, duration)
        node_to_index: Participant ID to array index
    """

    node_to_index = {
        node: i
        for i, node in enumerate(node_order)
    }

    adjacency = [
        []
        for _ in node_order
    ]

    for u, v, data in G.edges(data=True):

        i = node_to_index[u]
        j = node_to_index[v]

        duration = float(
            data.get("duration", 0)
        )

        if duration <= 0:
            raise ValueError(
                f"Invalid duration on edge {u}-{v}"
            )

        adjacency[i].append(
            (j, duration)
        )

        adjacency[j].append(
            (i, duration)
        )

    return adjacency, node_to_index


# ============================================================
# 3. INITIAL INFECTION SELECTION
# ============================================================

def choose_initial_infection(
    node_order,
    vaccinated,
    seed_order
):
    """
    Select the first unvaccinated participant from
    a shared random candidate ordering.

    This allows paired comparisons across strategies.
    """

    vaccinated_set = set(vaccinated)

    for candidate in seed_order:

        if candidate not in vaccinated_set:

            return candidate

    raise ValueError(
        "No susceptible participant available "
        "for initial infection."
    )


# ============================================================
# 4. SIR SIMULATION
# ============================================================

def simulate_sir(
    adjacency,
    node_order,
    vaccinated,
    beta,
    infectious_period=3,
    seed_order=None,
    random_state=None
):
    """
    Run one stochastic SIR simulation.

    Transmission probability:

        P(i -> j) = 1 - exp(-beta * duration_ij)

    The observed daily contact network is repeated
    once per infectious day.

    Infection attempts are synchronous:
        All currently infected individuals attempt
        transmission before new infections are added.

    Recovery occurs after infectious_period days.

    Parameters
    ----------
    adjacency:
        Prepared adjacency list.

    node_order:
        Ordered participant identifiers.

    vaccinated:
        List of vaccinated participant IDs.

    beta:
        Transmission rate per second of contact.

    infectious_period:
        Number of daily simulation steps an individual
        remains infectious.

    seed_order:
        Shared ordered list of initial-case candidates.

    random_state:
        Random seed for reproducibility.

    Returns
    -------
    Dictionary containing:
        final_outbreak_size
        attack_rate
        peak_infected
        outbreak_duration
        initial_case
        daily_infected
    """

    rng = np.random.default_rng(
        random_state
    )

    n = len(node_order)

    node_to_index = {
        node: i
        for i, node in enumerate(node_order)
    }

    # ----------------------------------------
    # Initialize participant states.
    # ----------------------------------------

    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
    VACCINATED = 3

    state = np.full(
        n,
        SUSCEPTIBLE,
        dtype=np.int8
    )

    vaccinated_indices = [
        node_to_index[node]
        for node in vaccinated
    ]

    state[vaccinated_indices] = VACCINATED

    vaccinated_set = set(vaccinated)

    # ----------------------------------------
    # Select initial infected participant.
    # ----------------------------------------

    if seed_order is None:

        seed_order = rng.permutation(
            node_order
        ).tolist()

    initial_case = choose_initial_infection(
        node_order,
        vaccinated_set,
        seed_order
    )

    initial_index = node_to_index[
        initial_case
    ]

    state[initial_index] = INFECTED

    # Age records the number of completed
    # infectious days for each participant.
    infection_age = np.zeros(
        n,
        dtype=int
    )

    # Record everyone who has ever been infected.
    ever_infected = np.zeros(
        n,
        dtype=bool
    )

    ever_infected[initial_index] = True

    initial_infected_count = 1

    peak_infected = initial_infected_count

    daily_infected = [
        initial_infected_count
    ]

    # ----------------------------------------
    # Main epidemic loop.
    # ----------------------------------------

    day = 0

    while np.any(state == INFECTED):

        day += 1

        infected_indices = np.where(
            state == INFECTED
        )[0]

        # Probability that each susceptible
        # participant escapes all exposures.
        failure_probability = np.ones(n)

        # ------------------------------------
        # A. Transmission attempts
        # ------------------------------------

        for i in infected_indices:

            for j, duration in adjacency[i]:

                if state[j] == SUSCEPTIBLE:

                    transmission_probability = (
                        -np.expm1(
                            -beta * duration
                        )
                    )

                    failure_probability[j] *= (
                        1.0 - transmission_probability
                    )

        # ------------------------------------
        # B. Determine new infections
        # ------------------------------------

        susceptible_indices = np.where(
            state == SUSCEPTIBLE
        )[0]

        infection_probability = (
            1.0 - failure_probability[
                susceptible_indices
            ]
        )


        infection_draws = rng.random(
            len(susceptible_indices)
        )

        newly_infected = susceptible_indices[
            infection_draws < infection_probability
        ]

        # ------------------------------------
        # C. Age existing infections
        # ------------------------------------

        infection_age[infected_indices] += 1

        recovered_indices = infected_indices[
            infection_age[infected_indices]
            >= infectious_period
        ]

        state[recovered_indices] = RECOVERED

        # ------------------------------------
        # D. Add new infections
        # ------------------------------------

        state[newly_infected] = INFECTED

        infection_age[newly_infected] = 0

        ever_infected[newly_infected] = True

        # ------------------------------------
        # E. Record epidemic statistics
        # ------------------------------------

        current_infected = int(
            np.sum(state == INFECTED)
        )

        daily_infected.append(
            current_infected
        )

        peak_infected = max(
            peak_infected,
            current_infected
        )

        # Safety check against an unintended
        # infinite simulation.
        if day > 365:

            raise RuntimeError(
                "Simulation exceeded 365 days."
            )

    # ----------------------------------------
    # Final results
    # ----------------------------------------

    final_outbreak_size = int(
        np.sum(ever_infected)
    )

    number_unvaccinated = (
        n - len(vaccinated_set)
    )

    attack_rate = (
        final_outbreak_size /
        number_unvaccinated
    )

    return {

        "final_outbreak_size":
            final_outbreak_size,

        "attack_rate":
            attack_rate,

        "peak_infected":
            peak_infected,

        "peak_fraction":
            peak_infected / number_unvaccinated,

        "outbreak_duration":
            day,

        "initial_case":
            initial_case,

        "daily_infected":
            daily_infected

    }



# ============================================================
# 5. LOAD COMMUNITY ASSIGNMENTS
# ============================================================

def load_community_assignments(node_order):
    """
    Load Day 1 and Day 2 community assignments.

    Participant IDs are standardized as strings.
    """

    community_file = (
        config.TABLE_DIR /
        "community_assignments_both_days.csv"
    )

    if not community_file.exists():

        raise FileNotFoundError(
            "Community assignments not found. "
            "Run spectral_analysis.py first."
        )

    communities = pd.read_csv(
        community_file
    )

    communities["id"] = (
        communities["id"]
        .astype(str)
        .str.strip()
    )

    communities = communities.set_index(
        "id"
    )

    ordered_ids = [
        str(node).strip()
        for node in node_order
    ]

    communities = communities.reindex(
        ordered_ids
    )

    required = [
        "community_day1",
        "community_day2"
    ]

    if communities[required].isna().any().any():

        raise ValueError(
            "Missing community assignments. "
            "Check participant IDs."
        )

    return (
        communities["community_day1"].to_numpy(),
        communities["community_day2"].to_numpy()
    )


# ============================================================
# 6. GENERATE PAIRED INITIAL INFECTION CANDIDATES
# ============================================================

def generate_seed_order(
    node_order,
    replicate
):
    """
    Generate one random participant ordering per replicate.

    The same ordering is shared across strategies,
    vaccination budgets, selection timings, and beta values.

    If the first candidate is vaccinated, the next
    eligible participant becomes the initial case.
    """

    rng = np.random.default_rng(
        RANDOM_SEED + replicate
    )

    return rng.permutation(
        np.asarray(node_order, dtype=object)
    ).tolist()


# ============================================================
# 7. CALCULATE VACCINATION SCORES FOR BOTH DAYS
# ============================================================

def calculate_day_scores(
    G1,
    G2,
    node_order,
    labels1,
    labels2
):
    """
    Calculate deterministic vaccination scores
    using the corresponding daily network.

    Day 1 scores:
        Previous-day vaccination selection.

    Day 2 scores:
        Same-day vaccination selection.
    """

    from vaccination import (
        calculate_all_strategy_scores
    )

    print("\nCalculating Day 1 vaccination scores...")

    scores_day1 = calculate_all_strategy_scores(
        G1,
        node_order,
        labels1
    )

    print("Calculating Day 2 vaccination scores...")

    scores_day2 = calculate_all_strategy_scores(
        G2,
        node_order,
        labels2
    )

    return {
        "Day 1": scores_day1,
        "Day 2": scores_day2
    }


# ============================================================
# 8. CREATE A VACCINATION SET
# ============================================================

def create_vaccination_set(
    scores,
    node_order,
    strategy,
    budget,
    replicate,
    budget_index,
    strategy_index
):
    """
    Generate a vaccination set.

    Random vaccination receives a different random
    sample for each replicate and budget.

    Deterministic strategies use the fixed rankings.
    """

    from vaccination import select_vaccinated

    random_state = (
        RANDOM_SEED
        + replicate * 10000
        + budget_index * 100
        + strategy_index
    )

    vaccinated = select_vaccinated(
        scores=scores,
        node_order=node_order,
        strategy=strategy,
        budget=budget,
        random_state=random_state
    )

    return vaccinated


# ============================================================
# 9. RUN ONE PAIRED EXPERIMENT
# ============================================================

def run_one_experiment(
    G2,
    adjacency,
    node_order,
    scores,
    selection_day,
    strategy,
    budget,
    beta,
    replicate,
    seed_order,
    infectious_period=INFECTIOUS_PERIOD,
    budget_index=0,
    strategy_index=0,
    beta_index=0
):
    """
    Run one SIR experiment.

    The outbreak always occurs on Day 2.

    selection_day determines whether vaccination
    rankings come from Day 1 or Day 2.
    """

    # ----------------------------------------
    # Select vaccinated participants.
    # ----------------------------------------

    vaccinated = create_vaccination_set(
        scores=scores,
        node_order=node_order,
        strategy=strategy,
        budget=budget,
        replicate=replicate,
        budget_index=budget_index,
        strategy_index=strategy_index
    )

    # ----------------------------------------
    # Set transmission random seed.
    # ----------------------------------------

    transmission_seed = (
        RANDOM_SEED
        + replicate * 1000000
        + budget_index * 10000
        + beta_index * 100
    )

    # ----------------------------------------
    # Run epidemic.
    # ----------------------------------------

    result = simulate_sir(
        adjacency=adjacency,
        node_order=node_order,
        vaccinated=vaccinated,
        beta=beta,
        infectious_period=infectious_period,
        seed_order=seed_order,
        random_state=transmission_seed
    )

    # ----------------------------------------
    # Store experimental information.
    # ----------------------------------------

    return {

        "replicate": replicate,

        "selection_day": selection_day,

        "outbreak_day": "Day 2",

        "strategy": strategy,

        "budget": budget,

        "n_vaccinated": len(vaccinated),

        "beta": beta,

        "infectious_period": infectious_period,

        "initial_case": result["initial_case"],

        "final_outbreak_size":
            result["final_outbreak_size"],

        "attack_rate":
            result["attack_rate"],

        "peak_infected":
            result["peak_infected"],

        "peak_fraction":
            result["peak_fraction"],

        "outbreak_duration":
            result["outbreak_duration"]

    }


# ============================================================
# 10. RUN THE COMPLETE EXPERIMENT
# ============================================================

def run_full_experiment(
    n_simulations=N_SIMULATIONS,
    transmission_rates=TRANSMISSION_RATES,
    vaccination_budgets=VACCINATION_BUDGETS,
    infectious_period=INFECTIOUS_PERIOD
):
    """
    Run all experimental conditions.

    Experimental dimensions:

        2 selection days
        6 vaccination strategies
        4 vaccination budgets
        4 transmission rates
        n_simulations replicates

    Every epidemic occurs on the Day 2 network.
    """

    from data_loader import load_school_networks

    from vaccination import STRATEGIES

    print("\n====================================")
    print("STAT 636 SIR SIMULATION")
    print("====================================")

    # ----------------------------------------
    # A. Load networks.
    # ----------------------------------------

    data = load_school_networks()

    G1 = data.G1
    G2 = data.G2

    node_order = [
        str(node).strip()
        for node in data.node_order
    ]

    # Ensure network IDs match node_order.
    G1 = nx.relabel_nodes(
        G1,
        {
            node: str(node).strip()
            for node in G1.nodes()
        }
    )

    G2 = nx.relabel_nodes(
        G2,
        {
            node: str(node).strip()
            for node in G2.nodes()
        }
    )

    # ----------------------------------------
    # B. Load community assignments.
    # ----------------------------------------

    labels1, labels2 = load_community_assignments(
        node_order
    )

    # ----------------------------------------
    # C. Prepare Day 2 outbreak network.
    # ----------------------------------------

    adjacency, node_to_index = (
        prepare_simulation_network(
            G2,
            node_order
        )
    )

    # ----------------------------------------
    # D. Calculate Day 1 and Day 2 scores.
    # ----------------------------------------

    day_scores = calculate_day_scores(
        G1,
        G2,
        node_order,
        labels1,
        labels2
    )

    # ----------------------------------------
    # E. Run experimental conditions.
    # ----------------------------------------

    results = []

    selection_days = [
        "Day 1",
        "Day 2"
    ]

    total_conditions = (
        len(selection_days)
        * len(STRATEGIES)
        * len(vaccination_budgets)
        * len(transmission_rates)
        * n_simulations
    )

    print("\nTotal simulation runs:", total_conditions)

    print("\nStarting epidemic simulations...")

    for day in selection_days:

        print(f"\nSelection information: {day}")

        scores = day_scores[day]

        for budget_index, budget in enumerate(
            vaccination_budgets
        ):

            print(
                f"Vaccination budget: {budget:.0%}"
            )

            for beta_index, beta in enumerate(
                transmission_rates
            ):

                for strategy_index, strategy in enumerate(
                    STRATEGIES
                ):

                    for replicate in range(
                        n_simulations
                    ):

                        # Same candidate ordering across
                        # strategies within each replicate.
                        seed_order = generate_seed_order(
                            node_order,
                            replicate
                        )

                        experiment = run_one_experiment(
                            G2=G2,
                            adjacency=adjacency,
                            node_order=node_order,
                            scores=scores,
                            selection_day=day,
                            strategy=strategy,
                            budget=budget,
                            beta=beta,
                            replicate=replicate,
                            seed_order=seed_order,
                            infectious_period=infectious_period,
                            budget_index=budget_index,
                            strategy_index=strategy_index,
                            beta_index=beta_index
                        )

                        results.append(experiment)

    # ----------------------------------------
    # F. Save individual simulation results.
    # ----------------------------------------

    results_df = pd.DataFrame(results)

    output_file = (
        config.TABLE_DIR /
        "sir_simulation_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nIndividual simulation results saved.")

    print(
        "Output:",
        output_file
    )

    # ----------------------------------------
    # G. Generate summary statistics.
    # ----------------------------------------

    summary = summarize_simulations(
        results_df
    )

    summary_file = (
        config.TABLE_DIR /
        "sir_simulation_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\nSimulation summary saved.")

    print(
        "Output:",
        summary_file
    )

    print("\n====================================")
    print("SIR SIMULATION COMPLETED")
    print("====================================")

    return {
        "results": results_df,
        "summary": summary
    }


# ============================================================
# 11. SUMMARIZE SIMULATION RESULTS
# ============================================================

def summarize_simulations(results):
    """
    Calculate means, standard deviations,
    and approximate 95% confidence intervals.

    Confidence intervals are calculated across
    independent simulation replicates within
    each experimental condition.
    """

    from scipy.stats import t

    grouping = [
        "selection_day",
        "outbreak_day",
        "strategy",
        "budget",
        "beta",
        "infectious_period"
    ]

    outcome_columns = [
        "final_outbreak_size",
        "attack_rate",
        "peak_infected",
        "peak_fraction",
        "outbreak_duration"
    ]

    summaries = []

    for keys, group in results.groupby(
        grouping,
        sort=True
    ):

        record = dict(
            zip(grouping, keys)
        )

        n = len(group)

        record["n_simulations"] = n

        for outcome in outcome_columns:

            values = group[outcome].to_numpy(
                dtype=float
            )

            mean = np.mean(values)

            sd = np.std(
                values,
                ddof=1
            ) if n > 1 else 0.0

            if n > 1:

                standard_error = sd / np.sqrt(n)

                critical_value = t.ppf(
                    0.975,
                    df=n - 1
                )

                margin = (
                    critical_value *
                    standard_error
                )

            else:

                margin = 0.0

            record[f"mean_{outcome}"] = mean

            record[f"sd_{outcome}"] = sd

            record[f"ci_lower_{outcome}"] = (
                max(0.0, mean - margin)
            )

            record[f"ci_upper_{outcome}"] = (
                mean + margin
            )

        summaries.append(record)

    return pd.DataFrame(summaries)


# ============================================================
# 12. EXECUTE DIRECTLY
# ============================================================

if __name__ == "__main__":

    results = run_full_experiment()

    print("\nSUMMARY PREVIEW:")

    print(
        results["summary"].head(20).to_string(
            index=False
        )
    )

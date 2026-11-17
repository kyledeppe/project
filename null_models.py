
"""
STAT 636 PROJECT
Chunk 8A: Null Model Generation

Models:
1. Erdős-Rényi
2. Configuration Model
3. Degree-Corrected SBM

Input:
    Aligned weighted adjacency matrix
    Node community assignments

Output:
    Simulated weighted adjacency matrices
"""

import numpy as np
import networkx as nx

from pathlib import Path

import config


# ============================================================
# 1. GENERAL VALIDATION
# ============================================================

def validate_adjacency(A):
    """Validate an undirected weighted adjacency matrix."""

    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    if not np.all(np.isfinite(A)):
        raise ValueError("Adjacency matrix contains non-finite values.")

    if np.any(A < 0):
        raise ValueError("Edge weights cannot be negative.")

    if not np.allclose(A, A.T):
        raise ValueError("Adjacency matrix must be symmetric.")

    if not np.allclose(np.diag(A), 0):
        raise ValueError("Self-loops are not allowed.")

    return A


def extract_empirical_edges(A):
    """Return upper-triangle empirical edge weights."""

    rows, cols = np.triu_indices(
        A.shape[0],
        k=1
    )

    mask = A[rows, cols] > 0

    return A[rows[mask], cols[mask]]


def assign_weights_to_edges(
    adjacency,
    empirical_weights,
    rng
):
    """
    Assign empirical edge weights randomly to
    generated edges.

    This preserves the empirical marginal
    distribution of positive edge weights,
    but not individual node strengths.
    """

    rows, cols = np.where(
        np.triu(adjacency, k=1) > 0
    )

    number_edges = len(rows)

    if number_edges == 0:
        return adjacency

    if len(empirical_weights) == 0:
        raise ValueError("No empirical weights available.")

    sampled_weights = rng.choice(
        empirical_weights,
        size=number_edges,
        replace=True
    )

    adjacency[rows, cols] = sampled_weights

    adjacency[cols, rows] = sampled_weights

    return adjacency


# ============================================================
# 2. ERDŐS-RÉNYI MODEL
# ============================================================

def generate_er(
    A,
    seed=None
):
    """
    Generate an undirected ER graph.

    Preserves:
        Number of nodes
        Expected density

    Does not preserve:
        Degree sequence
        Classroom structure
        Specific connections
    """

    A = validate_adjacency(A)

    rng = np.random.default_rng(seed)

    n = A.shape[0]

    number_possible_edges = n * (n - 1) / 2

    number_observed_edges = np.count_nonzero(
        np.triu(A, k=1)
    )

    p = (
        number_observed_edges /
        number_possible_edges
    )

    graph = nx.gnp_random_graph(
        n,
        p,
        seed=seed
    )

    generated = nx.to_numpy_array(
        graph,
        nodelist=range(n)
    )

    empirical_weights = extract_empirical_edges(A)

    generated = assign_weights_to_edges(
        generated,
        empirical_weights,
        rng
    )

    return generated


# ============================================================
# 3. CONFIGURATION MODEL
# ============================================================

def generate_configuration(
    A,
    seed=None
):
    """
    Generate a configuration-model graph.

    Preserves:
        Expected degree sequence through
        random stub pairing.

    Does not preserve:
        Original classroom structure
        Specific connections

    Notes:
        Self-loops and multiedges are removed.
        Consequently, the final simple graph
        may not exactly preserve every degree.
    """

    A = validate_adjacency(A)

    rng = np.random.default_rng(seed)

    n = A.shape[0]

    degrees = np.count_nonzero(
        A,
        axis=1
    ).astype(int)

    if degrees.sum() == 0:
        return np.zeros_like(A)

    if degrees.sum() % 2 == 1:

        # Make the stub total even.
        eligible = np.flatnonzero(
            degrees > 0
        )

        if len(eligible) == 0:
            return np.zeros_like(A)

        degrees[
            rng.choice(eligible)
        ] -= 1

    graph = nx.configuration_model(
        degrees.tolist(),
        seed=seed
    )

    graph = nx.Graph(graph)

    graph.remove_edges_from(
        nx.selfloop_edges(graph)
    )

    # Preserve all n participants.
    graph.add_nodes_from(
        range(n)
    )

    generated = nx.to_numpy_array(
        graph,
        nodelist=range(n)
    )

    empirical_weights = extract_empirical_edges(A)

    generated = assign_weights_to_edges(
        generated,
        empirical_weights,
        rng
    )

    return generated


# ============================================================
# 4. DEGREE-CORRECTED SBM
# ============================================================


def generate_dcsbm(
    A,
    communities,
    seed=None
):
    """
    Generate an approximate weighted DCSBM.

    Normalizes block probabilities by the
    degree-correction parameters so that the
    expected edge counts better match the
    empirical block mixing structure.

    This remains an approximate generative model,
    not a full weighted DCSBM likelihood fit.
    """

    A = validate_adjacency(A)

    rng = np.random.default_rng(seed)

    n = A.shape[0]

    communities = np.asarray(communities)

    if len(communities) != n:
        raise ValueError(
            "Community assignments must match node count."
        )

    _, labels = np.unique(
        communities,
        return_inverse=True
    )

    K = len(np.unique(labels))

    degrees = np.count_nonzero(
        A,
        axis=1
    ).astype(float)

    # --------------------------------------------------------
    # Degree correction within communities
    # --------------------------------------------------------

    theta = np.zeros(n)

    for k in range(K):

        members = np.flatnonzero(
            labels == k
        )

        mean_degree = degrees[members].mean()

        if mean_degree > 0:

            theta[members] = (
                degrees[members] / mean_degree
            )

    # --------------------------------------------------------
    # Estimate normalized block probabilities
    # --------------------------------------------------------

    block_probability = np.zeros((K, K))

    for k in range(K):

        members_k = np.flatnonzero(
            labels == k
        )

        for l in range(k, K):

            members_l = np.flatnonzero(
                labels == l
            )

            submatrix = A[
                np.ix_(members_k, members_l)
            ]

            if k == l:

                observed = np.count_nonzero(
                    np.triu(submatrix, k=1)
                )

                theta_k = theta[members_k]

                denominator = (
                    theta_k.sum() ** 2
                    - np.sum(theta_k ** 2)
                ) / 2

            else:

                observed = np.count_nonzero(
                    submatrix
                )

                denominator = (
                    theta[members_k].sum()
                    * theta[members_l].sum()
                )

            if denominator <= 0:

                if observed > 0:

                    raise ValueError(
                        "Cannot fit block probability: "
                        f"block ({k}, {l}) has edges "
                        "but zero degree-correction mass."
                    )

                probability = 0.0

            else:

                probability = observed / denominator

            if probability > 1:

                raise ValueError(
                    f"Block ({k}, {l}) requires "
                    f"probability {probability:.4f} > 1. "
                    "The current degree correction "
                    "cannot represent this block."
                )

            block_probability[k, l] = probability
            block_probability[l, k] = probability

    # --------------------------------------------------------
    # Generate undirected graph
    # --------------------------------------------------------

    generated = np.zeros(
        (n, n),
        dtype=float
    )

    for i in range(n):

        for j in range(i + 1, n):

            k = labels[i]
            l = labels[j]

            probability = (
                block_probability[k, l]
                * theta[i]
                * theta[j]
            )

            probability = np.clip(
                probability,
                0,
                1
            )

            if rng.random() < probability:

                generated[i, j] = 1
                generated[j, i] = 1

    # --------------------------------------------------------
    # Assign empirical contact-duration weights
    # --------------------------------------------------------

    empirical_weights = extract_empirical_edges(A)

    generated = assign_weights_to_edges(
        generated,
        empirical_weights,
        rng
    )

    return generated


# ============================================================
# 5. GENERATE ALL THREE NULL MODELS
# ============================================================

def generate_all_null_models(
    A,
    communities,
    seed=636
):
    """
    Generate one replicate from each null model.
    """

    return {

        "ER": generate_er(
            A,
            seed=seed
        ),

        "Configuration": generate_configuration(
            A,
            seed=seed + 1
        ),

        "DCSBM": generate_dcsbm(
            A,
            communities,
            seed=seed + 2
        )

    }


# ============================================================
# 6. SAVE GENERATED NETWORKS
# ============================================================

def save_null_network(
    adjacency,
    model_name,
    day_name
):
    """
    Save generated adjacency matrices as CSV files.
    """

    output_directory = (
        config.RESULTS_DIR / "null_models"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        f"{model_name.lower()}_{day_name.lower()}.csv"
    )

    output_path = (
        output_directory / filename
    )

    np.savetxt(
        output_path,
        adjacency,
        delimiter=","
    )

    print(
        "Saved:",
        output_path
    )


# END OF CHUNK 8A

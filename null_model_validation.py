
"""
STAT 636 PROJECT
Chunk 8B: Null Model Validation

Evaluates:
1. Density
2. Degree distribution
3. Node strength
4. Classroom mixing
5. Edge-weight distribution
6. Similarity to the empirical network
"""

import numpy as np
import pandas as pd

from pathlib import Path

import config


# ============================================================
# 1. GENERAL VALIDATION
# ============================================================

def validate_matrix(A):

    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:

        raise ValueError(
            "Adjacency matrix must be square."
        )

    if not np.all(np.isfinite(A)):

        raise ValueError(
            "Adjacency matrix contains invalid values."
        )

    if not np.allclose(A, A.T):

        raise ValueError(
            "Adjacency matrix must be symmetric."
        )

    if np.any(A < 0):

        raise ValueError(
            "Negative edge weights detected."
        )

    if not np.allclose(np.diag(A), 0):

        raise ValueError(
            "Self-loops detected."
        )

    return A


# ============================================================
# 2. BASIC NETWORK STATISTICS
# ============================================================

def calculate_network_statistics(A):

    A = validate_matrix(A)

    n = A.shape[0]

    binary = (A > 0).astype(int)

    degrees = binary.sum(axis=1)

    strengths = A.sum(axis=1)

    number_edges = np.count_nonzero(
        np.triu(binary, k=1)
    )

    possible_edges = n * (n - 1) / 2

    density = (
        number_edges / possible_edges
        if possible_edges > 0
        else 0
    )

    weights = A[
        np.triu_indices(n, k=1)
    ]

    weights = weights[
        weights > 0
    ]

    return {

        "nodes": n,

        "edges": number_edges,

        "density": density,

        "mean_degree": degrees.mean(),

        "sd_degree": degrees.std(),

        "min_degree": degrees.min(),

        "max_degree": degrees.max(),

        "mean_strength": strengths.mean(),

        "sd_strength": strengths.std(),

        "mean_edge_weight": (
            weights.mean()
            if len(weights) else 0
        ),

        "sd_edge_weight": (
            weights.std()
            if len(weights) else 0
        ),

        "total_contact_weight": A.sum() / 2

    }


# ============================================================
# 3. DEGREE DISTRIBUTION
# ============================================================

def calculate_degree_distribution(A):

    A = validate_matrix(A)

    degrees = (
        A > 0
    ).sum(axis=1)

    values, counts = np.unique(
        degrees,
        return_counts=True
    )

    return pd.DataFrame({

        "degree": values,

        "count": counts,

        "proportion": counts / len(degrees)

    })


# ============================================================
# 4. DEGREE CORRELATION
# ============================================================

def degree_correlation(
    original_A,
    simulated_A
):
    """
    Pearson correlation between corresponding
    node degree sequences.

    This measures whether high-degree individuals
    in the original network remain high-degree
    in the simulated network.
    """

    original_degree = (
        validate_matrix(original_A) > 0
    ).sum(axis=1)

    simulated_degree = (
        validate_matrix(simulated_A) > 0
    ).sum(axis=1)

    if np.std(original_degree) == 0:

        return np.nan

    if np.std(simulated_degree) == 0:

        return np.nan

    return np.corrcoef(
        original_degree,
        simulated_degree
    )[0, 1]


# ============================================================
# 5. DEGREE DISTRIBUTION DISTANCE
# ============================================================

def degree_distribution_distance(
    original_A,
    simulated_A
):
    """
    Calculate total variation distance between
    empirical and simulated degree distributions.

    Range:
        0 = identical distributions
        1 = maximally different distributions
    """

    original_degree = (
        validate_matrix(original_A) > 0
    ).sum(axis=1)

    simulated_degree = (
        validate_matrix(simulated_A) > 0
    ).sum(axis=1)

    maximum_degree = max(
        original_degree.max(),
        simulated_degree.max()
    )

    original_counts = np.bincount(
        original_degree,
        minlength=maximum_degree + 1
    )

    simulated_counts = np.bincount(
        simulated_degree,
        minlength=maximum_degree + 1
    )

    original_prob = (
        original_counts / original_counts.sum()
    )

    simulated_prob = (
        simulated_counts / simulated_counts.sum()
    )

    return 0.5 * np.abs(
        original_prob - simulated_prob
    ).sum()



# ============================================================
# 6. CLASSROOM MIXING MATRIX
# ============================================================

def calculate_classroom_mixing(
    A,
    classroom_labels
):
    """
    Calculate total contact weight between classrooms.

    Rows represent the classroom of participant i.
    Columns represent the classroom of participant j.

    Diagonal entries represent within-classroom contact.
    Off-diagonal entries represent between-classroom contact.
    """

    A = validate_matrix(A)

    labels = np.asarray(
        classroom_labels
    )

    if len(labels) != A.shape[0]:

        raise ValueError(
            "Classroom labels do not match matrix size."
        )

    communities = np.unique(labels)

    K = len(communities)

    mixing = np.zeros(
        (K, K)
    )

    for i, class_i in enumerate(communities):

        members_i = np.flatnonzero(
            labels == class_i
        )

        for j, class_j in enumerate(communities):

            members_j = np.flatnonzero(
                labels == class_j
            )

            mixing[i, j] = A[
                np.ix_(members_i, members_j)
            ].sum()

    return mixing, communities


# ============================================================
# 7. WITHIN-CLASSROOM CONTACT PROPORTION
# ============================================================

def within_classroom_contact_proportion(
    A,
    classroom_labels
):
    """
    Fraction of total edge weight occurring
    within classrooms.
    """

    A = validate_matrix(A)

    labels = np.asarray(
        classroom_labels
    )

    if len(labels) != A.shape[0]:

        raise ValueError(
            "Classroom labels do not match matrix size."
        )

    total_weight = np.triu(
        A,
        k=1
    ).sum()

    if total_weight == 0:
        return np.nan

    same_class = (
        labels[:, None] == labels[None, :]
    )

    within_weight = np.triu(
        A * same_class,
        k=1
    ).sum()

    return (
        within_weight / total_weight
    )


# ============================================================
# 8. EDGE-WEIGHT DISTRIBUTION
# ============================================================

def calculate_edge_weights(A):

    A = validate_matrix(A)

    weights = A[
        np.triu_indices(
            A.shape[0],
            k=1
        )
    ]

    return weights[
        weights > 0
    ]


def compare_weight_distributions(
    original_A,
    simulated_A
):
    """
    Compare empirical and generated edge weights.
    """

    original_weights = calculate_edge_weights(
        original_A
    )

    simulated_weights = calculate_edge_weights(
        simulated_A
    )

    if (
        len(original_weights) == 0
        or len(simulated_weights) == 0
    ):

        return {

            "weight_mean_difference": np.nan,

            "weight_median_difference": np.nan,

            "weight_std_difference": np.nan

        }

    return {

        "weight_mean_difference": (
            simulated_weights.mean()
            - original_weights.mean()
        ),

        "weight_median_difference": (
            np.median(simulated_weights)
            - np.median(original_weights)
        ),

        "weight_std_difference": (
            simulated_weights.std()
            - original_weights.std()
        )

    }


# ============================================================
# 9. CLASSROOM MIXING SUMMARY
# ============================================================

def classroom_mixing_summary(
    A,
    classroom_labels
):

    mixing, communities = calculate_classroom_mixing(
        A,
        classroom_labels
    )

    total = mixing.sum()

    within = np.trace(
        mixing
    )

    between = total - within

    return {

        "within_class_weight": within / 2,

        "between_class_weight": between / 2,

        "within_class_proportion": (
            within / total
            if total > 0
            else np.nan
        ),

        "between_class_proportion": (
            between / total
            if total > 0
            else np.nan
        ),

        "number_of_classrooms": len(communities)

    }



# ============================================================
# 10. VALIDATE ONE NULL MODEL
# ============================================================

def validate_null_model(
    original_A,
    simulated_A,
    classroom_labels,
    model_name
):
    """
    Calculate all diagnostics for one generated network.
    """

    original_stats = calculate_network_statistics(
        original_A
    )

    simulated_stats = calculate_network_statistics(
        simulated_A
    )

    classroom_stats = classroom_mixing_summary(
        simulated_A,
        classroom_labels
    )

    weight_stats = compare_weight_distributions(
        original_A,
        simulated_A
    )

    results = {

        "model": model_name,

        "nodes": simulated_stats["nodes"],

        "edges": simulated_stats["edges"],

        "density": simulated_stats["density"],

        "mean_degree": simulated_stats["mean_degree"],

        "sd_degree": simulated_stats["sd_degree"],

        "original_mean_degree": original_stats["mean_degree"],

        "original_sd_degree": original_stats["sd_degree"],

        "degree_correlation": degree_correlation(
            original_A,
            simulated_A
        ),

        "degree_distribution_distance": (
            degree_distribution_distance(
                original_A,
                simulated_A
            )
        ),

        "mean_strength": simulated_stats["mean_strength"],

        "total_contact_weight": simulated_stats[
            "total_contact_weight"
        ],

        "within_class_proportion": (
            classroom_stats["within_class_proportion"]
        ),

        "between_class_proportion": (
            classroom_stats["between_class_proportion"]
        ),

        **weight_stats

    }

    return results


# ============================================================
# 11. COMPARE ALL NULL MODELS
# ============================================================

def compare_all_null_models(
    original_A,
    null_networks,
    classroom_labels
):
    """
    Validate every generated null model.
    """

    results = []

    for model_name, simulated_A in null_networks.items():

        result = validate_null_model(
            original_A,
            simulated_A,
            classroom_labels,
            model_name
        )

        results.append(
            result
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# 12. SAVE VALIDATION RESULTS
# ============================================================

def save_validation_results(
    results,
    filename="null_model_validation.csv"
):

    output_directory = (
        config.RESULTS_DIR / "tables"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_directory / filename
    )

    results.to_csv(
        output_path,
        index=False
    )

    print(
        "Saved validation results:",
        output_path
    )

    return output_path


# ============================================================
# 13. VALIDATION REPORT
# ============================================================

def print_validation_report(results):

    print("\n======================================")

    print("NULL MODEL VALIDATION REPORT")

    print("======================================")

    print(
        results.round(4).to_string(
            index=False
        )
    )

    print("\nINTERPRETATION")

    print(
        "ER should reproduce expected density, "
        "but not the original degree distribution."
    )

    print(
        "Configuration should approximately reproduce "
        "the original degree distribution."
    )

    print(
        "DCSBM should reflect the supplied community "
        "structure and degree heterogeneity."
    )

    print(
        "Classroom mixing and weight differences "
        "should be interpreted as diagnostic outcomes."
    )


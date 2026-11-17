
"""
STAT 636 PROJECT
Chunk 8B: Generate and validate null networks.

Models:
1. Erdős-Rényi
2. Configuration model
3. Degree-corrected stochastic block model

Uses existing empirical adjacency matrices and
community assignments.

Random seed: 636
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pathlib import Path

import config
import null_models


# ============================================================
# 1. DIRECTORIES
# ============================================================

TABLE_DIR = config.RESULTS_DIR / "tables"

NULL_DIR = config.RESULTS_DIR / "null_models"

NULL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\nSTAT 636 NULL MODEL EXPERIMENT")
print("=" * 60)


# ============================================================
# 2. LOAD SAVED CSR ADJACENCY MATRICES
# ============================================================

def load_adjacency(path):

    archive = np.load(
        path,
        allow_pickle=False
    )

    required = {
        "data",
        "indices",
        "indptr",
        "shape",
        "node_order"
    }

    if not required.issubset(archive.files):

        raise ValueError(
            f"Invalid adjacency archive: {path}"
        )

    shape = tuple(
        archive["shape"].astype(int)
    )

    A = sp.csr_matrix(
        (
            archive["data"],
            archive["indices"],
            archive["indptr"]
        ),
        shape=shape
    )

    node_order = archive["node_order"].astype(str)

    return A.toarray(), node_order


# ============================================================
# 3. LOAD COMMUNITY ASSIGNMENTS
# ============================================================

community_path = (
    TABLE_DIR /
    "community_assignments_both_days.csv"
)

communities_df = pd.read_csv(
    community_path
)

communities_df["id"] = (
    communities_df["id"].astype(str)
)

if communities_df["id"].duplicated().any():

    raise ValueError(
        "Duplicate participant IDs in community assignments."
    )


# ============================================================
# 4. ALIGN COMMUNITIES WITH NETWORK NODE ORDER
# ============================================================

def align_communities(
    node_order,
    day
):

    column = f"community_{day}"

    if column not in communities_df.columns:

        raise ValueError(
            f"Missing community column: {column}"
        )

    indexed = communities_df.set_index("id")

    missing = (
        set(node_order) -
        set(indexed.index)
    )

    extra = (
        set(indexed.index) -
        set(node_order)
    )

    if missing or extra:

        raise ValueError(
            f"Community alignment error. "
            f"Missing: {missing}; Extra: {extra}"
        )

    aligned = indexed.loc[node_order]

    communities = aligned[column].to_numpy()

    if pd.isna(communities).any():

        raise ValueError(
            f"Missing community assignments for {day}."
        )

    return communities


# ============================================================
# 5. NETWORK SUMMARY FUNCTION
# ============================================================

def network_statistics(
    A,
    empirical_A,
    communities
):

    n = A.shape[0]

    upper = np.triu(
        A,
        k=1
    )

    empirical_upper = np.triu(
        empirical_A,
        k=1
    )

    degrees = np.count_nonzero(
        A,
        axis=1
    )

    empirical_degrees = np.count_nonzero(
        empirical_A,
        axis=1
    )

    possible_edges = n * (n - 1) / 2

    edge_count = np.count_nonzero(upper)

    density = edge_count / possible_edges

    strengths = A.sum(axis=1)

    # Correlation of generated and empirical degrees.
    if (
        np.std(degrees) > 0
        and np.std(empirical_degrees) > 0
    ):

        degree_correlation = np.corrcoef(
            degrees,
            empirical_degrees
        )[0, 1]

    else:

        degree_correlation = np.nan

    # Proportion of edge weight occurring within
    # the estimated communities.
    same_community = (
        communities[:, None] ==
        communities[None, :]
    )

    within_weight = (
        A * same_community
    ).sum() / 2

    total_weight = A.sum() / 2

    within_weight_fraction = (
        within_weight / total_weight
        if total_weight > 0
        else np.nan
    )

    return {

        "nodes": n,

        "edges": edge_count,

        "density": density,

        "mean_degree": degrees.mean(),

        "degree_sd": degrees.std(),

        "mean_strength": strengths.mean(),

        "total_edge_weight": total_weight,

        "degree_correlation_with_empirical":
            degree_correlation,

        "within_community_weight_fraction":
            within_weight_fraction

    }


# ============================================================
# 6. GENERATE NULL MODELS FOR BOTH DAYS
# ============================================================

all_results = []

network_objects = {}

for day_number, day in enumerate(
    ["day1", "day2"]
):

    print("\n" + "=" * 60)

    print(f"PROCESSING {day.upper()}")

    print("=" * 60)

    # Load empirical network.
    network_path = (
        config.RESULTS_DIR /
        f"adjacency_{day}.npz"
    )

    empirical_A, node_order = load_adjacency(
        network_path
    )

    empirical_A = null_models.validate_adjacency(
        empirical_A
    )

    n = empirical_A.shape[0]

    print("Participants:", n)

    print("Empirical edges:",
          np.count_nonzero(
              np.triu(empirical_A, k=1)
          )
    )

    # Align communities.
    communities = align_communities(
        node_order,
        day
    )

    print(
        "Estimated communities:",
        len(np.unique(communities))
    )

    # Save node ordering for future experiments.
    pd.DataFrame({
        "node_index": np.arange(n),
        "id": node_order,
        "community": communities
    }).to_csv(
        NULL_DIR / f"node_order_{day}.csv",
        index=False
    )

    # Record empirical network.
    empirical_stats = network_statistics(
        empirical_A,
        empirical_A,
        communities
    )

    empirical_stats.update({

        "day": day,

        "model": "Empirical"

    })

    all_results.append(
        empirical_stats
    )

    # Generate three null models.
    generated_networks = (
        null_models.generate_all_null_models(
            empirical_A,
            communities,
            seed=636 + day_number * 100
        )
    )

    network_objects[day] = {
        "empirical": empirical_A,
        "communities": communities,
        "node_order": node_order,
        "null_models": generated_networks
    }

    # Validate and save each generated network.
    for model_name, generated_A in (
        generated_networks.items()
    ):

        print("\nModel:", model_name)

        generated_A = (
            null_models.validate_adjacency(
                generated_A
            )
        )

        if generated_A.shape != empirical_A.shape:

            raise ValueError(
                f"{model_name} has incorrect dimensions."
            )

        if not np.allclose(
            np.diag(generated_A),
            0
        ):

            raise ValueError(
                f"{model_name} contains self-loops."
            )

        if not np.allclose(
            generated_A,
            generated_A.T
        ):

            raise ValueError(
                f"{model_name} is not symmetric."
            )

        # Save generated adjacency matrix.
        null_models.save_null_network(
            generated_A,
            model_name,
            day
        )

        # Save summary statistics.
        stats = network_statistics(
            generated_A,
            empirical_A,
            communities
        )

        stats.update({

            "day": day,

            "model": model_name

        })

        all_results.append(stats)

        print("Nodes:", stats["nodes"])

        print("Edges:", stats["edges"])

        print(
            "Density:",
            round(stats["density"], 4)
        )

        print(
            "Degree correlation:",
            round(
                stats[
                    "degree_correlation_with_empirical"
                ],
                4
            )
            if not np.isnan(
                stats[
                    "degree_correlation_with_empirical"
                ]
            )
            else "NA"
        )

        print("Validation: PASS")


# ============================================================
# 7. SAVE COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(
    all_results
)

column_order = [

    "day",

    "model",

    "nodes",

    "edges",

    "density",

    "mean_degree",

    "degree_sd",

    "mean_strength",

    "total_edge_weight",

    "degree_correlation_with_empirical",

    "within_community_weight_fraction"

]

results_df = results_df[column_order]

output_path = (
    TABLE_DIR /
    "null_model_validation.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 8. DISPLAY FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("NULL MODEL GENERATION COMPLETE")

print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved validation table:")

print(output_path)

print("\nGenerated networks:")

print(NULL_DIR)

print("\nSTAT 636 CHUNK 8B COMPLETE")


"""
STAT 636
Inspect saved network and community data.
Supports both NumPy and SciPy NPZ formats.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import inspect

import config
import null_models


# ============================================================
# 1. FLEXIBLE NPZ LOADER
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
            f"Missing required arrays: "
            f"{required - set(archive.files)}"
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

    node_order = archive["node_order"]

    return A.toarray(), node_order



# ============================================================
# 2. LOAD AND INSPECT ADJACENCY MATRICES
# ============================================================

network_data = {}

for day in ["day1", "day2"]:

    path = (
        config.RESULTS_DIR
        / f"adjacency_{day}.npz"
    )

    print("\n" + "=" * 55)
    print(f"{day.upper()} ADJACENCY MATRIX")
    print("=" * 55)

    A, node_order = load_adjacency(path)

    network_data[day] = {
        "A": A,
        "node_order": node_order
    }

    print("Shape:", A.shape)

    print("Participants:", len(node_order))

    print("First 10 participant IDs:")
    print(node_order[:10])

    print("Nonzero entries:", np.count_nonzero(A))

    print("Symmetric:", np.allclose(A, A.T))

    print("Total weight:", A.sum())

    print("Self-loops:", np.diag(A).sum())


# Verify node alignment.

ids_day1 = network_data["day1"]["node_order"]
ids_day2 = network_data["day2"]["node_order"]

print("\nNODE ALIGNMENT")

print(
    "Same participants:",
    set(ids_day1.tolist()) == set(ids_day2.tolist())
)

print(
    "Same node order:",
    np.array_equal(ids_day1, ids_day2)
)


# ============================================================
# 3. INSPECT METADATA
# ============================================================

metadata_path = (
    config.RESULTS_DIR
    / "tables"
    / "aligned_metadata.csv"
)

metadata = pd.read_csv(metadata_path)

print("\nMETADATA COLUMNS:")
print(metadata.columns.tolist())

print(metadata.head())


# ============================================================
# 4. INSPECT COMMUNITY ASSIGNMENTS
# ============================================================

community_path = (
    config.RESULTS_DIR
    / "tables"
    / "community_assignments_both_days.csv"
)

community_data = pd.read_csv(
    community_path
)

print("\nCOMMUNITY ASSIGNMENT COLUMNS:")
print(community_data.columns.tolist())

print("\nFIRST TEN ROWS:")
print(community_data.head(10))

print("\nDATA TYPES:")
print(community_data.dtypes)


# ============================================================
# 5. INSPECT NULL MODEL GENERATOR
# ============================================================

print("\nNULL MODEL FUNCTIONS")
print("=" * 55)

for name, function in inspect.getmembers(
    null_models,
    inspect.isfunction
):

    print("\nFunction:", name)

    try:

        print(
            "Arguments:",
            inspect.signature(function)
        )

    except (TypeError, ValueError):

        print("Signature unavailable")


print("\nINSPECTION COMPLETE")


"""
STAT 636
Test null model generation and validation.
"""

import numpy as np
import pandas as pd

from pathlib import Path

from null_models import generate_all_null_models

from null_model_validation import (
    compare_all_null_models,
    save_validation_results,
    print_validation_report
)

import config


# ============================================================
# 1. LOAD SAVED NETWORKS
# ============================================================

NETWORK_DIR = config.RESULTS_DIR / "networks"

day2_path = NETWORK_DIR / "day2_adjacency.csv"

if not day2_path.exists():
    raise FileNotFoundError(
        f"Cannot find {day2_path}. "
        "Check your network output directory."
    )

A = np.loadtxt(
    day2_path,
    delimiter=","
)


# ============================================================
# 2. LOAD NODE METADATA
# ============================================================

metadata_path = (
    config.DATA_DIR / "metadata.csv"
)

if not metadata_path.exists():
    raise FileNotFoundError(
        f"Cannot find {metadata_path}. "
        "Check your metadata filename."
    )

metadata = pd.read_csv(
    metadata_path
)

# Update these column names if necessary.
classroom_labels = metadata[
    "class"
].to_numpy()

communities = metadata[
    "community"
].to_numpy()


# ============================================================
# 3. CHECK ALIGNMENT
# ============================================================

if len(classroom_labels) != A.shape[0]:

    raise ValueError(
        "Classroom metadata and network have "
        "different numbers of participants."
    )

if len(communities) != A.shape[0]:

    raise ValueError(
        "Community assignments and network "
        "have different numbers of participants."
    )


# ============================================================
# 4. GENERATE NULL MODELS
# ============================================================

null_networks = generate_all_null_models(
    A,
    communities,
    seed=636
)


# ============================================================
# 5. VALIDATE NULL MODELS
# ============================================================

results = compare_all_null_models(
    A,
    null_networks,
    classroom_labels
)


# ============================================================
# 6. SAVE RESULTS
# ============================================================

save_validation_results(
    results
)

print_validation_report(
    results
)

print("\nNULL MODEL TEST COMPLETED.")

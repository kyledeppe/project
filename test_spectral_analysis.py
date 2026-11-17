
"""
STAT 636
Chunk 4: Spectral Analysis Tests
"""

import numpy as np

from data_loader import load_school_networks

from spectral_analysis import (
    spectral_preprocessing,
    compare_clustering_methods,
    calculate_degree_correction,
    calculate_community_bridge_entropy,
    estimate_block_matrix
)

# Load data.
data = load_school_networks()

G = data.G1
A = data.A1

metadata = data.metadata
node_order = data.node_order

# -----------------------------------
# TEST 1: SPECTRAL EMBEDDING
# -----------------------------------

spectral = spectral_preprocessing(
    A,
    dimensions=3
)

X = spectral["embedding"]
Y = spectral["normalized_embedding"]

assert X.shape == (242, 3)
assert Y.shape == (242, 3)

print("TEST 1 PASSED: Spectral embedding.")

# -----------------------------------
# TEST 2: NORMALIZATION
# -----------------------------------

norms = np.linalg.norm(Y, axis=1)

nonzero = norms > 0

assert np.allclose(
    norms[nonzero],
    1.0
)

print("TEST 2 PASSED: Spherical normalization.")

# -----------------------------------
# TEST 3: CLUSTERING
# -----------------------------------

ari_results, assignments, probabilities = (
    compare_clustering_methods(
        Y,
        metadata,
        node_order
    )
)

assert len(ari_results) == 8

print("TEST 3 PASSED: K-means and GMM.")

# -----------------------------------
# TEST 4: DEGREE CORRECTION
# -----------------------------------

labels = assignments[
    ("K-means", 10)
]

degree_results = calculate_degree_correction(
    G,
    node_order,
    labels
)

assert len(degree_results["theta"]) == 242

assert np.all(
    degree_results["theta"] >= 0
)

print("TEST 4 PASSED: Degree correction.")

# -----------------------------------
# TEST 5: BRIDGE ENTROPY
# -----------------------------------

bridge = calculate_community_bridge_entropy(
    G,
    node_order,
    labels
)

entropy = bridge["normalized_entropy"]

assert len(entropy) == 242

assert np.all(entropy >= 0)
assert np.all(entropy <= 1 + 1e-10)

print("TEST 5 PASSED: Community bridge entropy.")

# -----------------------------------
# TEST 6: BLOCK MATRIX
# -----------------------------------

block_matrix = estimate_block_matrix(
    G,
    node_order,
    labels
)

assert block_matrix.shape == (10, 10)

assert np.all(
    np.isfinite(block_matrix.to_numpy())
)

print("TEST 6 PASSED: Block matrix.")

# -----------------------------------
# FINAL
# -----------------------------------

print("\nARI comparison:")
print(ari_results.to_string(index=False))

print("\nALL CHUNK 4 TESTS COMPLETED.")

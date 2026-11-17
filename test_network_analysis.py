
"""
STAT 636
Chunk 3: Network Analysis Tests
"""

import numpy as np

from data_loader import load_school_networks

from network_analysis import (
    calculate_node_statistics,
    calculate_network_statistics,
    classroom_analysis
)

# Load networks.
data = load_school_networks()

G1 = data.G1
G2 = data.G2

metadata = data.metadata

# -----------------------------------
# TEST 1: NODE STATISTICS
# -----------------------------------

day1 = calculate_node_statistics(G1)
day2 = calculate_node_statistics(G2)

assert len(day1) == 242
assert len(day2) == 242

print("TEST 1 PASSED: Node statistics.")

# -----------------------------------
# TEST 2: NONNEGATIVE DEGREE
# -----------------------------------

assert (day1["degree"] >= 0).all()
assert (day2["degree"] >= 0).all()

print("TEST 2 PASSED: Degree.")

# -----------------------------------
# TEST 3: NONNEGATIVE STRENGTH
# -----------------------------------

assert (day1["strength"] >= 0).all()
assert (day2["strength"] >= 0).all()

print("TEST 3 PASSED: Strength.")

# -----------------------------------
# TEST 4: CENTRALITY BOUNDS
# -----------------------------------

assert day1["betweenness"].between(0, 1).all()
assert day2["betweenness"].between(0, 1).all()

assert day1["clustering"].between(0, 1).all()
assert day2["clustering"].between(0, 1).all()

print("TEST 4 PASSED: Centrality bounds.")

# -----------------------------------
# TEST 5: NETWORK SUMMARY
# -----------------------------------

summary1 = calculate_network_statistics(
    G1,
    "Day 1"
)

summary2 = calculate_network_statistics(
    G2,
    "Day 2"
)

assert summary1["nodes"] == 242
assert summary2["nodes"] == 242

print("TEST 5 PASSED: Network summaries.")

# -----------------------------------
# TEST 6: CLASSROOM ANALYSIS
# -----------------------------------

classroom1 = classroom_analysis(
    G1,
    metadata,
    "Day 1"
)

classroom2 = classroom_analysis(
    G2,
    metadata,
    "Day 2"
)

print("TEST 6 PASSED: Classroom analysis.")

# -----------------------------------
# FINAL RESULTS
# -----------------------------------

print("\nDay 1 network summary:")
print(summary1)

print("\nDay 2 network summary:")
print(summary2)

print("\nALL CHUNK 3 TESTS COMPLETED.")

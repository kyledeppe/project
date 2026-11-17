
"""
STAT 636 Project
Chunk 2: Data Loading and Network Construction

Team:
    Kyle Deppe
    Tyler Lowry
    Madhav Thamaran
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import networkx as nx

from scipy.sparse import csr_matrix

import config


# ============================================================
# 1. DATA CONTAINER
# ============================================================

class SchoolNetworkData:
    """
    Stores both daily contact networks, participant metadata,
    adjacency matrices, and participant ordering.
    """

    def __init__(
        self,
        G1,
        G2,
        metadata,
        node_order,
        A1,
        A2
    ):

        self.G1 = G1
        self.G2 = G2

        self.metadata = metadata

        self.node_order = node_order

        self.A1 = A1
        self.A2 = A2

        self.n_nodes = len(node_order)

        self.n_edges_day1 = G1.number_of_edges()
        self.n_edges_day2 = G2.number_of_edges()


# ============================================================
# 2. STANDARDIZE PARTICIPANT IDENTIFIERS
# ============================================================

def standardize_id(value):
    """
    Convert participant identifiers to consistent strings.
    """

    if pd.isna(value):
        raise ValueError("Missing participant ID.")

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    if not value:
        raise ValueError("Empty participant ID.")

    return value


# ============================================================
# 3. LOAD PARTICIPANT METADATA
# ============================================================

def load_metadata(filepath=None):
    """
    Load participant metadata.

    Required columns:
        id
        classroom
        gender
    """

    if filepath is None:
        filepath = config.METADATA_FILE

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {filepath}"
        )

    metadata = pd.read_csv(
        filepath,
        sep=None,
        engine="python",
        dtype=str
    )

    metadata.columns = [
        str(col).strip().lower()
        for col in metadata.columns
    ]

    # Recognize alternative column names.
    aliases = {
        "id": [
            "id", "node", "node_id",
            "participant", "participant_id"
        ],
        "classroom": [
            "classroom", "classname",
            "class", "group"
        ],
        "gender": [
            "gender", "sex"
        ]
    }

    rename = {}

    for standard, alternatives in aliases.items():

        for alternative in alternatives:

            if alternative in metadata.columns:

                rename[alternative] = standard
                break

    metadata = metadata.rename(columns=rename)

    required = ["id", "classroom", "gender"]

    missing = [
        col for col in required
        if col not in metadata.columns
    ]

    if missing:
        raise ValueError(
            f"Missing metadata columns: {missing}"
        )

    metadata = metadata[required].copy()

    metadata["id"] = (
        metadata["id"]
        .astype(str)
        .str.strip()
    )

    metadata["classroom"] = (
        metadata["classroom"].astype(str).str.strip()
    )

    metadata["gender"] = (
        metadata["gender"].astype(str).str.strip()
    )

    if metadata["id"].duplicated().any():
        raise ValueError(
            "Duplicate participant IDs in metadata."
        )

    metadata = metadata.set_index("id", drop=False)

    print("\nMetadata loaded successfully.")
    print("Participants:", len(metadata))
    print(
        "Classrooms:",
        metadata["classroom"].nunique()
    )

    return metadata


# ============================================================
# 4. LOAD GEXF NETWORK
# ============================================================

def load_gexf(filepath):
    """
    Import one daily contact network.

    Edge attributes:
        duration: contact duration in seconds
        count: number of contact intervals
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Network file not found: {filepath}"
        )

    G = nx.read_gexf(
        filepath,
        node_type=str
    )

    if G.is_directed():
        raise ValueError(
            "Expected an undirected network."
        )

    G = nx.relabel_nodes(
        G,
        {
            node: standardize_id(node)
            for node in G.nodes()
        }
    )

    # Combine multiple edges if necessary.
    if G.is_multigraph():

        H = nx.Graph()

        H.add_nodes_from(G.nodes(data=True))

        for u, v, data in G.edges(data=True):

            duration = float(data.get("duration", 0))
            count = float(data.get("count", 0))

            if H.has_edge(u, v):

                H[u][v]["duration"] += duration
                H[u][v]["count"] += count

            else:

                H.add_edge(
                    u,
                    v,
                    duration=duration,
                    count=count
                )

        G = H

    else:

        G = nx.Graph(G)

    # Validate contact attributes.
    for u, v, data in G.edges(data=True):

        if "duration" not in data:
            raise ValueError(
                f"Missing duration on edge {u}-{v}"
            )

        duration = float(data["duration"])
        count = float(data.get("count", 0))

        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(
                f"Invalid duration on edge {u}-{v}"
            )

        if not np.isfinite(count) or count < 0:
            raise ValueError(
                f"Invalid contact count on edge {u}-{v}"
            )

        data["duration"] = duration
        data["count"] = count
        data["weight"] = duration

    print(f"\nLoaded {filepath.name}")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    return G


# ============================================================
# 5. ALIGN DAILY NETWORKS
# ============================================================

def align_networks(G1, G2, metadata):
    """
    Ensure both networks contain the same participants.

    Metadata supplies participants who may be isolated
    on both observed days.
    """

    all_nodes = sorted(
        str(node).strip()
        for node in (
                set(G1.nodes()) |
                set(G2.nodes()) |
                set(metadata["id"])
        )
    )

    G1 = nx.relabel_nodes(
        G1,
        {node: str(node).strip() for node in G1.nodes()}
    )

    G2 = nx.relabel_nodes(
        G2,
        {node: str(node).strip() for node in G2.nodes()}
    )

    if len(all_nodes) != config.EXPECTED_NODES:

        warnings.warn(
            f"Expected {config.EXPECTED_NODES} participants; "
            f"found {len(all_nodes)}."
        )

    # Check metadata coverage.
    missing = (
        set(G1.nodes()) | set(G2.nodes())
    ) - set(metadata.index)

    if missing:
        warnings.warn(
            f"{len(missing)} network participants lack metadata."
        )

    G1 = G1.copy()
    G2 = G2.copy()

    # Preserve every participant on both days.
    for node in all_nodes:

        if node not in G1:
            G1.add_node(node)

        if node not in G2:
            G2.add_node(node)

    # Rebuild in identical node order.
    H1 = nx.Graph()
    H2 = nx.Graph()

    H1.add_nodes_from(
        (node, G1.nodes[node])
        for node in all_nodes
    )

    H2.add_nodes_from(
        (node, G2.nodes[node])
        for node in all_nodes
    )

    H1.add_edges_from(G1.edges(data=True))
    H2.add_edges_from(G2.edges(data=True))

    assert list(H1.nodes()) == list(H2.nodes())

    print("\nNetworks aligned.")
    print("Shared participants:", len(all_nodes))

    print(
        "Day 1 isolates:",
        nx.number_of_isolates(H1)
    )

    print(
        "Day 2 isolates:",
        nx.number_of_isolates(H2)
    )

    return H1, H2, all_nodes


# ============================================================
# 6. COMBINE PARTICIPANT ATTRIBUTES
# ============================================================

def combine_node_metadata(
    G1,
    G2,
    metadata,
    node_order
):
    """
    Combine external metadata and GEXF node attributes.
    """

    records = []

    for node in node_order:

        classroom = None
        gender = None

        if node in metadata.index:

            classroom = metadata.loc[node, "classroom"]
            gender = metadata.loc[node, "gender"]

        for graph in [G1, G2]:

            attrs = graph.nodes[node]

            if classroom is None or pd.isna(classroom):
                classroom = (
                    attrs.get("classname")
                    or attrs.get("class")
                )

            if gender is None or pd.isna(gender):
                gender = attrs.get("gender")

        records.append({
            "id": node,
            "classroom": classroom,
            "gender": gender
        })

    return pd.DataFrame(records)


# ============================================================
# 7. CREATE WEIGHTED ADJACENCY MATRIX
# ============================================================

def create_adjacency_matrix(G, node_order):
    """
    Construct weighted adjacency matrix.

    A[i,j] = total contact duration in seconds.
    """

    A = nx.to_scipy_sparse_array(
        G,
        nodelist=node_order,
        weight="duration",
        format="csr",
        dtype=float
    )

    A = csr_matrix(A)

    if not np.allclose(
        A.toarray(),
        A.T.toarray()
    ):
        raise ValueError(
            "Adjacency matrix is not symmetric."
        )

    if not np.all(np.isfinite(A.data)):
        raise ValueError(
            "Invalid adjacency matrix values."
        )

    if np.any(A.data < 0):
        raise ValueError(
            "Negative contact durations found."
        )

    return A


# ============================================================
# 8. VALIDATE NETWORKS
# ============================================================

def validate_networks(
    G1,
    G2,
    metadata,
    node_order
):
    """
    Validate the networks before further analysis.
    """

    checks = {
        "same_nodes":
            set(G1.nodes()) == set(G2.nodes()),

        "same_node_order":
            list(G1.nodes()) == list(G2.nodes()),

        "undirected_day1":
            not G1.is_directed(),

        "undirected_day2":
            not G2.is_directed(),

        "no_self_loops_day1":
            nx.number_of_selfloops(G1) == 0,

        "no_self_loops_day2":
            nx.number_of_selfloops(G2) == 0,

        "all_nodes_have_metadata":
            set(node_order).issubset(
                set(metadata["id"])
            )
    }

    for name, passed in checks.items():

        print(
            f"{name}: {'PASS' if passed else 'FAIL'}"
        )

    if not all(checks.values()):
        raise ValueError(
            "Network validation failed."
        )

    return checks


# ============================================================
# 9. NETWORK SUMMARY
# ============================================================

def network_summary(G, day):
    """
    Calculate basic descriptive statistics.
    """

    degrees = np.array([
        d for _, d in G.degree()
    ])

    strengths = np.array([
        s for _, s in G.degree(weight="duration")
    ])

    durations = np.array([
        data["duration"]
        for _, _, data in G.edges(data=True)
    ])

    counts = np.array([
        data["count"]
        for _, _, data in G.edges(data=True)
    ])

    components = list(
        nx.connected_components(G)
    )

    summary = {
        "day": day,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "average_degree": degrees.mean(),
        "degree_sd": degrees.std(ddof=1),
        "average_strength": strengths.mean(),
        "total_contact_duration": durations.sum(),
        "total_contact_intervals": counts.sum(),
        "isolated_nodes": nx.number_of_isolates(G),
        "connected_components": len(components),
        "largest_component": max(
            map(len, components)
        )
    }

    return summary


# ============================================================
# 10. SAVE DATA
# ============================================================

def save_network_data(
    G1,
    G2,
    metadata,
    A1,
    A2,
    node_order
):
    """
    Save aligned networks and adjacency matrices.
    """

    metadata.to_csv(
        config.TABLE_DIR / "aligned_metadata.csv",
        index=False
    )

    # Save sparse matrices.
    for name, A in [
        ("day1", A1),
        ("day2", A2)
    ]:

        np.savez_compressed(
            config.RESULTS_DIR / f"adjacency_{name}.npz",
            data=A.data,
            indices=A.indices,
            indptr=A.indptr,
            shape=A.shape,
            node_order=np.array(node_order)
        )

    nx.write_gexf(
        G1,
        config.RESULTS_DIR / "cleaned_day1.gexf"
    )

    nx.write_gexf(
        G2,
        config.RESULTS_DIR / "cleaned_day2.gexf"
    )

    summaries = pd.DataFrame([
        network_summary(G1, "Day 1"),
        network_summary(G2, "Day 2")
    ])

    summaries.to_csv(
        config.TABLE_DIR / "network_loading_summary.csv",
        index=False
    )

    print("\nNetwork files saved successfully.")


# ============================================================
# 11. MAIN LOADING PIPELINE
# ============================================================

def load_school_networks():
    """
    Complete data loading pipeline.

    Returns a SchoolNetworkData object.
    """

    print("\n================================")
    print("STAT 636 DATA LOADING")
    print("================================")

    # Load data.
    metadata = load_metadata()

    G1 = load_gexf(config.DAY1_FILE)
    G2 = load_gexf(config.DAY2_FILE)

    # Align participants.
    G1, G2, node_order = align_networks(
        G1,
        G2,
        metadata
    )

    # Combine attributes.
    combined_metadata = combine_node_metadata(
        G1,
        G2,
        metadata,
        node_order
    )

    # Validate.
    validate_networks(
        G1,
        G2,
        combined_metadata,
        node_order
    )

    # Construct adjacency matrices.
    A1 = create_adjacency_matrix(
        G1,
        node_order
    )

    A2 = create_adjacency_matrix(
        G2,
        node_order
    )

    # Save outputs.
    save_network_data(
        G1,
        G2,
        combined_metadata,
        A1,
        A2,
        node_order
    )

    dataset = SchoolNetworkData(
        G1,
        G2,
        combined_metadata,
        node_order,
        A1,
        A2
    )

    print("\nDATA LOADING COMPLETE")

    print("Participants:", dataset.n_nodes)

    print("Day 1 edges:", dataset.n_edges_day1)

    print("Day 2 edges:", dataset.n_edges_day2)

    return dataset

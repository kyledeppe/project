
"""
STAT 636 Project
Chunk 3: Exploratory Network Analysis

Part A:
    Basic graph statistics
    Degree
    Weighted degree (strength)
    Betweenness centrality
    Eigenvector centrality
    Clustering coefficients
"""

import numpy as np
import pandas as pd
import networkx as nx

import config


# ============================================================
# 1. DEGREE CENTRALITY
# ============================================================

def calculate_degree(G):
    """
    Calculate the number of distinct contacts for
    every participant.

    Degree:
        k_i = number of neighbors of participant i
    """

    degree = dict(G.degree())

    return degree


# ============================================================
# 2. WEIGHTED DEGREE (STRENGTH)
# ============================================================

def calculate_strength(G):
    """
    Calculate total contact duration for every participant.

    Strength:
        s_i = sum_j w_ij

    Units: seconds.
    """

    strength = dict(
        G.degree(weight="duration")
    )

    return strength


# ============================================================
# 3. DEGREE CENTRALITY
# ============================================================

def calculate_degree_centrality(G):
    """
    Calculate normalized degree centrality.

    Formula:
        C_D(i) = k_i / (n - 1)
    """

    return nx.degree_centrality(G)


# ============================================================
# 4. BETWEENNESS CENTRALITY
# ============================================================

def calculate_betweenness(G):
    """
    Calculate weighted shortest-path betweenness.

    Contact duration is converted into distance:

        distance_ij = 1 / duration_ij

    Longer contacts therefore correspond to shorter
    network distances.
    """

    H = G.copy()

    for u, v, data in H.edges(data=True):

        duration = float(data["duration"])

        data["distance"] = 1.0 / max(
            duration,
            config.MIN_EDGE_WEIGHT
        )

    betweenness = nx.betweenness_centrality(
        H,
        normalized=True,
        weight="distance"
    )

    return betweenness


# ============================================================
# 5. EIGENVECTOR CENTRALITY
# ============================================================

def calculate_eigenvector_centrality(G):
    """
    Calculate eigenvector centrality.

    Edges are weighted by contact duration.

    Participants connected to other highly connected
    participants receive higher scores.
    """

    try:

        centrality = nx.eigenvector_centrality(
            G,
            max_iter=2000,
            tol=1e-08,
            weight="duration"
        )

    except nx.PowerIterationFailedConvergence:

        warnings_message = (
            "Eigenvector centrality did not converge. "
            "Using the largest connected component."
        )

        print(warnings_message)

        largest_component = max(
            nx.connected_components(G),
            key=len
        )

        H = G.subgraph(
            largest_component
        ).copy()

        centrality = {
            node: 0.0
            for node in G.nodes()
        }

        component_scores = nx.eigenvector_centrality(
            H,
            max_iter=5000,
            weight="duration"
        )

        centrality.update(component_scores)

    return centrality


# ============================================================
# 6. CLUSTERING COEFFICIENT
# ============================================================

def calculate_clustering(G):
    """
    Calculate local clustering coefficients.

    Measures how interconnected a participant's
    immediate neighbors are.

    Uses contact duration as edge weight.
    """

    clustering = nx.clustering(
        G,
        weight="duration"
    )

    return clustering


# ============================================================
# 7. CONNECTED COMPONENT INFORMATION
# ============================================================

def calculate_components(G):
    """
    Calculate connected-component membership.

    Returns:
        Number of components
        Component size for each participant
        Largest component size
    """

    components = list(
        nx.connected_components(G)
    )

    component_sizes = {}

    for component_id, component in enumerate(components):

        for node in component:

            component_sizes[node] = len(component)

    largest_size = max(
        len(component)
        for component in components
    )

    return {
        "number_of_components": len(components),
        "component_sizes": component_sizes,
        "largest_component": largest_size
    }


# ============================================================
# 8. CREATE NODE-LEVEL ANALYSIS TABLE
# ============================================================

def calculate_node_statistics(G):
    """
    Calculate all node-level centrality statistics.

    Returns a pandas DataFrame.
    """

    degree = calculate_degree(G)

    strength = calculate_strength(G)

    degree_centrality = calculate_degree_centrality(G)

    betweenness = calculate_betweenness(G)

    eigenvector = calculate_eigenvector_centrality(G)

    clustering = calculate_clustering(G)

    components = calculate_components(G)

    results = []

    for node in G.nodes():

        results.append({

            "id": node,

            "degree": degree[node],

            "strength": strength[node],

            "degree_centrality":
                degree_centrality[node],

            "betweenness":
                betweenness[node],

            "eigenvector_centrality":
                eigenvector[node],

            "clustering":
                clustering[node],

            "component_size":
                components["component_sizes"][node]

        })

    return pd.DataFrame(results)


# ============================================================
# 9. BASIC NETWORK SUMMARY
# ============================================================

def calculate_network_statistics(G, day):
    """
    Calculate overall network-level statistics.
    """

    n = G.number_of_nodes()

    m = G.number_of_edges()

    degrees = np.array([
        degree
        for _, degree in G.degree()
    ])

    strengths = np.array([
        strength
        for _, strength in G.degree(
            weight="duration"
        )
    ])

    durations = np.array([
        data["duration"]
        for _, _, data in G.edges(data=True)
    ])

    components = list(
        nx.connected_components(G)
    )

    summary = {

        "day": day,

        "nodes": n,

        "edges": m,

        "density": nx.density(G),

        "average_degree": degrees.mean(),

        "degree_sd": degrees.std(ddof=1),

        "maximum_degree": degrees.max(),

        "average_strength": strengths.mean(),

        "strength_sd": strengths.std(ddof=1),

        "maximum_strength": strengths.max(),

        "average_contact_duration":
            durations.mean() if len(durations) else 0,

        "total_contact_duration":
            durations.sum(),

        "number_of_components":
            len(components),

        "largest_component":
            max(map(len, components)),

        "isolated_nodes":
            nx.number_of_isolates(G),

        "average_clustering":
            nx.average_clustering(
                G,
                weight="duration"
            )

    }

    return summary



# ============================================================
# 10. CLASSROOM ASSORTATIVITY
# ============================================================

def calculate_classroom_assortativity(G, metadata):
    """
    Calculate categorical assortativity by classroom.

    Positive assortativity indicates that participants
    tend to connect with participants from their own
    classroom.
    """

    classroom = dict(
        zip(
            metadata["id"],
            metadata["classroom"]
        )
    )

    H = G.copy()

    nx.set_node_attributes(
        H,
        classroom,
        "classroom"
    )

    return nx.attribute_assortativity_coefficient(
        H,
        "classroom"
    )


# ============================================================
# 11. WITHIN-CLASSROOM AND BETWEEN-CLASSROOM CONTACTS
# ============================================================

def calculate_classroom_contact_summary(G, metadata):
    """
    Calculate total contact duration within and between
    classrooms.

    Each undirected edge is counted once.
    """

    classroom = dict(
        zip(
            metadata["id"],
            metadata["classroom"]
        )
    )

    within_duration = 0.0
    between_duration = 0.0

    within_edges = 0
    between_edges = 0

    for u, v, data in G.edges(data=True):

        weight = float(data["duration"])

        class_u = classroom.get(u)
        class_v = classroom.get(v)

        if pd.isna(class_u) or pd.isna(class_v):
            continue

        if class_u == class_v:

            within_duration += weight
            within_edges += 1

        else:

            between_duration += weight
            between_edges += 1

    total = within_duration + between_duration

    within_proportion = (
        within_duration / total
        if total > 0 else 0
    )

    between_proportion = (
        between_duration / total
        if total > 0 else 0
    )

    return {

        "within_duration": within_duration,

        "between_duration": between_duration,

        "within_edges": within_edges,

        "between_edges": between_edges,

        "within_proportion": within_proportion,

        "between_proportion": between_proportion

    }


# ============================================================
# 12. CLASSROOM MIXING MATRIX
# ============================================================

def calculate_classroom_mixing_matrix(G, metadata):
    """
    Construct a weighted classroom mixing matrix.

    Entry (r,s) represents the total contact duration
    between classroom r and classroom s.

    Matrix is symmetric.
    """

    classroom = dict(
        zip(
            metadata["id"],
            metadata["classroom"]
        )
    )

    classes = sorted(
        metadata["classroom"].dropna().unique()
    )

    mixing = pd.DataFrame(
        0.0,
        index=classes,
        columns=classes
    )

    for u, v, data in G.edges(data=True):

        class_u = classroom.get(u)
        class_v = classroom.get(v)

        if pd.isna(class_u) or pd.isna(class_v):
            continue

        duration = float(data["duration"])

        mixing.loc[class_u, class_v] += duration

        if class_u != class_v:
            mixing.loc[class_v, class_u] += duration

    return mixing


# ============================================================
# 13. CLASSROOM CONTACT PROPORTIONS
# ============================================================

def calculate_classroom_contact_proportions(
    mixing_matrix
):
    """
    Convert classroom contact durations into proportions.

    Each row represents the distribution of contact
    duration originating from one classroom.
    """

    row_totals = mixing_matrix.sum(axis=1)

    proportions = mixing_matrix.div(
        row_totals.replace(0, np.nan),
        axis=0
    )

    return proportions.fillna(0)


# ============================================================
# 14. COMBINE CLASSROOM STATISTICS
# ============================================================

def classroom_analysis(G, metadata, day):
    """
    Return classroom-level mixing results.
    """

    assortativity = calculate_classroom_assortativity(
        G,
        metadata
    )

    contact_summary = calculate_classroom_contact_summary(
        G,
        metadata
    )

    mixing_matrix = calculate_classroom_mixing_matrix(
        G,
        metadata
    )

    proportions = calculate_classroom_contact_proportions(
        mixing_matrix
    )

    summary = {
        "day": day,
        "classroom_assortativity": assortativity,
        **contact_summary
    }

    return (
        summary,
        mixing_matrix,
        proportions
    )



# ============================================================
# 15. IMPORT VISUALIZATION PACKAGES
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_school_networks


# ============================================================
# 16. SAVE NODE-LEVEL STATISTICS
# ============================================================

def save_node_statistics(
    G1,
    G2,
    metadata
):
    """
    Calculate and save node-level statistics for both days.
    """

    print("\nCalculating Day 1 centrality...")

    day1 = calculate_node_statistics(G1)

    print("Calculating Day 2 centrality...")

    day2 = calculate_node_statistics(G2)

    # Add participant metadata.
    day1 = day1.merge(
        metadata,
        on="id",
        how="left"
    )

    day2 = day2.merge(
        metadata,
        on="id",
        how="left"
    )

    day1["day"] = "Day 1"
    day2["day"] = "Day 2"

    day1.to_csv(
        config.TABLE_DIR / "centrality_day1.csv",
        index=False
    )

    day2.to_csv(
        config.TABLE_DIR / "centrality_day2.csv",
        index=False
    )

    combined = pd.concat(
        [day1, day2],
        ignore_index=True
    )

    combined.to_csv(
        config.TABLE_DIR / "centrality_all_days.csv",
        index=False
    )

    return day1, day2


# ============================================================
# 17. SAVE NETWORK SUMMARY
# ============================================================

def save_network_summaries(G1, G2):
    """
    Calculate and save network-level statistics.
    """

    summary1 = calculate_network_statistics(
        G1,
        "Day 1"
    )

    summary2 = calculate_network_statistics(
        G2,
        "Day 2"
    )

    summary = pd.DataFrame([
        summary1,
        summary2
    ])

    summary.to_csv(
        config.TABLE_DIR / "network_summary.csv",
        index=False
    )

    return summary


# ============================================================
# 18. SAVE CLASSROOM ANALYSIS
# ============================================================

def save_classroom_analysis(
    G1,
    G2,
    metadata
):
    """
    Save classroom mixing statistics and matrices.
    """

    results = []

    for G, day in [
        (G1, "Day 1"),
        (G2, "Day 2")
    ]:

        summary, mixing, proportions = classroom_analysis(
            G,
            metadata,
            day
        )

        results.append(summary)

        mixing.to_csv(
            config.TABLE_DIR /
            f"classroom_mixing_{day.lower().replace(' ', '')}.csv"
        )

        proportions.to_csv(
            config.TABLE_DIR /
            f"classroom_proportions_{day.lower().replace(' ', '')}.csv"
        )

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        config.TABLE_DIR / "classroom_summary.csv",
        index=False
    )

    return result_df


# ============================================================
# 19. DEGREE DISTRIBUTION PLOT
# ============================================================

def plot_degree_distribution(day1, day2):
    """
    Compare degree distributions across the two days.
    """

    plt.figure(figsize=(9, 6))

    sns.histplot(
        day1["degree"],
        color="steelblue",
        label="Day 1",
        discrete=True,
        alpha=0.45
    )

    sns.histplot(
        day2["degree"],
        color="darkorange",
        label="Day 2",
        discrete=True,
        alpha=0.45
    )

    plt.xlabel("Degree")
    plt.ylabel("Number of Participants")

    plt.title(
        "Degree Distribution: Day 1 vs Day 2"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        config.FIGURE_DIR / "degree_distribution.png",
        dpi=config.FIGURE_DPI
    )

    plt.close()


# ============================================================
# 20. STRENGTH DISTRIBUTION PLOT
# ============================================================

def plot_strength_distribution(day1, day2):
    """
    Compare weighted degree distributions.
    """

    plt.figure(figsize=(9, 6))

    sns.histplot(
        day1["strength"],
        color="steelblue",
        label="Day 1",
        bins=25,
        alpha=0.45
    )

    sns.histplot(
        day2["strength"],
        color="darkorange",
        label="Day 2",
        bins=25,
        alpha=0.45
    )

    plt.xlabel("Total Contact Duration (seconds)")
    plt.ylabel("Number of Participants")

    plt.title(
        "Contact Strength Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        config.FIGURE_DIR / "strength_distribution.png",
        dpi=config.FIGURE_DPI
    )

    plt.close()


# ============================================================
# 21. DAY-TO-DAY CENTRALITY COMPARISON
# ============================================================

def plot_centrality_comparison(day1, day2):
    """
    Compare participant centrality across days.
    """

    merged = day1.merge(
        day2,
        on="id",
        suffixes=("_day1", "_day2")
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5)
    )

    # Degree comparison.
    axes[0].scatter(
        merged["degree_day1"],
        merged["degree_day2"],
        alpha=0.65,
        color="steelblue"
    )

    axes[0].set_xlabel("Day 1 Degree")
    axes[0].set_ylabel("Day 2 Degree")

    axes[0].set_title(
        "Degree Stability"
    )

    # Strength comparison.
    axes[1].scatter(
        merged["strength_day1"],
        merged["strength_day2"],
        alpha=0.65,
        color="darkorange"
    )

    axes[1].set_xlabel(
        "Day 1 Strength (seconds)"
    )

    axes[1].set_ylabel(
        "Day 2 Strength (seconds)"
    )

    axes[1].set_title(
        "Strength Stability"
    )

    plt.tight_layout()

    plt.savefig(
        config.FIGURE_DIR / "centrality_comparison.png",
        dpi=config.FIGURE_DPI
    )

    plt.close()


# ============================================================
# 22. CLASSROOM MIXING HEATMAP
# ============================================================

def plot_classroom_mixing(G1, G2, metadata):
    """
    Visualize contact duration between classrooms.
    """

    for G, day in [
        (G1, "Day 1"),
        (G2, "Day 2")
    ]:

        _, mixing, _ = classroom_analysis(
            G,
            metadata,
            day
        )

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            mixing,
            cmap="YlOrRd",
            annot=True,
            fmt=".0f"
        )

        plt.title(
            f"Classroom Contact Duration: {day}"
        )

        plt.xlabel("Contact Classroom")
        plt.ylabel("Participant Classroom")

        plt.tight_layout()

        filename = (
            f"classroom_heatmap_"
            f"{day.lower().replace(' ', '')}.png"
        )

        plt.savefig(
            config.FIGURE_DIR / filename,
            dpi=config.FIGURE_DPI
        )

        plt.close()


# ============================================================
# 23. COMPLETE EXPLORATORY ANALYSIS
# ============================================================

def run_network_analysis():
    """
    Execute the complete exploratory network analysis.
    """

    print("\n====================================")
    print("STAT 636 NETWORK ANALYSIS")
    print("====================================")

    # Load the networks.
    data = load_school_networks()

    G1 = data.G1
    G2 = data.G2

    metadata = data.metadata

    # Calculate centrality.
    day1, day2 = save_node_statistics(
        G1,
        G2,
        metadata
    )

    # Network-level statistics.
    summary = save_network_summaries(
        G1,
        G2
    )

    # Classroom structure.
    classroom_summary = save_classroom_analysis(
        G1,
        G2,
        metadata
    )

    # Figures.
    plot_degree_distribution(
        day1,
        day2
    )

    plot_strength_distribution(
        day1,
        day2
    )

    plot_centrality_comparison(
        day1,
        day2
    )

    plot_classroom_mixing(
        G1,
        G2,
        metadata
    )

    print("\nNETWORK ANALYSIS COMPLETED")

    print("\nNetwork summary:")
    print(summary.to_string(index=False))

    print("\nClassroom summary:")
    print(classroom_summary.to_string(index=False))

    print(
        "\nResults saved in:",
        config.RESULTS_DIR
    )

    return {
        "day1": day1,
        "day2": day2,
        "network_summary": summary,
        "classroom_summary": classroom_summary
    }


# ============================================================
# 24. EXECUTE DIRECTLY
# ============================================================

if __name__ == "__main__":

    results = run_network_analysis()

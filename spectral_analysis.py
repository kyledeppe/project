
"""
STAT 636 Project
Chunk 4: Spectral Clustering and DCSBM

Part A:
    Adjacency spectral embedding
    Eigenvalue decomposition
    Spherical normalization
"""

import numpy as np
import pandas as pd
import networkx as nx

from scipy.sparse.linalg import eigsh

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt
import seaborn as sns

import config


# ============================================================
# 1. ADJACENCY SPECTRAL EMBEDDING
# ============================================================

def adjacency_spectral_embedding(A, dimensions=3):
    """
    Calculate the adjacency spectral embedding (ASE).

    Parameters
    ----------
    A : scipy sparse matrix or numpy array
        Weighted adjacency matrix.

    dimensions : int
        Number of dimensions to retain.

    Returns
    -------
    X : numpy array
        Spectral embedding, with one row per participant.

    eigenvalues : numpy array
        Retained eigenvalues.
    """

    A = A.astype(float)

    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    if dimensions < 1 or dimensions >= n:
        raise ValueError("Invalid embedding dimension.")

    if not np.allclose(A.toarray() if hasattr(A, "toarray") else A,
                       (A.toarray() if hasattr(A, "toarray") else A).T):
        raise ValueError("Adjacency matrix must be symmetric.")

    # Use sparse eigendecomposition for larger networks.
    try:

        eigenvalues, eigenvectors = eigsh(
            A,
            k=dimensions,
            which="LM"
        )

    except Exception:

        # Dense fallback.
        dense_A = (
            A.toarray()
            if hasattr(A, "toarray")
            else np.asarray(A)
        )

        eigenvalues_all, eigenvectors_all = np.linalg.eigh(
            dense_A
        )

        order = np.argsort(
            np.abs(eigenvalues_all)
        )[::-1][:dimensions]

        eigenvalues = eigenvalues_all[order]
        eigenvectors = eigenvectors_all[:, order]

    # Sort by absolute eigenvalue magnitude.
    order = np.argsort(
        np.abs(eigenvalues)
    )[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # ASE:
    # X = U * |Lambda|^(1/2)
    X = eigenvectors * np.sqrt(
        np.abs(eigenvalues)
    )

    return X, eigenvalues


# ============================================================
# 2. SPHERICAL NORMALIZATION
# ============================================================

def spherical_normalization(X):
    """
    Normalize each participant's spectral embedding.

    Formula:
        Y_i = X_i / ||X_i||

    Zero-norm rows remain zero.

    This is important for isolated participants.
    """

    norms = np.linalg.norm(
        X,
        axis=1,
        keepdims=True
    )

    Y = np.divide(
        X,
        norms,
        out=np.zeros_like(X),
        where=norms > 0
    )

    return Y


# ============================================================
# 3. COMPLETE SPECTRAL PREPROCESSING
# ============================================================

def spectral_preprocessing(A, dimensions=3):
    """
    Perform ASE followed by spherical normalization.

    Returns:
        X: original spectral embedding
        Y: normalized spectral embedding
        eigenvalues: retained eigenvalues
    """

    X, eigenvalues = adjacency_spectral_embedding(
        A,
        dimensions=dimensions
    )

    Y = spherical_normalization(X)

    return {
        "embedding": X,
        "normalized_embedding": Y,
        "eigenvalues": eigenvalues
    }


# ============================================================
# 4. EMBEDDING DIAGNOSTICS
# ============================================================

def embedding_diagnostics(X, Y, eigenvalues):
    """
    Print basic spectral embedding diagnostics.
    """

    zero_rows = np.sum(
        np.linalg.norm(X, axis=1) == 0
    )

    print("\nSPECTRAL EMBEDDING DIAGNOSTICS")

    print("Embedding dimensions:", X.shape[1])

    print("Number of participants:", X.shape[0])

    print("Retained eigenvalues:", eigenvalues)

    print("Zero-norm participants:", zero_rows)

    print(
        "Mean normalized row length:",
        np.linalg.norm(Y, axis=1).mean()
    )

    return {
        "dimensions": X.shape[1],
        "participants": X.shape[0],
        "zero_norm_rows": int(zero_rows)
    }



# ============================================================
# 5. K-MEANS CLUSTERING
# ============================================================

def run_kmeans(Y, K, random_state=636):
    """
    Apply K-means to the normalized spectral embedding.
    """

    model = KMeans(
        n_clusters=K,
        n_init=30,
        max_iter=500,
        random_state=random_state
    )

    labels = model.fit_predict(Y)

    return labels, model


# ============================================================
# 6. GAUSSIAN MIXTURE MODEL
# ============================================================

def run_gmm(Y, K, random_state=636):
    """
    Apply a Gaussian mixture model to the normalized
    spectral embedding.

    Returns:
        Predicted labels
        Fitted model
        Membership probabilities
    """

    model = GaussianMixture(
        n_components=K,
        covariance_type="full",
        n_init=5,
        max_iter=500,
        reg_covar=1e-5,
        random_state=random_state
    )

    model.fit(Y)

    labels = model.predict(Y)

    probabilities = model.predict_proba(Y)

    return labels, model, probabilities


# ============================================================
# 7. PREPARE TRUE CLASSROOM LABELS
# ============================================================

def get_classroom_labels(metadata, node_order):
    """
    Obtain actual classroom labels in the same order
    as the adjacency matrix.

    Classroom labels are used only for validation.
    """

    indexed = metadata.set_index(
        "id"
    )

    classroom = indexed.reindex(
        node_order
    )["classroom"]

    if classroom.isna().any():
        raise ValueError(
            "Some participants are missing classroom labels."
        )

    return classroom.astype(str).to_numpy()


# ============================================================
# 8. COMPARE CLUSTERING METHODS
# ============================================================

def compare_clustering_methods(
    Y,
    metadata,
    node_order,
    k_values=(5, 8, 10, 12),
    random_state=636
):
    """
    Compare K-means and GMM for several community counts.

    Calculate ARI against actual classrooms.

    Returns:
        results_table
        assignments
        membership_probabilities
    """

    true_labels = get_classroom_labels(
        metadata,
        node_order
    )

    results = []

    assignments = {}

    membership_probabilities = {}

    for K in k_values:

        print(f"\nEvaluating K = {K}")

        if K >= len(Y):
            raise ValueError(
                f"K={K} must be smaller than the number "
                "of participants."
            )

        # ------------------------------------
        # K-MEANS
        # ------------------------------------

        kmeans_labels, kmeans_model = run_kmeans(
            Y,
            K,
            random_state
        )

        kmeans_ari = adjusted_rand_score(
            true_labels,
            kmeans_labels
        )

        results.append({

            "method": "K-means",

            "K": K,

            "ARI": kmeans_ari

        })

        assignments[
            ("K-means", K)
        ] = kmeans_labels

        # ------------------------------------
        # GAUSSIAN MIXTURE
        # ------------------------------------

        gmm_labels, gmm_model, probabilities = run_gmm(
            Y,
            K,
            random_state
        )

        gmm_ari = adjusted_rand_score(
            true_labels,
            gmm_labels
        )

        results.append({

            "method": "GMM",

            "K": K,

            "ARI": gmm_ari

        })

        assignments[
            ("GMM", K)
        ] = gmm_labels

        membership_probabilities[
            K
        ] = probabilities

        print(
            f"K-means ARI: {kmeans_ari:.4f}"
        )

        print(
            f"GMM ARI: {gmm_ari:.4f}"
        )

    results_table = pd.DataFrame(results)

    return (
        results_table,
        assignments,
        membership_probabilities
    )


# ============================================================
# 9. SELECT BEST CLUSTERING MODEL
# ============================================================

def select_best_clustering(results_table):
    """
    Select the method and K with the highest classroom ARI.

    This is a validation-based model comparison.
    """

    best_index = results_table["ARI"].idxmax()

    best = results_table.loc[
        best_index
    ]

    print("\nBEST CLUSTERING MODEL")

    print("Method:", best["method"])

    print("Number of communities:", int(best["K"]))

    print("ARI:", round(best["ARI"], 4))

    return best



# ============================================================
# 10. DEGREE-CORRECTION PARAMETERS
# ============================================================

def calculate_degree_correction(
    G,
    node_order,
    community_labels
):
    """
    Estimate descriptive degree-correction parameters.

    theta_i = strength_i / mean_strength_of_community_i

    A value greater than 1 indicates above-average
    contact strength within the participant's community.
    """

    strengths = dict(
        G.degree(weight="duration")
    )

    strength_array = np.array([
        strengths[node]
        for node in node_order
    ], dtype=float)

    labels = np.asarray(community_labels)

    theta = np.zeros(len(node_order))

    community_means = {}

    for community in np.unique(labels):

        mask = labels == community

        mean_strength = strength_array[mask].mean()

        community_means[community] = mean_strength

        if mean_strength > 0:

            theta[mask] = (
                strength_array[mask] / mean_strength
            )

        else:

            theta[mask] = 0.0

    return {
        "strength": strength_array,
        "theta": theta,
        "community_mean_strength": community_means
    }


# ============================================================
# 11. COMMUNITY BRIDGE ENTROPY
# ============================================================

def calculate_community_bridge_entropy(
    G,
    node_order,
    community_labels
):
    """
    Calculate normalized contact entropy for every node.

    q_ir = proportion of participant i's contact duration
           directed toward community r.

    H_i = -sum_r q_ir * log(q_ir)

    H*_i = H_i / log(K)

    Higher entropy indicates more distributed contact
    across communities.

    This is observed contact entropy, not latent
    mixed-membership probability.
    """

    labels = np.asarray(community_labels)

    communities = np.unique(labels)

    K = len(communities)

    community_index = {
        community: i
        for i, community in enumerate(communities)
    }

    node_index = {
        node: i
        for i, node in enumerate(node_order)
    }

    contact_matrix = np.zeros(
        (len(node_order), K)
    )

    # Allocate each participant's contact duration
    # to the community of their contact partner.
    for u, v, data in G.edges(data=True):

        i = node_index[u]
        j = node_index[v]

        weight = float(data["duration"])

        community_u = labels[i]
        community_v = labels[j]

        contact_matrix[i, community_index[community_v]] += weight

        contact_matrix[j, community_index[community_u]] += weight

    # Convert contact duration into proportions.
    totals = contact_matrix.sum(axis=1, keepdims=True)

    proportions = np.divide(
        contact_matrix,
        totals,
        out=np.zeros_like(contact_matrix),
        where=totals > 0
    )

    # Calculate Shannon entropy.
    log_proportions = np.zeros_like(proportions)

    positive = proportions > 0

    log_proportions[positive] = np.log(
        proportions[positive]
    )

    entropy = -np.sum(
        proportions * log_proportions,
        axis=1
    )

    # Normalize entropy to [0,1].
    if K > 1:

        normalized_entropy = (
            entropy / np.log(K)
        )

    else:

        normalized_entropy = np.zeros(
            len(node_order)
        )

    return {
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "contact_proportions": proportions,
        "communities": communities
    }


# ============================================================
# 12. WEIGHTED DCSBM BLOCK MATRIX
# ============================================================

def estimate_block_matrix(
    G,
    node_order,
    community_labels
):
    """
    Estimate descriptive weighted block connectivity.

    Off-diagonal:
        Total observed edge weight between blocks /
        number of possible cross-block pairs.

    Diagonal:
        Total observed edge weight within a block /
        number of possible within-block pairs.

    This is a descriptive weighted block matrix,
    not a fitted Bernoulli DCSBM likelihood.
    """

    labels = np.asarray(community_labels)

    communities = np.unique(labels)

    K = len(communities)

    node_index = {
        node: i
        for i, node in enumerate(node_order)
    }

    block_weight = np.zeros((K, K))

    community_sizes = np.array([
        np.sum(labels == community)
        for community in communities
    ])

    community_index = {
        community: i
        for i, community in enumerate(communities)
    }

    for u, v, data in G.edges(data=True):

        i = node_index[u]
        j = node_index[v]

        block_i = community_index[labels[i]]
        block_j = community_index[labels[j]]

        weight = float(data["duration"])

        block_weight[block_i, block_j] += weight

        if block_i != block_j:
            block_weight[block_j, block_i] += weight

    block_density = np.zeros((K, K))

    for r in range(K):

        for s in range(K):

            if r == s:

                possible = (
                    community_sizes[r]
                    * (community_sizes[r] - 1)
                    / 2
                )

            else:

                possible = (
                    community_sizes[r]
                    * community_sizes[s]
                )

            if possible > 0:

                block_density[r, s] = (
                    block_weight[r, s] / possible
                )

    return pd.DataFrame(
        block_density,
        index=communities,
        columns=communities
    )


# ============================================================
# 13. COMBINE PARTICIPANT-LEVEL COMMUNITY MEASURES
# ============================================================

def create_community_statistics(
    G,
    node_order,
    community_labels
):
    """
    Create a participant-level table containing:

        Community assignment
        Contact strength
        Degree correction
        Community bridge entropy
    """

    degree_results = calculate_degree_correction(
        G,
        node_order,
        community_labels
    )

    bridge_results = calculate_community_bridge_entropy(
        G,
        node_order,
        community_labels
    )

    results = pd.DataFrame({

        "id": node_order,

        "community": community_labels,

        "strength": degree_results["strength"],

        "degree_correction":
            degree_results["theta"],

        "bridge_entropy":
            bridge_results["entropy"],

        "normalized_bridge_entropy":
            bridge_results["normalized_entropy"]

    })

    return results


# ============================================================
# 14. CREATE WITHIN-COMMUNITY HUB SCORE
# ============================================================

def calculate_within_community_hubs(
    G,
    node_order,
    community_labels
):
    """
    Rank participants by their degree correction
    within their detected community.

    Higher values identify individuals with greater
    contact strength relative to their community.
    """

    result = calculate_degree_correction(
        G,
        node_order,
        community_labels
    )

    return pd.DataFrame({

        "id": node_order,

        "community": community_labels,

        "strength": result["strength"],

        "within_community_hub_score":
            result["theta"]

    })



# ============================================================
# 15. THREE-DIMENSIONAL SPECTRAL EMBEDDING
# ============================================================

def plot_spectral_embedding(
    Y,
    labels,
    day,
    filename
):
    """
    Plot the three-dimensional normalized spectral embedding.
    """

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    scatter = ax.scatter(
        Y[:, 0],
        Y[:, 1],
        Y[:, 2],
        c=labels,
        cmap="tab20",
        s=35,
        alpha=0.75
    )

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")

    ax.set_title(
        f"Spectral Embedding: {day}"
    )

    fig.colorbar(
        scatter,
        ax=ax,
        label="Community"
    )

    plt.tight_layout()

    plt.savefig(
        config.FIGURE_DIR / filename,
        dpi=config.FIGURE_DPI
    )

    plt.close()


# ============================================================
# 16. ARI COMPARISON PLOT
# ============================================================

def plot_ari_comparison(results, day):
    """
    Compare K-means and GMM using adjusted Rand index.
    """

    plt.figure(figsize=(9, 6))

    sns.lineplot(
        data=results,
        x="K",
        y="ARI",
        hue="method",
        marker="o",
        linewidth=2
    )

    plt.xlabel("Number of Communities (K)")

    plt.ylabel("Adjusted Rand Index")

    plt.title(
        f"Community Detection Performance: {day}"
    )

    plt.xticks(
        sorted(results["K"].unique())
    )

    plt.ylim(-0.1, 1.0)

    plt.tight_layout()

    filename = (
        f"ari_comparison_"
        f"{day.lower().replace(' ', '')}.png"
    )

    plt.savefig(
        config.FIGURE_DIR / filename,
        dpi=config.FIGURE_DPI
    )

    plt.close()


# ============================================================
# 17. BLOCK MATRIX HEATMAP
# ============================================================

def plot_block_matrix(block_matrix, day):
    """
    Visualize estimated weighted block connectivity.
    """

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        block_matrix,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f"
    )

    plt.xlabel("Community")

    plt.ylabel("Community")

    plt.title(
        f"Weighted DCSBM Block Matrix: {day}"
    )

    plt.tight_layout()

    filename = (
        f"block_matrix_"
        f"{day.lower().replace(' ', '')}.png"
    )

    plt.savefig(
        config.FIGURE_DIR / filename,
        dpi=config.FIGURE_DPI
    )

    plt.close()


# ============================================================
# 18. ANALYZE ONE DAILY NETWORK
# ============================================================

def analyze_one_day(
    G,
    A,
    metadata,
    node_order,
    day,
    primary_K=10
):
    """
    Complete spectral analysis for one daily network.

    Primary clustering:
        K-means with K=10.

    Alternative K values and GMM are used
    for sensitivity analysis.

    Classroom labels are used for validation only.
    """

    print(f"\n================================")
    print(f"SPECTRAL ANALYSIS: {day}")
    print("================================")

    # ----------------------------------------
    # A. Spectral embedding
    # ----------------------------------------

    spectral = spectral_preprocessing(
        A,
        dimensions=3
    )

    X = spectral["embedding"]

    Y = spectral["normalized_embedding"]

    eigenvalues = spectral["eigenvalues"]

    embedding_diagnostics(
        X,
        Y,
        eigenvalues
    )

    # ----------------------------------------
    # B. Compare clustering methods
    # ----------------------------------------

    ari_results, assignments, probabilities = (
        compare_clustering_methods(
            Y,
            metadata,
            node_order,
            k_values=(5, 8, 10, 12)
        )
    )

    # ----------------------------------------
    # C. Primary community solution
    # ----------------------------------------

    # We prespecify K=10 and K-means for the primary
    # analysis because there are 10 known classrooms.
    #
    # This avoids choosing the primary vaccination
    # strategy using classroom labels.

    primary_labels = assignments[
        ("K-means", primary_K)
    ]

    # ----------------------------------------
    # D. Degree correction and bridge entropy
    # ----------------------------------------

    community_stats = create_community_statistics(
        G,
        node_order,
        primary_labels
    )

    # ----------------------------------------
    # E. DCSBM block matrix
    # ----------------------------------------

    block_matrix = estimate_block_matrix(
        G,
        node_order,
        primary_labels
    )

    # ----------------------------------------
    # F. Save embedding
    # ----------------------------------------

    embedding_table = pd.DataFrame({

        "id": node_order,

        "dimension_1": X[:, 0],

        "dimension_2": X[:, 1],

        "dimension_3": X[:, 2],

        "normalized_dimension_1": Y[:, 0],

        "normalized_dimension_2": Y[:, 1],

        "normalized_dimension_3": Y[:, 2],

        "community": primary_labels

    })

    prefix = day.lower().replace(" ", "")

    embedding_table.to_csv(
        config.TABLE_DIR /
        f"spectral_embedding_{prefix}.csv",
        index=False
    )

    # ----------------------------------------
    # G. Save community statistics
    # ----------------------------------------

    community_stats.to_csv(
        config.TABLE_DIR /
        f"community_statistics_{prefix}.csv",
        index=False
    )

    # ----------------------------------------
    # H. Save ARI results
    # ----------------------------------------

    ari_results.to_csv(
        config.TABLE_DIR /
        f"ari_comparison_{prefix}.csv",
        index=False
    )

    # ----------------------------------------
    # I. Save block matrix
    # ----------------------------------------

    block_matrix.to_csv(
        config.TABLE_DIR /
        f"block_matrix_{prefix}.csv"
    )

    # ----------------------------------------
    # J. Save clustering membership probabilities
    # ----------------------------------------

    np.savez_compressed(
        config.RESULTS_DIR /
        f"gmm_probabilities_{prefix}.npz",
        **{
            f"K_{K}": probabilities[K]
            for K in probabilities
        }
    )

    # ----------------------------------------
    # K. Generate figures
    # ----------------------------------------

    plot_spectral_embedding(
        Y,
        primary_labels,
        day,
        f"spectral_embedding_{prefix}.png"
    )

    plot_ari_comparison(
        ari_results,
        day
    )

    plot_block_matrix(
        block_matrix,
        day
    )

    # ----------------------------------------
    # L. Print results
    # ----------------------------------------

    print("\nARI RESULTS")

    print(
        ari_results.to_string(index=False)
    )

    print("\nPRIMARY COMMUNITY SIZES")

    print(
        community_stats.groupby(
            "community"
        ).size()
    )

    print("\nCOMMUNITY ANALYSIS COMPLETE")

    return {
        "embedding": X,
        "normalized_embedding": Y,
        "eigenvalues": eigenvalues,
        "community_labels": primary_labels,
        "community_statistics": community_stats,
        "ari_results": ari_results,
        "block_matrix": block_matrix,
        "embedding_table": embedding_table
    }


# ============================================================
# 19. COMPLETE SPECTRAL ANALYSIS PIPELINE
# ============================================================

def run_spectral_analysis():
    """
    Run spectral analysis on both daily networks.
    """

    from data_loader import load_school_networks

    print("\n====================================")
    print("STAT 636 SPECTRAL ANALYSIS")
    print("====================================")

    # Load networks.
    data = load_school_networks()

    G1 = data.G1
    G2 = data.G2

    A1 = data.A1
    A2 = data.A2

    metadata = data.metadata

    node_order = data.node_order

    # Analyze day 1.
    day1_results = analyze_one_day(
        G1,
        A1,
        metadata,
        node_order,
        "Day 1"
    )

    # Analyze day 2.
    day2_results = analyze_one_day(
        G2,
        A2,
        metadata,
        node_order,
        "Day 2"
    )

    # ----------------------------------------
    # Combine community assignments
    # ----------------------------------------

    assignments = pd.DataFrame({

        "id": [str(node).strip() for node in node_order],

        "community_day1": day1_results["community_labels"],

        "community_day2": day2_results["community_labels"]

    })

    assignments = assignments.merge(
        metadata,
        on="id",
        how="left"
    )

    assignments.to_csv(
        config.TABLE_DIR /
        "community_assignments_both_days.csv",
        index=False
    )

    # ----------------------------------------
    # Combine bridge scores
    # ----------------------------------------

    bridge_scores = pd.DataFrame({

        "id": node_order,

        "bridge_entropy_day1":
            day1_results[
                "community_statistics"
            ]["normalized_bridge_entropy"],

        "bridge_entropy_day2":
            day2_results[
                "community_statistics"
            ]["normalized_bridge_entropy"]

    })

    bridge_scores.to_csv(
        config.TABLE_DIR /
        "community_bridge_scores.csv",
        index=False
    )

    print("\n====================================")
    print("SPECTRAL ANALYSIS COMPLETED")
    print("====================================")

    return {
        "day1": day1_results,
        "day2": day2_results,
        "assignments": assignments,
        "bridge_scores": bridge_scores
    }


# ============================================================
# 20. EXECUTE DIRECTLY
# ============================================================

if __name__ == "__main__":

    results = run_spectral_analysis()

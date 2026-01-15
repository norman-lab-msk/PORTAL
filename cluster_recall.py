import numpy as np
from collections import defaultdict
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection


def recall_complexes(
    delta,
    cluster_order,
    complex_gene,
    n_permutations=1000,
    window_size=10,
    random_seed=0,
):
    """
    Guide-level enrichment analysis.

    Args:
        delta: Data matrix (with gene_protospacer indices like 'RPL3_GAGGAAGACGAAGGGAGCTA')
        cluster_order: Cluster ordering indices
        complex_gene: Dictionary of known gene clusters (gene names, not guides)
        n_permutations: Number of permutations for testing
        window_size: Window size for neighborhood
        random_seed: Random seed for reproducibility (recommended to set)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Keep full guide names (don't collapse to genes)
    hierarchical_order = delta.index[cluster_order].to_numpy()

    complex_results, individual_results = calculate_normalized_scores(
        hierarchical_order=hierarchical_order,
        known_clusters=complex_gene,
        window_size=window_size,
        n_permutations=n_permutations,
    )

    complex_results_df = pd.DataFrame.from_dict(complex_results, orient="index")
    complex_results_df = complex_results_df.reset_index().rename(
        columns={"index": "cluster_name"}
    )
    complex_results_df = complex_results_df.sort_values("p_value")

    return complex_results_df, individual_results


def calculate_normalized_scores(
    hierarchical_order, known_clusters, window_size=10, n_permutations=1000
):
    """
    Calculate neighborhood enrichment scores at GUIDE level.

    For each guide, calculates what fraction of its neighbors (excluding itself)
    belong to genes in the same cluster. Each guide is treated independently.

    Args:
        hierarchical_order: numpy array of GUIDE names (gene_protospacer format)
        known_clusters: Dictionary mapping cluster names to lists of GENE names
        window_size: Size of window on each side of target guide
        n_permutations: Number of random permutations for significance testing

    Returns:
    tuple: (summary_dict, individual_scores_dict)
        summary_dict: cluster-level statistics
        individual_scores_dict: {cluster_name: list of (gene, position, score, p_value) tuples}
    """
    guide_to_gene = {}
    for guide in hierarchical_order:
        if "_" in guide:
            gene = guide.split("_")[0]
        else:
            gene = guide
        guide_to_gene[guide] = gene

    summary_results = {}
    individual_results = {}

    for cluster_name, cluster_genes in known_clusters.items():
        cluster_genes_set = set(cluster_genes)

        # Find ALL guide positions for genes in this cluster
        cluster_guide_positions = []
        for pos, guide in enumerate(hierarchical_order):
            gene = guide_to_gene[guide]
            if gene in cluster_genes_set:
                cluster_guide_positions.append(pos)

        if len(cluster_guide_positions) < 2:
            continue

        observed_scores = []
        individual_data = []

        # Test EACH guide independently
        for pos in cluster_guide_positions:
            guide = hierarchical_order[pos]
            start = max(0, pos - window_size)
            end = min(len(hierarchical_order), pos + window_size + 1)

            # Get neighborhood and exclude the guide itself
            neighborhood = hierarchical_order[start:end]
            neighborhood_without_self = [g for g in neighborhood if g != guide]

            # Calculate fraction of neighbors whose GENES are from same cluster
            if len(neighborhood_without_self) > 0:
                neighbor_genes = [guide_to_gene[g] for g in neighborhood_without_self]
                observed_count = sum(
                    1 for g in neighbor_genes if g in cluster_genes_set
                )
                score = observed_count / len(neighborhood_without_self)
            else:
                score = 0.0

            observed_scores.append(score)
            gene = guide_to_gene[guide]
            individual_data.append((gene, pos, score))

        # Permutation per guide
        random_scores = np.zeros(n_permutations)
        guide_random_scores = {
            pos: np.zeros(n_permutations) for pos in cluster_guide_positions
        }

        for perm_idx in range(n_permutations):
            shuffled_order = hierarchical_order.copy()
            np.random.shuffle(shuffled_order)

            perm_scores = []
            for pos in cluster_guide_positions:
                guide = hierarchical_order[pos]

                shuffle_pos = np.where(shuffled_order == guide)[0][0]
                start = max(0, shuffle_pos - window_size)
                end = min(len(shuffled_order), shuffle_pos + window_size + 1)

                neighborhood = shuffled_order[start:end]
                neighborhood_without_self = [g for g in neighborhood if g != guide]

                if len(neighborhood_without_self) > 0:
                    neighbor_genes = [
                        guide_to_gene[g] for g in neighborhood_without_self
                    ]
                    perm_count = sum(
                        1 for g in neighbor_genes if g in cluster_genes_set
                    )
                    perm_score = perm_count / len(neighborhood_without_self)
                else:
                    perm_score = 0.0

                perm_scores.append(perm_score)
                guide_random_scores[pos][perm_idx] = perm_score

            random_scores[perm_idx] = np.mean(perm_scores)

        observed_mean = np.mean(observed_scores)
        random_mean = np.mean(random_scores)
        random_std = np.std(random_scores)

        z_score = (observed_mean - random_mean) / random_std if random_std > 0 else 0

        p_value = (np.sum(random_scores >= observed_mean) + 1) / (n_permutations + 1)

        summary_results[cluster_name] = {
            "observed_score": observed_mean,
            "random_mean": random_mean,
            "random_std": random_std,
            "z_score": z_score,
            "p_value": p_value,
            "enrichment_ratio": (
                observed_mean / random_mean if random_mean > 0 else float("inf")
            ),
            "cluster_size": len(cluster_guide_positions),
        }

        individual_data_with_pvals = []
        for i, (gene, pos, observed_score) in enumerate(individual_data):
            guide_random = guide_random_scores[pos]
            guide_pval = (np.sum(guide_random >= observed_score) + 1) / (
                n_permutations + 1
            )
            individual_data_with_pvals.append((gene, pos, observed_score, guide_pval))

        individual_results[cluster_name] = individual_data_with_pvals

    return summary_results, individual_results


def assign_genes_to_single_cluster_with_distance_check(
    sig_df,
    known_clusters,
    linkage_matrix=None,
    hierarchical_order=None,
    distance_threshold=None,
    distance_percentile=75,
    merge_threshold=0.7,
    max_position_distance=80,
    window_size=10,
):
    """
    Assign each (gene, position) pair to a single cluster.

    Uses position-aware assignment and multi-level tie-breaking.

    Args:
        sig_df: DataFrame with columns ['cluster', 'gene', 'score', 'position', ...]
        known_clusters: Dict mapping cluster names to lists of genes
        max_position_distance: Max position distance for position-aware assignment (default: 80)
        window_size: Window size for proximity checks (default: 10)

    Returns:
        dict: Mapping of (gene, position) -> assigned cluster name
    """

    # Calculate median positions for each cluster
    cluster_positions = {}
    for cluster in sig_df["cluster"].unique():
        cluster_data = sig_df[sig_df["cluster"] == cluster]
        cluster_positions[cluster] = cluster_data["position"].median()

    # Track the order when each (gene, position, cluster) first appears
    first_seen_order = {}
    for idx, row in sig_df.iterrows():
        key = (row["gene"], row["position"], row["cluster"])
        if key not in first_seen_order:
            first_seen_order[key] = idx

    # Also track which cluster appeared FIRST for each (gene, position)
    first_seen_cluster = {}
    for _, row in sig_df.iterrows():
        key = (row["gene"], row["position"])
        if key not in first_seen_cluster:
            first_seen_cluster[key] = row["cluster"]

    # Group by (gene, position) to see which clusters each guide appears in
    gene_pos_clusters = (
        sig_df.groupby(["gene", "position"])["cluster"].apply(set).to_dict()
    )

    # Assign each (gene, position) to a cluster
    gene_pos_assignments = {}

    for (gene, position), clusters in gene_pos_clusters.items():
        if len(clusters) == 1:
            # Only in one cluster
            cluster = list(clusters)[0]
            gene_pos_assignments[(gene, position)] = cluster
        else:
            # In multiple clusters - choose based on size AND position proximity
            cluster_sizes = {}
            cluster_scores = {}
            cluster_dists = {}

            for cluster in clusters:
                # Count how many guides from this cluster are in the data
                cluster_sizes[cluster] = len(sig_df[sig_df["cluster"] == cluster])

                # Get score for this (gene, position, cluster)
                cluster_data = sig_df[
                    (sig_df["gene"] == gene)
                    & (sig_df["position"] == position)
                    & (sig_df["cluster"] == cluster)
                ]
                score = cluster_data["score"].iloc[0] if len(cluster_data) > 0 else 0
                # Convert NaN to 0 for proper comparison (NaN breaks tuple comparisons)
                cluster_scores[cluster] = 0 if pd.isna(score) else score

                # Calculate position distance
                if cluster in cluster_positions:
                    cluster_dists[cluster] = abs(position - cluster_positions[cluster])
                else:
                    cluster_dists[cluster] = float("inf")

            # Filter to nearby clusters (within max_position_distance)
            nearby_clusters = [
                c for c in clusters if cluster_dists[c] <= max_position_distance
            ]

            if nearby_clusters:
                # Pick largest among nearby clusters
                # Tie-breaking: (1) size, (2) score, (3) first-seen cluster, (4) cluster appearance order
                first_cluster = first_seen_cluster.get((gene, position))

                best_cluster = max(
                    nearby_clusters,
                    key=lambda c: (
                        cluster_sizes[c],
                        cluster_scores[c],
                        1 if c == first_cluster else 0,  # Prefer first-seen cluster
                        -first_seen_order.get(
                            (gene, position, c), float("inf")
                        ),  # Earlier appearance wins
                    ),
                )
            else:
                # All clusters are far - pick closest one (even if far)
                best_cluster = min(clusters, key=lambda c: cluster_dists[c])

            gene_pos_assignments[(gene, position)] = best_cluster

    return gene_pos_assignments


def expand_significant_genes_with_neighbors(
    sig_df, hierarchical_order, known_clusters, window_size=10
):
    """
    Expand significant guides to include their neighbors within the window.

    For each significant guide, adds all guides within its neighborhood window whose
    GENES belong to the same original cluster.

    Args:
        sig_df: DataFrame with columns ['gene', 'position', 'cluster', ...]
        hierarchical_order: Array of GUIDES in hierarchical order (gene_protospacer format)
        known_clusters: Dict mapping cluster names to lists of GENE names
        window_size: Size of window on each side (should match recall analysis)

    Returns:
        DataFrame with additional neighbor guides included (with original cluster assignments)
    """
    if isinstance(hierarchical_order, np.ndarray):
        ordered_guides = hierarchical_order
    else:
        ordered_guides = np.array(hierarchical_order)

    # Extract gene from each guide
    guide_to_gene = {}
    for guide in ordered_guides:
        if "_" in guide:
            gene = guide.split("_")[0]
        else:
            gene = guide
        guide_to_gene[guide] = gene

    # Create mapping: (gene_name, position, cluster) -> was it in input sig_df?
    originally_in_sig_df = set()
    for _, row in sig_df.iterrows():
        gene = row["gene"]
        position = row["position"]
        cluster = row["cluster"]
        originally_in_sig_df.add((gene, position, cluster))

    # Get cluster membership from known_clusters (these are GENE names)
    cluster_members = {cluster: set(genes) for cluster, genes in known_clusters.items()}

    # Collect all guides to include (significant + their cluster neighbors)
    genes_to_include = []
    processed = set()  # Track (gene_name, position, cluster) tuples we've already added

    for _, row in sig_df.iterrows():
        gene_name = row["gene"]
        position = row["position"]
        cluster = row["cluster"]

        # Add the significant guide itself
        if (gene_name, position, cluster) not in processed:
            genes_to_include.append(
                {
                    "gene": gene_name,
                    "position": position,
                    "cluster": cluster,
                    "score": row.get("score", None),
                    "p_value": row.get("p_value", None),
                    "fdr_pvalue": row.get("fdr_pvalue", None),
                    "is_originally_significant": True,
                }
            )
            processed.add((gene_name, position, cluster))

        # Get genes that belong to the same original cluster
        if cluster not in cluster_members:
            continue

        cluster_member_set = cluster_members[cluster]

        # Get neighborhood window
        start = max(0, position - window_size)
        end = min(len(ordered_guides), position + window_size + 1)

        # Check each neighbor GUIDE
        for neighbor_pos in range(start, end):
            if neighbor_pos == position:
                continue  # Skip self

            neighbor_guide = ordered_guides[neighbor_pos]
            neighbor_gene = guide_to_gene[neighbor_guide]

            # Only add neighbor if its GENE belongs to the same original cluster
            if neighbor_gene in cluster_member_set:
                if (neighbor_gene, neighbor_pos, cluster) not in processed:
                    # Check if this (gene, position, cluster) tuple was in the original input
                    was_significant = (
                        neighbor_gene,
                        neighbor_pos,
                        cluster,
                    ) in originally_in_sig_df

                    # Only add as neighbor if it was NOT FDR-significant
                    # (FDR-significant ones were already added in the first loop)
                    if not was_significant:
                        genes_to_include.append(
                            {
                                "gene": neighbor_gene,
                                "position": neighbor_pos,
                                "cluster": cluster,
                                "score": None,
                                "p_value": None,
                                "fdr_pvalue": None,
                                "is_originally_significant": False,
                            }
                        )
                        processed.add((neighbor_gene, neighbor_pos, cluster))

    expanded_df = pd.DataFrame(genes_to_include)

    if len(expanded_df) == 0:
        return pd.DataFrame(
            columns=["gene", "position", "cluster", "score", "p_value", "fdr_pvalue"]
        )

    return expanded_df


def recall_and_process_multiple_gene_sets_jointly(
    data_matrix,
    hierarchical_order,
    gene_sets_dict,
    linkage_matrix=None,
    n_permutations=10000,
    p_value_threshold=0.05,
    min_cluster_size=2,
    use_fdr=True,
    expand_with_neighbors=True,
    window_size=10,
    random_seed=None,
    distance_threshold=None,
    distance_percentile=75,
    merge_threshold=0.7,
    max_position_distance=80,
):
    """
    Run recall_complexes for multiple gene sets with JOINT FDR correction and assignment.

    This performs FDR correction across ALL gene sets together, then assigns each gene
    to only ONE cluster across all gene sets, prioritizing larger clusters but respecting
    distance in the dendrogram.

    Args:
        data_matrix: Guide matrix (DataFrame with gene_protospacer indices)
        hierarchical_order: Ordered indices from dendrogram (numpy array)
        gene_sets_dict: Dict mapping set_name -> gene_cluster_dict
                       Example: {'complexes': {cluster1: [genes], ...},
                                'transcriptional': {cluster2: [genes], ...}}
        linkage_matrix: Linkage matrix from hierarchical clustering (optional)
        n_permutations: Number of permutations for recall_complexes
        p_value_threshold: FDR threshold for significance (default 0.05)
        min_cluster_size: Minimum genes per cluster after filtering (default 2)
        use_fdr: Whether to use FDR correction (default True, recommended)
        expand_with_neighbors: Whether to include neighbors of significant genes (default True)
        window_size: Window size for both recall and neighbor expansion (default 10)
        random_seed: Random seed for reproducibility (recommended to set)
        max_position_distance: Max position distance for assignment (default 80)

    Returns:
        dict: Results dictionary with structure:
            {
                'filtered': DataFrame ready for plotting (combined across all sets),
                'significant': DataFrame with all significant genes (with p-values),
                }
            }
    """
    # Keep guides (don't collapse to genes)
    order_guides = data_matrix.index[hierarchical_order].to_numpy()

    # Step 1: Run recall for ALL gene sets and collect results
    print("Running recall for all gene sets...")
    all_individual_results = {}
    all_cluster_to_set = {}  # Maps cluster_name -> set_name

    for set_name, gene_set in gene_sets_dict.items():
        print(f"  Processing gene set: {set_name}")

        _, individual_results = recall_complexes(
            delta=data_matrix,
            cluster_order=hierarchical_order,
            complex_gene=gene_set,
            n_permutations=n_permutations,
            window_size=window_size,
            random_seed=random_seed,
        )

        all_individual_results[set_name] = individual_results

        # Track which set each cluster belongs to
        for cluster_name in individual_results.keys():
            all_cluster_to_set[cluster_name] = set_name

    # Step 2: Combine ALL results for joint FDR correction
    print("\nPerforming joint FDR correction across all gene sets...")
    all_results_flat = []

    for set_name, individual_results in all_individual_results.items():
        for cluster_name, gene_data in individual_results.items():
            for gene, pos, score, pval in gene_data:
                all_results_flat.append(
                    {
                        "set_name": set_name,
                        "cluster": cluster_name,
                        "gene": gene,
                        "position": pos,
                        "score": score,
                        "p_value": pval,
                    }
                )

    combined_df = pd.DataFrame(all_results_flat)

    if len(combined_df) == 0:
        print("No results found across all gene sets")
        return {
            "filtered": pd.DataFrame(),
            "significant": pd.DataFrame(),
            "by_set": {
                set_name: {"filtered": pd.DataFrame(), "significant": pd.DataFrame()}
                for set_name in gene_sets_dict.keys()
            },
        }

    # Apply joint FDR correction
    if use_fdr:
        _, fdr_pvals = fdrcorrection(combined_df["p_value"].values)
        combined_df["fdr_pvalue"] = fdr_pvals
        sig_combined_df = combined_df[
            combined_df["fdr_pvalue"] < p_value_threshold
        ].copy()
    else:
        combined_df["fdr_pvalue"] = combined_df["p_value"]
        sig_combined_df = combined_df[combined_df["p_value"] < p_value_threshold].copy()

    print(
        f"  Found {len(sig_combined_df)} significant gene-cluster pairs (FDR < {p_value_threshold})"
    )
    print(f"  Across {sig_combined_df['gene'].nunique()} unique genes")

    if len(sig_combined_df) == 0:
        print("  No significant results after FDR correction")
        return {
            "filtered": pd.DataFrame(),
            "significant": pd.DataFrame(),
            "by_set": {
                set_name: {"filtered": pd.DataFrame(), "significant": pd.DataFrame()}
                for set_name in gene_sets_dict.keys()
            },
        }

    # Step 3: Expand with neighbors BEFORE joint assignment
    if expand_with_neighbors:
        print("\nExpanding significant genes with neighbors...")

        # Combine all gene sets for expansion
        all_gene_sets_combined = {}
        for set_name, gene_set in gene_sets_dict.items():
            all_gene_sets_combined.update(gene_set)

        sig_combined_df = expand_significant_genes_with_neighbors(
            sig_combined_df,
            order_guides,
            all_gene_sets_combined,
            window_size=window_size,
        )

        print(f"  After expansion: {len(sig_combined_df)} total gene-cluster pairs")

    # Step 4: Joint assignment with distance check
    print("\nPerforming joint gene assignment with distance checking...")

    all_clusters_combined = {}
    for set_name, gene_set in gene_sets_dict.items():
        all_clusters_combined.update(gene_set)

    gene_assignments = assign_genes_to_single_cluster_with_distance_check(
        sig_df=sig_combined_df,
        known_clusters=all_clusters_combined,
        linkage_matrix=linkage_matrix,
        hierarchical_order=order_guides,
        distance_threshold=distance_threshold,
        distance_percentile=distance_percentile,
        merge_threshold=merge_threshold,
        max_position_distance=max_position_distance,
        window_size=window_size,
    )

    # Map using (gene, position) tuple
    sig_combined_df["cluster_reassigned"] = sig_combined_df.apply(
        lambda row: gene_assignments.get(
            (row["gene"], row["position"]), row["cluster"]
        ),
        axis=1,
    )

    sig_combined_df["set_name"] = sig_combined_df["cluster_reassigned"].map(
        all_cluster_to_set
    )

    print(f"  Assigned {len(gene_assignments)} (gene, position) pairs to clusters")

    # Step 4.5: Remove orphaned neighbors
    print(f"\nRemoving orphaned neighbors...")

    # Find FDR-significant guides (those with scores) in each reassigned cluster
    fdr_positions_by_cluster = {}
    for _, row in sig_combined_df.iterrows():
        if pd.notna(row["score"]):  # FDR-significant
            cluster = row["cluster_reassigned"]
            if cluster not in fdr_positions_by_cluster:
                fdr_positions_by_cluster[cluster] = set()
            fdr_positions_by_cluster[cluster].add(row["position"])

    # Check each neighbor
    rows_to_keep = []
    removed_count = 0

    for idx, row in sig_combined_df.iterrows():
        if pd.notna(row["score"]):
            # FDR-significant - always keep
            rows_to_keep.append(idx)
        else:
            # Neighbor - check if there's a nearby FDR-significant guide in same cluster
            cluster = row["cluster_reassigned"]
            position = row["position"]

            if cluster in fdr_positions_by_cluster:
                # Check if any FDR-sig guide in this cluster is within window_size
                fdr_positions = fdr_positions_by_cluster[cluster]
                has_nearby_parent = any(
                    abs(position - fdr_pos) <= window_size for fdr_pos in fdr_positions
                )

                if has_nearby_parent:
                    rows_to_keep.append(idx)
                else:
                    removed_count += 1
            else:
                # No FDR-sig guides in this cluster at all - remove
                removed_count += 1

    sig_combined_df = sig_combined_df.loc[rows_to_keep]

    print(f"  Removed {removed_count} orphaned neighbors")

    # Step 5: Create COMBINED filtered DataFrame for plotting
    combined_filtered = sig_combined_df[
        ["gene", "position", "cluster_reassigned"]
    ].drop_duplicates()

    # Filter by minimum cluster size
    cluster_counts = combined_filtered["cluster_reassigned"].value_counts()
    filtered_clusters = cluster_counts[cluster_counts >= min_cluster_size].index
    combined_filtered = combined_filtered.query(
        "cluster_reassigned in @filtered_clusters"
    )

    print(f"\nCombined results:")
    print(
        f"  Total: {len(combined_filtered)} genes in {combined_filtered['cluster_reassigned'].nunique()} clusters"
    )

    return {
        "filtered": combined_filtered,  # Use this for plotting
        "significant": sig_combined_df,  # Full significant results
    }

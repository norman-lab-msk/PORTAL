import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, dendrogram
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor
from adjustText import adjust_text
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import colorcet as cc


def extract_gene_from_guide(
    guide_id: str, separator: str = "_", position: int = 0
) -> str:
    return str(guide_id).split(separator)[position]


def _wrap_text(text, max_chars_per_line=12, delimiters=None):
    if delimiters is None:
        delimiters = [" ", ",", "-"]

    if len(text) <= max_chars_per_line:
        return text

    split_positions = []
    for i, char in enumerate(text):
        if char in delimiters:
            split_positions.append(i + 1)
        elif char == "(":
            split_positions.append(i)

    if not split_positions:
        lines = []
        for i in range(0, len(text), max_chars_per_line):
            lines.append(text[i : i + max_chars_per_line])
        return "\n".join(lines)

    lines = []
    start = 0
    current_line = ""

    for split_pos in split_positions:
        segment = text[start:split_pos]
        test_line = current_line + segment

        if len(test_line) <= max_chars_per_line:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = segment

        start = split_pos

    if start < len(text):
        remaining = text[start:]
        test_line = current_line + remaining
        if len(test_line) <= max_chars_per_line:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = remaining

    if current_line:
        lines.append(current_line)

    return "\n".join(lines) if len(lines) > 1 else text


def plot_reporter_single_effects(screen, figsize=(8, 6), topn=3, seed=42):
    np.random.seed(seed)

    data = screen.reporter_single_effects.copy()
    data.index = data.index.map(screen.guide_id_dict)

    topn_indices = data.nlargest(topn).index
    topn_values = data.nlargest(topn).values

    lower_percentile = data.quantile(0.01)
    upper_percentile = data.quantile(0.99)

    outliers_mask = (data < lower_percentile) | (data > upper_percentile)
    outlier_indices = data[outliers_mask].index
    outlier_values = data[outliers_mask].values

    sns.set_style("ticks")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 10,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxenplot(
        y=data.values,
        color="#808080",
        linewidth=0.8,
        ax=ax,
        width=0.5,
        outlier_prop=0.01,
        showfliers=False,
    )

    x_positions_all = np.random.normal(0, 0.02, len(outlier_values))
    ax.scatter(
        x_positions_all,
        outlier_values,
        s=30,
        color="#A0A0A0",
        alpha=0.6,
        zorder=2,
        edgecolors="none",
    )

    topn_positions = []
    for val in topn_values:
        idx = np.where(outlier_values == val)[0][0]
        topn_positions.append(x_positions_all[idx])

    ax.scatter(
        topn_positions,
        topn_values,
        s=30,
        color="#E63946",
        alpha=0.8,
        zorder=3,
        linewidth=0,
    )

    texts = []
    for i, (idx, val, x_pos) in enumerate(
        zip(topn_indices, topn_values, topn_positions)
    ):
        text = ax.text(
            x_pos,
            val,
            idx,
            fontsize=9,
            fontweight="normal",
            ha="left",
            va="center",
            color="#4D4D4D",
        )
        texts.append(text)

    adjust_text(
        texts,
        ax=ax,
        expand_points=(1.5, 1.5),
        expand_text=(1.2, 1.2),
        arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.8, alpha=0.7),
        force_points=(0.5, 0.5),
        force_text=(1, 1),
    )

    ax.set_ylabel("reporter single effect")
    ax.set_xlabel("guides")
    ax.set_xticks([])

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.04 * y_range, y_max + 0.06 * y_range)

    ax.set_xlim(-0.325, 0.325)

    sns.despine(ax=ax, left=False, bottom=False)

    plt.tight_layout()

    return fig, ax


def create_complex_interaction_matrix(
    interaction_df: pd.DataFrame,
    cluster_gene_dict: Dict[str, List[str]],
    separator: str = "_",
    position: int = 0,
) -> pd.DataFrame:

    guide_to_gene = {}
    for guide in interaction_df.index:
        guide_to_gene[guide] = extract_gene_from_guide(guide, separator, position)

    genes_in_matrix = set(guide_to_gene.values())

    valid_clusters = {}
    for cluster, genes in cluster_gene_dict.items():
        genes_present = [g for g in genes if g in genes_in_matrix]
        if genes_present:
            valid_clusters[cluster] = genes_present

    if not valid_clusters:
        print("Warning: No clusters have genes present in the interaction matrix")
        return pd.DataFrame()

    excluded_clusters = set(cluster_gene_dict.keys()) - set(valid_clusters.keys())
    if excluded_clusters:
        print(
            f"Excluded {len(excluded_clusters)} clusters with no genes in matrix: {sorted(excluded_clusters)}"
        )

    all_complexes = sorted(list(valid_clusters.keys()))

    complex_matrix = pd.DataFrame(
        index=all_complexes, columns=all_complexes, dtype=float
    )

    for complex1 in all_complexes:
        for complex2 in all_complexes:
            genes1 = set(valid_clusters[complex1])
            genes2 = set(valid_clusters[complex2])

            interactions = []

            for guide1 in interaction_df.index:
                gene1 = guide_to_gene.get(guide1)
                if gene1 in genes1:
                    for guide2 in interaction_df.columns:
                        gene2 = guide_to_gene.get(guide2)
                        if gene2 in genes2:
                            if complex1 != complex2 or guide1 != guide2:
                                value = interaction_df.loc[guide1, guide2]
                                if not pd.isna(value):
                                    interactions.append(value)

            if interactions:
                complex_matrix.loc[complex1, complex2] = np.nanmean(interactions)
            else:
                complex_matrix.loc[complex1, complex2] = np.nan

    print(
        f"\nCreated {len(all_complexes)}x{len(all_complexes)} complex interaction matrix"
    )
    return complex_matrix


def create_complex_interaction_matrix_from_sig_df(
    interaction_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    separator: str = "_",
    position: int = 0,
    cluster_column: str = "cluster_reassigned",
) -> pd.DataFrame:

    cluster_gene_dict = {}
    for _, row in sig_df.iterrows():
        gene = row["gene"]
        cluster = row[cluster_column]
        if cluster not in cluster_gene_dict:
            cluster_gene_dict[cluster] = []
        if gene not in cluster_gene_dict[cluster]:
            cluster_gene_dict[cluster].append(gene)

    return create_complex_interaction_matrix(
        interaction_df, cluster_gene_dict, separator=separator, position=position
    )


def plot_cluster_interaction_barplot(
    single_effects: pd.Series,
    double_effects_matrix: pd.DataFrame,
    cluster1_genes: List[str],
    cluster2_genes: List[str],
    cluster1_name: str = "Cluster 1",
    cluster2_name: str = "Cluster 2",
    separator: str = "_",
    position: int = 0,
    figsize: Tuple[float, float] = (12, 6),
    colors: Optional[List[str]] = None,
    show_individual_guides: bool = True,
    error_bars: bool = True,
    title: Optional[str] = None,
    ylabel: str = "Phenotype",
    ax: Optional[plt.Axes] = None,
    phenotype_type: str = "reporter",
    cluster1_is_list: bool = False,
    cluster2_is_list: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:

    if colors is None:
        if phenotype_type == "identity":
            colors = ["#a7af96", "#848e6c"]
        else:
            colors = ["#d6999a", "#a75f61"]

    cluster1_guides = [
        g
        for g in single_effects.index
        if extract_gene_from_guide(g, separator, position) in cluster1_genes
    ]
    cluster2_guides = [
        g
        for g in single_effects.index
        if extract_gene_from_guide(g, separator, position) in cluster2_genes
    ]

    cluster1_single = single_effects.loc[cluster1_guides]
    cluster2_single = single_effects.loc[cluster2_guides]

    double_values = []
    for g1 in cluster1_guides:
        for g2 in cluster2_guides:
            if (
                g1 in double_effects_matrix.index
                and g2 in double_effects_matrix.columns
            ):
                val = double_effects_matrix.loc[g1, g2]
                if not pd.isna(val):
                    double_values.append(val)
            elif (
                g2 in double_effects_matrix.index
                and g1 in double_effects_matrix.columns
            ):
                val = double_effects_matrix.loc[g2, g1]
                if not pd.isna(val):
                    double_values.append(val)

    cluster1_mean = cluster1_single.mean()
    cluster2_mean = cluster2_single.mean()

    cluster1_sem = cluster1_single.sem()
    cluster2_sem = cluster2_single.sem()

    def wrap_label(label, max_chars=12):
        if label.startswith("Transcription") and "cluster" in label:
            import re

            match = re.search(r"\((.*?)\)", label)
            if match:
                label = match.group(1)

        if len(label) <= max_chars:
            return label

        break_after = [" ", ",", "/", ":", "-"]
        break_before = ["("]
        after_break = [")"]

        break_points = []
        for i, char in enumerate(label):
            if char in break_after and i < len(label) - 1:
                break_points.append(i + 1)
            elif char in break_before and i > 0:
                break_points.append(i)
            elif i > 0 and label[i - 1] in after_break:
                break_points.append(i)

        if not break_points:
            lines = []
            for i in range(0, len(label), max_chars):
                lines.append(label[i : i + max_chars])
            return "\n".join(lines)

        lines = []
        start = 0

        for bp in break_points:
            segment = label[start:bp]

            if len(segment) > max_chars:
                if lines:
                    lines.append(label[start:bp])
                else:
                    for i in range(start, bp, max_chars):
                        lines.append(label[i : i + max_chars])
                start = bp
            elif start == 0 or len(lines[-1]) + len(segment) > max_chars:
                if segment.strip():
                    lines.append(segment)
                start = bp
            else:
                lines[-1] += segment
                start = bp

        if start < len(label):
            remaining = label[start:]
            if lines and len(lines[-1]) + len(remaining) <= max_chars:
                lines[-1] += remaining
            else:
                lines.append(remaining)

        return "\n".join(lines)

    if cluster1_is_list:
        cluster1_label = wrap_label(", ".join(cluster1_genes), max_chars=15)
    else:
        cluster1_label = wrap_label(cluster1_name, max_chars=15)

    if cluster2_is_list:
        cluster2_label = wrap_label(", ".join(cluster2_genes), max_chars=15)
    else:
        cluster2_label = wrap_label(cluster2_name, max_chars=15)

    means = [cluster1_mean, cluster2_mean, np.mean(double_values)]
    sems = [
        cluster1_sem,
        cluster2_sem,
        np.std(double_values, ddof=1) / np.sqrt(len(double_values)),
    ]

    x_positions = [0, 0.95, 1.9]
    bar_labels = [cluster1_label, cluster2_label, "Observed\ndouble"]
    bar_colors = [colors[0], colors[0], colors[1]]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    bars = ax.bar(
        x_positions,
        means,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.2,
        width=0.8,
        zorder=2,
    )

    if error_bars:
        ax.errorbar(
            x_positions,
            means,
            yerr=sems,
            fmt="none",
            ecolor="black",
            capsize=5,
            linewidth=1.5,
            capthick=1.5,
            zorder=3,
        )

    if show_individual_guides:
        np.random.seed(42)
        jitter = 0.15

        x1_vals = np.random.normal(x_positions[0], jitter, size=len(cluster1_single))
        ax.scatter(
            x1_vals,
            cluster1_single.values,
            color="white",
            s=30,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.7,
            zorder=4,
        )

        x2_vals = np.random.normal(x_positions[1], jitter, size=len(cluster2_single))
        ax.scatter(
            x2_vals,
            cluster2_single.values,
            color="white",
            s=30,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.7,
            zorder=4,
        )

        x_double = np.random.normal(x_positions[2], jitter, size=len(double_values))
        ax.scatter(
            x_double,
            double_values,
            color="white",
            s=30,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.7,
            zorder=4,
        )

    ax.axvline(x=1.425, color="gray", linestyle="--", linewidth=1, zorder=1)

    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bar_labels, fontsize=11, ha="center")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for i, (x_pos, n) in enumerate(
        [
            (x_positions[0], len(cluster1_single)),
            (x_positions[1], len(cluster2_single)),
            (x_positions[2], len(double_values)),
        ]
    ):
        ax.text(
            x_pos,
            -0.18,
            f"n = {n}",
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            transform=ax.get_xaxis_transform(),
        )

    plt.tight_layout()

    return fig, ax


def plot_two_interactions_comparison(
    screen,
    cluster1: Union[str, List[str]],
    cluster2_left: Union[str, List[str]],
    cluster2_right: Union[str, List[str]],
    sig_df: Optional[pd.DataFrame] = None,
    cluster_column: str = "cluster_reassigned",
    cluster1_name: Optional[str] = None,
    cluster2_left_name: Optional[str] = None,
    cluster2_right_name: Optional[str] = None,
    use_pcp: bool = False,
    use_identity: bool = False,
    use_guide_ids: bool = False,
    separator: str = "_",
    position: int = 0,
    figsize: Tuple[float, float] = (18, 6),
    colors: Optional[List[str]] = None,
    show_individual_guides: bool = True,
    error_bars: bool = True,
    suptitle: Optional[str] = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    phenotype_type = "identity" if use_identity else "reporter"

    if use_pcp:
        if use_identity:
            if not hasattr(screen, "identity_single_effects_pcp"):
                raise AttributeError(
                    "PCP identity data not available. Run perform_pcp() and save_guide_activities() first."
                )
            single_effects = screen.identity_single_effects_pcp
            double_effects = screen.identity_avg_guide_matrix_pcp
        else:
            if not hasattr(screen, "reporter_single_effects_pcp"):
                raise AttributeError(
                    "PCP reporter data not available. Run perform_pcp() and save_guide_activities() first."
                )
            single_effects = screen.reporter_single_effects_pcp
            double_effects = screen.reporter_avg_guide_matrix_pcp
    else:
        if use_identity:
            if not hasattr(screen, "identity_single_effects"):
                raise AttributeError(
                    "Identity data not available. Run save_guide_activities() first."
                )
            single_effects = screen.identity_single_effects
            double_effects = screen.identity_avg_guide_matrix
        else:
            if not hasattr(screen, "reporter_single_effects"):
                raise AttributeError(
                    "Reporter data not available. Run save_guide_activities() first."
                )
            single_effects = screen.reporter_single_effects
            double_effects = screen.reporter_avg_guide_matrix

    if use_guide_ids:
        if not hasattr(screen, "guide_id_dict"):
            raise AttributeError(
                "guide_id_dict not found. Run screen.generate_guide_ids() first."
            )
        single_effects = single_effects.rename(index=screen.guide_id_dict)
        double_effects = double_effects.rename(
            index=screen.guide_id_dict, columns=screen.guide_id_dict
        )

    def process_cluster(cluster, cluster_name, sig_df, cluster_column):
        if isinstance(cluster, str):
            if sig_df is None:
                raise ValueError("sig_df is required when using cluster names (str)")
            genes = sig_df[sig_df[cluster_column] == cluster]["gene"].unique().tolist()
            name = cluster if cluster_name is None else cluster_name
            is_list = False
        else:
            genes = cluster
            name = "Cluster" if cluster_name is None else cluster_name
            is_list = True
        return genes, name, is_list

    c1_left_genes, c1_left_name, c1_left_is_list = process_cluster(
        cluster1, None, sig_df, cluster_column
    )
    c2l_genes, c2l_name, c2l_is_list = process_cluster(
        cluster2_left, cluster2_left_name, sig_df, cluster_column
    )

    c1_right_genes, c1_right_name, c1_right_is_list = process_cluster(
        cluster1, cluster1_name, sig_df, cluster_column
    )
    c2r_genes, c2r_name, c2r_is_list = process_cluster(
        cluster2_right, cluster2_right_name, sig_df, cluster_column
    )

    plot_cluster_interaction_barplot(
        single_effects=single_effects,
        double_effects_matrix=double_effects,
        cluster1_genes=c1_left_genes,
        cluster2_genes=c2l_genes,
        cluster1_name=c1_left_name,
        cluster2_name=c2l_name,
        separator=separator,
        position=position,
        colors=colors,
        show_individual_guides=show_individual_guides,
        error_bars=error_bars,
        title=f"{c1_left_name} × {c2l_name}",
        ylabel="Phenotype",
        ax=axes[0],
        phenotype_type=phenotype_type,
        cluster1_is_list=c1_left_is_list,
        cluster2_is_list=c2l_is_list,
    )

    plot_cluster_interaction_barplot(
        single_effects=single_effects,
        double_effects_matrix=double_effects,
        cluster1_genes=c1_right_genes,
        cluster2_genes=c2r_genes,
        cluster1_name=c1_right_name,
        cluster2_name=c2r_name,
        separator=separator,
        position=position,
        colors=colors,
        show_individual_guides=show_individual_guides,
        error_bars=error_bars,
        title=f"{c1_right_name} × {c2r_name}",
        ylabel="",
        ax=axes[1],
        phenotype_type=phenotype_type,
        cluster1_is_list=c1_right_is_list,
        cluster2_is_list=c2r_is_list,
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    return fig, axes


def plot_split_symmetric_matrix(
    upper_matrix,
    lower_matrix,
    linkage_matrix,
    sig_genes_df_list=None,
    annotation_labels=None,
    annotation_colors_list=None,
    phenotype=None,
    phenotype_cmap="viridis",
    phenotype_vmin=None,
    phenotype_vmax=None,
    figsize=(12, 12),
    upper_cmap="RdBu_r",
    lower_cmap="viridis",
    upper_vmin=-2,
    upper_vmax=2,
    lower_vmin=None,
    lower_vmax=None,
    dendrogram_ratio=0.15,
    dendrogram_linewidth=0.5,
    show_labels=False,
    show_legend=True,
    keep_square=True,
    cluster_col="cluster_reassigned",
    color_palette="glasbey_bw_minc_20_maxl_70",
) -> Tuple[plt.Figure, dict, dict]:
    """
    Plot symmetric matrix with different data in upper/lower triangles.

    Useful for showing two different phenotypes (reporter/identity) or data types
    (raw/PCP) in a single visualization with cluster annotations from enrichment analysis.

    Args:
        upper_matrix: DataFrame for upper triangle (e.g., reporter GI)
        lower_matrix: DataFrame for lower triangle (e.g., identity GI)
        linkage_matrix: Scipy linkage matrix for dendrogram
        sig_genes_df_list: List of DataFrames from enrichment analysis with
                          'position' and cluster_col columns
        annotation_labels: Labels for each annotation track
        annotation_colors_list: Optional list of color dictionaries per annotation
        phenotype: Series or dict of phenotype values to show as color track
        phenotype_cmap: Colormap for phenotype track
        phenotype_vmin, phenotype_vmax: Phenotype color scale limits
        figsize: Figure size tuple
        upper_cmap, lower_cmap: Colormaps for upper/lower triangles
        upper_vmin, upper_vmax: Upper triangle color limits
        lower_vmin, lower_vmax: Lower triangle color limits
        dendrogram_ratio: Size of dendrogram relative to heatmap
        dendrogram_linewidth: Line width for dendrogram
        show_labels: Show guide labels on axes
        show_legend: Show cluster legends
        keep_square: Keep heatmap square
        cluster_col: Column name for cluster assignments
        color_palette: Colorcet palette name for cluster colors
                      Options: 'glasbey_bw_minc_20_maxl_70', 'glasbey_hv', etc.

    Returns:
        fig: Matplotlib Figure
        axes_dict: Dictionary of all axes
        all_cluster_colors: Dictionary mapping labels to cluster colors

    Example:
        # Show reporter (upper) and identity (lower) genetic interactions
        fig, axes, colors = plot_split_symmetric_matrix(
            upper_matrix=screen.reporter_GI_pcp,
            lower_matrix=screen.identity_GI_pcp,
            linkage_matrix=screen.reporter_Z_pcp,
            sig_genes_df_list=[enrichment_results['filtered']],
            annotation_labels=['Complexes'],
            show_legend=True
        )
    """

    # Get dendrogram ordering
    dend_data = dendrogram(linkage_matrix, no_plot=True)
    order = dend_data["leaves"]

    # Reorder matrices
    upper_ordered = upper_matrix.iloc[order, order]
    lower_ordered = lower_matrix.iloc[order, order]
    n = len(upper_ordered)

    # Create split matrices
    upper_data = np.full((n, n), np.nan)
    lower_data = np.full((n, n), np.nan)

    upper_indices = np.triu_indices(n)
    upper_data[upper_indices] = upper_ordered.values[upper_indices]

    lower_indices = np.tril_indices(n, -1)
    lower_data[lower_indices] = lower_ordered.values[lower_indices]

    if sig_genes_df_list is None:
        sig_genes_df_list = []

    row_sig_genes_df_list = sig_genes_df_list
    col_sig_genes_df_list = sig_genes_df_list  # Same for symmetric

    if annotation_labels is None:
        annotation_labels = [f"Set_{i+1}" for i in range(len(sig_genes_df_list))]

    row_annotation_labels = annotation_labels
    col_annotation_labels = annotation_labels

    if annotation_colors_list is None:
        annotation_colors_list = [None] * len(sig_genes_df_list)

    row_cluster_colors = annotation_colors_list
    col_cluster_colors = annotation_colors_list

    if phenotype is not None:
        if isinstance(phenotype, pd.Series):
            row_phenotype = phenotype
            col_phenotype = phenotype
        elif isinstance(phenotype, dict):
            first_key = list(phenotype.keys())[0]
            row_phenotype = phenotype[first_key]
            col_phenotype = phenotype[first_key]
    else:
        row_phenotype = None
        col_phenotype = None

    n_row_annotations = len(row_sig_genes_df_list)
    n_col_annotations = len(col_sig_genes_df_list)
    n_row_phenotypes = 1 if row_phenotype is not None else 0
    n_col_phenotypes = 1 if col_phenotype is not None else 0

    # Validate input dataframes
    if row_sig_genes_df_list is not None and len(row_sig_genes_df_list) > 0:
        for idx, sig_genes_df in enumerate(row_sig_genes_df_list):
            if "position" not in sig_genes_df.columns:
                raise ValueError(
                    f"sig_genes_df_list[{idx}] is missing required 'position' column. "
                    f"Available columns: {list(sig_genes_df.columns)}"
                )
            if cluster_col not in sig_genes_df.columns:
                raise ValueError(
                    f"sig_genes_df_list[{idx}] is missing required '{cluster_col}' column. "
                    f"Available columns: {list(sig_genes_df.columns)}"
                )

    # Process annotations - collect all clusters by label
    all_row_annotations = []
    all_col_annotations = []
    all_cluster_colors = {}
    all_sorted_clusters = {}

    # Collect ALL unique clusters from both rows and columns for each label
    all_clusters_by_label = {}

    if row_sig_genes_df_list is not None:
        for idx, (sig_genes_df, label) in enumerate(
            zip(row_sig_genes_df_list, row_annotation_labels)
        ):
            if label not in all_clusters_by_label:
                all_clusters_by_label[label] = []

            sig_genes_sorted = sig_genes_df.sort_values("position")

            for _, row in sig_genes_sorted.iterrows():
                cluster = row[cluster_col]
                if cluster and cluster not in all_clusters_by_label[label]:
                    all_clusters_by_label[label].append(cluster)

    # Add clusters from columns
    if col_sig_genes_df_list is not None:
        for idx, (sig_genes_df, label) in enumerate(
            zip(col_sig_genes_df_list, col_annotation_labels)
        ):
            if label not in all_clusters_by_label:
                all_clusters_by_label[label] = []

            sig_genes_sorted = sig_genes_df.sort_values("position")

            for _, row in sig_genes_sorted.iterrows():
                cluster = row[cluster_col]
                if cluster and cluster not in all_clusters_by_label[label]:
                    all_clusters_by_label[label].append(cluster)

    # Generate colors for ALL clusters upfront
    for label in all_clusters_by_label.keys():
        ordered_unique = all_clusters_by_label[label]
        n_clusters = len(ordered_unique)

        idx = row_annotation_labels.index(label)

        if row_cluster_colors[idx] is None:
            palette = getattr(cc, color_palette)

            if n_clusters <= len(palette):
                colors = palette[:n_clusters]
            else:
                colors = [palette[i % len(palette)] for i in range(n_clusters)]

            np.random.seed(0 + idx)
            shuffled_colors = list(colors)
            np.random.shuffle(shuffled_colors)

            row_cluster_colors[idx] = dict(zip(ordered_unique, shuffled_colors))

        all_cluster_colors[label] = row_cluster_colors[idx]
        all_sorted_clusters[label] = ordered_unique

    # Create annotation arrays for rows
    ordered_guides = upper_ordered.index.tolist()

    if row_sig_genes_df_list is not None:
        for idx, (sig_genes_df, label) in enumerate(
            zip(row_sig_genes_df_list, row_annotation_labels)
        ):
            row_annotations = [""] * len(ordered_guides)
            position_to_cluster = dict(
                zip(sig_genes_df["position"], sig_genes_df[cluster_col])
            )

            for pos, cluster in position_to_cluster.items():
                if pos < len(row_annotations):
                    row_annotations[pos] = cluster

            all_row_annotations.append(row_annotations)

    # Create annotation arrays for columns
    if col_sig_genes_df_list is not None:
        for idx, (sig_genes_df, label) in enumerate(
            zip(col_sig_genes_df_list, col_annotation_labels)
        ):
            col_annotations = [""] * len(ordered_guides)
            position_to_cluster = dict(
                zip(sig_genes_df["position"], sig_genes_df[cluster_col])
            )

            for pos, cluster in position_to_cluster.items():
                if pos < len(col_annotations):
                    col_annotations[pos] = cluster

            all_col_annotations.append(col_annotations)

    row_annotation_width = 0.015
    col_annotation_height = 0.015
    row_phenotype_width = 0.01
    col_phenotype_height = 0.01
    spacing = 0.002

    total_row_annotation_width = (
        n_row_annotations * row_annotation_width
        + max(0, n_row_annotations - 1) * spacing
    )
    total_col_annotation_height = (
        n_col_annotations * col_annotation_height
        + max(0, n_col_annotations - 1) * spacing
    )
    total_row_phenotype_width = n_row_phenotypes * row_phenotype_width
    total_col_phenotype_height = n_col_phenotypes * col_phenotype_height

    dendrogram_width = dendrogram_ratio
    dendrogram_height = dendrogram_ratio

    total_left_space = dendrogram_width + spacing
    total_bottom_space = (
        dendrogram_height
        + total_col_annotation_height
        + total_col_phenotype_height
        + spacing * 2
    )
    total_right_space = total_row_annotation_width + total_row_phenotype_width + spacing
    total_top_space = spacing

    available_width = 1.0 - total_left_space - total_right_space - 0.25
    available_height = 1.0 - total_bottom_space - total_top_space - 0.05

    if keep_square:
        heatmap_size = min(available_width, available_height)
        heatmap_width = heatmap_size
        heatmap_height = heatmap_size
    else:
        heatmap_width = available_width
        heatmap_height = available_height

    heatmap_left = total_left_space
    heatmap_bottom = total_bottom_space

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")

    ax_row_dend = fig.add_axes(
        [
            total_left_space - dendrogram_width - spacing,
            heatmap_bottom,
            dendrogram_width,
            heatmap_height,
        ]
    )

    ax_col_dend = fig.add_axes(
        [
            heatmap_left,
            heatmap_bottom + heatmap_height + spacing,
            heatmap_width,
            dendrogram_height,
        ]
    )

    ax_heatmap = fig.add_axes(
        [heatmap_left, heatmap_bottom, heatmap_width, heatmap_height]
    )

    # Plot dendrograms
    with plt.rc_context({"lines.linewidth": dendrogram_linewidth}):
        dendrogram(
            linkage_matrix,
            orientation="left",
            ax=ax_row_dend,
            color_threshold=0,
            above_threshold_color="black",
            no_labels=True,
        )
        dendrogram(
            linkage_matrix,
            orientation="top",
            ax=ax_col_dend,
            color_threshold=0,
            above_threshold_color="black",
            no_labels=True,
        )

    ax_row_dend.axis("off")
    ax_col_dend.axis("off")

    if lower_vmin is None:
        lower_vmin = np.nanpercentile(lower_data, 2)
    if lower_vmax is None:
        lower_vmax = np.nanpercentile(lower_data, 98)

    # Plot split heatmap
    im_upper = ax_heatmap.imshow(
        upper_data,
        cmap=upper_cmap,
        vmin=upper_vmin,
        vmax=upper_vmax,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        rasterized=True,
    )

    im_lower = ax_heatmap.imshow(
        lower_data,
        cmap=lower_cmap,
        vmin=lower_vmin,
        vmax=lower_vmax,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        rasterized=True,
    )

    ax_heatmap.set_xlim(-0.5, n - 0.5)
    ax_heatmap.set_ylim(n - 0.5, -0.5)

    for spine in ax_heatmap.spines.values():
        spine.set_visible(False)

    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])

    axes_dict = {
        "heatmap": ax_heatmap,
        "row_dend": ax_row_dend,
        "col_dend": ax_col_dend,
    }

    if row_phenotype is not None or col_phenotype is not None:
        pheno_series = row_phenotype if row_phenotype is not None else col_phenotype
        pheno_ordered = pheno_series.iloc[order]
        pheno_values = pheno_ordered.values

        if phenotype_vmin is None or phenotype_vmax is None:
            valid_mask = ~np.isnan(pheno_values)
            if valid_mask.any():
                data_min = np.nanmin(pheno_values)
                data_max = np.nanmax(pheno_values)

                max_abs = max(abs(data_min), abs(data_max))
                if phenotype_vmin is None:
                    phenotype_vmin = -max_abs
                if phenotype_vmax is None:
                    phenotype_vmax = max_abs
            else:
                if phenotype_vmin is None:
                    phenotype_vmin = -1
                if phenotype_vmax is None:
                    phenotype_vmax = 1

    current_right = heatmap_left + heatmap_width + spacing

    if row_phenotype is not None:
        pheno_ordered = row_phenotype.iloc[order]
        pheno_values = pheno_ordered.values
        valid_mask = ~np.isnan(pheno_values)

        if phenotype_vmax > phenotype_vmin:
            pheno_normalized = (pheno_values - phenotype_vmin) / (
                phenotype_vmax - phenotype_vmin
            )
            pheno_normalized = np.clip(pheno_normalized, 0, 1)
        else:
            pheno_normalized = np.zeros_like(pheno_values)

        cmap_obj = plt.get_cmap(phenotype_cmap)
        pheno_colors = cmap_obj(pheno_normalized)[:, :3]
        pheno_colors[~valid_mask] = [1.0, 1.0, 1.0]

        ax_row_pheno = fig.add_axes(
            [current_right, heatmap_bottom, row_phenotype_width, heatmap_height]
        )

        pheno_img = pheno_colors.reshape(n, 1, 3)
        ax_row_pheno.imshow(pheno_img, aspect="auto", interpolation="nearest")
        ax_row_pheno.set_xlim(-0.5, 0.5)
        ax_row_pheno.set_ylim(n - 0.5, -0.5)
        ax_row_pheno.set_xticks([])
        ax_row_pheno.set_yticks([])

        for spine in ax_row_pheno.spines.values():
            spine.set_visible(False)

        axes_dict["row_pheno"] = ax_row_pheno
        current_right += row_phenotype_width + spacing

    # Row annotations
    for idx, (annotations, label) in enumerate(
        zip(all_row_annotations, row_annotation_labels)
    ):
        color_dict = all_cluster_colors[label]
        default_color = np.array([1.0, 1.0, 1.0])

        colors_array = np.array(
            [
                (
                    color_dict[cluster]
                    if cluster and cluster in color_dict
                    else default_color
                )
                for cluster in annotations
            ]
        )

        ax_row_anno = fig.add_axes(
            [current_right, heatmap_bottom, row_annotation_width, heatmap_height]
        )

        colors_img = colors_array.reshape(n, 1, 3)
        ax_row_anno.imshow(colors_img, aspect="auto", interpolation="nearest")
        ax_row_anno.set_xlim(-0.5, 0.5)
        ax_row_anno.set_ylim(n - 0.5, -0.5)
        ax_row_anno.set_xticks([])
        ax_row_anno.set_yticks([])

        for spine in ax_row_anno.spines.values():
            spine.set_visible(False)

        axes_dict[f"row_anno_{idx}"] = ax_row_anno
        current_right += row_annotation_width + spacing

    current_bottom = heatmap_bottom - spacing

    if col_phenotype is not None:
        pheno_ordered = col_phenotype.iloc[order]
        pheno_values = pheno_ordered.values
        valid_mask = ~np.isnan(pheno_values)

        if phenotype_vmax > phenotype_vmin:
            pheno_normalized = (pheno_values - phenotype_vmin) / (
                phenotype_vmax - phenotype_vmin
            )
            pheno_normalized = np.clip(pheno_normalized, 0, 1)
        else:
            pheno_normalized = np.zeros_like(pheno_values)

        cmap_obj = plt.get_cmap(phenotype_cmap)
        pheno_colors = cmap_obj(pheno_normalized)[:, :3]
        pheno_colors[~valid_mask] = [1.0, 1.0, 1.0]

        current_bottom -= col_phenotype_height

        ax_col_pheno = fig.add_axes(
            [heatmap_left, current_bottom, heatmap_width, col_phenotype_height]
        )

        pheno_img = pheno_colors.reshape(1, n, 3)
        ax_col_pheno.imshow(pheno_img, aspect="auto", interpolation="nearest")
        ax_col_pheno.set_xlim(-0.5, n - 0.5)
        ax_col_pheno.set_ylim(-0.5, 0.5)
        ax_col_pheno.set_xticks([])
        ax_col_pheno.set_yticks([])

        for spine in ax_col_pheno.spines.values():
            spine.set_visible(False)

        axes_dict["col_pheno"] = ax_col_pheno
        current_bottom -= spacing

    # Column annotations
    for idx, (annotations, label) in enumerate(
        zip(all_col_annotations, col_annotation_labels)
    ):
        color_dict = all_cluster_colors[label]
        default_color = np.array([1.0, 1.0, 1.0])

        colors_array = np.array(
            [
                (
                    color_dict[cluster]
                    if cluster and cluster in color_dict
                    else default_color
                )
                for cluster in annotations
            ]
        )

        current_bottom -= col_annotation_height

        ax_col_anno = fig.add_axes(
            [heatmap_left, current_bottom, heatmap_width, col_annotation_height]
        )

        colors_img = colors_array.reshape(1, n, 3)
        ax_col_anno.imshow(colors_img, aspect="auto", interpolation="nearest")
        ax_col_anno.set_xlim(-0.5, n - 0.5)
        ax_col_anno.set_ylim(-0.5, 0.5)
        ax_col_anno.set_xticks([])
        ax_col_anno.set_yticks([])

        for spine in ax_col_anno.spines.values():
            spine.set_visible(False)

        axes_dict[f"col_anno_{idx}"] = ax_col_anno
        current_bottom -= spacing

    # Add labels to outermost tracks if requested
    if show_labels:
        if n_row_annotations > 0:
            rightmost_anno_ax = axes_dict[f"row_anno_{n_row_annotations-1}"]
            rightmost_anno_ax.set_yticks(range(n))
            rightmost_anno_ax.set_yticklabels(upper_ordered.index, fontsize=6)
            rightmost_anno_ax.yaxis.tick_right()
            rightmost_anno_ax.yaxis.set_label_position("right")

        if n_col_annotations > 0:
            bottommost_anno_ax = axes_dict[f"col_anno_{n_col_annotations-1}"]
            bottommost_anno_ax.set_xticks(range(n))
            bottommost_anno_ax.set_xticklabels(
                upper_ordered.columns, rotation=90, fontsize=6
            )
            bottommost_anno_ax.xaxis.tick_bottom()
            bottommost_anno_ax.xaxis.set_label_position("bottom")

    # Colorbars
    cbar_width = 0.015
    cbar_height = 0.04
    cbar_left = total_left_space - dendrogram_width - spacing + 0.005
    cbar_bottom = heatmap_bottom + heatmap_height + spacing + 0.005

    cbar_ax_upper = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar_upper = plt.colorbar(im_upper, cax=cbar_ax_upper, orientation="vertical")
    cbar_upper.ax.tick_params(labelsize=7, width=0.5, length=2)
    for spine in cbar_ax_upper.spines.values():
        spine.set_linewidth(0.5)

    cbar_ax_lower = fig.add_axes(
        [cbar_left + cbar_width + 0.01, cbar_bottom, cbar_width, cbar_height]
    )
    cbar_lower = plt.colorbar(im_lower, cax=cbar_ax_lower, orientation="vertical")
    cbar_lower.ax.tick_params(labelsize=7, width=0.5, length=2)
    for spine in cbar_ax_lower.spines.values():
        spine.set_linewidth(0.5)

    axes_dict["cbar_upper"] = cbar_ax_upper
    axes_dict["cbar_lower"] = cbar_ax_lower

    if show_legend:
        legend_left = heatmap_left + heatmap_width + total_right_space + 0.04
        legend_top = heatmap_bottom + heatmap_height

        for idx, label in enumerate(row_annotation_labels):
            ordered_unique = all_sorted_clusters[label]
            color_dict = all_cluster_colors[label]

            legend_patches = [
                mpatches.Patch(color=color_dict[cluster], label=cluster)
                for cluster in ordered_unique
                if cluster in color_dict
            ]

            if legend_patches:
                legend_ax = fig.add_axes(
                    [legend_left + 0.5 * idx, legend_top - 0.14, 0.15, 0.20]
                )
                legend_ax.axis("off")
                legend = legend_ax.legend(
                    handles=legend_patches,
                    loc="upper left",
                    frameon=False,
                    fontsize=figsize[0] * 0.66,
                    title=label,
                    title_fontsize=figsize[0] * 0.8,
                )
                legend._legend_box.align = "left"
                axes_dict[f"legend_{idx}"] = legend_ax

    return fig, axes_dict, all_cluster_colors


def plot_half_matrices(
    df1,
    df2,
    linkage_matrix,
    figsize=(12, 12),
    cmap="viridis",
    dendrogram_color="black",
    dendrogram_linewidth=0.5,
    titles=None,
    color_scale=2,
    ticklabels=False,
):
    """
    Plots two symmetric matrices in one plot:
    - Upper triangle + diagonal: df1 values
    - Lower triangle: df2 values
    - Single heatmap visualization
    """

    order = dendrogram(linkage_matrix, no_plot=True)["leaves"]
    df1_ordered = df1.iloc[order, order]
    df2_ordered = df2.iloc[order, order]

    std1 = np.nanstd(df1_ordered.values)
    std2 = np.nanstd(df2_ordered.values)

    # Create combined matrix
    combined = np.zeros_like(df1_ordered.values)
    n = combined.shape[0]

    # Fill upper triangle + diagonal with df1 values
    upper_indices = np.triu_indices(n)
    combined[upper_indices] = df1_ordered.values[upper_indices]

    # Fill lower triangle with df2 values
    lower_indices = np.tril_indices(n, -1)  # Exclude diagonal
    combined[lower_indices] = df2_ordered.values[lower_indices]
    np.fill_diagonal(combined, np.nan)

    combined = pd.DataFrame(
        combined, index=df1_ordered.index, columns=df1_ordered.columns
    )

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.1, 0.9],
        width_ratios=[0.1, 0.9],
        hspace=0.005,
        wspace=0.005,
    )

    ax_row_dend = fig.add_subplot(gs[1, 0])  # Left dendrogram
    ax_col_dend = fig.add_subplot(gs[0, 1])  # Top dendrogram
    ax_heatmap = fig.add_subplot(gs[1, 1])  # Main heatmap

    # Add colorbar in top left corner
    cbar_ax = fig.add_axes([0.14, 0.81, 0.015, 0.06])

    # Plot dendrograms
    with plt.rc_context({"lines.linewidth": dendrogram_linewidth}):
        dendrogram(
            linkage_matrix,
            orientation="left",
            ax=ax_row_dend,
            color_threshold=0,
            link_color_func=lambda x: dendrogram_color,
        )
        dendrogram(
            linkage_matrix,
            ax=ax_col_dend,
            color_threshold=0,
            link_color_func=lambda x: dendrogram_color,
        )
    ax_row_dend.axis("off")
    ax_col_dend.axis("off")

    vmin, vmax = -color_scale, color_scale

    im = sns.heatmap(
        combined,
        ax=ax_heatmap,
        cmap=cmap,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
        yticklabels=ticklabels,
        xticklabels=ticklabels,
        rasterized=True,
    )

    if ticklabels:
        ax_heatmap.yaxis.set_label_position("right")
        ax_heatmap.yaxis.tick_right()
        plt.setp(ax_heatmap.get_xticklabels(), rotation=90, ha="center", fontsize=6)
        plt.setp(ax_heatmap.get_yticklabels(), rotation=0, ha="left", fontsize=6)

    if titles:
        ax_col_dend.set_title(
            f"{titles[0]} (Upper) | {titles[1]} (Lower)", pad=20, fontsize=12
        )

    cbar = plt.colorbar(im.get_children()[0], cax=cbar_ax, orientation="vertical")
    cbar.ax.tick_params(labelsize=8, width=0.5, length=2)
    for spine in cbar_ax.spines.values():
        spine.set_linewidth(0.5)

    return fig


def plot_complex_heatmap(
    complex_matrix,
    figsize=(8, 8),
    cmap="RdBu_r",
    color_scale=None,
    use_cosine_clustering=True,
    optimal_leaf=True,
    title=None,
    show_labels=True,
    label_fontsize=10,
) -> plt.Figure:

    clean_matrix = complex_matrix.dropna(how="all", axis=0).dropna(how="all", axis=1)
    matrix_for_clustering = clean_matrix.fillna(0)

    if color_scale is None:
        color_scale = max(abs(clean_matrix.min().min()), abs(clean_matrix.max().max()))

    if use_cosine_clustering:
        similarity_matrix = cosine_similarity(matrix_for_clustering)

        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.nan_to_num(distance_matrix, nan=1.0)
        distance_matrix = np.clip(distance_matrix, 0, 2)

        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method="average")

        if optimal_leaf:
            linkage_matrix = optimal_leaf_ordering(linkage_matrix, condensed_dist)
    else:
        linkage_matrix = linkage(
            matrix_for_clustering, method="average", metric="correlation"
        )
        if optimal_leaf:
            condensed_dist = pdist(matrix_for_clustering, metric="correlation")
            linkage_matrix = optimal_leaf_ordering(linkage_matrix, condensed_dist)

    g = sns.clustermap(
        clean_matrix,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        cmap=cmap,
        center=0,
        vmin=-color_scale,
        vmax=color_scale,
        figsize=figsize,
        dendrogram_ratio=0.15,
        cbar_pos=(0.03, 0.91, 0.03, 0.08),
    )

    if show_labels:
        g.ax_heatmap.set_xticklabels([])
        g.ax_heatmap.set_xticks([])
        g.ax_heatmap.set_yticklabels(
            g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=label_fontsize
        )
    else:
        g.ax_heatmap.set_xticklabels([])
        g.ax_heatmap.set_yticklabels([])

    if title:
        g.fig.suptitle(title, y=0.98, fontsize=14)

    return g.fig


def plot_separate_model_query(
    query,
    screen=None,
    single_pairs_df=None,
    double_effects=None,
    use_pcp=False,
    use_identity=False,
    guide_id_dict=None,
    model_params=None,
    epsilon=1.35,
    alpha=0.0001,
    figsize=(10, 10),
    s=20,
    cmap="coolwarm",
    vmin=-2.5,
    vmax=2.5,
    highlight_genes=None,
    annotate_outliers=True,
    outlier_threshold=1.0,
    show_legend=True,
) -> plt.Figure:

    if screen is not None:
        if use_pcp:
            if use_identity:
                single_pairs_df = screen.identity_avg_singleton_pairs_pcp
                double_effects = screen.identity_avg_guide_matrix_pcp
            else:
                single_pairs_df = screen.reporter_avg_singleton_pairs_pcp
                double_effects = screen.reporter_avg_guide_matrix_pcp
        else:
            if use_identity:
                single_pairs_df = screen.identity_avg_singleton_pairs
                double_effects = screen.identity_avg_guide_matrix
            else:
                single_pairs_df = screen.reporter_avg_singleton_pairs
                double_effects = screen.reporter_avg_guide_matrix

        if guide_id_dict is None and hasattr(screen, "guide_id_dict"):
            guide_id_dict = screen.guide_id_dict

    if single_pairs_df is None or double_effects is None:
        raise ValueError(
            "Either provide 'screen' or both 'single_pairs_df' and 'double_effects'"
        )

    if isinstance(single_pairs_df, pd.DataFrame):
        single_effects = single_pairs_df.groupby("guide").mean().iloc[:, 0]
    else:
        single_effects = single_pairs_df.groupby("guide").mean()

    query_single = single_effects[query]
    target_singles = single_effects.copy()
    double_effects_for_query = double_effects.loc[query, target_singles.index]

    mask = ~np.isnan(double_effects_for_query.values)
    X_raw = target_singles.values[mask]
    y_raw = double_effects_for_query.values[mask]
    target_genes_masked = target_singles.index[mask]

    if model_params is not None and query in model_params:
        a_opt, b_opt, c_opt = model_params[query]
    else:

        def huber_loss(residuals, delta=epsilon):
            abs_residuals = np.abs(residuals)
            quadratic = 0.5 * residuals**2
            linear = delta * (abs_residuals - 0.5 * delta)
            return np.where(abs_residuals <= delta, quadratic, linear).sum()

        def objective(params, X, y, alpha_reg):
            b, a = params
            y_pred = a * X**2 + b * X
            residuals = y - y_pred
            loss = huber_loss(residuals, delta=epsilon)
            regularization = alpha_reg * (a**2 + b**2)
            return loss + regularization

        def monotonicity_constraint(params, X_min, X_max):
            b, a = params
            if abs(a) < 1e-10:
                return b
            if a > 0:
                min_deriv = 2 * a * X_min + b
            else:
                min_deriv = 2 * a * X_max + b
            return min_deriv

        y_adjusted = y_raw - query_single
        X = X_raw
        X_min, X_max = np.min(X), np.max(X)

        initial_params = np.array([1.0, 0.0])
        constraint = {
            "type": "ineq",
            "fun": lambda params: monotonicity_constraint(params, X_min, X_max),
        }

        result = minimize(
            objective,
            initial_params,
            args=(X, y_adjusted, alpha),
            method="SLSQP",
            constraints=[constraint],
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if result.success:
            b_opt, a_opt = result.x
        else:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_features = poly.fit_transform(X.reshape(-1, 1))
            model = HuberRegressor(epsilon=epsilon, alpha=alpha, fit_intercept=False)
            model.fit(X_features, y_adjusted)
            b_opt, a_opt = model.coef_[0], model.coef_[1]

        c_opt = query_single

    expected = a_opt * X_raw**2 + b_opt * X_raw + c_opt
    residuals = y_raw - expected
    sigma = np.std(residuals)
    z_scores = residuals / sigma

    r_squared = 1 - np.sum(residuals**2) / np.sum((y_raw - np.mean(y_raw)) ** 2)

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        X_raw,
        y_raw,
        c=residuals,
        s=s,
        alpha=0.8,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0,
    )

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("GI score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    x_fit = np.linspace(X_raw.min(), X_raw.max(), 1000)
    y_fit = a_opt * x_fit**2 + b_opt * x_fit + c_opt
    ax.plot(
        x_fit,
        y_fit,
        "grey",
        linestyle=":",
        linewidth=1.5,
        label="guide-specific quadratic fit",
    )

    texts = []

    if highlight_genes is not None:
        highlight_indices = [
            i for i, gene in enumerate(target_genes_masked) if gene in highlight_genes
        ]
        if highlight_indices:
            ax.scatter(
                X_raw[highlight_indices],
                y_raw[highlight_indices],
                s=100,
                facecolors="none",
                edgecolors="yellow",
                linewidth=2,
            )
            for idx in highlight_indices:
                gene = target_genes_masked[idx]
                display_name = guide_id_dict.get(gene, gene) if guide_id_dict else gene
                text = ax.annotate(display_name, (X_raw[idx], y_raw[idx]), fontsize=6)
                texts.append(text)

    if annotate_outliers:
        outlier_indices = np.where(np.abs(z_scores) > outlier_threshold)[0]
        for idx in outlier_indices:
            gene = target_genes_masked[idx]
            if highlight_genes is not None and gene in highlight_genes:
                continue
            display_name = guide_id_dict.get(gene, gene) if guide_id_dict else gene
            text = ax.annotate(
                display_name, (X_raw[idx], y_raw[idx]), fontsize=6, alpha=0.7
            )
            texts.append(text)

    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    display_query = guide_id_dict.get(query, query) if guide_id_dict else query
    ax.set_title(f"{display_query}", fontsize=10, pad=10)
    ax.set_xlabel("target guide single effects", fontsize=9)
    ax.set_ylabel("double knockout phenotypes", fontsize=9)

    if show_legend:
        ax.legend(fontsize=7, frameon=False)

    ax.set_box_aspect(1)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sns.despine()
    plt.tight_layout()

    return fig


def create_complex_matrix_from_screen(
    screen,
    cluster_gene_dict: Dict[str, List[str]],
    use_pcp: bool = False,
    use_identity: bool = False,
    separator: str = "_",
    position: int = 0,
) -> pd.DataFrame:

    if use_pcp:
        if use_identity:
            source = screen.identity_avg_guide_matrix_pcp
        else:
            source = screen.reporter_avg_guide_matrix_pcp
    else:
        if use_identity:
            source = screen.identity_avg_guide_matrix
        else:
            source = screen.reporter_avg_guide_matrix

    return create_complex_interaction_matrix(
        source, cluster_gene_dict, separator=separator, position=position
    )


def plot_cluster_correlations(
    screen,
    cluster: Union[str, List[str]],
    sig_df: Optional[pd.DataFrame] = None,
    cluster_col: str = "cluster_reassigned",
    matrix_type: str = "gi",
    use_pcp: bool = False,
    use_identity: bool = False,
    separator: str = "_",
    position: int = 0,
    similarity_metric: str = "pearson",
    mask_diagonal: bool = True,
    figsize: Tuple[float, float] = (10, 10),
    cmap: str = "RdBu_r",
    vmin: float = -1,
    vmax: float = 1,
    show_values: bool = False,
    value_fontsize: int = 8,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, pd.DataFrame]:

    if matrix_type == "gi":
        if use_pcp:
            if use_identity:
                source_matrix = screen.z_identity_GI_pcp
            else:
                source_matrix = screen.z_reporter_GI_pcp
        else:
            if use_identity:
                source_matrix = screen.z_identity_GI
            else:
                source_matrix = screen.z_reporter_GI
    else:
        if use_pcp:
            if use_identity:
                source_matrix = screen.identity_avg_guide_matrix_pcp
            else:
                source_matrix = screen.reporter_avg_guide_matrix_pcp
        else:
            if use_identity:
                source_matrix = screen.identity_avg_guide_matrix
            else:
                source_matrix = screen.reporter_avg_guide_matrix

    all_genes = []
    cluster_names = []
    gene_to_cluster = {}

    if isinstance(cluster, str):
        if sig_df is None:
            raise ValueError("sig_df is required when cluster is a string")
        genes = sig_df[sig_df[cluster_col] == cluster]["gene"].unique().tolist()
        all_genes.extend(genes)
        cluster_name = cluster
        cluster_names = [cluster]
        for gene in genes:
            gene_to_cluster[gene] = cluster
    elif isinstance(cluster, list) and len(cluster) > 0 and isinstance(cluster[0], str):
        if sig_df is not None and all(
            c in sig_df[cluster_col].unique() for c in cluster
        ):
            cluster_names = cluster
            for clust in cluster:
                genes = sig_df[sig_df[cluster_col] == clust]["gene"].unique().tolist()
                all_genes.extend(genes)
                for gene in genes:
                    gene_to_cluster[gene] = clust
            cluster_name = f"{len(cluster_names)} clusters"
        else:
            all_genes = cluster
            cluster_name = f"Manual ({len(all_genes)} genes)"
    else:
        all_genes = cluster
        cluster_name = f"Manual ({len(all_genes)} genes)"

    all_guides = []
    guide_to_cluster = {}
    for gene in all_genes:
        gene_guides = [
            idx for idx in source_matrix.index if idx.split(separator)[position] == gene
        ]
        all_guides.extend(gene_guides)

        if gene in gene_to_cluster:
            for guide in gene_guides:
                guide_to_cluster[guide] = gene_to_cluster[gene]

    if len(all_guides) == 0:
        raise ValueError(f"No guides found for genes: {all_genes}")

    cluster_data = source_matrix.loc[all_guides]

    if similarity_metric == "cosine":
        n = len(cluster_data)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                a = cluster_data.iloc[i].values
                b = cluster_data.iloc[j].values

                mask = ~np.isnan(a) & ~np.isnan(b)
                a_masked = a[mask]
                b_masked = b[mask]

                if len(a_masked) == 0:
                    sim_matrix[i, j] = np.nan
                else:
                    norm_a = np.linalg.norm(a_masked)
                    norm_b = np.linalg.norm(b_masked)
                    if norm_a == 0 or norm_b == 0:
                        sim_matrix[i, j] = np.nan
                    else:
                        sim_matrix[i, j] = np.dot(a_masked, b_masked) / (
                            norm_a * norm_b
                        )

        corr_matrix = pd.DataFrame(
            sim_matrix, index=cluster_data.index, columns=cluster_data.index
        )
    else:
        corr_matrix = cluster_data.T.corr(method=similarity_metric)

    if mask_diagonal:
        np.fill_diagonal(corr_matrix.values, np.nan)

    guide_id_mapping = {}
    if hasattr(screen, "guide_id_dict"):
        guide_id_dict = screen.guide_id_dict
        for orig_guide in corr_matrix.index:
            guide_id_mapping[guide_id_dict.get(orig_guide, orig_guide)] = orig_guide
        corr_matrix.index = [guide_id_dict.get(g, g) for g in corr_matrix.index]
        corr_matrix.columns = [guide_id_dict.get(g, g) for g in corr_matrix.columns]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        annot=show_values,
        fmt=".2f" if show_values else "",
        annot_kws={"fontsize": value_fontsize},
        cbar_kws={"label": similarity_metric + " similarity", "shrink": 0.5},
        ax=ax,
    )

    if len(cluster_names) > 1 and guide_to_cluster:
        guide_clusters = []
        for guide_id in corr_matrix.index:
            orig_guide = guide_id_mapping.get(guide_id, guide_id)
            guide_clusters.append(guide_to_cluster.get(orig_guide, None))

        cluster_blocks = {}
        for i, clust in enumerate(guide_clusters):
            if clust is not None:
                if clust not in cluster_blocks:
                    cluster_blocks[clust] = {"start": i, "end": i}
                else:
                    cluster_blocks[clust]["end"] = i

        for clust, bounds in cluster_blocks.items():
            start = bounds["start"]
            end = bounds["end"]
            width = end - start + 1

            rect = plt.Rectangle(
                (start, start),
                width,
                width,
                fill=False,
                edgecolor="black",
                linewidth=1,
                clip_on=False,
            )
            ax.add_patch(rect)

        ax.set_xticklabels([])

        for clust, bounds in cluster_blocks.items():
            start = bounds["start"]
            end = bounds["end"]
            center = (start + end) / 2

            wrapped_label = _wrap_text(clust)

            ax.text(
                center,
                len(corr_matrix) + 1,
                wrapped_label,
                ha="center",
                va="top",
                fontsize=10,
                rotation=0,
            )
    else:
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=9)

    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    if title is None:
        phenotype = "Identity" if use_identity else "Reporter"
        data_type = "PCP" if use_pcp else "Raw"
        matrix_desc = "GI" if matrix_type == "gi" else "Phenotype"
        metric_name = similarity_metric + " similarity"
        title = f"{phenotype} {matrix_desc} {metric_name}"

    ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()

    return fig, corr_matrix

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import gaussian_kde


def calculate_cluster_centers_density(
    umap_df,
    significant_df,
    x_col="x",
    y_col="y",
    cluster_col="cluster_reassigned",
    gene_col="gene",
    score_col="score",
):
    """
    Calculate cluster centers using density-based method (mode of KDE).
    Aggregates multiple guides per gene using mean of scores.

    Args:
        umap_df: DataFrame with gene names as index and UMAP coordinates
        significant_df: DataFrame from recall results (can have multiple guides per gene)
        x_col, y_col: Column names for coordinates in umap_df
        cluster_col: Column name for cluster assignments
        gene_col: Column name for gene names
        score_col: Column name for enrichment scores

    Returns:
        DataFrame with cluster centers

    Example:
        >>> centers = calculate_cluster_centers_density(
        ...     umap_df=merged_coordinates_df.set_index("gene")[["x", "y"]],
        ...     significant_df=cluster_significant,
        ...     x_col='x', y_col='y'
        ... )
    """
    cluster_genes = (
        significant_df.groupby([gene_col, cluster_col])[score_col].mean().reset_index()
    )

    print(
        f"Collapsed {len(significant_df)} guide-cluster pairs to {len(cluster_genes)} unique gene-cluster pairs"
    )

    cluster_centers = []

    for cluster_name in cluster_genes[cluster_col].unique():
        genes_data = cluster_genes[cluster_genes[cluster_col] == cluster_name]

        # Filter to genes with UMAP coordinates
        genes_with_coords = genes_data[genes_data[gene_col].isin(umap_df.index)]

        if len(genes_with_coords) == 0:
            print(
                f"Warning: No UMAP coordinates for cluster '{cluster_name}'. Skipping."
            )
            continue

        genes_list = genes_with_coords[gene_col].tolist()
        x_coords = umap_df.loc[genes_list, x_col].values
        y_coords = umap_df.loc[genes_list, y_col].values

        # Filter NaN coordinates
        valid_mask = ~np.isnan(x_coords) & ~np.isnan(y_coords)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]

        if len(x_coords) == 0:
            print(
                f"Warning: All coordinates NaN for cluster '{cluster_name}'. Skipping."
            )
            continue

        center_x, center_y = _calculate_density_center(x_coords, y_coords)

        if np.isnan(center_x) or np.isnan(center_y):
            print(f"Warning: Cluster '{cluster_name}' center is NaN. Skipping.")
            continue

        cluster_centers.append(
            {
                "cluster_name": cluster_name,
                "emb_variable_x": center_x,
                "emb_variable_y": center_y,
                "n_genes": len(genes_with_coords),
                "n_genes_valid_coords": len(x_coords),
            }
        )

    cluster_centers_df = pd.DataFrame(cluster_centers).set_index("cluster_name")

    print(f"\nCalculated centers for {len(cluster_centers_df)} clusters")
    print(
        f"Total genes with valid coordinates: {cluster_centers_df['n_genes_valid_coords'].sum()}"
    )

    return cluster_centers_df


def _calculate_density_center(x_coords, y_coords):
    """Calculate center as mode (highest density point) using KDE"""
    if len(x_coords) < 3:
        return np.median(x_coords), np.median(y_coords)

    coords = np.vstack([x_coords, y_coords])

    try:
        kde = gaussian_kde(coords, bw_method="scott")
    except np.linalg.LinAlgError:
        return np.median(x_coords), np.median(y_coords)

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1

    x_grid = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 50)
    y_grid = np.linspace(y_min - 0.1 * y_range, y_max + 0.1 * y_range, 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Find maximum density
    Z = kde(positions).reshape(X.shape)
    max_idx = np.unravel_index(Z.argmax(), Z.shape)

    return X[max_idx], Y[max_idx]


def add_complex_edges_curved(
    ax,
    complex_matrix,
    positions_df,
    threshold_positive,
    threshold_negative,
    max_edges_per_node=3,
    width_scale=2,
    alpha=0.4,
    zorder=1,
    curvature=0.03,
    node_avoidance_margin=1,
    background_points=None,
    background_x_col="x",
    background_y_col="y",
    background_avoidance_strength=0.2,
    negative_color="#9f8db8",
    positive_color="#eab679",
):
    """
    Add curved interaction edges with node avoidance.

    Example:
        >>> edges = add_complex_edges_curved(
        ...     ax=ax,
        ...     complex_matrix=reporter_cluster_matrix_recalled,
        ...     positions_df=cluster_centers_weighted,
        ...     threshold_positive=reporter_cluster_matrix_recalled.stack().quantile(0.975),
        ...     threshold_negative=abs(reporter_cluster_matrix_recalled.stack().quantile(0.025)),
        ...     max_edges_per_node=3,
        ...     background_points=merged_coordinates_df
        ... )
    """
    # Clean matrix
    clean_matrix = (
        complex_matrix.dropna(how="all", axis=0).dropna(how="all", axis=1).fillna(0)
    )
    valid_complexes = [c for c in clean_matrix.index if c in positions_df.index]

    if not valid_complexes:
        print("Warning: No complexes found in positions_df")
        return []

    # Collect edges
    positive_edges = []
    negative_edges = []

    for i, c1 in enumerate(valid_complexes):
        for j, c2 in enumerate(valid_complexes):
            if i < j:
                interaction = clean_matrix.loc[c1, c2]
                abs_interaction = abs(interaction)

                threshold = (
                    threshold_positive if interaction > 0 else threshold_negative
                )

                if abs_interaction >= threshold:
                    edge = {
                        "complex1": c1,
                        "complex2": c2,
                        "weight": abs_interaction,
                        "interaction": interaction,
                        "sign": "positive" if interaction > 0 else "negative",
                    }
                    (positive_edges if interaction > 0 else negative_edges).append(edge)

    print(
        f"Edges above threshold: {len(positive_edges)} positive, {len(negative_edges)} negative"
    )

    if not positive_edges and not negative_edges:
        print("No edges above thresholds")
        return []

    # Filter edges (same limit for all nodes)
    edges_to_draw = []
    if positive_edges:
        edges_to_draw.extend(_filter_edges(positive_edges, max_edges_per_node))
    if negative_edges:
        edges_to_draw.extend(_filter_edges(negative_edges, max_edges_per_node))

    if not edges_to_draw:
        return []

    max_weight = max(e["weight"] for e in edges_to_draw)

    # Prepare node positions
    node_positions = positions_df[["emb_variable_x", "emb_variable_y"]].values
    node_names = positions_df.index.tolist()
    center_x, center_y = np.mean(node_positions[:, 0]), np.mean(node_positions[:, 1])

    # Prepare background
    if background_points is not None:
        background_positions = background_points[
            [background_x_col, background_y_col]
        ].values
        print(
            f"Using {len(background_positions)} background points (strength={background_avoidance_strength})"
        )
    else:
        background_positions = None

    # Prepare edges with positions
    edges_with_positions = []
    for edge in edges_to_draw:
        edge_copy = edge.copy()
        edge_copy["pos1"] = (
            positions_df.loc[edge["complex1"], "emb_variable_x"],
            positions_df.loc[edge["complex1"], "emb_variable_y"],
        )
        edge_copy["pos2"] = (
            positions_df.loc[edge["complex2"], "emb_variable_x"],
            positions_df.loc[edge["complex2"], "emb_variable_y"],
        )
        edges_with_positions.append(edge_copy)

    # Draw edges
    edge_patches = []

    for edge in edges_to_draw:
        x1 = positions_df.loc[edge["complex1"], "emb_variable_x"]
        y1 = positions_df.loc[edge["complex1"], "emb_variable_y"]
        x2 = positions_df.loc[edge["complex2"], "emb_variable_x"]
        y2 = positions_df.loc[edge["complex2"], "emb_variable_y"]

        linewidth = (edge["weight"] / max_weight) * width_scale + 0.5
        color = positive_color if edge["sign"] == "positive" else negative_color

        # Calculate avoidance
        edge_curvature, direction = _calculate_avoidance(
            x1,
            y1,
            x2,
            y2,
            edge["complex1"],
            edge["complex2"],
            node_positions,
            node_names,
            curvature,
            node_avoidance_margin,
            background_positions,
            background_avoidance_strength,
            edges_with_positions,
            (center_x, center_y),
        )

        # Draw arc
        patch = _draw_arc(
            ax,
            x1,
            y1,
            x2,
            y2,
            color,
            linewidth,
            alpha,
            zorder,
            edge_curvature,
            direction,
        )
        edge_patches.append(patch)

    n_pos = sum(1 for e in edges_to_draw if e["sign"] == "positive")
    n_neg = sum(1 for e in edges_to_draw if e["sign"] == "negative")
    print(f"\nDrew {len(edges_to_draw)} edges: {n_pos} positive, {n_neg} negative")

    return edge_patches


def _filter_edges(edges, limit):
    """Simple greedy filtering by edge weight"""
    sorted_edges = sorted(edges, key=lambda e: e["weight"], reverse=True)
    accepted = set()
    node_degree = defaultdict(int)

    for edge in sorted_edges:
        n1, n2 = edge["complex1"], edge["complex2"]
        if node_degree[n1] < limit and node_degree[n2] < limit:
            accepted.add(tuple(sorted([n1, n2])))
            node_degree[n1] += 1
            node_degree[n2] += 1

    edge_lookup = {
        tuple(sorted([e["complex1"], e["complex2"]])): e for e in sorted_edges
    }
    return [edge_lookup[eid] for eid in accepted]


def _calculate_avoidance(
    x1,
    y1,
    x2,
    y2,
    node1_name,
    node2_name,
    node_positions,
    node_names,
    base_curvature,
    margin,
    background_positions,
    background_strength,
    all_edges,
    center_point,
):
    """Calculate curve direction and magnitude to avoid obstacles"""
    dx, dy = x2 - x1, y2 - y1
    edge_length = np.sqrt(dx**2 + dy**2)

    if edge_length < 1e-6:
        return base_curvature, 1

    # Perpendicular directions
    perp_x, perp_y = -dy / edge_length, dx / edge_length

    # Edge midpoint
    edge_mid_x, edge_mid_y = (x1 + x2) / 2, (y1 + y2) / 2

    # Outward direction preference
    center_x, center_y = center_point
    to_edge_x, to_edge_y = edge_mid_x - center_x, edge_mid_y - center_y
    to_edge_dist = np.sqrt(to_edge_x**2 + to_edge_y**2)

    if to_edge_dist > 1e-6:
        to_edge_x /= to_edge_dist
        to_edge_y /= to_edge_dist

    dot_plus = perp_x * to_edge_x + perp_y * to_edge_y
    dot_minus = -perp_x * to_edge_x + -perp_y * to_edge_y
    outward_direction = 1 if dot_plus > dot_minus else -1

    # Test both directions with 2x curvature
    test_curvature = base_curvature * 2.0
    direction_results = []

    for direction in [1, -1]:
        total_density = 0
        min_node_distance = float("inf")

        # Sample along curve
        for t in np.linspace(0.1, 0.9, 15):
            curve_height = 4 * t * (1 - t) * test_curvature * edge_length * direction
            px, py = x1 + t * dx, y1 + t * dy
            cx, cy = px + perp_x * curve_height, py + perp_y * curve_height

            # Position weight (middle matters most)
            position_weight = max(0.05, 1.0 - 2.0 * (t - 0.5) ** 4)

            # Node density
            density_radius = edge_length * 0.3 * margin
            for i, node_name in enumerate(node_names):
                if node_name in [node1_name, node2_name]:
                    continue

                node_x, node_y = node_positions[i]
                dist = np.sqrt((cx - node_x) ** 2 + (cy - node_y) ** 2)
                min_node_distance = min(min_node_distance, dist)

                if dist < density_radius:
                    weight = (1 - dist / density_radius) * position_weight
                    total_density += weight

            # Background density
            if background_positions is not None:
                bg_dists = np.sqrt(
                    (background_positions[:, 0] - cx) ** 2
                    + (background_positions[:, 1] - cy) ** 2
                )
                bg_radius = edge_length * 0.25 * margin
                bg_count = np.sum(bg_dists < bg_radius)
                total_density += bg_count * background_strength * position_weight

        # Outward bonus for long edges
        typical_spacing = np.median(
            np.sqrt(np.sum(np.diff(node_positions, axis=0) ** 2, axis=1))
        )
        is_long = edge_length > typical_spacing * 1.5
        outward_bonus = (
            min(edge_length / typical_spacing - 1.0, 3.0)
            if is_long and direction == outward_direction
            else 0
        )

        direction_results.append(
            {
                "direction": direction,
                "score": total_density - outward_bonus,
                "min_dist": min_node_distance,
            }
        )

    # Choose lower score (less dense)
    best = min(direction_results, key=lambda x: x["score"])
    best_direction = best["direction"]

    # Adjust curvature
    adjusted_curvature = base_curvature

    # Scale for long edges
    edge_length_ratio = edge_length / typical_spacing
    if edge_length_ratio > 1.0:
        adjusted_curvature *= 1.0 + min(edge_length_ratio**1.15 - 1.0, 5.0) * 1.5

    # Increase for dense regions
    if total_density > 1.0:
        adjusted_curvature *= 1 + min(total_density / 10, 2.0)

        collision_threshold = edge_length * 0.15 * margin
        if best["min_dist"] < collision_threshold:
            adjusted_curvature *= (
                1 + (collision_threshold - best["min_dist"]) / collision_threshold
            )

        adjusted_curvature = min(adjusted_curvature, base_curvature * 15.0)

    return adjusted_curvature, best_direction


def _draw_arc(
    ax, x1, y1, x2, y2, color, linewidth, alpha, zorder, curvature, direction
):
    """Draw curved edge using quadratic bezier"""
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 1e-6:
        return ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
            solid_capstyle="round",
        )[0]

    # Control point
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    perp_x, perp_y = -dy / dist, dx / dist
    offset = dist * curvature * direction
    ctrl_x, ctrl_y = mx + perp_x * offset, my + perp_y * offset

    # Bezier curve
    verts = [(x1, y1), (ctrl_x, ctrl_y), (x2, y2)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)

    patch = patches.PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
        capstyle="round",
    )
    ax.add_patch(patch)
    return patch


def add_complex_network_legend(
    ax,
    loc="lower left",
    fontsize=11,
    negative_color="#9f8db8",
    positive_color="#eab679",
    alpha=0.6,
):
    """Add legend for interaction edges"""
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=positive_color,
            linewidth=3,
            alpha=alpha,
            label="Positive Interaction",
        ),
        Line2D(
            [0],
            [0],
            color=negative_color,
            linewidth=3,
            alpha=alpha,
            label="Negative Interaction",
        ),
    ]

    return ax.legend(
        handles=legend_elements,
        loc=loc,
        fontsize=fontsize,
        frameon=False,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.8,
    )

import pandas as pd
import numpy as np
from sqrt_pcp import sqrt_pcp
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, dendrogram
from scipy.spatial.distance import squareform
from scipy import optimize


# ============================================================================
# Utility Functions
# ============================================================================


def nan_cosine_similarity(a, b):
    """Compute NaN-aware cosine similarity between two vectors"""
    mask = ~np.isnan(a) & ~np.isnan(b)
    a_masked = a[mask]
    b_masked = b[mask]
    if len(a_masked) == 0:
        return np.nan
    norm_a = np.linalg.norm(a_masked)
    norm_b = np.linalg.norm(b_masked)
    if norm_a == 0 or norm_b == 0:
        return np.nan
    return np.dot(a_masked, b_masked) / (norm_a * norm_b)


def nan_cosine_similarity_matrix(data):
    """Compute NaN-aware cosine similarity matrix"""
    if isinstance(data, pd.DataFrame):
        data = data.values
    n_samples = data.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                similarity_matrix[i, j] = nan_cosine_similarity(data[i], data[j])
    return similarity_matrix


def make_symmetric(df, keep_single_values=False, mask_diagonal=False):
    """Make a matrix symmetric by averaging with its transpose"""
    all_labels = df.index.union(df.columns)
    df_reindexed = df.reindex(index=all_labels, columns=all_labels)
    df_transposed = df_reindexed.T

    mask_both = df_reindexed.isna() & df_transposed.isna()
    mask_either = df_reindexed.isna() | df_transposed.isna()

    symmetric_avg = (df_reindexed.fillna(0) + df_transposed.fillna(0)) / 2

    # Handle single-direction values
    if keep_single_values:
        result = np.where(df_reindexed.isna(), df_transposed, df_reindexed)
        result = np.where(df_transposed.isna(), df_reindexed, result)
    else:
        result = np.where(mask_either, np.nan, symmetric_avg)

    result = np.where(~mask_either, symmetric_avg, result)
    result_df = pd.DataFrame(result, index=all_labels, columns=all_labels)

    if mask_diagonal:
        np.fill_diagonal(result_df.values, np.nan)

    # Restore original NaN pairs
    result_df[mask_both] = np.nan

    return result_df


def clustering_cosine(data, optimal_leaf=True):
    """Perform hierarchical clustering using cosine similarity"""
    similarity_matrix = nan_cosine_similarity_matrix(data)
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.nan_to_num(distance_matrix, nan=1.0)

    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method="average")
    if optimal_leaf:
        Z = optimal_leaf_ordering(Z, condensed_dist)

    return Z


def extract_single_effects(single_pairs, kind="guide"):
    """Extract single guide effects from paired data"""
    single_df = single_pairs.to_frame("val").reset_index()
    control_sgRNAs = single_df["control"].unique()
    single_effects = single_df.groupby(kind)["val"].mean()
    return single_effects, control_sgRNAs


# ============================================================================
# Genetic Interaction Calculator
# ============================================================================


class GeneticInteractionCalculator:
    """Handles genetic interaction calculations for dual-guide screens"""

    def __init__(self, dual_guide_portal):
        self.portal = dual_guide_portal

    def calculate(self, single_pairs, double_effects, kind="per_gene"):
        """
        Calculate genetic interactions

        Args:
            single_pairs: Single guide effect pairs
            double_effects: Double guide effect matrix
            kind: Method - 'per_gene' or 'joint'

        Returns:
            DataFrame of genetic interaction scores
        """
        methods = {"per_gene": self._calculate_per_gene, "joint": self._calculate_joint}

        if kind not in methods:
            raise ValueError(
                f"kind must be one of {list(methods.keys())}, got '{kind}'"
            )

        return methods[kind](single_pairs, double_effects)

    def _calculate_joint(
        self, single_pairs, double_effects, epsilon=1.35, alpha=0.0001
    ):
        """Calculate GI using joint quadratic model"""
        single_effects, _ = extract_single_effects(single_pairs)
        genes = single_effects.index

        # Prepare upper triangle data
        triu_indices = np.triu_indices(len(genes))
        gene1_idx, gene2_idx = triu_indices
        x1_vals = single_effects.iloc[gene1_idx].values
        x2_vals = single_effects.iloc[gene2_idx].values
        y_vals = double_effects.values[gene1_idx, gene2_idx]

        # Filter valid data
        valid_mask = ~np.isnan(y_vals)
        X1, X2, Y = x1_vals[valid_mask], x2_vals[valid_mask], y_vals[valid_mask]

        # Fit model
        X_features = np.column_stack([X1**2 + X2**2, X1 + X2, X1 * X2])
        model = linear_model.HuberRegressor(
            epsilon=epsilon, alpha=alpha, fit_intercept=False
        )
        model.fit(X_features, Y)
        a_joint, b_joint, e_joint = model.coef_

        # Predict expected values
        single_vals = single_effects.values
        X1_grid, X2_grid = np.meshgrid(single_vals, single_vals, indexing="ij")
        expected_vals = (
            a_joint * (X1_grid**2 + X2_grid**2)
            + b_joint * (X1_grid + X2_grid)
            + e_joint * X1_grid * X2_grid
        )
        expected = pd.DataFrame(expected_vals, index=genes, columns=genes)

        return double_effects - expected

    def _calculate_per_gene(
        self, single_pairs, double_effects, epsilon=1.35, alpha=0.0001
    ):
        """Calculate GI using per-gene quadratic models with monotonicity constraints"""
        from scipy.optimize import minimize

        single_effects, _ = extract_single_effects(single_pairs)
        genes = single_effects.index
        delta = pd.DataFrame(np.nan, index=genes, columns=genes)

        for query_idx, query in enumerate(genes):
            y_raw = double_effects.loc[query].values
            valid_mask = ~np.isnan(y_raw)

            if np.sum(valid_mask) > 5:
                X_raw = single_effects.values[valid_mask]
                y_valid = y_raw[valid_mask]
                query_phenotype = single_effects.iloc[query_idx]

                y_adjusted = y_valid - query_phenotype
                X = X_raw
                X_min, X_max = np.min(X), np.max(X)

                # Optimize with monotonicity constraint
                result = minimize(
                    lambda params: self._huber_objective(
                        params, X, y_adjusted, alpha, epsilon
                    ),
                    np.array([1.0, 0.0]),  # [b, a]
                    method="SLSQP",
                    constraints=[
                        {
                            "type": "ineq",
                            "fun": lambda params: self._monotonicity_constraint(
                                params, X_min, X_max
                            ),
                        }
                    ],
                    options={"maxiter": 1000, "ftol": 1e-9},
                )

                if result.success:
                    b_opt, a_opt = result.x
                else:
                    print("falling back to unconstrained")
                    # Fallback to unconstrained
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    X_features = poly.fit_transform(X.reshape(-1, 1))
                    model = linear_model.HuberRegressor(
                        epsilon=epsilon, alpha=alpha, fit_intercept=False
                    )
                    model.fit(X_features, y_adjusted)
                    b_opt, a_opt = model.coef_[0], model.coef_[1]

                # Compute GI scores
                all_X = single_effects.values
                expected_phenotypes = a_opt * all_X**2 + b_opt * all_X + query_phenotype
                gi_scores = y_raw - expected_phenotypes
                delta.iloc[query_idx] = gi_scores

        return self._symmetrize_gi_matrix(delta)

    @staticmethod
    def _huber_loss(residuals, delta):
        """Huber loss function"""
        abs_residuals = np.abs(residuals)
        quadratic = 0.5 * residuals**2
        linear = delta * (abs_residuals - 0.5 * delta)
        return np.where(abs_residuals <= delta, quadratic, linear).sum()

    def _huber_objective(self, params, X, y, alpha_reg, epsilon):
        """Objective function: Huber loss + L2 regularization"""
        b, a = params
        y_pred = a * X**2 + b * X
        residuals = y - y_pred
        loss = self._huber_loss(residuals, epsilon)
        regularization = alpha_reg * (a**2 + b**2)
        return loss + regularization

    @staticmethod
    def _monotonicity_constraint(params, X_min, X_max):
        """Constraint to ensure monotonically increasing function"""
        b, a = params

        if abs(a) < 1e-10:  # Linear case
            return b

        # For quadratic: dy/dx = 2a*x + b
        if a > 0:
            min_deriv = 2 * a * X_min + b
        else:
            min_deriv = 2 * a * X_max + b

        return min_deriv

    @staticmethod
    def _symmetrize_gi_matrix(delta):
        """Symmetrize genetic interaction matrix"""
        delta_T = delta.T
        valid_mask = ~delta.isna()
        valid_mask_T = ~delta_T.isna()

        genes = delta.index
        sym_gi = pd.DataFrame(np.nan, index=genes, columns=genes)

        # Both directions valid: average
        both_valid = valid_mask & valid_mask_T
        sym_gi[both_valid] = (delta[both_valid] + delta_T[both_valid]) / 2

        # Only one direction valid: use that value
        only_orig = valid_mask & ~valid_mask_T
        only_trans = ~valid_mask & valid_mask_T
        sym_gi[only_orig] = delta[only_orig]
        sym_gi[only_trans] = delta_T[only_trans]

        # Diagonal
        np.fill_diagonal(sym_gi.values, np.diag(delta.values))

        return sym_gi


# ============================================================================
# Base PORTAL Class
# ============================================================================


class BasePORTAL:
    """Base class for PORTAL analysis with shared functionality"""

    def __init__(
        self,
        umi_df,
        input_rep_data=None,
        guide_columns=None,
        barcode_columns=["barcode_mapped"],
        transcript_columns=["UMI_identity", "UMI_reporter"],
        sample_columns=None,
    ):
        self.guide_columns = guide_columns
        self.barcode_columns = barcode_columns
        self.transcript_columns = transcript_columns
        self.sample_columns = sample_columns

        self.rep_data = self._load_rep_data(input_rep_data) if input_rep_data else None
        if self.rep_data is not None:
            self.rep_col = self.rep_data.columns[-1]

        self.umi_df = self._merge_rep_data(umi_df)
        self._validate_data()

        self.filtered_umi_df = self.umi_df.copy()
        self.set_lineage_columns()

        self._calculate_measurement_counts()

    @classmethod
    def from_csv(cls, filepath, input_rep_data=None, **kwargs):
        """Create instance from CSV file"""
        df = pd.read_csv(filepath)
        return cls(df, input_rep_data, **kwargs)

    def _load_rep_data(self, input_rep_data):
        """Load representation data from file or DataFrame"""
        if isinstance(input_rep_data, str):
            return pd.read_csv(input_rep_data)
        elif isinstance(input_rep_data, pd.DataFrame):
            return input_rep_data
        raise TypeError("Representation data must be path string or DataFrame")

    def _merge_rep_data(self, umi_df):
        """Merge UMI data with representation data"""
        if self.rep_data is not None:
            return umi_df.merge(
                self.rep_data,
                left_on=self.guide_columns,
                right_on=self.guide_columns,
                how="left",
            ).fillna(0)
        return umi_df

    def _validate_data(self):
        """Validate input data"""
        if not isinstance(self.umi_df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame for UMI counts")
        if self.umi_df.empty:
            raise ValueError("UMI DataFrame cannot be empty")
        if self.rep_data is not None and not isinstance(self.rep_data, pd.DataFrame):
            raise TypeError("Representation data must be a pandas DataFrame")

    def get_summary(self):
        """Get summary statistics of UMI data"""
        return self.umi_df.describe()

    def set_lineage_columns(self):
        """Set columns that define unique lineages"""
        if self.sample_columns and "rep" in self.sample_columns:
            self.lineage_columns = self.guide_columns + self.barcode_columns + ["rep"]
        else:
            self.lineage_columns = self.guide_columns + self.barcode_columns

    def generate_guide_ids(self, separator="_", gene_position=0):
        """
        Generate clean guide IDs (GENE_1, GENE_2, etc.) from guide sequences.

        Creates mappings from full guide IDs (GENE_PROTOSPACER) to simplified
        guide IDs (GENE_1, GENE_2, etc.). Genes with only one guide keep just
        the gene name.

        Args:
            separator: Character separating gene from protospacer (default: "_")
            gene_position: Position of gene name after splitting (default: 0)

        Attributes created:
            guide_id_df: DataFrame with columns ['guide', 'gene', 'id', 'guide_id']
            guide_id_dict: Dict mapping guide -> guide_id
            guide_id_reverse_dict: Dict mapping guide_id -> guide
        """
        # Get unique guides from the first guide column
        unique_guides = pd.Series(self.filtered_umi_df[self.guide_columns[0]].unique())

        guide_id_df = pd.DataFrame(unique_guides, columns=["guide"])

        guide_id_df["gene"] = guide_id_df.guide.str.split(separator).str[gene_position]

        # Number guides per gene
        guide_id_df["id"] = guide_id_df.groupby("gene").cumcount() + 1

        # Create guide_id as GENE_NUMBER
        guide_id_df["guide_id"] = (
            guide_id_df["gene"] + separator + guide_id_df["id"].astype(str)
        )

        # For genes with only one guide, use just the gene name
        single_guide_genes = guide_id_df.gene.value_counts()[
            guide_id_df.gene.value_counts() == 1
        ].index
        guide_id_df.loc[guide_id_df.gene.isin(single_guide_genes), "guide_id"] = (
            guide_id_df.loc[guide_id_df.gene.isin(single_guide_genes), "gene"]
        )

        # Create mappings
        self.guide_id_df = guide_id_df
        self.guide_id_dict = guide_id_df.set_index("guide").guide_id.to_dict()
        self.guide_id_reverse_dict = guide_id_df.set_index("guide_id").guide.to_dict()

        n_genes = guide_id_df.gene.nunique()
        n_guides = len(guide_id_df)

        print(f"Generated guide IDs: {n_genes} genes with {n_guides} guides")

        return guide_id_df

    def get_guide_from_guide_id(self, guide_id):
        """
        Get full guide sequence from guide_id.

        Args:
            guide_id: Simplified guide ID (e.g., 'GCN5_1' or 'URA3')

        Returns:
            Full guide ID (e.g., 'GCN5_AGCTTAGC')
        """
        if not hasattr(self, "guide_id_reverse_dict"):
            raise AttributeError(
                "guide_id_reverse_dict not found. Run generate_guide_ids() first."
            )

        return self.guide_id_reverse_dict.get(guide_id, None)

    def _calculate_measurement_counts(self):
        """Calculate measurement counts per guide - must be implemented by subclass"""
        raise NotImplementedError(
            "Subclass must implement _calculate_measurement_counts()"
        )

    def filter_overrepresented_lineages(self, min_thres=0.05):
        """Filter out overrepresented lineages using mean + 3*std threshold"""
        lineage_df = self.filtered_umi_df.groupby(self.lineage_columns)[
            self.transcript_columns
        ].sum()

        # Calculate representation for each transcript
        thresholds = []
        for i, transcript in enumerate(self.transcript_columns):
            sums = lineage_df.groupby(self.guide_columns)[transcript].transform("sum")
            lineage_df[f"lineage_repr{i+1}"] = lineage_df[transcript] / sums

            threshold = (
                lineage_df[f"lineage_repr{i+1}"].fillna(0).mean()
                + 3 * lineage_df[f"lineage_repr{i+1}"].fillna(0).std()
            )
            if min_thres is not None:
                threshold = max(threshold, min_thres)
            thresholds.append(threshold)

        # Apply filter
        query_str = " and ".join(
            [
                f"lineage_repr{i+1} < {thresholds[i]}"
                for i in range(len(self.transcript_columns))
            ]
        )
        select_lineages = lineage_df.query(query_str).index

        print(
            f"Applied thresholds: "
            + ", ".join(
                [f"repr{i+1}={thresholds[i]:.4f}" for i in range(len(thresholds))]
            )
        )

        self.filtered_umi_df = (
            self.filtered_umi_df.set_index(self.lineage_columns)
            .loc[select_lineages]
            .reset_index()
        )

    def log_transform_UMIs(self, pseudocount=10):
        """Apply log2 transformation to UMI counts"""
        self.filtered_umi_df["total_UMI"] = (
            self.filtered_umi_df[self.transcript_columns[0]]
            + self.filtered_umi_df[self.transcript_columns[1]]
        )
        self.filtered_umi_df["log2_identity"] = np.log2(
            self.filtered_umi_df[self.transcript_columns[0]] + pseudocount
        )
        self.filtered_umi_df["log2_reporter"] = np.log2(
            self.filtered_umi_df[self.transcript_columns[1]] + pseudocount
        )
        self.filtered_umi_df["log2_UMI"] = np.log2(
            self.filtered_umi_df["total_UMI"] + pseudocount
        )

    def format_sample_columns(self):
        """Convert sample columns to dummy variables"""
        if not self.sample_columns:
            raise ValueError("Sample columns not provided")

        self.filtered_umi_df = pd.concat(
            [self.filtered_umi_df]
            + [
                pd.get_dummies(self.filtered_umi_df[c], prefix=c)
                for c in self.sample_columns
            ],
            axis=1,
        )

    def _get_categorical_columns(self):
        """Extract categorical dummy columns from formatted sample columns"""
        if not self.sample_columns:
            return []

        all_columns = self.filtered_umi_df.columns

        if "rep" in self.sample_columns:
            rep_cols = all_columns[all_columns.str.startswith("rep_")].tolist()
            other_cols = all_columns[
                all_columns.str.startswith(
                    tuple([f"{c}_" for c in np.setdiff1d(self.sample_columns, ["rep"])])
                )
            ].tolist()
            return rep_cols + other_cols
        else:
            return all_columns[
                all_columns.str.startswith(
                    tuple([f"{c}_" for c in self.sample_columns])
                )
            ].tolist()

    def regress_identity_poisson(
        self, resid_col="identity_resid", rescale=True, pseudocount=1
    ):
        """Regress identity transcript using Poisson model"""
        if not self.sample_columns:
            raise ValueError(
                "Can only regress identity when sample columns are provided"
            )

        col = self.transcript_columns[0]
        control = self.filtered_umi_df.query("control")
        cat_columns = self._get_categorical_columns()

        X = control[cat_columns]
        y = control[col]

        regr = linear_model.PoissonRegressor()
        regr.fit(X, y)

        samples = self.filtered_umi_df[cat_columns]
        identity = self.filtered_umi_df[col]
        y_pred = regr.predict(samples)

        self.filtered_umi_df[resid_col] = identity - y_pred

        # Rescale residuals to original UMI range
        if rescale:
            from sklearn.preprocessing import minmax_scale

            umi_min = self.filtered_umi_df[col].min()
            umi_max = self.filtered_umi_df[col].max()
            self.filtered_umi_df[f"{resid_col}_rescaled"] = minmax_scale(
                self.filtered_umi_df[resid_col], feature_range=(umi_min, umi_max)
            )

            # Also create log2-transformed rescaled version
            self.filtered_umi_df["log2_identity_rescaled"] = np.log2(
                self.filtered_umi_df[f"{resid_col}_rescaled"] + pseudocount
            )

    def regress_reporter(
        self, x_var="log2_identity", polynomial_degree=1, control_only=True
    ):
        """Regress reporter against identity with optional polynomial features"""
        fit_data = (
            self.filtered_umi_df.query("control")
            if control_only
            else self.filtered_umi_df
        )

        # Prepare continuous features
        x_continuous = fit_data[[x_var]]
        identity_continuous = self.filtered_umi_df[[x_var]]

        # Prepare categorical features
        if self.sample_columns:
            cat_columns = self._get_categorical_columns()
            x_categorical = fit_data[cat_columns]
            identity_categorical = self.filtered_umi_df[cat_columns]
        else:
            x_categorical = None
            identity_categorical = None

        # Add polynomial features
        if polynomial_degree > 1:
            poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
            x_poly = poly.fit_transform(x_continuous)
            identity_poly = poly.transform(identity_continuous)
        else:
            x_poly = x_continuous
            identity_poly = identity_continuous

        # Combine features
        if x_categorical is not None:
            X = np.hstack([x_poly, x_categorical.values])
            identity = np.hstack([identity_poly, identity_categorical.values])
        else:
            X = x_poly
            identity = identity_poly

        y = fit_data.log2_reporter.values.reshape(-1, 1)
        reporter = self.filtered_umi_df.log2_reporter.values.reshape(-1, 1)

        regr = linear_model.LinearRegression(fit_intercept=True)
        regr.fit(X, y)
        y_pred = regr.predict(identity)

        self.filtered_umi_df["reporter_resid"] = reporter - y_pred

        print(f"Coefficients: {regr.coef_}")
        print(f"Intercept: {regr.intercept_}")
        print(f"R^2 score: {regr.score(X, y)}")

    def calculate_summary_df(
        self,
        identity_col="identity_resid_rescaled",
        reporter_col="reporter_resid",
        pseudocount=1e-7,
    ):
        """
        Calculate summary statistics for both identity and reporter at guide level.
        Stores results in self.summary_df with both phenotypes.

        Works with or without representation data.
        """
        # Prepare identity column
        if "rescaled" in identity_col:
            self.filtered_umi_df["identity_scaled"] = self.filtered_umi_df[identity_col]
        else:
            from sklearn.preprocessing import minmax_scale

            if min(self.filtered_umi_df[identity_col]) < 0:
                self.filtered_umi_df["identity_scaled"] = minmax_scale(
                    self.filtered_umi_df[identity_col]
                )
            else:
                self.filtered_umi_df["identity_scaled"] = self.filtered_umi_df[
                    identity_col
                ]

        # Aggregate identity by guide
        if self.rep_data is not None:
            # With representation data - calculate depletion
            identity_agg = (
                self.filtered_umi_df.groupby(
                    self.guide_columns + [self.rep_col, "control"]
                )[["identity_scaled"]]
                .sum()
                .reset_index()
            )

            # Calculate identity depletion
            identity_agg["identity_screen_rep"] = (
                identity_agg.identity_scaled / identity_agg.identity_scaled.sum()
            )
            identity_agg["identity_log2fc"] = np.log2(
                identity_agg["identity_screen_rep"] + pseudocount
            ) - np.log2(identity_agg[self.rep_col] + pseudocount)

            # Z-score for identity (store raw value too)
            identity_metric = "identity_log2fc"
            control_mean_id = identity_agg.query("control")[identity_metric].mean()
            control_std_id = identity_agg.query("control")[identity_metric].std()
            identity_agg["identity_z"] = (
                identity_agg[identity_metric] - control_mean_id
            ) / control_std_id
        else:
            # Without representation data - use unnormalized identity
            identity_agg = (
                self.filtered_umi_df.groupby(self.guide_columns + ["control"])[
                    ["identity_scaled"]
                ]
                .mean()
                .reset_index()
            )

            # Z-score for unnormalized identity (keep raw value as identity_mean)
            identity_agg = identity_agg.rename(
                columns={"identity_scaled": "identity_mean"}
            )
            control_mean_id = identity_agg.query("control")["identity_mean"].mean()
            control_std_id = identity_agg.query("control")["identity_mean"].std()
            identity_agg["identity_z"] = (
                identity_agg["identity_mean"] - control_mean_id
            ) / control_std_id

        # Aggregate reporter by guide (keep raw mean)
        reporter_agg = (
            self.filtered_umi_df.groupby(self.guide_columns + ["control"])[
                [reporter_col]
            ]
            .mean()
            .reset_index()
            .rename(columns={reporter_col: "reporter_mean"})
        )

        # Z-score for reporter
        control_mean_rep = reporter_agg.query("control")["reporter_mean"].mean()
        control_std_rep = reporter_agg.query("control")["reporter_mean"].std()
        reporter_agg["reporter_z"] = (
            reporter_agg["reporter_mean"] - control_mean_rep
        ) / control_std_rep

        # Merge into single summary
        self.summary_df = identity_agg.merge(
            reporter_agg[self.guide_columns + ["reporter_mean", "reporter_z"]],
            on=self.guide_columns,
            how="left",
        )

        return self.summary_df


# ============================================================================
# Single-Guide PORTAL
# ============================================================================


class PORTAL(BasePORTAL):
    """Single-guide PORTAL analysis"""

    def __init__(
        self,
        umi_df,
        input_rep_data=None,
        guide_columns=[
            "protospacer_mapped"
        ],  # Columns that exist in umi_df for merging
        rep_guide_columns=["guide_identity"],  # Columns to use for analysis after merge
        barcode_columns=["barcode_mapped"],
        transcript_columns=["UMI_identity", "UMI_reporter"],
        sample_columns=None,
        auto_generate_guide_ids=True,
    ):
        self._merge_guide_columns = guide_columns  # For merging (exist in umi_df)
        self._rep_guide_columns = rep_guide_columns  # For _load_rep_data

        # Determine which columns to use for analysis (after merge)
        if rep_guide_columns is not None:
            final_guide_columns = rep_guide_columns
        else:
            final_guide_columns = guide_columns

        # Call parent - it will use final_guide_columns for analysis
        # But we override _merge_rep_data to use _merge_guide_columns
        super().__init__(
            umi_df,
            input_rep_data,
            guide_columns=final_guide_columns,
            barcode_columns=barcode_columns,
            transcript_columns=transcript_columns,
            sample_columns=sample_columns,
        )

        if auto_generate_guide_ids:
            self.generate_guide_ids()

    def _load_rep_data(self, input_rep_data):
        """Load representation data and extract protospacer if needed"""
        if isinstance(input_rep_data, str):
            rep_data = pd.read_csv(input_rep_data)
            # Extract protospacer from guide_identity if present
            if (
                hasattr(self, "_rep_guide_columns")
                and "guide_identity" in self._rep_guide_columns
            ):
                protospacer_mapped = rep_data["guide_identity"].str.split("_").str[-1]
                insertion_index = len(rep_data.columns) - 1
                rep_data.insert(
                    loc=insertion_index,
                    column="protospacer_mapped",
                    value=protospacer_mapped,
                )
            return rep_data
        elif isinstance(input_rep_data, pd.DataFrame):
            return input_rep_data
        raise TypeError("Representation data must be path string or DataFrame")

    def _merge_rep_data(self, umi_df):
        """Merge using _merge_guide_columns"""
        if self.rep_data is not None:
            # Use columns that exist in umi_df for merging
            return umi_df.merge(
                self.rep_data,
                left_on=self._merge_guide_columns,
                right_on=self._merge_guide_columns,
                how="left",
            ).fillna(0)
        return umi_df

    def _calculate_measurement_counts(self):
        """Calculate measurement counts per guide"""
        self.measurement_counts = (
            self.filtered_umi_df[self.lineage_columns]
            .drop_duplicates()[self.guide_columns]
            .value_counts()
        )

    def set_control(self, control_prefix="non"):
        """Mark control guides"""
        self.filtered_umi_df["control"] = self.filtered_umi_df[
            self.guide_columns[0]
        ].str.startswith(control_prefix)

    def rename_guide_ids_in_series(self, series, inplace=False):
        """
        Rename guide IDs in a Series index using guide_id_dict.

        Args:
            series: Series with guide IDs as index
            inplace: If True, modify series in place; else return copy

        Returns:
            Series with renamed guide IDs (if not inplace)
        """
        if not hasattr(self, "guide_id_dict"):
            raise AttributeError(
                "guide_id_dict not found. Run generate_guide_ids() first."
            )

        if inplace:
            series.rename(index=self.guide_id_dict, inplace=True)
            return None
        else:
            return series.rename(index=self.guide_id_dict)

    def run_standard_analysis(
        self,
        min_thres=0.05,
        control_prefix="non",
        pseudocount=10,
        polynomial_degree=1,
        control_only=True,
        rescale_identity=True,
    ):
        """
        Run standard single-guide screen analysis pipeline.

        Args:
            min_thres: Minimum threshold for overrepresented lineages (default: 0.05)
            control_prefix: Prefix for control guides (default: "non")
            pseudocount: Pseudocount for log transformation (default: 10)
            polynomial_degree: Degree for reporter regression (default: 1)
            control_only: Fit reporter regression on controls only (default: True)
            rescale_identity: Rescale identity residuals to UMI range (default: True)
        """
        print("Step 1/7: Filtering overrepresented lineages...")
        self.filter_overrepresented_lineages(min_thres=min_thres)

        print("Step 2/7: Setting controls...")
        self.set_control(control_prefix=control_prefix)

        print("Step 3/7: Log transforming UMIs...")
        self.log_transform_UMIs(pseudocount=pseudocount)

        print("Step 4/7: Formatting sample columns...")
        self.format_sample_columns()

        print("Step 5/7: Regressing reporter...")
        self.regress_reporter(
            polynomial_degree=polynomial_degree, control_only=control_only
        )

        print("Step 6/7: Regressing identity (Poisson)...")
        self.regress_identity_poisson(rescale=rescale_identity)

        print("Step 7/7: Calculating summary statistics...")
        identity_col = (
            "identity_resid_rescaled" if rescale_identity else "identity_resid"
        )
        self.calculate_summary_df(identity_col=identity_col)

        print("\nAnalysis complete!")
        print(f"Summary: {len(self.summary_df)} guides")
        print(f"  Controls: {self.summary_df['control'].sum()}")
        print(f"  Targeting: {(~self.summary_df['control']).sum()}")


# ============================================================================
# Dual-Guide PORTAL
# ============================================================================


class DualGuidePORTAL(BasePORTAL):
    """Dual-guide PORTAL analysis with genetic interaction capabilities"""

    def __init__(
        self,
        umi_df,
        input_rep_data=None,
        guide_columns=["p1_identity", "p2_identity"],
        barcode_columns=["barcode_mapped"],
        transcript_columns=["UMI_identity", "UMI_reporter"],
        sample_columns=None,
        auto_generate_guide_ids=True,
    ):
        super().__init__(
            umi_df,
            input_rep_data,
            guide_columns=guide_columns,
            barcode_columns=barcode_columns,
            transcript_columns=transcript_columns,
            sample_columns=sample_columns,
        )

        self.gi_calculator = GeneticInteractionCalculator(self)

        if auto_generate_guide_ids:
            self.generate_guide_ids()

    def _calculate_measurement_counts(self):
        """Calculate measurement counts per guide pair (returns unstacked matrix)"""
        self.measurement_counts = (
            self.filtered_umi_df[self.lineage_columns]
            .drop_duplicates()[self.guide_columns]
            .value_counts()
            .unstack()
        )

    def set_control(self, control_prefix="non"):
        """Mark control guide pairs (both guides must be controls)"""
        self.filtered_umi_df["control"] = self.filtered_umi_df[
            self.guide_columns[0]
        ].str.startswith(control_prefix) & self.filtered_umi_df[
            self.guide_columns[1]
        ].str.startswith(
            control_prefix
        )

    def get_activity_matrix(self, transcript="reporter"):
        """Get z-scored activity matrix for reporter or identity"""
        if not hasattr(self, "summary_df"):
            raise ValueError("Run calculate_summary_df() first")

        if transcript == "reporter":
            metric = "reporter_z"
        elif transcript == "identity":
            metric = "identity_z"
        else:
            raise ValueError("transcript must be 'reporter' or 'identity'")

        return self.summary_df.set_index(self.guide_columns)[metric].unstack()

    def calculate_activity_matrices(self):
        """Calculate both reporter and identity activity matrices"""
        self.reporter_activity = self.get_activity_matrix("reporter")
        self.identity_activity = self.get_activity_matrix("identity")

    def perform_pcp(self, mu=None, verbose=True):
        """Perform PCP decomposition using sqrt-pcp-base method"""
        self.reporter_pcp_results = sqrt_pcp(
            self.reporter_activity.values,
            mu=mu,
            max_iter=5000,
            abs_tol=1e-8,
            rel_tol=1e-8,
            verbose=verbose,
        )

        self.identity_pcp_results = sqrt_pcp(
            self.identity_activity.values,
            mu=mu,
            max_iter=5000,
            abs_tol=1e-8,
            rel_tol=1e-8,
            verbose=verbose,
        )

    def _get_singleton_pairs(self, stacked_matrix, control_prefix):
        """Extract singleton guide pairs (one guide is control)"""
        control_guide = stacked_matrix[
            stacked_matrix[self.guide_columns[0]].str.startswith(control_prefix)
            & ~stacked_matrix[self.guide_columns[1]].str.startswith(control_prefix)
        ]

        guide_control = stacked_matrix[
            stacked_matrix[self.guide_columns[1]].str.startswith(control_prefix)
            & ~stacked_matrix[self.guide_columns[0]].str.startswith(control_prefix)
        ]

        return control_guide, guide_control

    def _merge_singleton_pairs(self, control_guide, guide_control):
        """Merge and average bidirectional singleton measurements"""
        singleton_pairs = control_guide.merge(
            guide_control,
            left_on=self.guide_columns[::-1],
            right_on=self.guide_columns,
            how="outer",
            suffixes=("_cg", "_gc"),
        )

        singleton_pairs.loc[
            singleton_pairs[[f"{g}_cg" for g in self.guide_columns]].isna().all(axis=1),
            [f"{g}_cg" for g in self.guide_columns],
        ] = singleton_pairs.loc[
            singleton_pairs[[f"{g}_cg" for g in self.guide_columns]].isna().all(axis=1),
            [f"{g}_gc" for g in self.guide_columns[::-1]],
        ].values

        singleton_pairs.loc[
            singleton_pairs[[f"{g}_gc" for g in self.guide_columns[::-1]]]
            .isna()
            .all(axis=1),
            [f"{g}_gc" for g in self.guide_columns[::-1]],
        ] = singleton_pairs.loc[
            singleton_pairs[[f"{g}_gc" for g in self.guide_columns[::-1]]]
            .isna()
            .all(axis=1),
            [f"{g}_cg" for g in self.guide_columns],
        ].values

        singleton_pairs = singleton_pairs.set_index(
            [f"{g}_cg" for g in self.guide_columns]
        )[["val_cg", "val_gc"]]
        singleton_pairs.index.names = ["control", "guide"]
        avg_singleton_pairs = singleton_pairs.mean(axis=1)

        return singleton_pairs, avg_singleton_pairs

    def _get_guide_pairs(
        self, stacked_matrix, control_prefix, mask_diagonal, keep_single_values
    ):
        """Extract and process guide-guide pairs"""
        guide_matrix = (
            stacked_matrix[
                ~stacked_matrix[self.guide_columns[0]].str.startswith(control_prefix)
                & ~stacked_matrix[self.guide_columns[1]].str.startswith(control_prefix)
            ]
            .set_index(self.guide_columns)
            .val.unstack()
        )
        avg_guide_matrix = make_symmetric(
            guide_matrix,
            keep_single_values=keep_single_values,
            mask_diagonal=mask_diagonal,
        )

        return guide_matrix, avg_guide_matrix

    def get_guide_activities(
        self, matrix, control_prefix="non", mask_diagonal=True, keep_single_values=False
    ):
        """Extract singleton and pairwise guide activities from matrix"""
        stacked_matrix = matrix.stack().to_frame("val").reset_index()

        control_guide, guide_control = self._get_singleton_pairs(
            stacked_matrix, control_prefix
        )
        singleton_pairs, avg_singleton_pairs = self._merge_singleton_pairs(
            control_guide, guide_control
        )

        guide_matrix, avg_guide_matrix = self._get_guide_pairs(
            stacked_matrix, control_prefix, mask_diagonal, keep_single_values
        )

        return (singleton_pairs, avg_singleton_pairs, guide_matrix, avg_guide_matrix)

    def save_guide_activities(
        self,
        control_prefix="non",
        mask_diagonal=True,
        keep_single_values=False,
        mask_missing=True,
    ):
        """Extract and save guide activities from original and PCP-decomposed matrices"""
        # Original activities
        (
            self.reporter_singleton_pairs,
            self.reporter_avg_singleton_pairs,
            self.reporter_guide_matrix,
            self.reporter_avg_guide_matrix,
        ) = self.get_guide_activities(
            self.reporter_activity,
            control_prefix=control_prefix,
            mask_diagonal=mask_diagonal,
            keep_single_values=keep_single_values,
        )

        (
            self.identity_singleton_pairs,
            self.identity_avg_singleton_pairs,
            self.identity_guide_matrix,
            self.identity_avg_guide_matrix,
        ) = self.get_guide_activities(
            self.identity_activity,
            control_prefix=control_prefix,
            mask_diagonal=mask_diagonal,
            keep_single_values=keep_single_values,
        )

        # Compute single effects (averaged singleton pairs)
        self.reporter_single_effects = self.reporter_avg_singleton_pairs.groupby(
            "guide"
        ).mean()
        self.identity_single_effects = self.identity_avg_singleton_pairs.groupby(
            "guide"
        ).mean()

        # PCP activities
        try:
            reporter_low_rank = pd.DataFrame(
                self.reporter_pcp_results["L"],
                index=self.reporter_activity.index,
                columns=self.reporter_activity.columns,
            )
            identity_low_rank = pd.DataFrame(
                self.identity_pcp_results["L"],
                index=self.identity_activity.index,
                columns=self.identity_activity.columns,
            )

            if mask_missing:
                reporter_low_rank = reporter_low_rank[~self.reporter_activity.isna()]
                identity_low_rank = identity_low_rank[~self.identity_activity.isna()]

            (
                self.reporter_singleton_pairs_pcp,
                self.reporter_avg_singleton_pairs_pcp,
                self.reporter_guide_matrix_pcp,
                self.reporter_avg_guide_matrix_pcp,
            ) = self.get_guide_activities(
                reporter_low_rank,
                control_prefix=control_prefix,
                mask_diagonal=mask_diagonal,
                keep_single_values=keep_single_values,
            )

            (
                self.identity_singleton_pairs_pcp,
                self.identity_avg_singleton_pairs_pcp,
                self.identity_guide_matrix_pcp,
                self.identity_avg_guide_matrix_pcp,
            ) = self.get_guide_activities(
                identity_low_rank,
                control_prefix=control_prefix,
                mask_diagonal=mask_diagonal,
                keep_single_values=keep_single_values,
            )

            # Compute single effects for PCP
            self.reporter_single_effects_pcp = (
                self.reporter_avg_singleton_pairs_pcp.groupby("guide").mean()
            )
            self.identity_single_effects_pcp = (
                self.identity_avg_singleton_pairs_pcp.groupby("guide").mean()
            )

        except Exception as e:
            print("PCP activities not calculated")
            print(e)

    def get_GI(self, kind="per_gene"):
        """
        Calculate genetic interactions

        Args:
            kind: Method - 'per_gene' or 'joint'
        """
        self.reporter_GI = self.gi_calculator.calculate(
            self.reporter_avg_singleton_pairs, self.reporter_avg_guide_matrix, kind=kind
        )
        self.identity_GI = self.gi_calculator.calculate(
            self.identity_avg_singleton_pairs, self.identity_avg_guide_matrix, kind=kind
        )

        # Z-score normalize
        self.z_reporter_GI = self.reporter_GI / np.nanstd(self.reporter_GI)
        self.z_identity_GI = self.identity_GI / np.nanstd(self.identity_GI)

        # Calculate from PCP matrices if available
        try:
            self.reporter_GI_pcp = self.gi_calculator.calculate(
                self.reporter_avg_singleton_pairs_pcp,
                self.reporter_avg_guide_matrix_pcp,
                kind=kind,
            )
            self.identity_GI_pcp = self.gi_calculator.calculate(
                self.identity_avg_singleton_pairs_pcp,
                self.identity_avg_guide_matrix_pcp,
                kind=kind,
            )

            self.z_reporter_GI_pcp = self.reporter_GI_pcp / np.nanstd(
                self.reporter_GI_pcp
            )
            self.z_identity_GI_pcp = self.identity_GI_pcp / np.nanstd(
                self.identity_GI_pcp
            )
        except AttributeError:
            print("PCP interactions not calculated")

    def cluster_GI(self, optimal_leaf=True):
        """Cluster genetic interaction matrices"""
        self.reporter_Z = clustering_cosine(self.reporter_GI, optimal_leaf=optimal_leaf)
        self.reporter_order = dendrogram(self.reporter_Z, no_plot=True)["leaves"]

        self.identity_Z = clustering_cosine(self.identity_GI, optimal_leaf=optimal_leaf)
        self.identity_order = dendrogram(self.identity_Z, no_plot=True)["leaves"]

        try:
            self.reporter_Z_pcp = clustering_cosine(
                self.reporter_GI_pcp, optimal_leaf=optimal_leaf
            )
            self.reporter_order_pcp = dendrogram(self.reporter_Z_pcp, no_plot=True)[
                "leaves"
            ]

            self.identity_Z_pcp = clustering_cosine(
                self.identity_GI_pcp, optimal_leaf=optimal_leaf
            )
            self.identity_order_pcp = dendrogram(self.identity_Z_pcp, no_plot=True)[
                "leaves"
            ]
        except AttributeError:
            print("Clustering not performed for PCP interactions")

    def rename_guide_ids_in_matrix(self, matrix, inplace=False):
        """
        Rename guide IDs in a matrix (index and columns) using guide_id_dict.

        Args:
            matrix: DataFrame with guide IDs as index/columns
            inplace: If True, modify matrix in place; else return copy

        Returns:
            DataFrame with renamed guide IDs (if not inplace)
        """
        if not hasattr(self, "guide_id_dict"):
            raise AttributeError(
                "guide_id_dict not found. Run generate_guide_ids() first."
            )

        if inplace:
            matrix.rename(
                index=self.guide_id_dict, columns=self.guide_id_dict, inplace=True
            )
            return None
        else:
            return matrix.rename(index=self.guide_id_dict, columns=self.guide_id_dict)

    def run_standard_analysis(
        self,
        min_thres=0.05,
        control_prefix="non",
        pseudocount=10,
        polynomial_degree=1,
        control_only=True,
        perform_pcp=False,
        mu=None,
        mask_diagonal=False,
        keep_single_values=False,
        mask_missing=False,
        gi_kind="per_gene",
        optimal_leaf=True,
        rescale_identity=True,
    ):
        """
        Run standard dual-guide screen analysis pipeline.

        Args:
            min_thres: Minimum threshold for overrepresented lineages (default: 0.05)
            control_prefix: Prefix for control guides (default: "non")
            pseudocount: Pseudocount for log transformation (default: 10)
            polynomial_degree: Degree for reporter regression (default: 1)
            control_only: Fit reporter regression on controls only (default: True)
            perform_pcp: Whether to perform PCP decomposition (default: True)
            mu: PCP mu parameter (default: None for auto)
            mask_diagonal: Mask diagonal in guide activities (default: False)
            keep_single_values: Keep single-direction values (default: False)
            mask_missing: Mask missing values in PCP (default: False)
            gi_kind: GI calculation method - 'per_gene' or 'joint' (default: 'per_gene')
            optimal_leaf: Use optimal leaf ordering in clustering (default: True)
        """
        print("Step 1/12: Filtering overrepresented lineages...")
        self.filter_overrepresented_lineages(min_thres=min_thres)

        print("Step 2/12: Setting controls...")
        self.set_control(control_prefix=control_prefix)

        print("Step 3/12: Log transforming UMIs...")
        self.log_transform_UMIs(pseudocount=pseudocount)

        print("Step 4/12: Formatting sample columns...")
        self.format_sample_columns()

        print("Step 5/12: Regressing reporter...")
        self.regress_reporter(
            polynomial_degree=polynomial_degree, control_only=control_only
        )

        print("Step 6/12: Regressing identity (Poisson)...")
        self.regress_identity_poisson()

        print("Step 7/12: Calculating summary statistics...")
        identity_col = (
            "identity_resid_rescaled" if rescale_identity else "identity_resid"
        )
        self.calculate_summary_df(identity_col=identity_col)

        print("Step 8/12: Calculating activity matrices...")
        self.calculate_activity_matrices()

        if perform_pcp:
            print("Step 9/12: Performing PCP decomposition...")
            self.perform_pcp(mu=mu)
        else:
            print("Step 9/12: Skipping PCP decomposition...")

        print("Step 10/12: Saving guide activities...")
        self.save_guide_activities(
            mask_diagonal=mask_diagonal,
            keep_single_values=keep_single_values,
            mask_missing=mask_missing,
        )

        print("Step 11/12: Calculating genetic interactions...")
        self.get_GI(kind=gi_kind)

        print("Step 12/12: Clustering genetic interactions...")
        self.cluster_GI(optimal_leaf=optimal_leaf)

        print("\nAnalysis complete!")

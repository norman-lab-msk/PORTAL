import numpy as np
import pandas as pd
import math
from numba import jit, prange, types
from numba.typed import List
from scipy.stats import false_discovery_control
from tqdm import tqdm

# ===================================================================
# SECTION 1: CORE NUMBA HELPER FUNCTIONS
# ===================================================================


@jit(nopython=True, cache=True)
def numba_rankdata(arr):
    """A Numba-compatible equivalent of scipy.stats.rankdata(method='average')."""
    n = len(arr)
    ranks = np.empty(n, dtype=np.float64)
    sorter = np.argsort(arr)
    inv_sorter = np.empty(n, dtype=np.int64)
    inv_sorter[sorter] = np.arange(n)
    arr_sorted = arr[sorter]
    obs = np.ones(n, dtype=np.bool_)
    obs[1:] = arr_sorted[1:] != arr_sorted[:-1]
    dense = np.cumsum(obs) - 1
    count = np.bincount(dense)
    rank_vals = np.cumsum(np.concatenate((np.array([0.0]), count.astype(np.float64))))[
        :-1
    ]
    rank_vals += (count.astype(np.float64) - 1) / 2.0
    ranks = rank_vals[dense]
    return ranks[inv_sorter] + 1


@jit(nopython=True, cache=True)
def numba_unique_counts(arr):
    """A Numba-compatible function to find unique values and their counts."""
    if arr.shape[0] == 0:
        return np.empty(0, dtype=arr.dtype), np.empty(0, dtype=np.int64)
    sorted_arr = np.sort(arr)
    is_change = np.concatenate((np.array([True]), sorted_arr[1:] != sorted_arr[:-1]))
    change_indices = np.where(is_change)[0]
    unique_vals = sorted_arr[change_indices]
    counts = np.diff(np.concatenate((change_indices, np.array([len(arr)]))))
    return unique_vals, counts


@jit(nopython=True, cache=True)
def numba_mwu_asymptotic(x, y):
    """
    Final Numba MWU test implementing the standard asymptotic method,
    which always uses continuity correction.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan

    concatenated = np.concatenate((x, y))
    ranks = numba_rankdata(concatenated)

    r1 = np.sum(ranks[:n1])
    u1 = r1 - (n1 * (n1 + 1)) / 2.0
    u = min(u1, n1 * n2 - u1)
    cles = u1 / (n1 * n2)
    rbc = 2 * cles - 1
    mu_u = n1 * n2 / 2.0

    unique_ranks, rank_counts = numba_unique_counts(ranks)
    tie_correction_term = np.sum(rank_counts**3 - rank_counts)

    sigma_u_denominator = (n1 + n2) * (n1 + n2 - 1)
    if sigma_u_denominator == 0:
        return 1.0, cles, rbc

    sigma_u = np.sqrt(
        (n1 * n2 / 12.0) * (n1 + n2 + 1 - tie_correction_term / sigma_u_denominator)
    )
    if sigma_u == 0:
        return 1.0, cles, rbc

    # Standard continuity correction is always applied for the asymptotic test.
    numerator = np.abs(u - mu_u) - 0.5

    z = numerator / sigma_u
    p_val = math.erfc(z / math.sqrt(2.0))
    return p_val, cles, rbc


# ===================================================================
# SECTION 2: NUMBA PARALLEL KERNEL
# ===================================================================


@jit(nopython=True, parallel=True, cache=True)
def parallel_compare_to_controls(pert_values, control_values_list):
    """
    Compares a single perturbation's values against a list of control values in parallel.
    """
    n_controls = len(control_values_list)
    pvals = np.empty(n_controls, dtype=np.float64)
    cles_vals = np.empty(n_controls, dtype=np.float64)
    rbc_vals = np.empty(n_controls, dtype=np.float64)

    for i in prange(n_controls):
        ctrl_values = control_values_list[i]
        p, cles, rbc = numba_mwu_asymptotic(pert_values, ctrl_values)
        pvals[i] = p
        cles_vals[i] = cles
        rbc_vals[i] = rbc

    return pvals, cles_vals, rbc_vals


# ===================================================================
# SECTION 3: MAIN ANALYSIS FUNCTION
# ===================================================================


def run_mwu_analysis(pert_df, control_df, group_cols, outcomes, debug_limit=None):
    """
    Performs the full Mann-Whitney U analysis using the Numba-parallelized backend.
    """
    print("Preparing perturbation and control groups...")
    perts = {k: v[outcomes] for k, v in pert_df.groupby(group_cols)}
    control_perts = {k: v[outcomes] for k, v in control_df.groupby(group_cols)}

    n_perts = len(perts)
    pert_items = list(perts.items())

    if debug_limit is not None:
        print(
            f"--- DEBUG MODE: Limiting to {debug_limit} perturbations for testing. ---"
        )
        pert_items = pert_items[:debug_limit]
        n_perts = len(pert_items)

    print(
        f"Processing {n_perts} perturbations against {len(control_perts)} control groups..."
    )

    control_data_numba = {}
    for outcome in outcomes:
        typed_list = List.empty_list(types.float64[:])
        for ctrl_data in control_perts.values():
            clean_values = ctrl_data[outcome].dropna().values.astype(np.float64)
            typed_list.append(clean_values)
        control_data_numba[outcome] = typed_list

    all_results = {}
    for pert_key, pert_data in tqdm(
        pert_items, total=n_perts, desc="Processing perturbations"
    ):
        outcome_results = {}
        for outcome in outcomes:
            pert_values = pert_data[outcome].dropna().values.astype(np.float64)
            pvals, cles_vals, rbc_vals = parallel_compare_to_controls(
                pert_values, control_data_numba[outcome]
            )
            outcome_results[outcome] = {
                "pvals": pvals,
                "cles": cles_vals,
                "rbc": rbc_vals,
            }
        all_results[pert_key] = outcome_results

    print("Computing median statistics...")
    combined_results = []
    for pert_key, outcome_results in all_results.items():
        for outcome in outcomes:
            res = outcome_results[outcome]
            median_p = np.nanmedian(res["pvals"]) if len(res["pvals"]) > 0 else np.nan
            median_cles = np.nanmedian(res["cles"]) if len(res["cles"]) > 0 else np.nan
            median_rbc = np.nanmedian(res["rbc"]) if len(res["rbc"]) > 0 else np.nan

            combined_results.append(
                {
                    "perturbation": pert_key,
                    "outcome": outcome,
                    "rbc": median_rbc,
                    "cles": median_cles,
                    "p_value": median_p,
                }
            )

    results_df = pd.DataFrame(combined_results)
    print("Applying FDR correction...")
    results_df["p_adj"] = np.nan
    for outcome in outcomes:
        mask = results_df["outcome"] == outcome
        valid_pvals = results_df.loc[mask, "p_value"].dropna()
        if not valid_pvals.empty:
            p_adj = false_discovery_control(valid_pvals.values, method="bh")
            results_df.loc[valid_pvals.index, "p_adj"] = p_adj

    results_df["-log10_p_adj"] = -np.log10(results_df["p_adj"])

    return results_df.sort_values("p_value")

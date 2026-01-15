import numpy as np
from scipy import linalg
from tqdm import tqdm
import time


def prox_nuclear(Y, c):
    """
    Implements the proximal gradient method for the nuclear norm.

    Args:
        Y: The L matrix
        c: Amount by which to penalize singular values

    Returns:
        array: The thresholded L matrix
    """
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    S_new = np.sign(S) * np.maximum(np.abs(S) - c, 0)
    return U @ np.diag(S_new) @ Vt


def prox_l1(Y, c):
    """
    Implements the proximal gradient method for the L1 norm.

    Args:
        Y: The S matrix
        c: Amount by which to penalize values

    Returns:
        array: The thresholded S matrix
    """
    return np.sign(Y) * np.maximum(np.abs(Y) - c, 0)


def prox_fro(X, c):
    """
    Implements the proximal gradient method for the Frobenius norm.

    Args:
        X: The matrix
        c: Amount by which to penalize the norm

    Returns:
        array: The thresholded matrix
    """
    norm = linalg.norm(X, "fro")
    if norm <= c:
        return np.zeros_like(X)
    return (1 - c / norm) * X


def sqrt_pcp(
    D,
    lambda_param=None,
    mu=None,
    rho=0.2,
    verbose=False,
    max_iter=10000,
    abs_tol=1e-6,
    rel_tol=1e-6,
    accelerate=False,
    rho_aggressive=False,
    min_decrease_rate=0.001,
    decrease_check_window=100,
    force_timeout_mins=30,
):
    """
    Square Root PCP implementation.

    Args:
        D: Observed matrix with NaN for missing values
        lambda_param, mu, rho: Algorithm parameters
        verbose: Whether to print progress
        max_iter, abs_tol, rel_tol: Convergence parameters
        accelerate: Whether to use Nesterov acceleration
        rho_aggressive: Whether to use aggressive rho updates
        min_decrease_rate: Minimum acceptable decrease rate for early stopping
        decrease_check_window: Window size for checking decrease rate
        force_timeout_mins: Maximum runtime in minutes
    """
    n1, n2 = D.shape

    # Set default parameters as recommended in the paper
    if lambda_param is None:
        lambda_param = 1 / np.sqrt(n1)
    if mu is None:
        mu = np.sqrt(n2 / 2)

    mask_obs = ~np.isnan(D)

    D_filled = np.nan_to_num(D, nan=0)

    L1 = np.zeros_like(D)
    L2 = np.zeros_like(D)
    S1 = np.zeros_like(D)
    S2 = np.zeros_like(D)
    Z = np.zeros_like(D)
    Y1 = np.zeros_like(D)
    Y2 = np.zeros_like(D)
    Y3 = np.zeros_like(D)

    if accelerate:
        L1_accel = L1.copy()
        L2_accel = L2.copy()
        S1_accel = S1.copy()
        S2_accel = S2.copy()
        accel_param = 0.0

    best_residual = np.inf
    best_solution = {"L": L1.copy(), "S": S1.copy()}
    non_converged_count = 0
    unchanged_count = 0
    residual_history = []

    if rho_aggressive:
        rho_incr_factor = 1.5
        rho_decr_factor = 1 / 1.5
        rho_switch_threshold = 3
    else:
        rho_incr_factor = 1.2
        rho_decr_factor = 1 / 1.2
        rho_switch_threshold = 5

    iter_range = range(max_iter)
    if verbose:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=max_iter, position=0, leave=True)
        except ImportError:
            pbar = None
    else:
        pbar = None

    start_time = time.time()
    timeout = force_timeout_mins * 60

    for iter_num in iter_range:
        if time.time() - start_time > timeout:
            if verbose:
                print(f"Timeout after {force_timeout_mins} minutes.")
            break

        # For acceleration: store previous values and update acceleration parameter
        if accelerate and iter_num > 0:
            L1_prev, L2_prev = L1_accel.copy(), L2_accel.copy()
            S1_prev, S2_prev = S1_accel.copy(), S2_accel.copy()
            # Nesterov-style acceleration parameter
            accel_param = min(0.9, (iter_num - 1) / (iter_num + 3))

        # Store previous values for residual calculation
        L2_old = L2.copy()
        S2_old = S2.copy()

        # If using acceleration, work with accelerated variables
        if accelerate and iter_num > 0:
            L1_accel = L1 + accel_param * (L1 - L1_prev)
            L2_accel = L2 + accel_param * (L2 - L2_prev)
            S1_accel = S1 + accel_param * (S1 - S1_prev)
            S2_accel = S2 + accel_param * (S2 - S2_prev)

            L1_update = L2_accel - Y1 / rho
            S1_update = S2_accel - Y2 / rho
            Z_update = L2_accel + S2_accel - Y3 / rho
        else:
            L1_update = L2 - Y1 / rho
            S1_update = S2 - Y2 / rho
            Z_update = L2 + S2 - Y3 / rho

        # Update first primal variable (L1, S1, Z)
        L1 = prox_nuclear(L1_update, 1 / rho)
        S1 = prox_l1(S1_update, lambda_param / rho)

        # Handle missing values in Z update
        temp = Z_update
        temp_D = D_filled * mask_obs
        Z = prox_fro(temp - temp_D, mu / rho) + temp_D

        # Safety - prevent NaN/Inf
        L1 = np.nan_to_num(L1)
        S1 = np.nan_to_num(S1)
        Z = np.nan_to_num(Z)

        # Update second primal variables (L2, S2)
        L2_obs = mask_obs * (1 / 3 * (2 * L1 - S1 + Z + (2 * Y1 - Y2 + Y3) / rho))
        L2_unobs = (~mask_obs) * (L1 + Y1 / rho)
        L2 = L2_obs + L2_unobs

        S2_obs = mask_obs * (1 / 3 * (2 * S1 - L1 + Z + (2 * Y2 - Y1 + Y3) / rho))
        S2_unobs = (~mask_obs) * (S1 + Y2 / rho)
        S2 = S2_obs + S2_unobs

        # Safety
        L2 = np.nan_to_num(L2)
        S2 = np.nan_to_num(S2)

        # Update dual variables - standard version without confidence weights
        Y1 = Y1 + rho * (L1 - L2)
        Y2 = Y2 + rho * (S1 - S2)
        Y3 = Y3 + rho * mask_obs * (Z - (L2 + S2))

        # Safety
        Y1 = np.nan_to_num(Y1)
        Y2 = np.nan_to_num(Y2)
        Y3 = np.nan_to_num(Y3)

        # Calculate residuals safely
        try:
            res_primal = np.sqrt(
                linalg.norm(L1 - L2, "fro") ** 2
                + linalg.norm(S1 - S2, "fro") ** 2
                + linalg.norm(mask_obs * (Z - (L2 + S2)), "fro") ** 2
            )

            res_dual = rho * np.sqrt(
                linalg.norm(L2 - L2_old, "fro") ** 2
                + linalg.norm(S2 - S2_old, "fro") ** 2
                + linalg.norm(mask_obs * (L2 - L2_old + S2 - S2_old), "fro") ** 2
            )

            combined_res = res_primal + res_dual
            residual_history.append(combined_res)

            # Track best solution
            if np.isfinite(combined_res) and combined_res < best_residual:
                best_residual = combined_res
                best_solution = {"L": (L1 + L2) / 2, "S": (S1 + S2) / 2}
                non_converged_count = 0
            else:
                non_converged_count += 1

        except Exception as e:
            if verbose and iter_num % 100 == 0:
                print(f"Error in residual calculation: {str(e)}")
            non_converged_count += 1
            continue

        # Track if residual has hit small decrease rate
        if len(residual_history) >= decrease_check_window and iter_num % 10 == 0:
            # Calculate rate of decrease over window
            window_start = max(0, len(residual_history) - decrease_check_window)
            old_res = residual_history[window_start]
            if old_res > 0:  # Avoid division by zero
                decrease_rate = (old_res - combined_res) / old_res

                # If decrease rate is too small, consider stopping
                if 0 < decrease_rate < min_decrease_rate and combined_res < 1e-2:
                    if verbose:
                        print(
                            f"Slow convergence detected: decrease rate {decrease_rate:.6f} below threshold {min_decrease_rate}"
                        )
                    break

        # Check if residual is basically unchanged (possible cycling)
        if len(residual_history) > 1:
            prev_res = residual_history[-2]
            if abs(prev_res - combined_res) < 1e-8 * prev_res:
                unchanged_count += 1
            else:
                unchanged_count = 0

            # More aggressive early stopping for very small changes
            if unchanged_count > 30:
                if verbose:
                    print(f"Residual essentially unchanged for 30 iterations")
                break

        # Aggressive rho update
        if res_primal > rho_switch_threshold * res_dual:
            rho = min(rho * rho_incr_factor, 1e4)
        elif res_dual > rho_switch_threshold * res_primal:
            rho = max(rho * rho_decr_factor, 1e-4)

        # Dynamically scale the tolerance based on residual size
        if combined_res < 1e-2:
            dynamic_scale = max(1.0, 5.0 * np.log10(1e-2 / combined_res) + 1)
        else:
            dynamic_scale = 1.0

        # Scale factor based on matrix size
        n_entries = n1 * n2
        size_scale = np.log10(n_entries) / 5

        # Combined scaling factor
        scale_factor = size_scale * dynamic_scale

        eps_primal = abs_tol * scale_factor * np.sqrt(3 * n1 * n2) + rel_tol * max(
            np.sqrt(
                2 * linalg.norm(L1, "fro") ** 2
                + linalg.norm(S1, "fro") ** 2
                + linalg.norm(Z, "fro") ** 2
            ),
            np.sqrt(linalg.norm(L2, "fro") ** 2 + linalg.norm(S2, "fro") ** 2),
        )

        eps_dual = abs_tol * scale_factor * np.sqrt(3 * n1 * n2) + rel_tol * np.sqrt(
            linalg.norm(Y1, "fro") ** 2
            + linalg.norm(Y2, "fro") ** 2
            + linalg.norm(Y3, "fro") ** 2
        )

        if pbar is not None:
            pbar.update(1)
            pbar.set_description(f"Res: {combined_res:.8f}")

        # Early stopping if no improvement
        if non_converged_count > 200:
            if verbose:
                print(f"No improvement for 200 iterations")
            break

        # Standard convergence check
        if res_primal < eps_primal and res_dual < eps_dual:
            if verbose:
                print(f"Converged in {iter_num + 1} iterations")
            return {
                "L": (L1 + L2) / 2,
                "S": (S1 + S2) / 2,
                "final_iter": iter_num + 1,
                "converged": True,
                "res_primal": res_primal,
                "res_dual": res_dual,
                "runtime": time.time() - start_time,
            }

    if verbose:
        print(f"Algorithm stopped after {iter_num + 1} iterations")
        if "combined_res" in locals():
            print(f"Final residual: {combined_res:.8f}")

    return {
        "L": best_solution["L"],
        "S": best_solution["S"],
        "final_iter": iter_num + 1,
        "converged": False,
        "res_primal": res_primal if "res_primal" in locals() else np.nan,
        "res_dual": res_dual if "res_dual" in locals() else np.nan,
        "runtime": time.time() - start_time,
    }

"""
Hurst exponent estimation via quantile-graph mean-jump power-law scaling.

The estimator is described in Section 2 (Methods) of the paper.
For a fractional Brownian motion, the mean jump Delta(k) scales as k^H,
so a log-log regression of Delta(k) vs k yields the Hurst exponent H as
the slope.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Tuple

from src.quantile_graph import (
    assign_quantile_nodes,
    build_transition_matrix,
    random_walk_mean_jump_vectorized,
)


# Default parameters (paper-specified where possible; assumed where not)
DEFAULT_N_WALKS: int = 100       # paper states 100 random walks per QG (Fig. 3)
DEFAULT_WALK_LENGTH: int = 1000  # not specified; assumed
DEFAULT_N_REALIZATIONS: int = 100  # paper default protocol
DEFAULT_Q: int = 50              # paper default
DEFAULT_T: int = 10_000          # paper default (T = 10,000)


def build_lag_grid(
    k_min: int = 1,
    k_max: int = 1000,
    n_lags: int = 20,
) -> np.ndarray:
    """Return log-spaced integer lag grid from k_min to k_max.

    The paper shows data extending to k~1000 (Fig. 3); the complete
    lag grid is not specified.  A log-spaced grid of ~20 points spans
    more than two decades as required.  The exact lag grid is assumed.
    """
    lags_float = np.logspace(np.log10(k_min), np.log10(k_max), n_lags)
    lags = np.unique(np.round(lags_float).astype(int))
    lags = lags[(lags >= k_min) & (lags <= k_max)]
    return lags


def compute_delta_curve_fast(
    z: np.ndarray,
    Q: int,
    lags: np.ndarray,
    n_walks: int,
    walk_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute Delta(k) for each lag in ``lags`` using vectorised random walks.

    Parameters
    ----------
    z:
        Node-index array (values in 1..Q).
    Q, lags, n_walks, walk_length, rng:
        See ``estimate_hurst_single``.

    Returns
    -------
    deltas:
        Array of shape (len(lags),) with mean jump lengths.
    """
    deltas = np.empty(len(lags), dtype=np.float64)
    for idx, k in enumerate(lags):
        W = build_transition_matrix(z, Q, int(k))
        deltas[idx] = random_walk_mean_jump_vectorized(W, Q, n_walks, walk_length, rng)
    return deltas


def estimate_hurst_single(
    series: np.ndarray,
    Q: int,
    lags: np.ndarray,
    n_walks: int,
    walk_length: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Estimate H from a single time series using QG mean-jump scaling.

    For each lag k, the mean jump Delta(k) = (1/L)*sum |i-j| is computed
    via random walks on W_k (Section 2, Methods).  The slope of
    log(Delta(k)) vs log(k) is the estimated Hurst exponent.

    Parameters
    ----------
    series:
        1-D fBm (or cumsum of fGn) time series of length T.
    Q:
        Number of quantile nodes.
    lags:
        Integer lag array.
    n_walks:
        Number of random walks per lag-k graph (paper: 100 per QG).
    walk_length:
        Steps per random walk (not specified by paper; assumed).
    rng:
        NumPy random generator.

    Returns
    -------
    H_est:
        Estimated Hurst exponent (slope of log-log OLS regression).
    std_err:
        Standard error of slope from OLS.
    r_value:
        Pearson r of the log-log regression.
    deltas:
        Mean jump lengths for each lag (NaN if computation failed).
    valid_lags:
        Lags for which Delta(k) > 0 (used in regression).
    """
    z, _ = assign_quantile_nodes(series, Q)
    deltas = compute_delta_curve_fast(z, Q, lags, n_walks, walk_length, rng)

    # Exclude lags where Delta is zero or NaN to avoid log(0)
    valid = (deltas > 0) & np.isfinite(deltas)
    valid_lags = lags[valid]
    valid_deltas = deltas[valid]

    if len(valid_lags) < 2:
        return np.nan, np.nan, np.nan, deltas, valid_lags

    log_k = np.log10(valid_lags.astype(float))
    log_delta = np.log10(valid_deltas)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_delta)
    return float(slope), float(std_err), float(r_value), deltas, valid_lags


def estimate_hurst_ensemble(
    series_list: list[np.ndarray],
    Q: int,
    lags: np.ndarray,
    n_walks: int,
    walk_length: int,
    base_seed: int = 0,
) -> dict:
    """Estimate H from an ensemble of realizations.

    For each realization, ``estimate_hurst_single`` is called independently.
    The slope (H_est) is collected per realization.  The ensemble mean and
    95 % CI are reported.

    Parameters
    ----------
    series_list:
        List of 1-D time series (same fBm/cumsum process, different seeds).
    Q:
        Number of quantile nodes.
    lags:
        Integer lag array.
    n_walks:
        Number of random walks per lag-k graph.
    walk_length:
        Steps per random walk.
    base_seed:
        Base seed; each realization gets seed = base_seed + realization_index.

    Returns
    -------
    dict with keys:
        H_estimates: array of per-realization slopes
        mean_delta: mean Delta(k) curve averaged across realizations (shape: len(lags))
        std_delta: std of Delta(k) curves
        H_mean: ensemble mean of H estimates
        H_std: ensemble std
        H_ci95_lo: lower 95 % CI
        H_ci95_hi: upper 95 % CI
        std_errs: per-realization OLS standard errors
    """
    n = len(series_list)
    H_estimates = np.empty(n)
    std_errs = np.empty(n)
    delta_matrix = np.full((n, len(lags)), np.nan)

    for i, series in enumerate(series_list):
        rng = np.random.default_rng(base_seed + i)
        H_est, se, _, deltas, _ = estimate_hurst_single(
            series, Q, lags, n_walks, walk_length, rng
        )
        H_estimates[i] = H_est
        std_errs[i] = se
        delta_matrix[i, :] = deltas

    mean_delta = np.nanmean(delta_matrix, axis=0)
    std_delta = np.nanstd(delta_matrix, axis=0)

    valid_H = H_estimates[np.isfinite(H_estimates)]
    H_mean = float(np.nanmean(H_estimates))
    H_std = float(np.nanstd(H_estimates))

    # 95 % CI from t-distribution
    n_valid = int(np.sum(np.isfinite(H_estimates)))
    if n_valid > 1:
        t_crit = stats.t.ppf(0.975, df=n_valid - 1)
        margin = t_crit * H_std / np.sqrt(n_valid)
    else:
        margin = np.nan

    return {
        "H_estimates": H_estimates,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "H_mean": H_mean,
        "H_std": H_std,
        "H_ci95_lo": H_mean - (margin if np.isfinite(margin) else 0),
        "H_ci95_hi": H_mean + (margin if np.isfinite(margin) else 0),
        "std_errs": std_errs,
    }

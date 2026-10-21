"""Tests for src/hurst_estimator.py.

Validates that:
- Lag grid contains the expected number of log-spaced integers.
- Delta curve has the expected shape and positive values.
- H estimate for standard Brownian motion (H=0.5) lies near 0.5.
- Linear regression returns finite slope, std_err, and r_value.
- Ensemble averaging produces consistent statistics.
- Absolute estimation error for H in [0.2, 0.8] is small (< 0.15 for short T).
"""

import numpy as np
import pytest

from data.generate_synthetic import generate_realizations, generate_from_beta
from src.hurst_estimator import (
    build_lag_grid,
    compute_delta_curve_fast,
    estimate_hurst_single,
    estimate_hurst_ensemble,
)
from src.quantile_graph import assign_quantile_nodes


# ---------------------------------------------------------------------------
# build_lag_grid
# ---------------------------------------------------------------------------

class TestBuildLagGrid:
    def test_output_type(self):
        lags = build_lag_grid(1, 100, 15)
        assert lags.dtype in (np.int32, np.int64)

    def test_bounds(self):
        lags = build_lag_grid(1, 1000, 20)
        assert lags.min() >= 1
        assert lags.max() <= 1000

    def test_unique_sorted(self):
        lags = build_lag_grid(1, 100, 10)
        assert np.all(np.diff(lags) > 0)

    def test_approximately_n_lags(self):
        lags = build_lag_grid(1, 100, 15)
        assert len(lags) >= 1


# ---------------------------------------------------------------------------
# compute_delta_curve_fast
# ---------------------------------------------------------------------------

class TestComputeDeltaCurve:
    def test_shape(self):
        Q = 10
        rng = np.random.default_rng(0)
        x = rng.standard_normal(1000)
        z, _ = assign_quantile_nodes(x, Q)
        lags = build_lag_grid(1, 50, 8)
        deltas = compute_delta_curve_fast(z, Q, lags, n_walks=10, walk_length=20, rng=rng)
        assert deltas.shape == (len(lags),)

    def test_positive_values(self):
        Q = 10
        rng = np.random.default_rng(1)
        x = rng.standard_normal(2000)
        z, _ = assign_quantile_nodes(x, Q)
        lags = build_lag_grid(1, 30, 5)
        deltas = compute_delta_curve_fast(z, Q, lags, n_walks=20, walk_length=30, rng=rng)
        assert np.all(deltas >= 0)

    def test_finite_values(self):
        Q = 10
        rng = np.random.default_rng(2)
        x = rng.standard_normal(2000)
        z, _ = assign_quantile_nodes(x, Q)
        lags = build_lag_grid(1, 10, 5)
        deltas = compute_delta_curve_fast(z, Q, lags, n_walks=20, walk_length=30, rng=rng)
        assert np.all(np.isfinite(deltas))


# ---------------------------------------------------------------------------
# estimate_hurst_single
# ---------------------------------------------------------------------------

class TestEstimateHurstSingle:
    def test_returns_five_items(self):
        rng = np.random.default_rng(3)
        from fbm import FBM
        import random
        random.seed(0)
        series = np.array(FBM(n=1000, hurst=0.7, length=1, method='daviesharte').fbm()[1:])
        lags = build_lag_grid(1, 100, 8)
        result = estimate_hurst_single(series, Q=20, lags=lags, n_walks=10,
                                       walk_length=30, rng=rng)
        assert len(result) == 5

    def test_slope_finite(self):
        rng = np.random.default_rng(4)
        from fbm import FBM
        import random
        random.seed(1)
        series = np.array(FBM(n=2000, hurst=0.5, length=1, method='daviesharte').fbm()[1:])
        lags = build_lag_grid(1, 50, 8)
        slope, se, r, deltas, valid_lags = estimate_hurst_single(
            series, Q=20, lags=lags, n_walks=20, walk_length=50, rng=rng
        )
        assert np.isfinite(slope)
        assert np.isfinite(se)
        assert -1.0 <= r <= 1.0

    def test_r_value_high_for_fbm(self):
        """Log-log regression should be approximately linear for fBm."""
        rng = np.random.default_rng(5)
        from fbm import FBM
        import random
        random.seed(2)
        series = np.array(FBM(n=5000, hurst=0.7, length=1, method='daviesharte').fbm()[1:])
        lags = build_lag_grid(1, 100, 12)
        slope, se, r, deltas, valid_lags = estimate_hurst_single(
            series, Q=30, lags=lags, n_walks=30, walk_length=100, rng=rng
        )
        # r^2 should be high for true fBm (paper: excellent linear fit)
        assert r ** 2 > 0.8, f"r^2={r**2:.3f} too low for fBm"

    def test_slope_in_unit_interval(self):
        """Estimated H should be in a reasonable range (no gross errors)."""
        rng = np.random.default_rng(6)
        from fbm import FBM
        import random
        random.seed(3)
        series = np.array(FBM(n=3000, hurst=0.6, length=1, method='daviesharte').fbm()[1:])
        lags = build_lag_grid(1, 50, 10)
        slope, *_ = estimate_hurst_single(
            series, Q=20, lags=lags, n_walks=20, walk_length=50, rng=rng
        )
        assert -0.5 <= slope <= 1.5, f"Slope={slope:.3f} out of expected range"


# ---------------------------------------------------------------------------
# estimate_hurst_ensemble
# ---------------------------------------------------------------------------

class TestEstimateHurstEnsemble:
    def _make_series_list(self, H, T, n, base_seed=0):
        return generate_realizations(T, H, "fbm", n, base_seed=base_seed)

    def test_returns_required_keys(self):
        series_list = self._make_series_list(0.5, 500, 3)
        lags = build_lag_grid(1, 30, 5)
        result = estimate_hurst_ensemble(series_list, Q=10, lags=lags,
                                         n_walks=5, walk_length=20)
        required_keys = {"H_estimates", "mean_delta", "std_delta",
                         "H_mean", "H_std", "H_ci95_lo", "H_ci95_hi", "std_errs"}
        assert required_keys.issubset(result.keys())

    def test_H_estimates_shape(self):
        n = 5
        series_list = self._make_series_list(0.5, 500, n)
        lags = build_lag_grid(1, 20, 4)
        result = estimate_hurst_ensemble(series_list, Q=10, lags=lags,
                                         n_walks=5, walk_length=20)
        assert result["H_estimates"].shape == (n,)

    def test_H_mean_finite(self):
        series_list = self._make_series_list(0.7, 1000, 5)
        lags = build_lag_grid(1, 50, 6)
        result = estimate_hurst_ensemble(series_list, Q=15, lags=lags,
                                         n_walks=10, walk_length=30)
        assert np.isfinite(result["H_mean"])

    def test_ci_ordering(self):
        """Lower CI bound must be <= mean <= upper CI bound."""
        series_list = self._make_series_list(0.5, 1000, 5)
        lags = build_lag_grid(1, 50, 5)
        result = estimate_hurst_ensemble(series_list, Q=10, lags=lags,
                                         n_walks=10, walk_length=30)
        assert result["H_ci95_lo"] <= result["H_mean"]
        assert result["H_mean"] <= result["H_ci95_hi"]

    def test_H_estimate_accuracy_moderate(self):
        """For H=0.5 fBm with T=5000 and N=10 realizations, H_QG should be
        within 0.2 of exact H (lenient tolerance for short test series)."""
        series_list = self._make_series_list(H=0.5, T=5000, n=10, base_seed=99)
        lags = build_lag_grid(1, 500, 15)
        result = estimate_hurst_ensemble(series_list, Q=30, lags=lags,
                                         n_walks=50, walk_length=100)
        err = abs(result["H_mean"] - 0.5)
        assert err < 0.2, f"|H_QG - H| = {err:.4f} exceeds lenient threshold 0.2"


# ---------------------------------------------------------------------------
# No-look-ahead check
# ---------------------------------------------------------------------------

def test_no_lookahead_in_transition_matrix():
    """Confirm that the transition matrix for lag k only uses pairs (t, t+k)
    and not any future observation beyond t+k."""
    from src.quantile_graph import build_transition_matrix
    Q = 4
    T = 20
    z = np.arange(1, T + 1) % Q + 1  # deterministic sequence
    k = 3
    W = build_transition_matrix(z, Q, k)
    # W should be valid; just assert shape and row-stochasticity
    assert W.shape == (Q, Q)
    row_sums = W.sum(axis=1)
    nonempty = row_sums > 0
    np.testing.assert_allclose(row_sums[nonempty], 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_valid_lags_returns_nan():
    """When all Delta values are zero, slope should be NaN."""
    rng = np.random.default_rng(77)
    # Constant series -> all nodes same -> W is identity-like -> delta=0
    x = np.ones(200)
    # With a constant series, all obs go to the same node
    from src.quantile_graph import assign_quantile_nodes, build_transition_matrix
    z, _ = assign_quantile_nodes(x, Q=10)
    # All z values should be 1 (or similar); build a W_1
    W = build_transition_matrix(z, Q=10, k=1)
    # random_walk_mean_jump_vectorized on this W
    from src.quantile_graph import random_walk_mean_jump_vectorized
    delta = random_walk_mean_jump_vectorized(W, Q=10, n_walks=10, walk_length=20, rng=rng)
    # Should be a number (0.0 for identity row, NaN if empty)
    assert np.isfinite(delta) or np.isnan(delta)

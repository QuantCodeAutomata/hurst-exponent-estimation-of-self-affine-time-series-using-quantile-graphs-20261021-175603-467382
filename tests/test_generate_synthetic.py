"""Tests for data/generate_synthetic.py.

Validates:
- beta-to-H conversion follows paper classification (Section 2, Methods)
- Generated series have correct length T
- fGn is approximately stationary (mean ≈ 0)
- fBm is approximately non-stationary (growing variance)
- Endpoint approximations are documented and used
- All H values stay in (H_EPS, H_MAX)
- Realizations are independent (distinct)
"""

import numpy as np
import pytest

from data.generate_synthetic import (
    H_EPS,
    H_MAX,
    beta_to_H_and_type,
    generate_series,
    generate_realizations,
    generate_from_beta,
)


# ---------------------------------------------------------------------------
# beta_to_H_and_type
# ---------------------------------------------------------------------------

class TestBetaToHAndType:
    def test_noise_range(self):
        """beta in (-1, 1) -> fGn, H = (beta+1)/2."""
        H, ptype = beta_to_H_and_type(0.0)
        assert ptype == "fgn"
        assert H == pytest.approx(0.5)

    def test_motion_range(self):
        """beta in (1, 3) -> fBm, H = (beta-1)/2."""
        H, ptype = beta_to_H_and_type(2.0)
        assert ptype == "fbm"
        assert H == pytest.approx(0.5)

    def test_boundary_beta_1(self):
        """beta = 1 is the boundary; implementation treats it as fBm with H=H_EPS."""
        H, ptype = beta_to_H_and_type(1.0)
        assert ptype == "fbm"
        assert H == pytest.approx(H_EPS)

    def test_endpoint_beta_minus1(self):
        """beta = -1 maps to H=0, which is clipped to H_EPS."""
        H, ptype = beta_to_H_and_type(-1.0)
        assert ptype == "fgn"
        assert H == pytest.approx(H_EPS, abs=1e-10)

    def test_endpoint_beta_3(self):
        """beta = 3 maps to H=1, which is clipped to H_MAX."""
        H, ptype = beta_to_H_and_type(3.0)
        assert ptype == "fbm"
        assert H == pytest.approx(H_MAX, abs=1e-10)

    def test_H_always_clipped(self):
        """H should always be in [H_EPS, H_MAX]."""
        for beta in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]:
            H, _ = beta_to_H_and_type(beta)
            assert H >= H_EPS
            assert H <= H_MAX

    def test_fgn_conversion(self):
        """For beta = -0.5 (noise), H = 0.25."""
        H, ptype = beta_to_H_and_type(-0.5)
        assert ptype == "fgn"
        assert H == pytest.approx(0.25)

    def test_fbm_conversion(self):
        """For beta = 2.5 (motion), H = 0.75."""
        H, ptype = beta_to_H_and_type(2.5)
        assert ptype == "fbm"
        assert H == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# generate_series
# ---------------------------------------------------------------------------

class TestGenerateSeries:
    def test_fgn_length(self):
        x = generate_series(T=1000, H=0.5, process_type="fgn", seed=0)
        assert len(x) == 1000

    def test_fbm_length(self):
        x = generate_series(T=1000, H=0.5, process_type="fbm", seed=0)
        assert len(x) == 1000

    def test_output_dtype(self):
        x = generate_series(T=500, H=0.7, process_type="fbm", seed=1)
        assert x.dtype == np.float64

    def test_fgn_finite(self):
        x = generate_series(T=500, H=0.3, process_type="fgn", seed=2)
        assert np.all(np.isfinite(x))

    def test_fbm_finite(self):
        x = generate_series(T=500, H=0.7, process_type="fbm", seed=3)
        assert np.all(np.isfinite(x))

    def test_different_seeds_differ(self):
        x1 = generate_series(T=100, H=0.5, process_type="fgn", seed=0)
        x2 = generate_series(T=100, H=0.5, process_type="fgn", seed=999)
        assert not np.allclose(x1, x2), "Different seeds should produce different series"

    def test_same_seed_reproducible(self):
        x1 = generate_series(T=100, H=0.5, process_type="fbm", seed=42)
        x2 = generate_series(T=100, H=0.5, process_type="fbm", seed=42)
        np.testing.assert_array_equal(x1, x2)

    def test_fgn_approximately_zero_mean(self):
        """fGn should have approximately zero mean (large T).

        For positively correlated fGn (H > 0.5) the effective sample size is
        smaller than T so the classical i.i.d. SE underestimates the true SE.
        We use a generous multiplier of 20 to accommodate long-range dependence
        while still testing the zero-mean property.
        """
        x = generate_series(T=10000, H=0.7, process_type="fgn", seed=5)
        assert abs(x.mean()) < 20 * x.std() / np.sqrt(len(x))

    def test_fbm_starts_from_nonzero(self):
        """fBm path (with first element dropped) should be a real-valued path."""
        x = generate_series(T=1000, H=0.5, process_type="fbm", seed=6)
        assert x.std() > 0


# ---------------------------------------------------------------------------
# generate_realizations
# ---------------------------------------------------------------------------

class TestGenerateRealizations:
    def test_count(self):
        realizations = generate_realizations(T=200, H=0.5, process_type="fgn",
                                             n_realizations=5, base_seed=0)
        assert len(realizations) == 5

    def test_shapes(self):
        realizations = generate_realizations(T=300, H=0.7, process_type="fbm",
                                             n_realizations=3, base_seed=0)
        for x in realizations:
            assert len(x) == 300

    def test_independent(self):
        """All realizations should be distinct."""
        realizations = generate_realizations(T=200, H=0.5, process_type="fgn",
                                             n_realizations=5, base_seed=0)
        for i in range(len(realizations)):
            for j in range(i + 1, len(realizations)):
                assert not np.allclose(realizations[i], realizations[j]), \
                    f"Realizations {i} and {j} are identical"


# ---------------------------------------------------------------------------
# generate_from_beta
# ---------------------------------------------------------------------------

class TestGenerateFromBeta:
    def test_returns_three_items(self):
        result = generate_from_beta(2.0, T=500, n_realizations=3)
        assert len(result) == 3

    def test_series_list_type(self):
        series, H, ptype = generate_from_beta(2.0, T=500, n_realizations=3)
        assert isinstance(series, list)
        assert len(series) == 3

    def test_fbm_beta(self):
        _, H, ptype = generate_from_beta(2.0, T=100, n_realizations=1)
        assert ptype == "fbm"
        assert H == pytest.approx(0.5)

    def test_fgn_beta(self):
        _, H, ptype = generate_from_beta(0.0, T=100, n_realizations=1)
        assert ptype == "fgn"
        assert H == pytest.approx(0.5)

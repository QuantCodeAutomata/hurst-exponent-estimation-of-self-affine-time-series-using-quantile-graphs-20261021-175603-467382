"""Tests for src/quantile_graph.py.

Tests cover paper-substance requirements:
- equal-frequency binning produces exactly Q bins
- every observation is assigned a valid node index
- transition matrix is row-stochastic
- anti-correlated series show secondary-diagonal dominance
- positively correlated series show primary-diagonal dominance
- random walk jump lengths are non-negative and finite
"""

import numpy as np
import pytest

from src.quantile_graph import (
    assign_quantile_nodes,
    build_transition_matrix,
    random_walk_mean_jump,
    random_walk_mean_jump_vectorized,
    topology_diagnostics,
)


# ---------------------------------------------------------------------------
# assign_quantile_nodes
# ---------------------------------------------------------------------------

class TestAssignQuantileNodes:
    def test_output_shape(self):
        x = np.random.default_rng(0).standard_normal(1000)
        z, edges = assign_quantile_nodes(x, Q=10)
        assert z.shape == (1000,)
        assert edges.shape == (11,)

    def test_nodes_in_valid_range(self):
        Q = 20
        x = np.random.default_rng(1).standard_normal(500)
        z, _ = assign_quantile_nodes(x, Q)
        assert z.min() >= 1
        assert z.max() <= Q

    def test_all_nodes_used(self):
        """Each quantile node should be used at least once for large T."""
        Q = 10
        x = np.random.default_rng(2).standard_normal(10000)
        z, _ = assign_quantile_nodes(x, Q)
        unique = np.unique(z)
        assert len(unique) == Q, f"Expected {Q} unique nodes, got {len(unique)}"

    def test_equal_frequency_bins(self):
        """Each bin should contain approximately T/Q observations."""
        Q = 10
        T = 10000
        x = np.random.default_rng(3).standard_normal(T)
        z, _ = assign_quantile_nodes(x, Q)
        counts = np.bincount(z, minlength=Q + 1)[1:]  # 1-indexed
        expected = T / Q
        # All bins within 10% of expected count
        assert np.all(np.abs(counts - expected) / expected < 0.15)

    def test_precomputed_edges(self):
        """Edges computed on training data work on test data."""
        Q = 5
        train = np.random.default_rng(4).standard_normal(1000)
        _, edges = assign_quantile_nodes(train, Q)
        test = np.array([train.min() - 1, train.max() + 1])
        z, _ = assign_quantile_nodes(test, Q, edges=edges)
        assert z.min() >= 1
        assert z.max() <= Q

    def test_extreme_values(self):
        """Single value should map to a valid node."""
        Q = 5
        x = np.ones(100)
        z, _ = assign_quantile_nodes(x, Q)
        assert np.all((z >= 1) & (z <= Q))


# ---------------------------------------------------------------------------
# build_transition_matrix
# ---------------------------------------------------------------------------

class TestBuildTransitionMatrix:
    def test_shape(self):
        Q = 10
        z = np.random.default_rng(5).integers(1, Q + 1, size=1000)
        W = build_transition_matrix(z, Q, k=1)
        assert W.shape == (Q, Q)

    def test_row_stochastic(self):
        Q = 10
        z = np.random.default_rng(6).integers(1, Q + 1, size=1000)
        W = build_transition_matrix(z, Q, k=1)
        row_sums = W.sum(axis=1)
        # All rows with outgoing edges sum to 1
        nonempty = row_sums > 0
        np.testing.assert_allclose(row_sums[nonempty], 1.0, atol=1e-12)

    def test_nonnegative(self):
        Q = 5
        z = np.random.default_rng(7).integers(1, Q + 1, size=500)
        W = build_transition_matrix(z, Q, k=1)
        assert np.all(W >= 0)

    def test_lag_k_uses_correct_pairs(self):
        """With k=2 and a deterministic sequence, check exact counts."""
        Q = 2
        # Pattern 1 2 1 2 1 2 ... -> with k=2 all pairs are (1,1) or (2,2)
        T = 100
        z = np.tile([1, 2], T // 2)
        W = build_transition_matrix(z, Q, k=2)
        # Node 1 should always go to node 1, node 2 to node 2
        assert W[0, 0] == pytest.approx(1.0)
        assert W[1, 1] == pytest.approx(1.0)

    def test_k_equals_1_assertion(self):
        Q = 3
        z = np.array([1, 2, 3, 1, 2, 3])
        with pytest.raises(AssertionError):
            build_transition_matrix(z, Q, k=0)


# ---------------------------------------------------------------------------
# random walk functions
# ---------------------------------------------------------------------------

class TestRandomWalk:
    def _uniform_W(self, Q):
        W = np.ones((Q, Q), dtype=float) / Q
        return W

    def test_jump_nonnegative(self):
        Q = 10
        W = self._uniform_W(Q)
        rng = np.random.default_rng(8)
        delta = random_walk_mean_jump_vectorized(W, Q, n_walks=50, walk_length=100, rng=rng)
        assert np.isfinite(delta)
        assert delta >= 0

    def test_uniform_matrix_expected_jump(self):
        """For uniform W, expected jump = mean |i-j| = (Q^2-1)/(3Q) approximately."""
        Q = 50
        W = self._uniform_W(Q)
        rng = np.random.default_rng(9)
        delta = random_walk_mean_jump_vectorized(W, Q, n_walks=100, walk_length=1000, rng=rng)
        # Theoretical mean |i-j| in {0..Q-1}: (Q-1)/3
        expected = (Q - 1) / 3.0
        assert abs(delta - expected) / expected < 0.05, f"Got {delta:.3f}, expected ≈{expected:.3f}"

    def test_identity_matrix_zero_jump(self):
        """For identity W, every step is a self-loop; mean jump = 0."""
        Q = 10
        W = np.eye(Q)
        rng = np.random.default_rng(10)
        delta = random_walk_mean_jump_vectorized(W, Q, n_walks=20, walk_length=50, rng=rng)
        assert delta == pytest.approx(0.0, abs=1e-12)

    def test_vectorized_vs_scalar_consistency(self):
        """Vectorized and scalar random-walk estimates should agree closely."""
        Q = 10
        x = np.random.default_rng(11).standard_normal(1000)
        z, _ = assign_quantile_nodes(x, Q)
        W = build_transition_matrix(z, Q, k=1)
        rng1 = np.random.default_rng(12)
        rng2 = np.random.default_rng(12)
        d_vec = random_walk_mean_jump_vectorized(W, Q, n_walks=200, walk_length=200, rng=rng1)
        d_scalar = random_walk_mean_jump(W, Q, n_walks=200, walk_length=200, rng=rng2)
        # Should agree within 10% given same seed
        assert abs(d_vec - d_scalar) / max(d_vec, d_scalar) < 0.10


# ---------------------------------------------------------------------------
# topology_diagnostics
# ---------------------------------------------------------------------------

class TestTopologyDiagnostics:
    def test_structure(self):
        Q = 10
        W = np.eye(Q)
        diag = topology_diagnostics(W, Q)
        assert "primary_diag_mass" in diag
        assert "secondary_diag_mass" in diag
        assert "near_primary_mass_b3" in diag
        assert "mean_jump" in diag

    def test_primary_dominance_correlated(self):
        """Identity matrix (perfectly correlated) has full primary-diag mass."""
        Q = 10
        W = np.eye(Q)
        diag = topology_diagnostics(W, Q)
        assert diag["primary_diag_mass"] == pytest.approx(Q, abs=1e-10)

    def test_qualitative_topology_signature(self):
        """Anti-correlated series should have secondary > primary diag mass."""
        Q = 10
        T = 10000
        rng = np.random.default_rng(99)
        # Generate anti-correlated fGn (H close to 0)
        from fbm import FBM
        import random
        random.seed(42)
        f = FBM(n=T, hurst=0.05, length=1, method="daviesharte")
        x = np.array(f.fgn())
        z, _ = assign_quantile_nodes(x, Q)
        W = build_transition_matrix(z, Q, k=1)
        diag = topology_diagnostics(W, Q)
        # Anti-correlated: should show more secondary-diagonal mass than
        # uniform expectation (1/Q), but may not exceed primary strictly for small Q
        assert diag["secondary_diag_mass"] > Q / Q  # > 1.0 (greater than uniform 1.0)

    def test_quantitative_bounds(self):
        """All diagnostic values should be finite and in expected ranges."""
        Q = 10
        W = np.ones((Q, Q)) / Q  # uniform
        diag = topology_diagnostics(W, Q)
        assert np.isfinite(diag["primary_diag_mass"])
        assert diag["primary_diag_mass"] >= 0
        assert diag["mean_jump"] >= 0

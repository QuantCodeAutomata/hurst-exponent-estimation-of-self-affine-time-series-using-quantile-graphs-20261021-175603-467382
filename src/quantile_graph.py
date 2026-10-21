"""
Quantile-graph construction and random-walk utilities.

Based on: Hurst exponent estimation of self-affine time series using
quantile graphs (see paper Section 2. Methods).

The quantile graph maps a scalar time series to a directed weighted
graph whose Q nodes correspond to equal-frequency amplitude quantiles.
The lag-k transition matrix W_k is the row-normalized count of observed
(node_t, node_{t+k}) pairs.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def assign_quantile_nodes(
    x: np.ndarray,
    Q: int,
    edges: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each observation to an ordered quantile node index in {1, ..., Q}.

    Equal-frequency (equal-count) bins are used, following the paper's
    description of Q amplitude quantiles (Section 2, Methods).

    Parameters
    ----------
    x:
        1-D time series of length T.
    Q:
        Number of quantile bins / graph nodes.
    edges:
        Pre-computed bin edges of length Q+1.  If None, edges are derived
        from ``x`` itself (in-sample assignment).

    Returns
    -------
    z:
        Integer node-index array of shape (T,) with values in {1, ..., Q}.
    edges:
        Bin edges of length Q+1 used for the assignment.
    """
    if edges is None:
        probs = np.linspace(0.0, 1.0, Q + 1)
        edges = np.quantile(x, probs)
        # Ensure the left boundary is strictly less than all values
        edges[0] -= 1e-10 * (abs(edges[0]) + 1.0)
        # Ensure the right boundary is strictly greater than all values
        edges[-1] += 1e-10 * (abs(edges[-1]) + 1.0)

    # searchsorted with 'right' maps each x to the index of the first edge
    # strictly greater than x, giving bins 0..Q-1; clip to [0, Q-1].
    z = np.searchsorted(edges[1:], x, side="left")
    z = np.clip(z, 0, Q - 1) + 1  # shift to {1, ..., Q}
    return z.astype(np.int32), edges


def build_transition_matrix(
    z: np.ndarray,
    Q: int,
    k: int,
) -> np.ndarray:
    """Build the row-stochastic lag-k transition matrix W_k.

    C_ij = number of valid positions t where z_t == i and z_{t+k} == j
    (t = 0, ..., T-k-1 in 0-indexed terms; i.e. T-k valid pairs).
    W_k is obtained by row-normalising C.  Rows with no outgoing
    transitions are left as zero rows (should not occur for T >> Q).

    Parameters
    ----------
    z:
        Node-index array with values in {1, ..., Q}.
    Q:
        Number of nodes (graph dimension).
    k:
        Lag.

    Returns
    -------
    W:
        (Q, Q) row-stochastic transition matrix.
    """
    assert k >= 1, "Lag k must be >= 1"
    # Efficient counting via bincount on linearised (i,j) pairs
    pairs = (z[:-k] - 1) * Q + (z[k:] - 1)  # values in [0, Q*Q)
    counts = np.bincount(pairs, minlength=Q * Q).reshape(Q, Q).astype(np.float64)

    row_sums = counts.sum(axis=1, keepdims=True)
    # Safe divide: replace zero row-sums with 1 before dividing, then mask
    row_sums_safe = np.where(row_sums > 0, row_sums, 1.0)
    W = np.where(row_sums > 0, counts / row_sums_safe, 0.0)
    return W


def random_walk_mean_jump_vectorized(
    W: np.ndarray,
    Q: int,
    n_walks: int,
    walk_length: int,
    rng: np.random.Generator,
) -> float:
    """Fast vectorised version of random_walk_mean_jump.

    Runs all walks simultaneously using pre-sampled uniform random numbers.
    """
    # Pre-compute cumulative row probabilities for fast sampling
    cumW = np.cumsum(W, axis=1)

    # Starting nodes for all walks (uniform)
    current = rng.integers(0, Q, size=n_walks)  # (n_walks,)

    total_jump = 0.0
    total_steps = 0

    # Pre-generate all random numbers at once
    u = rng.random((walk_length, n_walks))  # (walk_length, n_walks)

    for step in range(walk_length):
        row_cum = cumW[current]  # (n_walks, Q)
        # For each walk, find the next node via inverse CDF
        next_node = (u[step, :, np.newaxis] >= row_cum).sum(axis=1)  # (n_walks,)
        next_node = np.clip(next_node, 0, Q - 1)
        total_jump += np.abs(current - next_node).sum()
        total_steps += n_walks
        current = next_node

    if total_steps == 0:
        return np.nan
    return float(total_jump / total_steps)


def random_walk_mean_jump(
    W: np.ndarray,
    Q: int,
    n_walks: int,
    walk_length: int,
    rng: np.random.Generator,
) -> float:
    """Estimate the mean jump length by simulating random walks on W.

    At each step, the walker at node i transitions to node j with
    probability W[i, j].  The jump length at that step is |i - j|
    (using 0-based node differences; the result is the same as
    |(i+1) - (j+1)|).

    Parameters
    ----------
    W:
        (Q, Q) row-stochastic transition matrix.
    Q:
        Number of nodes.
    n_walks:
        Number of independent random walks.
    walk_length:
        Number of transitions per walk.
    rng:
        NumPy random generator for reproducibility.

    Returns
    -------
    delta:
        Scalar mean jump length averaged over all steps and all walks.
    """
    nodes = np.arange(Q)
    total_jump = 0.0
    total_steps = 0

    for _ in range(n_walks):
        # Start from a uniformly random node
        current = rng.integers(0, Q)
        for _ in range(walk_length):
            row = W[current]
            if row.sum() == 0:
                # Dead-end node: restart uniformly
                current = rng.integers(0, Q)
                continue
            next_node = rng.choice(nodes, p=row)
            total_jump += abs(current - next_node)
            total_steps += 1
            current = next_node

    if total_steps == 0:
        return np.nan
    return total_jump / total_steps


def compute_delta_curve(
    z: np.ndarray,
    Q: int,
    lags: np.ndarray,
    n_walks: int,
    walk_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute Delta(k) for each lag in ``lags`` from node sequence ``z``.

    Delta(k) is the mean jump length of random walks on W_k.
    Quantile edges are computed once from the full series and reused for
    all lags (Section 2 of the paper).

    Parameters
    ----------
    z:
        Node-index array (1..Q).
    Q:
        Number of nodes.
    lags:
        Integer array of lag values.
    n_walks:
        Number of random walks per W_k.
    walk_length:
        Steps per walk.
    rng:
        NumPy generator.

    Returns
    -------
    deltas:
        Array of shape (len(lags),) with mean jump lengths.
    """
    deltas = np.empty(len(lags), dtype=np.float64)
    for idx, k in enumerate(lags):
        W = build_transition_matrix(z, Q, int(k))
        deltas[idx] = random_walk_mean_jump(W, Q, n_walks, walk_length, rng)
    return deltas


def topology_diagnostics(W: np.ndarray, Q: int) -> dict:
    """Compute supplementary topology statistics for a transition matrix.

    These statistics are implementation diagnostics rather than paper-
    defined quantities.

    Parameters
    ----------
    W:
        (Q, Q) row-stochastic transition matrix.
    Q:
        Number of nodes.

    Returns
    -------
    dict with keys:
        primary_diag_mass: sum_i W[i,i]
        secondary_diag_mass: sum_i W[i, Q-1-i]
        near_primary_mass_b3: mass within |i-j| <= 3
        mean_jump: expected |i-j| under uniform node distribution
    """
    primary = float(np.trace(W))
    secondary = float(sum(W[i, Q - 1 - i] for i in range(Q)))

    near_primary = 0.0
    for b in range(-3, 4):
        i_idx = np.arange(max(0, -b), min(Q, Q - b))
        j_idx = i_idx + b
        mask = (j_idx >= 0) & (j_idx < Q)
        near_primary += float(W[i_idx[mask], j_idx[mask]].sum())

    # Mean |i-j| under uniform starting node
    mean_jump = float(
        np.mean([abs(i - j) * W[i, j] for i in range(Q) for j in range(Q)])
    ) * Q

    return {
        "primary_diag_mass": primary,
        "secondary_diag_mass": secondary,
        "near_primary_mass_b3": near_primary,
        "mean_jump": mean_jump,
    }

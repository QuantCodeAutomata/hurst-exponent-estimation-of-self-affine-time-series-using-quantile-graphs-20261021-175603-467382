"""
Experiment 1: Quantile-graph transition topology across self-affine processes.

Reproduces Fig. 2 of the paper by generating synthetic self-affine time series
at different spectral exponents beta, constructing the lag-1 transition matrix
W_1, and visualising the resulting heatmaps.

Expected outcomes (Section 3, Results):
- Anti-correlated signals (beta < 0) -> secondary-diagonal dominance in W_1
- White noise (beta ≈ 0) -> approximately uniform W_1
- Positively correlated signals (beta > 0) -> primary-diagonal dominance in W_1

Deviation from paper:
- Davies-Harte method (fbm package) used instead of Abry-Sellan wavelet
  synthesis (see data/generate_synthetic.py docstring and RESULTS.md).
- beta = -1 and beta = 1 endpoint/boundary cases approximated with H = 0.01
  and H = 0.01 (fBm) respectively (see RESULTS.md).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data.generate_synthetic import generate_from_beta
from src.quantile_graph import (
    assign_quantile_nodes,
    build_transition_matrix,
    topology_diagnostics,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Paper-specified parameters (Section 2, Methods)
T: int = 10_000       # series length
Q: int = 10           # user override for speed (paper default: Q=50)
K_LAG: int = 1        # lag for Fig. 2
N_REALIZATIONS: int = 100  # paper default protocol

# Beta values shown in Fig. 2
BETA_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def compute_mean_transition_matrix(
    beta: float,
    T: int = T,
    Q: int = Q,
    k: int = K_LAG,
    n_realizations: int = N_REALIZATIONS,
    base_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Compute the ensemble-averaged lag-k transition matrix for a given beta.

    Parameters
    ----------
    beta, T, Q, k, n_realizations, base_seed:
        See module-level constants for descriptions.

    Returns
    -------
    W_mean:
        (Q, Q) matrix of ensemble-averaged row-stochastic W_k.
    W_std:
        (Q, Q) matrix of realization-level standard deviations.
    process_type:
        "fgn" or "fbm".
    H:
        Hurst exponent used in generation.
    """
    series_list, H, ptype = generate_from_beta(beta, T, n_realizations, base_seed)

    W_stack = np.empty((n_realizations, Q, Q))
    for i, x in enumerate(series_list):
        z, _ = assign_quantile_nodes(x, Q)
        W_stack[i] = build_transition_matrix(z, Q, k)

    # Per-realization W_1 is already row-stochastic; average them
    W_mean = W_stack.mean(axis=0)
    W_std = W_stack.std(axis=0)

    # Re-normalise mean to ensure row-stochasticity (floating-point safety)
    row_sums = W_mean.sum(axis=1, keepdims=True)
    W_mean = np.where(row_sums > 0, W_mean / row_sums, 0.0)

    return W_mean, W_std, ptype, H


def run_experiment(
    beta_values: list[float] = BETA_VALUES,
    T: int = T,
    Q: int = Q,
    n_realizations: int = N_REALIZATIONS,
) -> dict:
    """Run Experiment 1 and save artefacts to results/.

    Returns
    -------
    dict mapping beta -> dict with W_mean, W_std, diagnostics, H, process_type.
    """
    t0 = time.time()
    results = {}
    all_diag = []

    for beta in beta_values:
        print(f"  beta={beta:+.1f} ...", end=" ", flush=True)
        W_mean, W_std, ptype, H = compute_mean_transition_matrix(
            beta, T, Q, K_LAG, n_realizations
        )
        diag = topology_diagnostics(W_mean, Q)
        results[beta] = {
            "W_mean": W_mean,
            "W_std": W_std,
            "process_type": ptype,
            "H": H,
            "diagnostics": diag,
        }
        all_diag.append({"beta": beta, "H": H, "process_type": ptype, **diag})
        print(
            f"H={H:.3f} ({ptype}), primary_diag={diag['primary_diag_mass']:.3f}, "
            f"secondary_diag={diag['secondary_diag_mass']:.3f}"
        )

    elapsed = time.time() - t0
    print(f"  Experiment 1 completed in {elapsed:.1f}s")

    # ------------------------------------------------------------------ plots
    _plot_heatmaps(results, beta_values, Q)
    _plot_topology_summary(all_diag)

    # ------------------------------------------------------------------ save
    _save_diagnostics(all_diag)
    return results


def _plot_heatmaps(results: dict, beta_values: list[float], Q: int) -> None:
    """Plot transition-matrix heatmaps comparable to Fig. 2."""
    n = len(beta_values)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2), constrained_layout=True)
    if n == 1:
        axes = [axes]

    vmax = float(np.max([results[b]["W_mean"].max() for b in beta_values]))
    vmin = 0.0

    for ax, beta in zip(axes, beta_values):
        W = results[beta]["W_mean"]
        H = results[beta]["H"]
        ptype = results[beta]["process_type"]
        sns.heatmap(
            W,
            ax=ax,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            cbar=(ax == axes[-1]),
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_title(f"β={beta:+.1f}\nH={H:.2f} ({ptype})", fontsize=9)
        ax.set_xlabel("Node j")
        ax.set_ylabel("Node i")

    fig.suptitle(
        f"Lag-1 transition matrices W₁ for self-affine series\n"
        f"T={T}, Q={Q}, N={N_REALIZATIONS} realizations (ensemble mean)",
        fontsize=10,
    )
    out = RESULTS_DIR / "exp1_transition_heatmaps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_topology_summary(all_diag: list[dict]) -> None:
    """Plot topology diagnostics vs beta."""
    betas = [d["beta"] for d in all_diag]
    prim = [d["primary_diag_mass"] for d in all_diag]
    sec = [d["secondary_diag_mass"] for d in all_diag]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(betas, prim, "o-", label="Primary-diagonal mass")
    ax.plot(betas, sec, "s--", label="Secondary-diagonal mass")
    ax.axhline(1.0 / Q, color="gray", linestyle=":", label=f"Uniform (1/Q={1/Q:.3f})")
    ax.set_xlabel("β (spectral exponent)")
    ax.set_ylabel("Diagonal mass (sum of W₁ entries)")
    ax.set_title("Topology diagnostics of W₁ vs β (Exp. 1)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    out = RESULTS_DIR / "exp1_topology_diagnostics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _save_diagnostics(all_diag: list[dict]) -> None:
    """Save topology diagnostics as JSON and numpy arrays."""
    # JSON (scalars only)
    out_json = RESULTS_DIR / "exp1_diagnostics.json"
    with open(out_json, "w") as f:
        json.dump(all_diag, f, indent=2, default=float)
    print(f"  Saved: {out_json}")

    # Numpy arrays
    for diag in all_diag:
        beta_str = f"{diag['beta']:+.1f}".replace("+", "p").replace("-", "m")
        # Already saved as part of results dict; summary in JSON is sufficient


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 1: Quantile-graph transition topology")
    print(f"  T={T}, Q={Q}, k={K_LAG}, N_real={N_REALIZATIONS}")
    print(f"  Beta values: {BETA_VALUES}")
    print("=" * 60)
    results = run_experiment()
    print("Experiment 1 complete.")

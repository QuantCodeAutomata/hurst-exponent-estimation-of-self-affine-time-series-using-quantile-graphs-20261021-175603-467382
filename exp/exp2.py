"""
Experiment 2: Estimation of the Hurst exponent of fractional Brownian motion
from quantile-graph mean jump scaling.

Validates the core QG estimator (Section 2, Methods; Fig. 3, Fig. 4).

For each target beta (and corresponding H), 100 fBm realizations are generated
and the mean jump Delta(k) vs lag k is computed.  The log-log slope estimates H.

Expected outcomes (Section 3, Results):
- Power-law scaling of Delta(k) spans more than two decades (Fig. 3).
- Estimated H_QG is strongly linearly correlated with exact H (Fig. 4).
- Agreement is reported as excellent by the authors.

Deviation from paper:
- Davies-Harte method used instead of Abry-Sellan wavelet synthesis.
- Random-walk length L (steps per walk) not specified by paper; L=200 assumed.
- Lag grid (20 log-spaced points from 1 to 1000) not specified; assumed.
- Starting-node distribution is uniform; not specified by paper.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from data.generate_synthetic import generate_from_beta
from src.hurst_estimator import (
    build_lag_grid,
    estimate_hurst_single,
    estimate_hurst_ensemble,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Paper-specified parameters
T: int = 10_000
Q: int = 10           # user override for speed (paper default: Q=50)
N_REALIZATIONS: int = 100
N_WALKS: int = 100           # paper states 100 random walks per QG
WALK_LENGTH: int = 200       # not specified; assumed
K_MIN: int = 1
K_MAX: int = 1000
N_LAGS: int = 20             # log-spaced; not specified by paper

# Fig. 3 beta values for fBm: beta = 2H+1, so H = (beta-1)/2
# beta  -> H:  1.0->0.0, 1.5->0.25, 2.0->0.5, 2.5->0.75, 3.0->1.0
BETA_VALUES_FIG3 = [1.0, 1.5, 2.0, 2.5, 3.0]
# Fig. 4 uses a finer grid for scatter plot
BETA_VALUES_FIG4 = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]


def run_single_beta(
    beta: float,
    T: int,
    Q: int,
    lags: np.ndarray,
    n_realizations: int,
    n_walks: int,
    walk_length: int,
) -> tuple[dict, float, str]:
    """Generate fBm ensemble and estimate H for one beta value.

    Returns
    -------
    ensemble_result:
        Dict from estimate_hurst_ensemble.
    H_exact:
        Ground-truth Hurst exponent.
    process_type:
        "fbm" always for this experiment.
    """
    series_list, H_exact, ptype = generate_from_beta(
        beta, T, n_realizations, base_seed=int(round(beta * 100))
    )
    lags_arr = lags.astype(int)
    result = estimate_hurst_ensemble(
        series_list, Q, lags_arr, n_walks, walk_length, base_seed=0
    )
    return result, H_exact, ptype


def run_experiment(
    beta_values: list[float] = BETA_VALUES_FIG3,
    T: int = T,
    Q: int = Q,
    n_realizations: int = N_REALIZATIONS,
    n_walks: int = N_WALKS,
    walk_length: int = WALK_LENGTH,
    n_lags: int = N_LAGS,
    tag: str = "fig3",
) -> dict:
    """Run Experiment 2 for a given list of beta values.

    Returns
    -------
    dict mapping beta -> {'result': ensemble_dict, 'H_exact': float,
                          'process_type': str, 'H_error': float}
    """
    lags = build_lag_grid(K_MIN, K_MAX, n_lags)
    t0 = time.time()
    all_results = {}

    print(f"  Lags: {lags}")
    for beta in beta_values:
        print(f"  beta={beta:.1f} ...", end=" ", flush=True)
        result, H_exact, ptype = run_single_beta(
            beta, T, Q, lags, n_realizations, n_walks, walk_length
        )
        H_err = abs(result["H_mean"] - H_exact)
        all_results[beta] = {
            "result": result,
            "H_exact": H_exact,
            "process_type": ptype,
            "H_error": H_err,
            "lags": lags.tolist(),
        }
        print(
            f"H_exact={H_exact:.3f}, H_QG={result['H_mean']:.3f}±{result['H_std']:.3f}, "
            f"|err|={H_err:.4f}"
        )

    elapsed = time.time() - t0
    print(f"  {tag} completed in {elapsed:.1f}s")

    # --- Fig 3: log-log Delta(k) curves ---
    _plot_delta_curves(all_results, lags, tag)

    # --- Fig 4: H_QG vs H_exact scatter ---
    _plot_scatter(all_results, tag)

    # --- Save JSON ---
    _save_results(all_results, lags, tag)

    return all_results


def _plot_delta_curves(results: dict, lags: np.ndarray, tag: str) -> None:
    """Reproduce Fig. 3: log Delta(k) vs log k for each beta."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))

    for (beta, data), color in zip(sorted(results.items()), colors):
        H_exact = data["H_exact"]
        mean_delta = data["result"]["mean_delta"]
        valid = mean_delta > 0
        if valid.sum() < 2:
            continue
        ax.loglog(
            lags[valid], mean_delta[valid],
            "o-", color=color, markersize=4, linewidth=1.5,
            label=f"β={beta:.1f} (H={H_exact:.2f})"
        )

    ax.set_xlabel("Lag k")
    ax.set_ylabel("Mean jump Δ(k)")
    ax.set_title(
        f"Log-log scaling of mean jump Δ(k) vs lag k (Exp. 2, {tag})\n"
        f"T={T}, Q={Q}, N={N_REALIZATIONS}, n_walks={N_WALKS}, L={WALK_LENGTH}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    out = RESULTS_DIR / f"exp2_delta_curves_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_scatter(results: dict, tag: str) -> None:
    """Reproduce Fig. 4: H_QG vs H_exact with error bars."""
    betas = sorted(results.keys())
    H_exact_arr = np.array([results[b]["H_exact"] for b in betas])
    H_qg_arr = np.array([results[b]["result"]["H_mean"] for b in betas])
    H_lo = np.array([results[b]["result"]["H_ci95_lo"] for b in betas])
    H_hi = np.array([results[b]["result"]["H_ci95_hi"] for b in betas])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(
        H_exact_arr, H_qg_arr,
        yerr=[H_qg_arr - H_lo, H_hi - H_qg_arr],
        fmt="o", capsize=4, label="H_QG (mean ± 95% CI)"
    )
    lim = [0, 1.05]
    ax.plot(lim, lim, "k--", linewidth=1, label="Identity (H_QG = H)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Exact H")
    ax.set_ylabel("Estimated H_QG")
    ax.set_title(
        f"Estimated vs exact Hurst exponent (Exp. 2, {tag})\n"
        f"T={T}, Q={Q}, N={N_REALIZATIONS}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Compute linear correlation
    if len(H_exact_arr) > 2:
        r, p = stats.pearsonr(H_exact_arr, H_qg_arr)
        ax.text(0.05, 0.92, f"r={r:.4f}, p={p:.2e}", transform=ax.transAxes, fontsize=9)

    out = RESULTS_DIR / f"exp2_scatter_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _save_results(results: dict, lags: np.ndarray, tag: str) -> None:
    """Save JSON summary of Experiment 2 results."""
    summary = []
    for beta, data in sorted(results.items()):
        row = {
            "beta": beta,
            "H_exact": data["H_exact"],
            "process_type": data["process_type"],
            "H_QG_mean": data["result"]["H_mean"],
            "H_QG_std": data["result"]["H_std"],
            "H_QG_ci95_lo": data["result"]["H_ci95_lo"],
            "H_QG_ci95_hi": data["result"]["H_ci95_hi"],
            "abs_error": data["H_error"],
        }
        summary.append(row)
        print(
            f"  beta={beta:.1f}: H_exact={row['H_exact']:.3f}, "
            f"H_QG={row['H_QG_mean']:.3f}, |err|={row['abs_error']:.4f}"
        )

    out = RESULTS_DIR / f"exp2_results_{tag}.json"
    with open(out, "w") as f:
        json.dump({"lags": lags.tolist(), "results": summary}, f, indent=2, default=float)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 2: Hurst estimation from QG mean-jump scaling (fBm)")
    print(f"  T={T}, Q={Q}, N_real={N_REALIZATIONS}")
    print(f"  n_walks={N_WALKS}, walk_length={WALK_LENGTH}")
    print("=" * 60)

    print("\n-- Fig. 3 beta values --")
    run_experiment(BETA_VALUES_FIG3, tag="fig3")

    print("\n-- Fig. 4 extended beta grid --")
    run_experiment(BETA_VALUES_FIG4, tag="fig4")

    print("Experiment 2 complete.")

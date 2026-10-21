"""
Experiment 4: Extension of quantile-graph Hurst estimation to fractional
Gaussian noise by cumulative summation.

Tests the paper's proposed extension (Section 2, Methods; Section 4, Conclusions)
for applying the QG estimator to stationary fGn by first integrating each noise
series into a motion-like path.

Expected outcomes (Section 2, Methods):
- After cumsum, Delta(k) shows power-law scaling, enabling H estimation.
- White noise (H=0.5) after cumsum yields H_QG ≈ 0.5.
- Good agreement with exact H across the range (0,1).

Deviation from paper:
- Davies-Harte method used instead of Abry-Sellan wavelet synthesis.
- Random-walk length L=200 assumed (not specified by paper).
- H grid chosen by implementation (paper does not prescribe a grid for Fig.).
- No separate figure is provided in the paper for this extension numerically.
- No demeaning before cumsum (paper does not specify this step).
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

from data.generate_synthetic import generate_realizations, H_EPS, H_MAX
from src.hurst_estimator import build_lag_grid, estimate_hurst_ensemble

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Paper parameters
T: int = 10_000
Q: int = 10           # user override for speed (paper default: Q=50)
N_REALIZATIONS: int = 100
N_WALKS: int = 100
WALK_LENGTH: int = 200       # assumed
K_MIN: int = 1
K_MAX: int = 1000
N_LAGS: int = 20

# H grid for fGn; beta = 2H-1 in (-1,1)
H_VALUES_FGN = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def run_experiment(
    H_values: list[float] = H_VALUES_FGN,
    T: int = T,
    Q: int = Q,
    n_realizations: int = N_REALIZATIONS,
    n_walks: int = N_WALKS,
    walk_length: int = WALK_LENGTH,
) -> dict:
    """Run Experiment 4: QG estimation of fGn via cumulative summation.

    Returns
    -------
    dict mapping H -> result dict with H_QG estimate and diagnostics.
    """
    lags = build_lag_grid(K_MIN, K_MAX, N_LAGS)
    t0 = time.time()
    all_results = {}

    for H in H_values:
        H_clipped = float(np.clip(H, H_EPS, H_MAX))
        beta = 2.0 * H - 1.0
        print(f"  H={H:.2f} (beta={beta:+.2f}) ...", end=" ", flush=True)

        # Generate 100 fGn realizations
        fgn_list = generate_realizations(T, H_clipped, "fgn", n_realizations,
                                         base_seed=int(round(H * 1000)))

        # Cumulative sum to create motion-like paths (paper Section 2, Methods)
        # Cumsum starts at t=1: x_t = sum_{s=1}^t y_s, resulting in T values
        # First element of cumsum is y_1 (no zero prepended)
        cumsum_list = [np.cumsum(y) for y in fgn_list]

        # Apply QG estimator to integrated series
        result = estimate_hurst_ensemble(
            cumsum_list, Q, lags, n_walks, walk_length, base_seed=0
        )

        # Optional: apply QG directly to unintegrated fGn (diagnostic only)
        result_raw = estimate_hurst_ensemble(
            fgn_list, Q, lags, n_walks, walk_length, base_seed=10000
        )

        all_results[H] = {
            "H_exact": H_clipped,
            "beta": beta,
            "process_type": "fgn",
            "result_cumsum": result,
            "result_raw_fgn": result_raw,
            "H_error_cumsum": abs(result["H_mean"] - H_clipped),
        }
        print(
            f"H_QG_cumsum={result['H_mean']:.3f}±{result['H_std']:.3f}, "
            f"|err|={abs(result['H_mean'] - H_clipped):.4f}, "
            f"H_QG_raw={result_raw['H_mean']:.3f}"
        )

    elapsed = time.time() - t0
    print(f"  Experiment 4 completed in {elapsed:.1f}s")

    _plot_scatter(all_results)
    _plot_delta_comparison(all_results, lags)
    _save_results(all_results, lags)
    return all_results


def _plot_scatter(all_results: dict) -> None:
    """Scatter plot of H_QG vs H_exact for fGn via cumsum."""
    H_vals = sorted(all_results.keys())
    H_exact_arr = np.array([all_results[H]["H_exact"] for H in H_vals])
    H_cumsum = np.array([all_results[H]["result_cumsum"]["H_mean"] for H in H_vals])
    H_cumsum_lo = np.array([all_results[H]["result_cumsum"]["H_ci95_lo"] for H in H_vals])
    H_cumsum_hi = np.array([all_results[H]["result_cumsum"]["H_ci95_hi"] for H in H_vals])
    H_raw = np.array([all_results[H]["result_raw_fgn"]["H_mean"] for H in H_vals])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(
        H_exact_arr, H_cumsum,
        yerr=[H_cumsum - H_cumsum_lo, H_cumsum_hi - H_cumsum],
        fmt="o-", capsize=4, label="Cumsum fGn → H_QG"
    )
    ax.plot(H_exact_arr, H_raw, "s--", color="gray", alpha=0.6,
            label="Direct fGn → H_QG (diagnostic)")
    lim = [-0.1, 1.1]
    ax.plot(lim, lim, "k--", linewidth=1, label="Identity")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Exact H (fGn)")
    ax.set_ylabel("Estimated H_QG")
    ax.set_title(
        "QG Hurst estimation of fGn via cumulative summation (Exp. 4)\n"
        f"T={T}, Q={Q}, N={N_REALIZATIONS}"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Pearson correlation
    valid = np.isfinite(H_cumsum)
    if valid.sum() > 2:
        r, p = stats.pearsonr(H_exact_arr[valid], H_cumsum[valid])
        ax.text(0.05, 0.92, f"r={r:.4f}, p={p:.2e}",
                transform=ax.transAxes, fontsize=9)

    out = RESULTS_DIR / "exp4_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_delta_comparison(all_results: dict, lags: np.ndarray) -> None:
    """Show Delta(k) curves comparing cumsum vs raw fGn."""
    H_vals = sorted(all_results.keys())
    n = min(5, len(H_vals))
    selected = [H_vals[i] for i in np.linspace(0, len(H_vals) - 1, n, dtype=int)]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    titles = ["Cumsum fGn (proposed estimator)", "Raw fGn (diagnostic)"]
    result_keys = ["result_cumsum", "result_raw_fgn"]

    for ax, rkey, title in zip(axes, result_keys, titles):
        for H, color in zip(selected, colors):
            H_exact = all_results[H]["H_exact"]
            mean_delta = all_results[H][rkey]["mean_delta"]
            valid = mean_delta > 0
            if valid.sum() < 2:
                continue
            ax.loglog(
                lags[valid], mean_delta[valid],
                "o-", color=color, markersize=4,
                label=f"H={H_exact:.2f}"
            )
        ax.set_xlabel("Lag k")
        ax.set_ylabel("Mean jump Δ(k)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Δ(k) curves for fGn — cumsum vs raw (Exp. 4)\n"
        f"T={T}, Q={Q}, N={N_REALIZATIONS}",
        fontsize=10
    )
    out = RESULTS_DIR / "exp4_delta_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _save_results(all_results: dict, lags: np.ndarray) -> None:
    """Save JSON summary of Experiment 4 results."""
    summary = []
    for H in sorted(all_results.keys()):
        data = all_results[H]
        row = {
            "H_exact": data["H_exact"],
            "beta_fgn": data["beta"],
            "H_QG_cumsum_mean": data["result_cumsum"]["H_mean"],
            "H_QG_cumsum_std": data["result_cumsum"]["H_std"],
            "H_QG_cumsum_ci95_lo": data["result_cumsum"]["H_ci95_lo"],
            "H_QG_cumsum_ci95_hi": data["result_cumsum"]["H_ci95_hi"],
            "abs_error_cumsum": data["H_error_cumsum"],
            "H_QG_raw_fgn_mean": data["result_raw_fgn"]["H_mean"],
        }
        summary.append(row)

    out = RESULTS_DIR / "exp4_results.json"
    with open(out, "w") as f:
        json.dump({"lags": lags.tolist(), "results": summary}, f, indent=2, default=float)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 4: fGn via cumulative summation")
    print(f"  T={T}, Q={Q}, N_real={N_REALIZATIONS}")
    print(f"  n_walks={N_WALKS}, walk_length={WALK_LENGTH}")
    print(f"  H values: {H_VALUES_FGN}")
    print("=" * 60)
    run_experiment()
    print("Experiment 4 complete.")

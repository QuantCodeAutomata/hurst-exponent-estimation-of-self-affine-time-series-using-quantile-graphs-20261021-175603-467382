"""
Experiment 3: Sensitivity of quantile-graph Hurst estimates to the number
of quantiles Q.

Reproduces Fig. 5 of the paper by comparing absolute estimation errors
|H_QG - H| for Q=50 and Q=100 across a range of fBm spectral exponents.

Expected outcomes (Section 3, Results):
- Absolute errors below 0.02 for all tested beta values (both Q).
- Q=50 and Q=100 yield broadly similar errors, supporting weak Q-dependence.
- Fig. 5 values range approximately 0.0004–0.0172 (Q=50) and
  0.0014–0.0150 (Q=100).

Deviation from paper:
- Davies-Harte method used instead of Abry-Sellan wavelet synthesis.
- Random-walk length L=200 assumed (not specified by paper).
- Same realizations reused for Q=50 and Q=100 (paired comparison).
- Lag grid assumed (20 log-spaced points 1..1000).
- Fig. 5 beta grid inferred from paper's Fig. 5 caption.
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
from src.hurst_estimator import build_lag_grid, estimate_hurst_ensemble

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Paper parameters
T: int = 10_000
N_REALIZATIONS: int = 100
N_WALKS: int = 100
WALK_LENGTH: int = 200       # assumed
K_MIN: int = 1
K_MAX: int = 1000
N_LAGS: int = 20             # assumed

# Fig. 5: beta grid for fBm (beta = 2H+1, H = (beta-1)/2)
BETA_VALUES = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
Q_VALUES = [10, 20]          # user override for speed (paper: Q=50 and Q=100)


def run_experiment(
    beta_values: list[float] = BETA_VALUES,
    T: int = T,
    n_realizations: int = N_REALIZATIONS,
    n_walks: int = N_WALKS,
    walk_length: int = WALK_LENGTH,
) -> dict:
    """Run Experiment 3: compare absolute errors for Q=50 and Q=100.

    Returns
    -------
    dict mapping beta -> {Q: abs_error, 'H_exact': float, ...}
    """
    lags = build_lag_grid(K_MIN, K_MAX, N_LAGS)
    t0 = time.time()
    all_results = {}   # beta -> {50: result_dict, 100: result_dict, H_exact: float}

    for beta in beta_values:
        print(f"  beta={beta:.1f} ...", end=" ", flush=True)
        # Generate realizations once; reuse for both Q values
        series_list, H_exact, ptype = generate_from_beta(
            beta, T, n_realizations, base_seed=int(round(beta * 100))
        )

        q_results = {}
        for Q in Q_VALUES:
            result = estimate_hurst_ensemble(
                series_list, Q, lags, n_walks, walk_length, base_seed=0
            )
            q_results[Q] = result

        all_results[beta] = {
            "H_exact": H_exact,
            "process_type": ptype,
            "q_results": q_results,
        }
        errs = {Q: abs(q_results[Q]["H_mean"] - H_exact) for Q in Q_VALUES}
        err_str = ", ".join(f"|err|Q={Q}={errs[Q]:.4f}" for Q in Q_VALUES)
        print(f"H={H_exact:.3f}, {err_str}")

    elapsed = time.time() - t0
    print(f"  Experiment 3 completed in {elapsed:.1f}s")

    _plot_error_curves(all_results)
    _save_results(all_results, lags)
    return all_results


def _plot_error_curves(all_results: dict) -> None:
    """Reproduce Fig. 5: absolute error vs beta for each Q value."""
    betas = sorted(all_results.keys())
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    marker_cycle = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, Q in enumerate(Q_VALUES):
        errors = [
            abs(all_results[b]["q_results"][Q]["H_mean"] - all_results[b]["H_exact"])
            for b in betas
        ]
        ax.plot(
            betas, errors,
            marker=marker_cycle[idx % len(marker_cycle)],
            linestyle="-", color=palette[idx % len(palette)],
            label=f"Q={Q}"
        )

    ax.axhline(0.02, color="gray", linestyle=":", linewidth=1, label="|err|=0.02 threshold")
    ax.set_xlabel("β (spectral exponent)")
    ax.set_ylabel("|H_QG − H|")
    q_label = " and ".join(f"Q={q}" for q in Q_VALUES)
    ax.set_title(
        f"Absolute Hurst estimation error vs β for {q_label} (Exp. 3, Fig. 5)\n"
        f"T={T}, N={N_REALIZATIONS}, n_walks={N_WALKS}"
    )
    ax.legend()
    ax.grid(True, alpha=0.4)

    out = RESULTS_DIR / "exp3_q_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _save_results(all_results: dict, lags: np.ndarray) -> None:
    """Save JSON summary of Experiment 3."""
    summary = []
    for beta in sorted(all_results.keys()):
        H_exact = all_results[beta]["H_exact"]
        row = {"beta": beta, "H_exact": H_exact}
        for Q in Q_VALUES:
            r = all_results[beta]["q_results"][Q]
            row[f"H_QG_Q{Q}"] = r["H_mean"]
            row[f"abs_error_Q{Q}"] = abs(r["H_mean"] - H_exact)
            row[f"std_Q{Q}"] = r["H_std"]
        summary.append(row)
        err_str = ", ".join(f"|err|{Q}={row[f'abs_error_Q{Q}']:.4f}" for Q in Q_VALUES)
        print(f"  beta={beta:.1f}: H={H_exact:.3f}, {err_str}")

    out = RESULTS_DIR / "exp3_results.json"
    with open(out, "w") as f:
        json.dump({"lags": lags.tolist(), "results": summary}, f, indent=2, default=float)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 3: Q-sensitivity of Hurst estimation")
    print(f"  T={T}, Q_values={Q_VALUES}, N_real={N_REALIZATIONS}")
    print(f"  n_walks={N_WALKS}, walk_length={WALK_LENGTH}")
    print(f"  Beta values: {BETA_VALUES}")
    print("=" * 60)
    run_experiment()
    print("Experiment 3 complete.")

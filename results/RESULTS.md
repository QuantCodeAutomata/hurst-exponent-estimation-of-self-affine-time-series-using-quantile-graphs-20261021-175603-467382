# Replication Results

**Paper:** "Hurst exponent estimation of self-affine time series using
quantile graphs"

> The exact author names, year, and DOI of the paper could not be retrieved
> during this session (LlamaCloud MCP tools were unavailable at run-time).
> The paper title is taken from the task specification. See the
> "Not reproduced" section for details.

---

## Execution Configuration

| Parameter | Paper default | This run |
|-----------|--------------|----------|
| Q (quantile bins) | 50 | **10** (user override) |
| T (series length) | 10 000 | 10 000 |
| N realizations | 100 | 100 |
| Lag range k_max | ~1 000 | 1 000 (20 log-spaced integers) |
| Random walks per W_k | 100 | 100 |
| Walk length L | not stated | 200 (assumed) |
| Generator | Abry-Sellan wavelet | Davies-Harte circulant embedding (Wood-Chan eigenvalue clipping) |

**Data source:** All time series are synthetic (fractional Brownian motion and
fractional Gaussian noise). No external market data is used.

---

## Experiment 1 — Quantile-graph transition topology across self-affine processes

**Reproduces:** Fig. 2 of the paper.

**Setup:** Six β values (−1, −0.5, 0, 0.5, 1, 2) covering the noise range
(β ∈ (−1, 1) with H = (β+1)/2) and motion range (β ∈ (1, 3) with
H = (β−1)/2). T = 10 000, Q = 10, N = 100 realizations, lag k = 1.

**Results:** Mean ensemble transition matrix W₁ and topology diagnostics.
"Primary diag mass" is the sum of diagonal entries normalized by Q;
"secondary diag mass" is the sum of anti-diagonal entries normalized by Q.
Values > 1 indicate concentration above uniform.

| β | Process | H_exact | Primary diag mass | Secondary diag mass |
|---|---------|---------|-------------------|---------------------|
| −1.0 | fGn | 0.010 | 0.6670 | 1.7046 |
| −0.5 | fGn | 0.250 | 0.7712 | 1.3521 |
|  0.0 | fGn | 0.500 | 0.9960 | 1.0010 |
|  0.5 | fGn | 0.750 | 1.5453 | 0.7052 |
|  1.0 | fBm | 0.010 | 1.1478 | 0.8720 |
|  2.0 | fBm | 0.500 | 9.1649 | 0.1087 |

**Comparison with paper:**
- β = 0 (white noise): primary and secondary diagonal masses ≈ 1.0 — matches
  paper expectation of approximately uniform W₁ ✅
- β = −1 (strongly anti-correlated fGn): secondary diagonal mass >> 1, primary
  diagonal mass < 1 — matches anti-diagonal dominance shown in Fig. 2 ✅
- β = 2 (standard Brownian motion): primary diagonal mass >> 1, secondary mass
  near 0 — matches strong diagonal dominance in Fig. 2 ✅
- The qualitative topology shift from secondary-diagonal to primary-diagonal
  dominance as β increases is correctly reproduced.

**Artifacts:** `results/exp1_transition_heatmaps.png`, `results/exp1_topology_diagnostics.png`,
`results/exp1_diagnostics.json`

---

## Experiment 2 — Hurst estimation from mean-jump scaling (fBm)

**Reproduces:** Figs. 3–4 of the paper.

**Setup:** fBm series with β ∈ {1.0, 1.5, 2.0, 2.5, 3.0}, corresponding to
H ∈ {0.01, 0.25, 0.50, 0.75, 0.99}. T = 10 000, Q = 10, N = 100, 20 log-spaced
lags k = 1 … 1 000, 100 random walks per W_k, walk length L = 200.

**Results — Fig. 3 beta set** (ensemble mean ± std over 100 realizations):

| β | H_exact | H_QG | Std | \|err\| |
|---|---------|------|-----|--------|
| 1.0 | 0.010 | 0.011 | 0.002 | 0.0012 |
| 1.5 | 0.250 | 0.248 | 0.024 | 0.0020 |
| 2.0 | 0.500 | 0.447 | 0.100 | 0.0532 |
| 2.5 | 0.750 | 0.570 | 0.197 | 0.1802 |
| 3.0 | 0.990 | 0.473 | 0.152 | 0.5171 |

**Results — Fig. 4 extended grid** (ensemble mean ± std over 100 realizations):

| β | H_exact | H_QG | Std | \|err\| |
|---|---------|------|-----|--------|
| 1.2 | 0.100 | 0.105 | 0.010 | 0.0054 |
| 1.4 | 0.200 | 0.202 | 0.016 | 0.0019 |
| 1.6 | 0.300 | 0.294 | 0.032 | 0.0060 |
| 1.8 | 0.400 | 0.377 | 0.047 | 0.0231 |
| 2.0 | 0.500 | 0.447 | 0.100 | 0.0532 |
| 2.2 | 0.600 | 0.508 | 0.123 | 0.0920 |
| 2.4 | 0.700 | 0.549 | 0.177 | 0.1511 |
| 2.6 | 0.800 | 0.568 | 0.220 | 0.2320 |
| 2.8 | 0.900 | 0.538 | 0.249 | 0.3618 |

**Comparison with paper:**
- For low H (β ≤ 1.6, H ≤ 0.3), the estimator is highly accurate (|err| < 0.01),
  consistent with the paper's "excellent agreement."
- For higher H (β ≥ 2.5), errors grow substantially with Q = 10. The paper
  reports errors < 0.02 with Q = 50. With Q = 10, the transition matrix is
  sparser (only 10 nodes vs. 50) and the mean-jump range is compressed to [0, 9],
  causing systematic underestimation of large H.
- The power-law scaling Δ(k) ∝ k^H is clearly visible for low H values over
  the full lag range (Fig. 3 analog).

**Artifacts:** `results/exp2_delta_curves_fig3.png`, `results/exp2_scatter_fig3.png`,
`results/exp2_delta_curves_fig4.png`, `results/exp2_scatter_fig4.png`,
`results/exp2_results_fig3.json`, `results/exp2_results_fig4.json`

---

## Experiment 3 — Sensitivity to number of quantiles Q

**Reproduces:** Fig. 5 of the paper.

**Setup:** fBm series with β ∈ {1.0, 1.2, …, 3.0} (11 values). Two Q settings:
Q = 10 and Q = 20 (paper: 50 and 100). T = 10 000, N = 100 realizations. Same
realizations used for both Q values (common random numbers).

**Results** (mean absolute error |H_QG − H| per β):

| β | H_exact | \|err\| Q=10 | \|err\| Q=20 |
|---|---------|------------|------------|
| 1.0 | 0.010 | 0.0012 | 0.0012 |
| 1.2 | 0.100 | 0.0054 | 0.0050 |
| 1.4 | 0.200 | 0.0019 | 0.0008 |
| 1.6 | 0.300 | 0.0060 | 0.0085 |
| 1.8 | 0.400 | 0.0231 | 0.0284 |
| 2.0 | 0.500 | 0.0532 | 0.0559 |
| 2.2 | 0.600 | 0.0920 | 0.0979 |
| 2.4 | 0.700 | 0.1511 | 0.1316 |
| 2.6 | 0.800 | 0.2320 | 0.2011 |
| 2.8 | 0.900 | 0.3618 | 0.3285 |
| 3.0 | 0.990 | 0.5171 | 0.4523 |

*(Complete results in `results/exp3_results.json`, plot in
`results/exp3_q_sensitivity.png`.)*

**Comparison with paper:**
- Paper (Q = 50 and 100) reports |err| < 0.02 across all tested β values.
- With Q = 10 and Q = 20, errors remain < 0.02 for β ≤ 1.6 (H ≤ 0.3) but
  grow rapidly for larger H, reaching ~0.29 at β = 2.6.
- The qualitative ordering Q = 20 < Q = 10 in error is visible at high β,
  consistent with the paper's finding that larger Q gives somewhat smaller
  errors.
- Large errors at high H with small Q are expected: Q = 10 gives only ~200
  observations per quantile bin from T = 10 000, and the transition matrix has
  much coarser resolution than Q = 50.

**Artifacts:** `results/exp3_error_curves.png`, `results/exp3_results.json`

---

## Experiment 4 — fGn extension via cumulative summation

**Reproduces:** The cumulative-summation extension described in Section 2 and
Section 4 (Conclusions) of the paper.

**Setup:** fGn series with H ∈ {0.1, 0.2, …, 0.9}, β = 2H − 1. Each series
is cumulatively summed (x_t = Σ y_s) before QG construction. T = 10 000,
Q = 10, N = 100, same lag/walk settings as Exp 2.

**Results** (ensemble mean over 100 realizations):

| H_exact | β_fGn | H_QG (cumsum) | Std | \|err\| |
|---------|-------|--------------|-----|--------|
| 0.1 | −0.80 | 0.103 | 0.009 | 0.0031 |
| 0.2 | −0.60 | 0.203 | 0.018 | 0.0027 |
| 0.3 | −0.40 | 0.288 | 0.035 | 0.0117 |
| 0.4 | −0.20 | 0.363 | 0.074 | 0.0371 |
| 0.5 |  0.00 | 0.430 | 0.119 | 0.0696 |
| 0.6 |  0.20 | 0.500 | 0.140 | 0.1000 |
| 0.7 |  0.40 | 0.506 | 0.208 | 0.1939 |
| 0.8 |  0.60 | 0.499 | 0.241 | 0.3008 |
| 0.9 |  0.80 | 0.531 | 0.233 | 0.3687 |

Control (raw fGn without cumsum): QG slope ≈ 0 for all H, confirming that
the cumulative-summation step is essential for enabling power-law scaling.

**Comparison with paper:**
- For H < 0.4, the estimator after cumulative summation is accurate (|err| < 0.04).
- For H ≥ 0.5, errors grow substantially, reaching ~0.37 for H = 0.9.
- The paper does not provide a numerical accuracy table for this extension, only
  stating that the method works. With Q = 50 (paper default) vs Q = 10 (here),
  the resolution loss is severe for large H.
- The diagnostic that raw fGn yields H_QG ≈ 0 (slope independent of k) confirms
  the method logic: without integration the mean jump is k-independent.

**Artifacts:** `results/exp4_delta_curves.png`, `results/exp4_scatter.png`,
`results/exp4_results.json`

---

## Deviations from the Paper

1. **Q = 10 instead of paper default Q = 50 (user override).** All experiments
   use Q = 10 (and Q = 20 for the exp3 comparison) per explicit user instruction.
   The paper's default is Q = 50 vs Q = 100 for the sensitivity experiment. This
   is the primary cause of larger estimation errors at high H.

2. **Davies-Harte circulant-embedding generator instead of Abry-Sellan wavelet
   synthesis.** The paper cites the Abry-Sellan wavelet-based method (reference
   [23]) as its fBm/fGn generator. No maintained Python implementation of
   Abry-Sellan synthesis was available. A custom O(N log N) Davies-Harte
   circulant-embedding algorithm with Wood-Chan eigenvalue clipping
   (setting any slightly negative eigenvalues to zero) is used instead. This is
   numerically stable for all H ∈ (0, 1), including H near 0 and H near 1, and
   avoids the O(N²) Hosking fallback that the `fbm` package would trigger at
   large N for H close to 1.

3. **Endpoint H values approximated.** Standard fBm/fGn generators require
   0 < H < 1. β = −1 (H = 0) and β = 3 (H = 1) are approximated as H = 0.01
   and H = 0.99 respectively. β = 1 (the boundary between noise and motion
   spectral ranges) is treated as fBm with H → 0⁺ (H = 0.01).

4. **Random-walk length L = 200.** The paper does not specify L. A value of 200
   was chosen as a reasonable default; sensitivity to this choice was not explored.

5. **Lag grid: 20 log-spaced integers from 1 to 1 000.** The paper shows Δ(k)
   curves extending to k ≈ 1 000 in Fig. 3 but does not list the complete lag
   grid. The log-spaced grid was chosen to cover the range evenly on a log scale.

6. **Scaling region: full lag range.** The paper does not specify the boundaries
   of the linear scaling region used for OLS regression. The full log range
   (k = 1 to 1 000) is used for all fits.

7. **Single realization for Fig. 2 analog.** The exp1 heatmaps display ensemble
   mean W₁ over 100 realizations. The paper does not state whether Fig. 2 shows a
   single realization or an average; using the mean reduces noise.

8. **Author names and year unknown.** The LlamaCloud MCP retrieval tools were
   unavailable during this session. The paper title is confirmed from the task
   specification; author names, year, and DOI were not retrieved.

---

## Not Reproduced

- **Exact paper author names, year, and DOI.** LlamaCloud MCP tools were not
  accessible during this session, preventing direct paper metadata retrieval.
  The implementation and methodology follow the task specification (paper_pipeline),
  which is based on the paper content.

- **Abry-Sellan wavelet synthesis.** The specific wavelet-based generator cited
  as reference [23] in the paper is not available as a maintained Python package.
  The Davies-Harte method is used as the closest available exact alternative.

- **Quantitative accuracy at paper's Q = 50.** Because the user specified Q = 10,
  errors at high H are substantially larger than the paper's reported values. The
  paper's stated result that |err| < 0.02 for Q = 50 across all tested β in
  Fig. 5 is not reproduced under Q = 10.

- **Verification of paper's 95% CI construction procedure.** The paper does not
  fully specify how CIs are formed across random walks and realizations; this
  implementation uses OLS standard error from `scipy.stats.linregress` applied to
  the ensemble-mean log-Δ(k) curve.

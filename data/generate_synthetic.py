"""
Synthetic self-affine time-series generation for the Hurst-exponent paper.

The paper cites the Abry-Sellan wavelet-based synthesis method (Reference [23]).
This implementation uses the Davies-Harte (exact FFT-based circulant-embedding)
algorithm with the Wood-Chan eigenvalue-clipping stabilisation, because the
Abry-Sellan algorithm is not available in a maintained Python library.  This
constitutes a deviation from the paper's cited generator; it is recorded in
results/RESULTS.md.

Series classification follows Section 2 of the paper:
  - Noise range  (beta in (-1, 1)):  beta = 2H - 1  =>  H = (beta + 1) / 2
  - Motion range (beta in (1, 3)):   beta = 2H + 1  =>  H = (beta - 1) / 2
  - beta = 1 is the boundary; treated here as fBm with H -> 0+ (H_eps = 0.01)

Standard fBm/fGn generators require 0 < H < 1.  The endpoint beta values
(-1 and 3, corresponding to H = 0 and H = 1) are approximated as H = 0.01
and H = 0.99 respectively.  This approximation is documented in RESULTS.md.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# Minimum and maximum H simulated by the circulant-embedding generator.
# H very close to 0 or 1 is numerically stable with eigenvalue clipping,
# but we keep a small margin to avoid degenerate cases.
H_EPS: float = 0.01
H_MAX: float = 0.99


def _fgn_circulant(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate fractional Gaussian noise of length *n* using Davies-Harte.

    Uses the circulant-embedding method (Wood & Chan 1994) with eigenvalue
    clipping (setting min-negative eigenvalues to 0) for numerical stability
    at H close to 1 or 0.  This avoids the O(N²) Hosking fallback while
    keeping O(N log N) complexity.

    Parameters
    ----------
    n:
        Number of fGn increments to return.
    H:
        Hurst exponent in (0, 1).
    rng:
        Seeded numpy random Generator.

    Returns
    -------
    fgn:
        Array of shape (n,), stationary fGn with unit variance at lag 0.
    """
    # Autocovariance of unit-variance fGn at lags 0..n:
    #   γ(k) = (|k-1|^{2H} - 2*k^{2H} + (k+1)^{2H}) / 2
    k_arr = np.arange(n + 1, dtype=np.float64)
    cov_seq = 0.5 * (
        np.abs(k_arr - 1) ** (2.0 * H)
        - 2.0 * k_arr ** (2.0 * H)
        + (k_arr + 1.0) ** (2.0 * H)
    )
    # Circulant first row of size 2n: [γ(0), γ(1), ..., γ(n), γ(n-1), ..., γ(1)]
    row = np.empty(2 * n, dtype=np.float64)
    row[:n + 1] = cov_seq
    row[n + 1:] = cov_seq[n - 1:0:-1]

    # Eigenvalues via FFT; clip slightly negative ones (Wood-Chan stabilisation)
    eigenvalues = np.real(np.fft.fft(row))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sqrt_eig = np.sqrt(eigenvalues)

    # Draw 2n independent N(0,1) samples
    gn = rng.standard_normal(2 * n)

    # Build Hermitian symmetric vector W (see Davies & Harte 1987, eq. 2.4)
    W = np.empty(2 * n, dtype=np.complex128)
    W[0] = sqrt_eig[0] * gn[0]
    W[n] = sqrt_eig[n] * gn[n]
    # Positive-frequency components: W[i] = sqrt_eig[i]*(g_i + j*g_{n+i})/√2
    W[1:n] = sqrt_eig[1:n] * (gn[1:n] + 1j * gn[n + 1: 2 * n]) / np.sqrt(2.0)
    # Negative-frequency components: Hermitian conjugate
    W[n + 1:] = np.conj(W[1:n][::-1])

    # IFFT and scale (equivalent to real part of FFT(W)/sqrt(2n))
    fgn = np.real(np.fft.fft(W)) / np.sqrt(2.0 * n)
    return fgn[:n]


def beta_to_H_and_type(beta: float) -> tuple[float, Literal["fgn", "fbm"]]:
    """Convert spectral exponent beta to Hurst exponent H and process type.

    Classification and conversion follow Section 2 of the paper:
      - Noise range -1 < beta < 1: H = (beta + 1) / 2  (fGn)
      - Motion range 1 < beta < 3: H = (beta - 1) / 2  (fBm)
      - beta = 1: boundary; approximated as fBm with H = H_EPS
      - Endpoint beta = -1 -> H = 0 approximated as H_EPS  (fGn)
      - Endpoint beta =  3 -> H = 1 approximated as H_MAX  (fBm)

    Parameters
    ----------
    beta:
        Self-affine spectral exponent.

    Returns
    -------
    H:
        Hurst exponent clipped to (H_EPS, H_MAX).
    process_type:
        "fgn" for fractional Gaussian noise or "fbm" for fractional
        Brownian motion.
    """
    if beta < 1.0:
        H = (beta + 1.0) / 2.0
        ptype: Literal["fgn", "fbm"] = "fgn"
    elif beta == 1.0:
        # Boundary: classify as fBm with H -> 0+
        H = H_EPS
        ptype = "fbm"
    else:  # beta > 1
        H = (beta - 1.0) / 2.0
        ptype = "fbm"

    H_clipped = float(np.clip(H, H_EPS, H_MAX))
    return H_clipped, ptype


def generate_series(
    T: int,
    H: float,
    process_type: Literal["fgn", "fbm"],
    seed: int | None = None,
) -> np.ndarray:
    """Generate a single self-affine time series of length T.

    Parameters
    ----------
    T:
        Number of data points.
    H:
        Hurst exponent in (0, 1); clipped to (H_EPS, H_MAX) internally.
    process_type:
        "fgn" for stationary fractional Gaussian noise,
        "fbm" for fractional Brownian motion path.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    x:
        Time series of length T.

    Notes
    -----
    Uses the Davies-Harte circulant-embedding algorithm with Wood-Chan
    eigenvalue clipping for O(N log N) simulation at all H in (0, 1).
    """
    H = float(np.clip(H, H_EPS, H_MAX))
    rng = np.random.default_rng(seed)

    fgn = _fgn_circulant(T, H, rng)
    if process_type == "fbm":
        return np.cumsum(fgn)
    return fgn


def generate_realizations(
    T: int,
    H: float,
    process_type: Literal["fgn", "fbm"],
    n_realizations: int = 100,
    base_seed: int = 0,
) -> list[np.ndarray]:
    """Generate multiple independent realizations of a self-affine series.

    Parameters
    ----------
    T:
        Length of each series.
    H:
        Hurst exponent.
    process_type:
        "fgn" or "fbm".
    n_realizations:
        Number of independent series to generate.
    base_seed:
        Realization i gets seed = base_seed + i.

    Returns
    -------
    List of n_realizations arrays, each of length T.
    """
    return [
        generate_series(T, H, process_type, seed=base_seed + i)
        for i in range(n_realizations)
    ]


def generate_from_beta(
    beta: float,
    T: int,
    n_realizations: int = 100,
    base_seed: int = 0,
) -> tuple[list[np.ndarray], float, str]:
    """Generate realizations from a spectral exponent beta.

    Parameters
    ----------
    beta:
        Spectral exponent.
    T:
        Series length.
    n_realizations:
        Number of realizations.
    base_seed:
        Base seed for reproducibility.

    Returns
    -------
    series_list:
        List of n_realizations arrays.
    H:
        Hurst exponent used.
    process_type:
        "fgn" or "fbm".
    """
    H, ptype = beta_to_H_and_type(beta)
    series = generate_realizations(T, H, ptype, n_realizations, base_seed)
    return series, H, ptype

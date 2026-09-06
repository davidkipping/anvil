"""Convergence diagnostics: split rank-normalized R-hat and bulk ESS.

Implements the Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021)
recipes on host-side numpy in float64 (diagnostics are cheap after the
chain is reduced to stored draws). Self-contained: the normal quantile
function uses Acklam's rational approximation rather than pulling in scipy.

All functions take chains shaped ``(n_steps, n_chains, dim)`` (or
``(n_steps, n_chains)`` for a single parameter) as produced by
``Results.get_chain()``.
"""

from __future__ import annotations

import numpy as np

# --- Acklam's inverse normal CDF (max rel. error ~1.15e-9) -----------------

_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00)


def norm_ppf(p: np.ndarray) -> np.ndarray:
    """Vectorized standard-normal quantile function (Acklam)."""
    p = np.asarray(p, dtype=np.float64)
    out = np.empty_like(p)
    plow, phigh = 0.02425, 1 - 0.02425

    lo = p < plow
    hi = p > phigh
    mid = ~(lo | hi)

    if np.any(mid):
        q = p[mid] - 0.5
        r = q * q
        num = ((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]
        den = ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        out[mid] = num * q / den
    if np.any(lo):
        q = np.sqrt(-2.0 * np.log(p[lo]))
        num = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        out[lo] = num / den
    if np.any(hi):
        q = np.sqrt(-2.0 * np.log1p(-p[hi]))
        num = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        out[hi] = -num / den
    return out


# --- rank normalization -----------------------------------------------------

def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Fractional-rank normal scores over ALL draws, per parameter.

    x: (n_steps, n_chains). Returns same shape, z-scored via ranks.
    """
    flat = x.reshape(-1)
    ranks = np.empty_like(flat)
    order = np.argsort(flat, kind="stable")
    ranks[order] = np.arange(1, flat.size + 1, dtype=np.float64)
    p = (ranks - 3.0 / 8.0) / (flat.size + 1.0 / 4.0)
    return norm_ppf(p).reshape(x.shape)


def _split_chains(x: np.ndarray) -> np.ndarray:
    """Split each chain in half: (N, M) -> (N//2, 2M). Drops an odd step."""
    n = (x.shape[0] // 2) * 2
    first, second = x[: n // 2], x[n // 2 : n]
    return np.concatenate([first, second], axis=1)


# --- R-hat ------------------------------------------------------------------

def _rhat_single(x: np.ndarray) -> float:
    """Basic (non-split) R-hat on (N, M)."""
    n, m = x.shape
    chain_means = x.mean(axis=0)
    b = n * chain_means.var(ddof=1)
    w = x.var(axis=0, ddof=1).mean()
    var_plus = (n - 1) / n * w + b / n
    if w <= 0:
        return np.inf
    return float(np.sqrt(var_plus / w))


def split_rhat(chain: np.ndarray) -> np.ndarray:
    """Split rank-normalized R-hat per parameter. chain: (N, M[, dim])."""
    single = chain.ndim == 2
    if single:
        chain = chain[..., None]
    out = np.empty(chain.shape[-1])
    for d in range(chain.shape[-1]):
        z = _rank_normalize(chain[..., d].astype(np.float64))
        out[d] = _rhat_single(_split_chains(z))
    return out[0:1] if single else out


# --- ESS --------------------------------------------------------------------

def _autocov(x: np.ndarray) -> np.ndarray:
    """Per-chain autocovariance via FFT. x: (N, M) -> (N, M), biased (1/N)."""
    n, _ = x.shape
    xc = x - x.mean(axis=0)
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(xc, n=nfft, axis=0)
    acov = np.fft.irfft(f * np.conjugate(f), n=nfft, axis=0)[:n].real
    return acov / n


def _ess_single(x: np.ndarray) -> float:
    """Multi-chain bulk ESS on already rank-normalized, split chains (N, M),
    Stan-style with Geyer initial monotone sequence."""
    n, m = x.shape
    if n < 4:
        return float("nan")
    acov = _autocov(x)
    chain_var = acov[0] * n / (n - 1)
    w = chain_var.mean()
    var_plus = w * (n - 1) / n
    if m > 1:
        var_plus += x.mean(axis=0).var(ddof=1)
    if var_plus <= 0:
        return float("nan")

    rho = 1.0 - (w - acov.mean(axis=1)) / var_plus  # (N,)
    rho[0] = 1.0

    # Geyer: sum pair terms P_k = rho_{2k} + rho_{2k+1}; truncate at the
    # first negative pair; enforce a monotone non-increasing sequence.
    # tau = -1 + 2 * sum_k P_k   (since sum_k P_k = 1 + sum_{t>=1} rho_t).
    sum_pairs = 0.0
    prev_pair = np.inf
    for k in range(n // 2):
        pair = rho[2 * k] + rho[2 * k + 1]
        if pair < 0:
            break
        pair = min(pair, prev_pair)
        sum_pairs += pair
        prev_pair = pair
    tau = max(-1.0 + 2.0 * sum_pairs, 1.0 / np.log10(n * m + 10.0))
    return float(n * m / tau)


def ess_bulk(chain: np.ndarray) -> np.ndarray:
    """Rank-normalized bulk ESS per parameter. chain: (N, M[, dim])."""
    single = chain.ndim == 2
    if single:
        chain = chain[..., None]
    out = np.empty(chain.shape[-1])
    for d in range(chain.shape[-1]):
        z = _rank_normalize(chain[..., d].astype(np.float64))
        out[d] = _ess_single(_split_chains(z))
    return out[0:1] if single else out


def nested_rhat(chain: np.ndarray, n_superchains: int) -> np.ndarray:
    """Nested R-hat (Margossian et al. 2022) for the many-short-chains
    regime, where per-chain draws are too few for classic split R-hat.

    Chains are grouped contiguously into ``n_superchains`` groups (group
    chains that shared an initialization). chain: (N, M[, dim]) with M
    divisible by n_superchains. Values near 1 indicate the superchains
    agree; the diagnostic works even with a handful of draws per chain.
    """
    single = chain.ndim == 2
    if single:
        chain = chain[..., None]
    n, m, dim = chain.shape
    k = n_superchains
    if m % k:
        raise ValueError(f"n_chains={m} not divisible by n_superchains={k}")
    j = m // k
    x = chain.astype(np.float64).reshape(n, k, j, dim)

    chain_mean = x.mean(axis=0)                      # (K, J, dim)
    chain_var = x.var(axis=0, ddof=1) if n > 1 else np.zeros_like(chain_mean)
    super_mean = chain_mean.mean(axis=1)             # (K, dim)
    grand_mean = super_mean.mean(axis=0)             # (dim,)

    b_hat = ((super_mean - grand_mean) ** 2).mean(axis=0)          # (dim,)
    b_within = ((chain_mean - super_mean[:, None, :]) ** 2).mean(axis=1)
    w_within = chain_var.mean(axis=1)
    w_hat = (b_within + w_within).mean(axis=0)                     # (dim,)

    out = np.sqrt(1.0 + b_hat / np.maximum(w_hat, 1e-300))
    return out[0:1] if single else out


def summary(chain: np.ndarray, names: list[str] | None = None) -> str:
    """Human-readable per-parameter table: mean, sd, R-hat, bulk ESS."""
    if chain.ndim == 2:
        chain = chain[..., None]
    dim = chain.shape[-1]
    names = names or [f"p{d}" for d in range(dim)]
    rhat = split_rhat(chain)
    ess = ess_bulk(chain)
    lines = [f"{'param':>10s} {'mean':>12s} {'sd':>12s} {'rhat':>8s} {'ess_bulk':>10s}"]
    for d in range(dim):
        draws = chain[..., d]
        lines.append(
            f"{names[d]:>10s} {draws.mean():>12.5g} {draws.std():>12.5g} "
            f"{rhat[d]:>8.4f} {ess[d]:>10.0f}"
        )
    return "\n".join(lines)

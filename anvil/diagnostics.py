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

from dataclasses import dataclass

import mlx.core as mx
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


def _rank_normalize_all(chain: np.ndarray) -> np.ndarray:
    """Rank-normalize every parameter at once.

    Identical in intent to :func:`_rank_normalize`, but vectorized over
    parameters and executed through MLX, which makes the sort (~85% of the
    cost of the whole diagnostics suite) a GPU operation. Ranks are exact:
    they are integers below 2**24 for any sample this engine can store, so
    float32 carries them without loss; the only difference from the numpy
    path is the float32 evaluation of the normal quantile function, worth
    ~1e-6 in R-hat.

    chain: (N, M, dim) -> (N, M, dim) normal scores.
    """
    n, m, dim = chain.shape
    flat = mx.array(np.ascontiguousarray(
        chain.reshape(n * m, dim), dtype=np.float32))
    order = mx.argsort(flat, axis=0)
    pos = mx.broadcast_to(
        mx.arange(1, n * m + 1, dtype=mx.float32)[:, None], flat.shape)
    ranks = mx.put_along_axis(mx.zeros_like(flat), order, pos, axis=0)
    p = (ranks - 0.375) / (n * m + 0.25)
    return np.array(_norm_ppf_mx(p), dtype=np.float64).reshape(n, m, dim)


def _norm_ppf_mx(p: mx.array) -> mx.array:
    """Acklam's inverse normal CDF in MLX ops (see :func:`norm_ppf`)."""
    lo, hi = 0.02425, 1.0 - 0.02425
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    num_m = ((((_A[0]*r_mid + _A[1])*r_mid + _A[2])*r_mid + _A[3])*r_mid
             + _A[4])*r_mid + _A[5]
    den_m = ((((_B[0]*r_mid + _B[1])*r_mid + _B[2])*r_mid + _B[3])*r_mid
             + _B[4])*r_mid + 1.0
    mid = num_m * q_mid / den_m

    # both tails are evaluated everywhere, so their arguments must be
    # sanitized to stay in-domain (mx.where evaluates both branches)
    p_lo = mx.where(p < lo, p, 0.01)
    q_lo = mx.sqrt(-2.0 * mx.log(p_lo))
    p_hi = mx.where(p > hi, p, 0.99)
    q_hi = mx.sqrt(-2.0 * mx.log1p(-p_hi))

    def _tail(q):
        num = ((((_C[0]*q + _C[1])*q + _C[2])*q + _C[3])*q + _C[4])*q + _C[5]
        den = (((_D[0]*q + _D[1])*q + _D[2])*q + _D[3])*q + 1.0
        return num / den

    out = mx.where(p < lo, _tail(q_lo), mid)
    return mx.where(p > hi, -_tail(q_hi), out)


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
    z = _rank_normalize_all(chain)
    out = np.empty(chain.shape[-1])
    for d in range(chain.shape[-1]):
        out[d] = _rhat_single(_split_chains(z[..., d]))
    return out[0:1] if single else out


# --- ESS --------------------------------------------------------------------

def _autocov(x: np.ndarray) -> np.ndarray:
    """Per-chain autocovariance via FFT along axis 0, biased (1/N).

    Shape-agnostic beyond the first axis, so it batches over parameters:
    (N, M) -> (N, M) and (N, M, dim) -> (N, M, dim).
    """
    n = x.shape[0]
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    if x.size >= 1 << 18:
        # GPU float32 for large arrays: measured ~4x faster, and it moves
        # the resulting ESS by ~5e-9 relative (the autocovariance is an
        # average over M chains, so float32 noise averages away)
        xm = mx.array(np.ascontiguousarray(x, dtype=np.float32))
        xc = xm - mx.mean(xm, axis=0)
        f = mx.fft.rfft(xc, n=nfft, axis=0)
        acov = mx.fft.irfft(f * mx.conjugate(f), n=nfft, axis=0)[:n]
        return np.array(acov, dtype=np.float64) / n
    xc = x - x.mean(axis=0)
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


def _ess_batched(x: np.ndarray) -> np.ndarray:
    """Bulk ESS for every parameter at once. x: (N, M, dim), split chains.

    Same Geyer initial monotone sequence as :func:`_ess_single`; only the
    autocovariance is batched (one FFT over all parameters instead of one
    per parameter).
    """
    n, m, dim = x.shape
    if n < 4:
        return np.full(dim, np.nan)
    acov = _autocov(x)                       # (N, M, dim), FFT along axis 0
    w = (acov[0] * n / (n - 1)).mean(axis=0)                 # (dim,)
    var_plus = w * (n - 1) / n
    if m > 1:
        var_plus = var_plus + x.mean(axis=0).var(axis=0, ddof=1)
    rho = 1.0 - (w - acov.mean(axis=1)) / var_plus            # (N, dim)
    rho[0] = 1.0

    out = np.empty(dim)
    for d in range(dim):
        if var_plus[d] <= 0:
            out[d] = np.nan
            continue
        sum_pairs, prev_pair = 0.0, np.inf
        for k in range(n // 2):
            pair = rho[2 * k, d] + rho[2 * k + 1, d]
            if pair < 0:
                break
            pair = min(pair, prev_pair)
            sum_pairs += pair
            prev_pair = pair
        tau = max(-1.0 + 2.0 * sum_pairs, 1.0 / np.log10(n * m + 10.0))
        out[d] = n * m / tau
    return out


def ess_bulk(chain: np.ndarray) -> np.ndarray:
    """Rank-normalized bulk ESS per parameter. chain: (N, M[, dim])."""
    single = chain.ndim == 2
    if single:
        chain = chain[..., None]
    z = _rank_normalize_all(chain)
    zs = np.concatenate(
        [z[: z.shape[0] // 2], z[z.shape[0] // 2 : (z.shape[0] // 2) * 2]],
        axis=1)                                    # split chains, all params
    out = _ess_batched(zs)
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


@dataclass
class Diagnostics:
    """Convergence summary for one chain array."""

    rhat: np.ndarray
    ess_bulk: np.ndarray
    names: list[str]

    def __str__(self) -> str:
        lines = [f"{'param':>10s} {'rhat':>8s} {'ess_bulk':>10s}"]
        for i, n in enumerate(self.names):
            lines.append(f"{n:>10s} {self.rhat[i]:>8.4f} {self.ess_bulk[i]:>10.0f}")
        return "\n".join(lines)


def diagnose(chain: np.ndarray, names: list[str] | None = None) -> Diagnostics:
    """R-hat and bulk ESS in a single pass.

    Prefer this to calling :func:`split_rhat` and :func:`ess_bulk`
    separately: rank normalization is ~85% of the work and this shares it
    between the two statistics instead of repeating it.
    """
    if chain.ndim == 2:
        chain = chain[..., None]
    dim = chain.shape[-1]
    z = _rank_normalize_all(chain)
    rhat = np.array([_rhat_single(_split_chains(z[..., d])) for d in range(dim)])
    half = z.shape[0] // 2
    zs = np.concatenate([z[:half], z[half : 2 * half]], axis=1)
    return Diagnostics(rhat=rhat, ess_bulk=_ess_batched(zs),
                       names=names or [f"p{d}" for d in range(dim)])


def summary(chain: np.ndarray, names: list[str] | None = None) -> str:
    """Human-readable per-parameter table: mean, sd, R-hat, bulk ESS."""
    if chain.ndim == 2:
        chain = chain[..., None]
    dim = chain.shape[-1]
    names = names or [f"p{d}" for d in range(dim)]
    d = diagnose(chain, names)
    rhat, ess = d.rhat, d.ess_bulk
    lines = [f"{'param':>10s} {'mean':>12s} {'sd':>12s} {'rhat':>8s} {'ess_bulk':>10s}"]
    for d in range(dim):
        draws = chain[..., d]
        lines.append(
            f"{names[d]:>10s} {draws.mean():>12.5g} {draws.std():>12.5g} "
            f"{rhat[d]:>8.4f} {ess[d]:>10.0f}"
        )
    return "\n".join(lines)


def whitened_shape(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-parameter skew and excess kurtosis *after* whitening.

    Whitens the sample by its own covariance, which removes all linear
    correlation, and reports what is left. This predicts whether a dense
    mass matrix will help a gradient sampler, because a dense matrix
    removes exactly the linear part and nothing else:

    * both near zero -> the posterior is correlated but Gaussian-ish, and
      ``ChEESHMC(..., dense=True)`` should pay off (measured 5-40x more
      effective samples per gradient on correlated Gaussians);
    * large -> the posterior is *curved*, no single global mass matrix can
      linearize it, and an affine-invariant ensemble move is likely the
      better buy -- or better still, reparameterize the curvature away.

    ``draws``: (n_draws, dim), e.g. ``results.get_chain(flat=True)``.
    Returns ``(skew, excess_kurtosis)``, each (dim,).
    """
    x = np.asarray(draws, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"draws must be (n_draws, dim); got {x.shape}")
    xc = x - x.mean(axis=0)
    cov = np.cov(xc.T)
    cov = np.atleast_2d(cov)
    # whiten with the Cholesky factor: z = xc @ inv(L).T has identity cov
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]) * np.trace(cov))
    z = xc @ np.linalg.inv(L).T
    return (z ** 3).mean(axis=0), (z ** 4).mean(axis=0) - 3.0

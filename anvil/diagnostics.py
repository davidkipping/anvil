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


#: Rows above which ``mx.argsort`` stops returning a permutation. The two
#: cases are different, and the difference is worth 100x here:
#:
#:   * a **contiguous 1-D** sort is exact to at least 2**27 rows;
#:   * a **strided multi-column** sort (``axis=0`` of an (rows, dim) array
#:     with dim >= 2) breaks above ``1023 * 2048`` rows -- a tile counter,
#:     not a row counter, and identical at dim 2, 3, 8 and 105.
#:
#: Both failures are silent. Just past the limit the result repeats indices;
#: further past it, entries come back as 2143289344, which is the bit
#: pattern of float32 NaN, so the indices are being carried through float32
#: somewhere. Minimal reproducer (MLX 0.32.2, M2 Max):
#:
#:     x = mx.array(np.random.standard_normal((2_095_105, 2)).astype("f4"))
#:     o = np.array(mx.argsort(x, axis=0))
#:     np.array_equal(np.sort(o[:, 0]), np.arange(len(o)))   # False
#:     # ... and True at 2_095_104 rows, or at any size with dim == 1.
#:
#: An earlier version of this file guarded on ``rows > 2**21``, which is
#: BOTH too lenient and too strict: it let the corrupt multi-column sort run
#: for rows in (2_095_104, 2**21] -- 512 chains x 4096 draws lands exactly
#: there -- while sending every single-column sort above 2**21 to a numpy
#: fallback 100x slower than the GPU can do it.
_MX_SORT_COL_LIMIT = 1023 * 2048          # 2_095_104
_MX_SORT_1D_LIMIT = 2 ** 27
#: tail fraction of draws at each end whose normal score is recomputed on
#: the host in float64 -- see _rank_scores_1d
_TAIL_FRACTION = 1e-4

#: largest integer float32 represents exactly; above it ranks need int32 and
#: the quantile transform has to happen on the host in float64
_F32_EXACT_INT = 2 ** 24

#: Default ceiling on the working set of the sort and FFT stages, in bytes.
#: Diagnostics are not where a run should run out of memory: the old code's
#: peak grew as rows x chains x dim and reached 21 GB on a 16-parameter,
#: 16k-draw, 512-chain fit, which made a 32 GB machine swap. Peak device
#: memory comes out at about twice this, consistently (measured 0.96 GB at
#: 512 MiB, 3.85 GB at 2 GiB, 7.70 GB at 4 GiB), because the budget sizes
#: one stage's working set and MLX holds its own workspace beside it.
_DEFAULT_BUDGET = 1 << 31                 # 2 GiB

#: Grouping parameters amortizes the strided host gather, but only up to a
#: point: past ~2 GiB of working set the extra memory pressure costs more
#: than the batching saves. Measured at 8.4 M draws x 105 parameters --
#: 18.4 s at 2 parameters per group, 12.1 s at 9, 16.2 s at 18, 27.1 s at 36
#: -- so the heuristic is capped here even when a larger budget is given.
#: A *smaller* budget is always honoured; this only stops a generous one
#: from making things slower.
_GROUP_SOFT_CAP = 1 << 31                 # 2 GiB


def _rank_scores_1d(a: mx.array, rows: int) -> np.ndarray:
    """Normal scores for one parameter, ranked by a contiguous 1-D GPU sort.

    ``a`` is that parameter's draws as a (rows,) device array. The 1-D sort
    MLX gets right at any size we can store, and it is ~55x faster than
    numpy's stable argsort (measured 20 ms against 1.14 s at 8.4 M draws).
    MLX's sort is stable, so the ranks are bit-identical to
    ``np.argsort(kind="stable")`` even when ties are the rule rather than the
    exception (verified on 3 M draws taking only 8 distinct values)."""
    order = mx.argsort(a)
    if rows > _F32_EXACT_INT:
        # ranks stay exact as int32, but float32 cannot hold them, so the
        # quantile transform moves to the host (156 ms per parameter at
        # 8.4 M draws against 28 ms on the GPU -- only worth it up here)
        ranks = mx.put_along_axis(mx.zeros(rows, dtype=mx.int32), order,
                                  mx.arange(1, rows + 1, dtype=mx.int32),
                                  axis=0)
        return norm_ppf((np.array(ranks, dtype=np.int64) - 0.375)
                        / (rows + 0.25))
    ranks = mx.put_along_axis(mx.zeros(rows, dtype=mx.int32), order,
                              mx.arange(1, rows + 1, dtype=mx.int32), axis=0)
    p = (ranks.astype(mx.float32) - 0.375) / (rows + 0.25)
    z = np.array(_norm_ppf_mx_compiled(p), dtype=np.float64)

    # Float32 evaluates the quantile function poorly exactly where it is
    # steepest: at |z| ~ 5 it is off by 0.03, against 3.6e-4 for |z| < 4.
    # That is a handful of draws -- 1,679 of 8.4 M lie outside
    # p in [1e-4, 1-1e-4] -- and their ranks are known a priori (the k
    # smallest are ranks 1..k, the k largest are rows-k+1..rows), so only
    # their POSITIONS have to come back from the device. Redoing just those
    # in float64 costs ~nothing and removes the tail error entirely.
    k = int(np.ceil(_TAIL_FRACTION * rows))
    if k:
        lo_pos = np.array(order[:k], dtype=np.int64)
        hi_pos = np.array(order[rows - k:], dtype=np.int64)
        lo_rank = np.arange(1, k + 1, dtype=np.float64)
        z[lo_pos] = norm_ppf((lo_rank - 0.375) / (rows + 0.25))
        hi_rank = np.arange(rows - k + 1, rows + 1, dtype=np.float64)
        z[hi_pos] = norm_ppf((hi_rank - 0.375) / (rows + 0.25))
    return z


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
    rows = n * m
    if rows > _MX_SORT_COL_LIMIT:
        # The strided multi-column sort is wrong past this size, but the
        # contiguous 1-D one is not: rank each parameter on its own rather
        # than falling all the way back to the host.
        if rows > _MX_SORT_1D_LIMIT:
            return np.stack([_rank_normalize(chain[..., d].astype(np.float64))
                             for d in range(dim)], axis=-1)
        # Upload the whole group in one go. Pulling a single parameter out of
        # a (N, M, dim) host array reads one float in every `dim`, which cost
        # 46 ms per parameter at 8.4 M draws -- more than the sort itself --
        # while reading all `dim` adjacent columns costs barely more than
        # reading one. The per-parameter sorts then slice on the device.
        blk = mx.array(np.ascontiguousarray(chain.reshape(rows, dim),
                                            dtype=np.float32))
        out = np.empty((rows, dim), dtype=np.float64)
        for d in range(dim):
            out[:, d] = _rank_scores_1d(mx.contiguous(blk[:, d]), rows)
        return out.reshape(n, m, dim)
    flat = mx.array(np.ascontiguousarray(
        chain.reshape(n * m, dim), dtype=np.float32))
    order = mx.argsort(flat, axis=0)
    pos = mx.broadcast_to(
        mx.arange(1, n * m + 1, dtype=mx.float32)[:, None], flat.shape)
    ranks = mx.put_along_axis(mx.zeros_like(flat), order, pos, axis=0)
    p = (ranks - 0.375) / (n * m + 0.25)
    return np.array(_norm_ppf_mx(p), dtype=np.float64).reshape(n, m, dim)


def _param_group(rows: int, memory_budget: int) -> int:
    """How many parameters to score at once under ``memory_budget``.

    Per parameter and per draw: ~12 bytes on the GPU (the float32 column, its
    sort order and the ranks) and ~16 on the host (the float64 scores and
    their split-chain copy). Grouping is what makes the peak independent of
    ``dim``: the old code scored all of them at once, so a 105-parameter fit
    at 8.4 M draws asked for ~140 GB."""
    return max(1, int(min(memory_budget, _GROUP_SOFT_CAP)
                      // max(1, rows * 28)))


def _score_groups(chain: np.ndarray, memory_budget: int):
    """Yield ``(slice, normal scores)`` one memory-bounded parameter group at
    a time. chain: (N, M, dim)."""
    n, m, dim = chain.shape
    group = _param_group(n * m, memory_budget)
    for s0 in range(0, dim, group):
        sl = slice(s0, min(s0 + group, dim))
        yield sl, _rank_normalize_all(chain[..., sl])


def _norm_ppf_mx_compiled(p: mx.array) -> mx.array:
    """``_norm_ppf_mx`` with the whole rational approximation fused.

    Uncompiled it is ~30 ops over the full array, each writing a temporary:
    28 ms for 8.4 M draws, which was more than the sort. Fusing collapses
    that to one pass over memory. MLX retraces per shape, and diagnostics see
    only a handful."""
    return _norm_ppf_mx(p)


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


_norm_ppf_mx_compiled = mx.compile(_norm_ppf_mx_compiled)


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


def split_rhat(chain: np.ndarray,
               memory_budget: int = _DEFAULT_BUDGET) -> np.ndarray:
    """Split rank-normalized R-hat per parameter. chain: (N, M[, dim])."""
    single = chain.ndim == 2
    if single:
        chain = chain[..., None]
    out = np.empty(chain.shape[-1])
    for sl, z in _score_groups(chain, memory_budget):
        for j, d in enumerate(range(sl.start, sl.stop)):
            out[d] = _rhat_single(_split_chains(z[..., j]))
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


def _acov_chain_mean(x: np.ndarray,
                     memory_budget: int = _DEFAULT_BUDGET) -> np.ndarray:
    """Autocovariance averaged over chains, accumulated chunk by chunk.

    Bulk ESS uses the chain *average* of the autocovariance and its lag-0
    entry -- never the per-chain curves -- so the full ``(N, M, dim)``
    spectrum :func:`_autocov` builds was never needed. Summing over chunks of
    chains instead caps the working set at ``memory_budget`` whatever ``M``
    and ``dim`` are, which is the difference between 21 GB and a few hundred
    megabytes on a 16-parameter, 16k-draw, 512-chain fit.

    x: (N, M) or (N, M, dim). Returns (N,) or (N, dim), biased (1/N).
    """
    single = x.ndim == 2
    if single:
        x = x[..., None]
    n, m, dim = x.shape
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    if x.size < 1 << 18:
        # small enough that a GPU launch is not worth it; host float64
        xc = x - x.mean(axis=0)
        f = np.fft.rfft(xc, n=nfft, axis=0)
        acov = np.fft.irfft(f * np.conjugate(f), n=nfft, axis=0)[:n].real / n
        out = acov.mean(axis=1)
        return out[:, 0] if single else out
    # one chain in flight costs the half-spectrum (nfft/2+1 complex64), the
    # product and the inverse transform, over every parameter in the slice;
    # ~6 x nfft x 4 bytes per (chain, parameter) with MLX's own workspace
    per_chain = max(1, nfft * 4 * 6 * dim)
    step = max(1, min(m, int(memory_budget // per_chain)))
    total = np.zeros((n, dim), dtype=np.float64)
    for s0 in range(0, m, step):
        blk = mx.array(np.ascontiguousarray(x[:, s0:s0 + step],
                                            dtype=np.float32))
        blk = blk - mx.mean(blk, axis=0)
        f = mx.fft.rfft(blk, n=nfft, axis=0)
        acov = mx.fft.irfft(f * mx.conjugate(f), n=nfft, axis=0)[:n]
        total += np.array(mx.sum(acov, axis=1), dtype=np.float64)
        del blk, f, acov
    out = total / (n * m)
    return out[:, 0] if single else out


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


def _ess_batched(x: np.ndarray,
                 memory_budget: int = _DEFAULT_BUDGET) -> np.ndarray:
    """Bulk ESS for every parameter at once. x: (N, M, dim), split chains.

    Same Geyer initial monotone sequence as :func:`_ess_single`; the
    autocovariance is batched over parameters and accumulated over chunks of
    chains, so peak memory is set by ``memory_budget`` rather than by
    ``N x M x dim``.
    """
    n, m, dim = x.shape
    if n < 4:
        return np.full(dim, np.nan)
    acov = _acov_chain_mean(x, memory_budget)          # (N, dim), chain mean
    w = acov[0] * n / (n - 1)                                # (dim,)
    var_plus = w * (n - 1) / n
    if m > 1:
        var_plus = var_plus + x.mean(axis=0).var(axis=0, ddof=1)
    rho = 1.0 - (w - acov) / var_plus                         # (N, dim)
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


def ess_bulk(chain: np.ndarray,
             memory_budget: int = _DEFAULT_BUDGET) -> np.ndarray:
    """Rank-normalized bulk ESS per parameter. chain: (N, M[, dim])."""
    single = chain.ndim == 2
    if single:
        chain = chain[..., None]
    out = np.empty(chain.shape[-1])
    half = chain.shape[0] // 2
    for sl, z in _score_groups(chain, memory_budget):
        zs = np.concatenate([z[:half], z[half:2 * half]], axis=1)
        out[sl] = _ess_batched(zs, memory_budget)
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


def diagnose(chain: np.ndarray, names: list[str] | None = None,
             memory_budget: int = _DEFAULT_BUDGET) -> Diagnostics:
    """R-hat and bulk ESS in a single pass.

    Prefer this to calling :func:`split_rhat` and :func:`ess_bulk`
    separately: rank normalization is ~85% of the work and this shares it
    between the two statistics instead of repeating it.

    ``memory_budget`` (bytes, default 2 GiB) caps the working set of the
    sort and FFT stages; peak device memory lands at about twice it. Parameters are scored in groups that fit it and the
    chain-averaged autocovariance is accumulated over chunks of chains, so
    peak memory is set by the budget rather than by ``N x M x dim``. The
    results do not depend on it: R-hat is bit-identical and ESS agrees to
    float64 round-off (a budget 8192x smaller moves it by ~1e-15 relative,
    from reassociating the sum over chain chunks). Raise it to trade memory
    for fewer, larger GPU dispatches.
    """
    if chain.ndim == 2:
        chain = chain[..., None]
    dim = chain.shape[-1]
    rhat = np.empty(dim)
    ess = np.empty(dim)
    half = chain.shape[0] // 2
    for sl, z in _score_groups(chain, memory_budget):
        for j, d in enumerate(range(sl.start, sl.stop)):
            rhat[d] = _rhat_single(_split_chains(z[..., j]))
        zs = np.concatenate([z[:half], z[half:2 * half]], axis=1)
        ess[sl] = _ess_batched(zs, memory_budget)
        del z, zs
    return Diagnostics(rhat=rhat, ess_bulk=ess,
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


@dataclass
class WarmupReport:
    """Verdict on whether a run's warmup was long enough. See
    :func:`warmup_report`."""

    verdict: str
    settled_at: int | None      # iteration the spread stopped drifting
    n_warmup: int
    n_chains: int
    final_accept: float
    noise_floor: float          # relative sd of the spread estimate itself
    suggestion: str

    def __str__(self) -> str:
        at = "never" if self.settled_at is None else f"iteration {self.settled_at}"
        return (
            f"WarmupReport over {self.n_warmup} warmup iterations\n"
            f"  cross-chain spread settled : {at}\n"
            f"  final mean acceptance      : {self.final_accept:.3f}\n"
            f"  spread-estimate noise floor: {self.noise_floor:.1%} "
            f"({self.n_chains} chains)\n"
            f"  {self.verdict}: {self.suggestion}"
        )


def warmup_report(results, settle_tol: float = 0.05) -> WarmupReport:
    """Judge, after the fact, whether warmup was long enough.

    Warmup is pure overhead — it produces no samples — but cutting it too
    far biases everything downstream, so the useful thing is a measurement
    rather than a guess. This reads the cheap trace ``run()`` records and
    asks when the cross-chain spread stopped drifting.

    Two traps it is built to avoid:

    * **Noise masquerading as drift.** The cross-chain standard deviation
      is itself estimated from ``n_chains`` samples, with relative error
      ~1/sqrt(2(n_chains-1)) — 6% at 128 chains. Demanding tighter
      agreement than that never succeeds. The tolerance used is
      ``max(settle_tol, 3 x noise_floor)``.
    * **A stuck chain looking converged.** A chain that has stopped moving
      has a perfectly stable spread — and for HMC it also has acceptance
      near *one*, not near zero, because arbitrarily small steps are
      trivially accepted. The test that actually works is whether the
      spread ever changed at all: if it never moved off its initial value,
      the run is reported as inconclusive rather than converged.

    ``results``: a :class:`~anvil.engine.Results` from a run with
    ``warmup_probes > 0`` (the default).
    """
    tr = getattr(results, "warmup_trace", None)
    if not tr:
        raise ValueError(
            "no warmup trace on this result; run() needs warmup_probes > 0 "
            "and n_warmup > 0"
        )
    it, sd = tr["iter"], tr["sd"]
    # the requested warmup length, not merely the last probe (probes land
    # on multiples of the probe interval, so they usually stop just short)
    n_warm = int(getattr(results, "n_warmup", 0) or it[-1])
    n_chains = results.n_chains
    accept = float(tr["accept"][-1])
    eps = tr["step_size"]
    noise = 1.0 / np.sqrt(2.0 * max(n_chains - 1, 1))
    tol = max(settle_tol, 3.0 * noise)

    # reference: the last third of probes, which is the best estimate of
    # the stationary spread available from this run
    ref = np.median(sd[max(1, 2 * len(sd) // 3):], axis=0)
    ref = np.where(ref > 0, ref, 1.0)
    within = np.all(np.abs(sd / ref - 1.0) <= tol, axis=1)
    settled = None
    for k in range(len(within)):
        if within[k:].all():
            settled = int(it[k])
            break

    # Stuck-chain guards, checked before any "converged" verdict.
    # The decisive signal is that the spread never moved off its starting
    # value: a chain that is not exploring has a perfectly stable spread,
    # and for HMC its acceptance is near 1 (tiny steps are always
    # accepted), so acceptance alone would point the wrong way.
    drift = float(np.abs(sd[-1] / np.where(sd[0] > 0, sd[0], 1.0) - 1.0).max())
    tiny_eps = np.isfinite(eps).any() and float(eps[-1]) < 1e-4
    if drift <= tol:
        return WarmupReport(
            verdict="INCONCLUSIVE", settled_at=settled, n_warmup=n_warm,
            n_chains=n_chains, final_accept=accept, noise_floor=noise,
            suggestion=(
                f"the cross-chain spread never changed (by {drift:.1%} over "
                "the whole of warmup), so this cannot distinguish chains that "
                "began at the stationary distribution from chains that never "
                f"moved. Acceptance is {accept:.3f}"
                + (f" and the step size ended at {float(eps[-1]):.2g}, which is "
                   "small enough to suggest the latter" if tiny_eps else "")
                + ". Compare the spread against an independent estimate of the "
                "posterior width before trusting this run"
            ),
        )
    if accept < 0.01:
        return WarmupReport(
            verdict="FAILED", settled_at=settled, n_warmup=n_warm,
            n_chains=n_chains, final_accept=accept, noise_floor=noise,
            suggestion=(f"acceptance collapsed to {accept:.3g}; the chains are "
                        "barely moving, so warmup length is not the problem — "
                        "check the initialization and the model"),
        )

    if settled is None:
        return WarmupReport(
            verdict="TOO SHORT", settled_at=None, n_warmup=n_warm,
            n_chains=n_chains, final_accept=accept, noise_floor=noise,
            suggestion=("the cross-chain spread was still drifting at the end "
                        f"of warmup; try at least {2 * n_warm} iterations and "
                        "re-check"),
        )

    if settled <= n_warm // 2:
        return WarmupReport(
            verdict="LONGER THAN NEEDED", settled_at=settled, n_warmup=n_warm,
            n_chains=n_chains, final_accept=accept, noise_floor=noise,
            suggestion=(f"the spread settled by iteration {settled}; roughly "
                        f"{min(n_warm, 2 * settled)} warmup iterations would "
                        "do for this problem and initialization, leaving the "
                        "rest of the budget for draws"),
        )

    return WarmupReport(
        verdict="OK", settled_at=settled, n_warmup=n_warm, n_chains=n_chains,
        final_accept=accept, noise_floor=noise,
        suggestion=(f"the spread settled by iteration {settled}, comfortably "
                    "inside the warmup budget"),
    )

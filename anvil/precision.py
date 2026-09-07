"""Float32 conditioning tools: chunked reductions, fp64 anchoring, and the
precision validation harness.

The strategy: per-datum log-likelihood terms are computed in float32 on the
GPU, summed chunk-wise (a two-level tree, error O(eps*sqrt(n_chunks)) rather
than O(eps*n) for naive accumulation). Optionally the cross-chunk sum runs
in float64 on the CPU stream *inside the same lazy graph* ("fp64_anchor").
The default keeps the hot compiled step pure-fp32 and instead re-anchors
the *cached* log_prob against a full-fp64 CPU evaluation every K iterations
(the engine's ``reanchor_every``), so rounding drift cannot accumulate in
the Markov chain's accept/reject bookkeeping.

``validate_precision`` is the user-facing trust feature: it evaluates the
production fp32 path and a full-fp64 CPU path at given points and reports
the discrepancy in units that matter (the ~1-unit scale of Metropolis
accept decisions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import mlx.core as mx
import numpy as np


#: fixed-point scale for the exact reduction: terms are accumulated as
#: int64 multiples of 2**-30. Chosen so that (a) multiplying a float32 by
#: it is an exact exponent shift, (b) a per-term clamp at 2**43 still
#: admits |term| up to 8192 (residuals ~128 sigma), and (c) 2**20 clamped
#: terms cannot overflow int64.
_FIXED_SHIFT = 30
_FIXED_SCALE = float(2 ** _FIXED_SHIFT)
_FIXED_CLAMP = float(2 ** 43)


@dataclass
class PrecisionPolicy:
    #: "fp32_tree": chunked fp32 sums only (compilable, default).
    #: "fp64_anchor": cross-chunk sum in fp64 on the CPU stream (uncompiled).
    #: "fixed_point": accumulate terms as int64 fixed-point in a custom
    #: Metal kernel -- exact and order-independent, so the only rounding
    #: left is representing the answer as float32; falls back to
    #: "fp32_tree" off-GPU or for non-float32 inputs.
    reduction: Literal["fp32_tree", "fp64_anchor", "fixed_point"] = "fp32_tree"
    #: Data-axis tile width. Primarily an accuracy knob (it sets the depth
    #: of the summation tree), but it also bounds the reverse-mode tape:
    #: the chunk loop unrolls into one graph, so peak gradient memory falls
    #: roughly with the chunk. Measured at 1024 chains x 1e5 points,
    #: 65536 -> 16384 costs nothing in forward time, is marginally faster
    #: for gradients, and cuts peak memory ~1.4-1.7x; below ~4096 the
    #: dispatch count starts to cost at large N. Lower it further if a
    #: gradient run is memory-bound.
    chunk_size: int = 16384
    #: engine-level cadence for re-anchoring cached log_prob in fp64; 0 = off
    reanchor_every: int = 0
    #: Subtract the parameter-independent -N/2 from the summed terms so the
    #: float32 quantity is O(sqrt(N/2)) instead of O(N/2). Free, and it also
    #: shrinks the rounding of every DOWNSTREAM difference (notably HMC's
    #: energy difference, which validate_precision cannot see). See the
    #: log_offset_const note on ChunkedGaussianLogLike.
    recenter: bool = True


_FIXED_SRC = """
    uint chain = threadgroup_position_in_grid.y;
    uint tid   = thread_position_in_threadgroup.x;
    uint nth   = threads_per_threadgroup.x;
    uint m     = (uint)npts;
    uint base  = chain * m;
    long acc = 0;
    for (uint j = tid; j < m; j += nth) {
        // exact: multiplying a float32 by a power of two only shifts its
        // exponent, so rint() is the only rounding, at 2**-31 per term
        float f = metal::rint(terms[base + j] * SCALE);
        f = metal::clamp(f, -CLAMP, CLAMP);
        acc += (long)f;
    }
    threadgroup long tg[256];
    tg[tid] = acc;
    for (uint i = nth + tid; i < 256; i += nth) { tg[i] = 0; }  // pad: nth
    threadgroup_barrier(mem_flags::mem_threadgroup);            // need not be
    for (uint s = 128; s > 0; s >>= 1) {   // fixed power-of-two tree, exact
        if (tid < s) { tg[tid] += tg[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { out[chain] = tg[0]; }
""".replace("SCALE", f"{_FIXED_SCALE}f").replace("CLAMP", f"{_FIXED_CLAMP}f")

_fixed_kernel = None


def _fixed_point_partial(terms: mx.array) -> mx.array:
    """Exact int64 fixed-point sum over the last axis of (n_chains, m)."""
    global _fixed_kernel
    if _fixed_kernel is None:
        _fixed_kernel = mx.fast.metal_kernel(
            name="anvil_fixed_point_sum",
            input_names=["terms", "npts"],
            output_names=["out"],
            source=_FIXED_SRC,
        )
    n, m = terms.shape
    return _fixed_kernel(
        inputs=[terms, int(m)],
        output_shapes=[(n,)], output_dtypes=[mx.int64],
        grid=(256, n, 1), threadgroup=(256, 1, 1),
    )[0]


def _fixed_to_float32(acc: mx.array) -> mx.array:
    """int64 fixed-point -> float32, without losing bits in the cast.

    A direct ``astype(float32)`` of a ~2**37 integer discards 13 bits (and
    is not even reproducible across MLX's eager/compiled paths). Splitting
    the integer keeps both halves inside float32's exact-integer range, so
    each half converts and scales exactly and only the final add rounds --
    which is the unavoidable cost of returning a float32 at all.
    """
    split = 1 << 20
    hi = acc // split
    lo = acc - hi * split
    return (hi.astype(mx.float32) / float(1 << (_FIXED_SHIFT - 20))
            + lo.astype(mx.float32) / _FIXED_SCALE)


def _fixed_point_usable(policy: PrecisionPolicy) -> bool:
    return policy.reduction == "fixed_point" and mx.metal.is_available()


def chunked_sum(
    term_fn: Callable[[int, int], mx.array],
    n_data: int,
    policy: PrecisionPolicy,
) -> mx.array:
    """Sum per-datum terms over the data axis with controlled rounding.

    ``term_fn(start, stop)`` returns the fp32 terms for one data chunk,
    shape (n_chains, stop - start). Returns (n_chains,) fp32.
    """
    c = policy.chunk_size
    if _fixed_point_usable(policy):
        probe = term_fn(0, min(c, n_data))
        if probe.dtype == mx.float32:
            # accumulate int64 across chunks too, so the whole reduction is
            # exact and only the final float32 cast rounds
            acc = _fixed_point_partial(probe)
            for s in range(c, n_data, c):
                acc = acc + _fixed_point_partial(
                    term_fn(s, min(s + c, n_data)))
            return _fixed_to_float32(acc)
    partials = [
        mx.sum(term_fn(s, min(s + c, n_data)), axis=-1)
        for s in range(0, n_data, c)
    ]
    if len(partials) == 1:
        stacked = partials[0][:, None]
    else:
        stacked = mx.stack(partials, axis=-1)  # (n_chains, n_chunks)
    if policy.reduction == "fp64_anchor":
        s64 = mx.sum(
            stacked.astype(mx.float64, stream=mx.cpu), axis=-1, stream=mx.cpu
        )
        return s64.astype(mx.float32, stream=mx.cpu)
    return mx.sum(stacked, axis=-1)


class ChunkedGaussianLogLike:
    """Independent-Gaussian log-likelihood over a large dataset, chunked.

    ``model_fn(v, x)`` is the user's batched, dtype-polymorphic MLX model:
    parameters ``v`` (n_chains, dim) and abscissa ``x`` (m,) — or (c, m) for
    multi-channel abscissae such as (epoch-centered time, orbit number) —
    -> predictions (n_chains, m). It must use only MLX ops on its arguments
    so the same code runs fp32-on-GPU (production) and fp64-on-CPU
    (anchor/validation).

    Data enter as float64 numpy (the caller has already centered/scaled
    them into well-conditioned model units); fp32 copies feed the GPU path.

    TWO parameter-independent constants are deliberately kept out of the
    graph as float64 host scalars, because both cancel in MH ratios and
    both would otherwise force the float32 arithmetic to carry a large
    number:

    * ``log_norm_const`` = -sum(log(2*pi*sigma^2))/2, the Gaussian
      normalization;
    * ``log_offset_const`` = -N/2 (when ``policy.recenter``), which
      recentres the chi-squared. Because E[sum r^2] = N for a correct
      model, summing ``0.5*(1 - r^2)`` instead of ``-0.5*r^2`` leaves an
      O(sqrt(N/2)) quantity rather than an O(N/2) one -- at N = 1e5 that
      is ~200 instead of ~5e4, so the float32 ulp of the stored value
      falls by ~500x and every difference taken downstream (Metropolis
      ratios, and HMC's energy difference) gets correspondingly sharper.
      The per-term form ``0.5*(1-r)*(1+r)`` is used because ``1-r`` is
      exact for 0.5 <= r <= 2 (Sterbenz), avoiding cancellation near
      r = 1.

    So the true unnormalized log-likelihood is
    ``value + log_offset_const``, and the fully normalized one adds
    ``log_norm_const`` as well.
    """

    def __init__(
        self,
        model_fn: Callable[[mx.array, mx.array], mx.array],
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray,
        policy: PrecisionPolicy | None = None,
    ):
        self.model_fn = model_fn
        self.policy = policy or PrecisionPolicy()
        self._x64 = np.asarray(x, dtype=np.float64)
        self._y64 = np.asarray(y, dtype=np.float64)
        self._yerr64 = np.asarray(yerr, dtype=np.float64)
        self.n_data = self._y64.size

        self._x32 = mx.array(self._x64.astype(np.float32))
        self._y32 = mx.array(self._y64.astype(np.float32))
        self._w32 = mx.array((1.0 / self._yerr64).astype(np.float32))

        self.log_norm_const = float(
            -0.5 * np.sum(np.log(2.0 * np.pi * self._yerr64**2))
        )
        # exact in float64; add it back to recover the true chi-squared logL
        self.log_offset_const = (
            -0.5 * float(self.n_data) if self.policy.recenter else 0.0
        )

    def _terms(self, r: mx.array) -> mx.array:
        """Per-datum contribution, dtype-polymorphic."""
        if self.policy.recenter:
            return 0.5 * (1.0 - r) * (1.0 + r)   # == 0.5 - 0.5*r*r
        return -0.5 * r * r

    def __call__(self, v: mx.array) -> mx.array:
        """fp32 chunked chi-squared log-likelihood (n_chains,)."""

        def term_fn(s: int, e: int) -> mx.array:
            m = self.model_fn(v, self._x32[..., s:e])
            r = (self._y32[s:e] - m) * self._w32[s:e]
            return self._terms(r)

        return chunked_sum(term_fn, self.n_data, self.policy)

    #: chain-block size for the float64 path; bounds peak memory, since the
    #: fp64 CPU graph can hold dozens of (block, chunk)-sized intermediates
    hi_chain_block: int = 64

    def hi(self, v: mx.array) -> mx.array:
        """Full-float64 CPU-stream evaluation of the same likelihood.

        Processed in (chain-block x data-chunk) tiles with an eval per
        tile: an expensive model's fp64 graph over all chains x all data
        at once can transiently allocate tens of GB.
        """
        n_chains = v.shape[0]
        c = self.policy.chunk_size
        out = np.zeros(n_chains, dtype=np.float64)
        with mx.stream(mx.cpu):
            v64 = v.astype(mx.float64)
            for cb in range(0, n_chains, self.hi_chain_block):
                vb = v64[cb : cb + self.hi_chain_block]
                acc = mx.zeros(vb.shape[:1], dtype=mx.float64)
                for s in range(0, self.n_data, c):
                    e = min(s + c, self.n_data)
                    # dtype= is REQUIRED: mx.array() of a float64 numpy
                    # array silently yields float32, which would make this
                    # "float64 path" a second float32 path.
                    x = mx.array(self._x64[..., s:e], dtype=mx.float64)
                    y = mx.array(self._y64[s:e], dtype=mx.float64)
                    w = mx.array(1.0 / self._yerr64[s:e], dtype=mx.float64)
                    r = (y - self.model_fn(vb, x)) * w
                    acc = acc + mx.sum(self._terms(r), axis=-1)
                    mx.eval(acc)  # free this tile's graph before the next
                out[cb : cb + vb.shape[0]] = np.array(acc)
            return mx.array(out, dtype=mx.float64)


@dataclass
class PrecisionReport:
    n_points: int
    median_abs_err: float
    max_abs_err: float
    median_logl_magnitude: float

    def __str__(self) -> str:
        if self.max_abs_err < 0.1:
            verdict = ("OK: fp32 error is far below the ~1-unit scale of "
                       "Metropolis accept decisions")
        elif self.max_abs_err < 1.0:
            verdict = ("ACCEPTABLE: fp32 error is below the ~1-unit scale of "
                       "Metropolis accept decisions; residual distortion of "
                       "the sampled posterior is small. Read this as a "
                       "direct proxy for accept/reject distortion: the error "
                       "in the log-density *difference* between two states "
                       "does NOT cancel — measured 1.6-2.1x the pointwise "
                       "error at proposal-scale displacements.")
        else:
            verdict = ("WARNING: fp32 error reaches the scale of Metropolis "
                       "accept decisions — improve model conditioning "
                       "(offset parameters/data as in the docs) and/or "
                       "enable the fp64_anchor reduction")
        return (
            f"PrecisionReport over {self.n_points} states\n"
            f"  |logL| typical magnitude : {self.median_logl_magnitude:.4g}\n"
            f"  |fp32 - fp64| median     : {self.median_abs_err:.4g}\n"
            f"  |fp32 - fp64| max        : {self.max_abs_err:.4g}\n"
            f"  {verdict}"
        )


def validate_precision(target, u: mx.array) -> PrecisionReport:
    """Compare the production fp32 log_prob path against the fp64 path.

    ``target`` must expose ``log_prob(u)`` (fp32 production path) and
    ``log_prob_hi(u)`` (float64 CPU path). ``u``: (n_points, dim) states.
    """
    lp32 = np.array(target.log_prob(u), dtype=np.float64)
    lp64 = np.array(target.log_prob_hi(u), dtype=np.float64)
    err = np.abs(lp32 - lp64)
    return PrecisionReport(
        n_points=int(u.shape[0]),
        median_abs_err=float(np.median(err)),
        max_abs_err=float(err.max()),
        median_logl_magnitude=float(np.median(np.abs(lp64))),
    )


@dataclass
class Certificate:
    """Bias certificate and correction from :func:`certify`.

    All bias figures are in units of the parameter's own posterior
    standard deviation, so they compare directly against the Monte Carlo
    standard error 1/sqrt(ESS).
    """

    n_probe: int
    n_probe_needed: float    # >~ err_sd^2 * target_ess for the correction
    n_draws: int
    err_mean: float          # irrelevant to inference; reported for context
    err_sd: float            # the dispersion — this is what biases things
    chi2: float              # Var(err): importance-weight variance
    is_retention: float      # exp(-chi2): ESS kept by exact reweighting
    bias_std: np.ndarray     # (dim,) standardized bias per parameter
    max_bias_std: float
    target_ess: float
    ess_ceiling: float       # worst-case ESS certified by err_sd alone
    raw_mean: np.ndarray
    corrected_mean: np.ndarray
    probe_index: np.ndarray
    probe_err: np.ndarray
    names: list[str] | None
    verdict: str

    def correct(self, values: np.ndarray) -> np.ndarray:
        """Bias-correct the mean of any quantity evaluated on the draws.

        ``values`` has shape (n_draws,) or (n_draws, k), aligned with the
        ``draws`` passed to :func:`certify`. Returns the corrected mean(s):
        ``mean(f) - Cov(f, err)``, exact to O(err^2).
        """
        v = np.asarray(values, dtype=np.float64)
        flat = v.reshape(v.shape[0], -1)
        if flat.shape[0] != self.n_draws:
            raise ValueError(
                f"values has {flat.shape[0]} rows but certify() was given "
                f"{self.n_draws} draws; they must be the same sample"
            )
        probe = flat[self.probe_index]
        de = self.probe_err - self.probe_err.mean()
        cov = (probe - probe.mean(axis=0)).T @ de / len(de)
        out = flat.mean(axis=0) - cov
        return out.reshape(v.shape[1:]) if v.ndim > 1 else out[0]

    def __str__(self) -> str:
        lines = [
            f"Certificate from {self.n_probe} probe evaluations "
            f"of {self.n_draws} draws",
            f"  float32 error dispersion (sd) : {self.err_sd:.4g}"
            f"   [mean {self.err_mean:+.4g}, irrelevant: it cancels]",
            f"  exact-reweighting ESS retained: {self.is_retention:.6f}",
            f"  worst-case certified ESS      : {self.ess_ceiling:,.0f}",
            f"  probes used / needed          : {self.n_probe} / "
            f"{max(16.0, self.n_probe_needed):.0f}",
            f"  bias at ESS = {self.target_ess:,.0f} (units of MC standard error):",
        ]
        order = np.argsort(-np.abs(self.bias_std))
        names = self.names or [f"p{i}" for i in range(len(self.bias_std))]
        for i in order[:8]:
            rel = self.bias_std[i] * np.sqrt(self.target_ess)
            lines.append(f"    {names[i]:>12s} {rel:+8.3f}"
                         f"   ({self.bias_std[i]:+.3g} posterior sd)")
        lines.append(f"  {self.verdict}")
        return "\n".join(lines)


def certify(
    target,
    draws,
    *,
    n_probe: int = 256,
    target_ess: float = 1e4,
    seed: int = 0,
    names: list[str] | None = None,
) -> Certificate:
    """Quantify — and correct — the posterior bias caused by float32.

    Because the float32 log-density is a *deterministic* function of the
    parameters, the sampler is exactly stationary for a slightly tilted
    target ``pi * exp(err)``. The resulting bias in any posterior mean is
    ``Cov(f, err)`` to first order, so evaluating ``err`` on a small
    random subset of the stored draws both measures the bias and removes
    it — at a cost of ``n_probe`` float64 evaluations for the whole run,
    rather than one per iteration.

    ``draws``: (n_draws, dim) posterior draws in *sampling* (u) space,
    e.g. ``results.get_chain(flat=True)``. Requires ``target.log_prob_hi``.

    Pick ``n_probe`` >> err_sd^2 * target_ess (the returned ``err_sd``
    lets you check afterwards); the default 256 is generous for the
    error levels a well-conditioned model produces.
    """
    if not hasattr(target, "log_prob_hi"):
        raise ValueError(
            "certify() needs a float64 path (target.log_prob_hi); pass "
            "model_log_prob_hi= to TransformedLogDensity"
        )
    d = np.asarray(draws, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"draws must be (n_draws, dim); got {d.shape}")
    n_draws, dim = d.shape
    n_probe = int(min(n_probe, n_draws))
    idx = np.random.default_rng(seed).choice(n_draws, n_probe, replace=False)
    idx.sort()

    probe = mx.array(d[idx].astype(np.float32))
    lp32 = np.array(target.log_prob(probe), dtype=np.float64)
    lp64 = np.array(target.log_prob_hi(probe), dtype=np.float64)
    err = lp32 - lp64

    de = err - err.mean()
    dp = d[idx] - d[idx].mean(axis=0)
    bias = dp.T @ de / n_probe                   # Cov(u_j, err)
    sd = d.std(axis=0)
    bias_std = np.divide(bias, sd, out=np.zeros_like(bias), where=sd > 0)

    err_sd = float(err.std())
    chi2 = err_sd**2
    max_b = float(np.abs(bias_std).max()) if dim else 0.0
    rel = max_b * np.sqrt(target_ess)

    # the correction's own noise is sd*err_sd/sqrt(n_probe) versus the MC
    # error sd/sqrt(target_ess): negligible once n_probe >> err_sd^2*ess
    needed = 10.0 * chi2 * target_ess
    short = n_probe < max(16.0, needed)

    if max_b >= 0.1 or err_sd >= 0.5:
        verdict = ("WARNING: float32 bias is large in absolute terms "
                   "(>=0.1 posterior sd, or error dispersion >=0.5 nats) — "
                   "fix the model's conditioning; a correction cannot be "
                   "trusted this far out")
    elif rel < 0.1:
        verdict = (f"OK: bias is {rel:.3f}x the Monte Carlo standard error "
                   f"at ESS={target_ess:,.0f} — negligible")
    elif rel < 1.0:
        verdict = (f"ACCEPTABLE: bias is {rel:.2f}x the Monte Carlo standard "
                   f"error at ESS={target_ess:,.0f} — below the error bar, "
                   "and corrected_mean removes it")
    else:
        verdict = (f"WARNING: bias is {rel:.1f}x the Monte Carlo standard "
                   f"error at ESS={target_ess:,.0f} — report corrected_mean, "
                   "or improve conditioning")

    if short:
        verdict = (f"UNDER-PROBED: n_probe={n_probe} is too few to resolve "
                   f"the bias at ESS={target_ess:,.0f} (need >~{needed:.0f}). "
                   "Re-run certify with a larger n_probe. " + verdict)

    return Certificate(
        n_probe=n_probe, n_probe_needed=float(needed), n_draws=n_draws,
        err_mean=float(err.mean()), err_sd=err_sd, chi2=chi2,
        is_retention=float(np.exp(-chi2)),
        bias_std=bias_std, max_bias_std=max_b,
        target_ess=float(target_ess),
        ess_ceiling=float(1.0 / chi2) if chi2 > 0 else float("inf"),
        raw_mean=d.mean(axis=0), corrected_mean=d.mean(axis=0) - bias,
        probe_index=idx, probe_err=err, names=names, verdict=verdict,
    )

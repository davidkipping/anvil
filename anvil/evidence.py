"""Bayesian evidence (marginal likelihood), as an optional calculation.

Nothing here runs unless called: :func:`anvil.run` and the samplers never
compute an evidence. Three tools, in decreasing order of what they assume:

* :func:`smc` -- tempered sequential Monte Carlo. Particles drawn from the
  prior are carried to the posterior through L^beta, reweighted, resampled
  and moved at each step; the product of the mean weights is Z. Mode masses
  come from the weights rather than from chains crossing between modes, so
  it holds on posteriors MCMC cannot certify. Its particles are posterior
  draws.
* :func:`bridge_evidence` -- Meng and Wong's optimal bridge between a set of
  posterior draws and a Student-t fitted to half of them. Seconds, and
  exact on near-Gaussian posteriors, but it knows only the mode weights its
  draws have.
* :func:`laplace_evidence` -- the Gaussian approximation at the mode. A
  floor, not an answer.

Measured on Elome transit fits (10,000 points, hybrid4 limb darkening;
docs/evidence.md has the tables):

* Spherical: SMC at its defaults matched the reference ln Z to 0.02 nats in
  77 s, against 609 s for a 500 + 500 ChEES run of 512 chains.
* Oblate, with no oblateness in the data (a 2,000-point version): SMC was
  0.17-0.30 nats low at every population size tried (2,048 to 8,192),
  because its moves stop being accepted on the narrow curved posterior.
  It flags the stages where that happens (``SMCResult.stalled``) and
  warns.
* Laplace was 2.4 nats low on the spherical fit.

What ln Z includes: the priors declared on each ParamSpec, normalised (a
proper prior is required), and the likelihood's ``model_log_norm``. A
likelihood whose constants were not declared there (``ChunkedGaussianLogLike``
keeps them out of its graph) gives ln Z up to that constant -- which
cancels between models fitted to the same data with the same likelihood.
A penalty or prior written into ``model_log_prob`` itself is part of the
likelihood as far as anvil can tell, and is not normalised.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .compile import CompiledOrEager
from .logdensity import LogDensity
from .rng import KeyStream
from .transforms import TransformedLogDensity

#: stretch-move scale (Goodman & Weare's a)
_STRETCH_A = 2.0
#: sd of the groups' ln Z above which smc warns that groups are too small
_GROUP_SPREAD_WARN = 0.5
#: a stage that used all of max_moves, still ends with a coordinate
#: correlated above _STALL_RHO with the stage's start, and accepted fewer
#: than _STALL_ACCEPT of its moves has stalled. Both: a coordinate that
#: tells modes apart stays correlated however well particles move within
#: their modes, which the acceptance rate tells apart
_STALL_RHO = 0.9
_STALL_ACCEPT = 0.05
#: proposal-side ESS, as a fraction of the proposal draws, below which
#: bridge sampling warns
_BRIDGE_MIN_ESS = 0.01


def _require_proper(target) -> None:
    if not isinstance(target, TransformedLogDensity):
        raise ValueError(
            f"the evidence needs a TransformedLogDensity, whose priors are "
            f"declared and normalised; got {type(target).__name__}")
    if not target.is_proper:
        bad = [s.name for s, p in zip(target.transform.specs,
                                      target.transform.priors) if not p.is_proper]
        raise ValueError(
            f"the evidence is undefined under an improper prior: give {bad} "
            f"a finite box or an explicit prior")


def _logmeanexp(a: np.ndarray) -> float:
    m = np.max(a)
    if not np.isfinite(m):
        return -math.inf
    return float(m + math.log(np.mean(np.exp(a - m))))


# -- tempering ---------------------------------------------------------------

class TemperedLogDensity(LogDensity):
    """``beta * log L(v) + log p(v) + log|dv/du|`` for a TransformedLogDensity:
    the power posterior that tempering moves through. ``beta = 1`` is the
    target's own ``log_prob``; ``beta = 0`` the prior, as a density in u.

    ``beta`` may be changed between calls to :func:`anvil.run`, which
    retraces the kernel for it. A non-finite log-likelihood counts as
    ``-inf`` for every ``beta > 0`` (at ``beta = 0`` the point is
    excluded rather than given ``0 * inf``)."""

    def __init__(self, target: TransformedLogDensity, beta: float = 1.0):
        self.target = target
        self.beta = float(beta)
        self.dim = target.dim
        self.supports_grad = target.supports_grad

    def log_prob(self, u: mx.array) -> mx.array:
        ll, rest = _split(self.target, u)
        return mx.where(mx.isfinite(ll), self.beta * ll, -mx.inf) + rest


def _split(target: TransformedLogDensity, u: mx.array):
    """(log L, log prior + log|dv/du|) from a single map to model space."""
    tr = target.transform
    v = tr.to_model(u)
    ll = target._fn(v)
    rest = tr.log_det_jac(u)
    if tr.has_prior_terms:
        rest = rest + tr.log_prior(v)
    return mx.where(mx.isfinite(ll), ll, -mx.inf), rest


# -- SMC ---------------------------------------------------------------------

@dataclass
class SMCResult:
    """What :func:`smc` found.

    ``log_z`` is the log evidence. With ``n_groups > 1``, ``log_z_se`` is
    its standard error from the spread of ``log_z_groups`` (independent
    particle groups run in the same batched calls); with few groups it is
    itself uncertain: err/se follows a t distribution with n_groups - 1
    degrees of freedom. With one population (the default) it is NaN.

    ``stalled`` counts the stages that ran to ``max_moves`` with some
    coordinate still correlated above 0.9 with the stage's start and fewer
    than 5% of moves accepted: particles that stopped exploring, which
    biases ln Z low.

    ``u`` holds the final particles in sampling space, equal-weight
    posterior draws, ``(n_particles, dim)``; ``model`` and ``physical`` the
    same in model and reporting units.

    ``betas``, ``moves`` and ``accept`` record the path per stage;
    ``n_evals`` counts likelihood evaluations (rows). ``precision`` is the
    float32 error of log L at a few posterior draws against the target's
    float64 path, when it has one: report-only, not applied.
    """

    log_z: float
    log_z_se: float
    log_z_groups: np.ndarray
    names: list[str]
    u: np.ndarray
    model: np.ndarray
    physical: np.ndarray
    log_likelihood: np.ndarray
    betas: list[np.ndarray]
    moves: list[int]
    accept: list[float]
    decorrelation: list[float]
    n_evals: int
    wall_time: float
    precision: dict | None = None
    stalled: int = 0

    @property
    def n_stages(self) -> int:
        return len(self.moves)

    def __str__(self) -> str:
        G = len(self.log_z_groups)
        if G > 1:
            head = (f"anvil.smc: ln Z = {self.log_z:.4f} +/- {self.log_z_se:.4f} "
                    f"({G} independent groups)")
        else:
            head = f"anvil.smc: ln Z = {self.log_z:.4f} (one population)"
        lines = [head]
        if self.stalled:
            lines.append(f"  {self.stalled} of {self.n_stages} stages ended without "
                         f"the particles decorrelating: ln Z is likely low")
        lines.append(f"  {self.u.shape[0]} particles, {self.n_stages} stages, "
                     f"{sum(self.moves)} moves, {self.n_evals:,} likelihood "
                     f"evaluations, {self.wall_time:.1f} s")
        if self.precision is not None:
            lines.append(f"  float32 log L error at {self.precision['n']} "
                         f"posterior draws: {self.precision['mean']:+.2e} "
                         f"+/- {self.precision['sd']:.2e}")
        return "\n".join(lines)


class _Sweep:
    """Red-black stretch-move sweeps for G independent groups of 2K
    particles. A walker's partner comes from the other half of its own
    group, never another group's, so the groups stay independent. Proposals
    are evaluated in blocks of at most ``max_rows`` rows, each its own GPU
    command: one command over thousands of expensive light curves ran long
    enough for macOS to kill it ("Impacting Interactivity"). ``beta`` is an
    argument throughout, so nothing is retraced as it changes."""

    def __init__(self, target, G, K, d, a, max_rows, compile_step):
        self.G, self.K, self.d, self.a = G, K, d, a
        self.evaluate = CompiledOrEager(lambda u: _split(target, u),
                                        name="the SMC likelihood",
                                        enabled=compile_step)
        self.block = int(max_rows)

    def values(self, u: mx.array):
        """(log L, rest) at every row of u, block by block."""
        n = u.shape[0]
        b = min(self.block, n)
        lls, rs = [], []
        for s in range(0, n, b):
            ub = u[s:s + b]
            m = ub.shape[0]
            if m < b:       # one block shape, so one compiled trace
                ub = mx.concatenate([ub, mx.repeat(ub[:1], b - m, axis=0)], axis=0)
            ll, r = self.evaluate(ub)
            mx.eval(ll, r)
            lls.append(ll[:m])
            rs.append(r[:m])
        return mx.concatenate(lls), mx.concatenate(rs)

    def _half(self, keys, S, LLs, Rs, C, beta, de):
        G, K, d, a = self.G, self.K, self.d, self.a
        j = mx.random.randint(0, K, shape=(G, K), key=keys[0])
        Cp = mx.take_along_axis(C, j[..., None], axis=1)
        if de:
            # differential evolution, as DEMove: s + gamma (c1 - c2) + eps,
            # gamma = 1 one time in ten to jump between modes; symmetric
            j2 = (j + 1 + mx.random.randint(0, K - 1, shape=(G, K), key=keys[1])) % K
            C2 = mx.take_along_axis(C, j2[..., None], axis=1)
            gamma = mx.where(mx.random.uniform(shape=(G, K, 1), key=keys[3]) < 0.1,
                             1.0, 2.38 / math.sqrt(2 * d))
            Y = S + gamma * (Cp - C2) + 1e-5 * mx.random.normal(S.shape, key=keys[4])
            log_q = 0.0
        else:
            z = ((a - 1.0) * mx.random.uniform(shape=(G, K), key=keys[1]) + 1.0) ** 2 / a
            Y = Cp + z[..., None] * (S - Cp)
            log_q = (d - 1) * mx.log(z)
        ll_y, r_y = self.values(Y.reshape(G * K, d))
        ll_y, r_y = ll_y.reshape(G, K), r_y.reshape(G, K)
        log_ratio = log_q + beta * ll_y + r_y - (beta * LLs + Rs)
        ok = ((mx.log(mx.random.uniform(shape=(G, K), key=keys[2])) < log_ratio)
              & mx.all(mx.isfinite(Y), axis=-1) & mx.isfinite(ll_y))
        out = (mx.where(ok[..., None], Y, S), mx.where(ok, ll_y, LLs),
               mx.where(ok, r_y, Rs), ok)
        mx.eval(*out)
        return out

    def __call__(self, key, U, LL, R, beta, de=False):
        K = self.K
        k = mx.random.split(key, 10)
        A, LA, RA, okA = self._half(k[0:5], U[:, :K], LL[:, :K], R[:, :K], U[:, K:], beta, de)
        B, LB, RB, okB = self._half(k[5:10], U[:, K:], LL[:, K:], R[:, K:], A, beta, de)
        acc = (mx.sum(okA, axis=1) + mx.sum(okB, axis=1)) / (2.0 * K)
        return (mx.concatenate([A, B], axis=1), mx.concatenate([LA, LB], axis=1),
                mx.concatenate([RA, RB], axis=1), acc)


def _next_beta(ll: np.ndarray, beta: float, ess: float) -> float:
    """The largest increment whose weights L^dbeta keep an ESS of ess*n."""
    fin = np.isfinite(ll)

    def frac(db):
        lw = np.where(fin, db * ll, -np.inf)
        m = lw.max()
        w = np.exp(lw - m)
        return w.sum() ** 2 / (w * w).sum() / w.size

    hi = 1.0 - beta
    if frac(hi) >= ess:
        return hi
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if frac(mid) >= ess:
            lo = mid
        else:
            hi = mid
    return max(lo, 1e-300)


def _settled(hist: list[np.ndarray], target: float) -> bool:
    """Whether every group and coordinate has decorrelated from the start of
    the stage: correlation below ``target``, or no longer falling -- the
    drop over the last half of the sweeps under a quarter of the drop over
    the quarter before. A coordinate that tells modes apart keeps the
    correlation between modes, which no number of moves removes; the
    plateau test stops there instead of running to ``max_moves``."""
    k = len(hist)
    r_k, r_h, r_q = hist[k - 1], hist[k // 2 - 1], hist[k // 4 - 1]
    ok = r_k < target
    if k >= 16:
        ok |= (r_h - r_k) < 0.25 * (r_q - r_h)
    return bool(ok.all())


def _systematic(logw: np.ndarray, rng) -> np.ndarray:
    m = logw.max()
    w = np.exp(logw - m)
    c = np.cumsum(w / w.sum())
    n = w.size
    return np.minimum(np.searchsorted(c, (rng.uniform() + np.arange(n)) / n), n - 1)


def smc(target: TransformedLogDensity, n_particles: int = 8192, *,
        n_groups: int = 1, ess: float = 0.5, decorrelation: float | None = 0.1,
        de_fraction: float = 0.5, accepts_per_stage: float = 10.0,
        min_moves: int = 8, max_moves: int = 200, max_stages: int = 1000,
        n_probe: int = 32, max_rows: int = 1024,
        seed: int = 0, compile_step: bool = True, progress: bool = False,
        callback=None) -> SMCResult:
    """Evidence and posterior draws by tempered sequential Monte Carlo.

    ``n_particles`` particles start as prior draws. At each stage the
    population raises its ``beta`` as far as keeps the effective sample size
    of the new weights at ``ess`` of its size, adds the log mean weight to
    ln Z, resamples, and moves every particle at the new ``beta``. A move is
    a stretch move or, a ``de_fraction`` of the time, a differential-evolution
    move (the larger jumps that keep a broad, curved posterior mixing late in
    the run, where stretch moves stall). A stage's moves stop when every
    coordinate has decorrelated from the start of the stage -- correlation
    below ``decorrelation``, or no longer falling -- after at least
    ``min_moves`` and at most ``max_moves`` sweeps. ``decorrelation=None``
    instead stops after about ``accepts_per_stage`` accepted moves per
    particle.

    **ln Z is biased low when the particles stop moving.** Late in the run,
    on a narrow curved posterior, stretch and DE moves are rarely accepted;
    particles stay near where resampling left them and the part of the
    posterior they have not reached is under-counted. More particles do not
    fix it: on an oblate transit fit with no oblateness in the data, 8,192
    particles came out 0.21 nats low and 4,096 0.17 low. What does show it
    is the stage that runs to ``max_moves`` with the particles still
    correlated with where it began; ``smc`` counts those
    (``SMCResult.stalled``) and warns when there are any. A run without the
    warning is no proof of accuracy, but a run with it should not be trusted
    to better than a few tenths of a nat.

    ``n_groups > 1`` splits the particles into independent groups, each with
    its own schedule, evaluated in the same batched calls, and gives a
    standard error from their spread. Each group must be large enough by
    itself: smaller groups are each more biased, and their spread cannot
    show a bias they share. A warning fires when groups disagree by more
    than 0.5 nats, but agreement does not prove they are large enough.

    Needs a proper prior. Uses only log-likelihood values, no gradients.
    Proposals are evaluated in blocks of ``max_rows`` rows.
    ``n_probe`` posterior draws are re-evaluated on the float64 path, when
    the target has one, to report float32's error in log L.
    ``progress``/``callback`` as in :func:`anvil.run`; the callback gets
    ``("smc", stage, info)``.
    """
    _require_proper(target)
    t0 = time.perf_counter()
    G = int(n_groups)
    if G < 1 or n_particles % (2 * G):
        raise ValueError("n_particles must be a positive multiple of 2 * n_groups")
    n_g = n_particles // G
    K = n_g // 2
    d = target.dim
    if K < d:
        raise ValueError(f"{n_g} particles per group cannot span {d} dimensions "
                         f"with stretch moves; use more particles or fewer groups")
    rng = np.random.default_rng(seed)
    keys = KeyStream(seed)

    U = np.array(target.transform.sample_prior(n_particles, rng),
                 dtype=np.float32).reshape(G, n_g, d)
    step = _Sweep(target, G, K, d, _STRETCH_A, max_rows, compile_step)
    ll0, r0 = step.values(mx.array(U.reshape(-1, d)))
    LL = np.array(ll0, dtype=np.float64).reshape(G, n_g)
    R = np.array(r0, dtype=np.float64).reshape(G, n_g)
    if not np.isfinite(LL).any(axis=1).all():
        raise RuntimeError("the likelihood is not finite at any prior draw of "
                           "some group; nothing to temper from")

    beta = np.zeros(G)
    log_z = np.zeros(G)
    betas, moves, accepts, decorr = [], [], [], []
    stalled = 0
    n_evals = n_particles
    acc_prev = 0.25
    it = 0

    while (beta < 1.0).any():
        if len(moves) >= max_stages:
            raise RuntimeError(f"no convergence to beta = 1 in {max_stages} "
                               f"stages (reached {beta.min():.3g})")
        for g in np.flatnonzero(beta < 1.0):
            db = _next_beta(LL[g], beta[g], ess)
            logw = np.where(np.isfinite(LL[g]), db * LL[g], -np.inf)
            log_z[g] += _logmeanexp(logw)
            beta[g] = 1.0 if beta[g] + db >= 1.0 - 1e-12 else beta[g] + db
            # systematic resampling returns ancestors in order, so a mode's
            # copies sit together; shuffled, the red-black halves each hold a
            # share of every mode and a walker's partners come from its own
            idx = rng.permutation(_systematic(logw, rng))
            U[g], LL[g], R[g] = U[g][idx], LL[g][idx], R[g][idx]
        betas.append(beta.copy())

        Ud, LLd, Rd = mx.array(U), mx.array(LL.astype(np.float32)), mx.array(R.astype(np.float32))
        bd = mx.array(beta.astype(np.float32))[:, None]
        acc_sum = mx.zeros((G,))
        if decorrelation is None:
            M_fixed = int(min(max_moves, max(1, math.ceil(accepts_per_stage / max(acc_prev, 0.01)))))
        a0 = Ud - mx.mean(Ud, axis=1, keepdims=True)
        n0 = mx.sqrt(mx.sum(a0 * a0, axis=1))
        hist = []
        M = 0
        while True:
            de = bool(rng.uniform() < de_fraction)
            Ud, LLd, Rd, acc = step(keys.key(it), Ud, LLd, Rd, bd, de)
            acc_sum = acc_sum + acc
            b = Ud - mx.mean(Ud, axis=1, keepdims=True)
            rho = mx.sum(a0 * b, axis=1) / (n0 * mx.sqrt(mx.sum(b * b, axis=1)))
            mx.eval(acc_sum, rho)
            hist.append(np.nan_to_num(np.array(rho, dtype=np.float64), nan=1.0))
            it += 1
            M += 1
            if M >= max_moves:
                break
            if decorrelation is None:
                if M >= M_fixed:
                    break
            elif M >= min_moves and M % 4 == 0 and _settled(hist, decorrelation):
                break
        decorr.append(float(hist[-1].max()))
        U = np.array(Ud)
        LL = np.array(LLd, dtype=np.float64)
        R = np.array(Rd, dtype=np.float64)
        acc_prev = float(np.array(acc_sum).mean()) / M
        if M >= max_moves and decorr[-1] > _STALL_RHO and acc_prev < _STALL_ACCEPT:
            stalled += 1
        moves.append(M)
        accepts.append(acc_prev)
        n_evals += n_particles * M

        if progress or callback is not None:
            info = {"beta_min": float(beta.min()), "moves": M,
                    "accept": acc_prev, "decorrelation": hist[-1],
                    "elapsed": time.perf_counter() - t0}
            if progress:
                print(f"[smc] stage {len(moves)}: beta {beta.min():.3e}, "
                      f"{M} moves, accept {acc_prev:.2f}", flush=True)
            if callback is not None:
                callback("smc", len(moves), info)

    c = target.model_log_norm
    lz_mean = _logmeanexp(log_z)
    if stalled:
        warnings.warn(
            f"in {stalled} of {len(moves)} SMC stages the particles were still "
            f"correlated with the stage's start after {max_moves} moves: they "
            f"stopped exploring, and ln Z is likely biased low",
            RuntimeWarning, stacklevel=2)
    z = np.exp(log_z - log_z.max())
    se = float(z.std(ddof=1) / z.mean() / math.sqrt(G)) if G > 1 else math.nan
    if G > 1 and np.std(log_z, ddof=1) > _GROUP_SPREAD_WARN:
        warnings.warn(
            f"the {G} SMC groups disagree (ln Z spread {np.std(log_z, ddof=1):.2f} "
            f"nats): a group of {n_g} particles may be missing part of the "
            f"posterior, which biases ln Z low; use more particles",
            RuntimeWarning, stacklevel=2)

    U = U.reshape(-1, d).astype(np.float64)
    LL = LL.reshape(-1)
    tr = target.transform
    model = tr.model_np(U)

    precision = None
    if n_probe and target.has_hi:
        pick = rng.choice(U.shape[0], size=min(n_probe, U.shape[0]), replace=False)
        ll64 = np.array(target.log_likelihood_hi(mx.array(U[pick].astype(np.float32))),
                        dtype=np.float64)
        err = LL[pick] - ll64
        precision = {"n": int(pick.size), "mean": float(err.mean()),
                     "sd": float(err.std(ddof=1)) if pick.size > 1 else 0.0}

    return SMCResult(
        log_z=lz_mean + c, log_z_se=se, log_z_groups=log_z + c,
        names=list(tr.names), u=U, model=model,
        physical=tr.to_physical(model), log_likelihood=LL, betas=betas,
        moves=moves, accept=accepts, decorrelation=decorr, n_evals=n_evals,
        wall_time=time.perf_counter() - t0, precision=precision,
        stalled=stalled)


# -- the cheap tier ------------------------------------------------------------

@dataclass
class EvidenceResult:
    """An evidence estimate from :func:`bridge_evidence` or
    :func:`laplace_evidence`: ``log_z``, its standard error ``log_z_se``
    (NaN where the method gives none), the ``method``, and method-specific
    ``details``."""

    log_z: float
    log_z_se: float
    method: str
    details: dict = field(default_factory=dict)
    wall_time: float = 0.0

    def __str__(self) -> str:
        se = "" if not np.isfinite(self.log_z_se) else f" +/- {self.log_z_se:.4f}"
        extra = ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                          for k, v in self.details.items())
        return f"anvil {self.method}: ln Z = {self.log_z:.4f}{se}  ({extra})"


def _draws_of(draws):
    """Independent draws (n, d), or chains (n_draws, n_chains, d). Accepts
    either array, a :class:`Results`, or an :class:`SMCResult`."""
    if isinstance(draws, SMCResult):
        return np.asarray(draws.u, dtype=np.float64)
    if hasattr(draws, "get_chain"):
        draws = draws.get_chain()
    a = np.asarray(draws, dtype=np.float64)
    if a.ndim not in (2, 3):
        raise ValueError(f"draws must be (n, dim) or (n_draws, n_chains, dim); got {a.shape}")
    return a


def _mvt(mu, L, nu):
    d = mu.size
    logdet = 2.0 * float(np.log(np.diag(L)).sum())
    c = (math.lgamma((nu + d) / 2) - math.lgamma(nu / 2)
         - 0.5 * d * math.log(nu * math.pi) - 0.5 * logdet)

    def logpdf(x):
        z = np.linalg.solve(L, (x - mu).T).T
        return c - 0.5 * (nu + d) * np.log1p(np.einsum("ij,ij->i", z, z) / nu)

    def sample(n, rng):
        z = rng.standard_normal((n, d)) / np.sqrt(rng.chisquare(nu, n) / nu)[:, None]
        return mu + z @ L.T

    return logpdf, sample


def bridge_evidence(target: TransformedLogDensity, draws, *, nu: float = 5.0,
                    n_proposal: int | None = None, seed: int = 0,
                    max_rows: int = 1 << 15) -> EvidenceResult:
    """ln Z by Meng and Wong's optimal bridge sampling.

    ``draws`` are posterior draws in u: an array ``(n, dim)`` of independent
    draws (an :class:`SMCResult` qualifies), or ``(n_draws, n_chains, dim)``
    or a :class:`~anvil.Results`, whose autocorrelation the standard error
    then accounts for. A Student-t proposal (``nu`` degrees of freedom) is
    fitted to a random half; the bridge uses the other half and
    ``n_proposal`` proposal draws (default: as many). One batched
    evaluation of the target; the iteration runs on the host.

    The answer is only as good as the draws' mode weights: on a multimodal
    posterior, draws from MCMC chains that did not mix between modes give a
    wrong ln Z with a small error bar. ``details["proposal_ess"]`` (the
    effective sample size of the proposal-side ratios) falls when the
    proposal and posterior overlap poorly, and the standard error grows
    with it.
    """
    from .diagnostics import _ess_batched
    from .optimize import _Evaluator

    _require_proper(target)
    t0 = time.perf_counter()
    a = _draws_of(draws)
    rng = np.random.default_rng(seed)
    if a.ndim == 3:
        # chains: fit on every other chain, bridge on the rest, so each
        # bridged chain keeps the time order its autocorrelation needs
        if a.shape[1] < 2:
            raise ValueError("bridge_evidence needs at least two chains")
        fit = a[:, 0::2].reshape(-1, a.shape[-1])
        use_chains = a[:, 1::2]
        layout = use_chains.shape[:2]
        use = use_chains.reshape(-1, a.shape[-1])
    else:
        perm = rng.permutation(a.shape[0])
        fit, use = a[perm[: a.shape[0] // 2]], a[perm[a.shape[0] // 2:]]
        layout = None
    d = a.shape[-1]
    cov = np.cov(fit.T).reshape(d, d)
    L = np.linalg.cholesky(cov + 1e-12 * np.trace(cov) / d * np.eye(d))
    logq, sample = _mvt(fit.mean(axis=0), L, float(nu))
    n2 = int(n_proposal or use.shape[0])
    prop = sample(n2, rng)

    ev = _Evaluator(target.log_prob, compile_step=True, max_rows=max_rows)
    c = target.log_norm_const
    l1 = ev.value(use) + c - logq(use)          # log p/q at posterior draws
    l2 = ev.value(prop) + c - logq(prop)        # log p/q at proposal draws
    n1 = use.shape[0]
    s1, s2 = n1 / (n1 + n2), n2 / (n1 + n2)
    ls1, ls2 = math.log(s1), math.log(s2)

    def lse(a):
        m = a.max()
        return m + math.log(np.exp(a - m).sum())

    lr = float(np.median(l1[np.isfinite(l1)]))
    for _ in range(10_000):
        num = lse(l2 - np.logaddexp(ls1 + l2, ls2 + lr)) - math.log(n2)
        den = lse(-np.logaddexp(ls1 + l1, ls2 + lr)) - math.log(n1)
        new = num - den
        done = abs(new - lr) < 1e-12
        lr = new
        if done:
            break

    # relative-MSE estimate (Fruhwirth-Schnatter 2004), with the posterior
    # side's variance inflated by its integrated autocorrelation
    # in logs: with poor overlap most terms underflow, and var/mean^2 then
    # reads 0/0 in linear space
    lf1 = l2 - np.logaddexp(ls1 + l2, ls2 + lr)                 # at proposal draws
    lf2 = -np.logaddexp(ls1 + l1, ls2 + lr)                     # at posterior draws

    def rel_var(lf):
        # var(f) / mean(f)^2 = mean(f^2) / mean(f)^2 - 1
        k = lf.size
        return max(math.exp(lse(2 * lf) - math.log(k) - 2 * (lse(lf) - math.log(k))) - 1.0, 0.0)

    tau = 1.0
    if layout is not None:
        series = np.exp(lf2 - lf2.max()).reshape(layout)
        ess = float(_ess_batched(series[..., None])[0])
        if np.isfinite(ess):
            tau = max(1.0, series.size / max(ess, 1.0))
    re2 = rel_var(lf1) / n2 + tau * rel_var(lf2) / n1
    w = np.exp(l2 - l2.max())
    prop_ess = float(w.sum() ** 2 / (w * w).sum())
    if prop_ess < _BRIDGE_MIN_ESS * n2:
        warnings.warn(
            f"bridge sampling: the Student-t proposal overlaps the posterior "
            f"poorly (proposal ESS {prop_ess:.0f} of {n2}), so ln Z is "
            f"unreliable; anvil.smc does not depend on the proposal",
            RuntimeWarning, stacklevel=2)
    return EvidenceResult(
        log_z=float(lr), log_z_se=float(math.sqrt(re2)), method="bridge",
        details={"proposal_ess": prop_ess,
                 "n_posterior": int(n1), "n_proposal": n2, "nu": float(nu),
                 "tau": float(tau)},
        wall_time=time.perf_counter() - t0)


def laplace_evidence(target: TransformedLogDensity, opt=None, **optimize_kwargs) -> EvidenceResult:
    """ln Z from a Gaussian at the mode of the posterior in u.

    ``opt`` is an :class:`~anvil.OptimizeResult` from
    ``optimize(target, objective="density")``; if omitted, that call is
    made with ``optimize_kwargs``. A floor, not an answer: measured 2.4 nats
    low on a spherical transit fit, and undefined where the mode is on a
    bound or the curvature is not negative definite (it raises there).
    """
    from .optimize import optimize

    _require_proper(target)
    t0 = time.perf_counter()
    if opt is None:
        opt = optimize(target, objective="density", **optimize_kwargs)
    if opt.objective != "density":
        raise ValueError("laplace_evidence needs optimize(..., objective='density'): "
                         "the mode of the posterior in u, Jacobian included")
    if opt.cov_u is None or opt.at_bound.any():
        raise ValueError("no Laplace approximation: the mode is on a bound or its "
                         "curvature is not negative definite")
    sign, logdet = np.linalg.slogdet(opt.cov_u)
    lz = opt.log_value + 0.5 * target.dim * math.log(2 * math.pi) + 0.5 * logdet
    return EvidenceResult(log_z=float(lz), log_z_se=math.nan, method="laplace",
                          details={"converged": bool(opt.converged),
                                   "log_density_at_mode": float(opt.log_value)},
                          wall_time=time.perf_counter() - t0)

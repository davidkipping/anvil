"""Batched maximum-likelihood and MAP optimisation: anvil's third mode.

``optimize`` runs many starting points at once, the way the samplers run
many chains: every objective and gradient evaluation is one batched call
over all live starts. Each start keeps its own optimiser state; nothing is
shared between starts except the call.

Methods (measured on a 20,000-point transit likelihood and on Elome's
spherical transit model, against a float64 scipy reference optimum):

* ``"mixed"`` (default) -- L-BFGS for the large moves, then damped Newton to
  finish. L-BFGS alone stalled 0.01-0.1 nats short on Elome, whose
  curvature in u spans a condition number of ~8e6; Newton alone reached the
  optimum from fewer starts, because far from it the Hessian is a poor
  guide. Mixed reached it from 98% of starts in a ball, against 0% and 70%.
* ``"lbfgs"`` -- L-BFGS only: cheapest per iteration, enough on
  well-conditioned problems, and the choice at high dimension where
  Newton's 2 x dim gradient rows per iteration add up.
* ``"newton"`` -- damped Newton only.
* ``"de"`` -- differential evolution, for targets without gradients.

Turin's batched Adam was the baseline; it is not offered. With a fixed step
its last iterate keeps bouncing in the stiff directions: 0.6 nats short on
the transit target even from good starts, and on sharp posteriors turin
recorded it stopping hundreds to thousands of nats short.

float32 shapes the stopping rules. Near the optimum of a data-heavy
likelihood float32 cannot resolve the last step's change in value, but the
gradient still resolves the optimum. So convergence is judged by the Newton
decrement -- the gain a Newton step predicts, ``g^T (-H)^{-1} g / 2`` in
nats, from a Hessian built by central differences of float32 gradients --
and the line search accepts a step whose value has not fallen by more than
float32 resolution. The decrement has a floor of its own -- 2e-5 nats on
the transit target, 6e-4 to 1.2e-3 on Elome (20k to 100k points), whose
analytic gradient is not quite the derivative of its float32 values -- so
the default ``tol`` is 1e-2 nats, and a converged start still takes the
Newton step it has computed, which lands it far closer than that.

Where the predicted gain is small but the curvature is not negative
definite (a saddle, or a non-concave stretch such as Rosenbrock off its
valley), the step also moves along the most negative curvature, so a start
does not stop at a stationary point that is not a maximum.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .compile import CompiledOrEager
from .logdensity import LogDensity
from .transforms import TransformedLogDensity

_OBJECTIVES = ("mle", "map", "density")
_METHODS = ("mixed", "lbfgs", "newton", "de")

#: step scales tried per start by the line search, all in one batched call
#: (cuRobo's parallel line search) rather than a host sync per trial
_ALPHAS = np.array([4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.03, 0.01])
#: when no step scale helps, a start's next grid is this much shorter: the
#: grid's own span, so successive grids tile the scales without gaps. A
#: direction with no history is the unit gradient, and a posterior far
#: narrower than 0.01 in u (a tight parameter in a wide box) needs steps the
#: first grid never tries
_ALPHA_SHRINK = _ALPHAS[-1] / _ALPHAS[0]
#: |u| past which a bounded side counts as reached: tanh(7.5) = 1 - 6e-7, so
#: the parameter is within ~3e-7 of the box width of its bound, and dv/du --
#: hence the gradient -- has all but vanished
_U_BOUND = 15.0
#: clamp on every column with a finite bound, so exp() maps stay finite
_U_CLAMP = 30.0
#: |u| at which result.ball / laplace_starts place a column whose optimum is
#: on its bound: far enough out to be near it (tanh(4) = 0.9993), close
#: enough in that a sampler's Jacobian term is not already saturated
_U_START_AT_BOUND = 8.0
#: |u| past which a column with an outward gradient at the best point is
#: tried on its bound (tanh(2) = 0.964: within 2% of the box width)
_U_NEAR = 4.0
#: L-BFGS history length
_MEMORY = 10
#: iterations over which no gain of more than ``tol`` counts as stalled
_PATIENCE_LBFGS = 10
_PATIENCE_NEWTON = 5
#: L-BFGS iterations after which "mixed" hands a start to Newton regardless
_LBFGS_CAP = 200
#: finite-difference step (u units) of the gradient-difference Hessian
_FD_STEP = 1e-3
#: leading starts rescored on the target's float64 path to pick the best
_RESCORE = 16


# -- batched evaluation ----------------------------------------------------

class _Evaluator:
    """The objective and its gradient as padded, compiled batched calls.

    Row counts change as starts finish, so each call is padded to the next
    power of two (and split above ``max_rows``): a compiled function then
    sees a handful of shapes instead of one per call. Returns float64 numpy
    with non-finite values mapped to ``-inf``.
    """

    def __init__(self, f, *, compile_step: bool, max_rows: int):
        self.f = f
        self.max_rows = int(max_rows)

        def vg(u):
            out, vjps = mx.vjp(f, [u], [mx.ones(u.shape[:1], dtype=u.dtype)])
            return out[0], vjps[0]

        self._f = CompiledOrEager(f, name="the optimizer objective",
                                  enabled=compile_step)
        self._vg = CompiledOrEager(vg, name="the optimizer gradient",
                                   enabled=compile_step)
        self.n_value_rows = 0
        self.n_grad_rows = 0

    def _blocks(self, X):
        n = X.shape[0]
        for s in range(0, n, self.max_rows):
            blk = X[s:s + self.max_rows]
            m = blk.shape[0]
            size = min(1 << max(0, (m - 1).bit_length()), self.max_rows)
            pad = np.concatenate([blk, np.repeat(blk[:1], size - m, axis=0)])
            yield m, mx.array(pad.astype(np.float32))

    def value(self, X: np.ndarray) -> np.ndarray:
        out = []
        for m, xb in self._blocks(X):
            v = self._f(xb)
            mx.eval(v)
            out.append(np.array(v, dtype=np.float64)[:m])
        self.n_value_rows += X.shape[0]
        f = np.concatenate(out) if out else np.zeros(0)
        return np.where(np.isfinite(f), f, -np.inf)

    def value_and_grad(self, X: np.ndarray):
        vals, grads = [], []
        for m, xb in self._blocks(X):
            v, g = self._vg(xb)
            mx.eval(v, g)
            vals.append(np.array(v, dtype=np.float64)[:m])
            grads.append(np.array(g, dtype=np.float64)[:m])
        self.n_grad_rows += X.shape[0]
        d = X.shape[1]
        f = np.concatenate(vals) if vals else np.zeros(0)
        g = np.concatenate(grads) if grads else np.zeros((0, d))
        ok = np.isfinite(f) & np.isfinite(g).all(axis=1)
        return np.where(ok, f, -np.inf), np.where(ok[:, None], g, 0.0), ok


class _EagerValue:
    """The objective, uncompiled, for :meth:`OptimizeResult.laplace_starts`.
    A result kept around must not hold the search's compiled graphs: they
    cost memory, and ``mx.compile`` froze the target as it was when traced,
    so a target changed since would be scored as it was."""

    def __init__(self, f, max_rows: int):
        self.f = f
        self.max_rows = int(max_rows)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        out = []
        for s in range(0, X.shape[0], self.max_rows):
            v = self.f(mx.array(X[s:s + self.max_rows].astype(np.float32)))
            mx.eval(v)
            out.append(np.array(v, dtype=np.float64))
        f = np.concatenate(out) if out else np.zeros(0)
        return np.where(np.isfinite(f), f, -np.inf)


# -- objective -------------------------------------------------------------

def _objective(target: LogDensity, objective: str):
    """(f, f_hi, offset): the batched function maximised, its float64 twin
    (or None), and the float64 constant that makes its value absolute."""
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {_OBJECTIVES}, got {objective!r}")
    if objective == "density":
        f_hi = None
        if isinstance(target, TransformedLogDensity):
            f_hi = target.log_prob_hi if target.has_hi else None
            offset = target.log_norm_const if target.is_proper else target.model_log_norm
        else:
            offset = 0.0
        return target.log_prob, f_hi, offset
    if not isinstance(target, TransformedLogDensity):
        raise ValueError(
            f"objective={objective!r} needs a TransformedLogDensity, whose "
            f"likelihood and priors anvil can tell apart; a "
            f"{type(target).__name__}'s log_prob is one function. Pass "
            f"objective='density' to maximise it as written.")
    tr = target.transform
    if objective == "mle":
        f = target.log_likelihood
        f_hi = target.log_likelihood_hi if target.has_hi else None
        return f, f_hi, target.model_log_norm

    # "map": the posterior density in MODEL space -- likelihood plus the
    # declared priors, without the u-space Jacobian
    def f(u):
        out = target.log_likelihood(u)
        if tr.has_prior_terms:
            out = out + tr.log_prior(tr.to_model(u))
        return out

    f_hi = None
    if target.has_hi:
        def f_hi(u):
            u_np = np.array(u, dtype=np.float64)
            ll = np.array(target.log_likelihood_hi(u), dtype=np.float64)
            return mx.array(ll + tr.log_prior_np(tr.model_np(u_np)),
                            dtype=mx.float64)
    offset = target.model_log_norm + (tr.log_prior_norm if tr.is_proper else 0.0)
    return f, f_hi, offset


def _column_kinds(target, d):
    """Boolean masks (lower_bound, upper_bound) per column: which sides of
    each column carry a finite bound. Only TransformedLogDensity targets
    have bounds."""
    if isinstance(target, TransformedLogDensity):
        t = target.transform
        return t._bounded | t._lower, t._bounded | t._upper
    return np.zeros(d, bool), np.zeros(d, bool)


# -- result ----------------------------------------------------------------

@dataclass
class OptimizeResult:
    """What :func:`optimize` found.

    The best start's point is ``u`` (sampling space), ``model`` (model
    units) and ``physical`` (reporting units). ``value`` is the objective
    there as anvil computes it -- float64 when the target has a float64
    path -- and ``log_value`` the same with every float64 constant the
    graph leaves out added back: the absolute log-likelihood for ``"mle"``,
    the normalised model-space log posterior for ``"map"`` (when the priors
    are proper), the absolute log density for ``"density"``.

    Per start: ``u_all``, ``values``, ``status`` (``"converged"``,
    ``"saddle"``, ``"stalled"``, ``"max_iter"``, ``"failed"`` -- the
    objective was not finite at the start), ``n_iter``, ``decrement`` (the Newton decrement, in nats, at the end;
    NaN where none was computed).

    ``at_bound``: per parameter, whether the best point sits on a bound
    (the optimum is there, or beyond it).

    ``hessian_u``/``cov_u``: curvature of the objective at the best point in
    u, and its inverse; ``cov_model`` maps it to model units through the
    transform's Jacobian, and ``sd`` gives physical-unit standard
    deviations. Rows and columns of parameters on a bound are NaN. These
    are None when ``hessian=False``, without gradients, or when the
    curvature is not negative definite.

    ``optima``: distinct optima among the starts, best first, as dicts with
    ``value``, ``u`` and ``n_starts``. More than one within a few nats of
    the best is a sign of multimodality.
    """

    objective: str
    method: str
    names: list[str]
    u: np.ndarray
    model: np.ndarray
    physical: np.ndarray
    value: float
    log_value: float
    converged: bool
    status: np.ndarray
    u_all: np.ndarray
    values: np.ndarray
    n_iter: np.ndarray
    decrement: np.ndarray
    at_bound: np.ndarray
    hessian_u: np.ndarray | None
    cov_u: np.ndarray | None
    cov_model: np.ndarray | None
    sd: np.ndarray | None
    optima: list[dict]
    n_value_rows: int
    n_grad_rows: int
    wall_time: float
    _value_fn: object = field(default=None, repr=False)
    _lo_side: np.ndarray = field(default=None, repr=False)
    _hi_side: np.ndarray = field(default=None, repr=False)

    @property
    def n_starts(self) -> int:
        return self.u_all.shape[0]

    def _start_center(self) -> np.ndarray:
        """``u``, with a column on its bound brought in to
        ``_U_START_AT_BOUND`` on the same side."""
        c = self.u.copy()
        s = np.sign(c)
        c[self.at_bound] = s[self.at_bound] * _U_START_AT_BOUND
        return c

    def ball(self, n: int, spread: float = 1e-3, seed: int = 0) -> mx.array:
        """``n`` starts for :func:`anvil.run`: the best point plus
        ``spread`` x N(0, 1) in u. A small ball suits a round posterior;
        ChEES adapts to its starting spread, so a curved one is better
        started from :meth:`laplace_starts`."""
        rng = np.random.default_rng(seed)
        u = self._start_center() + spread * rng.standard_normal((n, self.u.size))
        return mx.array(u.astype(np.float32))

    def laplace_starts(self, n: int, *, scale: float = 0.5,
                       sigma_cap: float = 0.5, pull_nats: float = 25.0,
                       rounds: int = 12, seed: int = 0) -> mx.array:
        """``n`` starts drawn from the Laplace Gaussian at the best point:
        ``u + scale * N(0, cov_u)``, each principal sd capped at
        ``sigma_cap`` in u. A draw more than ``pull_nats`` below the best
        value is pulled halfway back to the centre, up to ``rounds`` times
        (turin's recipe). Columns on a bound get a 1e-3 ball."""
        if self.cov_u is None:
            raise ValueError("no Laplace covariance: optimize(..., hessian=True) "
                             "on a gradient target with a negative-definite "
                             "Hessian at the optimum")
        rng = np.random.default_rng(seed)
        d = self.u.size
        free = ~self.at_bound
        c = self._start_center()
        z = np.zeros((n, d))
        cov = self.cov_u[np.ix_(free, free)]
        w, V = np.linalg.eigh(cov)
        sd = np.minimum(np.sqrt(np.maximum(w, 0.0)), sigma_cap)
        z[:, free] = (rng.standard_normal((n, free.sum())) * sd) @ V.T * scale
        z[:, ~free] = 1e-3 * rng.standard_normal((n, (~free).sum()))
        u = c + z
        if self._value_fn is not None:
            ref = self.values.max() if np.isfinite(self.values).any() else -np.inf
            for _ in range(rounds):
                fv = self._value_fn(u)
                far = ~(fv >= ref - pull_nats)
                if not far.any():
                    break
                u[far] = c + 0.5 * (u[far] - c)
        return mx.array(u.astype(np.float32))

    def __str__(self) -> str:
        lines = [f"anvil.optimize: objective={self.objective!r} "
                 f"method={self.method!r}  {self.n_starts} starts, "
                 f"{self.wall_time:.2f} s"]
        n_conv = int(np.sum(self.status == "converged"))
        lines.append(f"  best: value {self.value:.6f} (absolute "
                     f"{self.log_value:.6f}), "
                     f"{'converged' if self.converged else 'NOT converged'}; "
                     f"{n_conv}/{self.n_starts} starts converged")
        for i, name in enumerate(self.names):
            sd = "" if self.sd is None or not np.isfinite(self.sd[i]) \
                else f" +/- {self.sd[i]:.4g}"
            flag = "  [on bound]" if self.at_bound[i] else ""
            lines.append(f"    {name:>12s} = {self.physical[i]:.10g}{sd}{flag}")
        close = [o for o in self.optima[1:] if self.value - o["value"] < 10.0]
        if close:
            lines.append(f"  {len(close)} other optimum/optima within 10 "
                         f"nats of the best: possible multimodality")
        return "\n".join(lines)


# -- the optimisers --------------------------------------------------------

#: step multipliers tried, in turn, for rows of the FD Hessian with a
#: non-finite point: an optimum beside an infeasible edge (a grazing
#: transit's b > 1 + k) puts a 1e-3 step past it
_FD_RETRY = (1.0, 0.1, 0.01)


def _fd_hessian(ev: _Evaluator, X: np.ndarray, steps: np.ndarray | None = None,
                G: np.ndarray | None = None):
    """Hessian of the objective at each row of X by central differences of
    gradients. ``steps`` (m, d, d): row k of steps[i] is the k-th step
    vector at X[i] (default ``_FD_STEP`` along each axis). Rows with a
    non-finite point are retried with shorter steps; given the gradients
    ``G`` at X, a step whose one side stays non-finite falls back to a
    one-sided difference. Returns (H (m, d, d) symmetric, ok (m,))."""
    m, d = X.shape
    if steps is None:
        steps = np.broadcast_to(_FD_STEP * np.eye(d), (m, d, d))
    H = np.zeros((m, d, d))
    ok = np.zeros(m, bool)
    todo = np.arange(m)
    for mult in _FD_RETRY:
        B = mult * steps[todo]
        # the points as the float32 graph sees them, and the steps actually
        # taken, so the solve divides by what was evaluated
        Xp = _f32(X[todo, None, :] + B)
        Xm = _f32(X[todo, None, :] - B)
        k = todo.size
        _, g, gok = ev.value_and_grad(np.concatenate([Xp, Xm], axis=1).reshape(k * 2 * d, d))
        g = g.reshape(k, 2, d, d)
        gok = gok.reshape(k, 2, d)
        Bp, Bm = Xp - X[todo, None, :], X[todo, None, :] - Xm
        both = gok[:, 0] & gok[:, 1]
        # (g(x + b+) - g(x - b-)) = H (b+ + b-): rows of B^T H, per step
        HB = np.where(both[..., None], g[:, 0] - g[:, 1], 0.0)
        BB = np.where(both[..., None], Bp + Bm, 0.0)
        if G is not None:
            Gt = G[todo, None, :]
            plus, minus = gok[:, 0] & ~gok[:, 1], gok[:, 1] & ~gok[:, 0]
            HB = np.where(plus[..., None], g[:, 0] - Gt, HB)
            BB = np.where(plus[..., None], Bp, BB)
            HB = np.where(minus[..., None], Gt - g[:, 1], HB)
            BB = np.where(minus[..., None], Bm, BB)
        have = (BB != 0).any(axis=2).all(axis=1)
        with np.errstate(all="ignore"):
            Ht = np.linalg.solve(np.where(have[:, None, None], BB, np.eye(d)), HB)
        Ht = 0.5 * (Ht + np.swapaxes(Ht, 1, 2))
        good = have & np.isfinite(Ht).all(axis=(1, 2))
        H[todo[good]] = Ht[good]
        ok[todo[good]] = True
        todo = todo[~good]
        if todo.size == 0:
            break
    return H, ok


def _newton_direction(H, g, frozen):
    """Saddle-free damped Newton ascent direction and decrement.

    Eigenvalues of -H are taken in absolute value and floored relative to
    the largest, so the step always ascends; ``frozen`` columns (on a
    bound, pushing outward) are taken out of the system. Returns
    (p, decrement = g.p/2, negative_definite, eigenvalues and
    eigenvectors of -H)."""
    n, d = g.shape
    A = -H.copy()
    g = np.where(frozen, 0.0, g)
    eye = np.eye(d)
    fz = frozen[:, :, None] | frozen[:, None, :]
    A = np.where(fz, 0.0, A) + np.where(frozen[:, :, None], eye[None], 0.0)
    w, V = np.linalg.eigh(A)
    negdef = (w > 0).all(axis=1)
    aw = np.abs(w)
    floor = np.maximum(1e-8 * aw.max(axis=1, keepdims=True), 1e-300)
    aw = np.maximum(aw, floor)
    p = np.einsum("nij,nj->ni", V, np.einsum("nji,nj->ni", V, g) / aw)
    p = np.where(frozen, 0.0, p)
    dec = 0.5 * np.einsum("ni,ni->n", g, p)
    return p, dec, negdef, w, V


class _LBFGS:
    """Per-start L-BFGS memory (minimising phi = -f)."""

    def __init__(self, n, d, m=_MEMORY):
        self.S = np.zeros((m, n, d))
        self.Y = np.zeros((m, n, d))
        self.rho = np.zeros((m, n))
        self.valid = np.zeros((m, n), bool)
        self.head = np.zeros(n, int)
        self.m = m

    def reset(self, idx):
        self.valid[:, idx] = False

    def direction(self, idx, g):
        """Ascent direction for starts ``idx`` with objective gradients
        ``g`` (len(idx), d). Two-loop recursion on phi = -f."""
        m = self.m
        S, Y, rho, valid = (self.S[:, idx], self.Y[:, idx],
                            self.rho[:, idx], self.valid[:, idx])
        head = self.head[idx]
        k = len(idx)
        q = -g.copy()                          # grad of phi
        a = np.zeros((m, k))
        rows = np.arange(k)
        # per-start slot of the i-th newest pair
        order = [(head - 1 - i) % m for i in range(m)]
        for i, j in enumerate(order):
            sj, yj = S[j, rows], Y[j, rows]
            a[i] = np.where(valid[j, rows], rho[j, rows] * np.einsum("nd,nd->n", sj, q), 0.0)
            q -= a[i][:, None] * yj
        jn = order[0]
        sn, yn = S[jn, rows], Y[jn, rows]
        sy = np.einsum("nd,nd->n", sn, yn)
        yy = np.einsum("nd,nd->n", yn, yn)
        gnorm = np.linalg.norm(g, axis=1)
        gamma = np.where(valid[jn, rows] & (yy > 0), sy / np.maximum(yy, 1e-300),
                         1.0 / np.maximum(gnorm, 1e-12))
        r = np.clip(gamma, 1e-12, 1e12)[:, None] * q
        for i in reversed(range(m)):           # oldest first
            j = order[i]
            sj, yj = S[j, rows], Y[j, rows]
            b = np.where(valid[j, rows], rho[j, rows] * np.einsum("nd,nd->n", yj, r), 0.0)
            r += sj * (a[i] - b)[:, None]
        p = -r                                  # descent on phi = ascent on f
        bad = ~(np.einsum("nd,nd->n", p, g) > 0) | ~np.isfinite(p).all(axis=1)
        if bad.any():
            # not an ascent direction: forget the history, take the gradient
            self.reset(idx[bad])
            p[bad] = g[bad] / np.maximum(gnorm[bad], 1e-12)[:, None]
        return p, 0.5 * np.einsum("nd,nd->n", g, p)

    def update(self, idx, s, y_phi):
        """Store pairs where the curvature condition holds (phi convex
        along s), skipping float32 noise."""
        sy = np.einsum("nd,nd->n", s, y_phi)
        keep = (sy > 1e-10 * np.linalg.norm(s, axis=1) * np.linalg.norm(y_phi, axis=1)) \
            & np.isfinite(sy) & (sy > 0)
        for i, ok, si, yi, syi in zip(idx, keep, s, y_phi, sy):
            if ok:
                h = self.head[i]
                self.S[h, i], self.Y[h, i] = si, yi
                self.rho[h, i] = 1.0 / syi
                self.valid[h, i] = True
                self.head[i] = (h + 1) % self.m


def _clamp(X, lo_side, hi_side):
    X = X.copy()
    lim = lo_side | hi_side
    X[:, lim] = np.clip(X[:, lim], -_U_CLAMP, _U_CLAMP)
    return X


def _on_bound(X, lo_side, hi_side):
    return ((X <= -_U_BOUND) & lo_side) | ((X >= _U_BOUND) & hi_side)


def _frozen(X, G, lo_side, hi_side):
    """Columns on a bound whose gradient pushes further out: held fixed."""
    out_lo = (X <= -_U_BOUND) & lo_side & (G <= 0)
    out_hi = (X >= _U_BOUND) & hi_side & (G >= 0)
    return out_lo | out_hi


def _f32(X):
    """Round to what the float32 graph sees, so host state and evaluated
    points agree exactly."""
    return X.astype(np.float32).astype(np.float64)


def _gradient_search(ev, X, method, *, max_iter, tol, lo_side, hi_side,
                     report):
    n, d = X.shape
    F, G, ok = ev.value_and_grad(X)
    status = np.where(ok, "max_iter", "failed").astype(object)
    active = ok.copy()
    phase = np.zeros(n, int) if method in ("mixed", "lbfgs") else np.ones(n, int)
    n_iter = np.zeros(n, int)
    n_lbfgs = np.zeros(n, int)
    decrement = np.full(n, np.nan)
    hist = [F.copy()]
    newton_since = np.zeros(n, int)
    newton_F0 = F.copy()
    last_negdef = np.ones(n, bool)
    step_scale = np.ones(n)
    mem = _LBFGS(n, d)
    tol_f32 = 4.0 * np.finfo(np.float32).eps

    for it in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        Xa, Fa, Ga = X[idx], F[idx], G[idx]
        frz = _frozen(Xa, Ga, lo_side, hi_side)
        P = np.zeros_like(Xa)
        done_now = np.zeros(idx.size, bool)
        final = np.zeros(idx.size, bool)

        # -- L-BFGS starts
        lb = phase[idx] == 0
        if lb.any():
            gl = np.where(frz[lb], 0.0, Ga[lb])
            p, gain = mem.direction(idx[lb], gl)
            P[lb] = np.where(frz[lb], 0.0, p)
            past = hist[-_PATIENCE_LBFGS] if len(hist) >= _PATIENCE_LBFGS else None
            stalled = (np.zeros(lb.sum(), bool) if past is None
                       else (Fa[lb] - past[idx[lb]] < tol))
            if method == "lbfgs":
                # its own estimate runs low on an ill-conditioned problem, so
                # ask much more of it; the Newton check at the end certifies
                small = gain < 1e-2 * tol
            else:
                small = gain < tol
            if method == "lbfgs":
                conv = small
                status[idx[lb][conv]] = "converged"
                status[idx[lb][stalled & ~conv]] = "stalled"
                done_now[np.flatnonzero(lb)[conv | stalled]] = True
            else:
                switch = small | stalled | (n_lbfgs[idx[lb]] >= _LBFGS_CAP)
                phase[idx[lb][switch]] = 1
                newton_since[idx[lb][switch]] = 0
                newton_F0[idx[lb][switch]] = Fa[lb][switch]
                n_lbfgs[idx[lb]] += 1

        # -- Newton starts (including any that just switched)
        nw = (phase[idx] == 1) & ~done_now
        if nw.any():
            j = np.flatnonzero(nw)
            H, hok = _fd_hessian(ev, Xa[j], G=Ga[j])
            p, dec, negdef, w, V = _newton_direction(H, Ga[j], frz[j])
            decrement[idx[j]] = np.where(hok, dec, np.nan)
            last_negdef[idx[j]] = negdef
            conv = hok & (dec < tol) & negdef
            # Little predicted gain but curvature that is not negative
            # definite: a saddle, or a non-concave stretch (Rosenbrock off its
            # valley). Step along the most negative curvature as well, far
            # enough that a quadratic would gain 100 tol there.
            esc = np.flatnonzero(hok & (dec < tol) & ~negdef)
            if esc.size:
                wk = w[esc, 0]
                v = V[esc, :, 0]
                sgn = np.where(np.einsum("nd,nd->n", Ga[j][esc], v) < 0, -1.0, 1.0)
                delta = np.minimum(np.sqrt(200.0 * tol / np.maximum(np.abs(wk), 1e-300)), 1.0)
                p[esc] += (sgn * delta)[:, None] * np.where(frz[j][esc], 0.0, v)
            newton_since[idx[j]] += 1
            stalled = (newton_since[idx[j]] > _PATIENCE_NEWTON) \
                & (Fa[j] - newton_F0[idx[j]] < tol) & ~conv
            reset = newton_since[idx[j]] > _PATIENCE_NEWTON
            newton_since[idx[j][reset]] = 0
            newton_F0[idx[j][reset]] = Fa[j][reset]
            status[idx[j][conv]] = "converged"
            # stopped without climbing where the curvature says it could:
            # stationary or flat, but not a maximum
            status[idx[j][stalled]] = np.where(negdef[stalled], "stalled", "saddle")
            # no Hessian even with shorter and one-sided steps: the start's
            # value is finite, so it stops uncertified rather than failed
            status[idx[j][~hok]] = "stalled"
            done_now[j[conv | stalled | ~hok]] = True
            # a converged start still takes the Newton step it has just
            # computed: with quadratic convergence that one step lands it far
            # inside tol, at the price of rows already in the batched call
            final[j[conv]] = True
            P[j] = p

        active[idx[done_now]] = False
        go = ~done_now | final
        if not go.any():
            continue
        gi, Xg, Pg, Fg = idx[go], Xa[go], P[go], Fa[go]

        # -- parallel line search: every step scale in one call
        K = _ALPHAS.size
        alphas = step_scale[gi][:, None] * _ALPHAS[None, :]
        cand = _f32(_clamp((Xg[:, None, :] + alphas[:, :, None] * Pg[:, None, :])
                           .reshape(-1, d), lo_side, hi_side))
        Fc = ev.value(cand).reshape(-1, K)
        k = np.argmax(Fc, axis=1)
        Fbest = Fc[np.arange(gi.size), k]
        slack = np.where(final[go], 0.0, tol_f32 * np.maximum(1.0, np.abs(Fg)))
        accept = Fbest >= Fg - slack
        Xn = cand.reshape(-1, K, d)[np.arange(gi.size), k]
        # no step scale helped: drop that start's L-BFGS history so its next
        # direction is the gradient rather than the same failed one
        rej = gi[~accept]
        mem.reset(rej[phase[rej] == 0])
        # and look shorter next time, or a rejected start repeats the same
        # grid until its patience runs out
        step_scale[rej] *= _ALPHA_SHRINK
        step_scale[gi[accept]] = 1.0

        a_idx = np.flatnonzero(accept)
        if a_idx.size:
            Fn, Gn, gok = ev.value_and_grad(Xn[a_idx])
            keep = gok
            ai = gi[a_idx][keep]
            s = Xn[a_idx][keep] - X[ai]
            y_phi = -(Gn[keep] - G[ai])
            lbk = phase[ai] == 0
            if lbk.any():
                mem.update(ai[lbk], s[lbk], y_phi[lbk])
            X[ai], F[ai], G[ai] = Xn[a_idx][keep], Fn[keep], Gn[keep]
        n_iter[gi] += 1
        hist.append(F.copy())
        if len(hist) > _PATIENCE_LBFGS + 1:
            hist.pop(0)
        report(it + 1, F, active)

    if method == "lbfgs":
        # L-BFGS's own predicted gain is unreliable when the problem is
        # ill-conditioned (it read < tol on Elome 0.1 nats from the optimum),
        # so a start it calls converged must pass one Newton decrement
        c = np.flatnonzero(status == "converged")
        if c.size:
            H, hok = _fd_hessian(ev, X[c], G=G[c])
            _, dec, negdef, _, _ = _newton_direction(
                H, G[c], _frozen(X[c], G[c], lo_side, hi_side))
            decrement[c] = np.where(hok, dec, np.nan)
            status[c[~(hok & (dec < tol) & negdef)]] = "stalled"

    return X, F, status, n_iter, decrement


def _de_partners(rng, n):
    """Two partners per member for rand/1: distinct from the member and from
    each other. Drawn independently, r1 == r2 makes the difference vector
    zero and the move a pure pull toward the best (a third of the time at
    n = 4)."""
    i = np.arange(n)
    r1 = rng.integers(0, n - 1, size=n)
    r1 += r1 >= i
    r2 = rng.integers(0, n - 2, size=n)
    lo, hi = np.minimum(i, r1), np.maximum(i, r1)
    r2 += r2 >= lo
    r2 += r2 >= hi
    return r1, r2


def _de_search(ev, X, rng, *, max_iter, tol, lo_side, hi_side, report):
    """Differential evolution, rand-to-best/1/bin with dithered F, the
    starts as the population. Converges when every member is within
    ``tol`` of the best."""
    n, d = X.shape
    if n < 4:
        raise ValueError("method='de' needs at least 4 starts")
    F = ev.value(X)
    n_iter = np.zeros(n, int)
    converged = False
    cr = 0.9
    for it in range(max_iter):
        b = int(np.argmax(F))
        fw = rng.uniform(0.5, 1.0, size=(n, 1))
        r1, r2 = _de_partners(rng, n)
        V = X + fw * (X[b] - X) + fw * (X[r1] - X[r2])
        cross = rng.uniform(size=(n, d)) < cr
        cross[np.arange(n), rng.integers(0, d, size=n)] = True
        T = _f32(_clamp(np.where(cross, V, X), lo_side, hi_side))
        FT = ev.value(T)
        better = FT >= F
        X[better], F[better] = T[better], FT[better]
        n_iter += 1
        report(it + 1, F, np.ones(n, bool))
        if np.isfinite(F).all() and F.max() - F.min() < tol:
            converged = True
            break
    status = np.full(n, "converged" if converged else "max_iter", dtype=object)
    status[~np.isfinite(F)] = "failed"
    return X, F, status, n_iter, np.full(n, np.nan)


# -- the entry point -------------------------------------------------------

#: candidate optima merged by the midpoint test (pairs are one batched call)
_MAX_CANDIDATES = 64


def _distinct_optima(ev, U, F, lo_side, hi_side, max_optima=10):
    """Distinct optima among the starts, best first.

    Starts within 0.05 in u and 0.1 nats are one optimum. That alone splits
    one optimum whose starts stop far apart in u -- on a bound, where any
    |u| from 15 to the clamp is the bound, or on a broad peak -- so
    candidates of nearly equal value are then merged when the objective does
    not dip between them: a single peak has no valley at the midpoint, and
    two separate ones do."""
    order = np.argsort(-F)
    reps: list[dict] = []
    for i in order:
        if not np.isfinite(F[i]):
            break
        for r in reps:
            if abs(r["value"] - F[i]) < 0.1 and np.max(np.abs(r["u"] - U[i])) < 0.05:
                r["n_starts"] += 1
                break
        else:
            reps.append({"value": float(F[i]), "u": U[i].copy(), "n_starts": 1})
    reps = reps[:_MAX_CANDIDATES]
    pairs = [(a, b) for a in range(len(reps)) for b in range(a + 1, len(reps))
             if abs(reps[a]["value"] - reps[b]["value"]) < 0.1]
    if pairs:
        mid = np.array([0.5 * (reps[a]["u"] + reps[b]["u"]) for a, b in pairs])
        fm = ev.value(_f32(_clamp(mid, lo_side, hi_side)))
        parent = list(range(len(reps)))

        def root(a):
            while parent[a] != a:
                a = parent[a]
            return a
        for (a, b), f in zip(pairs, fm):
            if f >= min(reps[a]["value"], reps[b]["value"]) - 0.1:
                ra, rb = root(a), root(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)   # keep the better one
        merged: dict[int, dict] = {}
        for a, r in enumerate(reps):
            ra = root(a)
            if ra in merged:
                merged[ra]["n_starts"] += r["n_starts"]
            else:
                merged[ra] = dict(r)
        reps = [merged[k] for k in sorted(merged)]
    return reps[:max_optima]


def _certified(ev, X, F, status, b, tol, lo_side, hi_side):
    """Whether the best start ``b`` sits on a certified maximum: it converged
    itself, or a converged start within ``tol`` below it lies on the same
    peak -- the objective does not dip between them. A start counts as
    converged once under ``tol`` nats remain, so the highest start is often
    a stalled one a little above a hundred converged ones at the same
    optimum; picking a converged one instead would hand back a lower point
    (0.008 nats lower on one of turin's fits), and reporting "NOT converged"
    would be wrong."""
    if status[b] == "converged":
        return True
    c = np.flatnonzero((status == "converged") & np.isfinite(F) & (F >= F[b] - tol))
    if c.size == 0:
        return False
    c = c[np.argsort(-F[c])][:_RESCORE]
    mid = _f32(_clamp(0.5 * (X[c] + X[b]), lo_side, hi_side))
    fm = ev.value(mid)
    return bool(np.any(fm >= np.minimum(F[c], F[b]) - tol))


def _settle_bounds(ev, u, lo_side, hi_side):
    """Move columns whose optimum is on a bound onto it.

    The stopping rule ends a start once under ``tol`` nats remain, and for a
    parameter whose optimum is a bound that happens while it is still a few
    units of u short (u ~ 6 for a likelihood rising 5 nats across the box),
    where ``at_bound`` cannot see it. A column past ``|u| = 4`` whose gradient
    still points out is set to the bound; the move is kept if the objective
    does not fall by more than float32 resolution. Returns the moved point,
    or None if nothing moved."""
    f0, g, ok = ev.value_and_grad(u[None])
    if not ok[0]:
        return None
    g = g[0]
    out = ((u > _U_NEAR) & hi_side & (g > 0)) | ((u < -_U_NEAR) & lo_side & (g < 0))
    cols = np.flatnonzero(out)
    if cols.size == 0:
        return None
    moved = u.copy()
    moved[cols] = np.sign(u[cols]) * _U_BOUND
    slack = 4.0 * np.finfo(np.float32).eps * max(1.0, abs(f0[0]))
    if ev.value(moved[None])[0] >= f0[0] - slack:
        return moved
    return None


def _jacobian_diag(target, u):
    """d(model)/du at u (float64, host), per column."""
    t = target.transform
    lo = np.array([s.lo for s in t.specs])
    hi = np.array([s.hi for s in t.specs])
    scale = np.array([s.scale for s in t.specs])
    sig = 1.0 / (1.0 + np.exp(-u))
    j = np.where(t._bounded, np.where(t._bounded, hi - lo, 1.0) * sig * (1 - sig), scale)
    j = np.where(t._lower, scale * np.exp(u), j)
    j = np.where(t._upper, scale * np.exp(-u), j)
    return j


def _laplace(ev, u, at_bound):
    """Hessian at u: one axis-aligned pass, then a second along its
    eigenvectors at 0.3 sd (capped at 0.5 in u), as SquishierPlanet and
    turin do. Returns (H, cov) or (None, None) if not negative definite."""
    d = u.size
    _, g0, gok = ev.value_and_grad(u[None])
    G = g0 if gok[0] else None
    H, ok = _fd_hessian(ev, u[None], G=G)
    if not ok[0]:
        return None, None
    H = H[0]
    free = ~at_bound
    A = -H[np.ix_(free, free)]
    w, V = np.linalg.eigh(A)
    if not (w > 0).all():
        return H, None
    sd = np.minimum(1.0 / np.sqrt(w), 0.5 / 0.3)
    B = _FD_STEP * np.eye(d)
    B[np.ix_(free, free)] = (V * (0.3 * sd)).T     # rows: step vectors
    H2, ok2 = _fd_hessian(ev, u[None], B[None], G=G)
    if ok2[0]:
        A2 = -H2[0][np.ix_(free, free)]
        if (np.linalg.eigvalsh(A2) > 0).all():
            H, A = H2[0], A2
    cov = np.full((d, d), np.nan)
    cov[np.ix_(free, free)] = np.linalg.inv(A)
    return H, cov


def optimize(target: LogDensity, u0=None, *, objective: str = "mle",
             method: str = "mixed", n_starts: int = 128,
             start_spread: float = 0.1, max_iter: int = 1000,
             tol: float = 1e-2, hessian: bool = True, seed: int = 0,
             compile_step: bool = True, max_rows: int = 4096,
             progress: bool | int = False, callback=None) -> OptimizeResult:
    """Find the maximum likelihood (or MAP, or density mode) from many
    starts at once.

    ``objective``:

    * ``"mle"`` -- the log-likelihood alone: no prior, no Jacobian. Needs a
      :class:`~anvil.TransformedLogDensity`, whose priors are declared on
      its ParamSpecs and so can be left out.
    * ``"map"`` -- likelihood plus declared log-priors, in model space (no
      Jacobian): the mode of the posterior in the units you wrote it in.
    * ``"density"`` -- the target's own ``log_prob``: for a plain
      LogDensity, the function as written; for a TransformedLogDensity, the
      mode in u (Jacobian included), which is what a sampler sees.

    The search runs in u, so no step leaves a box; without the Jacobian
    the optimum in u maps exactly onto the model-space one. An optimum on a
    bound lies at u -> +/-inf: such a column is held once it passes
    ``|u| = 15`` with the gradient still pushing out; and since a start can
    converge (under ``tol`` nats left) a few units of u short of that, a
    column of the best point past ``|u| = 4`` with an outward gradient is moved
    onto its bound if that does not lower the objective. Both are flagged
    in ``at_bound``.

    ``u0``: starts in u, ``(n, dim)``; a single point ``(dim,)`` is spread
    into ``n_starts`` with sd ``start_spread``; None draws ``n_starts``
    from the prior (needs a TransformedLogDensity with proper priors).
    Many wide starts find the global optimum more reliably than a few good
    ones; failed starts cost little.

    ``tol`` (nats): a start has converged when the Newton step predicts a
    gain below ``tol`` and the curvature is negative definite. Set it above
    the target's float32 floor (see the module notes). ``method="lbfgs"``
    stops on its own, stricter estimate, then checks each start it calls
    converged with one Newton decrement; on an ill-conditioned problem many
    end ``"stalled"``, which is the signal to use ``"mixed"``.

    ``hessian``: also return the curvature and Laplace covariance at the
    best point (2 x dim gradient evaluations, twice).

    ``max_rows``: largest batch sent to the target in one call; lower it if
    the target's memory per row is large.

    ``progress``/``callback``: as in :func:`anvil.run`; the callback gets
    ``("optimize", iteration, info)``.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if method != "de" and not target.supports_grad:
        raise ValueError(f"method={method!r} needs gradients and the target "
                         f"has supports_grad=False; use method='de'")
    t0 = time.perf_counter()
    f, f_hi, offset = _objective(target, objective)
    d = target.dim
    rng = np.random.default_rng(seed)

    if u0 is None:
        if not isinstance(target, TransformedLogDensity):
            raise ValueError("u0 is required for a target without a Transform")
        if not target.is_proper:
            raise ValueError("u0=None draws starts from the prior, which is "
                             "improper here; pass u0")
        X = np.array(target.transform.sample_prior(n_starts, rng), dtype=np.float64)
    else:
        X = np.array(u0, dtype=np.float64)
        if X.ndim == 1:
            X = X[None] + start_spread * rng.standard_normal((n_starts, X.size))
        if X.ndim != 2 or X.shape[1] != d:
            raise ValueError(f"u0 must have shape (n, {d}) or ({d},), got {X.shape}")
    lo_side, hi_side = _column_kinds(target, d)
    X = _f32(_clamp(X, lo_side, hi_side))

    ev = _Evaluator(f, compile_step=compile_step, max_rows=max_rows)

    every = 0
    if progress is True or (not progress and callback is not None):
        every = 25
    elif progress:
        every = max(1, int(progress))

    def report(it, F, active):
        if not every or it % every:
            return
        fin = F[np.isfinite(F)]
        info = {"best": float(fin.max()) if fin.size else -math.inf,
                "n_active": int(active.sum()), "n_starts": F.size}
        if progress:
            print(f"optimize {it:5d}: best {info['best']:.6f}, "
                  f"{info['n_active']}/{F.size} starts still moving", flush=True)
        if callback is not None:
            callback("optimize", it, info)

    if method == "de":
        X, F, status, n_iter, decrement = _de_search(
            ev, X, rng, max_iter=max_iter, tol=tol, lo_side=lo_side,
            hi_side=hi_side, report=report)
    else:
        X, F, status, n_iter, decrement = _gradient_search(
            ev, X, method, max_iter=max_iter, tol=tol, lo_side=lo_side,
            hi_side=hi_side, report=report)

    if not np.isfinite(F).any():
        raise RuntimeError("every start failed: the objective is not finite "
                           "at any of them, or at any point they reached")
    status = np.asarray(status, dtype=object)
    b = int(np.argmax(F))
    value = float(F[b])
    if f_hi is not None:
        # float32 cannot rank starts that all sit at the optimum (on Elome
        # its pick was up to 7e-4 nats below the float64 best), so rescore
        # the leading candidates on the float64 path and keep its winner
        top = np.argsort(-F)[:_RESCORE]
        top = top[np.isfinite(F[top])]
        v64 = np.array(f_hi(mx.array(X[top].astype(np.float32))), dtype=np.float64)
        b = int(top[np.nanargmax(v64)])
        value = float(np.nanmax(v64))
    u = X[b].copy()
    if target.supports_grad:
        u_set = _settle_bounds(ev, u, lo_side, hi_side)
        if u_set is not None:
            u = X[b] = u_set
            F[b] = ev.value(u[None])[0]
            value = float(F[b]) if f_hi is None else float(np.array(
                f_hi(mx.array(u[None].astype(np.float32))), dtype=np.float64)[0])
    at_bound = _on_bound(u[None], lo_side, hi_side)[0]
    converged = _certified(ev, X, F, status, b, tol, lo_side, hi_side)

    H = cov_u = cov_model = sd = None
    if hessian and target.supports_grad:
        H, cov_u = _laplace(ev, u, at_bound)

    if isinstance(target, TransformedLogDensity):
        tr = target.transform
        model = tr.model_np(u[None])[0]
        physical = tr.to_physical(model)
        if cov_u is not None:
            J = _jacobian_diag(target, u)
            cov_model = cov_u * np.outer(J, J)
            sd = np.sqrt(np.diag(cov_model)) * np.abs(tr._rep_scale)
        names = list(tr.names)
    else:
        model = physical = u.copy()
        if cov_u is not None:
            cov_model = cov_u
            sd = np.sqrt(np.diag(cov_u))
        names = [f"x{i}" for i in range(d)]

    return OptimizeResult(
        objective=objective, method=method, names=names, u=u, model=model,
        physical=physical, value=value, log_value=value + offset,
        converged=converged, status=status, u_all=X,
        values=F, n_iter=n_iter, decrement=decrement, at_bound=at_bound,
        hessian_u=H, cov_u=cov_u, cov_model=cov_model, sd=sd,
        optima=_distinct_optima(ev, X, F, lo_side, hi_side), n_value_rows=ev.n_value_rows,
        n_grad_rows=ev.n_grad_rows, wall_time=time.perf_counter() - t0,
        _value_fn=_EagerValue(f, max_rows), _lo_side=lo_side, _hi_side=hi_side)

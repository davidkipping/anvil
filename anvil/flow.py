"""Flow mode: sampling in the coordinates of a normalizing flow, for
posteriors with curved degeneracies. Optional; nothing else in anvil uses it.

A flow ``T`` maps a latent ``z`` to anvil's sampling coordinates ``u``. For
any fixed bijection, ::

    log p_z(z) = log p_u(T(z)) + log |det dT/dz|

is the same posterior in other coordinates. A sampler run on it, with its
draws mapped through ``T``, is exact whatever ``T`` is. The priors are
untouched: they are inside ``log p_u``. ``T`` only decides how hard the
sampling is, and the flow is trained to make the posterior close to a
standard normal in ``z``, where curved degeneracies become straight ones.
The flow has to be frozen before the draws that are kept.

The pieces, in the order :func:`train` uses them:

* :class:`Flow` -- whitening, then affine coupling layers (RealNVP) with
  smooth conditioners. :meth:`Flow.fit` trains it by maximum likelihood on
  draws.
* :class:`FlowTarget` -- a target in the flow's coordinates. ``run`` takes
  it like any other :class:`~anvil.LogDensity` and records its draws (and
  log-probabilities) in the target's own coordinates.
* :func:`train` -- tempered SMC from the prior, a flow fitted to its
  particles, a few short ensemble rounds in flow coordinates with a refit
  after each, then a final frozen flow and a :class:`FlowReport`.
* :func:`refine` -- one more fit, on the draws of a production run.
* :func:`anvil.flow_evidence` -- importance sampling with the flow as the
  proposal (optional, never called by ``run``).

Measured (research notes in the repository; 512 walkers, M2 Max):

* Oblate transit fits (hybrid4 limb darkening, 10 parameters, 2,000 and
  10,000 points, 3 seeds each): no plain or linearly preconditioned sampler
  converged (R-hat 2.8-35). The ensemble in flow coordinates agreed with
  itself across seeds to 0.02-0.07 posterior sd, at R-hat 1.015-1.027: a few
  walkers mix slowly through the tails. One more refit on production draws
  and a run four times longer took the 2,000-point fit to R-hat 1.006; the
  10,000-point fit stayed at 1.013.
* Spherical hybrid4 (curved limb-darkening prior): flow samplers at R-hat
  1.002-1.004 against 1.09-1.48 plain, at 30-90x the effective samples per
  second.
* Spherical hybrid2 (near-Gaussian posterior): no gain. Plain samplers were
  2-4x faster; the report's fit gain (0.1 nats, against 5-14 on the curved
  targets) says so before any sampling.
* ChEES works on a :class:`FlowTarget`, but on the oblate fit it trailed the
  ensemble 20-40x: a few walkers in badly mapped tails set its shared step
  size. The ensemble is the recommended kernel.
* A flow trained on draws from one basin (an optimiser's, say) fits that
  basin; sampling then speeds up there and reports healthy diagnostics while
  missing the rest. :func:`train` starts from the prior for this reason.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
import warnings
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from .logdensity import LogDensity

_LOG2PI = math.log(2.0 * math.pi)
#: fit gain over whitening (nats) below which a flow is not worth its cost:
#: 0.02-0.11 on a near-Gaussian transit posterior, 5-14 on curved ones
_GAIN_WORTHWHILE = 1.0
#: importance-sampling efficiency below which the flow is a poor fit; the
#: oblate fits that sampled well had 0.03-0.32
_EFF_POOR = 0.01
#: a validation loss this far above the whitened one (nats) is a diverged fit
_DIVERGED = 1.0


class Flow:
    """Whitening followed by ``n_layers`` affine coupling layers (RealNVP),
    each conditioned by a two-layer tanh network of width ``hidden``, so the
    latent target has continuous gradients. Scales are bounded by ``s_max``
    in log space. The last layer of every conditioner starts at zero: an
    untrained flow is the whitening alone.

    The default size (8 x 64, about 40,000 parameters at 10 dimensions) is
    the one measured. On a 7-dimensional curved target 4 x 32 needed 2-4x
    more gradients per iteration, and smaller flows lost the tails; a
    12 x 128 flow on the oblate fit was no better and once diverged in
    training.

    ``dim >= 2``: a coupling layer splits the coordinates in two."""

    def __init__(self, dim: int, n_layers: int = 8, hidden: int = 64,
                 s_max: float = 3.0, seed: int = 0):
        if dim < 2:
            raise ValueError("a coupling flow needs dim >= 2")
        rng = np.random.default_rng(seed)
        self.dim, self.n_layers, self.hidden = int(dim), int(n_layers), int(hidden)
        self.s_max = float(s_max)
        self.mu = mx.zeros((dim,))
        self.L = mx.eye(dim)
        self.Linv = mx.eye(dim)
        self.logdet_L = 0.0
        self.splits, self.params = [], []
        half = dim // 2
        for k in range(n_layers):
            if k % 2 == 0:
                perm = rng.permutation(dim)
                ia, ib = perm[:half], perm[half:]
            else:
                ia, ib = ib, ia
            self.splits.append(self._split(ia, ib))
            da, db = len(ia), len(ib)

            def w(m, n):
                return mx.array((rng.standard_normal((m, n)) / math.sqrt(m)).astype(np.float32))
            self.params.append({
                "W1": w(da, hidden), "b1": mx.zeros((hidden,)),
                "W2": w(hidden, hidden), "b2": mx.zeros((hidden,)),
                "W3": mx.zeros((hidden, 2 * db)), "b3": mx.zeros((2 * db,)),
            })
        # what every fit starts from: zero last layers, so the whitening alone
        self._init_params = [dict(p) for p in self.params]
        #: the :class:`FlowReport` from :func:`train` or :func:`refine`
        self.report: FlowReport | None = None
        #: draws in u that :meth:`start` picks from (set by :func:`train`)
        self.pool: np.ndarray | None = None

    @staticmethod
    def _split(ia, ib):
        ia, ib = np.asarray(ia), np.asarray(ib)
        inv = np.argsort(np.concatenate([ia, ib]))
        return mx.array(ia), mx.array(ib), mx.array(inv)

    # -- the map ----------------------------------------------------------
    def _st(self, p, xa, db):
        h = mx.tanh(xa @ p["W1"] + p["b1"])
        h = mx.tanh(h @ p["W2"] + p["b2"])
        out = h @ p["W3"] + p["b3"]
        return self.s_max * mx.tanh(out[:, :db] / self.s_max), out[:, db:]

    def forward(self, z: mx.array, params=None):
        """``z -> (u, log |det du/dz|)``, batched over rows."""
        params = self.params if params is None else params
        ld = mx.zeros(z.shape[:1])
        for (ia, ib, inv), p in zip(self.splits, params):
            za, zb = z[:, ia], z[:, ib]
            s, t = self._st(p, za, ib.shape[0])
            z = mx.concatenate([za, zb * mx.exp(s) + t], axis=1)[:, inv]
            ld = ld + mx.sum(s, axis=1)
        return self.mu + z @ self.L.T, ld + self.logdet_L

    def inverse(self, u: mx.array, params=None):
        """``u -> (z, log |det dz/du|)``, batched over rows."""
        params = self.params if params is None else params
        ld = mx.zeros(u.shape[:1])
        z = (u - self.mu) @ self.Linv.T
        for (ia, ib, inv), p in zip(reversed(self.splits), reversed(params)):
            za, zb = z[:, ia], z[:, ib]
            s, t = self._st(p, za, ib.shape[0])
            z = mx.concatenate([za, (zb - t) * mx.exp(-s)], axis=1)[:, inv]
            ld = ld - mx.sum(s, axis=1)
        return z, ld - self.logdet_L

    def log_q(self, u: mx.array, params=None) -> mx.array:
        """Log-density of the flow's own distribution at ``u``."""
        z, ld = self.inverse(u, params)
        return -0.5 * mx.sum(z * z, axis=1) - 0.5 * self.dim * _LOG2PI + ld

    def sample(self, n: int, key=None, seed: int = 0):
        """``n`` draws from the flow's distribution: ``(u, log q(u))``."""
        key = mx.random.key(seed) if key is None else key
        z = mx.random.normal((n, self.dim), key=key)
        u, ld = self.forward(z)
        return u, -0.5 * mx.sum(z * z, axis=1) - 0.5 * self.dim * _LOG2PI - ld

    @property
    def n_params(self) -> int:
        return sum(int(v.size) for p in self.params for v in p.values())

    # -- training ---------------------------------------------------------
    def _set_whitening(self, x: np.ndarray) -> None:
        x = x.astype(np.float64)
        mu = x.mean(axis=0)
        cov = np.atleast_2d(np.cov((x - mu).T))
        L = np.linalg.cholesky(cov + 1e-10 * np.trace(cov) / self.dim * np.eye(self.dim))
        self.mu = mx.array(mu.astype(np.float32))
        self.L = mx.array(L.astype(np.float32))
        self.Linv = mx.array(np.linalg.inv(L).astype(np.float32))
        self.logdet_L = float(np.sum(np.log(np.diag(L))))

    def fit(self, draws, *, max_steps: int = 6000, batch: int = 1024, lr: float = 2e-3,
            patience: int = 10, check_every: int = 100, retries: int = 2,
            seed: int = 0) -> dict:
        """Train by maximum likelihood on ``draws`` (``(n, dim)``, or a
        :class:`~anvil.Results` or chain array, in u), holding 10% out.

        Adam with a cosine-decayed learning rate; every ``check_every``
        steps the hold-out loss is checked, and training stops after
        ``patience`` checks without improvement. The best checkpoint is
        kept, and the whitening alone counts as the first checkpoint, so a
        fit can never come back worse than whitening. A fit that diverges
        (a non-finite loss, or a hold-out loss 1 nat above the whitening's)
        is retried from scratch at a quarter of the learning rate, up to
        ``retries`` times. Every fit starts from the untrained flow, so
        fitting a flow again replaces what it learned before; a
        :class:`FlowTarget` made from it earlier then has a different
        :attr:`~FlowTarget.resume_tag`, and resuming its runs raises.

        Exact duplicate rows (SMC particles that share an ancestor) are
        dropped first: a hold-out row with a copy in the training set
        makes the hold-out loss track the training loss.

        Returns the training facts: ``gain_nats`` (hold-out loss of the
        whitening alone minus the flow's, the measure of how far from
        Gaussian the draws are), ``val_nll``, ``steps``, ``n_train``,
        ``retries``, ``seconds``."""
        t0 = time.perf_counter()
        x = _as_draws(draws, self.dim)
        x = x[np.all(np.isfinite(x), axis=1)]
        x = np.unique(x, axis=0)
        if len(x) < 20:
            raise ValueError(f"fit needs at least 20 finite draws, got {len(x)}")
        rng = np.random.default_rng(seed)
        x = x[rng.permutation(len(x))]
        n_val = max(len(x) // 10, 1)
        xv, xt = mx.array(x[:n_val]), x[n_val:]
        self._set_whitening(xt)
        init = self._init_params
        self.params = [dict(p) for p in init]           # the whitening alone
        base = float(-mx.mean(self.log_q(xv)).item())

        for attempt in range(retries + 1):
            self.params = [dict(p) for p in init]
            best, best_params, steps, diverged = self._train(
                xt, xv, base, rng, max_steps, batch, lr / 4 ** attempt, patience,
                check_every)
            if not diverged:
                break
        self.params = best_params
        mx.eval(self.params)
        return {"gain_nats": base - best, "val_nll": best, "val_nll_whitened": base,
                "steps": steps, "n_train": len(xt), "retries": attempt,
                "diverged": diverged, "seconds": time.perf_counter() - t0,
                "n_params": self.n_params}

    def _train(self, xt, xv, base, rng, max_steps, batch, lr, patience, check_every):
        opt = optim.Adam(learning_rate=optim.cosine_decay(lr, max_steps, lr * 0.02))

        def loss(params, u):
            return -mx.mean(self.log_q(u, params))
        vg = mx.value_and_grad(loss)
        best, best_params, bad, diverged = base, [dict(p) for p in self.params], 0, False
        i = 0
        for i in range(max_steps):
            idx = rng.integers(0, len(xt), size=min(batch, len(xt)))
            l, g = vg(self.params, mx.array(xt[idx]))
            # the optimizer wants a dict at the top of the tree
            self.params = opt.apply_gradients({"p": g}, {"p": self.params})["p"]
            mx.eval(l, self.params, opt.state)
            if not math.isfinite(float(l.item())):
                diverged = True
                break
            if (i + 1) % check_every == 0:
                v = float(-mx.mean(self.log_q(xv)).item())
                if not math.isfinite(v) or v > base + _DIVERGED:
                    diverged = True
                    break
                if v < best - 1e-4:
                    best, bad = v, 0
                    best_params = [dict(p) for p in self.params]
                else:
                    bad += 1
                    if bad >= patience:
                        break
        return best, best_params, i + 1, diverged

    # -- use --------------------------------------------------------------
    def wrap(self, target: LogDensity) -> "FlowTarget":
        """``target`` in this flow's coordinates."""
        return FlowTarget(target, self)

    def to_latent(self, u) -> mx.array:
        """Positions in u (``(n, dim)``) mapped to the flow's coordinates."""
        u = mx.array(np.asarray(u, dtype=np.float32))
        return self.inverse(u)[0]

    def start(self, n: int, seed: int = 0) -> mx.array:
        """``n`` starting positions in the flow's coordinates: draws picked
        from :attr:`pool` (the last training round's, in u) and mapped in,
        or, for a flow without a pool, standard normal draws."""
        rng = np.random.default_rng(seed)
        if self.pool is not None and len(self.pool):
            pick = rng.choice(len(self.pool), n, replace=n > len(self.pool))
            return self.to_latent(self.pool[pick])
        return mx.array(rng.standard_normal((n, self.dim)).astype(np.float32))

    @property
    def fingerprint(self) -> str:
        """Hash of the flow's weights, the same for the same map."""
        h = hashlib.sha256()
        for a in self._arrays().values():
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()[:16]

    def _arrays(self) -> dict[str, np.ndarray]:
        out = {"mu": np.array(self.mu), "L": np.array(self.L), "Linv": np.array(self.Linv),
               "logdet_L": np.array(self.logdet_L, dtype=np.float64)}
        for k, ((ia, ib, _), p) in enumerate(zip(self.splits, self.params)):
            out[f"l{k}__ia"], out[f"l{k}__ib"] = np.array(ia), np.array(ib)
            for name, v in p.items():
                out[f"l{k}__{name}"] = np.array(v)
        return out

    def save(self, path) -> None:
        """Write the frozen flow to ``path`` (``.npz``, no pickle). The
        report and pool are not saved."""
        out = self._arrays()
        out["config"] = np.array([self.dim, self.n_layers, self.hidden], dtype=np.int64)
        out["s_max"] = np.array(self.s_max)
        path = os.fspath(path)
        np.savez(path, **out)
        if not path.endswith(".npz") and os.path.exists(path + ".npz"):
            os.replace(path + ".npz", path)

    @classmethod
    def load(cls, path) -> "Flow":
        """Read a flow written by :meth:`save`."""
        with np.load(os.fspath(path), allow_pickle=False) as z:
            dim, n_layers, hidden = (int(v) for v in z["config"])
            f = cls(dim, n_layers, hidden, float(z["s_max"]))
            f.mu, f.L, f.Linv = mx.array(z["mu"]), mx.array(z["L"]), mx.array(z["Linv"])
            f.logdet_L = float(z["logdet_L"])
            f.splits = [cls._split(z[f"l{k}__ia"], z[f"l{k}__ib"]) for k in range(n_layers)]
            f.params = [{name: mx.array(z[f"l{k}__{name}"])
                         for name in ("W1", "b1", "W2", "b2", "W3", "b3")}
                        for k in range(n_layers)]
        return f


class FlowTarget(LogDensity):
    """``target`` in the coordinates of a frozen ``flow``:
    ``log p_z(z) = target.log_prob(T(z)) + log |det dT/dz|``. Exact for any
    fixed flow; a latent point whose image overflows counts as outside the
    support.

    :func:`anvil.run` records its draws through :meth:`to_base`, so
    ``Results.get_chain()`` and ``get_log_prob()`` are in the wrapped
    target's coordinates and values, as if it had been sampled directly.
    The chain state (``Results.final_state``) and a resume state stay in
    the flow's coordinates, and a resume checks that the flow is the same
    one (:attr:`resume_tag`)."""

    def __init__(self, target: LogDensity, flow: Flow):
        if int(target.dim) != flow.dim:
            raise ValueError(f"the flow has dim {flow.dim}, the target {target.dim}")
        self.target, self.flow = target, flow
        self.dim = flow.dim
        self.supports_grad = getattr(target, "supports_grad", True)

    @property
    def resume_tag(self) -> str:
        """Identifies the flow's current weights, so a run cannot resume
        in another flow's coordinates (computed on each call: a flow
        fitted again after wrapping gets a new tag)."""
        return f"flow:{self.flow.fingerprint}"

    def log_prob(self, z: mx.array) -> mx.array:
        u, ld = self.flow.forward(z)
        lp = self.target.log_prob(u) + ld
        return mx.where(mx.isnan(lp), -mx.inf, lp)

    def to_base(self, z: mx.array, log_prob: mx.array):
        """Positions and their recorded log-probabilities, mapped to the
        wrapped target's coordinates: ``(T(z), log_prob - log|det dT/dz|)``."""
        u, ld = self.flow.forward(z)
        return u, log_prob - ld


@dataclass
class FlowReport:
    """What :func:`train` or :func:`refine` found, and whether the flow is
    worth sampling with.

    ``gain_nats``: the hold-out log-likelihood of the flow minus that of
    the whitening alone. Near zero, the posterior is already close to
    Gaussian after a linear map and a plain sampler (dense ChEES, or the
    ensemble) will be faster than one in flow coordinates; ``worthwhile``
    is False below 1 nat. ``is_efficiency``: the effective sample size of
    importance weights target/flow, as a fraction of the draws -- the one
    diagnostic that tracked the flow's quality in every test. It cannot see
    posterior mass the training draws never reached. ``warnings``: each
    concern in words.

    Nothing here predicts the slow tail mixing seen on the hardest fits
    (a few walkers on long excursions, R-hat 1.01-1.03): the training
    draws look Gaussian in flow coordinates by construction (excess
    kurtosis 0.2-1.7 on the oblate fits). The run's R-hat is the check."""

    gain_nats: float
    is_efficiency: float
    worthwhile: bool
    n_train: int
    fit: dict
    smc: dict | None = None
    seconds: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"anvil flow: fit gain {self.gain_nats:.2f} nats over whitening, "
                 f"importance efficiency {self.is_efficiency:.3f}, {self.n_train} training draws"]
        if self.smc is not None:
            lines.append(f"  smc: ln Z {self.smc['log_z']:.3f}, "
                         f"{self.smc['stalled']} of {self.smc['n_stages']} stages stalled")
        if self.seconds:
            lines.append("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in self.seconds.items()))
        lines += [f"  WARNING: {w}" for w in self.warnings]
        if not self.warnings:
            lines.append("  no warnings")
        return "\n".join(lines)


def _as_draws(draws, dim: int) -> np.ndarray:
    if hasattr(draws, "get_chain"):
        draws = draws.get_chain()
    elif hasattr(draws, "u") and not isinstance(draws, np.ndarray):
        draws = draws.u                                  # an SMCResult
    a = np.asarray(draws, dtype=np.float32)
    if a.shape[-1] != dim:
        raise ValueError(f"draws have {a.shape[-1]} columns, the flow {dim}")
    return a.reshape(-1, dim)


def _make_report(target, flow, info, smc_facts, seconds, seed, compile_step):
    from .evidence import _flow_log_weights

    t0 = time.perf_counter()
    lw = _flow_log_weights(target, flow, 16384, seed + 7, 1 << 10, compile_step)
    eff = 0.0
    if np.isfinite(lw.max()):
        w = np.exp(lw - lw.max())
        eff = float(w.sum() ** 2 / (w * w).sum() / lw.size)
    seconds = {**seconds, "report": time.perf_counter() - t0}
    gain = float(info["gain_nats"])
    warn = []
    worthwhile = gain >= _GAIN_WORTHWHILE
    if info.get("diverged"):
        warn.append(f"training diverged at every learning rate tried; this flow is the "
                    f"best checkpoint, gain {gain:.2f} nats")
    if not worthwhile:
        warn.append(f"no flow needed: the fit gains only {gain:.2f} nats over a linear "
                    f"map, so the posterior is close to Gaussian already and a plain "
                    f"sampler (dense ChEES or the ensemble) will be faster")
    if eff < _EFF_POOR:
        warn.append(f"importance efficiency {eff:.4f}: the flow fits the posterior poorly "
                    f"somewhere; expect slow mixing there, and do not trust flow_evidence")
    if smc_facts is not None and smc_facts["stalled"]:
        warn.append(f"smc stalled in {smc_facts['stalled']} stages; its particles may "
                    f"under-cover the posterior, which the refinement rounds then have "
                    f"to correct")
    return FlowReport(gain_nats=gain, is_efficiency=eff,
                      worthwhile=worthwhile, n_train=int(info["n_train"]), fit=info,
                      smc=smc_facts, seconds=seconds, warnings=warn)


def train(target: LogDensity, *, draws=None, rounds: int = 4, n_chains: int = 512,
          round_warmup: int = 200, round_samples: int = 1000, n_layers: int = 8,
          hidden: int = 64, seed: int = 0, smc_kwargs: dict | None = None,
          fit_kwargs: dict | None = None, progress: bool = False,
          compile_step: bool = True) -> Flow:
    """Train a frozen flow for ``target``. Returns it with
    :attr:`Flow.report` and :attr:`Flow.pool` set.

    1. Training draws that start from the prior: :func:`anvil.smc` on
       ``target`` (a :class:`~anvil.TransformedLogDensity` with proper
       priors), with ``smc_kwargs``. Or ``draws`` you supply -- an array in
       u, a :class:`~anvil.Results` or an :class:`~anvil.SMCResult`. They
       must cover the whole posterior: a flow fitted to one basin speeds
       sampling up there and misses the rest without any diagnostic saying
       so.
    2. ``rounds`` refinements: fit a flow to the latest draws, run the
       ensemble (stretch and differential-evolution moves, ``n_chains``
       walkers) in its coordinates for ``round_warmup + round_samples``
       iterations, keep every fifth state, refit on the last two rounds.
    3. A final, longer fit on the last two rounds' draws, frozen.
    4. The :class:`FlowReport`: fit gain, importance efficiency, and
       warnings. Read it before sampling.

    Then sample with::

        ft = flow.wrap(target)
        res = anvil.run(anvil.EnsembleKernel(ft, moves=...), ft, flow.start(512))

    ``compile_step=False`` evaluates the target uncompiled throughout (SMC,
    the refinement runs, the report), for one that leaves MLX inside
    ``log_prob``.

    Costs, measured on a 10-parameter oblate transit fit (10,000 points,
    66 ms per batched gradient): SMC 2,080 s, the four rounds 210 s. SMC is
    most of it; with ``draws`` supplied it is skipped.
    """
    from .engine import run
    from .evidence import smc
    from .kernels.ensemble import DEMove, EnsembleKernel, StretchMove
    from .transforms import TransformedLogDensity

    d = int(target.dim)
    fit_kwargs = dict(fit_kwargs or {})
    seconds: dict[str, float] = {}
    smc_facts = None
    t0 = time.perf_counter()
    if draws is None:
        if not (isinstance(target, TransformedLogDensity) and target.is_proper):
            raise ValueError(
                "flow training starts from the prior through anvil.smc, which needs a "
                "TransformedLogDensity with proper priors; give it one, or pass draws= "
                "that start from the prior")
        r = smc(target, **{"seed": seed, "progress": progress, "compile_step": compile_step,
                           **(smc_kwargs or {})})
        smc_facts = {"log_z": float(r.log_z), "stalled": int(r.stalled),
                     "n_stages": int(r.n_stages)}
        su = np.asarray(r.u, dtype=np.float32)
        seconds["smc"] = time.perf_counter() - t0
    else:
        su = _as_draws(draws, d)
    su = su[np.all(np.isfinite(su), axis=1)]
    if n_chains % 2:
        raise ValueError("n_chains must be even (ensemble moves split the walkers in two)")

    rng = np.random.default_rng(seed)
    u_last = su[rng.choice(len(su), n_chains, replace=n_chains > len(su))]
    pool, rounds_drawn = [su], []
    t1 = time.perf_counter()
    for k in range(1, rounds + 1):
        flow = Flow(d, n_layers, hidden, seed=100 * seed + k)
        info = flow.fit(np.concatenate(pool[-2:]), seed=100 * seed + k,
                        **{"max_steps": 3000, **fit_kwargs})
        ft = flow.wrap(target)
        kern = EnsembleKernel(ft, moves=[(StretchMove(), 0.5), (DEMove(), 0.5)])
        res = run(kern, ft, flow.to_latent(u_last), n_warmup=round_warmup,
                  n_samples=round_samples // 5, thin=5, seed=1000 * seed + k,
                  compile_step=compile_step)
        chain = res.get_chain()                          # in u, via FlowTarget.to_base
        pool.append(chain.reshape(-1, d))
        rounds_drawn.append(chain)
        u_last = chain[-1]
        if progress:
            print(f"[flow] round {k}/{rounds}: fit gain {info['gain_nats']:.2f} nats, "
                  f"accept {float(np.mean(res.accept_fraction)):.2f}", flush=True)
    if rounds:
        seconds["rounds"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    train_draws = np.concatenate(pool[-2:])
    flow = Flow(d, n_layers, hidden, seed=100 * seed + 99)
    info = flow.fit(train_draws, seed=100 * seed + 99, **fit_kwargs)
    seconds["final fit"] = time.perf_counter() - t2
    flow.pool = pool[-1] if rounds else su
    flow.report = _make_report(target, flow, info, smc_facts, seconds, seed, compile_step)
    if progress:
        print(flow.report, flush=True)
    for w in flow.report.warnings:
        warnings.warn(f"anvil.flow.train: {w}", RuntimeWarning, stacklevel=2)
    return flow


def refine(target: LogDensity, draws, *, seed: int = 0, n_layers: int = 8,
           hidden: int = 64, fit_kwargs: dict | None = None,
           compile_step: bool = True) -> Flow:
    """One more flow, fitted to the draws of a production run (a
    :class:`~anvil.Results` from a run in flow coordinates, whose chain is
    in u). Returns the new frozen flow, with a fresh report.

    A run on the new flow is a new Markov chain: start it with warmup from
    ``new.start(n)``, not by resuming the old one (``run`` refuses a resume
    across flows). Measured on the oblate fit at 2,000 points: a refit on
    256,000 production draws took R-hat at 10,000 iterations from 1.024 to
    1.012, and 1.006 at 40,000; at 10,000 points it reached 1.013 and went
    no further."""
    return train(target, draws=draws, rounds=0, seed=seed, n_layers=n_layers,
                 hidden=hidden, fit_kwargs=fit_kwargs, compile_step=compile_step)

"""Vectorized gradient-free ensemble moves (red-black scheme).

Naively updating every walker against the current ensemble breaks detailed
balance when vectorized. The correct scheme (as in emcee >= 3) splits the
ensemble into two halves: all walkers of one half are updated
simultaneously — their proposals depend only on the *frozen* complementary
half, so they are conditionally independent — then the second half updates
against the already-moved first half. Each half-update is one batched
log_prob call over n/2 walkers: ideal for the GPU.

Moves supplied: the Goodman-Weare stretch move (affine-invariant,
self-tuning, requires the z^(d-1) Jacobian factor in the acceptance) and a
differential-evolution move (mode-hopping friendly). Mixing is emcee-style:
``moves=[(StretchMove(), 0.8), (DEMove(), 0.2)]`` picks one move per
iteration with those probabilities.
"""

from __future__ import annotations

import math
import random as _pyrandom
import warnings

import mlx.core as mx

from ..logdensity import LogDensity
from ..state import ChainState
from .base import Kernel, StepInfo


class EnsembleMove:
    """One proposal rule: map (key, S, C) -> (proposal, extra log factor).

    ``S`` (K, dim) is the half being moved; ``C`` (K, dim) is the frozen
    complementary half. The extra log factor enters the acceptance ratio
    (e.g. the stretch move's (d-1) log z); zero for symmetric proposals.
    """

    def propose(self, key: mx.array, S: mx.array, C: mx.array):
        raise NotImplementedError


class StretchMove(EnsembleMove):
    """Goodman & Weare (2010) stretch move, z ~ g(z) proportional to
    1/sqrt(z) on [1/a, a], proposal y = c_j + z (s - c_j)."""

    def __init__(self, a: float = 2.0):
        if a <= 1.0:
            raise ValueError("stretch parameter a must exceed 1")
        self.a = float(a)

    def propose(self, key, S, C):
        k_z, k_j = mx.random.split(key)
        K, d = S.shape
        u = mx.random.uniform(shape=(K,), key=k_z)
        z = ((self.a - 1.0) * u + 1.0) ** 2 / self.a  # inverse-CDF sample
        j = mx.random.randint(0, K, shape=(K,), key=k_j)
        partner = C[j]
        prop = partner + z[:, None] * (S - partner)
        return prop, (d - 1.0) * mx.log(z)


class DEMove(EnsembleMove):
    """Differential-evolution move: y = s + gamma (c_j1 - c_j2) + eps.

    gamma defaults to the ter Braak scaling 2.38/sqrt(2 d) and switches to
    gamma = 1 with probability ``big_jump`` (proposes jumps between modes
    separated by the ensemble's own displacement vectors). Symmetric, so
    no Jacobian factor.
    """

    def __init__(self, gamma: float | None = None, sigma: float = 1e-5,
                 big_jump: float = 0.1):
        self.gamma = gamma
        self.sigma = float(sigma)
        self.big_jump = float(big_jump)

    def propose(self, key, S, C):
        k_j1, k_j2, k_big, k_eps = mx.random.split(key, 4)
        K, d = S.shape
        gamma0 = self.gamma if self.gamma is not None else 2.38 / math.sqrt(2 * d)
        j1 = mx.random.randint(0, K, shape=(K,), key=k_j1)
        # distinct second partner: offset by 1..K-1 cyclically
        j2 = (j1 + 1 + mx.random.randint(0, K - 1, shape=(K,), key=k_j2)) % K
        big = mx.random.uniform(shape=(K, 1), key=k_big) < self.big_jump
        gamma = mx.where(big, 1.0, gamma0)
        eps = self.sigma * mx.random.normal(shape=S.shape, key=k_eps)
        prop = S + gamma * (C[j1] - C[j2]) + eps
        return prop, mx.zeros((K,))


class EnsembleKernel(Kernel):
    """Red-black ensemble sampler over a mixture of moves.

    Non-adaptive (the stretch move is affine-invariant and self-tuning).
    Handles its own compilation — one compiled step per move — because the
    per-iteration move choice is host-side control flow.
    """

    needs_grad = False
    self_compiled = True

    def __init__(
        self,
        target: LogDensity,
        moves: list[tuple[EnsembleMove, float]] | None = None,
        seed: int = 0,
    ):
        self.target = target
        self.moves = moves or [(StretchMove(), 1.0)]
        total = sum(w for _, w in self.moves)
        self._weights = [w / total for _, w in self.moves]
        self._chooser = _pyrandom.Random(seed ^ 0x5EED)
        self._compiled = [mx.compile(self._make_step(m)) for m, _ in self.moves]

    def init(self, key, u0: mx.array, target: LogDensity) -> ChainState:
        self.target = target
        n, d = u0.shape
        if n % 2:
            raise ValueError("ensemble moves require an even number of walkers")
        if n < 2 * d:
            raise ValueError(
                f"n_walkers={n} < 2*dim={2*d}: each half-ensemble must "
                "affinely span parameter space"
            )
        if n < 4 * d:
            warnings.warn(
                f"n_walkers={n} < 4*dim: ensemble moves mix poorly with so "
                "few walkers; on this hardware use hundreds or thousands",
                stacklevel=2,
            )
        return {"u": u0, "log_prob": target.log_prob(u0)}

    def _make_step(self, move: EnsembleMove):
        def step(key, state, params):
            u, logp = state["u"], state["log_prob"]
            n = u.shape[0]
            K = n // 2
            keys = mx.random.split(key, 4)
            halves = [u[:K], u[K:]]
            logps = [logp[:K], logp[K:]]
            accept_probs = []
            accepted_flags = []
            for h in (0, 1):
                k_prop, k_acc = keys[2 * h], keys[2 * h + 1]
                S, C = halves[h], halves[1 - h]
                prop, extra = move.propose(k_prop, S, C)
                lp = self.target.log_prob(prop)
                log_ratio = extra + lp - logps[h]
                accept_prob = mx.minimum(1.0, mx.exp(log_ratio))
                accept = (
                    mx.log(mx.random.uniform(shape=(K,), key=k_acc)) < log_ratio
                )
                halves[h] = mx.where(accept[:, None], prop, S)
                logps[h] = mx.where(accept, lp, logps[h])
                accept_probs.append(accept_prob)
                accepted_flags.append(accept)
            new_state = {
                "u": mx.concatenate(halves, axis=0),
                "log_prob": mx.concatenate(logps, axis=0),
            }
            info: StepInfo = {
                "accept_prob": mx.concatenate(accept_probs, axis=0),
                "accepted": mx.concatenate(accepted_flags, axis=0),
            }
            return new_state, info

        return step

    def step(self, key, state, params):
        i = self._chooser.choices(range(len(self.moves)), self._weights)[0]
        return self._compiled[i](key, state, params)

"""ChEES trajectory-length adaptation (Hoffman & Sountsov, AISTATS 2021).

The Change-in-the-Estimator-of-the-Expected-Square criterion,

    ChEES = 1/4 E[ (||x' - E x'||^2 - ||x - E x||^2)^2 ],

peaks when trajectories are long enough to decorrelate the squared
distance from the posterior mean — a proxy for the largest-scale mixing —
and its gradient with respect to the trajectory length is estimable from
quantities every chain already computed: the proposed end state and its
end-of-trajectory velocity. Stochastic ascent on log T with an
RMSProp-style preconditioner (constants from the reference
implementation), followed by iterate averaging, gives the frozen
post-warmup trajectory length.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class ChEESState:
    log_T: mx.array      # scalar, current (max) trajectory length
    log_T_bar: mx.array  # scalar, averaged iterate
    s2: mx.array         # scalar, RMSProp second-moment accumulator


def init_chees(T0: float = 1.0) -> ChEESState:
    import math

    return ChEESState(
        log_T=mx.array(math.log(T0)),
        log_T_bar=mx.array(math.log(T0)),
        s2=mx.array(0.0),
    )


def chees_gradient(
    q_prev: mx.array,   # (n, dim) pre-step positions
    q_prop: mx.array,   # (n, dim) proposed end-of-trajectory positions
    v_end: mx.array,    # (n, dim) end-of-trajectory velocities M^{-1} p'
    accept_prob: mx.array,  # (n,)
    jitter: float,      # this iteration's Halton value h (chain rule t = h*T)
) -> mx.array:
    """Per-iteration stochastic gradient of the ChEES criterion w.r.t. T."""
    mu_prev = mx.mean(q_prev, axis=0)
    a_sum = mx.sum(accept_prob) + 1e-20
    mu_prop = mx.sum(accept_prob[:, None] * q_prop, axis=0) / a_sum
    dc_prev = q_prev - mu_prev
    dc_prop = q_prop - mu_prop
    d2 = mx.sum(dc_prop * dc_prop, axis=-1) - mx.sum(dc_prev * dc_prev, axis=-1)
    gi = d2 * mx.sum(dc_prop * v_end, axis=-1)
    ok = (accept_prob > 1e-4) & mx.isfinite(gi)
    w = mx.where(ok, accept_prob, 0.0)
    g = mx.sum(w * mx.where(ok, gi, 0.0)) / (mx.sum(w) + 1e-20)
    return jitter * g


def update_chees(
    st: ChEESState,
    grad: mx.array,
    t: int,
    eps: mx.array,
    max_leapfrog: int = 1000,
    adaptation_rate: float = 0.025,
    sq_grad_rate: float = 0.05,
    max_step: float = 0.35,
) -> ChEESState:
    """RMSProp-in-log ascent + iterate averaging; ``t`` is 1-based."""
    s2 = (1.0 - sq_grad_rate) * st.s2 + sq_grad_rate * grad * grad
    s2_hat = s2 / (1.0 - (1.0 - sq_grad_rate) ** t)  # zero-debias
    step = adaptation_rate * grad / mx.sqrt(s2_hat + 1e-20)
    log_T = st.log_T + mx.clip(step, -max_step, max_step)
    # keep T within [eps, eps * max_leapfrog]
    log_T = mx.clip(log_T, mx.log(eps), mx.log(eps * max_leapfrog))
    w = t ** (-0.5)
    log_T_bar = w * log_T + (1.0 - w) * st.log_T_bar
    return ChEESState(log_T=log_T, log_T_bar=log_T_bar, s2=s2)

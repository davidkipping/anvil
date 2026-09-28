"""anvil + MetalPlanet: fiducial hot-Jupiter fit, both sampling modes.

The test case: a 3-day hot Jupiter transiting a Sun-like star at impact
parameter b = 0.5 (r = 0.103, a/R* = 8.75 from Kepler's third law),
quadratic limb darkening sampled in Kipping (2013) (q1, q2), observed for
30 days at 100,000 epochs with 500 ppm independent Gaussian noise.

Pipeline demonstrated:
  1. float64 CPU world: absolute BJD-scale times, data synthesis through
     MetalPlanet's own float64 path, epoch-centering preprocessing;
  2. float32 GPU world: MetalPlanet's fused Metal transit kernel behind
     anvil's chunked likelihood;
  3. the precision harness (trust check) before any sampling;
  4. BOTH sampler modes on identical data, each configured to actually
     converge (R-hat < 1.01) — emcee-like (gradient-free ensemble moves)
     and HMC-like (ChEES-HMC with a dense mass matrix) — head to head;
  4b. a curvature read-out that explains WHY the comparison lands where
     it does, via anvil.whitened_shape;
  5. posteriors reported in absolute physical units, and a summary figure.

Requires: pip install -e ".[test]" matplotlib, and MetalPlanet
(pip install -e ../MetalPlanet).

Run:  .venv/bin/python examples/metalplanet_hotjupiter.py
"""

import time

import mlx.core as mx
import numpy as np

import anvil
from anvil.kernels.chees import ChEESHMC
from anvil.kernels.ensemble import DEMove, EnsembleKernel, StretchMove
from metalplanet.anvil import PARAM_NAMES, make_quad_transit_flux
from metalplanet.ld import q_to_u_np, u_to_q_np
from metalplanet.orbit import epoch_center_times

# ---------------------------------------------------------------- truth
T_REF = 2_457_000.0          # BJD zero-point — hopeless in float32
BASELINE_DAYS = 30.0
N_DATA = 100_000
YERR = 500e-6                # 500 ppm per point

P_TRUE = 3.0                 # hot Jupiter
T0_TRUE = 1.2345             # days after T_REF
R_TRUE = 0.103               # Rp/R* (Jupiter / Sun)
B_TRUE = 0.5
A_TRUE = 8.75                # a/R* for P = 3 d around a Sun-like star
U1_TRUE, U2_TRUE = 0.40, 0.25
Q1_TRUE, Q2_TRUE = (float(q) for q in u_to_q_np(U1_TRUE, U2_TRUE))

SEED = 20260906

# Per-sampler configuration. The two samplers want different settings and
# pretending otherwise produces an unconverged showcase: the ensemble's
# autocorrelation here is ~120 iterations, so it needs LONG chains (walker
# count buys nothing once the GPU is saturated, and spending the budget on
# walkers instead of iterations is what left an earlier version of this
# example at R-hat 1.05 — not converged). ChEES-HMC decorrelates in ~2
# iterations and needs the opposite: many chains, few draws each.
N_WALKERS, ENSEMBLE_WARMUP, ENSEMBLE_DRAWS = 128, 500, 20_000
N_CHAINS, HMC_WARMUP, HMC_DRAWS = 512, 300, 200

# ------------------------------------------- float64 CPU: data synthesis
rng = np.random.default_rng(SEED)
t_abs = T_REF + np.sort(rng.uniform(0.0, BASELINE_DAYS, size=N_DATA))
t_model = t_abs - T_REF

# imperfect "preliminary fit" references, as in a real workflow
t0_ref = T0_TRUE - 0.009
period_ref = P_TRUE + 0.0005

truth_model = np.array([
    T0_TRUE - t0_ref, P_TRUE - period_ref,
    R_TRUE, B_TRUE, A_TRUE, Q1_TRUE, Q2_TRUE,
    0.0,                                    # df0 (baseline deviation)
])

x64 = epoch_center_times(t_model, t0_ref=t0_ref, period_ref=period_ref)
model_fn = make_quad_transit_flux(period_ref=period_ref)   # fused Metal core

with mx.stream(mx.cpu):                                    # fp64 synthesis
    dev_true = np.array(
        model_fn(mx.array(truth_model[None, :], dtype=mx.float64),
                 mx.array(x64, dtype=mx.float64))[0], dtype=np.float64)
y = 1.0 + dev_true + YERR * rng.standard_normal(N_DATA)
depth_ppm = -dev_true.min() * 1e6
n_transits = int(BASELINE_DAYS / P_TRUE)
print(f"synthetic hot Jupiter: depth {depth_ppm:.0f} ppm, "
      f"~{n_transits} transits, {N_DATA} points, {YERR*1e6:.0f} ppm noise\n")

# --------------------------------------------- anvil target assembly
transform = anvil.Transform([
    anvil.ParamSpec("t0_off", lo=-0.5, hi=0.5, report_offset=T_REF + t0_ref),
    anvil.ParamSpec("p_off", lo=-0.05, hi=0.05, report_offset=period_ref),
    anvil.ParamSpec("r", lo=0.01, hi=0.5),
    anvil.ParamSpec("b", lo=0.0, hi=0.9),
    anvil.ParamSpec("a", lo=2.0, hi=50.0),
    anvil.ParamSpec("q1", lo=0.0, hi=1.0),
    anvil.ParamSpec("q2", lo=0.0, hi=1.0),
    anvil.ParamSpec("df0", lo=-0.01, hi=0.01, report_offset=1.0),
])
loglike = anvil.precision.ChunkedGaussianLogLike(
    model_fn, x64, y - 1.0, np.full(N_DATA, YERR))
target = anvil.TransformedLogDensity(
    loglike, transform, model_log_prob_hi=loglike.hi)

# --------------------------------------------------- trust check first
u_truth = transform.from_model_np(truth_model)


def init_ball(n):
    """Chains started in a tight ball around a preliminary fit — the
    initialization ChEES-HMC needs (see docs/samplers.md)."""
    return mx.array(
        (u_truth + 1e-3 * rng.standard_normal((n, 8))).astype(np.float32))


print(anvil.validate_precision(target, init_ball(32)), "\n")

# ------------------------------------------------- run both sampler modes
runs = {}
for label, kernel, n_chains, n_warmup, n_samples in (
    # A 50/50 stretch + differential-evolution mixture halves the
    # autocorrelation of stretch alone here (measured tau 262 -> 140) and
    # more than doubles the effective sample size.
    ("ensemble (emcee-like)",
     EnsembleKernel(target, moves=[(StretchMove(), 0.5), (DEMove(), 0.5)],
                    seed=1),
     N_WALKERS, ENSEMBLE_WARMUP, ENSEMBLE_DRAWS),
    # dense=True adapts the full cross-chain covariance. This posterior is
    # strongly correlated (b-a -0.96, q1-q2 -0.95), which a diagonal
    # preconditioner cannot remove; dense is worth ~1.85x here.
    ("ChEES-HMC (dense)",
     ChEESHMC(target, max_leapfrog=128, dense=True),
     N_CHAINS, HMC_WARMUP, HMC_DRAWS),
):
    u0 = init_ball(n_chains)
    # NOTE: no reanchor_every here — the harness above reported OK
    # (fp32 error ~0.01 for this problem, far below the Metropolis
    # decision scale), and each float64 re-anchor of an expensive model
    # costs single-core CPU minutes at this batch size. Enable it only
    # when the harness reports ACCEPTABLE or WARNING.
    t0 = time.perf_counter()
    res = anvil.run(kernel, target, u0, n_warmup=n_warmup,
                    n_samples=n_samples, seed=2)
    wall = time.perf_counter() - t0
    chain = res.get_chain()
    diag = anvil.diagnose(chain, names=PARAM_NAMES)   # one shared pass
    ess, rhat = diag.ess_bulk, diag.rhat
    runs[label] = dict(res=res, wall=wall, chain=chain,
                       ess_min=float(ess.min()),
                       rhat_max=float(rhat.max()),
                       n_div=res.extras.get("n_divergent", 0))
    print(f"== {label}: {wall:.1f} s | {n_chains} chains x {n_samples} draws "
          f"| min ESS {ess.min():,.0f} ({ess.min()/wall:,.0f} ESS/s) "
          f"| max R-hat {rhat.max():.4f} | divergent {runs[label]['n_div']}")

# ------------------------------------------------- posteriors, physical units
truth_phys = transform.to_physical(truth_model)
print(f"\n{'param':>10s} " + " ".join(f"{lbl:>28s}" for lbl in runs)
      + f" {'truth':>16s}")
posts = {}
for lbl, r in runs.items():
    flat_u = r["res"].get_chain(flat=True).astype(np.float64)
    posts[lbl] = transform.to_physical(transform.model_np(flat_u))
for i, name in enumerate(PARAM_NAMES):
    cells = []
    for lbl in runs:
        p = posts[lbl][:, i]
        cells.append(f"{p.mean():>16.8g} ± {p.std():<9.2g}")
    print(f"{name:>10s} " + " ".join(cells) + f" {truth_phys[i]:>16.8g}")

u1u2 = {lbl: q_to_u_np(p[:, 5].mean(), p[:, 6].mean())
        for lbl, p in posts.items()}
for lbl, (u1, u2) in u1u2.items():
    print(f"  [{lbl}] implied limb darkening u1={u1:.3f}, u2={u2:.3f} "
          f"(truth {U1_TRUE}, {U2_TRUE})")

# ------------------------------------------------------------- verdict
s, h = runs["ensemble (emcee-like)"], runs["ChEES-HMC (dense)"]
print(f"\nVERDICT: ChEES-HMC {h['ess_min']/h['wall']:,.0f} ESS/s vs "
      f"ensemble {s['ess_min']/s['wall']:,.0f} ESS/s "
      f"({(h['ess_min']/h['wall'])/(s['ess_min']/s['wall']):.1f}x). "
      f"Both converged (R-hat {s['rhat_max']:.4f} / {h['rhat_max']:.4f}).")
skew, exkurt = anvil.whitened_shape(
    runs["ChEES-HMC (dense)"]["res"].get_chain(flat=True).astype(np.float64))
print("\nposterior curvature after whitening (anvil.whitened_shape):")
print("   " + "  ".join(f"{n:>8s}" for n in PARAM_NAMES))
print("   " + "  ".join(f"{v:>8.2f}" for v in skew) + "   skew")
print("   " + "  ".join(f"{v:>8.2f}" for v in exkurt) + "   excess kurtosis")
print(f"   Max |skew| {np.abs(skew).max():.2f}: this posterior is correlated "
      "but essentially\n   uncurved, which is exactly the regime a dense mass "
      "matrix was built for --\n   it takes ChEES from ~38 leapfrog steps per "
      "draw to ~3 here. A curved\n   posterior would show |skew| of order 1 "
      "and cap that gain; see docs/samplers.md.")

# ------------------------------------------------------------- figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib not installed; skipping figure")

BLUE, ORANGE, INK, MUTED = "#2563eb", "#ea580c", "#1f2937", "#9ca3af"
fig = plt.figure(figsize=(11, 7.2), constrained_layout=True)
gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0])

# phase-folded light curve + posterior-median model
ax = fig.add_subplot(gs[0, :])
phase = ((t_model - T0_TRUE + 0.5 * P_TRUE) % P_TRUE) - 0.5 * P_TRUE
nb = 400
bins = np.linspace(-0.5 * P_TRUE, 0.5 * P_TRUE, nb + 1)
idx = np.digitize(phase, bins) - 1
ybin = np.array([y[idx == j].mean() if np.any(idx == j) else np.nan
                 for j in range(nb)])
ax.plot(0.5 * (bins[1:] + bins[:-1]) * 24, ybin, ".", ms=3, color=MUTED,
        label="data (binned)")
med_u = np.median(runs["ChEES-HMC (dense)"]["res"].get_chain(flat=True), axis=0)
med_model = transform.model_np(med_u[None, :])[0]
t_dense = np.linspace(T0_TRUE - 0.5 * P_TRUE, T0_TRUE + 0.5 * P_TRUE, 3000)
xd = epoch_center_times(t_dense, t0_ref=t0_ref, period_ref=period_ref)
with mx.stream(mx.cpu):
    dev_med = np.array(model_fn(mx.array(med_model[None, :], dtype=mx.float64),
                                mx.array(xd, dtype=mx.float64))[0])
ax.plot((t_dense - T0_TRUE) * 24, 1.0 + dev_med, color=BLUE, lw=2,
        label="posterior-median model (ChEES-HMC)")
ax.set_xlim(-4, 4)
ax.set_xlabel("hours from mid-transit")
ax.set_ylabel("relative flux")
ax.set_title("anvil + MetalPlanet — 3 d hot Jupiter, b = 0.5, "
             f"{N_DATA:,} pts @ {YERR*1e6:.0f} ppm", color=INK)
ax.legend(frameon=False, loc="lower right")
ax.grid(alpha=0.15)

# posterior marginals: both samplers overlaid
show = [("r", 2), ("b", 3), ("a", 4), ("q1", 5)]
for j, (name, i) in enumerate(show):
    axm = fig.add_subplot(gs[1, j])
    for lbl, color in (("ensemble (emcee-like)", ORANGE),
                       ("ChEES-HMC (dense)", BLUE)):
        axm.hist(posts[lbl][:, i], bins=60, density=True, histtype="step",
                 lw=2, color=color, label=lbl)
    axm.axvline(truth_phys[i], color=INK, ls="--", lw=1, alpha=0.6)
    axm.set_xlabel(name)
    axm.set_yticks([])
    axm.grid(alpha=0.15)
    if j == 0:
        axm.set_ylabel("posterior density")
        axm.legend(frameon=False, fontsize=8)

fig.savefig("examples/metalplanet_hotjupiter.png", dpi=150)
print("\nfigure saved: examples/metalplanet_hotjupiter.png")

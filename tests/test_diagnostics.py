import numpy as np

from anvil.diagnostics import ess_bulk, norm_ppf, split_rhat, summary


def test_norm_ppf_matches_known_quantiles():
    # Known standard-normal quantiles
    assert abs(norm_ppf(np.array([0.5]))[0]) < 1e-9
    assert abs(norm_ppf(np.array([0.975]))[0] - 1.959964) < 1e-5
    assert abs(norm_ppf(np.array([0.0013498980316301]))[0] + 3.0) < 1e-6


def test_rhat_near_one_for_iid_chains():
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(500, 32, 3))
    rhat = split_rhat(chain)
    assert rhat.shape == (3,)
    assert np.all(rhat < 1.01)


def test_rhat_flags_stuck_chain():
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(500, 32))
    chain[:, 0] = 5.0  # one chain frozen far away
    assert split_rhat(chain)[0] > 1.05


def test_rhat_flags_trending_chains():
    # all chains still trending -> within-half disagreement -> split-Rhat > 1
    rng = np.random.default_rng(1)
    n, m = 400, 16
    chain = np.cumsum(rng.normal(size=(n, m)), axis=0)  # random walks
    assert split_rhat(chain)[0] > 1.1


def test_ess_iid_close_to_total():
    rng = np.random.default_rng(2)
    n, m = 500, 32
    chain = rng.normal(size=(n, m))
    ess = ess_bulk(chain)[0]
    assert 0.5 * n * m < ess < 1.5 * n * m


def test_ess_correlated_much_smaller():
    rng = np.random.default_rng(3)
    n, m, rho = 2000, 8, 0.95
    x = np.zeros((n, m))
    eps = rng.normal(size=(n, m)) * np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    ess = ess_bulk(x)[0]
    # AR(1): tau = (1+rho)/(1-rho) = 39 -> ESS ~ n*m/39
    assert ess < 0.1 * n * m
    assert ess > 0.005 * n * m


def test_summary_runs():
    rng = np.random.default_rng(4)
    text = summary(rng.normal(size=(200, 8, 2)), names=["a", "b"])
    assert "a" in text and "rhat" in text

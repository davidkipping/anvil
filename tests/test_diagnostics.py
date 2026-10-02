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


def test_diagnose_matches_separate_calls_and_shares_work():
    from anvil.diagnostics import diagnose
    rng = np.random.default_rng(11)
    chain = rng.normal(size=(120, 256, 4))
    chain[:, :5, 0] += 0.5                      # a genuinely non-trivial case
    d = diagnose(chain, names=list("abcd"))
    np.testing.assert_allclose(d.rhat, split_rhat(chain), rtol=1e-12)
    np.testing.assert_allclose(d.ess_bulk, ess_bulk(chain), rtol=1e-12)
    assert d.names == list("abcd") and "rhat" in str(d)


def test_batched_and_gpu_paths_match_the_per_parameter_reference():
    """The shared/batched implementation must reproduce the original
    per-parameter numpy computation, including across the size threshold
    where the autocovariance switches to the GPU."""
    import anvil.diagnostics as D
    rng = np.random.default_rng(12)
    # big enough to cross the x.size >= 2**18 GPU threshold
    chain = rng.normal(size=(160, 512, 6))
    chain[:, :7, 1] += 0.3
    ref_rhat = np.array([
        D._rhat_single(D._split_chains(D._rank_normalize(chain[..., d])))
        for d in range(6)])
    ref_ess = np.array([
        D._ess_single(D._split_chains(D._rank_normalize(chain[..., d])))
        for d in range(6)])
    d = D.diagnose(chain)
    assert np.abs(d.rhat - ref_rhat).max() < 1e-6
    assert np.abs(d.ess_bulk - ref_ess).max() / ref_ess.max() < 1e-5


def test_autocov_batches_over_parameters():
    from anvil.diagnostics import _autocov
    rng = np.random.default_rng(13)
    x = rng.normal(size=(64, 32, 3))
    batched = _autocov(x)
    for d in range(3):
        np.testing.assert_allclose(batched[..., d], _autocov(x[..., d]),
                                   rtol=1e-10, atol=1e-12)


def test_ranks_are_exact_across_the_mlx_sort_boundary():
    """MLX's argsort is wrong past a size, and the size depends on shape:
    a contiguous 1-D sort is exact to at least 2**27 rows, but the strided
    multi-column sort breaks above 1023*2048 = 2,095,104 rows, whatever the
    dim. The previous guard here keyed off 2**21 rows, which let the corrupt
    path run for 512 chains x 4096 draws. Whichever path diagnose picks, the
    scores must match the host float64 reference.
    """
    import anvil.diagnostics as D
    rng = np.random.default_rng(5)
    lim = D._MX_SORT_COL_LIMIT
    for rows, dim in ((lim, 2), (lim + 1, 2), (2 ** 21, 2), (2 ** 21, 8),
                      (2 ** 21 + 1, 1)):
        chain = rng.standard_normal((rows, 1, dim)).astype(np.float32)
        got = D._rank_normalize_all(chain)
        ref = np.stack([D._rank_normalize(chain[..., d].astype(np.float64))
                        for d in range(dim)], axis=-1)
        # a corrupted permutation moves scores by O(1); float32's quantile
        # evaluation moves them by ~3e-4 in the far tail
        assert np.abs(got - ref).max() < 1e-3, (rows, dim,
                                                np.abs(got - ref).max())


def test_memory_budget_does_not_change_the_answer():
    """The budget trades dispatch size for peak memory, nothing else."""
    import anvil.diagnostics as D
    rng = np.random.default_rng(6)
    chain = rng.standard_normal((400, 128, 5)).astype(np.float32)
    big = D.diagnose(chain, memory_budget=1 << 29)
    small = D.diagnose(chain, memory_budget=1 << 14)      # one param, tiny chunks
    np.testing.assert_array_equal(big.rhat, small.rhat)
    # ESS sums the autocovariance over chain chunks, so a different chunking
    # reassociates a float64 sum: equal to round-off, not bit-identical
    np.testing.assert_allclose(big.ess_bulk, small.ess_bulk, rtol=1e-12)
    np.testing.assert_array_equal(
        D.split_rhat(chain, memory_budget=1 << 14), big.rhat)
    np.testing.assert_allclose(
        D.ess_bulk(chain, memory_budget=1 << 14), big.ess_bulk, rtol=1e-12)


def test_memory_budget_is_detectable_and_bounds_the_peak():
    """Downstream feature-detects by signature (turin's _has_param), and the
    peak must stop scaling with dim -- that was the 21 GB."""
    import inspect

    import mlx.core as mx

    import anvil.diagnostics as D
    for fn in (D.diagnose, D.split_rhat, D.ess_bulk):
        assert "memory_budget" in inspect.signature(fn).parameters

    rng = np.random.default_rng(7)
    budget = 1 << 24                                   # 16 MiB
    peaks = {}
    for dim in (2, 16):
        chain = rng.standard_normal((2000, 256, dim)).astype(np.float32)
        mx.clear_cache()
        mx.reset_peak_memory()
        D.diagnose(chain, memory_budget=budget)
        peaks[dim] = mx.get_peak_memory()
    # 8x the parameters must not mean 8x the peak
    assert peaks[16] < 2 * peaks[2], peaks
    # and the peak stays within a small multiple of what was asked for
    assert peaks[16] < 12 * budget, (peaks, budget)



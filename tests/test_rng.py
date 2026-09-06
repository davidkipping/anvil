import mlx.core as mx

from anvil.halton import halton_jitter, van_der_corput
from anvil.rng import KeyStream


def test_van_der_corput_values():
    assert [van_der_corput(n) for n in (1, 2, 3, 4, 5)] == [
        0.5, 0.25, 0.75, 0.125, 0.625,
    ]


def test_halton_jitter_zero_based():
    assert halton_jitter(0) == 0.5
    assert halton_jitter(3) == 0.125


def test_halton_fills_evenly():
    vals = sorted(halton_jitter(t) for t in range(64))
    gaps = [b - a for a, b in zip(vals, vals[1:])]
    assert max(gaps) <= 2.0 / 64  # low discrepancy: no big holes


def test_keys_deterministic():
    a = KeyStream(seed=7)
    b = KeyStream(seed=7)
    xa = mx.random.normal((4,), key=a.key(3, KeyStream.PROPOSAL))
    xb = mx.random.normal((4,), key=b.key(3, KeyStream.PROPOSAL))
    assert mx.allclose(xa, xb).item()


def test_keys_differ_across_iteration_role_seed():
    ks = KeyStream(seed=7)
    base = mx.random.normal((4,), key=ks.key(3, 0))
    for other in (
        ks.key(4, 0),                    # different iteration
        ks.key(3, 1),                    # different role
        KeyStream(seed=8).key(3, 0),     # different seed
    ):
        assert not mx.allclose(base, mx.random.normal((4,), key=other)).item()

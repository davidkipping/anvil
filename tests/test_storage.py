import numpy as np

from anvil.storage import MemoryBackend, NpyStreamBackend


def _fill(backend, n_kept=10, n_chains=8, dim=3, seed=0):
    rng = np.random.default_rng(seed)
    backend.reserve(n_kept, n_chains, dim)
    frames = []
    for _ in range(n_kept):
        u = rng.standard_normal((n_chains, dim)).astype(np.float32)
        lp = rng.standard_normal(n_chains).astype(np.float32)
        backend.append(u, lp)
        frames.append((u, lp))
    return frames


def test_memory_backend_roundtrip():
    b = MemoryBackend()
    frames = _fill(b)
    chain = b.get_chain()
    assert chain.shape == (10, 8, 3)
    np.testing.assert_array_equal(chain[4], frames[4][0])
    np.testing.assert_array_equal(b.get_log_prob()[7], frames[7][1])
    assert b.get_chain(discard=6).shape == (4, 8, 3)
    assert b.get_chain(thin=2).shape == (5, 8, 3)
    assert b.get_chain(flat=True).shape == (80, 3)


def test_npy_stream_backend_matches_memory(tmp_path):
    mem, disk = MemoryBackend(), NpyStreamBackend(str(tmp_path / "run"))
    frames_a = _fill(mem, seed=1)
    frames_b = _fill(disk, seed=1)
    np.testing.assert_array_equal(mem.get_chain(), disk.get_chain())
    np.testing.assert_array_equal(mem.get_log_prob(), disk.get_log_prob())
    disk.flush()
    # files readable as plain .npy
    loaded = np.load(str(tmp_path / "run_chain.npy"), mmap_mode="r")
    np.testing.assert_array_equal(np.asarray(loaded), mem.get_chain())

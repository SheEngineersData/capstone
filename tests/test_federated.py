import numpy as np

from potatofl import federated as F


def labels(n=1200, c=8, seed=0):
    return np.random.default_rng(seed).integers(0, c, n)


def test_partition_is_a_disjoint_cover():
    y = labels()
    idx = np.arange(len(y))
    parts = F.dirichlet_partition(y, idx, 10, 0.5, np.random.default_rng(1))
    allp = np.concatenate(parts)
    assert len(allp) == len(idx) and len(set(allp.tolist())) == len(idx)
    assert min(map(len, parts)) >= 10


def test_smaller_alpha_gives_more_label_skew():
    y = labels()
    idx = np.arange(len(y))
    skew = lambda a: F.label_skew(F.dirichlet_partition(y, idx, 10, a, np.random.default_rng(2)), y, 8)
    assert skew(0.1) > skew(1.0) > skew(100.0)


def test_partition_is_reproducible():
    y = labels()
    idx = np.arange(len(y))
    a = F.dirichlet_partition(y, idx, 5, 0.5, np.random.default_rng(3))
    b = F.dirichlet_partition(y, idx, 5, 0.5, np.random.default_rng(3))
    assert all((p == q).all() for p, q in zip(a, b))


def test_int8_compression_error_and_bytes():
    d = np.random.default_rng(0).normal(size=10_000).astype(np.float32)
    out, nbytes = F.compress(d, "int8")
    assert nbytes == d.size + 4
    assert np.abs(out - d).max() <= np.abs(d).max() / 127 / 2 + 1e-6


def test_topk_keeps_largest_entries():
    d = np.arange(100, dtype=np.float32)
    out, nbytes = F.compress(d, "topk", 0.1)
    assert np.count_nonzero(out) == 10 and out[-1] == 99 and nbytes == 80


def test_privatize_clips_norm_before_noise():
    d = np.full(1000, 1.0, np.float32)
    out = F.privatize(d, clip=1.0, noise_mult=0.0, rng=np.random.default_rng(0))
    assert abs(np.linalg.norm(out) - 1.0) < 1e-4
    noisy = F.privatize(d, clip=1.0, noise_mult=0.5, rng=np.random.default_rng(0))
    assert abs((noisy - out).std() - 0.5) < 0.05

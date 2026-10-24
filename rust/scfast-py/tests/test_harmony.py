"""Tests of scfast.harmony (the Python wrapper and the native harmony_dense).

Run with the built extension on the path, e.g.
    python -m pytest rust/scfast-py/tests/test_harmony.py
"""

import numpy as np
import pytest

import scfast


def blobs(n_per=150, types=3, batches=3, d=10, seed=0):
    """Gaussian cell types with a per-batch shift; returns (emb, batch labels, types)."""
    rng = np.random.default_rng(seed)
    centres = 10 * rng.normal(size=(types, d))
    shifts = 4 * rng.normal(size=(batches, d))
    ty = np.tile(np.arange(types), n_per * batches)
    b = np.repeat(np.arange(batches), n_per * types)
    rng.shuffle(b)
    emb = centres[ty] + shifts[b] + rng.normal(size=(ty.size, d))
    return emb, np.array([f"batch{i}" for i in b]), ty


def batch_spread(emb, b, ty):
    """Mean distance between the batch centroids of each type."""
    out = []
    for t in np.unique(ty):
        cs = [emb[(ty == t) & (b == x)].mean(0) for x in np.unique(b)]
        out += [np.linalg.norm(cs[i] - cs[j]) for i in range(len(cs)) for j in range(i + 1, len(cs))]
    return float(np.mean(out))


def test_removes_batch_shift_and_keeps_dtype():
    emb, b, ty = blobs()
    out64 = scfast.harmony(emb, b)
    assert out64.dtype == np.float64 and out64.shape == emb.shape
    assert batch_spread(out64, b, ty) < 0.2 * batch_spread(emb, b, ty)
    type_centres = np.array([out64[ty == t].mean(0) for t in range(3)])
    gaps = [np.linalg.norm(type_centres[i] - type_centres[j]) for i in range(3) for j in range(i + 1, 3)]
    assert min(gaps) > 5 * batch_spread(out64, b, ty)

    out32, info = scfast.harmony(emb.astype(np.float32), b, return_info=True)
    assert out32.dtype == np.float32
    assert np.allclose(out32, out64, atol=1e-3)
    assert info["iters"] == len(info["objective"]) - 1
    assert info["nclust"] == 45  # round(1350 / 30)
    assert info["device"] == "cpu"
    assert isinstance(info["converged"], bool)


def test_reproducible_and_options():
    emb, b, _ = blobs(n_per=60, seed=1)
    a = scfast.harmony(emb, b, seed=3)
    assert np.array_equal(a, scfast.harmony(emb, b, seed=3))
    assert not np.array_equal(a, scfast.harmony(emb, b, seed=4))
    # integer labels and string labels of the same partition give the same result
    codes = np.unique(b, return_inverse=True)[1]
    assert np.array_equal(a, scfast.harmony(emb, codes, seed=3))
    # Fortran-ordered input is accepted
    assert np.array_equal(a, scfast.harmony(np.asfortranarray(emb), b, seed=3))
    _, info = scfast.harmony(emb, b, lambda_=1.0, nclust=7, max_iter=3, early_stop=False, return_info=True)
    assert info["iters"] == 3 and info["nclust"] == 7
    init = emb[:5].astype(np.float32)
    _, info = scfast.harmony(emb, b, init_centroids=init, return_info=True)
    assert info["nclust"] == 5


def test_argument_errors():
    emb, b, _ = blobs(n_per=20)
    with pytest.raises(ValueError, match="one label per cell"):
        scfast.harmony(emb, b[1:])
    with pytest.raises(ValueError, match="2-D"):
        scfast.harmony(emb[:, 0], b)
    with pytest.raises(ValueError, match="theta"):
        scfast.harmony(emb, b, theta=-1)
    with pytest.raises(ValueError, match="sigma"):
        scfast.harmony(emb, b, sigma=0)
    with pytest.raises(ValueError, match="nclust"):
        scfast.harmony(emb, b, nclust=0)
    with pytest.raises(ValueError, match="init_centroids"):
        scfast.harmony(emb, b, init_centroids=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="non-finite"):
        bad = emb.copy()
        bad[3, 2] = np.nan
        scfast.harmony(bad, b)
    with pytest.raises(ValueError, match="at least 6 cells"):
        scfast.harmony(emb[:5], b[:5])
    with pytest.raises(TypeError):
        scfast.harmony(emb.astype(str), b)
    with pytest.raises(RuntimeError, match="GPU"):
        scfast.harmony(emb, b, device="cuda")  # no CUDA Harmony (or no CUDA device) here

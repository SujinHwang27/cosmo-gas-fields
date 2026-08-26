"""Hann-windowed slab r(k): identity gate, null gate, sensitivity to noise."""

import numpy as np

from cosmo_gas_fields.analysis import gaussian_smooth_periodic, slab_identity_gate, slab_null_gate, slab_r_k
from cosmo_gas_fields.analysis.field_stats import rk_first_crossing

BOX = 60.0
N = 32


def _cube(seed):
    rng = np.random.default_rng(seed)
    return gaussian_smooth_periodic(rng.standard_normal((N, N, N)), BOX, 2.0)


def test_identity_gate_passes():
    slab = _cube(0)[24:32]
    assert slab_identity_gate(slab, BOX, N) < 1e-9


def test_null_gate_passes():
    slab = _cube(1)[20:32]
    kf = 2 * np.pi / BOX
    assert slab_null_gate(slab, BOX, N, k_min=4 * kf, threshold=0.3, seed=0) < 0.3


def test_noisy_copy_loses_coherence_at_high_k():
    cube = _cube(2)
    rng = np.random.default_rng(3)
    noisy = cube + 0.7 * rng.standard_normal(cube.shape)
    kc, rk, cnt = slab_r_k(cube[16:32], noisy[16:32], BOX, N)
    assert kc.shape == rk.shape == cnt.shape
    ok = np.isfinite(rk)
    assert rk[ok][0] > rk[ok][-1]
    assert np.isfinite(rk_first_crossing(kc, rk))

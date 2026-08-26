"""Field statistics: identity, null floor, r(k) crossing, controls, bootstrap."""

import numpy as np
import pytest

from cosmo_gas_fields.analysis import (
    block_bootstrap_delta_rs,
    gaussian_smooth_periodic,
    lowpass_sharp,
    nccf,
    paired_fisher_t,
    pearson,
    per_octant_pearson,
    phase_randomized,
    phase_randomized_null_band,
    rk_coherence,
    rk_first_crossing,
    smoothed_pearson_ladder,
    spearman,
)

BOX = 60.0
N = 32


def _field(seed, sigma=3.0):
    rng = np.random.default_rng(seed)
    return gaussian_smooth_periodic(rng.standard_normal((N, N, N)), BOX, sigma)


def test_pearson_identity_and_degenerate():
    x = _field(0)
    assert pearson(x, x) == pytest.approx(1.0, abs=1e-12)
    assert np.isnan(pearson(x, np.zeros_like(x)))
    assert spearman(x, x) == pytest.approx(1.0, abs=1e-12)


def test_smoothed_pearson_ladder_identity():
    x = _field(1)
    ladder = smoothed_pearson_ladder(x, x, BOX, sigmas=(1.0, 2.0, 4.0))
    assert set(ladder) == {1.0, 2.0, 4.0}
    assert all(v == pytest.approx(1.0, abs=1e-10) for v in ladder.values())


def test_phase_randomized_keeps_power_and_kills_correlation():
    x = _field(2)
    null = phase_randomized(x, seed=7)
    amp_x = np.abs(np.fft.rfftn(x - x.mean()))
    amp_n = np.abs(np.fft.rfftn(null - null.mean()))
    np.testing.assert_allclose(amp_n, amp_x, rtol=1e-8, atol=1e-8)
    assert abs(pearson(x, null)) < 0.2
    band = phase_randomized_null_band(x, BOX, sigma=2.0, n_draws=6, seed=3)
    assert band["values"].shape == (6,)
    assert abs(band["mean"]) < 3 * band["sd"] + 0.2


def test_rk_coherence_identity_and_noise_crossing():
    x = _field(3)
    res = rk_coherence(x, x, BOX)
    ok = np.isfinite(res["r_k"])
    assert ok.any()
    np.testing.assert_allclose(res["r_k"][ok], 1.0, atol=1e-9)
    assert np.isnan(rk_first_crossing(res["k_centers"], res["r_k"]))
    rng = np.random.default_rng(4)
    noisy = x + 0.5 * rng.standard_normal(x.shape)
    res2 = rk_coherence(x, noisy, BOX)
    k_half = rk_first_crossing(res2["k_centers"], res2["r_k"])
    assert np.isfinite(k_half) and k_half > 0


def test_lowpass_control_scores_between_null_and_one():
    x = _field(5, sigma=2.0)
    lp = lowpass_sharp(x, BOX, k_c=0.5)
    r = pearson(x, lp)
    assert 0.3 < r < 1.0


def test_nccf_identity_and_validity_mask():
    x = _field(6)
    res = nccf(x, x, BOX)
    assert not res["undefined"]
    assert res["pearson_zero_lag"] == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(res["nccf"][res["valid"]], 1.0, atol=1e-9)
    assert nccf(x, np.zeros_like(x), BOX)["undefined"]


def test_inference_helpers():
    x = _field(8)
    rng = np.random.default_rng(9)
    good = x + 0.3 * rng.standard_normal(x.shape)
    bad = x + 2.0 * rng.standard_normal(x.shape)
    t = paired_fisher_t(per_octant_pearson(x, good), per_octant_pearson(x, bad))
    assert t["pass"] and t["df"] == 7
    bs = block_bootstrap_delta_rs(x, good, bad, n_blocks_side=4, n_boot=200, seed=1)
    assert bs["excludes_zero"] and bs["ci95"][0] > 0

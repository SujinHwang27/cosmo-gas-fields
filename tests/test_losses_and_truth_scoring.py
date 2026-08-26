"""Data losses and the truth-scoring diagnostic on synthetic rays."""

import numpy as np
import pytest
import torch

from cosmo_gas_fields.diagnostics import (
    identifiability_margin,
    project_to_grid_basis,
    score_truth_under_loss,
)
from cosmo_gas_fields.models import FieldsModel, volume_render_physics
from cosmo_gas_fields.training import (
    masked_log1p_mse,
    masked_mean_flux,
    mean_flux_anchor_loss,
    mean_flux_linearized_coefficient,
)


def test_masked_log1p_mse_zero_on_identity_and_masking():
    tau = torch.rand(4, 16) * 3
    assert float(masked_log1p_mse(tau, tau)) == 0.0
    other = tau + 1.0
    mask = torch.zeros(4, 16, dtype=torch.bool)
    assert float(masked_log1p_mse(tau, other, mask)) == 0.0        # fully masked -> finite 0
    mask[:, :8] = True
    full = float(masked_log1p_mse(tau, other))
    half = float(masked_log1p_mse(tau, other, mask))
    assert full > 0 and half > 0


def test_cap_ignores_saturated_differences():
    a = torch.full((2, 8), 50.0)
    b = torch.full((2, 8), 500.0)
    assert float(masked_log1p_mse(a, b, tau_max=10.0)) == 0.0


def test_sat_weight_one_is_uniform_form():
    tau = torch.rand(3, 12)
    other = tau * 1.5
    sat = torch.rand(3, 12) > 0.5
    assert float(masked_log1p_mse(tau, other, sat_mask=sat, sat_weight=1.0)) == pytest.approx(
        float(masked_log1p_mse(tau, other)))
    assert float(masked_log1p_mse(tau, other, sat_mask=sat, sat_weight=3.0)) != pytest.approx(
        float(masked_log1p_mse(tau, other)))


def test_mean_flux_linearization_matches_true_gradient():
    """Gradient of c * mean_F equals gradient of lambda (mean_F - obs)^2 at the linearization point."""
    tau = (torch.rand(5, 20) * 2).requires_grad_(True)
    lam, obs = 3.0, 0.7
    mF = masked_mean_flux(tau)
    (g_true,) = torch.autograd.grad(mean_flux_anchor_loss(mF, obs, lam), tau)
    c = mean_flux_linearized_coefficient(float(mF), obs, lam)
    (g_lin,) = torch.autograd.grad(c * masked_mean_flux(tau), tau)
    assert torch.allclose(g_true, g_lin, rtol=1e-5, atol=1e-8)


def _synthetic_truth(n_rays=6, n_bins=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    coords = torch.rand(n_rays, n_bins, 3, generator=g)
    rho = 0.5 + torch.rand(n_rays, n_bins, generator=g) * 3
    fields = torch.stack([rho, 1.0e4 * rho ** 0.3, 2e-5 * rho ** 0.6, torch.zeros(n_rays, n_bins)], -1)
    vel = torch.linspace(0.0, 1200.0, n_bins)
    return fields, coords, vel


def test_truth_floor_is_zero_when_observation_is_rendered_truth():
    fields, coords, vel = _synthetic_truth()
    with torch.no_grad():
        tau_gt = volume_render_physics(FieldsModel(fields), coords, vel, tau_amp=3.0e4, window=6)
    res = score_truth_under_loss(fields, coords, vel, tau_gt, window=6,
                                 amps=np.array([1.0e4, 2.0e4, 3.0e4, 4.0e4]),
                                 extended_amps=np.array([1.0e4, 3.0e4]))
    assert res["wiring_check_pass"]
    assert res["best_tau_amp"] == pytest.approx(3.0e4)
    assert res["value"] == pytest.approx(0.0, abs=1e-10)
    assert identifiability_margin(0.01, 0.0025) == pytest.approx(4.0)


def test_truth_floor_positive_under_perturbed_observation():
    fields, coords, vel = _synthetic_truth(seed=1)
    with torch.no_grad():
        tau_gt = volume_render_physics(FieldsModel(fields), coords, vel, tau_amp=3.0e4, window=6)
    tau_gt = tau_gt * (1 + 0.3 * torch.randn_like(tau_gt)).clamp_min(0.1)
    res = score_truth_under_loss(fields, coords, vel, tau_gt, window=6,
                                 amps=np.linspace(1e4, 5e4, 9), extended_amps=np.linspace(1e3, 1e5, 20))
    assert res["value"] > 0 and np.isfinite(res["value"])


def test_projection_onto_grid_basis_is_idempotent_and_capacity_limited():
    n_rays, n_bins, G = 3, 64, 8
    rng = np.random.default_rng(0)
    rho = 0.5 + rng.random((n_rays, n_bins)) * 2
    fields = np.stack([rho, 1e4 * np.ones_like(rho), 1e-4 * np.ones_like(rho), np.zeros_like(rho)], -1)
    pos = np.linspace(0, 1, n_bins)
    deg = project_to_grid_basis(fields, pos, G)
    assert deg.shape == fields.shape
    deg2 = project_to_grid_basis(deg, pos, G)
    np.testing.assert_allclose(deg2[..., 0], deg[..., 0], rtol=1e-6, atol=1e-8)
    # A G-knot piecewise-linear function has far less high-frequency power than the input.
    assert np.abs(np.diff(deg[..., 0], axis=1)).mean() < np.abs(np.diff(fields[..., 0], axis=1)).mean()

"""Torch P_F estimator equals the NumPy reference; autograd survives; GradNorm balances."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from cosmo_gas_fields.analysis import compute_p_flux
from cosmo_gas_fields.training import (
    GradNormWrapper,
    cross_coherence_per_bin,
    pf_log_mse_loss,
    torch_p_flux,
)

SEED = 1234
N_RAYS, N_BINS = 256, 1024


def _synth_flux(rng):
    dv = 1.5
    vel = (np.arange(N_BINS) * dv).astype(np.float64)
    smooth = 0.05 * rng.standard_normal((N_RAYS, N_BINS))
    k = np.fft.rfftfreq(N_BINS, d=dv)
    sk = np.fft.rfft(smooth, axis=1) * np.exp(-((2 * np.pi * k) ** 2) / (2 * 0.05 ** 2))
    smooth = np.fft.irfft(sk, n=N_BINS, axis=1)
    pos = rng.integers(0, N_BINS, size=(N_RAYS, 20))[:, :, None]
    strength = rng.uniform(0.2, 2.0, size=(N_RAYS, 20))[:, :, None]
    width = rng.uniform(2.0, 8.0, size=(N_RAYS, 20))[:, :, None]
    bins = np.arange(N_BINS)[None, None, :]
    tau = (strength / (1.0 + ((bins - pos) / width) ** 2)).sum(axis=1)
    F = np.clip(np.exp(-tau) * (1.0 + smooth), 1e-6, 1.0)
    return F, vel


def test_torch_pf_matches_numpy_reference():
    rng = np.random.default_rng(SEED)
    for _ in range(3):
        F, vel = _synth_flux(rng)
        centers_np, P_np = compute_p_flux(F, vel)
        centers_t, P_t = torch_p_flux(torch.from_numpy(F), torch.from_numpy(vel))
        np.testing.assert_allclose(centers_t.numpy(), centers_np, rtol=1e-12)
        P_t_avg = P_t.to(torch.float64).numpy().mean(axis=0)
        ok = np.isfinite(P_np)
        assert ok.any()
        d_abs = np.abs(P_t_avg[ok] - P_np[ok])
        d_rel = d_abs / np.abs(P_np[ok]).clip(min=1e-30)
        assert (d_abs.max() <= 1e-6) or (d_rel.max() <= 1e-4)


def test_torch_pf_autograd_through_F():
    rng = np.random.default_rng(SEED + 1)
    F, vel = _synth_flux(rng)
    F_t = torch.from_numpy(F[:16]).clone().requires_grad_(True)
    _, P = torch_p_flux(F_t, torch.from_numpy(vel))
    P.sum().backward()
    assert F_t.grad is not None and torch.isfinite(F_t.grad).all() and float(F_t.grad.abs().max()) > 0


def test_pf_log_mse_zero_on_identical_and_scale_invariant():
    rng = np.random.default_rng(SEED + 2)
    F, vel = _synth_flux(rng)
    F_t = torch.from_numpy(F[:32])
    v = torch.from_numpy(vel)
    assert float(pf_log_mse_loss(F_t, F_t, v)) == pytest.approx(0.0, abs=1e-12)
    assert float(pf_log_mse_loss(0.7 * F_t, F_t, v)) == pytest.approx(0.0, abs=1e-10)


def test_cross_coherence_one_for_identical_fields():
    rng = np.random.default_rng(SEED + 3)
    F, vel = _synth_flux(rng)
    F_t = torch.from_numpy(F[:32])
    gamma = cross_coherence_per_bin(F_t, F_t, torch.from_numpy(vel))
    assert gamma.numel() > 0
    assert torch.allclose(gamma[torch.isfinite(gamma)], torch.ones(1, dtype=gamma.dtype), atol=1e-9)


@pytest.mark.parametrize("simplified", [False, True])
def test_gradnorm_weights_diverge_under_imbalanced_losses(simplified):
    torch.manual_seed(0)
    model = nn.Linear(1, 1)
    x = torch.tensor([[1.0]])
    y_a, y_b = torch.tensor([[10.0]]), torch.tensor([[-10.0]])
    gn = GradNormWrapper(initial_w=(1.0, 1.0), alpha=0.12, simplified=simplified)
    opt = torch.optim.Adam([gn.w_tau, gn.w_pf], lr=0.025)
    for _ in range(50):
        out = model(x)
        loss_tau = 100.0 * (out - y_a).pow(2).mean()
        loss_pf = (out - y_b).pow(2).mean()
        gn.initialize_L0(loss_tau, loss_pf)
        gn_loss = gn.compute_gradnorm_loss(loss_tau, loss_pf, shared_params=list(model.parameters()))
        opt.zero_grad()
        gn_loss.backward()
        opt.step()
        gn.renormalize_weights()
    w_t, w_p = gn.w_tau.item(), gn.w_pf.item()
    assert abs(w_t - 1.0) > 0.05 and abs(w_p - 1.0) > 0.05
    assert abs((w_t + w_p) - 2.0) < 1e-5


def test_gradnorm_total_loss_shape_and_ratio():
    gn = GradNormWrapper()
    total = gn.compute_total_loss(torch.tensor(2.0), torch.tensor(3.0))
    assert total.shape == () and total.item() == pytest.approx(5.0)
    assert gn.weight_ratio == pytest.approx(1.0)

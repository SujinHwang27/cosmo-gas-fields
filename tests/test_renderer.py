"""Differentiable renderer: gradient flow, amplitude linearity, RSD shift, identity."""

import torch

from cosmo_gas_fields.models import FieldsModel, IGMNeRF, volume_render_physics


def _rays(n_rays=4, n_bins=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    coords = torch.rand(n_rays, n_bins, 3, generator=g)
    vel = torch.linspace(0.0, 1000.0, n_bins)
    return coords, vel


def test_gradient_reaches_every_mlp_parameter():
    torch.manual_seed(0)
    coords, vel = _rays()
    mlp = IGMNeRF(hidden_dim=32, num_layers=6, L=3)
    tau = volume_render_physics(mlp, coords, vel, tau_amp=1.0e4, window=6)
    assert tau.shape == (4, 32)
    assert torch.all(tau >= 0)
    tau.sum().backward()
    for name, p in mlp.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_tau_is_linear_in_tau_amp():
    torch.manual_seed(1)
    coords, vel = _rays()
    mlp = IGMNeRF(hidden_dim=32, num_layers=6, L=3)
    with torch.no_grad():
        t1 = volume_render_physics(mlp, coords, vel, tau_amp=1.0, window=6)
        t2 = volume_render_physics(mlp, coords, vel, tau_amp=2.5, window=6)
    assert torch.allclose(t2, 2.5 * t1, rtol=1e-6, atol=1e-8)


def test_tau_amp_receives_gradient():
    torch.manual_seed(2)
    coords, vel = _rays()
    mlp = IGMNeRF(hidden_dim=32, num_layers=6, L=3)
    log_amp = torch.nn.Parameter(torch.tensor(8.0))
    tau = volume_render_physics(mlp, coords, vel, tau_amp=torch.exp(log_amp), window=6)
    tau.mean().backward()
    assert log_amp.grad is not None and torch.isfinite(log_amp.grad)


def test_peculiar_velocity_shifts_absorption():
    """Same fields with v_pec = 0 vs a uniform +v shift: tau must move, not change total."""
    n_rays, n_bins = 2, 64
    coords, vel = _rays(n_rays, n_bins)
    rho = torch.ones(n_rays, n_bins)
    rho[:, 30] = 50.0                                  # one absorber per ray
    temp = torch.full((n_rays, n_bins), 1.0e4)
    xhi = torch.full((n_rays, n_bins), 1.0e-4)
    f0 = torch.stack([rho, temp, xhi, torch.zeros(n_rays, n_bins)], -1)
    dv = float(vel[1] - vel[0])
    f1 = torch.stack([rho, temp, xhi, torch.full((n_rays, n_bins), 5 * dv)], -1)
    with torch.no_grad():
        tau0 = volume_render_physics(FieldsModel(f0), coords, vel, tau_amp=1.0e3, window=10)
        tau1 = volume_render_physics(FieldsModel(f1), coords, vel, tau_amp=1.0e3, window=10)
    assert tau0.argmax(dim=1).tolist() == [30, 30]
    assert tau1.argmax(dim=1).tolist() == [35, 35]
    # The absorber's integrated optical depth is conserved under the shift
    # (compare equal windows around each peak; the uniform shift also pushes
    # the last 5 source bins off the observed grid, so the full-ray sum differs).
    assert torch.allclose(tau0[:, 20:41].sum(1), tau1[:, 25:46].sum(1), rtol=1e-4)


def test_fields_model_identity_render_is_deterministic():
    n_rays, n_bins = 3, 40
    coords, vel = _rays(n_rays, n_bins)
    fields = torch.stack([torch.rand(n_rays, n_bins) + 0.5, torch.full((n_rays, n_bins), 1.5e4),
                          torch.full((n_rays, n_bins), 3e-5), torch.zeros(n_rays, n_bins)], -1)
    with torch.no_grad():
        a = volume_render_physics(FieldsModel(fields), coords, vel, tau_amp=1.0e4, window=8)
        b = volume_render_physics(FieldsModel(fields), coords, vel, tau_amp=1.0e4, window=8)
    assert torch.equal(a, b)

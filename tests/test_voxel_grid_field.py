"""Voxel-grid field: output contract, head ranges, gradient to all grids, interpolation."""

import math

import pytest
import torch

from cosmo_gas_fields.models import IGMNeRF, VoxelGridField, volume_render_physics
from cosmo_gas_fields.models.voxel_grid_field import DENSITY_LOG_EPS, MEAN_LOG_RHO_INIT
from cosmo_gas_fields.training import masked_log1p_mse


def _coords(n_rays, n_bins, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_rays, n_bins, 3, generator=g)


def test_output_shape_matches_neural_field_contract():
    coords = _coords(8, 16)
    assert VoxelGridField(grid_size=16)(coords).shape == IGMNeRF(hidden_dim=16, L=2)(coords).shape == (8, 16, 4)


def test_head_map_ranges():
    grid = VoxelGridField(grid_size=8, init_noise_std=0.0)
    with torch.no_grad():
        grid.log_rho_grid.fill_(3.0)
        grid.temp_grid.uniform_(-5.0, 30.0)
        grid.xhi_grid.uniform_(-30.0, 30.0)
        grid.vpec_grid.uniform_(-30.0, 30.0)
    density, temp, h1, vpec = grid(_coords(32, 8, 1)).unbind(-1)
    assert torch.all(density >= 0.0)
    assert torch.all(temp >= 1.0e3 - 1e-3) and torch.all(temp <= 1.0e7)
    assert torch.all(h1 >= 0.0) and torch.all(h1 <= 1.0)
    assert torch.all(vpec >= -500.0 - 1e-3) and torch.all(vpec <= 500.0 + 1e-3)


def test_softplus_vs_linear_log_contract():
    coords = _coords(4, 4, 2)
    g_soft = VoxelGridField(grid_size=8, init_noise_std=0.0, density_head="softplus")
    g_log = VoxelGridField(grid_size=8, init_noise_std=0.0, density_head="linear-log")
    with torch.no_grad():
        g_log.log_rho_grid.uniform_(-2.0, 2.0)
        g_soft.log_rho_grid.copy_(g_log.log_rho_grid)
    d_soft = g_soft(coords)[..., 0]
    d_log_lin = VoxelGridField.density_log_to_linear(g_log(coords)[..., 0])
    assert torch.allclose(d_soft, d_log_lin, atol=1e-5)


def test_constant_mean_init_recovers_unit_density():
    grid = VoxelGridField(grid_size=8, init_noise_std=0.0)
    density = grid(_coords(4, 4, 3))[..., 0]
    assert MEAN_LOG_RHO_INIT == pytest.approx(math.log10(1.0 + DENSITY_LOG_EPS))
    assert torch.allclose(density, torch.ones_like(density), atol=1e-4)


def test_gradient_flows_to_all_four_grids_under_flux_loss():
    torch.manual_seed(7)
    coords = _coords(6, 24, 5)
    vel_axis = torch.linspace(0.0, 2000.0, 24)
    grid = VoxelGridField(grid_size=12, init_noise_std=0.01)
    tau_pred = volume_render_physics(grid, coords, vel_axis, tau_amp=torch.nn.Parameter(torch.tensor(1.0)))
    loss = masked_log1p_mse(tau_pred, torch.rand(6, 24) * 2.0)
    loss.backward()
    for name in ("log_rho_grid", "temp_grid", "xhi_grid", "vpec_grid"):
        p = getattr(grid, name)
        assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0.0, name


def test_vpec_grid_feeds_rsd_shift():
    torch.manual_seed(11)
    coords = _coords(4, 24, 6)
    vel_axis = torch.linspace(0.0, 2000.0, 24)
    grid = VoxelGridField(grid_size=10, init_noise_std=0.0)
    with torch.no_grad():
        grid.log_rho_grid.uniform_(0.0, 1.0)
        grid.xhi_grid.fill_(0.0)
        grid.vpec_grid.fill_(0.0)
        tau_zero = volume_render_physics(grid, coords, vel_axis).clone()
        grid.vpec_grid.uniform_(-5.0, 5.0)
        tau_shift = volume_render_physics(grid, coords, vel_axis)
    assert not torch.allclose(tau_zero, tau_shift, atol=1e-4)


def test_interp_hits_corner_voxel():
    grid = VoxelGridField(grid_size=4, init_noise_std=0.0)
    with torch.no_grad():
        grid.log_rho_grid.fill_(0.0)
        grid.log_rho_grid[0, 0, 0] = 5.0
    raw = grid._sample_grid(grid.log_rho_grid, torch.tensor([[[0.0, 0.0, 0.0]]]))
    assert float(raw.reshape(-1)[0].detach()) == pytest.approx(5.0, abs=1e-4)


def test_conditioning_args_rejected():
    grid = VoxelGridField(grid_size=6)
    coords = _coords(2, 4, 9)
    with pytest.raises(RuntimeError):
        grid(coords, g=torch.zeros(2, 4, 1))
    with pytest.raises(RuntimeError):
        grid(coords, physics_id=torch.zeros(2, dtype=torch.long))

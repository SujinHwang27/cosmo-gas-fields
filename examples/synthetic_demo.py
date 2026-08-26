#!/usr/bin/env python
"""End-to-end CPU demo on a tiny synthetic volume (runs in well under a minute).

1. Build a small random log-density field on a periodic grid (a smoothed
   Gaussian random field, the simplest stand-in for a cosmological volume).
2. Sample sightlines through it, render optical depth with the differentiable
   Voigt renderer, and treat those spectra as the "observations".
3. Fit a small neural field to the spectra with the masked log(1+tau) loss
   and the mean-flux anchor, a few dozen optimizer steps.
4. Score the true field under the same loss (truth floor) and report the
   identifiability margin, plus r_s between the recovered and true density on
   the grid.

Nothing here is tuned; the point is that every piece is wired and
differentiable.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from cosmo_gas_fields.analysis import gaussian_smooth_periodic, pearson, phase_randomized
from cosmo_gas_fields.diagnostics import identifiability_margin, score_truth_under_loss
from cosmo_gas_fields.models import FieldsModel, IGMNeRF, volume_render_physics
from cosmo_gas_fields.training import (
    masked_log1p_mse,
    masked_mean_flux,
    mean_flux_anchor_loss,
)

torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. truth
N = 16                  # grid cells per side
BOX = 1.0               # unit cube
N_RAYS, N_BINS = 48, 64
white = rng.standard_normal((N, N, N))
log_rho = 0.6 * gaussian_smooth_periodic(white, BOX, 0.08)
log_rho -= log_rho.mean()
rho_truth = 10.0 ** log_rho                          # rho/<rho>


def sample_truth(coords: np.ndarray) -> np.ndarray:
    """Nearest-cell lookup of the truth cube at unit-cube coords (n, 3)."""
    idx = np.clip((coords * N).astype(int), 0, N - 1)
    return rho_truth[idx[:, 0], idx[:, 1], idx[:, 2]]


# Rays parallel to axis 0 at random transverse positions.
pos = np.linspace(0.0, 1.0, N_BINS, endpoint=False) + 0.5 / N_BINS
yz = rng.random((N_RAYS, 2))
coords = np.zeros((N_RAYS, N_BINS, 3))
coords[..., 0] = pos[None, :]
coords[..., 1:] = yz[:, None, :]
rho_rays = sample_truth(coords.reshape(-1, 3)).reshape(N_RAYS, N_BINS)

# Simple thermal / ionization state: T-rho power law, X_HI ~ rho^0.6 (toy).
temp_rays = 1.0e4 * rho_rays ** 0.3
xhi_rays = 2.0e-5 * rho_rays ** 0.6
vpec_rays = 30.0 * rng.standard_normal((N_RAYS, N_BINS))
truth_fields = torch.tensor(np.stack([rho_rays, temp_rays, xhi_rays, vpec_rays], -1), dtype=torch.float32)

coords_t = torch.tensor(coords, dtype=torch.float32)
vel_axis = torch.linspace(0.0, 1500.0, N_BINS)  # km/s, dv ~ 24 km/s
TAU_AMP_TRUE = 6.0e5

with torch.no_grad():
    tau_obs = volume_render_physics(FieldsModel(truth_fields), coords_t, vel_axis,
                                    tau_amp=TAU_AMP_TRUE, window=8)
mask = torch.ones_like(tau_obs, dtype=torch.bool)
mean_F_obs = float(torch.exp(-tau_obs).mean())
print(f"[demo] observations: {N_RAYS} rays x {N_BINS} bins, <F>={mean_F_obs:.3f}, "
      f"tau max={tau_obs.max():.2f}")

# ------------------------------------------------------------------- 2. fit
model = IGMNeRF(hidden_dim=64, num_layers=6, L=3)
log_tau_amp = torch.nn.Parameter(torch.log(torch.tensor(TAU_AMP_TRUE)))
opt = torch.optim.Adam(list(model.parameters()) + [log_tau_amp], lr=1e-3)
LAMBDA_F = 10.0
t0 = time.time()
N_STEPS = 300
for step in range(N_STEPS):
    opt.zero_grad(set_to_none=True)
    tau_pred = volume_render_physics(model, coords_t, vel_axis, tau_amp=torch.exp(log_tau_amp), window=8)
    loss_data = masked_log1p_mse(tau_pred, tau_obs, mask)
    loss_anchor = mean_flux_anchor_loss(masked_mean_flux(tau_pred, mask), mean_F_obs, LAMBDA_F)
    loss = loss_data + loss_anchor
    loss.backward()
    opt.step()
    if step % 50 == 0 or step == N_STEPS - 1:
        print(f"[demo] step {step:3d}  loss_data={loss_data.item():.5f}  "
              f"anchor={loss_anchor.item():.2e}  tau_amp={torch.exp(log_tau_amp).item():.3g}")
print(f"[demo] {N_STEPS} steps in {time.time() - t0:.1f}s")

# ------------------------------------------------------- 3. truth scoring
floor = score_truth_under_loss(truth_fields, coords_t, vel_axis, tau_obs, mask, window=8,
                               amps=np.linspace(0.5, 2.0, 31) * TAU_AMP_TRUE,
                               extended_amps=np.linspace(0.1, 10.0, 100) * TAU_AMP_TRUE)
margin = identifiability_margin(floor["value"], loss_data.item())
print(f"[demo] truth floor T0={floor['value']:.3e} at amp={floor['best_tau_amp']:.3g} "
      f"(wiring check {'OK' if floor['wiring_check_pass'] else 'FAIL'}); "
      f"model loss={loss_data.item():.3e}; margin T0/L={margin:.2f} "
      f"({'model below truth floor: objective not identifying' if margin > 1 else 'model above truth floor'})")

# ------------------------------------------------------------ 4. r_s score
g = np.linspace(0.0, 1.0, N, endpoint=False) + 0.5 / N
gx, gy, gz = np.meshgrid(g, g, g, indexing="ij")
grid_coords = torch.tensor(np.stack([gx, gy, gz], -1).reshape(-1, 1, 3), dtype=torch.float32)
with torch.no_grad():
    rho_hat = model(grid_coords)[..., 0].reshape(N, N, N).numpy()
x_true = np.log10(np.maximum(rho_truth, 1e-3))
x_hat = np.log10(np.maximum(rho_hat, 1e-3))
sigma = 2.0 * BOX / N
r_s = pearson(gaussian_smooth_periodic(x_true, BOX, sigma), gaussian_smooth_periodic(x_hat, BOX, sigma))
null = pearson(gaussian_smooth_periodic(x_true, BOX, sigma),
               gaussian_smooth_periodic(phase_randomized(x_true, 1), BOX, sigma))
if np.isnan(r_s):
    print("[demo] r_s undefined: reconstruction has zero variance (collapsed to a constant)")
else:
    print(f"[demo] r_s(sigma=2 cells) recon vs truth = {r_s:.3f}; phase-randomized null = {null:.3f}")
print("[demo] done")

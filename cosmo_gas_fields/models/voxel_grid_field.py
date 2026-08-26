"""Explicit four-field voxel grid: a drop-in, non-neural field producer.

Where :class:`~cosmo_gas_fields.models.neural_field.IGMNeRF` maps a coordinate
through an MLP, :class:`VoxelGridField` maps the same coordinate through
trilinear interpolation of four dense (G, G, G) parameter grids, one per
output channel, then applies the same output-head maps:

    density : grid stores log10(rho/<rho> + eps); channel 0 is linear rho/<rho>
              (``density_head='softplus'`` contract) or the raw log value
              (``density_head='linear-log'`` contract)
    temp    : softplus(raw) * 1e4 + 1e3
    h1_frac : sigmoid(raw)
    vpec    : tanh(raw) * 500

Because the forward contract is identical, it plugs into
``volume_render_physics`` unchanged. Its role is a capacity-matched
*explicit* baseline: if a free grid trained under the same flux loss beats the
neural field, the neural field's inductive bias is the bottleneck, not the loss.

Interpolation is ``torch.nn.functional.grid_sample`` (autograd-live); the grids
are leaf parameters with no in-place ops in the forward.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DENSITY_LOG_EPS = 1.0e-3
# Constant-mean log-density init: log10(1 + eps).
MEAN_LOG_RHO_INIT = math.log10(1.0 + DENSITY_LOG_EPS)


def _inv_softplus(y: float) -> float:
    return float(math.log(math.expm1(y)))


def _logit(p: float) -> float:
    return float(math.log(p / (1.0 - p)))


class VoxelGridField(nn.Module):
    """Four independent dense (G, G, G) grids -> (..., 4) field stack.

    Args:
        grid_size: G.
        init_noise_std: symmetry-breaking Gaussian noise on the constant init.
        density_head: 'softplus' (channel 0 = linear rho/<rho>) or 'linear-log'
            (channel 0 = raw stored log10 value).
        align_corners: True maps [0,1] box edges to the outermost voxel centers.
        init_temp_K, init_xhi: constant-mean inits for temperature and
            neutral fraction (v_pec initializes at 0).
    """

    def __init__(
        self,
        grid_size: int = 128,
        init_noise_std: float = 0.01,
        density_head: str = "softplus",
        align_corners: bool = True,
        init_temp_K: float = 1.0e4,
        init_xhi: float = 1.0e-5,
    ):
        super().__init__()
        if density_head not in ("softplus", "linear-log"):
            raise ValueError(
                f"density_head must be 'softplus' or 'linear-log'; got {density_head!r}"
            )
        if grid_size < 2:
            raise ValueError(f"grid_size must be >= 2; got {grid_size}")
        self.grid_size = int(grid_size)
        self.density_head = density_head
        self.align_corners = bool(align_corners)
        self.init_noise_std = float(init_noise_std)
        G = self.grid_size

        rho_mean = MEAN_LOG_RHO_INIT
        temp_softplus_target = (init_temp_K - 1.0e3) / 1.0e4
        if temp_softplus_target <= 0:
            raise ValueError(
                f"init_temp_K={init_temp_K} below the 1e3 K floor; softplus target must be > 0."
            )
        temp_raw = _inv_softplus(temp_softplus_target)
        xhi_raw = _logit(init_xhi)
        vpec_raw = 0.0

        def _init_grid(mean_val: float) -> torch.Tensor:
            base = torch.full((G, G, G), float(mean_val), dtype=torch.float32)
            if self.init_noise_std > 0:
                base = base + torch.randn_like(base) * self.init_noise_std
            return base

        self.log_rho_grid = nn.Parameter(_init_grid(rho_mean))
        self.temp_grid = nn.Parameter(_init_grid(temp_raw))
        self.xhi_grid = nn.Parameter(_init_grid(xhi_raw))
        self.vpec_grid = nn.Parameter(_init_grid(vpec_raw))

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def density_log_to_linear(log_density: torch.Tensor) -> torch.Tensor:
        """rho = clamp(10**log_density - eps, min=0)."""
        return torch.clamp(torch.pow(10.0, log_density) - DENSITY_LOG_EPS, min=0.0)

    def _sample_grid(self, grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Autograd-live trilinear sample of one (G,G,G) grid at coords in [0,1]^3."""
        orig_shape = coords.shape[:-1]
        n = int(torch.tensor(orig_shape).prod().item()) if len(orig_shape) else 1
        flat = coords.reshape(n, 3)
        gs = flat * 2.0 - 1.0  # grid_sample wants [-1, 1]
        inp = grid[None, None]  # (1, 1, G, G, G) over (D, H, W)
        # grid_sample's last dim is (x, y, z) indexing (W, H, D); reorder so
        # coordinate axis i addresses grid axis i.
        samp = torch.stack([gs[:, 2], gs[:, 1], gs[:, 0]], dim=-1).view(1, n, 1, 1, 3)
        out = F.grid_sample(
            inp, samp, mode="bilinear", padding_mode="border",
            align_corners=self.align_corners,
        ).reshape(n)
        return out.reshape(orig_shape) if len(orig_shape) else out.reshape(())

    def forward(
        self,
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        physics_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """coords -> (..., 4) stack [density, temp, h1_frac, vpec].

        ``g`` / ``physics_id`` are accepted for signature parity with the neural
        field but unsupported; passing a non-None value raises so a
        misconfigured caller fails loudly.
        """
        if g is not None:
            raise RuntimeError("VoxelGridField does not support velocity-gradient conditioning (g).")
        if physics_id is not None:
            raise RuntimeError("VoxelGridField does not support physics embedding (physics_id).")

        log_rho = self._sample_grid(self.log_rho_grid, x)
        temp_raw = self._sample_grid(self.temp_grid, x)
        xhi_raw = self._sample_grid(self.xhi_grid, x)
        vpec_raw = self._sample_grid(self.vpec_grid, x)

        if self.density_head == "softplus":
            density = self.density_log_to_linear(log_rho)
        else:
            density = log_rho
        temp = self.softplus(temp_raw) * 10**4 + 10**3
        h1_frac = self.sigmoid(xhi_raw)
        vpec = torch.tanh(vpec_raw) * 500
        return torch.stack([density, temp, h1_frac, vpec], dim=-1)

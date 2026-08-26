"""Differentiable, unbinned P_F(k_||) PSD (torch) + inertial-band mean.

Same conventions as :mod:`cosmo_gas_fields.analysis.flux_power` (mean-divide
contrast, Hann window, dv/sum w^2, k = 2 pi f) but WITHOUT log binning: the
band-integrated residual only needs a uniform mean over the FFT bins inside
the band, and skipping ``digitize`` keeps the graph simple. For the binned,
NumPy-equivalent estimator see
:func:`cosmo_gas_fields.training.flux_power_loss.torch_p_flux`.
"""

from __future__ import annotations

import torch

_K_MIN_INERTIAL = 10.0 ** -2.5
_K_MAX_INERTIAL = 10.0 ** -1.5


def compute_p_flux_torch(F: torch.Tensor, dv: float) -> "tuple[torch.Tensor, torch.Tensor]":
    """Per-sightline one-sided PSD on the full rfft frequency grid.

    Returns (k_axis (n_freq,), psd (n_sightlines, n_freq)); DC bin included.
    """
    if F.dim() != 2:
        raise ValueError(f"F must be 2D (n_sightlines, n_bins); got {tuple(F.shape)}")
    n_bins = F.shape[1]
    if dv <= 0:
        raise ValueError("dv must be > 0")
    F_mean = F.mean(dim=1, keepdim=True)
    delta_F = F / F_mean.clamp(min=1e-8) - 1.0
    window = torch.hann_window(n_bins, periodic=False, dtype=F.dtype, device=F.device)
    sum_w2 = (window * window).sum()
    F_k = torch.fft.rfft(delta_F * window.unsqueeze(0), dim=1)
    psd = (F_k.real ** 2 + F_k.imag ** 2) * (dv / sum_w2)
    correction = torch.ones_like(psd)
    if n_bins % 2 == 0:
        correction[:, 1:-1] = 2.0
    else:
        correction[:, 1:] = 2.0
    psd = psd * correction
    k_axis = 2.0 * torch.pi * torch.fft.rfftfreq(n_bins, d=dv).to(F.device)
    return k_axis, psd


def band_mean_inertial(
    psd: torch.Tensor,
    k_axis: torch.Tensor,
    k_min: float = _K_MIN_INERTIAL,
    k_max: float = _K_MAX_INERTIAL,
) -> torch.Tensor:
    """Sightline-wise mean PSD over [k_min, k_max]. Raises if no bin is in band."""
    band_mask = (k_axis >= k_min) & (k_axis <= k_max)
    if int(band_mask.sum().item()) == 0:
        raise ValueError(
            f"No FFT bins fall in band [{k_min:.4g}, {k_max:.4g}] s/km. "
            f"k_axis spans [{k_axis.min().item():.4g}, {k_axis.max().item():.4g}]."
        )
    return psd[:, band_mask].mean(dim=1)

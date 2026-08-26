"""1D flux power spectrum P_F(k_||) — NumPy evaluation reference.

The standard Lyman-alpha forest statistic. Conventions:

- input is transmitted flux F = exp(-tau), shape (n_sightlines, n_bins);
- contrast delta_F = F/<F> - 1 per sightline (invariant under F -> r F);
- Hann-windowed periodogram, one-sided PSD normalized by dv / sum(w^2);
- angular wavenumber k_|| = 2 pi f in s/km;
- log-spaced output bins between ``k_min`` and ``k_max``; empty bins -> NaN.

NumPy-only and vectorized over sightlines. The torch mirror used inside the
training loss lives in :mod:`cosmo_gas_fields.training.flux_power_loss`.
"""

from __future__ import annotations

import numpy as np


def compute_p_flux(
    F: np.ndarray,
    vel_axis_kms: np.ndarray,
    k_min: float = 10 ** -3,
    k_max: float = 10 ** -1,
    n_kbins: int = 20,
) -> "tuple[np.ndarray, np.ndarray]":
    """Sightline-averaged P_F(k_||). Returns (k_centers, P_F), both in s/km."""
    F = np.asarray(F)
    vel_axis_kms = np.asarray(vel_axis_kms)
    if F.ndim != 2:
        raise ValueError(f"F must be 2D (n_sightlines, n_bins); got {F.shape}")
    n_sl, n_bins = F.shape
    if vel_axis_kms.shape != (n_bins,):
        raise ValueError(f"vel_axis_kms shape {vel_axis_kms.shape} != ({n_bins},)")
    dv = float(vel_axis_kms[1] - vel_axis_kms[0])
    if dv <= 0:
        raise ValueError("vel_axis_kms must be strictly increasing")
    if not np.allclose(np.diff(vel_axis_kms), dv, rtol=1e-3):
        raise ValueError("vel_axis_kms must be uniformly spaced for FFT-based PSD")

    F_mean = F.mean(axis=1, keepdims=True)
    delta_F = F / F_mean - 1.0
    window = np.hanning(n_bins)
    delta_F = delta_F * window[None, :]
    F_k = np.fft.rfft(delta_F, axis=1)
    psd = (np.abs(F_k) ** 2) * (dv / np.sum(window ** 2))
    if n_bins % 2 == 0:
        psd[:, 1:-1] *= 2.0
    else:
        psd[:, 1:] *= 2.0
    psd_mean = psd.mean(axis=0)

    k_axis = 2.0 * np.pi * np.fft.rfftfreq(n_bins, d=dv)
    log_edges = np.linspace(np.log10(k_min), np.log10(k_max), n_kbins + 1)
    edges = 10 ** log_edges
    centers = 10 ** (0.5 * (log_edges[:-1] + log_edges[1:]))

    P_binned = np.full(n_kbins, np.nan, dtype=np.float64)
    valid = k_axis > 0
    k_pos = k_axis[valid]
    psd_pos = psd_mean[valid]
    idx = np.digitize(k_pos, edges) - 1
    in_range = (idx >= 0) & (idx < n_kbins)
    idx_ir = idx[in_range]
    psd_ir = psd_pos[in_range]
    if idx_ir.size > 0:
        sums = np.bincount(idx_ir, weights=psd_ir, minlength=n_kbins)
        cnts = np.bincount(idx_ir, minlength=n_kbins)
        nz = cnts > 0
        P_binned[nz] = sums[nz] / cnts[nz]
    return centers, P_binned

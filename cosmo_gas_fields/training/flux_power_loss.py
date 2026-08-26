"""Flux power spectrum loss + GradNorm multi-task balancing.

Components
----------
1. :func:`torch_p_flux` — differentiable torch re-implementation of the
   NumPy evaluation estimator :func:`cosmo_gas_fields.analysis.flux_power.compute_p_flux`
   (mean-divide contrast, Hann window with ``dv / sum w^2`` leakage compensation,
   ``rfft``, log-spaced k binning). Tested to 1e-6 abs / 1e-4 rel against the
   NumPy reference on non-empty bins.

2. :func:`pf_log_mse_loss` — log-MSE over an inertial k band, with the ray
   average taken INSIDE the log. Per-ray periodograms have chi^2_2 statistics;
   averaging in linear space first drops the log-domain noise floor to
   ~1/sqrt(N_rays) instead of accumulating the chi-square tail.

3. :func:`pf_knorm_loss` — linear-domain residual sum weighted by an
   EMA-tracked truth-side per-mode variance (inverse-variance weighting).

4. :class:`GradNormWrapper` — Chen et al. (2018) GradNorm with two task
   weights updated by a separate optimizer step on the gradient-norm balance
   loss. The full path uses ``torch.autograd.grad(create_graph=True)`` so the
   gradient norms carry autograd back to the task weights (a second-order
   graph through the FFT). A ``simplified`` proxy (G_i = w_i |L_i|) is
   provided for platforms where double-backward through FFT graphs is unstable.

Precision: the estimator computes in float64 internally so that the
equivalence test against the float64 NumPy reference is meaningful; output is
cast back to the input dtype. Empty bins are 0.0 in training (NaN would poison
the whole batch) and NaN in evaluation.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

# Inertial-range edges in s/km (angular wavenumber).
K_MIN_INERTIAL = 10.0 ** -2.5
K_MAX_INERTIAL = 10.0 ** -1.5

_DEFAULT_K_MIN = 10.0 ** -3
_DEFAULT_K_MAX = 10.0 ** -1
_DEFAULT_N_KBINS = 20


def torch_p_flux(
    F: torch.Tensor,
    vel_axis_kms: torch.Tensor,
    k_min: float = _DEFAULT_K_MIN,
    k_max: float = _DEFAULT_K_MAX,
    n_kbins: int = _DEFAULT_N_KBINS,
    empty_bin_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable per-sightline P_F(k_||), log-binned.

    Steps: delta_F = F/<F> - 1 (mean-divide, so a uniform rescaling of F
    leaves P_F invariant); Hann apodization with dv/sum(w^2) compensation;
    rfft; one-sided PSD; angular wavenumber k = 2 pi f; log-spaced binning.
    The bin index depends only on the fixed frequency grid, so the bin
    aggregation (a matrix product with a 0/1 membership matrix) is
    autograd-clean in F.

    Returns
    -------
    k_centers : (n_kbins,)   geometric bin centers, s/km
    P_binned  : (n_sightlines, n_kbins)   per-sightline P_F, s/km
    """
    if F.dim() != 2:
        raise ValueError(f"F must be 2D (n_sightlines, n_bins); got {tuple(F.shape)}")
    n_sl, n_bins = F.shape
    if vel_axis_kms.shape != (n_bins,):
        raise ValueError(f"vel_axis_kms shape {tuple(vel_axis_kms.shape)} != ({n_bins},)")

    out_dtype = F.dtype
    device = F.device
    F64 = F.to(torch.float64)
    vel64 = vel_axis_kms.to(torch.float64).to(device)
    dv = float((vel64[1] - vel64[0]).item())
    if dv <= 0:
        raise ValueError("vel_axis_kms must be strictly increasing")

    F_mean = F64.mean(dim=1, keepdim=True)
    delta_F = F64 / F_mean - 1.0

    # periodic=False matches numpy.hanning (symmetric N-point form).
    window = torch.hann_window(n_bins, periodic=False, dtype=torch.float64, device=device)
    sum_w2 = (window * window).sum()
    delta_F = delta_F * window.unsqueeze(0)

    F_k = torch.fft.rfft(delta_F, dim=1)
    psd = (F_k.real ** 2 + F_k.imag ** 2) * (dv / sum_w2)
    correction = torch.ones_like(psd)
    if n_bins % 2 == 0:
        correction[:, 1:-1] = 2.0
    else:
        correction[:, 1:] = 2.0
    psd = psd * correction

    freqs = torch.fft.rfftfreq(n_bins, d=dv).to(torch.float64).to(device)
    k_axis = 2.0 * torch.pi * freqs

    log_edges = torch.linspace(
        float(torch.log10(torch.tensor(k_min, dtype=torch.float64))),
        float(torch.log10(torch.tensor(k_max, dtype=torch.float64))),
        n_kbins + 1, dtype=torch.float64, device=device,
    )
    edges = 10.0 ** log_edges
    centers = 10.0 ** (0.5 * (log_edges[:-1] + log_edges[1:]))

    valid_freq = k_axis > 0
    k_pos = k_axis[valid_freq]
    psd_pos = psd[:, valid_freq]
    bin_idx = torch.bucketize(k_pos, edges) - 1
    in_range = (bin_idx >= 0) & (bin_idx < n_kbins)

    if in_range.any():
        bin_idx_clamped = bin_idx.clamp(min=0, max=n_kbins - 1)
        membership = torch.zeros((k_pos.shape[0], n_kbins), dtype=torch.float64, device=device)
        membership[torch.arange(k_pos.shape[0], device=device), bin_idx_clamped] = 1.0
        membership = membership * in_range.unsqueeze(1).to(torch.float64)
        counts = membership.sum(dim=0)
        sums = psd_pos @ membership
        P_binned = sums / counts.clamp(min=1.0).unsqueeze(0)
        empty_mask = counts == 0
        if bool(empty_mask.any()):
            sentinel = torch.full_like(P_binned, empty_bin_value)
            P_binned = torch.where(empty_mask.unsqueeze(0).expand_as(P_binned), sentinel, P_binned)
    else:
        P_binned = torch.full((n_sl, n_kbins), empty_bin_value, dtype=torch.float64, device=device)

    return centers.to(out_dtype), P_binned.to(out_dtype)


def pf_log_mse_loss(
    F_pred: torch.Tensor,
    F_truth: torch.Tensor,
    vel_axis_kms: torch.Tensor,
    k_min_inertial: float = K_MIN_INERTIAL,
    k_max_inertial: float = K_MAX_INERTIAL,
    n_kbins: int = _DEFAULT_N_KBINS,
    k_min: float = _DEFAULT_K_MIN,
    k_max: float = _DEFAULT_K_MAX,
    eps: float = 1e-30,
    reduction: str = "sum",
) -> torch.Tensor:
    """sum_k (log10 <P_pred>_rays(k) - log10 <P_truth>_rays(k))^2 over the inertial band.

    ``reduction='mean'`` divides by the number of inertial bins so the scale is
    independent of the binning.
    """
    if F_pred.shape != F_truth.shape:
        raise ValueError(f"F_pred {tuple(F_pred.shape)} != F_truth {tuple(F_truth.shape)}")

    centers_p, P_pred = torch_p_flux(F_pred, vel_axis_kms, k_min=k_min, k_max=k_max, n_kbins=n_kbins)
    _, P_truth = torch_p_flux(F_truth, vel_axis_kms, k_min=k_min, k_max=k_max, n_kbins=n_kbins)

    P_pred_ravg = P_pred.to(torch.float64).mean(dim=0)
    P_truth_ravg = P_truth.to(torch.float64).mean(dim=0)
    c64 = centers_p.to(torch.float64)
    band_mask = (c64 >= k_min_inertial) & (c64 <= k_max_inertial)
    if not bool(band_mask.any()):
        raise ValueError(
            f"No log-k bin centers fall in inertial range [{k_min_inertial:.4g}, "
            f"{k_max_inertial:.4g}] s/km. Check n_kbins / k_min / k_max."
        )
    log_pred = torch.log10(P_pred_ravg[band_mask].clamp_min(eps))
    log_truth = torch.log10(P_truth_ravg[band_mask].clamp_min(eps))
    sq = (log_pred - log_truth) ** 2
    if reduction == "sum":
        loss = sq.sum()
    elif reduction == "mean":
        loss = sq.mean()
    else:
        raise ValueError(f"unsupported reduction={reduction!r}; expected 'sum' or 'mean'")
    return loss.to(F_pred.dtype)


def compute_sigma_k_squared_ema(
    P_truth_batch: torch.Tensor,
    ema_prev: "torch.Tensor | None",
    decay: float = 0.99,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """EMA of the per-mode truth-side P_F(k) variance across rays.

    Truth-side (not prediction-side) so the weights are well defined at step 0,
    when all predicted rays are near-identical and their variance is ~0.
    The first call initializes directly from the batch variance.
    """
    if P_truth_batch.dim() != 2:
        raise ValueError(f"P_truth_batch must be 2D (n_rays, n_kbins); got {tuple(P_truth_batch.shape)}")
    with torch.no_grad():
        batch_var = P_truth_batch.to(torch.float64).var(dim=0, unbiased=False)
        if ema_prev is None:
            ema_new = batch_var.clone()
        else:
            ema_new = decay * ema_prev.to(batch_var.dtype) + (1.0 - decay) * batch_var
    return ema_new, ema_new


def pf_knorm_loss(
    F_pred: torch.Tensor,
    F_truth: torch.Tensor,
    vel_axis_kms: torch.Tensor,
    sigma_k_squared_truth_ema: torch.Tensor,
    k_min_inertial: float = K_MIN_INERTIAL,
    k_max_inertial: float = K_MAX_INERTIAL,
    n_kbins: int = _DEFAULT_N_KBINS,
    k_min: float = _DEFAULT_K_MIN,
    k_max: float = _DEFAULT_K_MAX,
    floor_rel: float = 0.01,
) -> torch.Tensor:
    """sum_k (P_pred(k) - P_truth(k))^2 / max(sigma_k^2, floor), linear domain.

    ``floor = floor_rel * median_k(sigma_k^2)`` over the band: a *relative*
    floor, because an absolute one lets any sub-floor mode dominate by orders
    of magnitude at typical P_F scales (1e-5 .. 1e-3 s/km).
    """
    if F_pred.shape != F_truth.shape:
        raise ValueError(f"F_pred {tuple(F_pred.shape)} != F_truth {tuple(F_truth.shape)}")
    if sigma_k_squared_truth_ema.dim() != 1 or sigma_k_squared_truth_ema.shape[0] != n_kbins:
        raise ValueError(
            f"sigma_k_squared_truth_ema must be 1D of shape ({n_kbins},); "
            f"got {tuple(sigma_k_squared_truth_ema.shape)}"
        )
    centers_p, P_pred = torch_p_flux(F_pred, vel_axis_kms, k_min=k_min, k_max=k_max, n_kbins=n_kbins)
    _, P_truth = torch_p_flux(F_truth, vel_axis_kms, k_min=k_min, k_max=k_max, n_kbins=n_kbins)
    P_pred_ravg = P_pred.to(torch.float64).mean(dim=0)
    P_truth_ravg = P_truth.to(torch.float64).mean(dim=0)
    c64 = centers_p.to(torch.float64)
    band_mask = (c64 >= k_min_inertial) & (c64 <= k_max_inertial)
    if not bool(band_mask.any()):
        raise ValueError(
            f"No log-k bin centers fall in inertial range [{k_min_inertial:.4g}, {k_max_inertial:.4g}] s/km."
        )
    sigma_band = sigma_k_squared_truth_ema.to(torch.float64)[band_mask].detach()
    floor = (floor_rel * sigma_band.median().clamp_min(1e-30)).detach()
    weights = torch.clamp_min(sigma_band, floor)
    resid_sq = (P_pred_ravg[band_mask] - P_truth_ravg[band_mask]) ** 2
    return (resid_sq / weights).sum().to(F_pred.dtype)


def inertial_rel_residual(
    F_pred: torch.Tensor,
    F_truth: torch.Tensor,
    vel_axis_kms: torch.Tensor,
    k_min_inertial: float = K_MIN_INERTIAL,
    k_max_inertial: float = K_MAX_INERTIAL,
    n_kbins: int = _DEFAULT_N_KBINS,
    k_min: float = _DEFAULT_K_MIN,
    k_max: float = _DEFAULT_K_MAX,
) -> torch.Tensor:
    """Mean over inertial bins of |<P_pred> - <P_truth>| / <P_truth> (diagnostic)."""
    centers, P_pred = torch_p_flux(F_pred, vel_axis_kms, k_min=k_min, k_max=k_max, n_kbins=n_kbins)
    _, P_truth = torch_p_flux(F_truth, vel_axis_kms, k_min=k_min, k_max=k_max, n_kbins=n_kbins)
    P_pred_ravg = P_pred.to(torch.float64).mean(dim=0)
    P_truth_ravg = P_truth.to(torch.float64).mean(dim=0)
    c64 = centers.to(torch.float64)
    band_mask = (c64 >= k_min_inertial) & (c64 <= k_max_inertial)
    rel = (P_pred_ravg[band_mask] - P_truth_ravg[band_mask]).abs() / P_truth_ravg[band_mask].clamp_min(1e-30)
    return rel.mean()


def cross_coherence_per_bin(
    F_pred: torch.Tensor,
    F_truth: torch.Tensor,
    vel_axis_kms: torch.Tensor,
    k_min_inertial: float = K_MIN_INERTIAL,
    k_max_inertial: float = K_MAX_INERTIAL,
    n_kbins: int = _DEFAULT_N_KBINS,
    k_min: float = _DEFAULT_K_MIN,
    k_max: float = _DEFAULT_K_MAX,
) -> torch.Tensor:
    """Segment-averaged magnitude-squared coherence |<S_xy>|^2 / (<S_xx><S_yy>) per inertial bin.

    The cross spectrum is averaged as a COMPLEX number over (ray, FFT bin)
    samples before taking |.|^2; a single-realization periodogram coherence is
    identically 1 by Cauchy-Schwarz, so averaging over independent samples is
    what makes the statistic informative. 1 when F_pred == F_truth (or any
    rescaling), ~0 when uncorrelated.
    """
    if F_pred.shape != F_truth.shape:
        raise ValueError("F_pred and F_truth must share shape.")
    n_sl, n_bins = F_pred.shape
    device = F_pred.device
    F64p = F_pred.to(torch.float64)
    F64t = F_truth.to(torch.float64)
    vel64 = vel_axis_kms.to(torch.float64).to(device)
    dv = float((vel64[1] - vel64[0]).item())

    delta_p = F64p / F64p.mean(dim=1, keepdim=True) - 1.0
    delta_t = F64t / F64t.mean(dim=1, keepdim=True) - 1.0
    window = torch.hann_window(n_bins, periodic=False, dtype=torch.float64, device=device)
    sum_w2 = (window * window).sum()
    Fk_p = torch.fft.rfft(delta_p * window.unsqueeze(0), dim=1)
    Fk_t = torch.fft.rfft(delta_t * window.unsqueeze(0), dim=1)

    norm = dv / sum_w2
    Sxx = (Fk_p.real ** 2 + Fk_p.imag ** 2) * norm
    Syy = (Fk_t.real ** 2 + Fk_t.imag ** 2) * norm
    Sxy_re = (Fk_p.real * Fk_t.real + Fk_p.imag * Fk_t.imag) * norm
    Sxy_im = (Fk_p.imag * Fk_t.real - Fk_p.real * Fk_t.imag) * norm

    freqs = torch.fft.rfftfreq(n_bins, d=dv).to(torch.float64).to(device)
    k_axis = 2.0 * torch.pi * freqs
    log_edges = torch.linspace(
        float(torch.log10(torch.tensor(k_min, dtype=torch.float64))),
        float(torch.log10(torch.tensor(k_max, dtype=torch.float64))),
        n_kbins + 1, dtype=torch.float64, device=device,
    )
    edges = 10.0 ** log_edges
    centers = 10.0 ** (0.5 * (log_edges[:-1] + log_edges[1:]))

    valid = k_axis > 0
    k_pos = k_axis[valid]
    bin_idx = torch.bucketize(k_pos, edges) - 1
    in_range = (bin_idx >= 0) & (bin_idx < n_kbins)
    band_mask = (centers >= k_min_inertial) & (centers <= k_max_inertial)
    band_bins = torch.where(band_mask)[0]
    out = torch.zeros(int(band_mask.sum().item()), dtype=torch.float64, device=device)
    for out_i, b in enumerate(band_bins.tolist()):
        sel = in_range & (bin_idx == b)
        if int(sel.sum().item()) == 0:
            out[out_i] = float("nan")
            continue
        Sxx_avg = Sxx[:, valid][:, sel].mean()
        Syy_avg = Syy[:, valid][:, sel].mean()
        Sxy_re_avg = Sxy_re[:, valid][:, sel].mean()
        Sxy_im_avg = Sxy_im[:, valid][:, sel].mean()
        out[out_i] = (Sxy_re_avg ** 2 + Sxy_im_avg ** 2) / (Sxx_avg * Syy_avg).clamp_min(1e-30)
    return out.to(F_pred.dtype)


class GradNormWrapper(nn.Module):
    """Chen et al. (2018) GradNorm for two tasks (``alpha=0.12`` paper default).

    Two trainable task weights ``w_tau``, ``w_pf`` (init 1.0) are updated by a
    SEPARATE optimizer step on the gradient-norm balance loss; the model
    optimizer sees ``w_tau * L_tau + w_pf * L_pf`` with the current weights.

    Usage::

        gn = GradNormWrapper(initial_w=(1.0, 1.0), alpha=0.12)
        gn_opt = torch.optim.Adam(gn.parameters(), lr=1e-3)
        # per step
        total = gn.compute_total_loss(loss_tau, loss_pf)
        gn_loss = gn.compute_gradnorm_loss(loss_tau, loss_pf, shared_params)
        total.backward(retain_graph=True); model_opt.step(); model_opt.zero_grad()
        gn_opt.zero_grad(); gn_loss.backward(); gn_opt.step()
        gn.renormalize_weights()   # keep w_tau + w_pf = T

    ``simplified=True`` substitutes G_i = w_i |L_i| for the true gradient norm
    (no second-order autograd). Use it when double-backward through an
    FFT-bearing graph is unstable on the target platform.
    """

    def __init__(self, initial_w: Tuple[float, float] = (1.0, 1.0),
                 alpha: float = 0.12, simplified: bool = False):
        super().__init__()
        if len(initial_w) != 2:
            raise ValueError("GradNormWrapper currently supports exactly 2 tasks.")
        self.simplified = bool(simplified)
        self.w_tau = nn.Parameter(torch.tensor(float(initial_w[0]), dtype=torch.float32))
        self.w_pf = nn.Parameter(torch.tensor(float(initial_w[1]), dtype=torch.float32))
        self.alpha = float(alpha)
        # Initial losses L_i(0), pinned on the first call and frozen thereafter.
        self.register_buffer("_L0_tau", torch.tensor(float("nan")))
        self.register_buffer("_L0_pf", torch.tensor(float("nan")))
        self.T = float(initial_w[0] + initial_w[1])

    @property
    def weights_clamped(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.w_tau.clamp_min(1e-4), self.w_pf.clamp_min(1e-4)

    @property
    def weight_ratio(self) -> float:
        w_t, w_p = self.weights_clamped
        return float((w_t / w_p.clamp_min(1e-30)).item())

    def initialize_L0(self, loss_tau: torch.Tensor, loss_pf: torch.Tensor) -> None:
        """Pin L_i(0). Idempotent: only the first call writes."""
        if not bool(torch.isnan(self._L0_tau)):
            return
        with torch.no_grad():
            self._L0_tau.copy_(loss_tau.detach())
            self._L0_pf.copy_(loss_pf.detach())

    def renormalize_weights(self) -> None:
        """Rescale so that w_tau + w_pf == T."""
        with torch.no_grad():
            scale = self.T / (self.w_tau + self.w_pf).clamp_min(1e-8)
            self.w_tau.mul_(scale)
            self.w_pf.mul_(scale)

    def compute_total_loss(self, loss_tau: torch.Tensor, loss_pf: torch.Tensor) -> torch.Tensor:
        w_t, w_p = self.weights_clamped
        return w_t * loss_tau + w_p * loss_pf

    def compute_gradnorm_loss(
        self,
        loss_tau: torch.Tensor,
        loss_pf: torch.Tensor,
        shared_params: Iterable[torch.Tensor],
    ) -> torch.Tensor:
        """GradNorm balance loss L_grad = sum_i |G_i - G_avg * r_i^alpha| (targets detached).

        G_i = ||grad_{shared} (w_i L_i)||_2, r_i = (L_i / L_i(0)) / mean_j(L_j / L_j(0)).
        Autograd-live in (w_tau, w_pf) only; the model parameters receive no
        gradient from this loss.
        """
        if bool(torch.isnan(self._L0_tau)):
            self.initialize_L0(loss_tau, loss_pf)
        w_t, w_p = self.weights_clamped

        if self.simplified:
            G_tau = w_t * loss_tau.detach().abs().clamp_min(1e-30)
            G_pf = w_p * loss_pf.detach().abs().clamp_min(1e-30)
        else:
            params = [p for p in shared_params
                      if p.requires_grad and p is not self.w_tau and p is not self.w_pf]
            if not params:
                raise ValueError(
                    "shared_params must contain at least one parameter with "
                    "requires_grad=True (excluding the GradNorm task weights)."
                )
            grads_tau = torch.autograd.grad(outputs=w_t * loss_tau, inputs=params,
                                            create_graph=True, retain_graph=True, allow_unused=True)
            grads_pf = torch.autograd.grad(outputs=w_p * loss_pf, inputs=params,
                                           create_graph=True, retain_graph=True, allow_unused=True)
            G_tau = (torch.sqrt(sum((g * g).sum() for g in grads_tau if g is not None).clamp_min(1e-30))
                     if any(g is not None for g in grads_tau) else torch.tensor(0.0, device=w_t.device))
            G_pf = (torch.sqrt(sum((g * g).sum() for g in grads_pf if g is not None).clamp_min(1e-30))
                    if any(g is not None for g in grads_pf) else torch.tensor(0.0, device=w_p.device))

        G_avg = (G_tau + G_pf) / 2.0
        r_tau = loss_tau.detach() / self._L0_tau.clamp_min(1e-30)
        r_pf = loss_pf.detach() / self._L0_pf.clamp_min(1e-30)
        r_mean = ((r_tau + r_pf) / 2.0).clamp_min(1e-30)
        G_target_tau = (G_avg.detach() * ((r_tau / r_mean) ** self.alpha)).detach()
        G_target_pf = (G_avg.detach() * ((r_pf / r_mean) ** self.alpha)).detach()
        return (G_tau - G_target_tau).abs() + (G_pf - G_target_pf).abs()


__all__ = [
    "K_MIN_INERTIAL", "K_MAX_INERTIAL", "torch_p_flux", "pf_log_mse_loss",
    "pf_knorm_loss", "compute_sigma_k_squared_ema", "inertial_rel_residual",
    "cross_coherence_per_bin", "GradNormWrapper",
]

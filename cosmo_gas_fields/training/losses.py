"""Data losses on rendered optical depth.

Two terms make up the flux-domain objective:

1. **Masked, capped log(1+tau) MSE** (:func:`masked_log1p_mse`).
   ``log1p`` compresses the long optical-depth tail so a handful of saturated
   absorbers do not dominate; the cap ``tau_max`` stops the loss from chasing
   exact tau values where F = exp(-tau) is numerically zero; the mask drops
   bins flagged as damped systems (or any other excluded region). The masked
   mean stays finite even when a micro-batch is fully masked (zero weight ->
   zero gradient).

2. **Mean-flux anchor** (:func:`mean_flux_anchor_loss` and the two-pass
   linearized form). A soft constraint ``lambda_F (<F_pred> - <F>_obs)^2`` on
   the *global* mean transmitted flux fixes the amplitude degeneracy between
   the field and the free ``tau_amp``. Because <F_pred> is a mean over every
   ray in an accumulation cycle, the squared loss cannot be back-propagated
   micro-batch by micro-batch directly. The two-pass trick:

   * pass 1 (no grad): compute the cycle mean ``F_cycle`` over all micro-batches;
   * pass 2: for each micro-batch back-propagate ``c * mean_F_mb`` with
     ``c = 2 lambda_F (F_cycle - F_obs)`` (:func:`mean_flux_linearized_coefficient`).

   The gradient of the surrogate equals the gradient of the true squared loss
   at the linearization point, and is invariant to how many accumulation
   steps the cycle is split into.
"""

from __future__ import annotations

from typing import Optional

import torch

DEFAULT_TAU_MAX = 10.0


def masked_log1p_mse(
    tau_pred: torch.Tensor,
    tau_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    tau_max: float = DEFAULT_TAU_MAX,
    sat_mask: Optional[torch.Tensor] = None,
    sat_weight: float = 1.0,
) -> torch.Tensor:
    """Masked mean of (log1p(min(tau_pred, tau_max)) - log1p(min(tau_gt, tau_max)))^2.

    Parameters
    ----------
    tau_pred, tau_gt : (n_rays, n_bins)
    mask : (n_rays, n_bins) bool, True = include. None = all bins.
    sat_mask, sat_weight
        Optional per-bin up-weighting of a "saturation band" (bins whose
        truth flux sits in a chosen range). Weight = mask + (sat_weight - 1) * sat_mask,
        so ``sat_weight=1`` reduces exactly to the uniform-mask form. The band
        must depend on the truth only, never on ``tau_pred``.
    """
    tpe = tau_pred.clamp_max(tau_max)
    tge = tau_gt.clamp_max(tau_max)
    diff = torch.log1p(tpe) - torch.log1p(tge)
    diff_sq = diff * diff
    if mask is None:
        weight = torch.ones_like(diff_sq)
    else:
        weight = mask.to(diff_sq.dtype)
    if sat_mask is not None and sat_weight != 1.0:
        weight = weight + (sat_weight - 1.0) * sat_mask.to(diff_sq.dtype)
    return (diff_sq * weight).sum() / weight.sum().clamp(min=1.0)


def masked_mean_flux(tau_pred: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Masked mean of F = exp(-tau) over all (ray, bin) entries. Autograd-live."""
    F_pred = torch.exp(-tau_pred)
    if mask is None:
        return F_pred.mean()
    m = mask.to(F_pred.dtype)
    return (F_pred * m).sum() / m.sum().clamp(min=1.0)


def mean_flux_anchor_loss(mean_F_pred: torch.Tensor, mean_F_obs: float, lambda_F: float) -> torch.Tensor:
    """Direct form lambda_F (<F_pred> - <F>_obs)^2 (single-batch training)."""
    return lambda_F * (mean_F_pred - mean_F_obs) ** 2


def mean_flux_linearized_coefficient(mean_F_cycle: float, mean_F_obs: float, lambda_F: float) -> float:
    """c = d/dF [lambda_F (F - F_obs)^2] evaluated at the cycle mean.

    Back-propagating ``c * masked_mean_flux(tau_pred_mb, mask_mb)`` for each
    micro-batch yields the same parameter gradient as the true squared loss on
    the whole cycle, with memory bounded by one micro-batch.
    """
    return 2.0 * lambda_F * (mean_F_cycle - mean_F_obs)


__all__ = [
    "DEFAULT_TAU_MAX",
    "masked_log1p_mse",
    "masked_mean_flux",
    "mean_flux_anchor_loss",
    "mean_flux_linearized_coefficient",
]

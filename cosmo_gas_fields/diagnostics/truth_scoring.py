"""Truth scoring: score the TRUE field under the training loss.

The question this answers: *if the optimizer had found the true field, what
loss would it see?* Call that the truth floor T0. If a trained model reaches a
data loss well BELOW T0, the loss is rewarding something other than the truth
— the objective does not identify the field, and a lower loss is not evidence
of a better reconstruction. The ratio ``margin = T0 / L_model`` summarizes this:
a margin much larger than 1 signals a non-identifying objective, not a win.

Steps
-----
1. Sample the true fields (density, T, X_HI, v_pec) along the same rays used in
   training and render them with the same renderer
   (:func:`render_fields_in_chunks`).
2. Because the renderer carries a free amplitude ``tau_amp``, scan it and take
   the minimum loss (:func:`amplitude_sweep`); the floor must not depend on a
   nuisance parameter the model is free to tune. If the optimum lands on the
   scan edge and the loss range is non-trivial, the scan is extended once.
3. Optionally build a *capacity-matched* truth: project each true field onto
   the basis a low-resolution model can represent (per-ray least squares onto
   the trilinear knots, :func:`project_to_grid_basis`) and score that too (T4).
   min(T0, T4) is the floor a model of that capacity could hope to reach.
4. Wiring check: the loss of the rendered truth against itself must be exactly 0.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from ..models.neural_field import FieldsModel, volume_render_physics
from ..training.losses import DEFAULT_TAU_MAX, masked_log1p_mse

LossFn = Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


def default_loss(tau_pred: torch.Tensor, tau_gt: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    return float(masked_log1p_mse(tau_pred, tau_gt, mask, tau_max=DEFAULT_TAU_MAX))


@torch.no_grad()
def render_fields_in_chunks(fields: torch.Tensor, coords: torch.Tensor, vel_axis: torch.Tensor,
                            tau_amp: float = 1.0, chunk: int = 128, window: int = 64) -> torch.Tensor:
    """Render known per-ray fields (n_rays, n_bins, 4) to tau (n_rays, n_obs) in ray chunks."""
    n_rays = fields.shape[0]
    n_obs = vel_axis.shape[0]
    tau = torch.zeros((n_rays, n_obs), dtype=torch.float32)
    m = FieldsModel(fields)
    for i in range(0, n_rays, chunk):
        j = min(i + chunk, n_rays)
        m.slice = slice(i, j)
        tau[i:j] = volume_render_physics(m, coords[i:j], vel_axis, tau_amp=tau_amp, window=window)
    return tau


def amplitude_sweep(tau_base: torch.Tensor, tau_gt: torch.Tensor, mask: Optional[torch.Tensor],
                    amps: np.ndarray, loss_fn: Callable = default_loss) -> Tuple[float, Dict[float, float]]:
    """Scan tau_amp over ``amps`` (tau = amp * tau_base); return (best_amp, {amp: loss})."""
    losses = {float(a): float(loss_fn(a * tau_base, tau_gt, mask)) for a in amps}
    best = min(losses, key=losses.get)
    return float(best), losses


def score_truth_under_loss(
    truth_fields: torch.Tensor,
    coords: torch.Tensor,
    vel_axis: torch.Tensor,
    tau_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    loss_fn: Callable = default_loss,
    amps: Optional[np.ndarray] = None,
    extended_amps: Optional[np.ndarray] = None,
    edge_range_rel: float = 0.005,
    chunk: int = 128,
    window: int = 64,
) -> Dict:
    """Truth floor: min over tau_amp of loss(render(truth) * amp, tau_gt).

    Returns a dict with ``value`` (the floor), ``best_tau_amp``, ``edge_flag``,
    ``extended_grid_used``, ``loss_at_amp1``, and ``wiring_check_pass`` (loss of
    the rendered truth against itself is exactly 0).
    """
    if amps is None:
        amps = np.concatenate([np.linspace(0.2, 4.0, 96), [1.0]])
    if extended_amps is None:
        extended_amps = np.concatenate([np.linspace(0.2, 16.0, 396), [1.0]])
    tau_base = render_fields_in_chunks(truth_fields, coords, vel_axis, 1.0, chunk=chunk, window=window)
    best_amp, losses = amplitude_sweep(tau_base, tau_gt, mask, amps, loss_fn)
    lvals = list(losses.values())
    loss_range_rel = (max(lvals) - min(lvals)) / max(min(lvals), 1e-30)
    at_edge = best_amp in (float(amps.min()), float(amps.max()))
    extended = False
    if at_edge and loss_range_rel > edge_range_rel:
        best_amp, losses = amplitude_sweep(tau_base, tau_gt, mask, extended_amps, loss_fn)
        extended = True
    tau_self = best_amp * tau_base
    wiring = float(loss_fn(tau_self, tau_self, mask))
    return {
        "value": losses[best_amp],
        "best_tau_amp": best_amp,
        "edge_flag": bool(at_edge),
        "grid_loss_range_rel": float(loss_range_rel),
        "extended_grid_used": bool(extended),
        "loss_at_amp1": losses.get(1.0),
        "wiring_check_pass": bool(wiring == 0.0),
        "tau_base": tau_base,
    }


def identifiability_margin(truth_floor: float, model_loss: float) -> float:
    """margin = truth_floor / model_loss. >> 1 means the loss rewards non-truth solutions."""
    return float(truth_floor) / max(float(model_loss), 1e-30)


# ---------------------------------------------- capacity-matched projection

def _inv_softplus(y):
    y = np.clip(y, 1e-9, None)
    return np.where(y > 20, y, np.log(np.expm1(y)))


def _softplus(s):
    return np.logaddexp(0.0, s)


def fields_to_raw(density, temp, h1, vpec, eps: float = 1.0e-3):
    """Physical fields -> raw pre-activation head space (inverse of the head maps)."""
    r_d = np.log10(np.maximum(density, 0.0) + eps)
    r_t = _inv_softplus((np.maximum(temp, 1.0e3 + 1e-6) - 1.0e3) / 1.0e4)
    h1c = np.clip(h1, 1e-12, 1 - 1e-12)
    r_x = np.log(h1c / (1 - h1c))
    r_v = np.arctanh(np.clip(vpec / 500.0, -1 + 1e-9, 1 - 1e-9))
    return r_d, r_t, r_x, r_v


def raw_to_fields(r_d, r_t, r_x, r_v, eps: float = 1.0e-3):
    density = np.maximum(np.power(10.0, r_d) - eps, 0.0)
    temp = _softplus(r_t) * 1.0e4 + 1.0e3
    h1 = 1.0 / (1.0 + np.exp(-r_x))
    vpec = np.tanh(r_v) * 500.0
    return density, temp, h1, vpec


def project_to_grid_basis(fields: np.ndarray, pos_axis_unit: np.ndarray, grid_size: int) -> np.ndarray:
    """Per-ray least-squares projection of each raw field onto a 1D trilinear knot basis.

    ``fields``: (n_rays, n_bins, 4) physical fields along rays that run parallel
    to one axis; ``pos_axis_unit``: (n_bins,) positions in [0, 1] along that axis;
    ``grid_size``: G knots at c_i = i / (G - 1) (align_corners convention).
    Returns the capacity-degraded (n_rays, n_bins, 4) physical fields.
    """
    n_rays, nbins, _ = fields.shape
    G = grid_size
    p = pos_axis_unit * (G - 1)
    A = np.zeros((nbins, G))
    i0 = np.floor(p).astype(int)
    fr = p - i0
    i0c = np.clip(i0, 0, G - 1)
    i1c = np.clip(i0 + 1, 0, G - 1)
    A[np.arange(nbins), i0c] += 1.0 - fr
    A[np.arange(nbins), i1c] += fr
    AtA = A.T @ A
    raw = fields_to_raw(fields[..., 0], fields[..., 1], fields[..., 2], fields[..., 3])
    degraded = []
    for f in raw:
        K = np.linalg.solve(AtA, A.T @ f.T)   # (G, n_rays)
        degraded.append((A @ K).T)            # (n_rays, nbins)
    d, t, x, v = raw_to_fields(*degraded)
    return np.stack([d, t, x, v], axis=-1)

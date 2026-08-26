"""Training-loop contract tests: overfit-one-batch and the step-N contract.

Failures these prevent:

* :func:`overfit_one_batch` — a model/loss/optimizer wiring that cannot drive
  the loss on TWO examples down by 10x in 50 steps is broken (dead gradient,
  wrong target, detached graph). Run it before any multi-hour job.
* :func:`assert_step_contract` — at a fixed early step (e.g. 100) the
  prediction must have non-trivial spread and the loss must be below its
  starting value. A model that has collapsed to a constant passes ordinary
  "loss decreased" checks; this one catches it. Call it INSIDE the training
  loop, OUTSIDE any try/except, so it raises loud.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch


def overfit_one_batch(model: torch.nn.Module, batch, loss_fn: Callable,
                      steps: int = 50, lr: float = 1.0e-3, ratio: float = 0.1,
                      grad_clip: float = 1.0) -> Dict:
    """Train on one fixed batch; gate: loss(steps) <= ratio * loss(0).

    ``batch`` is ``(inputs, targets)``; ``loss_fn(model(inputs), targets)``.
    ``loss(0)`` is the pre-update loss, ``loss(k)`` the loss after k updates.
    """
    x, y = batch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    losses: List[float] = []
    model.train()
    for step in range(steps + 1):
        loss = loss_fn(model(x), y)
        losses.append(float(loss.item()))
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
    gate_pass = losses[steps] <= ratio * losses[0]
    return {"losses": losses, "loss_0": losses[0], "loss_final": losses[steps],
            "ratio": losses[steps] / max(losses[0], 1e-30), "rule": f"loss({steps}) <= {ratio} * loss(0)",
            "verdict": "PASS" if gate_pass else "FAIL", "pass": bool(gate_pass)}


def assert_step_contract(step: int, losses: List[float], pred_std: float,
                         contract_step: int = 100, min_pred_std: float = 0.01) -> None:
    """At ``step == contract_step`` assert pred_std > min_pred_std and loss(step) < loss(1).

    ``losses`` is 1-indexed by position (losses[k-1] = loss at step k).
    """
    if step != contract_step:
        return
    assert pred_std > min_pred_std, (
        f"CONTRACT FAIL: prediction std {pred_std:.6f} <= {min_pred_std} at step {step} "
        "(model collapsed to a constant)"
    )
    assert losses[step - 1] < losses[0], (
        f"CONTRACT FAIL: loss({step})={losses[step - 1]:.6f} >= loss(1)={losses[0]:.6f}"
    )

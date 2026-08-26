"""Tepper-Garcia Voigt kernel: line-center limit, branch continuity, symmetry, gradients."""

import math

import pytest
import torch

from cosmo_gas_fields.models import tepper_garcia_voigt

_A_LYA = 4.7182e-4  # damping parameter at b = 12.85 km/s (T = 1e4 K)


def test_line_center_returns_one_minus_2a_over_sqrtpi():
    a = torch.tensor(_A_LYA, dtype=torch.float64)
    x = torch.tensor(0.0, dtype=torch.float64)
    expected = 1.0 - 2.0 * _A_LYA / math.sqrt(math.pi)
    assert tepper_garcia_voigt(a, x).item() == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_taylor_branch_matches_main_branch_at_handoff():
    a = torch.tensor(_A_LYA, dtype=torch.float64)
    x2 = torch.tensor(1.0e-4, dtype=torch.float64)
    exp_x2 = torch.exp(-x2)
    exp_2x2 = torch.exp(-2.0 * x2)
    poly = 4 * x2 ** 2 + 7 * x2 + 4 + 1.5 / x2
    bracket = exp_2x2 * poly - 1.5 / x2 - 1.0
    H_main = (exp_x2 - (a / (math.sqrt(math.pi) * x2)) * bracket).item()
    H_small = (torch.exp(-x2) - 2.0 * a / math.sqrt(math.pi)).item()
    assert abs(H_main - H_small) < 5e-7


def test_production_regime_matches_closed_form():
    a = torch.tensor(_A_LYA, dtype=torch.float64)
    x = torch.linspace(0.05, 5.0, 100, dtype=torch.float64)
    H = tepper_garcia_voigt(a, x)
    P = x * x
    R = torch.exp(-P)
    Q = 1.5 / P
    bracket = R * R * (4 * P * P + 7 * P + 4 + Q) - Q - 1.0
    H_ref = torch.clamp(R - (a / (math.sqrt(math.pi) * P)) * bracket, min=0.0)
    assert (H - H_ref).abs().max().item() < 1e-12


def test_gradient_finite_at_line_center():
    a = torch.tensor(_A_LYA, dtype=torch.float64, requires_grad=True)
    x = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    tepper_garcia_voigt(a, x).backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(x.grad).all()
    assert a.grad.item() == pytest.approx(-2.0 / math.sqrt(math.pi), rel=1e-12)


def test_symmetry_in_x():
    a = torch.tensor(_A_LYA, dtype=torch.float64)
    x = torch.linspace(0.01, 5.0, 50, dtype=torch.float64)
    assert torch.allclose(tepper_garcia_voigt(a, x), tepper_garcia_voigt(a, -x), atol=1e-14)

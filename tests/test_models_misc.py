"""U-Net / ResNet / neural-field construction and shape tests."""

import pytest
import torch

from cosmo_gas_fields.models import IGMNeRF, UNet3D, resnet18_3d
from cosmo_gas_fields.models.cnn3d import MeanOverdensityBaseline, MeanVarSkewKurtBaseline, MeanVarianceBaseline


def test_unet3d_param_envelope_and_shape():
    net = UNet3D()
    assert 5.0e6 <= net.n_parameters() <= 25.0e6
    out = net(torch.randn(1, 2, 16, 16, 16))
    assert out.shape == (1, 1, 16, 16, 16)


def test_resnet_and_baselines_shapes():
    x = torch.randn(2, 1, 16, 16, 16)
    assert resnet18_3d()(x).shape == (2, 4)
    assert MeanOverdensityBaseline()(x).shape == (2, 4)
    assert MeanVarianceBaseline()(x).shape == (2, 4)
    assert MeanVarSkewKurtBaseline()(x).shape == (2, 4)


@pytest.mark.parametrize("body_arch", ["current", "skip-rich-mlp"])
@pytest.mark.parametrize("density_head", ["softplus", "linear-log"])
def test_neural_field_variants(body_arch, density_head):
    m = IGMNeRF(hidden_dim=16, num_layers=6, L=2, body_arch=body_arch, density_head=density_head)
    out = m(torch.rand(3, 5, 3))
    assert out.shape == (3, 5, 4)
    if density_head == "softplus":
        assert torch.all(out[..., 0] >= 0)
    assert torch.all(out[..., 2] >= 0) and torch.all(out[..., 2] <= 1)


def test_physics_embedding_and_conditioning():
    m = IGMNeRF(hidden_dim=16, num_layers=6, L=2, use_physics_embedding=True, n_physics=3,
                use_velocity_gradient_conditioning=True)
    x = torch.rand(4, 6, 3)
    g = torch.randn(4, 6, 1)
    pid = torch.tensor([0, 1, 2, 1])
    assert m(x, g=g, physics_id=pid).shape == (4, 6, 4)
    with pytest.raises(RuntimeError):
        m(x, g=g)
    with pytest.raises(ValueError):
        IGMNeRF(body_arch="bogus")

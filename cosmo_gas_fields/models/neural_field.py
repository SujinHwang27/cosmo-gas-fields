"""Continuous neural field for the intergalactic medium + differentiable Voigt renderer.

Two pieces:

* :class:`IGMNeRF` maps a 3D position in the unit cube to four physical fields
  (overdensity rho/<rho>, temperature, neutral fraction, peculiar velocity).
* :func:`volume_render_physics` turns those fields, sampled along a sightline,
  into a Lyman-alpha optical depth profile tau(v) through an analytic Voigt
  line profile with redshift-space (peculiar velocity) displacement. Every
  operation is autograd-live, so a loss on tau back-propagates to the field.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fourier feature encoding: x -> [x, sin(2^l pi x), cos(2^l pi x)]_{l<L}."""

    def __init__(self, L: int = 10):
        super().__init__()
        self.L = L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., 3) -> (..., 3 + 6 L)."""
        res = [x]
        for l in range(self.L):
            freq = (2.0 ** l) * torch.pi
            res.append(torch.sin(freq * x))
            res.append(torch.cos(freq * x))
        return torch.cat(res, dim=-1)


# Numerical stabilizer for the log-density head: log10(rho/<rho> + eps).
DENSITY_LOG_EPS = 1.0e-3


class IGMNeRF(nn.Module):
    """MLP mapping 3D position -> (density, temperature, neutral fraction, v_pec).

    Parameters
    ----------
    hidden_dim, num_layers, L
        Width, depth, and number of Fourier frequencies.
    use_velocity_gradient_conditioning
        Concatenate a per-sample scalar feature ``g`` (e.g. a z-scored
        line-of-sight velocity gradient) onto the encoded coordinate.
    use_physics_embedding, n_physics, physics_embedding_dim
        Learn one embedding vector per simulation variant and inject it at the
        first layer only, so a single field can be fit jointly to several
        variants of the same volume.
    body_arch
        ``'current'`` = 4 + 4 layers with one mid-network skip re-injecting the
        encoded coordinate. ``'skip-rich-mlp'`` = every layer consumes
        ``concat([h, encoded])`` (DeepSDF-style).
    density_head
        ``'softplus'`` = channel 0 is Softplus(raw), i.e. linear rho/<rho> >= 0.
        ``'linear-log'`` = channel 0 is the raw output, interpreted as
        log10(rho/<rho> + eps). With the log head, compute the training loss
        directly on the raw output; ``density_log_to_linear`` is for probes
        only (its clamp kills the gradient below raw = -3).
    """

    def __init__(self, hidden_dim=256, num_layers=8, L=10,
                 use_velocity_gradient_conditioning: bool = False,
                 use_physics_embedding: bool = False,
                 n_physics: int = 4,
                 physics_embedding_dim: int = 16,
                 body_arch: str = "current",
                 density_head: str = "softplus"):
        super().__init__()
        if body_arch not in ("current", "skip-rich-mlp"):
            raise ValueError(
                f"body_arch must be 'current' or 'skip-rich-mlp'; got {body_arch!r}"
            )
        if density_head not in ("softplus", "linear-log"):
            raise ValueError(
                f"density_head must be 'softplus' or 'linear-log'; got {density_head!r}"
            )
        self.body_arch = body_arch
        self.density_head = density_head
        self.encoding = PositionalEncoding(L)
        self.use_velocity_gradient_conditioning = use_velocity_gradient_conditioning
        self.use_physics_embedding = use_physics_embedding
        self.n_physics = n_physics
        self.physics_embedding_dim = physics_embedding_dim

        encoded_dim = 3 + 2 * 3 * L
        g_dim = 1 if use_velocity_gradient_conditioning else 0
        e_dim = physics_embedding_dim if use_physics_embedding else 0
        in_dim = encoded_dim + g_dim + e_dim
        # The skip re-injects (encoded, g) but NOT the physics embedding.
        skip_dim = encoded_dim + g_dim

        if use_physics_embedding:
            self.physics_embedding = nn.Embedding(n_physics, physics_embedding_dim)

        if body_arch == "current":
            self.layers1 = nn.ModuleList()
            for i in range(4):
                dim = in_dim if i == 0 else hidden_dim
                self.layers1.append(nn.Linear(dim, hidden_dim))
            self.layers2 = nn.ModuleList()
            for i in range(num_layers - 4):
                dim = hidden_dim + skip_dim if i == 0 else hidden_dim
                self.layers2.append(nn.Linear(dim, hidden_dim))
            out_in_dim = hidden_dim if (num_layers - 4) > 0 else (hidden_dim + skip_dim)
        else:
            # Skip-rich body: every layer consumes concat([h, skip_in]).
            self.layers1 = nn.ModuleList()
            self.layers2 = nn.ModuleList()
            for i in range(num_layers):
                in_features = (in_dim + skip_dim) if i == 0 else (hidden_dim + skip_dim)
                self.layers2.append(nn.Linear(in_features, hidden_dim))
            out_in_dim = hidden_dim

        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(out_in_dim, 4)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g: Optional[torch.Tensor] = None,
                physics_id: Optional[torch.Tensor] = None):
        encoded = self.encoding(x)
        if g is not None:
            # g is already ~N(0,1); no positional encoding on it.
            skip_in = torch.cat([encoded, g], dim=-1)
        else:
            skip_in = encoded

        if physics_id is not None:
            if not self.use_physics_embedding:
                raise RuntimeError(
                    "physics_id passed to forward() but model was built with "
                    "use_physics_embedding=False."
                )
            e_p = self.physics_embedding(physics_id)  # (n_rays, e_dim)
            target_shape = list(encoded.shape[:-1]) + [self.physics_embedding_dim]
            e_p_expanded = e_p.unsqueeze(1).expand(target_shape)
            h_in = torch.cat([skip_in, e_p_expanded], dim=-1)
        else:
            if self.use_physics_embedding:
                raise RuntimeError(
                    "Model was built with use_physics_embedding=True but no "
                    "physics_id was passed to forward()."
                )
            h_in = skip_in

        h = h_in
        if self.body_arch == "current":
            for layer in self.layers1:
                h = self.relu(layer(h))
            h = torch.cat([h, skip_in], dim=-1)
            for layer in self.layers2:
                h = self.relu(layer(h))
        else:
            for layer in self.layers2:
                h = self.relu(layer(torch.cat([h, skip_in], dim=-1)))

        out = self.out_layer(h)
        if self.density_head == "softplus":
            density = self.softplus(out[..., 0])       # rho/<rho> >= 0
        else:
            density = out[..., 0]                      # log10(rho/<rho> + eps), raw
        temp = self.softplus(out[..., 1]) * 10**4 + 10**3  # K, ~1e3 .. 1e6
        h1_frac = self.sigmoid(out[..., 2])            # neutral fraction in [0, 1]
        vpec = torch.tanh(out[..., 3]) * 500           # km/s, +/- 500
        return torch.stack([density, temp, h1_frac, vpec], dim=-1)

    @staticmethod
    def density_log_to_linear(log_density: torch.Tensor) -> torch.Tensor:
        """rho = clamp(10**log_density - eps, min=0). Probe-side only (see class doc)."""
        return torch.clamp(torch.pow(10.0, log_density) - DENSITY_LOG_EPS, min=0.0)


def tepper_garcia_voigt(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Tepper-Garcia (2006) analytic approximation to the Voigt-Hjerting function H(a, x).

    A Taylor branch handles |x| << 1, where the closed form suffers a
    catastrophic cancellation (1.5/x^2 terms). Gradient-safe everywhere.
    """
    x2 = x ** 2
    small = x2 < 1e-4
    x2_safe = torch.where(small, torch.ones_like(x2), x2)

    exp_x2 = torch.exp(-x2_safe)
    exp_2x2 = torch.exp(-2.0 * x2_safe)
    poly = 4 * x2_safe**2 + 7 * x2_safe + 4 + 1.5 / x2_safe
    bracket = exp_2x2 * poly - 1.5 / x2_safe - 1.0
    H_main = exp_x2 - (a / (math.sqrt(math.pi) * x2_safe)) * bracket

    H_small = torch.exp(-x2) - 2.0 * a / math.sqrt(math.pi)
    H = torch.where(small, H_small, H_main)
    return torch.clamp(H, min=0.0)


def volume_render_physics(mlp, ray_points, vel_axis, tau_amp=None, window=64,
                          z=0.3, return_tau_local=False, g=None, physics_id=None):
    """Differentiable Lyman-alpha optical depth with windowed redshift-space convolution.

    Discrete form of

        tau(v_obs) = A * sum_{src in window(obs)} n_HI[src] * phi(v_obs - v_src - v_pec[src])

    where phi = H(a, x) / (b sqrt(pi)) is the normalized Voigt profile, b the
    thermal Doppler width, and a the damping parameter. For each source bin the
    profile is evaluated only at the +/- ``window`` nearest observed bins around
    the displaced line center; the profile decays faster than exp(-x^2/2), so
    truncation error is negligible for typical b and window.

    Parameters
    ----------
    mlp
        Any callable ``f(ray_points, g=None, physics_id=None) -> (..., 4)`` with
        channels (rho/<rho>, T [K], X_HI, v_pec [km/s]).
    ray_points : (n_rays, n_bins, 3)
        Sample coordinates in the unit cube.
    vel_axis : (n_obs,)
        Uniform velocity grid (km/s) shared by source and observed bins.
    tau_amp
        Optional scalar amplitude (absorbs the mean hydrogen column and the
        oscillator-strength prefactor). A free ``nn.Parameter`` in training.
    window : int
        Half-width of the convolution kernel in bins.
    return_tau_local
        Also return the per-source-bin "local" optical depth (the delta-profile
        limit) and the sampled density / temperature, for regularizers that
        compare against a power-law tau(rho, T) closure.

    Returns
    -------
    tau : (n_rays, n_obs)
    """
    fields = mlp(ray_points, g=g, physics_id=physics_id)  # (n_rays, n_bins, 4)
    density = fields[..., 0]
    temp = fields[..., 1]
    h1_frac = fields[..., 2]
    vpec = fields[..., 3]

    # Thermal Doppler width (km/s) and damping parameter.
    b = 12.85 * torch.sqrt(temp / 10000.0)
    # a = Gamma * lambda / (4 pi b); 6.063e-3 km/s yields a = 4.72e-4 at T = 1e4 K.
    a = 6.063e-3 / b

    # n_HI proxy: rho/<rho> * X_HI (mean column absorbed into tau_amp).
    n_hi = density * h1_frac

    n_obs = vel_axis.shape[0]
    n_rays, n_src = density.shape
    device = vel_axis.device
    dtype = density.dtype

    dv_per_bin = (vel_axis[-1] - vel_axis[0]) / (n_obs - 1)
    v_source = vel_axis[None, :] + vpec                          # (n_rays, n_src)
    center_idx = ((v_source - vel_axis[0]) / dv_per_bin).long()

    offsets = torch.arange(-window, window + 1, device=device)
    obs_idx = center_idx[..., None] + offsets[None, None, :]     # (n_rays, n_src, K)
    valid_mask = (obs_idx >= 0) & (obs_idx < n_obs)
    obs_idx_safe = obs_idx.clamp(0, n_obs - 1)

    v_obs_window = vel_axis[obs_idx_safe]
    dv_window = v_obs_window - v_source[..., None]
    x = dv_window / b[..., None]
    H = tepper_garcia_voigt(a[..., None], x)
    H = H * valid_mask.to(dtype)

    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype))
    contrib = (n_hi[..., None] * H) / (b[..., None] * sqrt_pi)

    tau = torch.zeros((n_rays, n_obs), dtype=dtype, device=device)
    tau.scatter_add_(1, obs_idx_safe.reshape(n_rays, -1), contrib.reshape(n_rays, -1))

    if tau_amp is not None:
        tau = tau * tau_amp

    if return_tau_local:
        amp_scalar = tau_amp if tau_amp is not None else 1.0
        tau_local = amp_scalar * n_hi * dv_per_bin / (b * sqrt_pi)
        return tau, {"tau_local": tau_local, "density": density, "temp": temp}
    return tau


class FieldsModel:
    """Adapter exposing pre-computed per-ray fields through the ``mlp`` interface.

    Lets :func:`volume_render_physics` render a *known* field (e.g. the
    simulation truth sampled along the rays) without a network.
    ``fields``: (n_rays, n_bins, 4) tensor. Slice with ``model.slice = slice(i, j)``
    to render a chunk of rays.
    """

    def __init__(self, fields: torch.Tensor):
        self.fields = fields
        self.slice = slice(None)

    def __call__(self, ray_points, g=None, physics_id=None):
        return self.fields[self.slice]

    def eval(self):
        return self

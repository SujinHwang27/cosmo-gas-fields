"""cosmo_gas_fields: neural fields for 3D gas density from 1D absorption spectra.

Subpackages
-----------
models       neural field, voxel-grid field, 3D U-Net, 3D ResNet
training     data losses (masked log-tau, mean-flux anchor), flux-power loss, GradNorm
analysis     flux power spectrum (numpy + torch), field statistics, slab coherence
diagnostics  truth scoring: score the true field under the training loss
mlops        tracker with offline fallback, identity pins, MLflow replay, contract tests
"""

__version__ = "0.1.0"

from . import analysis, diagnostics, mlops, models, training  # noqa: F401

__all__ = ["analysis", "diagnostics", "mlops", "models", "training", "__version__"]

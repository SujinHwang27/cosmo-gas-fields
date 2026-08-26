"""Cross-coherence r(k) on a non-periodic slab (e.g. a held-out test region).

When the test region is a slab cut out of a periodic cube, axis 0 is no longer
periodic; a plain FFT would ring at the cut. :func:`slab_r_k` applies a Hann
window along axis 0 only, keeps the transverse axes periodic, and shell-bins on
the anisotropic physical |k|. Two gates guard the estimator:

* identity gate — truth vs truth must return r(k) = 1 in every valid bin;
* null gate — truth vs a phase-scrambled copy of itself must return
  mean |r(k)| below a small threshold at high k (the estimator must not
  manufacture coherence).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .field_stats import default_k_edges


def slab_r_k(x: np.ndarray, y: np.ndarray, box: float, n_full: int,
             k_edges: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """r(k) between two (n0, N, N) slabs cut from an (N, N, N) periodic cube of side ``box``.

    Parameters
    ----------
    x, y : (n0, N, N) arrays; axis 0 is the cut (non-periodic) axis.
    box : side length of the full periodic cube.
    n_full : N, the full cube resolution (sets the transverse k grid).
    k_edges : shell edges; default = the same log-spaced shells used for full cubes.

    Returns
    -------
    (k_centers, r_k, mode_counts)
    """
    assert x.shape == y.shape and x.ndim == 3
    n0, n1, n2 = x.shape
    assert n1 == n_full and n2 == n_full, "transverse axes must be the full periodic size"
    box0 = box * n0 / n_full
    w0 = np.hanning(n0)[:, None, None]
    dx = (np.asarray(x, dtype=np.float64) - x.mean()) * w0
    dy = (np.asarray(y, dtype=np.float64) - y.mean()) * w0
    Fx = np.fft.rfftn(dx)
    Fy = np.fft.rfftn(dy)
    kf0, kf = 2 * np.pi / box0, 2 * np.pi / box
    k0 = np.fft.fftfreq(n0, d=1.0 / n0) * kf0
    k1 = np.fft.fftfreq(n_full, d=1.0 / n_full) * kf
    k2 = np.fft.rfftfreq(n_full, d=1.0 / n_full) * kf
    kmag = np.sqrt(k0[:, None, None] ** 2 + k1[None, :, None] ** 2 + k2[None, None, :] ** 2)
    w = np.full(Fx.shape, 2.0)
    w[..., 0] = 1.0
    if n_full % 2 == 0:
        w[..., -1] = 1.0
    pxy = (np.conj(Fx) * Fy).real * w
    pxx = (np.abs(Fx) ** 2) * w
    pyy = (np.abs(Fy) ** 2) * w
    if k_edges is None:
        k_edges = default_k_edges(box, k_max=np.pi * n_full / box * 0.9)
    idx = np.digitize(kmag.ravel(), k_edges) - 1
    nb = len(k_edges) - 1
    ok = (idx >= 0) & (idx < nb)
    io = idx[ok]
    sxy = np.bincount(io, weights=pxy.ravel()[ok], minlength=nb)
    sxx = np.bincount(io, weights=pxx.ravel()[ok], minlength=nb)
    syy = np.bincount(io, weights=pyy.ravel()[ok], minlength=nb)
    cnt = np.bincount(io, minlength=nb)
    rk = np.full(nb, np.nan)
    nz = (cnt > 0) & (sxx > 0) & (syy > 0)
    rk[nz] = sxy[nz] / np.sqrt(sxx[nz] * syy[nz])
    kc = np.sqrt(k_edges[:-1] * k_edges[1:])
    return kc, rk, cnt


def slab_identity_gate(slab: np.ndarray, box: float, n_full: int, tol: float = 1e-9) -> float:
    """Assert r(k)(slab, slab) == 1 in every valid bin; returns the max deviation."""
    _, rk, _ = slab_r_k(slab, slab, box, n_full)
    valid = np.isfinite(rk)
    dev = float(np.max(np.abs(rk[valid] - 1.0)))
    assert dev < tol, f"identity gate FAIL: max |r(k) - 1| = {dev:.3e}"
    return dev


def phase_scramble(slab: np.ndarray, seed: int) -> np.ndarray:
    """Same |FFT| amplitudes as the slab, random phases (real output)."""
    rng = np.random.default_rng(seed)
    F = np.fft.rfftn(slab - slab.mean())
    phases = np.exp(2j * np.pi * rng.random(F.shape))
    return np.fft.irfftn(np.abs(F) * phases, s=slab.shape, axes=(0, 1, 2))


def slab_null_gate(slab: np.ndarray, box: float, n_full: int, k_min: float,
                   threshold: float = 0.2, seed: int = 0) -> float:
    """Assert mean |r(k)| < threshold over k > k_min against a phase-scrambled slab."""
    kc, rk, _ = slab_r_k(slab, phase_scramble(slab, seed), box, n_full)
    hi = np.isfinite(rk) & (kc > k_min)
    null_mean = float(np.mean(np.abs(rk[hi])))
    assert null_mean < threshold, f"null gate FAIL: mean |r(k)| = {null_mean:.3f} >= {threshold}"
    return null_mean

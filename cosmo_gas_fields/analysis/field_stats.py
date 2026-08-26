"""Field-vs-field statistics for evaluating a reconstructed 3D density cube.

Evaluation protocol
-------------------
* **r_s(sigma)** — zero-lag Pearson correlation between truth and
  reconstruction after BOTH are Gaussian-smoothed at scale sigma (periodic
  FFT smoothing on the full cube; any mask is applied afterwards). Reported
  as a ladder over several sigma. It is a true Pearson coefficient.
* **r(k)** — Fourier-space cross-coherence P_xy / sqrt(P_xx P_yy) per k shell,
  plus the first k at which it drops below 0.5.
* **NCCF(r)** — configuration-space shell statistic C_xy(r)/sqrt(C_xx C_yy).
  NOT a Pearson coefficient (Cauchy-Schwarz does not bound the shell ratio);
  only reported inside a validity domain where both autocovariances are
  well above zero.
* **Chance floor** — the phase-randomized null: a field with the SAME power
  spectrum as the truth but random phases. A reconstruction that only matches
  two-point statistics scores at this level.
* **Controls** — sharp low-pass of the truth ("what a band-limited perfect
  reconstruction would score"), which acts as an achievable ceiling.
* **Inference** — Fisher-z paired t over octants and a paired block bootstrap
  for the difference of two reconstructions' r_s.

All estimators are FFT-based, periodic-box, mean-subtracted, float64.
Degenerate-variance inputs return NaN / an 'undefined' flag, never a silent 0.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

EPS_D_PINNED = 0.01
DEGENERATE_STD = 1e-12


# ------------------------------------------------------------------ helpers

def _check_cubes(x: np.ndarray, y: np.ndarray) -> int:
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.ndim != 3 or len(set(x.shape)) != 1:
        raise ValueError(f"inputs must be cubic 3D, got {x.shape}")
    return x.shape[0]


def periodic_r_grid(n: int, box: float) -> np.ndarray:
    """|r| lattice with minimum-image (periodic) offsets."""
    cell = box / n
    coord = np.arange(n) * cell
    coord = np.where(coord > box / 2.0, coord - box, coord)
    rx, ry, rz = np.meshgrid(coord, coord, coord, indexing="ij")
    return np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)


def k_grids(n: int, box: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Angular k components (2 pi / box units) for the rfftn layout."""
    kf = 2.0 * np.pi / box
    kx = np.fft.fftfreq(n, d=1.0 / n) * kf
    kz = np.fft.rfftfreq(n, d=1.0 / n) * kf
    return kx, kx.copy(), kz


def cross_corr_cube(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """C_xy(r_vec) = (1/N^3) sum_u dx(u) dy(u + r_vec), periodic, float64."""
    n = _check_cubes(x, y)
    dx = np.asarray(x, dtype=np.float64)
    dy = np.asarray(y, dtype=np.float64)
    dx = dx - dx.mean()
    dy = dy - dy.mean()
    Fx = np.fft.rfftn(dx)
    Fy = np.fft.rfftn(dy)
    return np.fft.irfftn(np.conj(Fx) * Fy, s=dx.shape, axes=(0, 1, 2)) / float(n) ** 3


def shell_bin(cube3d: np.ndarray, rmag: np.ndarray, r_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Shell means of a lag cube over |r| bins -> (means, counts)."""
    n_bins = len(r_edges) - 1
    idx = np.digitize(rmag.ravel(), r_edges) - 1
    ok = (idx >= 0) & (idx < n_bins)
    idxo = idx[ok]
    vals = cube3d.ravel()[ok]
    s = np.bincount(idxo, weights=vals, minlength=n_bins)
    c = np.bincount(idxo, minlength=n_bins)
    means = np.full(n_bins, np.nan)
    nz = c > 0
    means[nz] = s[nz] / c[nz]
    return means, c


# ----------------------------------------------------- smoothing + Pearson

def gaussian_smooth_periodic(x: np.ndarray, box: float, sigma: float) -> np.ndarray:
    """Periodic FFT Gaussian smoothing with kernel exp(-k^2 sigma^2 / 2)."""
    xf = np.asarray(x, dtype=np.float64)
    n = xf.shape[0]
    kx, ky, kz = k_grids(n, box)
    F = np.fft.rfftn(xf)
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    F *= np.exp(-0.5 * k2 * sigma ** 2)
    return np.fft.irfftn(F, s=xf.shape, axes=(0, 1, 2))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    da = a - a.mean()
    db = b - b.mean()
    sa = da.std()
    sb = db.std()
    if sa < DEGENERATE_STD or sb < DEGENERATE_STD:
        return float("nan")
    return float((da * db).mean() / (sa * sb))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata
    return pearson(rankdata(np.ravel(a), method="average"), rankdata(np.ravel(b), method="average"))


def smoothed_pearson(truth: np.ndarray, recon: np.ndarray, box: float, sigma: float,
                     mask: Optional[np.ndarray] = None) -> float:
    """r_s(sigma): smooth BOTH cubes on the full periodic box, then Pearson (optionally masked)."""
    ts = gaussian_smooth_periodic(truth, box, sigma)
    rs = gaussian_smooth_periodic(recon, box, sigma)
    if mask is not None:
        return pearson(ts[mask], rs[mask])
    return pearson(ts, rs)


def smoothed_pearson_ladder(truth: np.ndarray, recon: np.ndarray, box: float,
                            sigmas: Sequence[float] = (1.0, 2.0, 4.0),
                            mask: Optional[np.ndarray] = None) -> Dict[float, float]:
    return {float(s): smoothed_pearson(truth, recon, box, s, mask) for s in sigmas}


def per_octant_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson within each of the 8 octants of the cube."""
    n = a.shape[0]
    h = n // 2
    out = []
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                s = (slice(i * h, (i + 1) * h), slice(j * h, (j + 1) * h), slice(k * h, (k + 1) * h))
                out.append(pearson(a[s], b[s]))
    return np.array(out)


# ------------------------------------------------------------------- NCCF

def default_r_edges(box: float, n: int, r_max_frac: float = 0.25, n_bins: int = 15) -> np.ndarray:
    """Log-spaced shells from 2 cells to r_max_frac * box."""
    return np.geomspace(2.0 * box / n, r_max_frac * box, n_bins + 1)


def shell_zero_crossing(corr_cube: np.ndarray, rmag: np.ndarray, box: float, dr: float) -> float:
    """First radius where the fine-binned shell-mean autocovariance goes <= 0."""
    edges = np.arange(0.0, box / 2.0 + dr, dr)
    means, cnt = shell_bin(corr_cube, rmag, edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ok = cnt > 0
    m = means[ok]
    c = centers[ok]
    neg = np.where(m <= 0)[0]
    if len(neg) == 0:
        return float("nan")
    i = int(neg[0])
    if i == 0:
        return float(c[0])
    r0, r1 = c[i - 1], c[i]
    v0, v1 = m[i - 1], m[i]
    return float(r0 + (r1 - r0) * (v0 / (v0 - v1)))


def nccf(x: np.ndarray, y: np.ndarray, box: float,
         r_edges: Optional[np.ndarray] = None, eps_d: float = EPS_D_PINNED) -> Dict:
    """Normalized cross-correlation function, ratio-of-shell-means.

    Only shells with min(C_xx(r), C_yy(r)) > eps_d * C_xx(0) are reported;
    others are NaN. Also returns the zero-lag Pearson and the zero-crossing
    radii of both autocovariances.
    """
    n = _check_cubes(x, y)
    if r_edges is None:
        r_edges = default_r_edges(box, n)
    sx = float(np.asarray(x, dtype=np.float64).std())
    sy = float(np.asarray(y, dtype=np.float64).std())
    if sx < DEGENERATE_STD or sy < DEGENERATE_STD:
        return {"undefined": True, "reason": f"degenerate variance (std_x={sx:.3e}, std_y={sy:.3e})"}
    cxy = cross_corr_cube(x, y)
    cxx = cross_corr_cube(x, x)
    cyy = cross_corr_cube(y, y)
    rmag = periodic_r_grid(n, box)
    m_xy, cnt = shell_bin(cxy, rmag, r_edges)
    m_xx, _ = shell_bin(cxx, rmag, r_edges)
    m_yy, _ = shell_bin(cyy, rmag, r_edges)
    cxx0 = float(cxx[0, 0, 0])
    cyy0 = float(cyy[0, 0, 0])
    cxy0 = float(cxy[0, 0, 0])
    valid = (np.minimum(m_xx, m_yy) > eps_d * cxx0) & (cnt > 0)
    prof = np.full(len(m_xy), np.nan)
    prof[valid] = m_xy[valid] / np.sqrt(m_xx[valid] * m_yy[valid])
    dr = box / n
    return {
        "undefined": False,
        "r_edges": r_edges,
        "r_centers": np.sqrt(r_edges[:-1] * r_edges[1:]),
        "nccf": prof,
        "shell_C_xy": m_xy, "shell_C_xx": m_xx, "shell_C_yy": m_yy,
        "mode_counts": cnt, "valid": valid, "eps_d": eps_d,
        "C_xx0": cxx0, "C_yy0": cyy0, "C_xy0": cxy0,
        "pearson_zero_lag": cxy0 / np.sqrt(cxx0 * cyy0),
        "r_zc_xx": shell_zero_crossing(cxx, rmag, box, dr),
        "r_zc_yy": shell_zero_crossing(cyy, rmag, box, dr),
    }


# ------------------------------------------------------------------- r(k)

def default_k_edges(box: float, k_max: float, n_bins: int = 16) -> np.ndarray:
    """Log-spaced shells from the fundamental mode to k_max."""
    kf = 2.0 * np.pi / box
    return np.geomspace(kf * 0.999, k_max, n_bins + 1)


def rk_coherence(x: np.ndarray, y: np.ndarray, box: float,
                 k_edges: Optional[np.ndarray] = None) -> Dict:
    """r(k) = P_xy / sqrt(P_xx P_yy) per |k| shell on a periodic cube."""
    n = _check_cubes(x, y)
    if k_edges is None:
        k_edges = default_k_edges(box, k_max=np.pi * n / box * 0.9)
    dx = np.asarray(x, dtype=np.float64)
    dy = np.asarray(y, dtype=np.float64)
    dx = dx - dx.mean()
    dy = dy - dy.mean()
    Fx = np.fft.rfftn(dx)
    Fy = np.fft.rfftn(dy)
    kx, ky, kz = k_grids(n, box)
    kmag = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)
    # rfftn stores half the kz planes; interior planes represent +/- kz modes.
    w = np.full(Fx.shape, 2.0)
    w[..., 0] = 1.0
    if n % 2 == 0:
        w[..., -1] = 1.0
    pxy = (np.conj(Fx) * Fy).real * w
    pxx = (np.abs(Fx) ** 2) * w
    pyy = (np.abs(Fy) ** 2) * w
    idx = np.digitize(kmag.ravel(), k_edges) - 1
    n_bins = len(k_edges) - 1
    ok = (idx >= 0) & (idx < n_bins)
    idxo = idx[ok]
    sxy = np.bincount(idxo, weights=pxy.ravel()[ok], minlength=n_bins)
    sxx = np.bincount(idxo, weights=pxx.ravel()[ok], minlength=n_bins)
    syy = np.bincount(idxo, weights=pyy.ravel()[ok], minlength=n_bins)
    cnt = np.bincount(idxo, minlength=n_bins)
    rk = np.full(n_bins, np.nan)
    nz = (cnt > 0) & (sxx > 0) & (syy > 0)
    rk[nz] = sxy[nz] / np.sqrt(sxx[nz] * syy[nz])
    return {"k_edges": k_edges, "k_centers": np.sqrt(k_edges[:-1] * k_edges[1:]),
            "r_k": rk, "mode_counts": cnt}


def rk_first_crossing(k_centers: np.ndarray, rk: np.ndarray, level: float = 0.5) -> float:
    """First k (linear interpolation) where r(k) drops below `level`; NaN if never."""
    ok = np.isfinite(rk)
    kc = k_centers[ok]
    rr = rk[ok]
    if len(rr) == 0:
        return float("nan")
    below = np.where(rr < level)[0]
    if len(below) == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(kc[0])
    k0, k1 = kc[i - 1], kc[i]
    v0, v1 = rr[i - 1], rr[i]
    return float(k0 + (k1 - k0) * ((v0 - level) / (v0 - v1)))


# ---------------------------------------------------- nulls and controls

def amplitude_matched_grf(amplitude: np.ndarray, seed: int, shape: Tuple[int, int, int]) -> np.ndarray:
    """Random-phase real field with EXACTLY the given rfftn |amplitude| (DC zeroed).

    Phases come from the rfftn of a white-noise realization, which guarantees
    the Hermitian symmetry a real field requires.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(shape)
    W = np.fft.rfftn(w)
    mag = np.abs(W)
    mag[mag == 0] = 1.0
    F = amplitude * (W / mag)
    F[0, 0, 0] = 0.0
    return np.fft.irfftn(F, s=shape, axes=(0, 1, 2))


def field_amplitude(x: np.ndarray) -> np.ndarray:
    dx = np.asarray(x, dtype=np.float64)
    return np.abs(np.fft.rfftn(dx - dx.mean()))


def phase_randomized(x: np.ndarray, seed: int) -> np.ndarray:
    """The chance-floor null: identical |FFT| amplitudes, randomized phases, mean restored."""
    out = amplitude_matched_grf(field_amplitude(x), seed, x.shape)
    return out + float(np.asarray(x, dtype=np.float64).mean())


def phase_randomized_null_band(truth: np.ndarray, box: float, sigma: float,
                               n_draws: int = 20, seed: int = 0,
                               mask: Optional[np.ndarray] = None) -> Dict:
    """Distribution of r_s(sigma) between the truth and n phase-randomized copies.

    Returns mean, SD, and the 3*SD bar: any reconstruction inside +/- 3 SD of
    this band is indistinguishable from "right power spectrum, wrong structure".
    """
    ts = gaussian_smooth_periodic(truth, box, sigma)
    vals = []
    for i in range(n_draws):
        null = gaussian_smooth_periodic(phase_randomized(truth, seed + i), box, sigma)
        vals.append(pearson(ts[mask], null[mask]) if mask is not None else pearson(ts, null))
    vals = np.asarray(vals)
    sd = float(vals.std(ddof=1)) if n_draws > 1 else float("nan")
    return {"values": vals, "mean": float(vals.mean()), "sd": sd, "bar_3sd": 3.0 * sd}


def lowpass_sharp(x: np.ndarray, box: float, k_c: float) -> np.ndarray:
    """Achievable-ceiling control: sharp k-space cutoff retaining |k| <= k_c."""
    xf = np.asarray(x, dtype=np.float64)
    n = xf.shape[0]
    kx, ky, kz = k_grids(n, box)
    kmag = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)
    F = np.fft.rfftn(xf)
    F[kmag > k_c] = 0.0
    return np.fft.irfftn(F, s=xf.shape, axes=(0, 1, 2))


# ------------------------------------------------------------- inference

def fisher_z(r) -> np.ndarray:
    r = np.clip(np.asarray(r, dtype=np.float64), -1 + 1e-12, 1 - 1e-12)
    return np.arctanh(r)


def paired_fisher_t(r_a_oct: np.ndarray, r_b_oct: np.ndarray, t_crit: float = 2.365) -> Dict:
    """Paired t over octants on Fisher-z transformed per-octant correlations (df = 7)."""
    d = fisher_z(r_a_oct) - fisher_z(r_b_oct)
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    t = m / (sd / np.sqrt(len(d))) if sd > 0 else float("inf") * np.sign(m or 1.0)
    return {"mean_dz": m, "sd_dz": sd, "t": float(t), "df": len(d) - 1,
            "t_crit": t_crit, "pass": bool(t >= t_crit)}


def _block_sums(a: np.ndarray, b: np.ndarray, n_blocks_side: int):
    n = a.shape[0]
    bs = n // n_blocks_side
    a4 = a.reshape(n_blocks_side, bs, n_blocks_side, bs, n_blocks_side, bs)
    b4 = b.reshape(n_blocks_side, bs, n_blocks_side, bs, n_blocks_side, bs)
    ax = (1, 3, 5)
    return {
        "n_cell": float(bs ** 3),
        "sa": a4.sum(axis=ax).ravel(), "sb": b4.sum(axis=ax).ravel(),
        "saa": (a4 ** 2).sum(axis=ax).ravel(),
        "sbb": (b4 ** 2).sum(axis=ax).ravel(),
        "sab": (a4 * b4).sum(axis=ax).ravel(),
    }


def _pearson_from_sums(st, idx) -> float:
    ncell = st["n_cell"] * len(idx)
    sa = st["sa"][idx].sum(); sb = st["sb"][idx].sum()
    saa = st["saa"][idx].sum(); sbb = st["sbb"][idx].sum()
    sab = st["sab"][idx].sum()
    va = saa / ncell - (sa / ncell) ** 2
    vb = sbb / ncell - (sb / ncell) ** 2
    if va <= 0 or vb <= 0:
        return float("nan")
    return float((sab / ncell - sa * sb / ncell ** 2) / np.sqrt(va * vb))


def block_bootstrap_delta_rs(truth: np.ndarray, obj_a: np.ndarray, obj_b: np.ndarray,
                             n_blocks_side: int = 8, n_boot: int = 1000, seed: int = 0) -> Dict:
    """Paired block bootstrap of Delta r = r(a, truth) - r(b, truth).

    Sub-cube blocks are resampled with replacement, the same resample for both
    objects. The 95% CI should exclude 0 for a claimed difference.
    """
    st_a = _block_sums(truth, obj_a, n_blocks_side)
    st_b = _block_sums(truth, obj_b, n_blocks_side)
    nb = n_blocks_side ** 3
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, nb, nb)
        deltas[i] = _pearson_from_sums(st_a, idx) - _pearson_from_sums(st_b, idx)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"n_boot": n_boot, "n_blocks": nb, "block_cells_side": truth.shape[0] // n_blocks_side,
            "delta_mean": float(np.nanmean(deltas)), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}

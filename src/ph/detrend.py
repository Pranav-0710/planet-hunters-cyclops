"""Detrending shared by every feature family (issue #3).

clean = relative flux minus the camera/CCD common trend (top principal components of that CCD's stars,
        fitted per star), with data gaps set to NaN
slow  = ~1 day running median of clean (rotation, spots, slow pulsation)
fast  = clean - slow, with momentum-dump and scattered-light cadences also set to NaN (dip and flare search)

Everything is computed within one sector, so train and test each get their own trends.
"""

import numpy as np
import pandas as pd

from ph.data import Split, relative_flux

N_COMPONENTS = 3
SLOW_WINDOW = 49  # cadences, ~1 day at 30 min
DUMP_PAD = 1  # also mask this many cadences on each side of a momentum dump


def quality_masks(cadences: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(gap, noisy): gap = quality & 4; noisy = momentum dumps (padded) or scattered light."""
    q = cadences.quality.to_numpy()
    gap = (q & 4) > 0
    dump = (q & 1) > 0
    padded = dump.copy()
    for k in range(1, DUMP_PAD + 1):
        padded[k:] |= dump[:-k]
        padded[:-k] |= dump[k:]
    return gap, padded | ((q & 2) > 0)


def _robust_scale(f: np.ndarray) -> np.ndarray:
    med = np.nanmedian(f, 1, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(f - med), 1, keepdims=True) + 1e-9


def ccd_basis(f: np.ndarray, n_components: int = N_COMPONENTS) -> np.ndarray:
    """Top principal components (n_components, t) of one CCD's noise-normalized curves."""
    z = f / _robust_scale(f)
    z = np.clip(np.nan_to_num(z), -5, 5)  # strong variables and transits do not drive the basis
    _, _, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
    return vt[:n_components]


def _weighted_fit(design: np.ndarray, f: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-row least squares with 0/1 weights, all rows at once via normal equations. Returns (n, k)."""
    y = np.where(w, f, 0.0)
    ata = np.einsum("tk,nt,tj->nkj", design, w.astype(np.float64), design)
    aty = np.einsum("tk,nt->nk", design, y)
    return np.linalg.solve(ata + 1e-9 * np.eye(design.shape[1]), aty[..., None])[..., 0]


def remove_common_trend(f: np.ndarray, basis: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Least-squares fit of each row on [1, basis] over valid points, with one 4-sigma clipping pass."""
    design = np.vstack([np.ones(basis.shape[1]), basis]).T
    ok = valid[None, :] & ~np.isnan(f)
    coef = _weighted_fit(design, f, ok)
    resid = np.where(ok, f - coef @ design.T, np.nan)
    sig = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid, 1, keepdims=True)), 1, keepdims=True)
    keep = ok & (np.abs(np.nan_to_num(resid)) < 4 * sig + 1e-12)
    enough = keep.sum(1) > design.shape[1] * 10
    coef[enough] = _weighted_fit(design, f[enough], keep[enough])
    return f - coef[:, 1:] @ design[:, 1:].T


def running_median(f: np.ndarray, window: int = SLOW_WINDOW) -> np.ndarray:
    """Centered running median along time that skips NaN; NaN where the input is NaN."""
    df = pd.DataFrame(f.T)
    slow = df.rolling(window, center=True, min_periods=window // 4).median().to_numpy().T.copy()
    slow[np.isnan(f)] = np.nan
    return slow


def detrend_split(s: Split) -> dict[str, np.ndarray]:
    """Return clean, slow, fast (n, t) float32 and cen (n, 2, t) float32 for one sector."""
    gap, noisy = quality_masks(s.cadences)
    f = relative_flux(s.flux)
    f[:, gap] = np.nan

    cen = np.asarray(s.centroid, np.float32)
    cen = cen - np.nanmedian(cen, axis=2, keepdims=True)
    cen[:, :, gap] = np.nan

    clean = np.empty_like(f)
    for _, idx in s.stars.groupby(["camera", "ccd"]).indices.items():
        basis = ccd_basis(f[idx])
        clean[idx] = remove_common_trend(f[idx], basis, valid=~gap & ~noisy)
        for c in range(2):
            cen[idx, c] -= np.nanmedian(cen[idx, c], axis=0)

    slow = running_median(clean)
    fast = clean - slow
    fast[:, noisy] = np.nan
    return {
        "clean": clean.astype(np.float32),
        "slow": slow.astype(np.float32),
        "fast": fast.astype(np.float32),
        "cen": cen.astype(np.float32),
    }

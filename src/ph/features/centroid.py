"""Centroid features (issue #5): does the light centroid move during the dip?

Inputs per star: t, fast relative flux, cen (2, t) detrended centroid in millipixels, the BLS ephemeris.
"""

import numpy as np
from scipy import ndimage

from ph.features.transit import in_transit_mask, robust_std


def high_pass(x: np.ndarray, window: int = 49) -> np.ndarray:
    ok = ~np.isnan(x)
    if ok.sum() < window:
        return np.full_like(x, np.nan)
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[ok], x[ok])
    out = filled - ndimage.median_filter(filled, size=window, mode="nearest")
    out[~ok] = np.nan
    return out


def centroid_features(t: np.ndarray, fast: np.ndarray, cen: np.ndarray, ephem: dict) -> dict:
    feats: dict[str, float] = {}
    intr = in_transit_mask(t, ephem["period"], ephem["duration"], ephem["t0"])
    shifts_snr = []
    depth = -np.nanmean(fast[intr]) if intr.any() else 0.0
    for c, name in ((0, "col"), (1, "row")):
        x = high_pass(cen[c])
        noise = robust_std(x)
        ok_in = intr & ~np.isnan(x)
        ok_out = ~intr & ~np.isnan(x)
        shift = np.nanmean(x[ok_in]) - np.nanmean(x[ok_out]) if ok_in.sum() > 1 else 0.0
        snr = shift / (noise / np.sqrt(max(ok_in.sum(), 1)))
        shifts_snr.append(snr)
        feats[f"cen_{name}_noise"] = noise
        feats[f"cen_{name}_shift"] = shift
        feats[f"cen_{name}_shift_snr"] = snr

        both = ~np.isnan(x) & ~np.isnan(fast)
        if both.sum() > 10:
            a, b = fast[both] - fast[both].mean(), x[both] - x[both].mean()
            feats[f"cen_{name}_corr"] = float((a * b).sum() / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-12))
        else:
            feats[f"cen_{name}_corr"] = 0.0

        d = np.diff(x)
        feats[f"cen_{name}_jumps"] = float(np.nansum(np.abs(d) > 6 * robust_std(d)))
        feats[f"cen_{name}_maxabs_sig"] = float(np.nanmax(np.abs(x)) / noise) if np.isfinite(x).any() else 0.0

    feats["cen_shift_snr"] = float(np.hypot(*shifts_snr))
    feats["cen_shift_per_depth"] = float(
        np.hypot(feats["cen_col_shift"], feats["cen_row_shift"]) / (abs(depth) * 1e3 + 1e-6)
    )
    return feats

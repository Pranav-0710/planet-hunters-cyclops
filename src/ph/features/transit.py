"""Transit and eclipse features (issue #4): BLS search on the fast curve, then vet the best signal.

Every function takes one star: t (days), fast relative flux with NaN gaps, and returns floats.
"""

import numpy as np
from astropy.timeseries import BoxLeastSquares

PERIODS = 1 / np.linspace(1 / 13.7, 1 / 0.4, 6000)
DURATIONS = [0.04, 0.08, 0.13, 0.2, 0.3]
SINGLE_WIDTHS = (3, 6, 12)


def robust_std(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) + 1e-9


def in_transit_mask(t: np.ndarray, period: float, duration: float, t0: float, frac: float = 1.0) -> np.ndarray:
    phase = (t - t0 + 0.5 * period) % period - 0.5 * period
    return np.abs(phase) < 0.5 * duration * frac


def bls_search(t: np.ndarray, f: np.ndarray, durations: list[float] = DURATIONS) -> tuple[BoxLeastSquares, dict]:
    ok = ~np.isnan(f)
    bls = BoxLeastSquares(t[ok], f[ok], dy=np.full(ok.sum(), robust_std(f)))
    r = bls.power(PERIODS, durations, objective="likelihood")
    i = int(np.argmax(r.power))
    best = {
        "period": float(r.period[i]), "duration": float(r.duration[i]), "t0": float(r.transit_time[i]),
        "depth": float(r.depth[i]), "depth_snr": float(r.depth_snr[i]), "power": float(r.power[i]),
    }
    # How peaked the periodogram is: best power vs the typical power level
    best["power_z"] = float((r.power[i] - np.median(r.power)) / (robust_std(r.power) + 1e-12))
    return bls, best


def transit_features(
    t: np.ndarray, fast: np.ndarray, dump_times: np.ndarray, durations: list[float] = DURATIONS
) -> tuple[dict, dict]:
    """Return (features, ephemeris). Ephemeris feeds the centroid features."""
    sig = robust_std(fast)
    bls, b = bls_search(t, fast, durations)
    st = bls.compute_stats(b["period"], b["duration"], b["t0"])

    d_odd, e_odd = st["depth_odd"]
    d_even, e_even = st["depth_even"]
    d_half, e_half = st["depth_half"]
    counts = np.asarray(st["per_transit_count"])
    ll = np.asarray(st["per_transit_log_likelihood"])
    ll_pos = np.clip(ll, 0, None)

    in_all = in_transit_mask(t, b["period"], b["duration"], b["t0"]) & ~np.isnan(fast)
    in_core = in_transit_mask(t, b["period"], b["duration"], b["t0"], frac=0.5) & ~np.isnan(fast)
    depth_all = -np.nanmean(fast[in_all]) if in_all.any() else 0.0
    depth_core = -np.nanmean(fast[in_core]) if in_core.any() else 0.0

    tt = np.asarray(st["transit_times"])
    near_dump = (
        np.mean([np.min(np.abs(dump_times - x)) < 0.1 for x in tt[counts > 0]])
        if len(dump_times) and (counts > 0).any() else 0.0
    )

    feats = {
        "bls_period": b["period"],
        "bls_log_period": np.log10(b["period"]),
        "bls_duration": b["duration"],
        "bls_duration_frac": b["duration"] / b["period"],
        "bls_depth": b["depth"],
        "bls_depth_sig": b["depth"] / sig,
        "bls_depth_snr": b["depth_snr"],
        "bls_power": b["power"],
        "bls_power_z": b["power_z"],
        "odd_even_snr": abs(d_odd - d_even) / np.sqrt(e_odd**2 + e_even**2 + 1e-18),
        "odd_even_ratio": min(d_odd, d_even) / (max(d_odd, d_even) + 1e-12),
        "secondary_snr": d_half / (e_half + 1e-12),
        "secondary_ratio": d_half / (b["depth"] + 1e-12),
        "harmonic_dll": float(st["harmonic_delta_log_likelihood"]),
        "harmonic_amp_ratio": float(st["harmonic_amplitude"]) / (b["depth"] + 1e-12),
        "n_transits": float((counts > 0).sum()),
        "single_transit_dominance": float(ll_pos.max() / (ll_pos.sum() + 1e-12)) if len(ll) else 1.0,
        "per_transit_ll_cv": float(np.std(ll[counts > 0]) / (abs(np.mean(ll[counts > 0])) + 1e-12)) if (counts > 0).sum() > 1 else 0.0,
        "v_shape": depth_core / (depth_all + 1e-12),
        "transit_near_dump_frac": float(near_dump),
        "fast_sig": sig,
    }

    # Deepest single event at a few box widths (single transits, glitch dips)
    filled = np.where(np.isnan(fast), 0.0, fast)
    valid = (~np.isnan(fast)).astype(float)
    best_snr, best_w = 0.0, 0
    for w in SINGLE_WIDTHS:
        k = np.ones(w)
        n = np.convolve(valid, k, "same")
        mean = np.convolve(filled, k, "same") / np.maximum(n, 1)
        snr = np.where(n >= w * 0.6, -mean / (sig / np.sqrt(np.maximum(n, 1))), 0.0)
        if snr.max() > best_snr:
            best_snr, best_w = float(snr.max()), w
    feats["single_dip_snr"] = best_snr
    feats["single_dip_width"] = float(best_w)
    feats["single_vs_bls"] = best_snr / (abs(b["depth_snr"]) + 1e-9)

    ephem = {"period": b["period"], "duration": b["duration"], "t0": b["t0"]}
    return feats, ephem

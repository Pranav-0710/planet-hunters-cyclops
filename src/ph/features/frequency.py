"""Frequency, variability and glitch features (issue #6).

Separates starspots (days, evolving), pulsating stars (hours, stable, possibly aliased near Nyquist),
flares (bright spikes with tails), instrument glitches (steps, ramps, one-orbit anomalies) and quiet stars.
"""

import numpy as np
from astropy.timeseries import LombScargle
from scipy import stats

from ph.features.transit import robust_std

NYQUIST = 24.0  # cycles/day at 30-minute cadence
STEP_WINDOW = 24


def _peaks(freq: np.ndarray, power: np.ndarray, k: int = 3) -> list[int]:
    idx = []
    p = power.copy()
    for _ in range(k):
        i = int(np.argmax(p))
        idx.append(i)
        lo, hi = np.searchsorted(freq, [freq[i] * 0.9, freq[i] * 1.1])
        p[lo:hi] = 0
    return idx


def periodogram_features(t: np.ndarray, f: np.ndarray, fmin: float, fmax: float, prefix: str) -> dict:
    ok = ~np.isnan(f)
    freq = np.linspace(fmin, fmax, 4000)
    power = LombScargle(t[ok], f[ok]).power(freq)
    top = _peaks(freq, power)
    f1 = freq[top[0]]
    harm = np.argmin(np.abs(freq - 2 * f1))
    return {
        f"{prefix}freq1": float(f1),
        f"{prefix}power1": float(power[top[0]]),
        f"{prefix}power2_ratio": float(power[top[1]] / (power[top[0]] + 1e-12)),
        f"{prefix}freq2_ratio": float(freq[top[1]] / f1),
        f"{prefix}harm_power": float(power[harm] / (power[top[0]] + 1e-12)) if 2 * f1 <= fmax else 0.0,
        f"{prefix}power_median": float(np.median(power)),
        f"{prefix}n_strong": float((power > 0.1).sum() / len(power)),
    }


def acf_at(f: np.ndarray, lag: int) -> float:
    x = np.where(np.isnan(f), 0.0, f - np.nanmean(f))
    if lag <= 0 or lag >= len(x) - 10:
        return 0.0
    return float((x[:-lag] * x[lag:]).sum() / ((x**2).sum() + 1e-18))


def variability_features(t: np.ndarray, clean: np.ndarray, slow: np.ndarray, fast: np.ndarray) -> dict:
    sig = robust_std(fast)
    feats = {}
    feats.update(periodogram_features(t, clean, 1 / 13.7, 4.0, "ls_slow_"))
    feats.update(periodogram_features(t, fast, 1.0, NYQUIST, "ls_fast_"))
    feats["ls_fast_near_nyquist"] = float(feats["ls_fast_freq1"] > 0.8 * NYQUIST)

    p_slow = 1 / feats["ls_slow_freq1"]
    lag = int(round(p_slow / (t[1] - t[0])))
    feats["acf_slow_period"] = acf_at(clean, lag)
    feats["acf_two_periods"] = acf_at(clean, 2 * lag)
    feats["acf_1day"] = acf_at(clean, 48)
    feats["acf_1cad_fast"] = acf_at(fast, 1)

    s = slow[~np.isnan(slow)]
    c = clean[~np.isnan(clean)]
    fz = (fast - np.nanmedian(fast)) / sig
    feats["slow_amp"] = float(np.percentile(s, 95) - np.percentile(s, 5)) if len(s) else 0.0
    feats["slow_amp_sig"] = feats["slow_amp"] / sig
    feats["clean_amp"] = float(np.percentile(c, 99) - np.percentile(c, 1))
    feats["clean_skew"] = float(stats.skew(c))
    feats["clean_kurt"] = float(stats.kurtosis(c))
    feats["fast_skew"] = float(stats.skew(fz[~np.isnan(fz)]))
    feats["fast_kurt"] = float(stats.kurtosis(fz[~np.isnan(fz)]))

    # Bright excursions: single points (cosmic rays) vs runs with a tail (flares)
    hi = np.nan_to_num(fz) > 4
    runs = np.diff(np.concatenate([[0], hi.astype(int), [0]]))
    starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]
    lengths = ends - starts
    tails = [np.nanmean(fz[e:e + 4]) for e in ends if e + 4 <= len(fz)]
    feats["n_hi_single"] = float((lengths == 1).sum())
    feats["n_hi_runs"] = float((lengths >= 2).sum())
    feats["hi_max_z"] = float(np.nanmax(fz))
    feats["hi_tail_z"] = float(np.nanmean(tails)) if tails else 0.0
    feats["n_lo4"] = float((np.nan_to_num(fz) < -4).sum())
    return feats


def glitch_features(clean: np.ndarray, sig: float, gap: np.ndarray) -> dict:
    """Steps, ramps and one-orbit anomalies on the clean curve."""
    x = clean
    w = STEP_WINDOW
    best = 0.0
    for i in range(w, len(x) - w, 4):
        a, b = x[i - w:i], x[i:i + w]
        if np.isnan(a).mean() > 0.5 or np.isnan(b).mean() > 0.5:
            continue
        best = max(best, abs(np.nanmedian(b) - np.nanmedian(a)))
    gap_idx = np.where(gap)[0]
    split = int(gap_idx.mean()) if len(gap_idx) else len(x) // 2
    o1, o2 = x[:split], x[split:]
    s1, s2 = robust_std(np.diff(o1)), robust_std(np.diff(o2))
    idx = np.arange(len(x))

    def slope(seg: np.ndarray, ix: np.ndarray) -> float:
        ok = ~np.isnan(seg)
        return float(np.polyfit(ix[ok], seg[ok], 1)[0]) if ok.sum() > 20 else 0.0

    return {
        "step_max_sig": best / sig,
        "orbit_noise_log_ratio": float(np.log(s1 / s2)),
        "orbit_level_diff_sig": float(abs(np.nanmedian(o1) - np.nanmedian(o2)) / sig),
        "orbit_slope_diff_sig": float(abs(slope(o1, idx[:split]) - slope(o2, idx[split:])) * len(x) / sig),
    }

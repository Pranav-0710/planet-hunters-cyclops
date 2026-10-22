"""Vectorized distribution statistics over many stars at once (n, t) -> columns."""

import numpy as np
import pandas as pd
from scipy import stats


def robust_sigma(f: np.ndarray) -> np.ndarray:
    return 1.4826 * np.nanmedian(np.abs(f - np.nanmedian(f, 1, keepdims=True)), 1)


def p2p_noise(f: np.ndarray) -> np.ndarray:
    """White-noise level from point-to-point differences: blind to variability slower than a few cadences."""
    return np.nanmedian(np.abs(np.diff(f, axis=1)), 1) / (0.6745 * np.sqrt(2)) + 1e-12


def variability_block(clean: np.ndarray, slow: np.ndarray, fast: np.ndarray) -> pd.DataFrame:
    """Amplitudes as multiples of the white noise, so a strongly variable star is not mistaken for a noisy one."""
    noise = p2p_noise(clean)
    amp = lambda x, lo, hi: np.nanpercentile(x, hi, 1) - np.nanpercentile(x, lo, 1)  # noqa: E731
    fz = (fast - np.nanmedian(fast, 1, keepdims=True)) / noise[:, None]
    return pd.DataFrame({
        "noise_p2p_log": np.log10(noise),
        "clean_std_over_noise": np.nanstd(clean, 1) / noise,
        "clean_amp90_over_noise": amp(clean, 5, 95) / noise,
        "slow_amp90_over_noise": amp(slow, 5, 95) / noise,
        "fast_std_over_noise": np.nanstd(fast, 1) / noise,
        "fast_amp90_over_noise": amp(fast, 5, 95) / noise,
        "n_hi_single_p2p5": np.nansum(fz > 5, 1),
        "max_hi_p2p": np.nanmax(fz, 1),
        "min_lo_p2p": np.nanmin(fz, 1),
    })


def stat_block(f: np.ndarray, prefix: str) -> pd.DataFrame:
    sig = robust_sigma(f) + 1e-12
    z = (f - np.nanmedian(f, 1, keepdims=True)) / sig[:, None]
    cols = {
        f"{prefix}std_sig": np.nanstd(f, 1) / sig,
        f"{prefix}sig": sig,
        f"{prefix}p2p_sig": np.nanmedian(np.abs(np.diff(f, axis=1)), 1) / sig,
        f"{prefix}min_sig": np.nanmin(z, 1),
        f"{prefix}max_sig": np.nanmax(z, 1),
        f"{prefix}skew": stats.skew(f, 1, nan_policy="omit"),
        f"{prefix}kurt": stats.kurtosis(f, 1, nan_policy="omit"),
        **{f"{prefix}z_p{q}": np.nanpercentile(z, q, 1) for q in (1, 5, 95, 99)},
        **{f"{prefix}n_lo{k}": np.nansum(z < -k, 1) for k in (3, 5)},
        **{f"{prefix}n_hi{k}": np.nansum(z > k, 1) for k in (3, 5)},
    }
    return pd.DataFrame(cols)

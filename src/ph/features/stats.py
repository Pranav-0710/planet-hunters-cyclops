"""Vectorized distribution statistics over many stars at once (n, t) -> columns."""

import numpy as np
import pandas as pd
from scipy import stats


def robust_sigma(f: np.ndarray) -> np.ndarray:
    return 1.4826 * np.nanmedian(np.abs(f - np.nanmedian(f, 1, keepdims=True)), 1)


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

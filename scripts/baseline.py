"""Baseline: starter statistics plus cheap extras (per-CCD trend removal, scatter, outliers, centroid spread).

Trains a gradient-boosting model on folds.csv, reports OOF macro-F1, writes oof/ and submissions/01_baseline.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ph.data import LABELS, ROOT, SEED, load_folds, load_split, relative_flux  # noqa: E402
from check_submission import check  # noqa: E402

META = ["tmag", "teff", "logg", "radius", "contamination"]


def robust_sigma(f: np.ndarray) -> np.ndarray:
    return 1.4826 * np.nanmedian(np.abs(f - np.nanmedian(f, 1, keepdims=True)), 1)


def remove_ccd_trend(f: np.ndarray, stars: pd.DataFrame) -> np.ndarray:
    """Subtract the median relative-flux curve of each camera/CCD (computed within this sector only)."""
    out = f.copy()
    for _, idx in stars.groupby(["camera", "ccd"]).indices.items():
        out[idx] -= np.nanmedian(f[idx], axis=0)
    return out


def stat_block(f: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    sig = robust_sigma(f)
    z = (f - np.nanmedian(f, 1, keepdims=True)) / sig[:, None]
    return {
        f"{prefix}std": np.nanstd(f, 1),
        f"{prefix}sig": sig,
        f"{prefix}p2p": np.nanmedian(np.abs(np.diff(f, axis=1)), 1),
        f"{prefix}min": np.nanmin(f, 1),
        f"{prefix}max": np.nanmax(f, 1),
        f"{prefix}skew": stats.skew(f, 1, nan_policy="omit"),
        f"{prefix}kurt": stats.kurtosis(f, 1, nan_policy="omit"),
        **{f"{prefix}p{q}": np.nanpercentile(f, q, 1) for q in (1, 5, 25, 75, 95, 99)},
        **{f"{prefix}n_lo{k}": np.nansum(z < -k, 1) for k in (3, 5)},
        **{f"{prefix}n_hi{k}": np.nansum(z > k, 1) for k in (3, 5)},
    }


def features(split_name: str) -> pd.DataFrame:
    s = load_split(split_name, mmap=False)
    f = remove_ccd_trend(relative_flux(s.flux), s.stars)
    gaps = np.isnan(f)
    slow = ndimage.median_filter(np.where(gaps, 0.0, f), size=(1, 49), mode="nearest")
    slow[gaps] = np.nan
    fast = f - slow

    cen = np.asarray(s.centroid, np.float32)
    cen = cen - np.nanmedian(cen, axis=2, keepdims=True)
    feats = {**stat_block(f, "raw_"), **stat_block(fast, "fast_")}
    feats["slow_std"] = np.nanstd(slow, 1)
    feats["fast_to_slow"] = feats["fast_std"] / (feats["slow_std"] + 1e-9)
    for c, name in ((0, "col"), (1, "row")):
        feats[f"cen_{name}_std"] = np.nanstd(cen[:, c], 1)
        feats[f"cen_{name}_maxabs"] = np.nanmax(np.abs(cen[:, c]), 1)
        valid = ~gaps & ~np.isnan(cen[:, c])
        a = np.where(valid, fast, 0.0)
        b = np.where(valid, cen[:, c], 0.0)
        a = a - a.sum(1, keepdims=True) / valid.sum(1, keepdims=True)
        b = b - b.sum(1, keepdims=True) / valid.sum(1, keepdims=True)
        a[~valid] = 0
        b[~valid] = 0
        feats[f"cen_{name}_corr"] = (a * b).sum(1) / (np.sqrt((a**2).sum(1) * (b**2).sum(1)) + 1e-12)
    X = pd.DataFrame(feats)
    return pd.concat([X, s.stars[META].reset_index(drop=True)], axis=1)


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train_stars.csv")
    test = pd.read_csv(ROOT / "data" / "test_stars.csv")
    X, X_test = features("train"), features("test")
    y = train.label.map({lab: i for i, lab in enumerate(LABELS)}).to_numpy()
    folds = load_folds().to_numpy()

    oof = np.zeros((len(X), len(LABELS)))
    test_proba = np.zeros((len(X_test), len(LABELS)))
    for k in range(5):
        tr, va = folds != k, folds == k
        model = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, class_weight="balanced", random_state=SEED
        )
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])
        test_proba += model.predict_proba(X_test) / 5
        print(f"fold {k}: macro-F1 {f1_score(y[va], oof[va].argmax(1), average='macro'):.4f}")

    print(f"OOF macro-F1: {f1_score(y, oof.argmax(1), average='macro'):.4f}")
    print(classification_report(y, oof.argmax(1), target_names=LABELS, digits=3))

    (ROOT / "oof").mkdir(exist_ok=True)
    (ROOT / "submissions").mkdir(exist_ok=True)
    pd.concat([train[["id"]], pd.DataFrame(oof, columns=LABELS)], axis=1).to_csv(ROOT / "oof/baseline_oof.csv", index=False)
    pd.concat([test[["id"]], pd.DataFrame(test_proba, columns=LABELS)], axis=1).to_csv(ROOT / "oof/baseline_test.csv", index=False)
    out = ROOT / "submissions/01_baseline.csv"
    pd.DataFrame({"id": test.id, "label": np.array(LABELS)[test_proba.argmax(1)]}).to_csv(out, index=False)
    check(out)


if __name__ == "__main__":
    main()

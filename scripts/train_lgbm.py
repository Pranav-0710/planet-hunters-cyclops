"""LightGBM on all features (issue #9). Needs build_detrended.py and build_features.py first.

Two stages: train on folds.csv, then down-weight training stars whose OOF probability for their own
label is very low (about 2% of labels are vetting mistakes) and train again. Writes oof/lgbm_{oof,test}.csv.
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ph.data import DATA, LABELS, ROOT, SEED, load_folds  # noqa: E402
from ph.features.stats import stat_block  # noqa: E402

CACHE = DATA / "cache"
META = ["tmag", "teff", "logg", "radius", "contamination"]
PARAMS = dict(
    objective="multiclass", num_class=len(LABELS), learning_rate=0.03, num_leaves=63,
    min_child_samples=20, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, class_weight="balanced", n_estimators=3000, random_state=SEED, verbose=-1,
)
NOISY_PROBA = 0.05
NOISY_WEIGHT = 0.2


def load_features(name: str) -> pd.DataFrame:
    stars = pd.read_csv(DATA / f"{name}_stars.csv")
    clean = np.load(CACHE / f"{name}_clean.npy")
    fast = np.load(CACHE / f"{name}_fast.npy")
    star = pd.read_csv(CACHE / f"{name}_feat_star.csv")
    blocks = pd.concat([stat_block(clean, "clean_"), stat_block(fast, "fast_")], axis=1)
    star = star.drop(columns=[c for c in star.columns if c in blocks.columns])
    return pd.concat([blocks, star, stars[META]], axis=1)


def cv(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, folds: np.ndarray, w: np.ndarray):
    oof = np.zeros((len(X), len(LABELS)))
    test = np.zeros((len(X_test), len(LABELS)))
    imp = np.zeros(X.shape[1])
    for k in range(5):
        tr, va = folds != k, folds == k
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(
            X[tr], y[tr], sample_weight=w[tr], eval_set=[(X[va], y[va])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[va] = model.predict_proba(X[va])
        test += model.predict_proba(X_test) / 5
        imp += model.booster_.feature_importance("gain")
        print(f"  fold {k}: {f1_score(y[va], oof[va].argmax(1), average='macro'):.4f} ({model.best_iteration_} trees)")
    return oof, test, pd.Series(imp, X.columns).sort_values(ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lgbm")
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train_stars.csv")
    test = pd.read_csv(DATA / "test_stars.csv")
    X, X_test = load_features("train"), load_features("test")
    y = train.label.map({lab: i for i, lab in enumerate(LABELS)}).to_numpy()
    folds = load_folds().to_numpy()
    print(f"{X.shape[1]} features")

    print("stage 1")
    oof, _, _ = cv(X, y, X_test, folds, np.ones(len(y)))
    print(f"stage 1 OOF macro-F1: {f1_score(y, oof.argmax(1), average='macro'):.4f}")

    own = oof[np.arange(len(y)), y]
    w = np.where(own < NOISY_PROBA, NOISY_WEIGHT, 1.0)
    print(f"stage 2: down-weighting {(w < 1).sum()} suspected label errors ({(w < 1).mean():.1%})")
    oof, test_proba, imp = cv(X, y, X_test, folds, w)
    print(f"stage 2 OOF macro-F1: {f1_score(y, oof.argmax(1), average='macro'):.4f}")
    print(classification_report(y, oof.argmax(1), target_names=LABELS, digits=3))
    print("top features:\n" + imp.head(25).round(0).to_string())

    drift = pd.DataFrame({
        "train_share": pd.Series(y).value_counts(normalize=True).sort_index().to_numpy(),
        "test_pred_share": np.bincount(test_proba.argmax(1), minlength=len(LABELS)) / len(test_proba),
    }, index=LABELS)
    print("class mix, train labels vs test predictions:\n" + drift.round(3).to_string())

    (ROOT / "oof").mkdir(exist_ok=True)
    pd.concat([train[["id"]], pd.DataFrame(oof, columns=LABELS)], axis=1).to_csv(ROOT / f"oof/{args.name}_oof.csv", index=False)
    pd.concat([test[["id"]], pd.DataFrame(test_proba, columns=LABELS)], axis=1).to_csv(ROOT / f"oof/{args.name}_test.csv", index=False)


if __name__ == "__main__":
    main()

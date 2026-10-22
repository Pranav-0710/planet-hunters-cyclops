"""Sector-shift check: how well can a model tell train stars from test stars? (issue #9)

AUC near 0.5 = features look the same in both sectors. Prints the AUC and the features driving it.
Usage: python scripts/adversarial.py
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ph.data import SEED  # noqa: E402
from train_lgbm import load_features  # noqa: E402


def main() -> None:
    X, Xt = load_features("train"), load_features("test")
    A = pd.concat([X, Xt], ignore_index=True)
    s = np.r_[np.zeros(len(X)), np.ones(len(Xt))]
    oof = np.zeros(len(s))
    imp = np.zeros(A.shape[1])
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(A, s):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=8, verbose=-1)
        m.fit(A.iloc[tr], s[tr])
        oof[va] = m.predict_proba(A.iloc[va])[:, 1]
        imp += m.booster_.feature_importance("gain")
    print(f"adversarial AUC: {roc_auc_score(s, oof):.3f}")
    print((pd.Series(imp, A.columns) / imp.sum()).sort_values(ascending=False).head(15).round(3).to_string())
    ks = pd.Series({c: ks_2samp(X[c].dropna(), Xt[c].dropna()).statistic for c in X.columns})
    print("largest KS drift:\n" + ks.sort_values(ascending=False).head(15).round(3).to_string())


if __name__ == "__main__":
    main()

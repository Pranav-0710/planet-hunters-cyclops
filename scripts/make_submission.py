"""Blend model probabilities from oof/ and write a submission (issue #11).

Usage: python scripts/make_submission.py --models lgbm cnn --weights 0.6 0.4 --out 03_lgbm_cnn [--tune]

--tune adds per-class log-probability offsets fitted on OOF to maximize macro-F1.
Reproducible: same oof/ files + same arguments -> same CSV. The commit hash is printed for the log.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ph.data import DATA, LABELS, ROOT  # noqa: E402
from check_submission import check  # noqa: E402


def blend(models: list[str], weights: list[float], part: str) -> np.ndarray:
    logp = sum(w * np.log(pd.read_csv(ROOT / f"oof/{m}_{part}.csv")[LABELS].to_numpy() + 1e-9)
               for m, w in zip(models, weights))
    return logp / sum(weights)


def tune_offsets(logp: np.ndarray, y: np.ndarray, rounds: int = 3) -> np.ndarray:
    off = np.zeros(len(LABELS))
    grid = np.linspace(-1.5, 1.5, 31)
    for _ in range(rounds):
        for c in range(len(LABELS)):
            scores = []
            for g in grid:
                trial = off.copy()
                trial[c] = g
                scores.append(f1_score(y, (logp + trial).argmax(1), average="macro"))
            off[c] = grid[int(np.argmax(scores))]
    return off


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()
    weights = args.weights or [1.0] * len(args.models)

    train = pd.read_csv(DATA / "train_stars.csv")
    test = pd.read_csv(DATA / "test_stars.csv")
    y = train.label.map({lab: i for i, lab in enumerate(LABELS)}).to_numpy()
    oof, tst = blend(args.models, weights, "oof"), blend(args.models, weights, "test")
    print(f"blend OOF macro-F1: {f1_score(y, oof.argmax(1), average='macro'):.4f}")

    off = np.zeros(len(LABELS))
    if args.tune:
        off = tune_offsets(oof, y)
        print(f"tuned OOF macro-F1: {f1_score(y, (oof + off).argmax(1), average='macro'):.4f} (optimistic: fitted on OOF)")
        print("offsets: " + ", ".join(f"{lab}={o:+.2f}" for lab, o in zip(LABELS, off)))

    out = ROOT / "submissions" / f"{args.out}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"id": test.id, "label": np.array(LABELS)[(tst + off).argmax(1)]}).to_csv(out, index=False)
    check(out)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {commit}; args: {' '.join(sys.argv[1:])}")


if __name__ == "__main__":
    main()

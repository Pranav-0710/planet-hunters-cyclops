"""Check a submission CSV against the upload rules. Usage: python scripts/check_submission.py submissions/01_x.csv"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ph.data import DATA, LABELS  # noqa: E402


def check(path: Path) -> None:
    sub = pd.read_csv(path)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    assert list(sub.columns) == ["id", "label"], f"columns are {list(sub.columns)}"
    assert len(sub) == len(sample), f"{len(sub)} rows, expected {len(sample)}"
    assert sub.id.is_unique, "duplicate ids"
    assert set(sub.id) == set(sample.id), "ids differ from sample_submission"
    bad = set(sub.label) - set(LABELS)
    assert not bad, f"unknown labels {bad}"
    print(f"OK {path} ({len(sub)} rows)")
    print(sub.label.value_counts().to_string())


if __name__ == "__main__":
    check(Path(sys.argv[1]))

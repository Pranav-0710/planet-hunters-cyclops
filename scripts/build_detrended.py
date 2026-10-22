"""Build data/cache/{train,test}_{clean,slow,fast,cen}.npy (issue #3).

Usage: python scripts/build_detrended.py [--union]

--union masks both sectors with the union of their bad cadences (same cadence index grid), so they share
one observing window.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ph.data import DATA, load_split  # noqa: E402
from ph.detrend import apply_mask, detrend_split  # noqa: E402

CACHE = DATA / "cache"
CACHE.mkdir(exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--union", action="store_true")
args = ap.parse_args()

outs = {}
for name in ("train", "test"):
    t0 = time.time()
    outs[name] = detrend_split(load_split(name, mmap=False))
    print(f"{name}: {len(outs[name]['clean'])} stars in {time.time() - t0:.0f}s; "
          f"masked cadences besides the gap: {np.where(outs[name]['noisy'])[0].tolist()}")

union = outs["train"]["noisy"] | outs["test"]["noisy"]
if args.union:
    print(f"union mask: {union.sum()} cadences")
for name, out in outs.items():
    if args.union:
        out = apply_mask(out, union)
    for key in ("clean", "slow", "fast", "cen"):
        np.save(CACHE / f"{name}_{key}.npy", out[key])

"""Per-star features from the detrended cache, in parallel. Writes data/cache/{split}_feat_star.csv.

Usage: python scripts/build_features.py [--limit N]   (needs scripts/build_detrended.py first)
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ph.data import DATA, load_split  # noqa: E402
from ph.features.centroid import centroid_features  # noqa: E402
from ph.features.frequency import glitch_features, variability_features  # noqa: E402
from ph.features.transit import transit_features  # noqa: E402

CACHE = DATA / "cache"
CHUNK = 250


def _chunk(args: tuple[str, int, int]) -> list[dict]:
    name, start, stop = args
    fast = np.load(CACHE / f"{name}_fast.npy", mmap_mode="r")
    clean = np.load(CACHE / f"{name}_clean.npy", mmap_mode="r")
    slow = np.load(CACHE / f"{name}_slow.npy", mmap_mode="r")
    cen = np.load(CACHE / f"{name}_cen.npy", mmap_mode="r")
    cad = pd.read_csv(DATA / f"{name}_cadences.csv")
    t = cad.time_btjd.to_numpy()
    q = cad.quality.to_numpy()
    dump_times = t[(q & 1) > 0]
    gap = (q & 4) > 0
    rows = []
    for i in range(start, stop):
        f = np.asarray(fast[i], np.float64)
        c = np.asarray(clean[i], np.float64)
        try:
            tr, ephem = transit_features(t, f, dump_times)
            ce = centroid_features(t, f, np.asarray(cen[i], np.float64), ephem)
            va = variability_features(t, c, np.asarray(slow[i], np.float64), f)
            gl = glitch_features(c, tr["fast_sig"], gap)
            rows.append({**tr, **ce, **va, **gl})
        except Exception as exc:  # one bad star must not kill the run; it gets NaN features
            print(f"{name}[{i}] failed: {exc}", flush=True)
            rows.append({})
    return rows


def build(name: str, limit: int | None) -> None:
    n = len(load_split(name).stars) if limit is None else limit
    jobs = [(name, s, min(s + CHUNK, n)) for s in range(0, n, CHUNK)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = [r for chunk in pool.map(_chunk, jobs) for r in chunk]
    out = pd.DataFrame(rows)
    out.to_csv(CACHE / f"{name}_feat_star.csv", index=False)
    print(f"{name}: {len(out)} stars, {out.shape[1]} features in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    for split in ("train", "test"):
        build(split, args.limit)

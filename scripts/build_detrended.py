"""Build data/cache/{train,test}_{clean,slow,fast,cen}.npy (issue #3). Usage: python scripts/build_detrended.py"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ph.data import DATA, load_split  # noqa: E402
from ph.detrend import detrend_split  # noqa: E402

CACHE = DATA / "cache"
CACHE.mkdir(exist_ok=True)

for name in ("train", "test"):
    t0 = time.time()
    out = detrend_split(load_split(name, mmap=False))
    for key, arr in out.items():
        np.save(CACHE / f"{name}_{key}.npy", arr)
    print(f"{name}: {len(out['clean'])} stars in {time.time() - t0:.0f}s")

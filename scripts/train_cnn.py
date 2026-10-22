"""1D CNN on detrended light curves (issue #10). Runs on CPU; needs build_detrended.py and build_features.py.

Two views per star:
  global: clean, fast, centroid col/row (high-passed), valid mask; each standardized by the star's noise
  folded: fast and centroids folded on the BLS period into 128 phase bins, dip centered
plus star properties. Same 5 folds as LightGBM. Writes oof/cnn_{oof,test}.csv.

Usage: python scripts/train_cnn.py [--epochs 12] [--folds 0 1 2 3 4]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ph.data import DATA, LABELS, ROOT, SEED, load_folds  # noqa: E402
from ph.detrend import running_median  # noqa: E402

CACHE = DATA / "cache"
META = ["tmag", "teff", "logg", "radius", "contamination"]
N_BINS = 128
CLIP = 20.0
GLOBAL_POOL = 2


def _robust_sig(x: np.ndarray) -> np.ndarray:
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, -1, keepdims=True)), -1, keepdims=True) + 1e-9


def fold_view(t: np.ndarray, series: np.ndarray, period: np.ndarray) -> np.ndarray:
    """Fold (n, c, t) series on each star's period into N_BINS bins, rolled so the deepest flux bin is centered."""
    n, c, _ = series.shape
    out = np.zeros((n, c, N_BINS), np.float32)
    for i in range(n):
        b = ((t - t[0]) / period[i] % 1.0 * N_BINS).astype(int)
        for k in range(c):
            x = series[i, k]
            ok = ~np.isnan(x)
            s = np.bincount(b[ok], x[ok], N_BINS)
            m = np.bincount(b[ok], minlength=N_BINS)
            out[i, k] = np.where(m > 0, s / np.maximum(m, 1), 0.0)
        out[i] = np.roll(out[i], N_BINS // 2 - int(np.argmin(out[i, 0])), axis=-1)
    return out


def build_inputs(name: str) -> dict[str, np.ndarray]:
    path = CACHE / f"{name}_cnn.npz"
    if path.exists():
        return dict(np.load(path))
    t = pd.read_csv(DATA / f"{name}_cadences.csv").time_btjd.to_numpy()
    clean = np.load(CACHE / f"{name}_clean.npy").astype(np.float64)
    fast = np.load(CACHE / f"{name}_fast.npy").astype(np.float64)
    cen = np.load(CACHE / f"{name}_cen.npy").astype(np.float64)
    cen_hp = np.stack([cen[:, k] - running_median(cen[:, k]) for k in range(2)], axis=1)
    sig_f = _robust_sig(fast)
    sig_c = _robust_sig(cen_hp)
    glob = np.stack([clean / sig_f, fast / sig_f, cen_hp[:, 0] / sig_c[:, 0], cen_hp[:, 1] / sig_c[:, 1]], axis=1)
    valid = ~np.isnan(fast)
    period = pd.read_csv(CACHE / f"{name}_feat_star.csv").bls_period.fillna(1.0).to_numpy()
    folded = fold_view(t, glob[:, 1:], period)
    glob = np.concatenate([np.nan_to_num(np.clip(glob, -CLIP, CLIP)), valid[:, None].astype(float)], axis=1)
    stars = pd.read_csv(DATA / f"{name}_stars.csv")
    meta = np.column_stack([stars[META].to_numpy(), np.log10(sig_f[:, 0]), np.log10(period)])
    out = {"glob": glob.astype(np.float32), "fold": np.clip(folded, -CLIP, CLIP), "meta": meta.astype(np.float32)}
    np.savez(path, **out)
    return out


def block(cin: int, cout: int, k: int = 7) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.GELU(),
                         nn.Conv1d(cout, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.GELU(), nn.MaxPool1d(2))


class Net(nn.Module):
    """Sized for a laptop CPU: the global view is averaged into 1-hour bins; the folded view keeps full detail."""

    def __init__(self, n_meta: int):
        super().__init__()
        self.glob = nn.Sequential(block(5, 16, 5), block(16, 32, 5), block(32, 48, 5), block(48, 64, 5))
        self.fold = nn.Sequential(block(3, 16, 5), block(16, 32, 5), block(32, 64, 5))
        self.head = nn.Sequential(nn.Linear(2 * 64 + 2 * 64 + n_meta, 128), nn.GELU(), nn.Dropout(0.3),
                                  nn.Linear(128, len(LABELS)))

    def forward(self, g: torch.Tensor, f: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        a = self.glob(nn.functional.avg_pool1d(g, GLOBAL_POOL))
        b = self.fold(f)
        z = torch.cat([a.mean(-1), a.amax(-1), b.mean(-1), b.amax(-1), m], dim=1)
        return self.head(z)


def augment(g: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Random circular time shift and a few extra masked cadence triplets (fake dumps)."""
    g = torch.roll(g, int(rng.integers(-60, 61)), dims=-1).clone()
    for c0 in rng.integers(0, g.shape[-1] - 3, size=int(rng.integers(0, 10))):
        g[:, :, c0:c0 + 3] = 0.0
    return g


def predict(model: Net, g: np.ndarray, f: np.ndarray, m: np.ndarray, bs: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(g), bs):
            out.append(torch.softmax(model(torch.from_numpy(g[s:s + bs]), torch.from_numpy(f[s:s + bs]),
                                           torch.from_numpy(m[s:s + bs])), 1).numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--name", default="cnn")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    torch.set_num_threads(16)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tr, te = build_inputs("train"), build_inputs("test")
    mu, sd = tr["meta"].mean(0), tr["meta"].std(0) + 1e-6
    m_tr = ((tr["meta"] - mu) / sd).astype(np.float32)
    m_te = ((te["meta"] - mu) / sd).astype(np.float32)
    train = pd.read_csv(DATA / "train_stars.csv")
    test = pd.read_csv(DATA / "test_stars.csv")
    y = train.label.map({lab: i for i, lab in enumerate(LABELS)}).to_numpy()
    folds = load_folds().to_numpy()
    weights = torch.tensor(len(y) / (len(LABELS) * np.bincount(y)), dtype=torch.float32)

    oof = np.zeros((len(y), len(LABELS)))
    test_proba = np.zeros((len(te["glob"]), len(LABELS)))
    for k in args.folds:
        t0 = time.time()
        idx_tr, idx_va = np.where(folds != k)[0], np.where(folds == k)[0]
        model = Net(m_tr.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
        steps = args.epochs * int(np.ceil(len(idx_tr) / 128))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
        loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
        for ep in range(args.epochs):
            model.train()
            perm = rng.permutation(idx_tr)
            for s in range(0, len(perm), 128):
                b = perm[s:s + 128]
                g = augment(torch.from_numpy(tr["glob"][b]), rng)
                loss = loss_fn(model(g, torch.from_numpy(tr["fold"][b]), torch.from_numpy(m_tr[b])),
                               torch.from_numpy(y[b]))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            if ep % 3 == 2 or ep == args.epochs - 1:
                p = predict(model, tr["glob"][idx_va], tr["fold"][idx_va], m_tr[idx_va])
                print(f"  fold {k} epoch {ep + 1}: val macro-F1 {f1_score(y[idx_va], p.argmax(1), average='macro'):.4f}",
                      flush=True)
        oof[idx_va] = predict(model, tr["glob"][idx_va], tr["fold"][idx_va], m_tr[idx_va])
        test_proba += predict(model, te["glob"], te["fold"], m_te) / len(args.folds)
        print(f"fold {k} done in {time.time() - t0:.0f}s", flush=True)

    done = np.isin(folds, args.folds)
    print(f"CNN OOF macro-F1 on folds {args.folds}: {f1_score(y[done], oof[done].argmax(1), average='macro'):.4f}")
    if len(args.folds) == 5:
        pd.concat([train[["id"]], pd.DataFrame(oof, columns=LABELS)], axis=1).to_csv(ROOT / f"oof/{args.name}_oof.csv", index=False)
        pd.concat([test[["id"]], pd.DataFrame(test_proba, columns=LABELS)], axis=1).to_csv(ROOT / f"oof/{args.name}_test.csv", index=False)


if __name__ == "__main__":
    main()

"""python compare.py BASE.npz VARIANT.npz: paired NLL delta (bootstrap over whole windows, 10k, seed 173), top-1 agreement."""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
assert (a["target"] == b["target"]).all()
d = b["nll"].mean(1) - a["nll"].mean(1)  # per window
rng = np.random.default_rng(173)
boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
lo, hi = np.percentile(boots, [2.5, 97.5])
agree = (a["argmax"] == b["argmax"]).mean()
print(f"NLL {a['nll'].mean():.4f} -> {b['nll'].mean():.4f}  delta {d.mean():+.4f} CI [{lo:+.4f}, {hi:+.4f}]  "
      f"top1-agree {agree:.4f}  top1-text {(a['argmax'] == a['target']).mean():.4f} -> {(b['argmax'] == b['target']).mean():.4f}")
for s in np.unique(a["sources"]):
    m = a["sources"] == s
    print(f"  {s}: delta {d[m].mean():+.4f}, agree {(a['argmax'][m] == b['argmax'][m]).mean():.4f}")
if "kl" in b:
    k = b["kl"].mean(1)
    lo, hi = np.percentile(k[rng.integers(0, len(k), (10000, len(k)))].mean(1), [2.5, 97.5])
    print(f"  KL(ref||variant) {k.mean():.4f} CI [{lo:.4f}, {hi:.4f}]  median-token {np.median(b['kl']):.4f}")

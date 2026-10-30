"""Paired base-vs-tuned comparison of two eval_code score files (same items): accuracy, paired bootstrap 95% CI of the
difference, exact McNemar p-value. python compare.py base.scores.json tuned.scores.json"""
import json, random, sys
from math import comb

a, b = (json.load(open(p)) for p in sys.argv[1:3])
ids = sorted(set(a) & set(b))
for k in ("base", "plus"):
    x, y = [a[i][k] for i in ids], [b[i][k] for i in ids]
    d = [int(q) - int(p) for p, q in zip(x, y)]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(10000))
    up, down = d.count(1), d.count(-1)
    n = up + down
    p = min(1.0, 2 * sum(comb(n, j) for j in range(min(up, down) + 1)) / 2 ** n) if n else 1.0
    print(json.dumps({"metric": k, "n": len(ids), "base": sum(x) / len(ids), "tuned": sum(y) / len(ids),
                      "diff": sum(d) / len(d), "ci95": [boots[250], boots[9749]], "tuned_wins": up, "tuned_losses": down, "mcnemar_p": round(p, 4)}))

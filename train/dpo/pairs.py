"""Preference pairs from scored rollouts (correctness first, then length).

  python3 pairs.py <rollouts.scored.jsonl> <pairs.jsonl> [--max-per-problem 2] [--long-ratio 1.5]

Per problem: chosen = shortest passing, uncapped rollout. Rejected, in order of preference:
  1. failing rollouts (capped ones first, then longest)            weight 1.0
  2. passing but capped, or >= long-ratio x the chosen's length    weight 0.5
Problems where nothing passes, or every pass is about as long as the others, give no pair.
"""
import json
import sys
from collections import defaultdict


def make_pairs(rows, max_pp=2, long_ratio=1.5):
    by = defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)
    out = []
    for pid, rs in by.items():
        ok = sorted((r for r in rs if r["passed"] and not r["capped"]), key=lambda r: r["tokens"])
        if not ok:
            continue
        ch = ok[0]
        bad = sorted((r for r in rs if not r["passed"]), key=lambda r: (not r["capped"], -r["tokens"]))
        long = sorted((r for r in rs if r is not ch and r["passed"] and (r["capped"] or r["tokens"] >= long_ratio * ch["tokens"])),
                      key=lambda r: -r["tokens"])
        rej = [(r, 1.0) for r in bad] + [(r, 0.5) for r in long]
        for r, w in rej[:max_pp]:
            out.append({"id": pid, "prompt_ids": ch["prompt_ids"], "chosen_ids": ch["out_ids"], "rejected_ids": r["out_ids"],
                        "chosen_tokens": ch["tokens"], "rejected_tokens": r["tokens"], "rejected_capped": r["capped"], "rejected_kind": "fail" if w == 1.0 else "long",
                        "weight": w})
    return out


if __name__ == "__main__":
    a = sys.argv
    opt = dict(zip(a[3::2], a[4::2]))
    rows = [json.loads(l) for l in open(a[1])]
    ps = make_pairs(rows, int(opt.get("--max-per-problem", 2)), float(opt.get("--long-ratio", 1.5)))
    with open(a[2], "w") as f:
        for p in ps:
            f.write(json.dumps(p) + "\n")
    n_fail = sum(p["rejected_kind"] == "fail" for p in ps)
    print(f"rollouts {len(rows)} problems {len({r['id'] for r in rows})} pass@1 {sum(r['passed'] for r in rows) / max(len(rows), 1):.3f} "
          f"pairs {len(ps)} (fail {n_fail}, long {len(ps) - n_fail}) problems with pairs {len({p['id'] for p in ps})}")

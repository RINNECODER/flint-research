"""python nextmask.py PARENT_KEEP.json PARENT_STATS.npz N OUT.json: keep the top-N experts by REAP (mean g*||y|| over routed tokens)
of the parent variant, as ids of the original 256."""
import json, sys
import numpy as np
parent = json.load(open(sys.argv[1]))["keep_ids"]; s = np.load(sys.argv[2]); n = int(sys.argv[3])
reap = s["sum_reap"] / np.maximum(s["count"], 1)
keep = [sorted(np.array(parent[l])[np.argsort(-reap[l])[:n]].tolist()) for l in range(47)]
assert all(len(k) == n and set(k) <= set(p) for k, p in zip(keep, parent))
json.dump({"keep_ids": keep, "method": f"REAP top-{n} of {sys.argv[1]} by {sys.argv[2]}"}, open(sys.argv[4], "w"))

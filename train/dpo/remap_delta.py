"""Remap an int4 expert delta (original 256-expert ids) onto a REAP-pruned model, dropping experts that were pruned.
  python3 remap_delta.py <delta.safetensors> <keep_ids.json> <out.safetensors>
keep_ids[layer-2][j] = original id of kept expert j. Output feeds apply_delta.py <pruned folder> <out> <new folder>."""
import json, sys
import torch
from safetensors.torch import load_file, save_file


def remap(dl, keep):
    out = {}
    for key in [k for k in dl if k.endswith(".experts")]:
        layer = int(key.split(".")[0])
        pos = {e: j for j, e in enumerate(keep[layer - 2])}
        rows = [(r, pos[e]) for r, e in enumerate(dl[key].tolist()) if e in pos]
        if not rows:
            continue
        idx = torch.tensor([r for r, _ in rows])
        out[key] = torch.tensor([j for _, j in rows], dtype=torch.int32)
        for k in dl:
            if k.startswith(f"{layer}.") and not k.endswith(".experts"):
                out[k] = dl[k][idx].contiguous()
    return out


if __name__ == "__main__":
    dl = load_file(sys.argv[1])
    out = remap(dl, json.load(open(sys.argv[2]))["keep_ids"])
    save_file(out, sys.argv[3])
    print("experts kept", sum(len(v) for k, v in out.items() if k.endswith(".experts")), "of", sum(len(v) for k, v in dl.items() if k.endswith(".experts")))

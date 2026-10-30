"""Write a full model folder: the base int4 checkpoint with the ESFT delta's experts patched in, byte for byte in the
same layout (headers, shapes, dtypes and offsets unchanged; only the trained experts' weight/scales/biases slices).

  python apply_delta.py <base folder> <esft_delta.safetensors> <out folder>
"""
import hashlib
import json
import os
import shutil
import struct
import sys

import torch
from safetensors.torch import load_file


def main(base, delta_path, out):
    os.makedirs(out, exist_ok=True)
    delta = load_file(delta_path)
    patched = 0
    for name in sorted(os.listdir(base)):
        src, dst = os.path.join(base, name), os.path.join(out, name)
        shutil.copyfile(src, dst)  # follows HF-cache symlinks
        if not name.endswith(".safetensors"):
            continue
        with open(dst, "r+b") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            for key, t in header.items():
                if ".mlp.switch_mlp." not in key:
                    continue
                layer = key.split(".layers.")[1].split(".")[0]
                proj_kind = key.split(".switch_mlp.")[1]  # e.g. gate_proj.weight
                if f"{layer}.experts" not in delta:
                    continue
                b, e = t["data_offsets"]
                per = (e - b) // t["shape"][0]
                new = delta[f"{layer}.{proj_kind}"].to({"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "U8": torch.uint8}[t["dtype"]])
                assert tuple(new.shape[1:]) == tuple(t["shape"][1:]) and new[0].numel() * new.element_size() == per, (key, new.shape, t)
                for row, ex in zip(new, delta[f"{layer}.experts"].tolist()):
                    f.seek(8 + n + b + ex * per)
                    f.write(row.contiguous().view(torch.uint8).numpy().tobytes())
                    patched += 1
    meta = [f for f in os.listdir(out) if f.endswith("-int4.json")]
    for mf in meta:  # keep the checkpoint manifest's hashes true
        p = os.path.join(out, mf)
        m = json.load(open(p))
        for f in m.get("checkpoint_files", {}):
            h = hashlib.sha256()
            with open(os.path.join(out, f), "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            m["checkpoint_files"][f] = h.hexdigest()
        json.dump(m, open(p, "w"), indent=1)
    print("patched expert slices", patched)


if __name__ == "__main__":
    main(*sys.argv[1:])

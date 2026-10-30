"""Slice Flint's routed experts: python prune.py SRC_FOLDER KEEP_JSON OUT_FOLDER

KEEP_JSON: {"keep_ids": [[ids for MoE layer 2], ..., [ids for layer 48]]}, ids index the ORIGINAL 256 experts.
Slices dim 0 of every mlp.switch_mlp.* tensor and the router mlp.gate.weight rows (no router bias in this build),
writes raw bytes (no dequant), sets num_experts/num_local_experts in config.json."""
import json
import shutil
import struct
import sys
from pathlib import Path

src, keep_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
keep = json.loads(keep_path.read_text())["keep_ids"]
assert len(keep) == 47 and len({len(k) for k in keep}) == 1, "47 MoE layers, same count each"
n_keep = len(keep[0])
for k in keep:
    assert k == sorted(set(k)) and 0 <= k[0] and k[-1] < 256
out.mkdir(parents=True, exist_ok=True)

for f in sorted(src.glob("model*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        base = 8 + n
        new_header, chunks, off = {}, [], 0
        if "__metadata__" in header:
            new_header["__metadata__"] = header["__metadata__"]
        for name, t in sorted(((k, v) for k, v in header.items() if k != "__metadata__"), key=lambda kv: kv[1]["data_offsets"][0]):
            b, e = t["data_offsets"]
            fh.seek(base + b)
            data = fh.read(e - b)
            shape = list(t["shape"])
            if ".layers." in name and (".mlp.switch_mlp." in name or name.endswith(".mlp.gate.weight")):
                layer = int(name.split(".layers.")[1].split(".")[0])
                assert layer >= 2 and shape[0] == 256, (name, shape)
                row = len(data) // 256
                data = b"".join(data[i * row:(i + 1) * row] for i in keep[layer - 2])
                shape[0] = n_keep
            new_header[name] = {"dtype": t["dtype"], "shape": shape, "data_offsets": [off, off + len(data)]}
            chunks.append(data)
            off += len(data)
    hb = json.dumps(new_header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    tmp = out / (f.name + ".tmp")
    with open(tmp, "wb") as w:
        w.write(struct.pack("<Q", len(hb)))
        w.write(hb)
        for c in chunks:
            w.write(c)
    tmp.rename(out / f.name)
    del chunks
    print("wrote", f.name, flush=True)

for p in src.iterdir():
    if p.suffix != ".safetensors" and p.name not in ("config.json", "flint-abliterated-int4.json") and p.is_file():
        shutil.copy(p.resolve(), out / p.name)
cfg = json.loads((src / "config.json").read_text())
cfg["num_experts"] = cfg["num_local_experts"] = n_keep
cfg["pruned_from"] = {"source": str(src), "keep_ids_file": str(keep_path), "kept_per_layer": n_keep}
(out / "config.json").write_text(json.dumps(cfg, indent=2))
shutil.copy(keep_path, out / "keep_ids.json")
print("done", out)

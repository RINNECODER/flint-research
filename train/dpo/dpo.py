"""DPO on Flint's top coding experts (the ESFT expert set), reference log-probs precomputed so one model sits in VRAM.

  ref    <model> <delta|none> <pairs.jsonl> <ref.jsonl> [--max-len 3072]
  train  <model> <delta|none> <stats.json> <pairs.jsonl> <ref.jsonl> <outdir> [--beta 0.1] [--lr 5e-7] [--accum 8]
         [--epochs 2] [--rpo 0.0] [--minutes 300] [--max-len 3072] [--sync /root/ckpt_sync] [--resume 1]
  bench  <model> <stats.json> <seq len>            one DPO step on two random sequences: time + peak VRAM
  (train ends by writing outdir/esft_delta.safetensors, the int4 delta that apply_delta.py / eval_code.py take)

<delta> is the starting model's own int4 expert delta (e.g. an SFT run), applied before anything else; the DPO export
holds only the DPO-trained experts, so the shipped model = base + start delta + DPO delta (apply in that order).
Loss per pair: -w * logsigmoid(beta * ((pc - rc) - (pr - rr)))  [+ rpo * chosen NLL per token]. Sums over reply tokens.
Replies longer than max-len (prompt + reply) are truncated from the end, as TRL does.
"""
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/ft")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import esft
import flint_torch as ft
from rollout import apply_delta

EOT = 150001


def seqs(p, max_len):
    """(ids, mask) for chosen and rejected; None if the prompt alone leaves under 256 reply tokens."""
    pr = p["prompt_ids"]
    if len(pr) > max_len - 256:
        return None
    out = []
    for key, capped in (("chosen_ids", False), ("rejected_ids", p.get("rejected_capped", False))):
        r = (p[key] + ([] if capped else [EOT]))[: max_len - len(pr)]
        out.append((pr + r, [0] * len(pr) + [1] * len(r)))
    return out


def logp(m, ids, mask):
    s, k = m.loss(torch.tensor([ids]), mask=torch.tensor([mask]))
    return -s, k


def ref(a):
    model_dir, delta, pairs, out = a[:4]
    opt = dict(zip(a[4::2], a[5::2]))
    max_len = int(opt.get("--max-len", 3072))
    m = ft.load(model_dir, esft.DEV).eval()
    if delta != "none":
        apply_delta(m, delta)
    ps = [json.loads(l) for l in open(pairs)]
    done = sum(1 for _ in open(out)) if os.path.exists(out) else 0
    t0 = time.time()
    with open(out, "a") as f, torch.no_grad():
        for i in range(done, len(ps)):
            s = seqs(ps[i], max_len)
            rc, rr = (logp(m, *x)[0].item() for x in s) if s else (None, None)
            f.write(json.dumps({"i": i, "rc": rc, "rr": rr}) + "\n")
            f.flush()
            if i % 50 == 0:
                esft.log(f"ref {i}/{len(ps)} {(time.time() - t0) / 60:.1f} min")
    esft.log("ref done", len(ps))


def delta_int4(m, path):
    """Non-destructive int4 export of the trained experts (same format as esft.export)."""
    from safetensors.torch import save_file
    delta = {}
    for i, moe in esft.moes(m):
        if not len(moe.esft):
            continue
        es = sorted({int(k.split("_")[0]) for k in moe.esft.keys()})
        delta[f"{i}.experts"] = torch.tensor(es, dtype=torch.int32)
        for n in ("gate", "up", "down"):
            q = [esft.quantize(moe.esft[f"{e}_{n}"].detach()) for e in es]
            for j, suf in enumerate(("weight", "scales", "biases")):
                delta[f"{i}.{n}_proj.{suf}"] = torch.stack([x[j].cpu() for x in q])
    save_file(delta, path + ".tmp")
    os.replace(path + ".tmp", path)


def save_state(m, opt, step, cursor, hist, outdir, sync):
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items() if ".esft." in k}
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    st = {"params": sd, "opt": {n: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in opt.state[p].items()}
                                for n, p in m.named_parameters() if p.requires_grad and p in opt.state},
          "step": step, "cursor": cursor, "names": names}
    torch.save(st, f"{outdir}/state.pt.tmp")
    os.replace(f"{outdir}/state.pt.tmp", f"{outdir}/state.pt")
    json.dump(hist, open(f"{outdir}/train_hist.json", "w"))
    if sync:
        os.makedirs(sync, exist_ok=True)
        delta_int4(m, f"{sync}/dpo_delta_int4.safetensors")
        json.dump(hist, open(f"{sync}/train_hist.json", "w"))
        json.dump({"step": step, "cursor": cursor, "time": time.strftime("%F %T")}, open(f"{sync}/train_state.json", "w"))
    esft.log("saved state step", step, "cursor", cursor)


def train(a):
    model_dir, delta, stats, pairs, refs, outdir = a[:6]
    o = dict(zip(a[6::2], a[7::2]))
    beta, lr, accum = float(o.get("--beta", 0.1)), float(o.get("--lr", 5e-7)), int(o.get("--accum", 8))
    epochs, rpo, minutes = float(o.get("--epochs", 2)), float(o.get("--rpo", 0.0)), float(o.get("--minutes", 300))
    max_len, sync = int(o.get("--max-len", 3072)), o.get("--sync", "/root/ckpt_sync")
    os.makedirs(outdir, exist_ok=True)
    m = ft.load(model_dir, esft.DEV)
    if delta != "none":
        apply_delta(m, delta)
    params = esft.select(m, json.load(open(stats)))
    esft.log("trainable B", round(sum(p.numel() for p in params) / 1e9, 2))
    opt = esft.AdamSR(params, lr=lr)
    ps, rf = [json.loads(l) for l in open(pairs)], [json.loads(l) for l in open(refs)]
    data = [(p, r) for p, r in zip(ps, rf) if r["rc"] is not None]
    order = list(range(len(data)))
    random.Random(0).shuffle(order)
    total = int(len(data) * epochs) // accum
    step, cursor, hist = 0, 0, {"args": o, "steps": []}
    if o.get("--resume") and os.path.exists(f"{outdir}/state.pt"):
        st = torch.load(f"{outdir}/state.pt")
        m.load_state_dict(st["params"], strict=False)
        pmap = dict(m.named_parameters())
        for n, s in st["opt"].items():
            opt.state[pmap[n]] = {k: (v.to(pmap[n].device) if torch.is_tensor(v) else v) for k, v in s.items()}
        step, cursor = st["step"], st["cursor"]
        hist = json.load(open(f"{outdir}/train_hist.json"))
        esft.log("resumed at step", step)
    sched = lambda s: lr * min(1.0, (s + 1) / 5) * (0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    m.train()
    m.grad_ckpt = True
    t0 = last = time.time()
    while step < total:
        for g in opt.param_groups:
            g["lr"] = sched(step)
        acc = {"loss": 0.0, "margin": 0.0, "win": 0}
        for _ in range(accum):
            p, r = data[order[cursor % len(data)]]
            cursor += 1
            (ci, cm), (ri, rm) = seqs(p, max_len)
            pc, nc = logp(m, ci, cm)
            pr, _ = logp(m, ri, rm)
            z = beta * ((pc - r["rc"]) - (pr - r["rr"]))
            loss = -p["weight"] * F.logsigmoid(z) + (rpo * -pc / nc if rpo else 0)
            (loss / accum).backward()
            acc["loss"] += loss.item() / accum
            acc["margin"] += z.item() / beta / accum
            acc["win"] += int(z.item() > 0)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        row = {"step": step, **{k: round(v, 4) for k, v in acc.items()}, "gnorm": round(gn, 3), "lr": sched(step),
               "min": round((time.time() - t0) / 60, 1)}
        hist["steps"].append(row)
        esft.log(json.dumps(row))
        step += 1
        if time.time() - last > 1800:
            save_state(m, opt, step, cursor, hist, outdir, sync)
            last = time.time()
        if (time.time() - t0) / 60 > minutes:
            esft.log("time budget reached")
            break
    save_state(m, opt, step, cursor, hist, outdir, sync)
    delta_int4(m, f"{outdir}/esft_delta.safetensors")
    esft.log("done; delta", f"{outdir}/esft_delta.safetensors")


def bench(a):
    model_dir, stats, L = a[0], a[1], int(a[2])
    m = ft.load(model_dir, esft.DEV)
    params = esft.select(m, json.load(open(stats)))
    opt = esft.AdamSR(params, lr=1e-7)
    m.train()
    m.grad_ckpt = True
    P = 600
    p = {"prompt_ids": torch.randint(1000, 100000, (P,)).tolist(), "chosen_ids": torch.randint(1000, 100000, (L - P,)).tolist(),
         "rejected_ids": torch.randint(1000, 100000, (L - P,)).tolist(), "rejected_capped": True}
    for _ in range(2):
        torch.cuda.synchronize()
        t = time.time()
        (ci, cm), (ri, rm) = seqs(p, L + 1)
        pc, _ = logp(m, ci, cm)
        pr, _ = logp(m, ri, rm)
        (-F.logsigmoid(0.1 * (pc - pr))).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        esft.log(json.dumps({"len": L, "pair_step_s": round(time.time() - t, 1),
                             "peak_GB": round(torch.cuda.max_memory_allocated() / 2**30, 1)}))
    t = time.time()
    delta_int4(m, "/root/bench_delta.safetensors")
    esft.log("delta_int4 s", round(time.time() - t), "MB", os.path.getsize("/root/bench_delta.safetensors") >> 20)
    os.remove("/root/bench_delta.safetensors")




if __name__ == "__main__":
    {"ref": ref, "train": train, "bench": bench}[sys.argv[1]](sys.argv[2:])

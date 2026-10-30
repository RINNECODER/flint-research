"""LoRA SFT of Flint on attention (q/k/v/o) + the shared expert (dense MLP in layers 0-1); routed experts stay frozen int4.

  probe  <model> <train.jsonl> [--r R]                     attention self-check + 4 timed steps on the longest rows
  train  <model> <train.jsonl> <val.jsonl> <outdir> [--lr 1e-4] [--r 32] [--alpha 64] [--accum 8] [--minutes M] [--frac F]
  export <model> <outdir> <val.jsonl> [bits=4]             merge LoRA, re-quantize (MLX affine g64, mse; 4 or 8 bit), delta
                                                           in the checkpoint's own key layout + val NLL of the int4 model

Resumable: every <=25 min and at the end, LoRA + AdamW state + step + data cursor go to /root/ckpt_sync (RESUME.md).
A train call finding /root/ckpt_sync/state.pt with the same args resumes from it.
"""
import json
import math
import os
import random
import sys
import time

import torch

import esft
import flint_torch as ft
from esft import log, nll, rows

SYNC = "/root/ckpt_sync"


def targets(m):
    """(name, QLinear) for every LoRA target, names in the checkpoint's key layout."""
    for i, l in enumerate(m.layers):
        for t in ("q_proj", "k_proj", "v_proj", "o_proj"):
            yield f"model.layers.{i}.self_attn.{t}", getattr(l.self_attn, t)
        mlp, path = (l.mlp.shared_expert, "mlp.shared_expert") if isinstance(l.mlp, ft.MoE) else (l.mlp, "mlp")
        for t in ("gate_proj", "up_proj", "down_proj"):
            yield f"model.layers.{i}.{path}.{t}", getattr(mlp, t)


def add_lora(m, r, alpha):
    for p in m.parameters():
        p.requires_grad_(False)
    for _, q in targets(m):
        q.add_lora(r, alpha)
    return [p for n, p in m.named_parameters() if "lora_" in n]


def lora_sd(m):
    return {k: v.detach().cpu() for k, v in m.state_dict().items() if "lora_" in k}


def save_sync(m, opt, step, it, hist, args):
    os.makedirs(SYNC, exist_ok=True)
    torch.save({"lora": lora_sd(m), "opt": opt.state_dict(), "step": step, "it": it, "hist": hist, "args": args},
               f"{SYNC}/state.pt.tmp")
    os.replace(f"{SYNC}/state.pt.tmp", f"{SYNC}/state.pt")
    with open(f"{SYNC}/RESUME.md", "w") as f:
        f.write(f"# Flint LoRA SFT resume state ({time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())})\n\n"
                f"- state.pt: LoRA weights (fp32), AdamW state, optimizer step {step} done, data cursor {it} "
                f"(index into train rows shuffled with random.Random(0)), loss/val history.\n- args: {json.dumps(args)}\n"
                f"- Resume: re-run the same `sft.py train ...` command on a pod with /root/ckpt_sync/state.pt in place.\n")
    log("synced step", step, "it", it)


def train(model, trp, vap, outdir, opt_args):
    lr, r, alpha = float(opt_args.get("--lr", 1e-4)), int(opt_args.get("--r", 32)), int(opt_args.get("--alpha", 64))
    accum, minutes, frac = int(opt_args.get("--accum", 8)), float(opt_args.get("--minutes", 600)), float(opt_args.get("--frac", 1))
    args = {"lr": lr, "r": r, "alpha": alpha, "accum": accum, "frac": frac}
    m = ft.load(model, esft.DEV)
    params = add_lora(m, r, alpha)
    log("trainable M", round(sum(p.numel() for p in params) / 1e6, 1))
    tr, va = rows(trp), rows(vap)
    random.Random(0).shuffle(tr)
    tr = tr[:int(len(tr) * frac)]
    total = len(tr) // accum
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = lambda s: lr * min(1.0, (s + 1) / 10) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    step, it, hist = 0, 0, {"args": args, "total_steps": total, "val": [], "steps": []}
    if os.path.exists(f"{SYNC}/state.pt"):
        st = torch.load(f"{SYNC}/state.pt", weights_only=False)
        assert st["args"] == args, (st["args"], args)
        m.load_state_dict({k: v.to(dict(m.named_parameters())[k].device) for k, v in st["lora"].items()}, strict=False)
        opt.load_state_dict(st["opt"])
        step, it, hist = st["step"], st["it"], st["hist"]
        log("resumed at step", step, "it", it)
    m.train()
    m.grad_ckpt = True
    if not hist["val"]:
        hist["val"].append({"step": 0, "nll": nll(m, va)})
        log("val", hist["val"][-1])
    t0 = last = time.time()
    while step < total:
        if step == total // 2 and all(v["step"] != step for v in hist["val"]):
            hist["val"].append({"step": step, "nll": nll(m, va)})
            log("val", hist["val"][-1])
        for g in opt.param_groups:
            g["lr"] = sched(step)
        batch = tr[it:it + accum]
        ntok = sum(sum(x["mask"][1:]) for x in batch)
        lsum = 0.0
        for x in batch:
            s, _ = m.loss(torch.tensor([x["ids"]]), mask=torch.tensor([x["mask"]]))
            (s / ntok).backward()
            lsum += s.item()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step, it = step + 1, it + accum
        row = {"step": step, "loss": round(lsum / ntok, 4), "gnorm": round(gn, 3), "lr": sched(step - 1),
               "toks": sum(len(x["ids"]) for x in batch), "min": round((time.time() - t0) / 60, 1),
               "peak_GB": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        hist["steps"].append(row)
        log(json.dumps(row))
        if time.time() - last > 25 * 60:
            save_sync(m, opt, step, it, hist, args)
            last = time.time()
        if (time.time() - t0) / 60 > minutes:
            log("time budget reached")
            break
    if step == total:
        hist["val"].append({"step": step, "nll": nll(m, va)})
        log("val", hist["val"][-1])
    save_sync(m, opt, step, it, hist, args)
    os.makedirs(outdir, exist_ok=True)
    torch.save({"lora": lora_sd(m), "args": args}, f"{outdir}/lora.pt")
    json.dump(hist, open(f"{outdir}/train_hist.json", "w"), indent=1)


@torch.no_grad()
def export(model, outdir, vap, bits=4):
    from safetensors.torch import save_file
    lo = torch.load(f"{outdir}/lora.pt")
    m = ft.load(model, esft.DEV)
    a = lo["args"]
    add_lora(m, a["r"], a["alpha"])
    m.load_state_dict({k: v.to(dict(m.named_parameters())[k].device) for k, v in lo["lora"].items()}, strict=False)
    delta, rel, keep, resid = {}, [], [], []
    for name, q in targets(m):
        W = q.weight().float()
        d = (q.lora_b @ q.lora_a) * q.lora_scale
        Wn = W + d
        pw, s, b = esft.quantize(Wn, bits=bits)
        Wq = ft.dequant(pw, s.to(ft.CDT), b.to(ft.CDT), bits).float()
        rel.append(((Wq - Wn) ** 2).mean().item() / (Wn ** 2).mean().item())  # ESFT's metric
        keep.append(((Wq - W) * d).sum().item() / (d * d).sum().item())  # share of the LoRA delta the int4 weight keeps
        resid.append(((Wq - Wn).norm() / d.norm()).item())  # re-quant error relative to the delta's size
        q.load(pw, s, b)
        q.bits = bits
        q.lora_a = q.lora_b = None
        delta.update({f"{name}.weight": pw.cpu(), f"{name}.scales": s.cpu(), f"{name}.biases": b.cpu()})
    save_file(delta, f"{outdir}/sft_delta_int{bits}.safetensors")
    mean = lambda x: sum(x) / len(x)
    res = {"tensors": len(rel), "rel_mse_mean": mean(rel), "delta_kept_mean": mean(keep), "requant_err_over_delta_mean": mean(resid),
           "val_nll_int4": nll(m, rows(vap))}
    log("export", json.dumps(res))
    json.dump(res, open(f"{outdir}/export_int{bits}.json", "w"), indent=1)


def probe(model, trp, opt_args):
    m = ft.load(model, esft.DEV)
    log("loaded GB", round(torch.cuda.memory_allocated() / 2**30, 1))
    # self-check: memory-efficient attention == the original dense fp32 attention (one sliding, one full layer)
    with torch.no_grad():
        for li in (2, 5):
            at = m.layers[li].self_attn
            x = torch.randn(1, 700, 2048, device=at.q_proj.w.device)
            ft.SDPA = True
            y1 = at(x)
            ft.SDPA = False
            y0 = at(x)
            ft.SDPA = True
            err = ((y1 - y0).norm() / y0.norm()).item()
            log("attn check layer", li, "sliding", at.sliding, "rel err", round(err, 5))
            assert err < 2e-2, err
    params = add_lora(m, int(opt_args.get("--r", 32)), 64)
    log("trainable M", round(sum(p.numel() for p in params) / 1e6, 1))
    opt = torch.optim.AdamW(params, lr=1e-4)
    tr = sorted(rows(trp), key=lambda x: len(x["ids"]))[-4:]
    m.train()
    m.grad_ckpt = True
    for x in tr:
        torch.cuda.synchronize()
        t = time.time()
        s, k = m.loss(torch.tensor([x["ids"]]), mask=torch.tensor([x["mask"]]))
        (s / k).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        log(json.dumps({"tokens": len(x["ids"]), "step_s": round(time.time() - t, 2), "loss": round(s.item() / k, 4),
                        "peak_GB": round(torch.cuda.max_memory_allocated() / 2**30, 1)}))


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "probe":
        probe(a[2], a[3], dict(zip(a[4::2], a[5::2])))
    elif a[1] == "train":
        train(a[2], a[3], a[4], a[5], dict(zip(a[6::2], a[7::2])))
    elif a[1] == "export":
        export(a[2], a[3], a[4], int(a[5]) if len(a) > 5 else 4)

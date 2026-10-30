"""Expert-Specialized Fine-Tuning (ESFT, DeepSeek 2024) of Flint on top of flint_torch.

  stats  <model> <data.jsonl> <out.json> [n]      routed gate mass per expert over n examples; picks experts per layer
  train  <model> <stats.json> <train.jsonl> <val.jsonl> <outdir> [--minutes M] [--lr LR]
  export <model> <outdir>                          trained experts -> int4 (affine g64, MSE scale search) delta file
  bench  <model> <stats.json> <train.jsonl>        3 timed ESFT steps (GPU pick)

Only the selected routed experts train (fp32 master, bf16 compute); everything else stays packed int4.
"""
import json
import math
import os
import random
import sys
import time

import torch

import flint_torch as ft

DEV = ["cuda:%d" % i for i in range(torch.cuda.device_count())] or ["cpu"]
EOS = (150001, 150003, 150005, 150018)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def moes(m):
    return [(i, l.mlp) for i, l in enumerate(m.layers) if isinstance(l.mlp, ft.MoE)]


def rows(path, n=None):
    out = [json.loads(l) for l in open(path)]
    return out[:n] if n else out


# ---------------------------------------------------------------- expert relevance
@torch.no_grad()
def stats(m, data, cover):
    for _, moe in moes(m):
        moe.stats = torch.zeros(moe.router_w.shape[0], device=moe.router_w.device)
    toks = 0
    for r in data:
        m(torch.tensor([r["ids"]]), hidden=True)
        toks += len(r["ids"])
    res = {"tokens": toks, "cover": cover, "layers": {}}
    for i, moe in moes(m):
        share = (moe.stats / moe.stats.sum()).cpu()
        order = share.argsort(descending=True)
        n = int((share[order].cumsum(0) < cover).sum()) + 1  # smallest set whose gate mass reaches `cover`
        res["layers"][i] = {"experts": order[:n].tolist(), "share": [round(x, 6) for x in share.tolist()]}
        moe.stats = None
    return res


def select(m, st):
    for i, moe in moes(m):
        moe.make_trainable(st["layers"][str(i)]["experts"])
    for p in m.parameters():
        p.requires_grad_(False)
    params = [p for n, p in m.named_parameters() if ".esft." in n]
    for p in params:
        p.requires_grad_(True)
    return params


# ---------------------------------------------------------------- training
def sr_bf16(x):
    """fp32 -> bf16 with stochastic rounding (unbiased, so updates far below bf16's resolution still add up)."""
    i = x.view(torch.int32)
    return ((i + torch.randint_like(i, 0, 1 << 16)) & -65536).view(torch.float32).bfloat16()


class AdamSR(torch.optim.Optimizer):
    """AdamW for bf16 parameters: bf16 moments, the step computed in fp32 and stochastically rounded back to bf16.
    8 bytes per parameter instead of 16, which is what lets ~1.9B expert weights train beside the int4 model on 48 GB."""

    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                st["t"] += 1
                gr = p.grad.float()
                m = st["m"].float().mul_(b1).add_(gr, alpha=1 - b1)
                v = st["v"].float().mul_(b2).addcmul_(gr, gr, value=1 - b2)
                st["m"].copy_(m)
                st["v"].copy_(v)
                step = (m / (1 - b1 ** st["t"])) / (v / (1 - b2 ** st["t"])).sqrt_().add_(g["eps"])
                p.copy_(sr_bf16(p.float().add_(step, alpha=-g["lr"])))


def nll(m, data):
    m.eval()
    tot = n = 0
    with torch.no_grad():
        for r in data:
            s, k = m.loss(torch.tensor([r["ids"]]), mask=torch.tensor([r["mask"]]))
            tot, n = tot + s.item(), n + k
    m.train()
    return tot / max(n, 1)


def train(m, params, tr, va, outdir, minutes, lr, accum, budget_steps=None):
    os.makedirs(outdir, exist_ok=True)
    opt = AdamSR(params, lr=lr)
    m.train()
    m.grad_ckpt = True
    random.Random(0).shuffle(tr)
    total = budget_steps or len(tr) // accum
    sched = lambda s: lr * min(1.0, (s + 1) / 10) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    hist = {"val_base": nll(m, va), "steps": []}
    log("val nll base", hist["val_base"])
    t0 = last_ck = time.time()
    it = 0
    for step in range(total):
        for g in opt.param_groups:
            g["lr"] = sched(step)
        lsum = nsum = 0
        batch = [tr[(it + j) % len(tr)] for j in range(accum)]
        it += accum
        ntok = sum(sum(r["mask"][1:]) for r in batch)
        for r in batch:
            s, k = m.loss(torch.tensor([r["ids"]]), mask=torch.tensor([r["mask"]]))
            (s / ntok).backward()
            lsum, nsum = lsum + s.item(), nsum + k
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        row = {"step": step, "loss": lsum / nsum, "gnorm": round(gn, 3), "lr": sched(step), "min": round((time.time() - t0) / 60, 1)}
        hist["steps"].append(row)
        log(json.dumps(row))
        if time.time() - last_ck > 1800:
            save(m, outdir)
            last_ck = time.time()
        if (time.time() - t0) / 60 > minutes:
            log("time budget reached")
            break
    save(m, outdir)
    hist["val_tuned"] = nll(m, va)
    log("val nll tuned", hist["val_tuned"])
    json.dump(hist, open(f"{outdir}/train_hist.json", "w"))


def save(m, outdir):
    sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in m.state_dict().items() if ".esft." in k}
    torch.save(sd, f"{outdir}/esft_ckpt.pt.tmp")
    os.replace(f"{outdir}/esft_ckpt.pt.tmp", f"{outdir}/esft_ckpt.pt")
    log("checkpoint", len(sd))


# ---------------------------------------------------------------- int4 export (MLX affine, 4 bit, group 64)
def quantize(w, group=64, grid=None, bits=4):
    """w [..., in] -> (uint8 packed [..., in/2], scales, biases bf16 [..., in/group]). Per group, try min/max ranges
    shrunk by each factor in `grid` and keep the one with the least squared error (an 'mse' scale rule)."""
    grid = grid if grid is not None else [1.0 - 0.01 * i for i in range(25)]
    g = w.float().unflatten(-1, (-1, group))
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    best_err = best = None
    for a in grid:
        l, h = lo * a, hi * a
        s = ((h - l) / (2 ** bits - 1)).clamp(min=1e-8).bfloat16().float()
        b = l.bfloat16().float()
        q = ((g - b) / s).round().clamp(0, 2 ** bits - 1)
        err = ((q * s + b - g) ** 2).sum(-1, keepdim=True)
        if best is None:
            best_err, best = err, (q, s, b)
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best = tuple(torch.where(take, new, old) for new, old in zip((q, s, b), best))
    q, s, b = best
    q = q.to(torch.uint8).flatten(-2)
    packed = q[..., 0::2] | (q[..., 1::2] << 4) if bits == 4 else q  # 8 bit: one byte per weight, as the embedding
    return packed, s[..., 0].bfloat16(), b[..., 0].bfloat16()


def export(m, outdir):
    """Quantize each trained expert into the model's packed buffers (so the model in memory is the shipped one) and
    write the delta: per MoE layer and projection, the selected experts' new weight/scales/biases."""
    from safetensors.torch import save_file
    delta, errs = {}, []
    for i, moe in moes(m):
        if not len(moe.esft):
            continue
        es = sorted({int(k.split("_")[0]) for k in moe.esft.keys()})
        delta[f"{i}.experts"] = torch.tensor(es, dtype=torch.int32)
        for n in ("gate", "up", "down"):
            ws, ss, bs = [], [], []
            for e in es:
                w = moe.esft[f"{e}_{n}"].detach()
                pw, s, b = quantize(w)
                getattr(moe, f"{n}_w")[e] = pw
                getattr(moe, f"{n}_s")[e] = s.to(ft.CDT)
                getattr(moe, f"{n}_b")[e] = b.to(ft.CDT)
                errs.append(((ft.dequant(pw, s.to(ft.CDT), b.to(ft.CDT), 4).float() - w.float()) ** 2).mean().item() / w.float().pow(2).mean().item())
                ws.append(pw.cpu()); ss.append(s.cpu()); bs.append(b.cpu())
            delta[f"{i}.{n}_proj.weight"] = torch.stack(ws)
            delta[f"{i}.{n}_proj.scales"] = torch.stack(ss)
            delta[f"{i}.{n}_proj.biases"] = torch.stack(bs)
        moe.esft = torch.nn.ParameterDict()
    save_file(delta, f"{outdir}/esft_delta.safetensors")
    log("export: experts", sum(len(v) for k, v in delta.items() if k.endswith(".experts")), "rel mse mean", sum(errs) / len(errs))


def load_ckpt(m, outdir):
    sd = torch.load(f"{outdir}/esft_ckpt.pt")
    for i, moe in moes(m):
        es = sorted({int(k.split(".esft.")[1].split("_")[0]) for k in sd if k.startswith(f"layers.{i}.")})
        moe.make_trainable(es)
    missing = m.load_state_dict(sd, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys[:3]


# ---------------------------------------------------------------- batched greedy generation
@torch.no_grad()
def generate(m, prompts, max_new, eos=EOS, prefill_rows=16):
    """Greedy decode of left-padded prompts with a KV cache; finished rows leave the batch. Returns (tokens, hit_cap)."""
    T0 = max(map(len, prompts))
    ids = torch.tensor([[0] * (T0 - len(p)) + p for p in prompts])
    valid = torch.tensor([[False] * (T0 - len(p)) + [True] * len(p) for p in prompts])
    pos = (valid.cumsum(1) - 1).clamp(min=0)
    parts = []
    for s in range(0, len(prompts), prefill_rows):  # prefill a few rows at a time (the MoE's padded matmuls are big)
        g = {"cache": {}, "pos": pos[s:s + prefill_rows], "valid": valid[s:s + prefill_rows]}
        parts.append((m(ids[s:s + prefill_rows], gen=g)[:, -1], g["cache"]))
    logits = torch.cat([p[0] for p in parts])
    gen = {"cache": {k: tuple(torch.cat([p[1][k][i] for p in parts]) for i in range(4)) for k in parts[0][1]}}
    nxt, live = pos[:, -1:] + 1, list(range(len(prompts)))
    out, capped = [[] for _ in prompts], [False] * len(prompts)
    for step in range(max_new):
        tok = logits.argmax(-1).cpu()
        keep = []
        for j, (r, t) in enumerate(zip(live, tok.tolist())):
            if t not in eos:
                out[r].append(t)
                keep.append(j)
        if step == max_new - 1:
            for j in keep:
                capped[live[j]] = True
            break
        if not keep:
            break
        if len(keep) < len(live):
            sel = torch.tensor(keep)
            gen["cache"] = {k: tuple(x[sel.to(x.device)] for x in c) for k, c in gen["cache"].items()}
            live, tok, nxt = [live[j] for j in keep], tok[sel], nxt[sel]
        gen["pos"], gen["valid"] = nxt, torch.ones_like(nxt, dtype=torch.bool)
        nxt = nxt + 1
        logits = m(tok[:, None], gen=gen)[:, -1]
    return out, capped


# ---------------------------------------------------------------- CLI
def main():
    cmd, model_dir = sys.argv[1], sys.argv[2]
    m = ft.load(model_dir, DEV)
    log("loaded", [round(torch.cuda.memory_allocated(d) / 2**30, 1) for d in DEV if d != "cpu"])
    if cmd == "stats":
        data = rows(sys.argv[3], int(sys.argv[5]) if len(sys.argv) > 5 else 300)
        res = stats(m.eval(), data, float(os.environ.get("COVER", "0.2")))
        json.dump(res, open(sys.argv[4], "w"))
        n = [len(v["experts"]) for v in res["layers"].values()]
        log("experts per layer", n, "total", sum(n), "params B", round(sum(n) * 3 * 512 * 2048 / 1e9, 2))
    elif cmd == "bench":
        params = select(m, json.load(open(sys.argv[3])))
        log("trainable B", round(sum(p.numel() for p in params) / 1e9, 2))
        tr = sorted(rows(sys.argv[4]), key=lambda r: len(r["ids"]))[-3:]  # the longest: peak memory
        opt = AdamSR(params, lr=1e-5)
        m.train(); m.grad_ckpt = True
        for r in tr:
            torch.cuda.synchronize(); t = time.time()
            ids = torch.tensor([r["ids"]])
            s, k = m.loss(ids, mask=torch.tensor([r["mask"]]))
            (s / k).backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            log(json.dumps({"step_s": round(time.time() - t, 2), "loss": s.item() / k, "peak_GB": [round(torch.cuda.max_memory_allocated(d) / 2**30, 1) for d in DEV]}))
    elif cmd == "train":
        a = sys.argv
        params = select(m, json.load(open(a[3])))
        log("trainable B", round(sum(p.numel() for p in params) / 1e9, 2))
        opt = dict(zip(a[7::2], a[8::2]))
        train(m, params, rows(a[4]), rows(a[5]), a[6], float(opt.get("--minutes", 180)), float(opt.get("--lr", 1e-5)),
              int(opt.get("--accum", 8)), int(opt["--steps"]) if "--steps" in opt else None)
    elif cmd == "export":
        load_ckpt(m, sys.argv[3])
        export(m, sys.argv[3])


if __name__ == "__main__":
    main()

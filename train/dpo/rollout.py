"""On-policy rollouts for DPO: k sampled replies per problem, thinking on, continuous batching (finished rows are
refilled with new prompts, so a few long replies don't idle the batch).

  python3 rollout.py <model dir> <delta|none> <problems.jsonl> <out.jsonl> [--k 4] [--temp 0.8] [--max-new 4096]
                     [--bs 96] [--refill 16] [--n 600]

Resumable: (id, k) pairs already in out.jsonl are skipped. Each line: id, k, tokens, capped, out_ids, text.
Score with score.py afterwards (CPU only).
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/root/ft")
import esft
import flint_torch as ft


def apply_delta(m, path):
    from safetensors.torch import load_file
    dl = load_file(path)
    for i, moe in esft.moes(m):
        if f"{i}.experts" not in dl:
            continue
        es = dl[f"{i}.experts"].tolist()
        for n in ("gate", "up", "down"):
            dev = getattr(moe, f"{n}_w").device
            getattr(moe, f"{n}_w")[es] = dl[f"{i}.{n}_proj.weight"].to(dev)
            getattr(moe, f"{n}_s")[es] = dl[f"{i}.{n}_proj.scales"].to(dev, ft.CDT)
            getattr(moe, f"{n}_b")[es] = dl[f"{i}.{n}_proj.biases"].to(dev, ft.CDT)
    esft.log("applied delta", path)


def _pad_left(c, T):
    """Left-pad one layer cache (k, v, kpos, kvalid) along time to T keys with invalid slots."""
    k, v, p, mk = c
    d = T - k.shape[2]
    if d <= 0:
        return c
    z = lambda x: torch.zeros(*x.shape[:2], d, *x.shape[3:], dtype=x.dtype, device=x.device)
    return (torch.cat((z(k), k), 2), torch.cat((z(v), v), 2), torch.cat((torch.zeros(p.shape[0], d, dtype=p.dtype, device=p.device), p), 1),
            torch.cat((torch.zeros(mk.shape[0], d, dtype=mk.dtype, device=mk.device), mk), 1))


def _merge(a, b):
    """Row-concatenate two caches (dict layer -> 4-tuple) whose key lengths may differ."""
    if a is None:
        return b
    out = {}
    for key in a:
        T = max(a[key][0].shape[2], b[key][0].shape[2])
        x, y = _pad_left(a[key], T), _pad_left(b[key], T)
        out[key] = tuple(torch.cat((u, w.to(u.device))) for u, w in zip(x, y))
    return out


@torch.no_grad()
def _prefill(m, prompts, rows=16):
    """Left-padded prefill; returns (last logits, cache, next position per row)."""
    T0 = max(map(len, prompts))
    ids = torch.tensor([[0] * (T0 - len(p)) + p for p in prompts])
    valid = torch.tensor([[False] * (T0 - len(p)) + [True] * len(p) for p in prompts])
    pos = (valid.cumsum(1) - 1).clamp(min=0)
    logits, cache = [], None
    for s in range(0, len(prompts), rows):
        g = {"cache": {}, "pos": pos[s:s + rows], "valid": valid[s:s + rows]}
        logits.append(m(ids[s:s + rows], gen=g)[:, -1])
        cache = _merge(cache, g["cache"])
    return torch.cat(logits), cache, pos[:, -1:] + 1


@torch.no_grad()
def sample_stream(m, jobs, max_new, temp, bs, refill, eos=esft.EOS, seed=0):
    """jobs: list of (key, prompt ids). Yields (key, out ids, capped) as rows finish. Keeps up to bs rows live and
    prefills new prompts whenever at least `refill` slots are free (or the queue would otherwise stall)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    queue = list(jobs)
    live, outs, cache, logits, nxt = [], [], None, None, None
    while queue or live:
        free = bs - len(live)
        if queue and (free >= refill or not live):
            new = [queue.pop(0) for _ in range(min(free, len(queue)))]
            lg, c, nx = _prefill(m, [p for _, p in new])
            logits = lg if logits is None else torch.cat((logits, lg.to(logits.device)))
            nxt = nx if nxt is None else torch.cat((nxt, nx))
            cache = _merge(cache, c)
            live += [k for k, _ in new]
            outs += [[] for _ in new]
        if temp > 0:
            tok = torch.multinomial(torch.softmax(logits.float().cpu() / temp, -1), 1, generator=g)[:, 0]
        else:
            tok = logits.argmax(-1).cpu()
        keep = []
        for j, t in enumerate(tok.tolist()):
            if t in eos:
                yield live[j], outs[j], False
                continue
            outs[j].append(t)
            if len(outs[j]) >= max_new:
                yield live[j], outs[j], True
            else:
                keep.append(j)
        if not keep:
            live, outs, cache, logits, nxt = [], [], None, None, None
            continue
        if len(keep) < len(live):
            sel = torch.tensor(keep)
            cache = {k: tuple(x[sel.to(x.device)] for x in c) for k, c in cache.items()}
            live, outs, tok, nxt = [live[j] for j in keep], [outs[j] for j in keep], tok[sel], nxt[sel]
        gen = {"cache": cache, "pos": nxt, "valid": torch.ones_like(nxt, dtype=torch.bool)}
        logits = m(tok[:, None], gen=gen)[:, -1]
        cache, nxt = gen["cache"], nxt + 1


def main(a):
    from jinja2 import Environment
    from tokenizers import Tokenizer
    model_dir, delta, probs_path, out = a[0], a[1], a[2], a[3]
    opt = dict(zip(a[4::2], a[5::2]))
    K, temp, max_new = int(opt.get("--k", 4)), float(opt.get("--temp", 0.8)), int(opt.get("--max-new", 4096))
    bs, refill, n = int(opt.get("--bs", 96)), int(opt.get("--refill", 16)), int(opt.get("--n", 600))
    tmpl = Environment().from_string(open(f"{model_dir}/chat_template.jinja").read())
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    m = ft.load(model_dir, esft.DEV).eval()
    if delta != "none":
        apply_delta(m, delta)
    probs = [json.loads(l) for l in open(probs_path)][:n]
    done = {(r["id"], r["k"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    jobs = []
    for p in probs:
        ids = tok.encode(tmpl.render(messages=[{"role": "user", "content": p["prompt"]}], add_generation_prompt=True),
                         add_special_tokens=False).ids
        jobs += [((p["id"], k, ids), ids) for k in range(K) if (p["id"], k) not in done]
    esft.log("jobs", len(jobs), "done", len(done))
    t0, ntok, nrow = time.time(), 0, 0
    with open(out, "a") as f:
        for (pid, k, pids), o, capped in sample_stream(m, jobs, max_new, temp, bs, refill, seed=len(done)):
            f.write(json.dumps({"id": pid, "k": k, "tokens": len(o), "capped": capped, "prompt_ids": pids, "out_ids": o,
                                "text": tok.decode(o, skip_special_tokens=False)}) + "\n")
            f.flush()
            ntok, nrow = ntok + len(o), nrow + 1
            if nrow % 50 == 0:
                esft.log(f"{nrow}/{len(jobs)} rows, {ntok / (time.time() - t0):.0f} tok/s")
    esft.log("done", nrow, "rows", ntok, "tokens", f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[1:])

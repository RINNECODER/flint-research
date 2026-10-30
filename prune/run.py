"""Streamed-Flint measurements in-process (Locals' loader: run with the Python of a Locals install, see docs/pruning.md).

  python run.py windows                      -> data/calib_windows.json, data/heldout_windows.json (token ids)
  python run.py stats MODEL OUT.npz          -> per-layer per-expert routing / REAP stats on the calib windows
  python run.py nll MODEL OUT.npz            -> per-token NLL and argmax on the 32 held-out windows
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
TOKENIZER_SRC = sorted((Path.home() / ".cache/huggingface/hub/models--dima6312--Flint-40B-A2.5B-abliterated-Rec-4bit/snapshots").iterdir())[0]  # hf download dima6312/Flint-40B-A2.5B-abliterated-Rec-4bit first
SOURCES = ["code", "swe-agent", "openhands"]
CALIB_TOKENS = 70_000  # per source
CALIB_WIN = 4096
HELD_WIN = 1025  # 1,024 transitions
HELD_N = {"code": 11, "swe-agent": 11, "openhands": 10}


def encode_docs(path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_SRC))
    for line in open(path):
        d = json.loads(line)
        if "text" in d:
            ids = tok.encode(d["text"], add_special_tokens=False)
        else:
            try:
                ids = tok.encode(tok.apply_chat_template(d["messages"], tokenize=False), add_special_tokens=False)
            except Exception:
                ids = tok.encode("\n\n".join(f"{m['role']}: {m['content']}" for m in d["messages"]), add_special_tokens=False)
        yield d["source"], ids


def windows():
    calib, used = [], {s: 0 for s in SOURCES}
    for src, ids in encode_docs(DATA / "calib.jsonl"):
        for i in range(0, len(ids), CALIB_WIN):
            w = ids[i:i + CALIB_WIN]
            if len(w) < 256 or used[src] >= CALIB_TOKENS:
                break
            calib.append({"source": src, "ids": w})
            used[src] += len(w)
    held, n = [], {s: 0 for s in SOURCES}
    for src, ids in encode_docs(DATA / "heldout.jsonl"):
        if n[src] < HELD_N[src] and len(ids) >= HELD_WIN:  # one window per document, from its middle
            mid = (len(ids) - HELD_WIN) // 2
            held.append({"source": src, "ids": ids[mid:mid + HELD_WIN]})
            n[src] += 1
    (DATA / "calib_windows.json").write_text(json.dumps(calib))
    (DATA / "heldout_windows.json").write_text(json.dumps(held))
    print("calib tokens", used, "windows", len(calib), "| heldout", n)


def load(model):
    from locals.lm import load as lload

    import os
    m, _ = lload(model, dtype=os.environ.get("FLINT_DTYPE", "auto"))
    return m


def stats(model, out):
    import mlx.core as mx
    from locals.models import flint_streamed as fs

    m = load(model)
    E = m.args.num_experts
    L = len(m.layers) - 2
    S = {k: np.zeros((L, E), np.float64) for k in ("count", "sum_gate", "sum_reap", "sum_norm", "sum_sigmoid")}
    by_src = np.zeros((len(SOURCES), L, E), np.int64)
    cur = {"src": 0}

    def call(self, x, residual=None):  # SparseMoe.__call__'s unfused path, plus the stats
        g = self.gate
        logits = g(x).astype(mx.float32) if "scales" in g else x.astype(mx.float32) @ g.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        inds = mx.argpartition(-scores, kth=self.k - 1, axis=-1)[..., : self.k]
        sel = mx.take_along_axis(scores, inds, axis=-1)
        w = sel / (sel.sum(axis=-1, keepdims=True) + 1e-20)
        y = self.experts(x, inds)
        norms = mx.linalg.norm(y.astype(mx.float32), axis=-1)
        mx.eval(inds, w, norms, sel)
        li = self.experts.layer - 2
        i, wn, nn_, sg = (np.array(a).reshape(-1) for a in (inds, w, norms, sel))
        np.add.at(S["count"][li], i, 1)
        np.add.at(S["sum_gate"][li], i, wn)
        np.add.at(S["sum_reap"][li], i, wn * nn_)
        np.add.at(S["sum_norm"][li], i, nn_)
        np.add.at(S["sum_sigmoid"][li], i, sg)
        np.add.at(by_src[cur["src"], li], i, 1)
        y = (y * w[..., None].astype(y.dtype)).sum(axis=-2) + self.shared_expert(x)
        return y if residual is None else residual + y

    fs.SparseMoe.__call__ = call
    wins = json.loads((DATA / "calib_windows.json").read_text())
    tokens = np.zeros(len(SOURCES), np.int64)
    t0 = time.time()
    for n, w in enumerate(wins):
        cur["src"] = SOURCES.index(w["source"])
        mx.eval(m.model(mx.array([w["ids"]]))[:, -1])  # hidden states only: the stats are in the MoE layers
        tokens[cur["src"]] += len(w["ids"])
        mx.clear_cache()
        if n % 5 == 0:
            print(f"{n + 1}/{len(wins)} windows, {tokens.sum()} tokens, {time.time() - t0:.0f}s, "
                  f"active {mx.get_active_memory() / 2**30:.1f} GB peak {mx.get_peak_memory() / 2**30:.1f} GB", flush=True)
    np.savez(out, **S, count_by_source=by_src, tokens_by_source=tokens, sources=np.array(SOURCES),
             layers=np.arange(2, 2 + L), num_experts=E)
    print("saved", out, tokens.sum(), "tokens", f"{time.time() - t0:.0f}s")


def nll(model, out):
    """REF=ref.npz (a run without REF, which saves its top-256 log-probs): adds per-token KL(ref || this model)."""
    import os

    import mlx.core as mx

    m = load(model)
    wins = json.loads((DATA / "heldout_windows.json").read_text())
    ref = np.load(os.environ["REF"]) if os.environ.get("REF") else None
    chunk = int(os.environ.get("CHUNK", "0"))  # prefill in chunks (a numerics-only perturbation, for a noise floor)
    nlls, tops, kls, ridx, rlp = [], [], [], [], []
    t0 = time.time()
    for n, w in enumerate(wins):
        ids = mx.array([w["ids"]])
        if chunk:
            from mlx_lm.models.cache import make_prompt_cache

            c = make_prompt_cache(m)
            logits = mx.concatenate([m(ids[:, i:min(i + chunk, ids.shape[1] - 1)], cache=c)[0]
                                     for i in range(0, ids.shape[1] - 1, chunk)]).astype(mx.float32)
        else:
            logits = m(ids[:, :-1]).astype(mx.float32)[0]
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = ids[0, 1:]
        nl = -mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
        am = mx.argmax(logits, axis=-1)
        if ref is None:
            ix = mx.argpartition(-lp, kth=255, axis=-1)[:, :256]
            v = mx.take_along_axis(lp, ix, axis=-1)
            mx.eval(ix, v)
            ridx.append(np.array(ix).astype(np.int32))
            rlp.append(np.array(v).astype(np.float32))
        else:
            rl = mx.array(ref["ref_lp"][n])
            q = mx.take_along_axis(lp, mx.array(ref["ref_idx"][n]), axis=-1)
            kls.append(np.array((mx.exp(rl) * (rl - q)).sum(-1)))
        mx.eval(nl, am)
        nlls.append(np.array(nl))
        tops.append(np.array(am))
    nlls, tops = np.stack(nlls), np.stack(tops)
    tgt = np.array([w["ids"][1:] for w in wins])
    extra = {"ref_idx": np.stack(ridx), "ref_lp": np.stack(rlp).astype(np.float16)} if ref is None else {"kl": np.stack(kls)}
    np.savez(out, nll=nlls, argmax=tops, target=tgt, sources=np.array([w["source"] for w in wins]), **extra)
    kl = f", KL {np.stack(kls).mean():.4f}" if kls else ""
    print(f"saved {out}: mean NLL {nlls.mean():.4f}, top-1 vs text {(tops == tgt).mean():.4f}{kl}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "windows":
        windows()
    elif cmd == "stats":
        stats(sys.argv[2], sys.argv[3])
    elif cmd == "nll":
        nll(sys.argv[2], sys.argv[3])

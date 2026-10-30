# --- appended after flint_jax.py by build.sh ---
import time, subprocess, json as _json
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
log("start", time.strftime("%H:%M:%S"), jax.__version__, jax.devices())
log(subprocess.run("df -h / /tmp /kaggle/working | tail -3; free -g | head -2; nproc", shell=True, capture_output=True, text=True).stdout)
from huggingface_hub import hf_hub_download, list_repo_files
REPO = "RecursiveSI/flint"  # BF16
files = sorted(f for f in list_repo_files(REPO) if f.endswith(".safetensors"))
cfg = _json.load(open(hf_hub_download(REPO, "config.json", local_dir="/tmp/flint")))
tokp = hf_hub_download(REPO, "tokenizer.json", local_dir="/tmp/flint")
def stream():  # one shard on disk at a time (80 GB total)
    for f in files:
        p = hf_hub_download(REPO, f, local_dir="/tmp/flint")
        log("shard", f)
        yield from iter_tensors([p])
devs = jax.devices()
params = load(cfg, stream(), devs, on_file_done=os.remove)
model = Flint(cfg, params)
for d in devs:
    s = d.memory_stats() or {}
    log(d, "weights GB", round(s.get("bytes_in_use", 0) / 2**30, 2))
from tokenizers import Tokenizer
tok = Tokenizer.from_file(tokp)
REF = _json.loads(REF_JSON)
res = {"weights": "RecursiveSI/flint BF16", "parity": []}
PADN = 64
for r in REF:
    p, c = r["prompt_ids"], r["ref_ids"]
    seq = (p + c + [cfg["pad_token_id"]] * PADN)[:PADN]
    t = time.time()
    lg = np.asarray(model.logits(jnp.asarray([seq])))[0, len(p) - 1: len(p) + len(c) - 1]
    fwd_s = time.time() - t
    lp = lg - np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1, keepdims=True)) - lg.max(-1, keepdims=True)
    agree = lp.argmax(-1) == np.array(c)
    gap = lp.max(-1) - lp[np.arange(len(c)), c]
    ids = list(p)
    for _ in range(32):
        s = (ids + [cfg["pad_token_id"]] * PADN)[:PADN]
        ids.append(int(np.asarray(model.logits(jnp.asarray([s])))[0, len(ids) - 1].argmax()))
    mine = tok.decode(ids[len(p):])
    div = next((i for i, (a, b) in enumerate(zip(mine, r["ref_text"])) if a != b), min(len(mine), len(r["ref_text"])))
    row = {"prompt": r["prompt"][:40], "n": len(c), "top1_agree": float(agree.mean()), "mean_logprob_gap": float(gap.mean()),
           "max_logprob_gap": float(gap.max()), "finite": bool(np.isfinite(lg).all()), "ref_text": r["ref_text"],
           "jax_greedy": mine, "same_chars_before_divergence": div, "fwd_s_incl_compile": round(fwd_s, 2)}
    res["parity"].append(row); log(_json.dumps(row))

lo = model.init_lora(16, 32)
log("lora params M", round(sum(x.size for x in jax.tree.leaves(lo)) / 1e6, 2))
st = (jax.tree.map(jnp.zeros_like, lo), jax.tree.map(jnp.zeros_like, lo))
text = tok.encode(open(__file__).read() * 4, add_special_tokens=False).ids
vg = jax.value_and_grad(model.loss)
res["lora"], step = [], 0
for L in SEQ_LENS:
    batch = jnp.asarray([[cfg["bos_token_id"]] + text[: L - 1]])
    for i in range(STEPS):
        step += 1
        t = time.time()
        l, g = vg(lo, batch)
        lo, st = adamw(lo, g, st, 1e-4, step=step)
        jax.block_until_ready(lo)
        peak = [round((d.memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30, 2) for d in devs]
        row = {"seq": L, "step": i, "loss": float(l), "step_s": round(time.time() - t, 2), "note": "step 0 includes compile", "peak_GB": peak}
        res["lora"].append(row); log(_json.dumps(row))
res["lora_b_abs_sum_after"] = float(sum(jnp.abs(x["b"]).sum() for lay in lo for x in lay.values()))
log("lora_b moved:", res["lora_b_abs_sum_after"])
_json.dump(res, open("/kaggle/working/result.json", "w"), indent=1)
log("done")

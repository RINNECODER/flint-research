# --- appended after flint_torch.py by build.sh ---
import time, sys, subprocess, json as _json
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
log("start", time.strftime("%H:%M:%S"), torch.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
log(subprocess.run("df -h /tmp /kaggle/working | tail -2; free -g | head -2", shell=True, capture_output=True, text=True).stdout)
from huggingface_hub import snapshot_download
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
path = snapshot_download("RecursiveSI/flint-int4", local_dir="/tmp/flint-int4", max_workers=8)
log("downloaded")
devs = ["cuda:0", "cuda:1"]
model = load(path, devs).eval()
for d in devs:
    log(d, "weights GB", round(torch.cuda.memory_allocated(d) / 2**30, 2))
from tokenizers import Tokenizer
tok = Tokenizer.from_file(f"{path}/tokenizer.json")
REF = _json.loads(REF_JSON)
res = {"parity": []}

# ---- parity: teacher-forced on Locals' greedy continuation, then our own greedy 32 tokens
with torch.no_grad():
    for r in REF:
        p, c = r["prompt_ids"], r["ref_ids"]
        t = time.time()
        logits = model(torch.tensor([p + c]))[0, len(p) - 1: len(p) + len(c) - 1].float()
        fwd_s = time.time() - t
        lp = logits.log_softmax(-1)
        ref_t = torch.tensor(c, device=lp.device)
        agree = (lp.argmax(-1) == ref_t).float()
        gap = (lp.max(-1).values - lp.gather(-1, ref_t[:, None])[:, 0])
        ids = list(p)
        for _ in range(32):
            ids.append(int(model(torch.tensor([ids]))[0, -1].argmax()))
        mine = tok.decode(ids[len(p):])
        first_div = next((i for i, (a, b) in enumerate(zip(mine, r["ref_text"])) if a != b), min(len(mine), len(r["ref_text"])))
        row = {"prompt": r["prompt"][:40], "n": len(c), "top1_agree": agree.mean().item(), "first_miss": int((agree == 0).nonzero()[0]) if (agree == 0).any() else None,
               "mean_logprob_gap": gap.mean().item(), "max_logprob_gap": gap.max().item(), "finite": bool(torch.isfinite(logits).all()),
               "ref_text": r["ref_text"], "torch_greedy": mine, "same_chars_before_divergence": first_div, "fwd_s": round(fwd_s, 2)}
        res["parity"].append(row); log(_json.dumps(row))

# ---- LoRA steps (attention q/k/v/o, r16 alpha32), gradient checkpointing per layer
params = model.add_lora(16, 32)
log("lora params M", round(sum(p.numel() for p in params) / 1e6, 2))
model.train(); model.grad_ckpt = True
opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
text = tok.encode(open(__file__).read() * 4, add_special_tokens=False).ids
res["lora"] = []
for L in SEQ_LENS:
    for d in devs:
        torch.cuda.reset_peak_memory_stats(d)
    batch = torch.tensor([[150000] + text[: L - 1]])
    for step in range(STEPS):
        for d in devs: torch.cuda.synchronize(d)
        t = time.time()
        loss, n = model.loss(batch)
        loss = loss / n
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step(); opt.zero_grad(set_to_none=True)
        for d in devs: torch.cuda.synchronize(d)
        row = {"seq": L, "step": step, "loss": loss.item(), "grad_norm": gnorm, "step_s": round(time.time() - t, 2),
               "peak_GB": [round(torch.cuda.max_memory_allocated(d) / 2**30, 2) for d in devs]}
        res["lora"].append(row); log(_json.dumps(row))
        del loss
lb = sum(p.abs().sum().item() for n, p in model.named_parameters() if "lora_b" in n)
res["lora_b_abs_sum_after"] = lb; log("lora_b moved:", lb)
_json.dump(res, open("/kaggle/working/result.json", "w"), indent=1)
log("done")

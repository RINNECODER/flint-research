"""flint_torch on the tiny MLX-built checkpoint must reproduce MLX's logits; LoRA and ESFT must train; cached batched
generation must match full forward passes; the int4 quantizer must round-trip an untouched expert. CPU, seconds."""
import sys, os, numpy as np, torch
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_root, "model"), os.path.join(_root, "train")]
import flint_torch as ft
import esft
torch.manual_seed(0)
d = sys.argv[1]
m = ft.load(d, ["cpu", "cpu"]).eval()
ids = torch.tensor(np.load(f"{d}/ids.npy"))
ref = torch.tensor(np.load(f"{d}/logits.npy"))
with torch.no_grad():
    got = m(ids)
err = (got - ref).abs().max().item() / ref.abs().max().item()
print("rel max err", err, "top1 agree", (got.argmax(-1) == ref.argmax(-1)).float().mean().item())
assert err < (2e-2 if ft.CDT == torch.float16 else 0.15), err  # bf16 keeps 8 mantissa bits

# cached generation: two left-padded rows, 12 greedy tokens, must equal greedy decoding by full forward passes
seqs = [ids[0, :7].tolist(), ids[0, 3:6].tolist()]
outs, capped = esft.generate(m, seqs, 12, eos=(), prefill_rows=1)
assert all(capped)
with torch.no_grad():
    for p, o in zip(seqs, outs):
        cur = list(p)
        for _ in range(12):
            cur.append(int(m(torch.tensor([cur]))[0, -1].argmax()))
        assert cur[len(p):] == o, (cur[len(p):], o)
print("cached generation matches")

# quantizer: an untouched expert's dequantized weights re-quantize to the same values
moe = m.layers[2].mlp
w = ft.dequant(moe.gate_w[1], moe.gate_s[1], moe.gate_b[1], 4).float()
pw, s, b = esft.quantize(w)
rt = ft.dequant(pw, s.to(ft.CDT), b.to(ft.CDT), 4).float()
print("requant rel err", ((rt - w) ** 2).mean().item() / (w ** 2).mean().item())
assert ((rt - w) ** 2).mean() / (w ** 2).mean() < 1e-4  # int4 error itself is ~1e-2

# ESFT: train experts 0,1 of every MoE layer
st = {"layers": {str(i): {"experts": [0, 1]} for i, _ in esft.moes(m)}}
params = esft.select(m, st); m.train(); m.grad_ckpt = True
opt = esft.AdamSR(params, lr=1e-2)
losses = []
for _ in range(5):
    s, k = m.loss(ids, chunk=6, mask=torch.ones_like(ids)); (s / k).backward(); opt.step(); opt.zero_grad()
    losses.append(s.item() / k)
print("esft losses", [round(x, 3) for x in losses]); assert losses[-1] < losses[0]
assert all(p.grad is None for n, p in m.named_parameters() if ".esft." not in n)
import tempfile; td = tempfile.mkdtemp()
m.eval()
with torch.no_grad():
    before = m(ids)
esft.export(m, td)
with torch.no_grad():
    after = m(ids)
print("export logit drift", (after - before).abs().max().item() / before.abs().max().item())
assert (after - before).abs().max() / before.abs().max() < 0.1

# the delta patched into a copy of the checkpoint files loads as the same model
import apply_delta
apply_delta.main(d, f"{td}/esft_delta.safetensors", f"{td}/out")
with torch.no_grad():
    again = ft.load(f"{td}/out", ["cpu", "cpu"]).eval()(ids)
assert torch.equal(again, after), (again - after).abs().max()
print("patched checkpoint reproduces the exported model")

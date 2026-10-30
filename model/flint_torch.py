"""Recursive's Flint in plain PyTorch, run from the MLX INT4 checkpoint (RecursiveSI/flint-int4), for attention-only LoRA.

Written from the architecture (Locals' locals/models/flint.py, Transformers' modeling_flint): Qwen3-MoE plus
- a learned per-head sink logit in each attention softmax's denominator (softmax_offset);
- sliding layers see the current token and the 512 before it (sliding_window 513) and use RoPE; full layers use none;
- sigmoid routing, top 6, renormalized; an ungated shared expert in every MoE block; layers 0-1 dense SwiGLU.

Every weight stays packed as in the checkpoint (4-bit affine, group 64; the embedding 8-bit) and is dequantized in
the forward pass, so the 40B model fits two 16 GB T4s. T4 has no bf16: matmuls run in fp16, the residual stream,
norms, router, attention softmax and LoRA in fp32.
"""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

GROUP = 64
CDT = getattr(torch, os.environ.get("FLINT_CDT", "float16"))  # matmul dtype (bfloat16 on Ampere+)
SDPA = os.environ.get("FLINT_ATTN", "sdpa") == "sdpa"  # "dense": the original fp32 [L, L] attention (T4, checks)
BMM_MOE = os.environ.get("FLINT_MOE", "bmm") == "bmm"  # "loop": one expert at a time (least memory, T4)


def dequant(w, scales, biases, bits):
    """MLX affine dequant: w is uint8 [..., in*bits/8], scales/biases [..., in/64] -> [..., in] in CDT."""
    if bits == 4:  # byte b holds element 2b in its low nibble, 2b+1 in its high one (MLX packs little-endian)
        q = torch.stack((w & 0xF, w >> 4), dim=-1).flatten(-2)
    else:
        q = w
    q = q.unflatten(-1, (-1, GROUP)).to(CDT)
    return (q * scales.unsqueeze(-1) + biases.unsqueeze(-1)).flatten(-2)


class QLinear(nn.Module):
    """A frozen packed linear layer, optionally with a trainable LoRA (fp32 A/B; B starts at zero)."""

    def __init__(self, bits=4):
        super().__init__()
        self.bits, self.lora_a, self.lora_b = bits, None, None

    def load(self, w, s, b):
        self.register_buffer("w", w, persistent=False)
        self.register_buffer("s", s.to(CDT), persistent=False)
        self.register_buffer("b", b.to(CDT), persistent=False)

    def weight(self):
        return dequant(self.w, self.s, self.b, self.bits)

    def add_lora(self, r, alpha):
        out_f, in_f = self.w.shape[0], self.w.shape[1] * 8 // self.bits
        self.lora_a = nn.Parameter(torch.empty(r, in_f, device=self.w.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.lora_b = nn.Parameter(torch.zeros(out_f, r, device=self.w.device))
        self.lora_scale = alpha / r

    def forward(self, x):
        y = F.linear(x.to(CDT), self.weight()).float()
        if self.lora_a is not None:
            y = y + F.linear(F.linear(x.float(), self.lora_a), self.lora_b) * self.lora_scale
        return y


class RMSNorm(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def load(self, w):
        self.register_buffer("weight", w.float(), persistent=False)

    def forward(self, x):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope(x, theta, pos):  # non-interleaved (rotate-half) RoPE, as MLX nn.RoPE(traditional=False)
    d = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = (pos.float().view(-1, pos.shape[-1])[..., None] * inv).unsqueeze(1)  # pos [L] or per-row [B, L]
    cos, sin = torch.cat((ang, ang), -1).cos(), torch.cat((ang, ang), -1).sin()
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat((-x2, x1), -1) * sin


class Attention(nn.Module):
    def __init__(self, cfg, sliding):
        super().__init__()
        self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.sliding, self.window, self.theta = sliding, cfg["sliding_window"], cfg["rope_theta"]
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = QLinear(), QLinear(), QLinear(), QLinear()
        self.q_norm, self.k_norm = RMSNorm(cfg["rms_norm_eps"]), RMSNorm(cfg["rms_norm_eps"])

    def forward(self, x, gen=None):
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, L, self.nh, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, L, self.nkv, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nkv, self.hd).transpose(1, 2)
        if gen is not None:
            return self.o_proj(self._cached(q, k, v, gen))
        pos = torch.arange(L, device=x.device)
        if self.sliding:
            q, k = rope(q, self.theta, pos), rope(k, self.theta, pos)
        return self.o_proj(self._sdpa(q, k, v, pos) if SDPA else self._dense(q, k, v, pos))

    def _sdpa(self, q, k, v, pos):
        """Memory-efficient attention (no [L, L] matrices; 6k-token training fits). The sink becomes an extra key:
        q gets a constant 1 in a padded dim where the sink key holds sink*sqrt(hd), and its value is 0."""
        B, _, L, _ = q.shape
        rep, P = self.nh // self.nkv, 8
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        qa = F.pad(q, (0, P)).to(CDT)
        qa[..., self.hd] = 1
        sink = F.pad((self.softmax_offset * math.sqrt(self.hd)).view(1, self.nh, 1, 1), (self.hd, P - 1))
        ka = torch.cat((sink.expand(B, -1, -1, -1), F.pad(k, (0, P))), 2).to(CDT)
        va = F.pad(v, (0, P, 1, 0)).to(CDT)  # zero value row for the sink, head dim padded to match
        i, j = pos[:, None], pos[None]
        allowed = (j <= i) & ((i - j < self.window) if self.sliding else True)
        allowed = torch.cat((allowed.new_ones(L, 1), allowed), 1)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            o = F.scaled_dot_product_attention(qa, ka, va, attn_mask=allowed, scale=1 / math.sqrt(self.hd))
        return o[..., :self.hd].float().transpose(1, 2).reshape(B, L, -1)

    def _dense(self, q, k, v, pos):
        B, _, L, _ = q.shape
        rep = self.nh // self.nkv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)  # fp32
        i, j = pos[:, None], pos[None]
        allowed = (j <= i) & ((i - j < self.window) if self.sliding else True)
        scores = scores.masked_fill(~allowed, float("-inf"))
        sink = self.softmax_offset.view(1, self.nh, 1, 1).expand(B, -1, L, 1)
        p = torch.softmax(torch.cat((scores, sink), -1), -1)[..., :L]  # the sink takes probability mass, outputs nothing
        return (p @ v).transpose(1, 2).reshape(B, L, -1)

    def _cached(self, q, k, v, gen):
        """Batched generation over left-padded rows: gen["pos"] [B, L] positions of the new tokens, gen["valid"]
        [B, L] False on padding, gen["cache"][id(self)] this layer's (k, v, key positions, key valid). Sliding layers
        keep only their last `window` keys."""
        B, _, L, _ = q.shape
        pos, valid = gen["pos"].to(q.device), gen["valid"].to(q.device)
        if self.sliding:
            q, k = rope(q, self.theta, pos), rope(k, self.theta, pos)
        k, v = k.to(CDT), v.to(CDT)
        if id(self) in gen["cache"]:
            ck, cv, cp, cm = gen["cache"][id(self)]
            k, v, kp, km = torch.cat((ck, k), 2), torch.cat((cv, v), 2), torch.cat((cp, pos), 1), torch.cat((cm, valid), 1)
        else:
            kp, km = pos, valid
        i, j = pos[:, None, None, :, None], kp[:, None, None, None, :]
        allowed = km[:, None, None, None, :] & (j <= i) & ((i - j < self.window) if self.sliding else True)
        rep = self.nh // self.nkv
        qg = q.to(CDT).view(B, self.nkv, rep, L, self.hd)
        scores = (qg @ k[:, :, None].transpose(-1, -2)).float() / math.sqrt(self.hd)  # [B, nkv, rep, L, T]
        scores = scores.masked_fill(~allowed, float("-inf"))
        sink = self.softmax_offset.view(1, self.nkv, rep, 1, 1).expand(B, -1, -1, L, 1)
        p = torch.softmax(torch.cat((scores, sink), -1), -1)[..., :-1]
        o = (p.to(CDT) @ v[:, :, None]).float().reshape(B, self.nh, L, self.hd).transpose(1, 2).reshape(B, L, -1)
        if self.sliding:
            k, v, kp, km = k[:, :, -self.window:], v[:, :, -self.window:], kp[:, -self.window:], km[:, -self.window:]
        gen["cache"][id(self)] = (k, v, kp, km)
        return o


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = QLinear(), QLinear(), QLinear()

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.k = cfg["num_experts_per_tok"]
        self.shared_expert = MLP()
        self.stats = None  # set to a zero [experts] fp32 tensor to accumulate each expert's routed gate mass
        self.esft = nn.ParameterDict()  # "<e>_<gate|up|down>" -> trainable full-precision weights of expert e

    def _expert(self, e):
        if f"{e}_gate" in self.esft:
            return tuple(self.esft[f"{e}_{n}"].to(CDT) for n in ("gate", "up", "down"))
        return tuple(dequant(getattr(self, f"{n}_w")[e], getattr(self, f"{n}_s")[e], getattr(self, f"{n}_b")[e], 4)
                     for n in ("gate", "up", "down"))

    def make_trainable(self, experts):
        for e in experts:
            for n, t in zip(("gate", "up", "down"), self._expert(e)):
                self.esft[f"{e}_{n}"] = nn.Parameter(t.to(torch.bfloat16))  # bf16 + stochastic-rounding Adam (esft.AdamSR)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        scores = torch.sigmoid(x.float() @ self.router_w.T)  # router in fp32
        w, idx = scores.topk(self.k, -1)
        w = w / w.sum(-1, keepdim=True)
        if self.stats is not None:
            self.stats.index_add_(0, idx.flatten(), w.flatten())
        if len(idx) * self.k <= 384 and not len(self.esft):  # decode: gather the routed experts, three batched matmuls
            fi = idx.flatten()
            g, u, d = (dequant(getattr(self, f"{n}_w")[fi], getattr(self, f"{n}_s")[fi], getattr(self, f"{n}_b")[fi], 4)
                       for n in ("gate", "up", "down"))
            h = x.to(CDT).repeat_interleave(self.k, 0)[:, :, None]
            y = torch.bmm(d, F.silu(torch.bmm(g, h).float()).to(CDT) * torch.bmm(u, h)).float()[..., 0]
            return ((y.view(-1, self.k, shape[-1]) * w[..., None]).sum(1) + self.shared_expert(x)).view(shape)
        flat = idx.flatten()
        order = flat.argsort()
        if BMM_MOE:
            return (self._bmm(x, flat, order, w) + self.shared_expert(x)).view(shape)
        counts = torch.bincount(flat, minlength=self.router_w.shape[0]).tolist()  # one host sync per layer
        toks, ws = order // self.k, w.flatten()[order]
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        xc = x.to(CDT)
        start = 0
        for e, c in enumerate(counts):
            if not c:
                continue
            tok, we = toks[start:start + c], ws[start:start + c, None]
            start += c
            # dequantize each used expert as it runs, so only one expert's fp16 weights exist at a time
            g, u, d = self._expert(e)
            h = xc[tok]
            y = F.linear(F.silu(F.linear(h, g).float()).to(CDT) * F.linear(h, u), d).float()
            out.index_add_(0, tok, y * we)
        return (out + self.shared_expert(x)).view(shape)

    def _bmm(self, x, flat, order, w):
        """All experts at once: dequantize the layer's experts, pad each expert's tokens to the busiest expert's count,
        three batched matmuls. ~10x faster than the per-expert loop on a 48 GB card; costs ~1.5 GB per layer.
        shortcut: padding wastes FLOPs when routing is very uneven; switch to grouped GEMM if that shows in profiles."""
        E = self.router_w.shape[0]
        counts = torch.bincount(flat, minlength=E)
        cap = int(counts.max())  # one host sync per layer
        es = flat[order]
        slot = torch.arange(len(order), device=x.device) - (counts.cumsum(0) - counts)[es]
        toks = order // self.k
        X = x.new_zeros(E, cap, x.shape[-1], dtype=CDT)
        X[es, slot] = x.to(CDT)[toks]
        g, u, d = (dequant(getattr(self, f"{n}_w"), getattr(self, f"{n}_s"), getattr(self, f"{n}_b"), 4) for n in ("gate", "up", "down"))
        if len(self.esft):
            sel = sorted({int(k.split("_")[0]) for k in self.esft.keys()})
            ix = torch.tensor(sel, device=x.device)
            g, u, d = (t.index_copy(0, ix, torch.stack([self.esft[f"{e}_{n}"] for e in sel]).to(CDT))
                       for t, n in zip((g, u, d), ("gate", "up", "down")))
        H = F.silu((X @ g.transpose(1, 2)).float()).to(CDT) * (X @ u.transpose(1, 2))
        Y = (H @ d.transpose(1, 2))[es, slot].float() * w.flatten()[order, None]
        return torch.zeros(x.shape, dtype=torch.float32, device=x.device).index_add_(0, toks, Y)


class Layer(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.self_attn = Attention(cfg, cfg["layer_types"][i] == "sliding_attention")
        self.mlp = MLP() if i in cfg["mlp_only_layers"] else MoE(cfg)
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(cfg["rms_norm_eps"]), RMSNorm(cfg["rms_norm_eps"])

    def forward(self, x, gen=None):
        h = x + self.self_attn(self.input_layernorm(x), gen)
        return h + self.mlp(self.post_attention_layernorm(h))


class Flint(nn.Module):
    def __init__(self, cfg, devices):
        super().__init__()
        self.cfg, n = cfg, cfg["num_hidden_layers"]
        self.layers = nn.ModuleList(Layer(cfg, i) for i in range(n))
        self.norm = RMSNorm(cfg["rms_norm_eps"])
        self.lm_head = QLinear()
        self.embed = QLinear(bits=8)
        # split layers evenly by weight (MoE layers dominate) across devices
        self.dev = [devices[min(len(devices) - 1, i * len(devices) // n)] for i in range(n)]
        self.grad_ckpt = False

    def embed_tokens(self, ids):
        e = self.embed
        return dequant(e.w[ids], e.s[ids], e.b[ids], 8).float()

    def forward(self, ids, hidden=False, gen=None):
        h = self.embed_tokens(ids.to(self.dev[0]))
        for layer, dev in zip(self.layers, self.dev):
            h = h.to(dev)
            h = checkpoint(layer, h, use_reentrant=False) if self.grad_ckpt and self.training else layer(h, gen)
        if gen is not None:
            h = h[:, -1:]  # generation only needs the last position's logits
        h = self.norm(h.to(self.dev[-1]))
        return h if hidden else self.lm_head(h)  # fp32 logits

    def loss(self, ids, chunk=512, mask=None):
        """Next-token cross-entropy, the 150k-vocab logits made chunk tokens at a time (full logits OOM a T4 at 2k).
        mask [1, L]: 1 where ids[t] is a trained target; returns (sum of losses, number of targets)."""
        h, tgt = self(ids[:, :-1], hidden=True)[0], ids[0, 1:].to(self.dev[-1])
        if mask is not None:
            tgt = tgt.masked_fill(mask[0, 1:].to(tgt.device) == 0, -100)
        ce = lambda hc, tc: F.cross_entropy(self.lm_head(hc), tc, reduction="sum", ignore_index=-100)
        total = sum(checkpoint(ce, h[i:i + chunk], tgt[i:i + chunk], use_reentrant=False) for i in range(0, len(tgt), chunk))
        return total, int((tgt != -100).sum())

    def add_lora(self, r=16, alpha=32, targets=("q_proj", "k_proj", "v_proj", "o_proj")):
        for p in self.parameters():
            p.requires_grad_(False)
        for layer in self.layers:
            for t in targets:
                getattr(layer.self_attn, t).add_lora(r, alpha)
        return [p for n, p in self.named_parameters() if "lora_" in n]


def load(path, devices):
    """Build Flint from a flint-int4 folder, placing each layer's tensors straight onto its device."""
    from safetensors import safe_open
    cfg = json.load(open(os.path.join(path, "config.json")))
    m = Flint(cfg, devices)
    pending = {}
    shards = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    for f in shards:
        with safe_open(os.path.join(path, f), framework="pt") as st:
            for key in st.keys():
                _place(m, key, st.get_tensor(key), pending)
    assert not pending, f"incomplete quantized tensors: {list(pending)[:5]}"
    return m


def _place(m, key, t, pending):
    parts = key.split(".")
    if parts[0] == "lm_head" or parts[1] in ("embed_tokens", "norm"):
        mod = m.lm_head if parts[0] == "lm_head" else (m.embed if parts[1] == "embed_tokens" else m.norm)
        dev = m.dev[-1] if mod is not m.embed else m.dev[0]
        attr = parts[-1]
    else:
        i = int(parts[2])
        dev, mod = m.dev[i], m.layers[i]
        rest = parts[3:-1]
        attr = parts[-1]
        if rest[:2] == ["mlp", "switch_mlp"]:  # routed experts: stored as buffers on the MoE block
            setattr_buf(mod.mlp, f"{rest[2].split('_')[0]}_{attr[0]}", t.to(dev) if attr == "weight" else t.to(dev, CDT))
            return
        if rest == ["mlp", "gate"]:
            setattr_buf(mod.mlp, "router_w", t.to(dev, torch.float32))
            return
        if rest == ["self_attn"] and attr == "softmax_offset":
            setattr_buf(mod.self_attn, "softmax_offset", t.to(dev, torch.float32))
            return
        for r in rest:
            mod = getattr(mod, r)
    if isinstance(mod, RMSNorm):
        mod.load(t.to(dev))
        return
    slot = pending.setdefault(id(mod), {})
    slot[attr] = t.to(dev)
    if len(slot) == 3:
        mod.load(slot["weight"], slot["scales"], slot["biases"])
        del pending[id(mod)]


def setattr_buf(mod, name, t):
    mod.register_buffer(name, t, persistent=False)

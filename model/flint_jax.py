"""Flint in JAX for a TPU v5e-8: the same computation as flint_torch.py, written for XLA.

Loads RecursiveSI/flint (BF16, per-expert tensors) or flint-int4 (MLX packed, dequantized at load). Layers are spread
over the devices (a pipeline in one process: each layer's weights live on one chip, the hidden state moves); the
routed experts run with lax.ragged_dot on tokens sorted by expert, so every shape is static. Attention-only LoRA.
"""
import json
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

DT = jnp.bfloat16
GROUP = 64


# ---------------------------------------------------------------- loading
def _dequant(w, s, b, bits):
    w = np.asarray(w, np.uint8)
    q = np.stack((w & 0xF, w >> 4), -1).reshape(*w.shape[:-1], -1) if bits == 4 else w
    q = q.reshape(*q.shape[:-1], -1, GROUP).astype(np.float32)
    return (q * np.asarray(s, np.float32)[..., None] + np.asarray(b, np.float32)[..., None]).reshape(*q.shape[:-2], -1)


def iter_tensors(files):
    """(name, numpy float32/uint8 array) from safetensors files, one file at a time (the caller may delete it after)."""
    from safetensors import safe_open
    cpu = jax.devices("cpu")[0]
    for f in files:
        with jax.default_device(cpu), safe_open(f, framework="flax") as st:
            for k in st.keys():
                t = st.get_tensor(k)
                yield k, np.asarray(t if t.dtype == jnp.uint8 else t.astype(jnp.float32))
        yield None, f  # file finished


def load(cfg, tensors, devices, on_file_done=None):
    """Params: a list of per-layer dicts (each on its device) plus embed / norm / lm_head."""
    n, E = cfg["num_hidden_layers"], cfg["num_experts"]
    bits = {"embed_tokens": 8}
    dev = [devices[i * len(devices) // n] for i in range(n)]
    layers = [dict() for _ in range(n)]
    top, packed, experts = {}, {}, {}

    def put(i, name, arr):
        tgt = layers[i] if i is not None else top
        d = dev[i] if i is not None else (devices[0] if name == "embed" else devices[-1])
        keep32 = name in ("router", "sinks") or name.endswith("norm")
        a = arr.T if name.startswith(("q", "k", "v", "o", "sh_", "d_", "lm_head")) and arr.ndim == 2 and not name.endswith("norm") else arr
        tgt[name] = jax.device_put(np.asarray(a, np.float32 if keep32 else DT), d)  # straight from host, no default-device copy

    def name_of(parts):  # checkpoint path -> our short name
        p = ".".join(parts)
        m = {"self_attn.q_proj": "q", "self_attn.k_proj": "k", "self_attn.v_proj": "v", "self_attn.o_proj": "o",
             "self_attn.q_norm": "q_norm", "self_attn.k_norm": "k_norm", "input_layernorm": "ln1_norm",
             "post_attention_layernorm": "ln2_norm", "mlp.gate": "router", "self_attn.softmax_offset": "sinks",
             "mlp.gate_proj": "d_gate", "mlp.up_proj": "d_up", "mlp.down_proj": "d_down",
             "mlp.shared_expert.gate_proj": "sh_gate", "mlp.shared_expert.up_proj": "sh_up", "mlp.shared_expert.down_proj": "sh_down"}
        return m[p]

    for key, t in tensors:
        if key is None:
            on_file_done and on_file_done(t)
            continue
        parts = key.split(".")
        attr = parts[-1] if parts[-1] in ("weight", "scales", "biases") else "weight"
        base = parts[:-1] if parts[-1] in ("weight", "scales", "biases") else parts
        if attr != "weight" or (t.dtype == np.uint8):  # MLX packed: gather the three parts
            slot = packed.setdefault(tuple(base), {})
            slot[attr] = t
            if len(slot) < 3:
                continue
            del packed[tuple(base)]
            t = _dequant(slot["weight"], slot["scales"], slot["biases"], bits.get(base[-1], 4))
        if base[0] == "lm_head":
            put(None, "lm_head", t)
        elif base[1] == "embed_tokens":
            put(None, "embed", t)
        elif base[1] == "norm":
            put(None, "final_norm", t)
        else:
            i, rest = int(base[2]), base[3:]
            if rest[:2] == ["mlp", "switch_mlp"]:  # int4: stacked [E, out, in]
                experts[(i, rest[2])] = t
            elif rest[:2] == ["mlp", "experts"]:  # bf16: one expert at a time
                experts.setdefault((i, rest[3]), [None] * E)[int(rest[2])] = t
            else:
                put(i, name_of(rest), t)
                continue
            k = (i, rest[2] if rest[1] == "switch_mlp" else rest[3])
            v = experts[k]
            if isinstance(v, list) and any(x is None for x in v):
                continue
            w = np.stack(v) if isinstance(v, list) else v  # [E, out, in]
            del experts[k]
            # ragged_dot wants [E, in, out]
            put(i, {"gate_proj": "e_gate", "up_proj": "e_up", "down_proj": "e_down"}[k[1]], np.swapaxes(w, 1, 2))
    assert not packed and not experts, (list(packed)[:3], list(experts)[:3])
    return {"layers": layers, "top": top, "dev": dev, "last": devices[-1]}


# ---------------------------------------------------------------- model
def rms(x, w, eps):
    x = x.astype(jnp.float32)
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps) * w


def rope(x, theta):  # [B, H, L, D], rotate-half
    L, d = x.shape[-2], x.shape[-1]
    inv = 1.0 / theta ** (jnp.arange(0, d, 2, dtype=jnp.float32) / d)
    ang = jnp.arange(L, dtype=jnp.float32)[:, None] * inv[None]
    ang = jnp.concatenate([ang, ang], -1)
    rot = jnp.concatenate([-x[..., d // 2:], x[..., : d // 2]], -1)
    return x * jnp.cos(ang) + rot * jnp.sin(ang)


def lin(x, w, lora=None, scale=2.0):  # w is [in, out]
    y = jnp.matmul(x.astype(DT), w, preferred_element_type=jnp.float32)
    if lora is not None:
        y = y + (x.astype(jnp.float32) @ lora["a"].T) @ lora["b"].T * scale
    return y


def layer_fn(p, lora, h, cfg, sliding, moe):
    B, L, D = h.shape
    nh, nkv, hd, eps = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"]
    x = rms(h, p["ln1_norm"], eps)
    lo = lora or {}
    q = rms(lin(x, p["q"], lo.get("q")).reshape(B, L, nh, hd), p["q_norm"], eps).transpose(0, 2, 1, 3)
    k = rms(lin(x, p["k"], lo.get("k")).reshape(B, L, nkv, hd), p["k_norm"], eps).transpose(0, 2, 1, 3)
    v = lin(x, p["v"], lo.get("v")).reshape(B, L, nkv, hd).transpose(0, 2, 1, 3)
    if sliding:
        q, k = rope(q, cfg["rope_theta"]), rope(k, cfg["rope_theta"])
    k, v = jnp.repeat(k, nh // nkv, 1), jnp.repeat(v, nh // nkv, 1)
    s = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hd)
    i, j = jnp.arange(L)[:, None], jnp.arange(L)[None]
    ok = (j <= i) & ((i - j < cfg["sliding_window"]) if sliding else True)
    s = jnp.where(ok, s, -jnp.inf)
    sink = jnp.broadcast_to(p["sinks"][None, :, None, None], (B, nh, L, 1))
    a = jax.nn.softmax(jnp.concatenate([s, sink], -1), -1)[..., :L]
    o = (a @ v.astype(jnp.float32)).transpose(0, 2, 1, 3).reshape(B, L, -1)
    h = h + lin(o, p["o"], lo.get("o"))
    x = rms(h, p["ln2_norm"], eps)
    if not moe:
        return h + lin(jax.nn.silu(lin(x, p["d_gate"])) * lin(x, p["d_up"]), p["d_down"])
    xf = x.reshape(-1, D)
    k_ = cfg["num_experts_per_tok"]
    scores = jax.nn.sigmoid(xf @ p["router"].T)  # fp32 router
    w, idx = jax.lax.top_k(scores, k_)
    w = w / w.sum(-1, keepdims=True)
    flat = idx.reshape(-1)
    order = jnp.argsort(flat)
    tok = order // k_
    gs = jnp.bincount(flat, length=cfg["num_experts"]).astype(jnp.int32)
    xs = xf[tok].astype(DT)
    g = jax.lax.ragged_dot(xs, p["e_gate"], gs, preferred_element_type=jnp.float32)
    u = jax.lax.ragged_dot(xs, p["e_up"], gs, preferred_element_type=jnp.float32)
    y = jax.lax.ragged_dot((jax.nn.silu(g) * u).astype(DT), p["e_down"], gs, preferred_element_type=jnp.float32)
    y = y * w.reshape(-1)[order][:, None]
    routed = jnp.zeros_like(xf).at[tok].add(y)
    shared = lin(jax.nn.silu(lin(xf, p["sh_gate"])) * lin(xf, p["sh_up"]), p["sh_down"])
    return h + (routed + shared).reshape(B, L, D)


class Flint:
    def __init__(self, cfg, params):
        self.cfg, self.p = cfg, params
        self.kinds = [(cfg["layer_types"][i] == "sliding_attention", i not in cfg["mlp_only_layers"]) for i in range(cfg["num_hidden_layers"])]
        self._fns = {}

    def fn(self, kind, ckpt):
        if (kind, ckpt) not in self._fns:
            f = partial(layer_fn, cfg=self.cfg, sliding=kind[0], moe=kind[1])
            self._fns[(kind, ckpt)] = jax.jit(jax.checkpoint(f) if ckpt else f)
        return self._fns[(kind, ckpt)]

    def hidden(self, ids, loras=None, ckpt=False):
        top = self.p["top"]
        h = top["embed"][ids].astype(jnp.float32)
        for i, kind in enumerate(self.kinds):
            h = jax.device_put(h, self.p["dev"][i])
            h = self.fn(kind, ckpt)(self.p["layers"][i], loras[i] if loras else None, h)
        h = jax.device_put(h, self.p["last"])
        return rms(h, top["final_norm"], self.cfg["rms_norm_eps"])

    def logits(self, ids):
        return _head(self.hidden(ids), self.p["top"]["lm_head"])

    def loss(self, loras, ids, chunk=512):
        h, tgt = self.hidden(ids[:, :-1], loras, ckpt=True)[0], ids[0, 1:]
        tgt = jax.device_put(tgt, self.p["last"])
        tot = sum(_ce(h[i:i + chunk], tgt[i:i + chunk], self.p["top"]["lm_head"]) for i in range(0, tgt.shape[0], chunk))
        return tot / tgt.shape[0]

    def init_lora(self, r=16, alpha=32, seed=0):
        key, out = jax.random.PRNGKey(seed), []
        for i, p in enumerate(self.p["layers"]):
            d, lo = self.p["dev"][i], {}
            for n in ("q", "k", "v", "o"):
                key, sub = jax.random.split(key)
                fin, fout = p[n].shape
                bound = 1 / np.sqrt(fin)  # kaiming_uniform(a=sqrt 5), as PEFT
                lo[n] = {"a": jax.device_put(jax.random.uniform(sub, (r, fin), jnp.float32, -bound, bound), d),
                         "b": jax.device_put(jnp.zeros((fout, r), jnp.float32), d)}
            out.append(lo)
        assert alpha / r == 2.0  # lin() uses scale 2.0
        return out


@jax.jit
def _head(h, w):
    return jnp.matmul(h.astype(DT), w, preferred_element_type=jnp.float32)


@partial(jax.jit)
@jax.checkpoint
def _ce(h, t, w):
    lg = jnp.matmul(h.astype(DT), w, preferred_element_type=jnp.float32)
    return -(jax.nn.log_softmax(lg, -1)[jnp.arange(t.shape[0]), t]).sum()


def adamw(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8, step=1):
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state[0], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state[1], grads)
    new = jax.tree.map(lambda p, m, v: p - lr * (m / (1 - b1 ** step)) / (jnp.sqrt(v / (1 - b2 ** step)) + eps), params, m, v)
    return new, (m, v)

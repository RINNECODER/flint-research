"""Tiny random Flint built with Locals' MLX reference (locals/models/flint.py) on the CPU, quantized like flint-int4
(4-bit affine, embedding 8-bit, router kept), saved in the checkpoint's layout with its logits. Run with Locals' venv."""
import json, sys, os
import numpy as np
import mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_flatten
mx.set_default_device(mx.cpu)
sys.path.insert(0, os.environ.get("LOCALS_SRC", "Locals"))  # a Locals checkout with patches/locals applied
from locals.models.flint import Model, ModelArgs
out = sys.argv[1]
cfg = dict(model_type="flint", hidden_size=128, num_hidden_layers=5, intermediate_size=256, num_attention_heads=4,
           num_key_value_heads=2, head_dim=64, num_experts=8, num_experts_per_tok=3, moe_intermediate_size=64,
           shared_expert_intermediate_size=128, mlp_only_layers=[0, 1], sliding_window=5, rms_norm_eps=1e-6, vocab_size=320,
           layer_types=["full_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention"],
           rope_theta=10000.0, tie_word_embeddings=False)
mx.random.seed(0)
m = Model(ModelArgs(**cfg))
for l in m.layers:  # make the sinks and norms non-trivial
    l.self_attn.softmax_offset = mx.random.normal((4,))
    for n in (l.input_layernorm, l.post_attention_layernorm, l.self_attn.q_norm, l.self_attn.k_norm):
        n.weight = 1 + 0.3 * mx.random.normal(n.weight.shape)
nn.quantize(m, 64, 4, class_predicate=lambda p, mod: hasattr(mod, "to_quantized") and not p.endswith("mlp.gate") and "embed" not in p)
nn.quantize(m, 64, 8, class_predicate=lambda p, mod: "embed_tokens" in p)
ids = mx.array(np.random.RandomState(0).randint(0, 320, (1, 16)))
logits = m(ids)
os.makedirs(out, exist_ok=True)
w = {k: np.array(v.astype(mx.float32) if v.dtype != mx.uint32 else v) for k, v in tree_flatten(m.parameters())}
w = {k: (v.view(np.uint8) if v.dtype == np.uint32 else v) for k, v in w.items()}
from safetensors.numpy import save_file
save_file(w, f"{out}/model-00000.safetensors")
json.dump(cfg, open(f"{out}/config.json", "w"))
np.save(f"{out}/ids.npy", np.array(ids)); np.save(f"{out}/logits.npy", np.array(logits.astype(mx.float32)))
print("saved", len(w), "tensors", logits.shape)

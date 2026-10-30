# Architecture and the int4 layout

## Flint in one table

The values come from `model/ref/int4_config.json`, a copy of the int4 release's `config.json`.

| | value |
|---|---|
| layers | 49 (0-48). Layers 0-1 are dense SwiGLU (intermediate 6144); layers 2-48 are MoE (47 MoE layers) |
| hidden size | 2048 |
| attention | 32 query heads and 4 KV heads (GQA 8:1), head_dim 128, no bias |
| attention sink | a learned per-head logit added to each softmax denominator (`softmax_type: learnable`) |
| layer pattern | 14 full-attention layers (0, 1, 5, 9, 13, …, 45, 48) with **no RoPE**; the other 35 are sliding-window with RoPE θ=10,000 |
| sliding window | 513: the current token plus the 512 before it |
| routed experts | 256 per MoE layer, SwiGLU with intermediate size 512 |
| routing | sigmoid scores; the top 6 are renormalized to sum to 1 (`norm_topk_prob`). There is no router bias and no aux loss |
| shared expert | one per MoE layer, intermediate size 1024, **ungated** (always added) |
| vocab | 150,144; `<|end_of_text|>` = 150001 (also pad), BOS 150000 |
| context | 262,144 positions |

Total ≈ 40B params, ≈ 2.5B active per token (6 of 256 routed experts plus the shared expert and attention).

## The int4 checkpoint (RecursiveSI/flint-int4, and dima6312's abliterated "Rec 4-bit" build)

- MLX affine quantization: `w ≈ scales * q + biases` per group of 64 input columns. Linear layers are 4-bit; the
  embedding is 8-bit; norms and the router are full precision.
- A layer's routed experts are stacked along dim 0 of nine tensors:
  `mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}`, each with shape `[256, out, in/8 or in/64]`.
  The router is `mlp.gate.weight` `[256, 2048]`.
- `model/ref/hdr0.json` and `hdr4.json` are the safetensors headers of two shards, kept for offsets and dtypes.
  `bf16hdr.json` is a bf16 shard header. These were used to write deltas byte for byte.

## `model/flint_torch.py`

This is a from-scratch PyTorch Flint that runs directly from the packed int4 checkpoint.

- Weights stay packed and are dequantized inside the forward pass, so the 40B model fits two 16 GB T4s (Kaggle) or
  one 48 GB A6000 with room for training.
- On a T4 (no bf16), matmuls run in fp16, while the residual stream, norms, router, attention softmax and LoRA run in
  fp32. On an A6000, `FLINT_CDT=bfloat16` sets the compute dtype.
- **Attention.** By default it uses SDPA's memory-efficient kernel, with the sink as an extra key, so 6k-token training
  fits. `FLINT_ATTN=dense` gives the explicit path, which matches within 0.5% relative error and is the only one that
  works on CPU.
- **MoE.** The default `FLINT_MOE=bmm` path uses batched padded matmuls over the selected experts. Decode with more than
  64 rows (N·6 > 384) falls back to it from the gather path.
- **Trainable parts.** LoRA can sit on q/k/v/o and on the shared expert (`sft.py`). Selected routed experts can be
  un-packed to bf16 and trained (`esft.py`). Everything else stays int4.
- **Export.** Trained weights are re-quantized with an MSE scale search into the checkpoint's own keys.
  `train/apply_delta.py` then patches only those byte ranges into a copy of the checkpoint (headers, offsets and dtypes
  unchanged), so the result loads in Locals unchanged.

Parity:

- **Tiny model.** `tests/make_tiny_mlx.py` builds a tiny random Flint with Locals' MLX model (`locals/models/flint.py`),
  quantizes it like flint-int4, and saves its logits. `tests/test_tiny.py` must match them: max relative error 0.003,
  100% top-1 agreement.
- **Full model.** `model/parity/get_reference.py` stored greedy continuations and logprobs from Locals for 3 prompts
  (`reference.json`). The Kaggle T4 kernel reproduced them with 96.9% top-1 agreement over 32 tokens each, and a mean
  logprob gap below 0.01 (`results/kaggle/t4_result.json`).

## `model/flint_jax.py`

This is a JAX port of the same forward pass, written for a Kaggle TPU v5e-8. It has a tiny CPU test
(`tests/test_tiny_jax.py`, which needs jax installed; it was not re-run for this repo). The TPU proof kernel never got a TPU: it stayed in Kaggle's queue for more
than 24 h (`results/kaggle/tpu_watch.log`).

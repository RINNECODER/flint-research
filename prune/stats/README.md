# Flint expert calibration stats

Per-layer, per-routed-expert statistics of dima6312/Flint-40B-A2.5B-abliterated-Rec-4bit (Rec int4) on coding / coding-agent text, collected 2026-10-29 in-process with Locals' streamed Flint
(`prune/run.py stats`). For pruning (REAP) and for Expert-Specialized Fine-Tuning.

## Files

- `stats_256experts.npz`: the full 256-expert model. Use this one for ESFT.
- `stats_reap224.npz`, `stats_reap192.npz`: the same stats on the pruned 224 and 192 variants. Expert axis = position
  in that variant; map it to the original ids with `../masks/keep_reap224.json` / `../masks/keep_reap192.json` (`keep_ids[layer-2][j]`).
- `../masks/keep_reap{224,192,160}.json`: the kept expert ids (original 0-255 numbering), one list per MoE layer 2..48.

## Arrays (npz)

Axis 0 = MoE layer index 0..46 = model layers 2..48 (`layers`). Axis 1 = expert.

| key | shape | meaning |
|---|---|---|
| `count` | (47, E) | routing slots: tokens that selected the expert in their top-6 |
| `sum_gate` | (47, E) | sum of the normalized gate weight (sigmoid scores renormalized over the top-6) |
| `sum_reap` | (47, E) | sum of gate weight x L2 norm of the expert's output; REAP score = `sum_reap / count` |
| `sum_norm` | (47, E) | sum of the expert-output L2 norm |
| `sum_sigmoid` | (47, E) | sum of the raw sigmoid router score (before renormalization) |
| `count_by_source` | (3, 47, E) | `count` split by data source (order in `sources`) |
| `tokens_by_source` | (3,) | tokens per source; each token fills 6 slots per layer |
| `sources` | (3,) | `code`, `swe-agent`, `openhands` |

Routing frequency = `count / (6 * tokens_by_source.sum())`; mean gate = `sum_gate / count`.

## Calibration data (217,606 tokens; public HF datasets via the datasets-server rows API)

- `code`: codeparrot/codeparrot-clean-valid rows 0-399 (Python source), 73,314 tokens.
- `swe-agent`: nebius/SWE-agent-trajectories rows 0-59 (multi-turn agent transcripts with shell tool use), 70,428 tokens.
- `openhands`: SWE-Gym/OpenHands-SFT-Trajectories split train.success.oss rows 0-39 (agent transcripts with function
  calls), 73,864 tokens.

Chats rendered with the model's own chat template; windows of up to 4,096 tokens, one forward pass each, no BOS.
Compute: bf16 activations, Locals ExpertStore (streamed experts), router logits in float32 as Locals computes them.
Held-out evaluation used disjoint rows (code 30000+, swe-agent 60000+, openhands 420+).

Also published as an HF dataset: [RinneCODER/flint-expert-routing-stats](https://huggingface.co/datasets/RinneCODER/flint-expert-routing-stats).

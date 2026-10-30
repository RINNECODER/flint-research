# Training runs

**Summary.** Two coding fine-tunes ran: ESFT and SFT LoRA. Both made Flint worse on HE+/MBPP+. A third step, on-policy
DPO, was fully prepared but never ran, because the compute budget ran out. That is the only reason the work stopped.
The negative results are **single runs with fixed hyperparameters**: they show these recipes failed, not that the
direction is closed.

Compute: about **$6.8 of paid GPU time in total on Lium** (RTX A6000 at $0.42/h). ESFT cost ≈ $3.4, SFT ≈ $2.6, and the
rest went on probes and a pod that died. Kaggle's free T4s were used for the proof of concept. The base for all runs was
`dima6312/Flint-40B-A2.5B-abliterated-Rec-4bit`, snapshot c0d2388f.

## 0. Proof of concept on Kaggle (free)

- **Script:** `kaggle/t4/` runs `model/flint_torch.py` with attention LoRA on 2× T4 16 GB.
- **Parity:** 96.9% top-1 agreement with Locals over 32 tokens, on three prompts.
- **Speed:** a 2k-token step took 39 s, which projects to ~54 GPU-hours per 5k examples. That is too slow for real
  runs on the free tier, so the real runs moved to a rented A6000.
- **TPU:** the JAX/TPU twin (`kaggle/tpu/`) never got a TPU slot; see [compute.md](compute.md).

## 1. ESFT on R1 competitive-programming traces: don't ship

**Method.** Expert-Specialized Fine-Tuning (DeepSeek, 2024): train only the routed experts that the task actually uses.

- `train/esft.py stats` sums routed gate mass per expert over the training data (301,827 tokens). With cover 0.2, it
  keeps the experts that hold the top 20% of each layer's gate mass. That gave **593 experts (1.87B params)**
  (`results/esft/stats.json`).
- Training unpacks those experts to bf16, uses AdamSR (bf16 moments with stochastic rounding, because fp32 Adam does not
  fit), and gradient-checkpoints per layer.
- **Data:** `data/v1`, 2,144 examples and 4.4M tokens.
  - Sources: DeepSeek-R1 traces from [nvidia/OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning)
    and [open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots).
  - Each trace was **verified by running its code** on public and private tests.
  - 27% are wrapped as synthetic `run_python` tool-use trajectories.
  - Decontaminated against HE+/MBPP+ with 13-grams.
- **Run:** batch 1 × accumulation 8, LR 1e-5 cosine with 10 warmup steps, 1 epoch = 268 steps, 4 h 55 min. Peak VRAM was
  41.6 of 47.4 GB.
  - Train loss: 0.95 → 0.58.
  - Val NLL: 0.956 → 0.529 (bf16 experts, before re-quantizing).

**Result (n=96, paired):**

| | base | ESFT | diff (95% CI) | wins/losses | McNemar p |
|---|---|---|---|---|---|
| plus tests | 83.3% | 81.3% | −2.1 pp [−8.3, +4.2] | 4/6 | 0.75 |
| base tests | 92.7% | 89.6% | −3.1 pp [−8.3, +1.0] | 1/4 | 0.38 |

Mean reply length went from 972 to 1,513 tokens (+55%). Replies hitting the 4,096-token cap went from 6 to 8. The model
learned R1's long "Okay, let's see…" style. It coded no better, and it took longer to answer.

Files: `results/esft/eval_{base,tuned}.jsonl` and `.scores.json`. The checkpoint was lost when the pod died (see
[compute.md](compute.md)); it was not worth re-running.

## 2. SFT LoRA on frontier-model traces: don't ship

**Method.** `train/sft.py`: LoRA with r 32 and α 64 on q/k/v/o and on the shared expert (the dense MLP in layers 0-1).
That is 343 tensors and 42.7M params. The routed experts stay frozen int4.

- **Data:** `data/v2`, 2,053 examples and 5.8M tokens, with a 6,144-token cap.
  - About 1,020 agentic tool-use trajectories, attributed to Claude Fable 5, from public HF sets: DSFFGFG456,
    armand0e, AlinCiocan and TeichAI Cursor. They are mapped onto one bash/read/write/edit/grep/glob toolset.
  - 1,033 coding-chat answers from Opus 4.6/4.7 (angrygiraffe).
  - The candidate review is in `data/v2/SOURCES.md`.
- **Run:** AdamW fp32, LR 1e-4 cosine to a 10% floor, accumulation 8. It used 85% of train (1,745 rows, ~5M tokens) for
  218 steps in 307 min. Peak VRAM was 40.1 GB. Attention ran on SDPA mem-efficient; dense attention does not fit 6k tokens.
- **Val NLL:** 1.462 → 1.062 (bf16 LoRA) → **1.093 after merging and re-quantizing to int4**.
  - Re-quantizing kept 76% of the delta and 92% of the NLL gain (`results/sft/export_int4.json`).

**Result (n=96, paired):**

| | base | SFT | diff (95% CI) | wins/losses | McNemar p |
|---|---|---|---|---|---|
| plus tests | 83.3% | **72.9%** | −10.4 pp [−17.7, −3.1] | 2/12 | 0.013 |
| base tests | 92.7% | 80.2% | −12.5 pp [−19.8, −6.3] | 0/12 | 0.0005 |

Mean reply length went from 972 to 524 tokens, and thinking shrank to ~600-800 characters. 3 of the 12 plus-losses are a
missing `from typing import`; the rest are real logic errors.

**Why.** The agentic traces rarely contain thinking: Claude Code logs redact it, and only 178 of 1,020 have a `<think>`.
The chat set's `<think>` blocks are short and synthetic. The model learned to think less, and it coded worse.

Files: `results/sft/` (train log and history, export stats, eval generations and scores).

## Lessons

1. **Lower NLL did not mean better code.** In both runs, validation NLL on the teacher's data fell a lot while pass@1
   fell. Use a task eval as the gate, never NLL on teacher traces.
2. **Style transfers first.** Reply length moved in the direction of the teacher in both runs (+55% with R1, −46% with
   short-thinking traces). For a reasoning model, the amount of thinking in the data matters as much as its correctness.
3. **A strong base has more to lose.** Flint already scores 83% plus / 93% base here. Off-policy imitation of public
   traces pulled it toward other models' habits.
4. **Re-quantizing to int4 erases small deltas.** Re-quantizing keeps only about 4% of deltas smaller than 3% of the
   weight norm. `esft.quantize(..., bits=8)` keeps about 100% of them. Measure the post-quantization model, not the
   bf16 one.
5. **Hence on-policy.** The next experiment was meant to train only on the model's own outputs, so that no foreign
   style gets in.

## 3. On-policy DPO from the base model: prepared, never run

`train/dpo/` holds the prepared run; full details are in [train/dpo/README.md](../train/dpo/README.md).

- **Problems:** 600 Codeforces stdin/stdout problems from `open-r1/codeforces-cots`, each with 3-8 tests that the
  reference solution passes.
- **Pipeline:**
  1. Rollouts: 4 samples per problem at temperature 0.8, with continuous batching.
  2. Scoring by execution.
  3. Pairs: chosen = the shortest uncapped passing reply; rejected = a failing reply, or a passing one that is capped or
     at least 1.5× longer.
  4. DPO (β 0.1, LR 5e-7, RPO 0.2) on the 593 ESFT experts, with resumable state.
- `score.py`, `pairs.py` and `remap_delta.py` were checked on CPU. The GPU bench (`pod_queue.sh`) never ran.
- **Estimated cost:** about $3 and ~7 A6000-hours for rollouts, reference log-probs, training and eval.
- The plan was first to run DPO on top of the SFT model. After SFT failed, the next step became DPO from the base model
  (`<sft_delta>` = `none`).

## Shipping a delta

`apply_delta.py <base folder> <delta> <out folder>` writes the patched checkpoint byte for byte. For the REAP-224 model,
first remap the expert ids with `train/dpo/remap_delta.py` and `prune/masks/keep_reap224.json`. Then pack the model for
Locals with `python -m locals.models.pack`.

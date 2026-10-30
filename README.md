# flint-research

**Paper:** [Running, Pruning and Trying to Fine-Tune a 40B MoE on a 24 GB Mac](paper/flint-report.pdf) (technical report; LaTeX source in [paper/](paper/))

Research notes, code and results from about a week (2026-10-28 to 10-30) of work on **Flint**, a 40B-parameter
mixture-of-experts model, on a 24 GB Apple-silicon Mac and on cheap rented GPUs. The work covers three things:

1. **Running it fast on a 24 GB Mac**, with experts streamed from SSD: 10 → ~36 tok/s.
2. **Pruning experts with REAP**: REAP-224 is published on Hugging Face. It is 12% smaller and about 14% faster,
   and it lost one task of 40 on a small HumanEval+ test (36 → 35), which is within noise.
3. **Tuning it for coding**: ESFT and LoRA SFT. **Both runs made the model worse.** That result is reported here as
   plainly as the wins.

The project is finished. Nobody runs this model any more, and the repo is a record of the work. Nothing here is maintained.

- Published model: [RinneCODER/Flint-40B-A2.5B-abliterated-REAP224-Rec-4bit](https://huggingface.co/RinneCODER/Flint-40B-A2.5B-abliterated-REAP224-Rec-4bit)
- Expert routing stats and masks: [RinneCODER/flint-expert-routing-stats](https://huggingface.co/datasets/RinneCODER/flint-expert-routing-stats) (also in `prune/stats/` and `prune/masks/`)

## What Flint is

[RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) is a 40B-A2.5B MoE: a Qwen3-MoE-style decoder with
these properties:

- **Layers:** 49 layers. Layers 0-1 are dense SwiGLU; layers 2-48 are MoE.
- **Experts:** 256 routed experts per MoE layer (intermediate size 512), with sigmoid routing, the top 6 renormalized,
  and one ungated shared expert (1024).
- **Attention:** hidden size 2048, 32 query heads and 4 KV heads with head_dim 128. Each head has a learned sink logit
  in the softmax denominator.
- **Sliding and full layers:** 35 sliding-window layers (513 tokens, RoPE) and 14 full-attention layers (no RoPE):
  layers 0, 1, 5, 9, …, 45 and 48.
- **Vocabulary:** 150,144 tokens.

All work here used the int4 release layout: MLX affine 4-bit with group size 64 and an 8-bit embedding. Most of it ran
on [dima6312/Flint-40B-A2.5B-abliterated-Rec-4bit](https://huggingface.co/dima6312/Flint-40B-A2.5B-abliterated-Rec-4bit),
an abliterated build with the same layout. See [docs/architecture.md](docs/architecture.md).

## Results at a glance

### Speed on a 24 GB M5 Pro Mac (experts streamed from SSD, Locals)

| step | decode tok/s | what changed |
|---|---|---|
| stock streaming | ~10 | [Locals](https://github.com/Generated-Computers/Locals) streams the int4 experts from SSD |
| + expert pack | ~22 | `python -m locals.models.pack`: one contiguous file per layer, so each expert is a single read |
| + GPU expert cache | ~28 | `LOCALS_EXPERT_CACHE_GB=6 --cache-gb 2` (≈7.8 GB resident) |
| + DFlash 2 drafts | ~34-36 | [incoai/Flint-DFlash2](https://huggingface.co/incoai/Flint-DFlash2) block drafter, 2 tokens per pass ([patches/locals](patches/locals)) |
| + REAP-224 | **36.8** | 32 of 256 experts per layer removed (+14% over 256 experts with DFlash 2, measured 32.3 → 36.8) |

These are prose and coding prompts at temperature 0 with 1,024 tokens. Prompt-lookup drafts reach ~46 tok/s on agent
file rewrites, where the output copies the input. The steps were measured on different days, so the early rows are
approximate. See [docs/streaming-and-drafting.md](docs/streaming-and-drafting.md).

### Expert pruning (abliterated Flint, coding and coding-agent text)

| variant | experts kept | KL from 256 ↓ | top-1 agree ↑ | HumanEval+ pass@1 (n=40) | tok/s DFlash on / off |
|---|---|---|---|---|---|
| original | 256 | — | — | 36/40 | 32.3 / 27.4 (mean of 2 sessions) |
| **REAP-224** (published) | 224 | **0.92** | **78.0%** | **35/40** | **36.8** / 30.3 |
| REAP-224, sum score | 224 | 0.90 | 78.0% | not run | — |
| frequency-224 | 224 | 0.98 | 75.4% | not run | — |
| RESMP RCO-224 mask | 224 | 1.49 | 68.3% | not run | — |
| random-224 | 224 | 1.47 | 68.6% | not run | — |
| REAP-192 | 192 | 1.21 | 73.7% | 32/40 | 36.5 / 31.6 |
| REAP-160 | 160 | 1.45 | 69.9% | 29/40 | 38.3 / 33.1 |

KL and top-1 are measured on 32 held-out windows of 1,024 tokens. The noise floor is not zero: re-running the
256-expert model with chunked prefill alone gives KL 0.42 and 85.8% top-1 agreement.
The published [RESMP RCO-224](https://huggingface.co/RESMP-DEV/Flint-40B-A2.5B-RCO-224-MLX-GPTQ-MXFP4) mask did no
better than random on this coding text. See [docs/pruning.md](docs/pruning.md).

### Coding fine-tunes: both failed

The eval runs greedy with thinking on and a 4,096-token cap, on a fixed seed-0 sample of 48 HumanEval+ and 48 MBPP+
problems (n=96), paired against the same base model. See [docs/eval.md](docs/eval.md).

**ESFT** (Expert-Specialized Fine-Tuning): the top 593 coding experts (1.87B params) trained on 2,144 verified DeepSeek-R1
competitive-programming traces, 1 epoch.

| n=96 | base | ESFT | diff (95% CI) | wins/losses | McNemar p |
|---|---|---|---|---|---|
| plus tests | 83.3% | 81.3% | −2.1 pp [−8.3, +4.2] | 4/6 | 0.75 |
| base tests | 92.7% | 89.6% | −3.1 pp [−8.3, +1.0] | 1/4 | 0.38 |
| mean reply | 972 tok | 1,513 tok | **+55%** | | |

**SFT LoRA** (r32 on attention + the shared expert): trained on 2,053 frontier-model agent and chat traces (Claude Fable 5 /
Opus 4.6-4.7 public sets).

| n=96 | base | SFT | diff (95% CI) | wins/losses | McNemar p |
|---|---|---|---|---|---|
| plus tests | 83.3% | **72.9%** | −10.4 pp [−17.7, −3.1] | 2/12 | 0.013 |
| base tests | 92.7% | 80.2% | −12.5 pp [−19.8, −6.3] | 0/12 | 0.0005 |
| mean reply | 972 tok | 524 tok | −46% | | |

**Lessons**

- **NLL fell while pass@1 fell.** ESFT validation NLL went from 0.956 to 0.529, and SFT validation NLL from 1.462 to
  1.093. The model learned the teacher's text well and still coded no better, or worse. Validation loss on the
  teacher's traces is not a proxy for capability here.
- **The teacher's style transfers before its skill does.** R1 traces made Flint think 55% longer. Short-thinking
  frontier traces (mostly synthetic `<think>` or redacted thinking) made it think about half as much, and it got 10
  points worse.
- **Base Flint is already strong at this size** (83% plus, 93% base on HE+/MBPP+). Off-policy SFT on public traces
  has more room to hurt than to help.
- **Re-quantizing to int4 erases small deltas.** Re-quantizing kept 76% of the LoRA delta and 92% of its NLL gain.
  int4 keeps only about 4% of deltas smaller than 3% of the weight norm.

**Why the tuning stopped.** It stopped because the compute budget ran out, not because the approach was shown to be a
dead end. The planned next step was **on-policy DPO from the base model**: the model's own shortest passing answer
against its failing or over-long answers, on 600 test-verified Codeforces problems. Its prep is complete in
[train/dpo/](train/dpo/), but it never ran. It would have cost about **$3 and ~7 A6000-hours**. Total paid compute for the
whole project was about **$6.8 on Lium**, plus free Kaggle. The ESFT and SFT results are **single runs with fixed
hyperparameters**, so they show that these recipes failed, not that tuning Flint for coding can't work.
See [docs/training.md](docs/training.md).

## Repository layout

| path | contents |
|---|---|
| `model/` | `flint_torch.py` (PyTorch Flint on packed int4 weights, LoRA, used for ESFT/SFT/DPO), `flint_jax.py` (JAX port for TPU), `ref/` (Transformers reference code, config, safetensors headers), `parity/` (reference logits from Locals) |
| `train/` | `esft.py`, `sft.py`, `apply_delta.py` (patch an int4 delta into a checkpoint byte for byte), `budget_guard.sh`, `pod/probe.sh`, `dpo/` (prepared on-policy DPO) |
| `eval/` | `eval_code.py` (HE+/MBPP+ generate and score), `compare.py` (paired McNemar/bootstrap) |
| `prune/` | REAP pipeline: `fetch.py`, `run.py` (stats / NLL in-process on Locals), `prune.py` (slice experts), `nextmask.py`, `compare.py`, `humaneval.py`, `speed.py`, eval scripts; `masks/` (kept-expert ids), `stats/` (per-expert routing stats) |
| `data/v1/`, `data/v2/` | build scripts and data cards (`DATA.md`, `SOURCES.md`) for the R1 set (ESFT) and the frontier-trace set (SFT). The jsonl files are not included. |
| `kaggle/` | free-tier T4 (PyTorch) and TPU v5e-8 (JAX) proof kernels |
| `tests/` | tiny random Flint built with Locals' MLX model, plus CPU parity, training and round-trip tests |
| `patches/locals/` | 4 patches on Locals `b35c46c`: prompt-lookup drafts, DFlash 2 drafter, width 2, BufferError fix |
| `results/` | eval generations and scores, training histories, pruning logs, KL tables, Kaggle logs |
| `docs/` | deeper write-ups (below) |

Docs: [architecture and int4 layout](docs/architecture.md) · [streaming and drafting](docs/streaming-and-drafting.md) ·
[pruning](docs/pruning.md) · [training runs](docs/training.md) · [eval protocol](docs/eval.md) · [compute notes](docs/compute.md)

## Quickstart

**Run the published REAP-224 model on a Mac** (Apple silicon, 24 GB or more). This uses Locals with the patches:

```bash
git clone https://github.com/Generated-Computers/Locals && cd Locals
git checkout b35c46c && git am /path/to/flint-research/patches/locals/*.patch
# install Locals as its README says, then:
hf download RinneCODER/Flint-40B-A2.5B-abliterated-REAP224-Rec-4bit --local-dir ~/models/flint-reap224
LOCALS_DRAFTER=dflash LOCALS_EXPERT_CACHE_GB=6 locals start --model ~/models/flint-reap224 --cache-gb 2
```

**Check the PyTorch reimplementation on CPU** in seconds, against logits from Locals' MLX model:

```bash
pip install torch safetensors numpy
FLINT_ATTN=dense python tests/test_tiny.py tests/tiny   # CPU has no mem-efficient SDPA kernel, so use the dense path
```

The test checks MLX logit parity, cached generation, LoRA/ESFT training steps, int4 export and the byte-exact
`apply_delta` round trip.

## Reproducing each experiment

| experiment | hardware | time | cost | how |
|---|---|---|---|---|
| speed path | M5 Pro 24 GB | minutes per config | free | [docs/streaming-and-drafting.md](docs/streaming-and-drafting.md) |
| expert stats + REAP masks | same Mac | ~1 min/variant for NLL; stats ~minutes | free | `prune/fetch.py` → `run.py windows` → `run.py stats` → `nextmask.py` → `prune.py` ([docs/pruning.md](docs/pruning.md)) |
| HumanEval+ 40 + speed per variant | same Mac | not timed exactly (40 reasoning replies + 8 speed runs) | free | `prune/evalmodel.sh LABEL MODEL` |
| T4 LoRA proof | Kaggle 2×T4 (free) | ~39 s/step at 2k tokens | free | `kaggle/t4/build.sh` then `kaggle kernels push` ([docs/compute.md](docs/compute.md)) |
| ESFT | 1× RTX A6000 48 GB (Lium) | 4 h 55 min train + ~1 h eval per model | ≈ $3.4 incl. evals | `train/esft.py stats/train/export`, `eval/eval_code.py` ([docs/training.md](docs/training.md)) |
| SFT LoRA | 1× RTX A6000 | 307 min train + ~1 h eval | ≈ $2.6 | `train/sft.py probe/train/export` |
| DPO (never run) | 1× RTX A6000 | ~7 h estimated | ≈ $3 estimated | [train/dpo/README.md](train/dpo/README.md) |

Training data: the build scripts in `data/v1/build` and `data/v2/build` re-download the public source datasets and
rebuild `train.jsonl` / `val.jsonl`. The data cards list every source, revision and license.

## Limitations

- **Small evals.** Pruning used HumanEval+ with 40 tasks; tuning used 96 HE+/MBPP+ items. These catch big regressions,
  not small ones. Long agentic sessions were never benchmarked.
- **Single runs.** ESFT and SFT were each run once, with one learning rate, one data mix and no sweep. DPO never ran.
  The work stopped when the budget ran out.
- **Coding-only calibration.** The REAP masks were chosen on code and agent text, so other domains may lose more.
  Refusal behaviour was not re-tested after pruning.
- **One machine for speed.** All tok/s numbers come from one M5 Pro 24 GB Mac, and SSD streaming speed depends on the
  disk and on memory pressure. The steps of the speed path were measured on different days.
- **Kaggle TPU.** The JAX TPU proof kernel stayed queued for more than a day and never ran. The JAX port is tested only
  on CPU.
- **Not published:** checkpoints (the ESFT checkpoint was lost when a pod died), the training jsonl, and the 50 MB NLL
  reference `nll_256.npz` (rebuild it with `prune/run.py nll`).

## License and credits

Apache-2.0 ([LICENSE](LICENSE)), the same as Flint and Locals. `model/ref/*.py` are from the Hugging Face Transformers
Flint implementation (Apache-2.0, © Recursive and the HuggingFace team). `patches/locals` modify
[Generated-Computers/Locals](https://github.com/Generated-Computers/Locals) (Apache-2.0).

- [Flint](https://huggingface.co/RecursiveSI/flint) by RecursiveSI.
- Abliteration and the Rec 4-bit build by [dima6312](https://huggingface.co/dima6312).
- [Locals](https://github.com/Generated-Computers/Locals) by Generated-Computers: SSD expert streaming, the expert pack, the server.
- DFlash by [z-lab](https://github.com/z-lab/dflash); the Flint drafter [incoai/Flint-DFlash2](https://huggingface.co/incoai/Flint-DFlash2).
- REAP by Lasby et al. (Cerebras), [arXiv:2510.13999](https://arxiv.org/abs/2510.13999).
- ESFT by DeepSeek (Wang et al., 2024). RESMP-DEV for the RCO-224 mask used as a comparison.
- Training data authors: see `data/v1/DATA.md` and `data/v2/DATA.md`.

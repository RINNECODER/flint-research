# Expert pruning (REAP)

Goal: fewer distinct experts per layer, so the SSD streams less and the GPU cache hits more, without losing coding
quality. Everything ran on the Mac, in-process, on Locals' streamed Flint (`prune/run.py`). No GPU rental was needed.

## Calibration and held-out data

`prune/fetch.py` pulls these rows through the HF datasets-server rows API:

| source | dataset | calibration rows | held-out rows |
|---|---|---|---|
| `code` | [codeparrot/codeparrot-clean-valid](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid) | 0-399 (73,314 tokens) | 30000+ |
| `swe-agent` | [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | 0-59 (70,428 tokens) | 60000+ |
| `openhands` | [SWE-Gym/OpenHands-SFT-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories) (train.success.oss) | 0-39 (73,864 tokens) | 420+ |

- **Calibration:** 217,606 tokens in windows of up to 4,096 tokens, rendered with the model's chat template.
- **Held-out:** 32 windows of 1,025 tokens (11 / 11 / 10 per source).

The raw text is not redistributed; `fetch.py` rebuilds it.

## Statistics

`run.py stats MODEL OUT.npz` records these for each MoE layer and expert:

- `count`: routing slots;
- `sum_gate`: the renormalized gate weight;
- `sum_reap`: gate × ‖expert output‖₂;
- `sum_norm`;
- `sum_sigmoid`;
- `count_by_source`.

The arrays are in `prune/stats/` (format in its README) and published as
[RinneCODER/flint-expert-routing-stats](https://huggingface.co/datasets/RinneCODER/flint-expert-routing-stats).

## Masks (`prune/masks/`, `{"keep_ids": [[ids for layer 2], …, [layer 48]]}`, original 0-255 numbering)

| file | rule |
|---|---|
| `keep_reap224.json` | REAP score = `sum_reap / count` (mean of g·‖y‖ over routed tokens); keep the top 224 per layer |
| `keep_sumreap224.json` | top 224 by `sum_reap` (frequency-weighted REAP) |
| `keep_freq224.json` | top 224 by `count` |
| `keep_random224.json` | 224 at random per layer |
| `keep_reap192.json`, `keep_reap160.json` | `nextmask.py`: top-N by REAP, re-scored on the parent variant (224 → 192 → 160) |

The RESMP RCO-224 mask came from
[RESMP-DEV/Flint-40B-A2.5B-RCO-224-MLX-GPTQ-MXFP4](https://huggingface.co/RESMP-DEV/Flint-40B-A2.5B-RCO-224-MLX-GPTQ-MXFP4)
(`pruning-mask.json`). That mask was built for the original Flint, not the abliterated one; it is not copied here.

`prune.py SRC KEEP OUT` slices dim 0 of every `mlp.switch_mlp.*` tensor and the router rows as raw bytes (no
dequantization) and sets `num_experts` in `config.json`. The result keeps the Rec int4 layout.

## Metrics

- `run.py nll MODEL OUT.npz` computes per-token NLL, argmax and (with `REF=nll_256.npz`) the per-token KL(256 ‖ variant)
  on the 32 held-out windows. `compare.py` reports the bootstrap CI over windows.
- HumanEval+, every 4th task (40), runs greedy with full reasoning through the Locals server with DFlash on
  (`humaneval.py` + `score.py`, evalplus base and plus tests).
- Speed is measured with `speed.py` (see [streaming-and-drafting.md](streaming-and-drafting.md)).

## Results

Full output: `results/prune/kl_vs_256.txt`. HumanEval+: `results/prune/<variant>/score.log`. Speed: `speed_{on,off}.log`.

| variant | KL (95% CI) | top-1 agree | NLL vs text | HE+ plus (n=40) | tok/s DFlash on / off | dropped routing mass |
|---|---|---|---|---|---|---|
| 256, chunked prefill only (noise floor) | 0.42 [0.31, 0.54] | 85.8% | 3.287 | — | — | 0 |
| **REAP-224** | **0.92** [0.71, 1.14] | 78.0% | 3.276 | **35/40** | **36.8** / 30.1-30.4 | 10.1% of slots |
| sum-REAP-224 | 0.90 [0.70, 1.12] | 78.0% | 3.275 | — | — | |
| freq-224 | 0.98 [0.79, 1.18] | 75.4% | 3.282 | — | — | 3.5% |
| RESMP RCO-224 | 1.49 [1.25, 1.75] | 68.3% | 3.063 | — | — | 12.3% |
| random-224 | 1.47 [1.21, 1.73] | 68.6% | 2.927 | — | — | 12.6% |
| REAP-192 | 1.21 [0.96, 1.47] | 73.7% | 3.302 | 32/40 | 36.5 / 31.6 | |
| REAP-160 | 1.45 [1.18, 1.74] | 69.9% | 3.259 | 29/40 | 38.3 / 33.1 | |
| 256 (original) | — | — | 3.283 | 36/40 | 30.1-34.5 / 25.4-29.3 | |

Readings:

- **REAP and sum-REAP tie at 224, and both beat frequency.** Frequency is only modestly worse; random and RESMP are much
  worse.
- **RESMP RCO-224 is no better than random on this text.** Its mask was optimized for the original Flint's GPTQ-MXFP4
  build and a different objective and data. It is not evidence against that method on its own target.
- **"NLL vs text" is misleading.** The random and RESMP masks have *lower* NLL against the held-out text while
  diverging most from the base model. The abliterated model is the reference to match, not the text, so KL and top-1
  agreement are the metrics used.
- **192 and 160 trade quality for little speed.** REAP-192 is no faster than 224 with drafts on (acceptance drops
  from 0.68 to 0.61). REAP-160 gains 4% speed for 6 lost HE+ tasks.
- **Speed.** REAP-224 is +14% over 256 with DFlash (32.3 mean → 36.8), ~12% smaller on disk (19 vs 21 GB), and has
  higher draft acceptance (0.68 vs 0.64).

REAP-224 is published as
[RinneCODER/Flint-40B-A2.5B-abliterated-REAP224-Rec-4bit](https://huggingface.co/RinneCODER/Flint-40B-A2.5B-abliterated-REAP224-Rec-4bit).

## Reproduce

```bash
cd prune
PY=/path/to/locals/.venv/bin/python        # the Python Locals is installed in (run.py imports locals)
python fetch.py                             # -> data/{calib,heldout}.jsonl
$PY run.py windows                          # -> data/*_windows.json (token ids)
$PY run.py stats <256 model> stats.npz
$PY run.py nll <256 model> ../results/prune/nll_256.npz
LOCALS_PY=$PY BASE=<256 model> ./ab224.sh reap224 random224   # build, KL-eval and delete each mask
python nextmask.py masks/keep_reap224.json stats_reap224.npz 192 masks/keep_reap192.json
LOCALS_BIN=locals LOCALS_PY=$PY EVAL_PY=<python with evalplus> ./evalmodel.sh reap224 <pruned model>
```

The tokenizer is read from the HF cache of `dima6312/Flint-40B-A2.5B-abliterated-Rec-4bit`, so run
`hf download` for it first.

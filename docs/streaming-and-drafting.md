# Streaming and drafting on a 24 GB Mac

The machine was an M5 Pro MacBook with 24 GB of unified memory. The int4 model is ~21 GB, so it can't sit in GPU memory
next to the OS. The GPU wired limit was left at its default, with no `sudo sysctl iogpu.wired_limit_mb`.

## The stack

[Locals](https://github.com/Generated-Computers/Locals) (Apache-2.0) serves Flint with **routed experts streamed from SSD**.

- Attention, the shared experts, the router and the embeddings stay resident.
- For each token and layer, the 6 selected experts are read from disk (uncached reads) or found in an LRU expert cache
  in GPU memory.
- It exposes an OpenAI-compatible server (`locals start --model … --port 8321`).

## The speed path

| step | tok/s | setting |
|---|---|---|
| stock streaming from the safetensors shards | ~10 | each expert is 9 slices spread across 9 tensors, so 4+ reads |
| expert pack | ~22 | `python -m locals.models.pack <model>` writes `~/.cache/locals/experts/<model>-<hash>/`, one file per layer, laid out expert by expert. One expert = one contiguous read (Locals measured 0.76 ms vs 1.28 ms, and 4.4 vs 2.6 GB/s) |
| GPU expert cache | ~28 | `LOCALS_EXPERT_CACHE_GB=6 locals start … --cache-gb 2` (~7.8 GB total). An 8 GB cache gave 31-32 tok/s, but memory pressure was preferred low |
| DFlash 2 speculative drafts | 30-34.5 (256 experts) | `LOCALS_DRAFTER=dflash` with patches 0002/0003 |
| REAP-224 + DFlash 2 | **36.8** | the published pruned model: fewer distinct experts, so more cache hits |

All measurements use `prune/speed.py`: 2 coding prompts × 2 runs, temperature 0, 1,024 tokens. Decode time runs from
the first streamed token to the last. Results are in `results/prune/*/speed_{on,off}.log`. The base model was measured
in two sessions (25.4 / 29.3 off, 30.1 / 34.5 on), and SSD and thermal state move these numbers by ±10%.

Tried and measured as worse or neutral (keep the defaults):

- a RAM tier for experts (`LOCALS_EXPERT_RAM_GB`): slower;
- page-cache reads instead of `F_NOCACHE`: neutral;
- Locals' guess/read/refine/solo/nax/busy knobs: the defaults were best.

For reference, Locals' own docs give ~138 tok/s for a full-memory 3-bit build. That needs ~20 GB wired, which this
setup avoided.

## Drafting patches (`patches/locals/`, on top of Locals `b35c46c`)

Apply them with `git am patches/locals/*.patch`. They were verified to apply cleanly on `b35c46c`.

1. **0001: prompt-lookup drafts for Flint, streamed included.** A rejected draft's keys stayed in Flint's
   sliding-window ring buffers (`BatchRotatingKVCache`) past `trim()`, because the next one-token step trimmed from the
   ring length, not its index. `_solo_rewind` now cuts them. Before this fix, a rejected draft left stale keys in the
   sliding window. The solo path also failed to recognize `flint_streamed` by module name. `LOCALS_DRAFTS=0` turns this
   off. Prompt lookup gives ~46 tok/s (1.85×) on agent file rewrites, where the output copies the input.
2. **0002: the DFlash 2 drafter (`LOCALS_DRAFTER=dflash`).** This ports z-lab's DFlash MLX model for
   [incoai/Flint-DFlash2](https://huggingface.co/incoai/Flint-DFlash2): a 5-layer, 0.4B bf16 block drafter that proposes
   7 tokens in one pass from Flint's hidden states at layers 5, 15, 26, 36 and 46. FlintModel keeps a tape of those
   states for one-row passes. A request's context is matched against the tape by token. It replaces prompt lookup in
   the solo path for unconstrained rows; grammars still use lookup.
3. **0003: draft width 2 by default (`LOCALS_DFLASH_WIDTH`).** With streamed experts, a verify pass costs ~60 ms plus
   ~20 ms per drafted token (more experts to read), against ~7 ms for the drafter. So 2 tokens per pass beat 7. That
   gives ~1.45× on coding and ~1.3× on prose. Acceptance is 0.64 (256 experts) and 0.68 (REAP-224).
4. **0004: BufferError fix.** A prompt-cache entry reused as a strided KV view raised `BufferError: not C-contiguous`
   in `hybrid_batch._marks`, which surfaced as an HTTP 500. That chunk now skips its rewind marks.

The drafter is lossless at temperature 0: drafts are verified by the full model.

# Eval protocol

Two evals were used. They differ in size, in harness and in what they were used for.

## A. Training evals: `eval/eval_code.py` (GPU, PyTorch)

- **Items.** A fixed seed-0 sample of 48 HumanEval+ and 48 MBPP+ problems (`random.Random(0).sample`), the same for
  every model. The data comes from evalplus.
- **Prompt.** One user message asking for the complete solution in one ```python block. HumanEval shows the signature
  and docstring; MBPP gives the task text and the first test as an example.
- **Decoding.** Greedy, chat template with thinking on, 4,096 new tokens at most, batch 64-96.
  - The code scored is the longest ```python block after `</think>`.
  - A reply cut off at the cap is scored on whatever code it holds, which is usually none.
- **Scoring.** `eval_code.py score` runs the evalplus base tests and the stricter plus tests.
- **Comparison.** `eval/compare.py a.scores.json b.scores.json` reports the paired difference, a bootstrap 95% CI,
  wins/losses and an exact McNemar p-value.
- **Deltas.** `eval_code.py gen <model> <delta|none>` applies an int4 delta in memory: either ESFT expert keys or SFT
  keys in the checkpoint's own `model.layers…` layout.
- **Cost.** 45-65 min per model on one A6000. Decode takes ~0.6 s per step whatever the batch size, so one big batch is
  best.

Base Flint (abliterated, int4) scores **83.3% plus / 92.7% base**, with a mean reply of 972 tokens
(`results/esft/eval_base.scores.json`).

## B. Pruning evals: `prune/` (Mac, Locals server)

- **HumanEval+, every 4th task (40):** sent to the Locals OpenAI-compatible server with greedy decoding, full
  reasoning and DFlash 2 on. `humaneval.py` sanitizes the replies with evalplus, and `score.py` /
  `evalplus.evaluate` scores them. Results are in `results/prune/<variant>/he/` and `score.log`.
- **KL and top-1 agreement against the 256-expert model:** measured on 32 held-out windows of coding and agent text
  (see [pruning.md](pruning.md)), with a window-level bootstrap.
- **Speed:** `speed.py`, 2 prompts × 2 runs, 1,024 tokens, temperature 0, with DFlash on and off.

## Caveats

- n=40 and n=96 detect regressions of about 10 points; differences of 2-3 points are noise (see the CIs).
- The prompts don't force a format beyond "one python block". Some losses are formatting, such as a missing import.
- HumanEval+ and MBPP+ are single-function tasks, not agentic. Agent behaviour was only spot-checked (pi with Locals:
  read, fix and write a file with tools), not benchmarked.
- `results/prune/base256/score.log` is a failed first scoring attempt ("Missing problems in samples"). The base number
  comes from the completed `base256/he/padded_eval_results.json`. Unsampled tasks are padded with canonical solutions,
  so count only the 40 task ids in `samples.jsonl`. That gives 36/40 plus and 37/40 base for 256 experts, and 35/40 and
  37/40 for REAP-224, which loses HumanEval/48. The HF model card first said 35/40 for both; it now carries a correction note.

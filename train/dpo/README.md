# Flint DPO: prepared, never run (2026-10-29)

> Status: the coding-tuning work stopped here because the compute budget ran out. Everything below is ready but no DPO
> step ever ran on a GPU. Estimated cost of the full run: about $3 and ~7 A6000-hours. The original plan ran DPO on
> top of the SFT model; after SFT failed, the intended next step became on-policy DPO from the base model (`<sft_delta>` = `none`).
> Commands assume the repo is at /root/ft on the pod.

Plan change: SFT on frontier-model traces runs first; DPO then runs on-policy FROM THE SFT MODEL.
So no rollouts were generated from the base model. Everything below is prep, ready to run.

## Status
- Pod brave-matrix-97 went unreachable ~17:20 UTC Oct 29, reboot failed, the coordinator terminated it. Nothing of this
  prep ran on a GPU. Everything is in train/dpo/ of this repo.
- CPU-checked: score.py, pairs.py and remap_delta.py (assert self-checks), and problems.jsonl was built and verified.
- NOT yet run on the GPU, so do this first on the new pod (about 10 min): `bash train/dpo/pod_queue.sh`. It runs `dpo.py bench`
  at 3072 and 4608 tokens (peak VRAM, s/pair; ESFT peaked at 41.6 GB at 2.8k, so 4608 may OOM and then use
  --max-len 3072) and `test_rollout.py`, which checks that the refill decoder reproduces esft.generate's greedy
  tokens and that sampling varies. The REAP-224 eval at the end of the queue is optional (~1 h); it needs /root/eval_base.jsonl.
- Lost with the pod: /root/stats.json, /root/eval_base.jsonl and the ESFT int4 delta. Copies in this repo:
  results/esft/stats.json (593 experts) and results/esft/eval_base.jsonl. The ESFT checkpoint is not published.
- New pod setup: upload this repo to /root/ft, build train/dpo/problems.jsonl with build_problems.py, copy
  results/esft/stats.json to /root/stats.json, and set the model at /root/flint (HF dima6312/Flint-40B-A2.5B-abliterated-Rec-4bit).

## Files (repo: train/dpo/, pod: /root/ft/train/dpo/)
- `problems.jsonl` (not in the repo; rebuild with `build_problems.py`): 600 Codeforces stdin/stdout problems (open-r1/codeforces-cots solutions_py_decontaminated,
  checker type "diff", seed-0 shuffle). Each has 3-8 tests (public + up to 6 private). Kept only if the dataset's
  own R1 reference code passes every kept test. Field `prompt` is the dataset's prompt (asks for one ```python block).
- `score.py`: runs the reply's last ```python block (after </think>) on the tests, 5 s CPU / test. CPU only.
- `rollout.py`: k sampled replies per problem, thinking on, continuous batching (finished rows refilled).
  Resumable by (id, k). Takes `<delta|none>` so it can sample from the SFT model.
- `pairs.py`: chosen = shortest passing uncapped; rejected = failing (w 1.0) else passing but capped or
  >=1.5x longer (w 0.5); up to 2 pairs per problem.
- `dpo.py`: `ref` (reference log-probs, one model in VRAM), `train` (DPO on the 593 ESFT coding experts from
  /root/stats.json, AdamSR bf16, cosine LR, resumable), `bench` (VRAM/time check).
- `remap_delta.py`: maps an int4 expert delta onto REAP-224 (keep_ids.json) so apply_delta.py can patch the pruned model.
- `pod_queue.sh`: the GPU queue run during prep (bench, rollout check, REAP-224 eval).

## Steps once the SFT model exists (delta = its int4 expert delta, or `none`)
    export FLINT_CDT=bfloat16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd /root/ft
    python3 train/dpo/rollout.py /root/flint <sft_delta> train/dpo/problems.jsonl /root/dpo/roll.jsonl --k 4 --temp 0.8 --max-new 4096 --bs 96 --n 400
    python3 train/dpo/score.py train/dpo/problems.jsonl /root/dpo/roll.jsonl          # -> roll.scored.jsonl
    python3 train/dpo/pairs.py /root/dpo/roll.scored.jsonl /root/dpo/pairs.jsonl
    python3 train/dpo/dpo.py ref /root/flint <sft_delta> /root/dpo/pairs.jsonl /root/dpo/ref.jsonl --max-len 3072
    python3 train/dpo/dpo.py train /root/flint <sft_delta> /root/stats.json /root/dpo/pairs.jsonl /root/dpo/ref.jsonl /root/dpo/run \
        --beta 0.1 --lr 5e-7 --accum 8 --epochs 2 --rpo 0.2 --max-len 3072 --sync /root/ckpt_sync
    # resume after a stop: same train command + --resume 1 (reads /root/dpo/run/state.pt: params, AdamSR moments, step, cursor)
    python3 eval/eval_code.py gen /root/flint /root/dpo/run/esft_delta.safetensors /root/eval_dpo.jsonl --he 48 --mbpp 48 --max-new 4096 --bs 64

Caveat: the DPO delta holds only DPO-trained experts. If the SFT delta changed other experts, the eval/ship model
is base + SFT delta + DPO delta (apply in that order). eval_code.py takes one delta: merge them first (later wins per expert).

## Ship path
apply_delta.py <base folder> <delta> <out folder> writes Recursive's int4 layout byte for byte; then on the Mac
`python -m locals.models.pack` (in a Locals install). REAP-224: remap_delta.py first (with prune/masks/keep_reap224.json),
then apply_delta.py on the REAP-224 folder (RinneCODER/Flint-40B-A2.5B-abliterated-REAP224-Rec-4bit).

## Pod-loss fallback
/root/ckpt_sync holds `dpo_delta_int4.safetensors` (int4 of the trained experts, ~1 GB) + train_state.json.
Full bf16 params + optimizer (~11 GB) stay in /root/dpo/run/state.pt (too big for the sync). If the pod is lost,
restart training from the int4 delta (apply as start delta, fresh optimizer, skip `cursor` pairs).

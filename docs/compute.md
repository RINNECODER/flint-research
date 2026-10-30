# Compute notes

Total paid compute was about **$6.8 on Lium**. Everything else ran on free Kaggle or on the Mac.

| where | what | cost |
|---|---|---|
| M5 Pro 24 GB Mac | serving, speed path, expert stats, REAP masks, KL and HumanEval+ evals | free |
| Kaggle 2× T4 | PyTorch LoRA proof (parity + timed steps) | free |
| Kaggle TPU v5e-8 | JAX proof: never ran (queued for more than 24 h) | free |
| Lium 1× RTX A6000 48 GB, $0.42/h | ESFT (≈ $3.4 including evals), SFT LoRA (≈ $2.6), probes and a dead pod | ≈ $6.8 |
| not spent | on-policy DPO, estimated ~7 A6000-hours | ≈ $3 |

## Kaggle T4 (`kaggle/t4/`)

- `build.sh` concatenates `model/flint_torch.py`, the Locals reference (`model/parity/reference.json`) and `run.py` into
  one script (`flint_t4.py`). Kaggle script kernels are single files.
- Set your Kaggle username in `kernel-metadata.json` (`YOUR_KAGGLE_USER`), then run `kaggle kernels push -p kaggle/t4`.
  The model downloads from HF inside the kernel (internet on).
- **Result** (`results/kaggle/t4_result.json`):
  - 96.9% top-1 parity with Locals.
  - ~39 s per 2k-token LoRA step, with chunked loss so that 2k sequences fit in 2× 16 GB.
- That speed projects to ~54 GPU-hours per 5k examples. Kaggle's weekly quota makes that impractical, so the real runs
  moved to rented A6000s.

## Kaggle TPU v5e-8 (`kaggle/tpu/`, `kaggle/tpu-probe/`)

- Kaggle allows one TPU session per user, and a queued one counts. `watch.sh` (needs `KAGGLE_USER`) waits for the probe,
  pushes the proof, and saves both outputs.
- The probe sat in `QUEUED` from 2026-10-28 23:37 to 10-29 19:22 CT (~20 h). It showed 8 TPU devices, jax 0.11 and
  torch_xla 2.8 (`results/kaggle/tpu_probe.log`).
- The proof kernel was then queued in turn and had still not started more than 24 h later. The JAX path is unproven at
  scale.

## Lium (RTX A6000)

- **Power caps.** Every A6000 and L40 node offered was capped at 276 W of its 300 W default (8% under). Check with
  `nvidia-smi -q -d POWER`.
  - The ESFT node ran at 276 W with no active throttle reasons. It was pass-through (not vGPU/MIG), with 46.9 GB usable
    and ~110 bf16 TFLOPs measured.
  - The SFT node ran at the full 300 W.
  - `train/pod/probe.sh` runs these checks, plus a matmul benchmark, disk/RAM checks and an HF download speed test,
    before any money is spent on training.
- **Budget guard.** `train/budget_guard.sh <pod> [floor]` runs every 45 min.
  - It logs the account balance.
  - It pulls the pod's `/root/ckpt_sync` (adapters, optimizer, step, data cursor) to `ckpt/pod/`.
  - Below the floor (default $0.75), it does a final pull and then terminates the pod, so a run can't drain the account.
  - The scripts write resumable state there every ≤ 25 min.
- **A pod died.** The ESFT pod stopped answering ssh about 17:45 UTC on 2026-10-29, right after training and both evals.
  - The provider's reboot ended in `REBOOT_FAILED` ("Executor unreachable"), and the pod was terminated.
  - The 3.7 GB bf16 checkpoint copy had reached only 1.5 GB, and the int4 delta was never copied. Only the code, data,
    stats and eval scores survived on the Mac.
  - Lesson: pull small artifacts first (int4 delta, scores) and big ones after, and sync on a timer during the run,
    which is what `budget_guard.sh` was then built for.
- **Gotchas:**
  - `pkill -f <pattern>` over ssh kills the ssh shell itself (and once killed an eval). Use `pgrep -f "python3 X"`, then
    `kill <pid>`.
  - For tar from macOS, use `COPYFILE_DISABLE=1 tar --no-xattrs`.
  - Launch long jobs with `setsid nohup … < /dev/null &`.
  - Set `FLINT_CDT=bfloat16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` always.
  - Prefill OOMs at 64 rows at once, so `esft.generate` prefills 16 rows at a time.

## Mac

The Mac ran everything in `prune/` in-process on Locals' streamed model, with no GPU rental:

- ~50 s per held-out NLL pass (32 × 1k tokens);
- pruned builds are I/O-bound byte slicing (`prune.py`);
- one expert pack per variant (`python -m locals.models.pack`).

The GPU wired limit was left at its default, and the expert cache was kept at 6 GB to keep memory pressure low.

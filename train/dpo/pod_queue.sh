#!/bin/bash
# GPU queue on the pod (run detached): DPO VRAM bench, rollout check, REAP-224 build + eval. Logs in /root/q_*.log.
export FLINT_CDT=bfloat16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/ft  # the repo root, uploaded to the pod
[ -d /root/flint-reap224 ] || (python3 prune/prune.py /root/flint prune/masks/keep_reap224.json /root/flint-reap224 > /root/q_prune.log 2>&1 &)
for L in 3072 4608; do python3 train/dpo/dpo.py bench /root/flint /root/stats.json $L > /root/q_bench_$L.log 2>&1; done
python3 train/dpo/test_rollout.py > /root/q_rollout_test.log 2>&1
while [ ! -f /root/flint-reap224/config.json ] || pgrep -f "prune.py" >/dev/null; do sleep 20; done
python3 eval/eval_code.py gen /root/flint-reap224 none /root/eval_reap224.jsonl --he 48 --mbpp 48 --max-new 4096 --bs 64 > /root/q_reap_gen.log 2>&1
python3 eval/eval_code.py score /root/eval_reap224.jsonl > /root/q_reap_score.log 2>&1
mkdir -p /root/ckpt_sync && cp /root/eval_reap224.jsonl* /root/q_*.log /root/ckpt_sync/ 2>/dev/null
echo done > /root/q_done

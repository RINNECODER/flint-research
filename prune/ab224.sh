#!/bin/zsh
# ab224.sh VARIANT...: build each candidate mask (masks/keep_VARIANT.json) from the 256-expert model, KL-eval vs 256, delete.
# Needs results/prune/nll_256.npz first: $LOCALS_PY run.py nll <256 model> ../results/prune/nll_256.npz
# Env: LOCALS_PY (Python of a Locals install), BASE (the dima6312/Flint-40B-A2.5B-abliterated-Rec-4bit folder), WORK (scratch dir).
cd ${0:A:h}
PY=${LOCALS_PY:-python3}; R=../results/prune; W=${WORK:-/tmp/flint-ab}
S=${BASE:-$(echo ~/.cache/huggingface/hub/models--dima6312--Flint-40B-A2.5B-abliterated-Rec-4bit/snapshots/*)}
for v in "$@"; do
  $PY -I prune.py $S masks/keep_$v.json $W/flint-abl-$v > $R/logs/build_$v.log 2>&1 || exit 1
  REF=$R/nll_256.npz LOCALS_EXPERT_CACHE_GB=2 $PY run.py nll $W/flint-abl-$v $R/nll_$v.npz 2>&1 | grep saved
  $PY compare.py $R/nll_256.npz $R/nll_$v.npz
  rm -rf $W/flint-abl-$v
done

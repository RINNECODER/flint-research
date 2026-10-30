#!/bin/zsh
# heonly.sh LABEL MODEL: HumanEval+ 40 (dflash on, resumable) on :8322, then score. Env as in evalmodel.sh.
L=$1; M=$2; T=${0:A:h}; O=$T/../results/prune/$L; mkdir -p $O
LOC=${LOCALS_BIN:-locals}; EPY=${EVAL_PY:-python3}
[ -n "${LOCALS_SRC:-}" ] && export PYTHONPATH=$LOCALS_SRC
export LOCALS_DRAFTER=dflash
LOCALS_EXPERT_CACHE_GB=6 $LOC start --model $M --cache-gb 2 --max-tokens 32768 --port 8322 > $O/start_he.log 2>&1
$EPY $T/humaneval.py $O/he 8322 >> $O/he.log 2>&1
$LOC stop --port 8322 >/dev/null 2>&1; sleep 3
$EPY -m evalplus.evaluate --dataset humaneval --samples $O/he/samples.jsonl > $O/score.log 2>&1
tail -4 $O/score.log

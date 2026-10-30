#!/bin/zsh
# evalmodel.sh LABEL MODEL [skip-he]: HumanEval+ (40, dflash on) + speed dflash on/off, on :8322. Logs to results/prune/LABEL/.
# Env: LOCALS_BIN (the `locals` CLI), LOCALS_PY (its Python), EVAL_PY (a Python with evalplus),
#      LOCALS_SRC (optional: a Locals checkout with patches/locals applied, put first on PYTHONPATH).
set -u
L=$1; M=$2; T=${0:A:h}; O=$T/../results/prune/$L; mkdir -p $O
LOC=${LOCALS_BIN:-locals}; PY=${LOCALS_PY:-python3}; EPY=${EVAL_PY:-python3}
[ -n "${LOCALS_SRC:-}" ] && export PYTHONPATH=$LOCALS_SRC
serve() {  # $1 on|off
  if [ $1 = on ]; then export LOCALS_DRAFTER=dflash; unset LOCALS_DRAFTS; else export LOCALS_DRAFTS=0; unset LOCALS_DRAFTER; fi
  LOCALS_EXPERT_CACHE_GB=6 $LOC start --model $M --cache-gb 2 --max-tokens 32768 --port 8322 > $O/start_$1.log 2>&1
  for i in {1..300}; do curl -sf localhost:8322/v1/models >/dev/null && break; sleep 2; done
}
stop() { $LOC stop --port 8322 >/dev/null 2>&1; sleep 3; }
serve on
[ "${3:-}" = skip-he ] || $EPY $T/humaneval.py $O/he 8322 > $O/he.log 2>&1
$PY $T/speed.py 8322 $L-dflash > $O/speed_on.log 2>&1
stop
serve off
$PY $T/speed.py 8322 $L-nodraft > $O/speed_off.log 2>&1
stop
[ "${3:-}" = skip-he ] || $EPY $T/score.py $O/he > $O/score.log 2>&1
tail -1 $O/speed_on.log $O/speed_off.log

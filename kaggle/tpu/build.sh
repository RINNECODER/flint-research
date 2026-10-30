#!/bin/bash
# Builds the single-file Kaggle TPU script: flint_jax.py + embedded Locals reference + run_tpu.py
set -e; cd "$(dirname "$0")"
{ cat ../../model/flint_jax.py; echo; echo "REF_JSON = r'''$(cat ../../model/parity/reference.json)'''"; echo "SEQ_LENS = ${SEQ_LENS:-[512, 1024, 2048]}"; echo "STEPS = ${STEPS:-3}"; cat run_tpu.py; } > flint_tpu.py
python3 -c "import ast;ast.parse(open('flint_tpu.py').read())"

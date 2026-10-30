#!/bin/bash
# Builds the single-file Kaggle script: flint_torch.py + embedded Locals reference + run.py
set -e; cd "$(dirname "$0")"
{ cat ../../model/flint_torch.py; echo; echo "REF_JSON = r'''$(cat ../../model/parity/reference.json)'''"; echo "SEQ_LENS = ${SEQ_LENS:-[512, 1024, 2048]}"; echo "STEPS = ${STEPS:-3}"; cat run.py; } > flint_t4.py
python3 -c "import ast;ast.parse(open('flint_t4.py').read())"

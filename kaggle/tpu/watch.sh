#!/bin/bash
# Detached TPU watcher: Kaggle allows one TPU session (queued counts), so the proof kernel can only be pushed once the
# queued probe has run. Waits for the probe, saves its output, pushes flint-lora-tpu, waits, saves its output.
# Never deletes anything. Progress: watch.log; finished: out/DONE.
cd "$(dirname "$0")"; mkdir -p out/probe out/proof; U=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
st() { kaggle kernels status $U/$1 2>&1 | tail -1; }
log() { echo "$(date '+%F %T %Z') $*" >> watch.log; }
log "watching probe: $(st flint-tpu-probe)"
until st flint-tpu-probe | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 300; done
log "probe finished: $(st flint-tpu-probe)"; kaggle kernels output $U/flint-tpu-probe -p out/probe >/dev/null 2>&1
until kaggle kernels push -p . 2>&1 | tee -a watch.log | grep -q "successfully pushed"; do sleep 300; done
log "proof pushed"
sleep 60
until st flint-lora-tpu | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 300; done
log "proof finished: $(st flint-lora-tpu)"; kaggle kernels output $U/flint-lora-tpu -p out/proof >/dev/null 2>&1
touch out/DONE

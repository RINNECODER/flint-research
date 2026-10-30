#!/bin/zsh
# re-download raw inputs into ../raw (all public HF, no auth). Then: python3 -I conv_dsf.py conv_cc.py conv_cursor.py conv_ag.py assemble.py maskcheck.py
R=${0:A:h}/../raw; mkdir -p $R/{s_dsf,s_fable_cc,s_alin,s_cursor,s_ag,ep}
# DSFFGFG456: parquet shards -> dsf_load.py keeps the final (longest) row per trajectory -> raw/dsf_full.jsonl
curl -s "https://datasets-server.huggingface.co/parquet?dataset=DSFFGFG456/fable-5-coding-and-debugging-traces" | jq -r '.parquet_files[].url' | (cd $R/s_dsf && xargs -P8 -I{} curl -sL -O {})
python3 -I dsf_load.py $R/s_dsf $R/dsf_full.jsonl
for d in armand0e/claude-fable-5-claude-code:s_fable_cc AlinCiocan/fable-5-claude-code-traces:s_alin; do id=${d%%:*}; dir=${d##*:}
  curl -s https://huggingface.co/api/datasets/$id | jq -r '.siblings[].rfilename|select(endswith(".jsonl"))' | (cd $R/$dir && xargs -P6 -I{} curl -sL -o {} https://huggingface.co/datasets/$id/resolve/main/{}); done
curl -sL -o $R/s_cursor/train.jsonl https://huggingface.co/datasets/TeichAI/Fable-5-Cursor-Traces/resolve/main/train.jsonl
curl -sL -o $R/s_ag/code_train.jsonl https://huggingface.co/datasets/angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k/resolve/main/code_train.jsonl
for d in evalplus/humanevalplus evalplus/mbppplus; do n=${d#*/}; mkdir -p $R/ep/$n; curl -s "https://datasets-server.huggingface.co/parquet?dataset=$d" | jq -r '.parquet_files[]|select(.split=="test")|.url' | while read u; do curl -sL -o $R/ep/$n/$(basename $u) "$u"; done; done

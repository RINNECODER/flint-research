#!/bin/bash
# Phase 0 probe: GPU cap checks, bf16 matmul TFLOPs, disk/RAM, HF download speed. Writes /root/probe/*.txt
mkdir -p /root/probe; cd /root/probe
nvidia-smi --query-gpu=name,memory.total,memory.used,power.limit,power.default_limit,power.max_limit,clocks.max.sm,clocks.sm,ecc.mode.current,mig.mode.current,compute_mode --format=csv > gpu.csv 2>&1
nvidia-smi -q > smi_q.txt 2>&1
grep -iA12 "Clocks Event Reasons\|Clocks Throttle Reasons" smi_q.txt | head -16 > throttle.txt
grep -i "virtualization\|Host VGPU\|vGPU" smi_q.txt > virt.txt
df -h / /root > disk.txt; free -g > ram.txt; nproc > cpu.txt
python3 -c "import torch;print(torch.__version__, torch.version.cuda)" > torch.txt 2>&1
pip install -q safetensors tokenizers jinja2 evalplus hf_transfer huggingface_hub > pip.log 2>&1
python3 - > matmul.txt 2>&1 <<'PY'
import torch, time
for d in range(torch.cuda.device_count()):
    a = torch.randn(8192, 8192, device=f"cuda:{d}", dtype=torch.bfloat16); b = torch.randn_like(a)
    free, tot = torch.cuda.mem_get_info(d)
    for _ in range(5): a @ b
    torch.cuda.synchronize(d); t = time.time(); n = 200
    for _ in range(n): a @ b
    torch.cuda.synchronize(d); dt = time.time() - t
    x = torch.empty(int(free * 0.97) // 2, dtype=torch.bfloat16, device=f"cuda:{d}"); x.fill_(1); ok = bool(x[-1] == 1); del x
    print(f"cuda:{d} bf16 TFLOPs {2*8192**3*n/dt/1e12:.1f} free_GB {free/2**30:.1f} total_GB {tot/2**30:.1f} alloc97%_ok {ok}")
PY
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv >> matmul.txt 2>&1
t=$(date +%s); HF_HUB_ENABLE_HF_TRANSFER=1 hf download dima6312/Flint-40B-A2.5B-abliterated-Rec-4bit --local-dir /root/flint > dl.log 2>&1; echo "download_s $(( $(date +%s) - t )) GB $(du -sh /root/flint | cut -f1)" > dl.txt
echo PROBE_DONE > done.txt

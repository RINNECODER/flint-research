import subprocess, time, os
print("START", time.strftime("%H:%M:%S"))
for c in ["df -h / /kaggle/working /tmp", "free -g", "nproc", "python -c 'import jax;print(jax.devices())'", "pip list 2>/dev/null | grep -i -E 'torch|jax|xla'"]:
    print("$", c); print(subprocess.run(c, shell=True, capture_output=True, text=True, timeout=300).stdout[-2000:])

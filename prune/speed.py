"""python speed.py PORT LABEL: decode tok/s on 2 coding prompts x 2 runs (temperature 0, 1024 tokens), draft acceptance
from /stats. Decode time = first streamed token to the last; tokens from usage."""
import json
import sys
import time
import urllib.request

port, label = sys.argv[1], sys.argv[2]
BASE = f"http://127.0.0.1:{port}"
PROMPTS = [
    "Write a Python module implementing an LRU cache with TTL expiry, thread safety, and type hints. Include a short "
    "docstring for each method and a few unit tests with unittest.",
    "In TypeScript, write an Express middleware that rate-limits requests per API key using a sliding window stored "
    "in Redis, with error handling and configuration options. Explain briefly after the code.",
]


def stats():
    try:
        return json.load(urllib.request.urlopen(BASE + "/stats", timeout=10))
    except Exception:
        return {}


def find(d, key):
    if isinstance(d, dict):
        if key in d and isinstance(d[key], (int, float)):
            return d[key]
        for v in d.values():
            r = find(v, key)
            if r is not None:
                return r
    return None


res = []
s0 = stats()
for p in PROMPTS:
    for run in range(2):
        req = {"messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": 1024, "stream": True,
               "stream_options": {"include_usage": True}}
        r = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(req).encode(), {"Content-Type": "application/json"})
        t0 = time.time()
        first = last = None
        usage = None
        with urllib.request.urlopen(r, timeout=1800) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    usage = d["usage"]
                if d.get("choices") and (d["choices"][0].get("delta", {}).get("content") or d["choices"][0].get("delta", {}).get("reasoning_content")):
                    first = first or time.time()
                    last = time.time()
        n = usage["completion_tokens"]
        tps = (n - 1) / (last - first)
        res.append(tps)
        print(f"{label} prompt{PROMPTS.index(p)} run{run}: {n} tokens, decode {tps:.1f} tok/s, ttft {first - t0:.1f}s", flush=True)
s1 = stats()
dr = (find(s1, "drafted") or 0) - (find(s0, "drafted") or 0)
dt = (find(s1, "draft_tokens") or 0) - (find(s0, "draft_tokens") or 0)
print(json.dumps({"label": label, "gpu_peak_mb": find(s1, "gpu_peak_mb"), "decode_tps_mean": sum(res) / len(res), "runs": res,
                  "acceptance": dr / dt if dt else None, "drafted": dr, "draft_tokens": dt}))

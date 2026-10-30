"""Greedy 32-token continuations from the live Locals server (flint-int4 weights) for fixed prompts. Read-only."""
import json, urllib.request
from tokenizers import Tokenizer
tok = Tokenizer.from_file("../ref/tokenizer.json")
PROMPTS = [
    "The capital of France is Paris. The capital of Japan is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Photosynthesis is the process by which plants",
]
out = []
for p in PROMPTS:
    ids = [150000] + tok.encode(p, add_special_tokens=False).ids  # BOS + text
    req = {"model": "locals/flint-int4", "prompt": ids, "max_tokens": 32, "temperature": 0, "ignore_eos": True}
    r = urllib.request.Request("http://127.0.0.1:8093/v1/completions", json.dumps(req).encode(), {"Content-Type": "application/json"})
    text = json.load(urllib.request.urlopen(r, timeout=600))["choices"][0]["text"]
    out.append({"prompt": p, "prompt_ids": ids, "ref_text": text, "ref_ids": tok.encode(text, add_special_tokens=False).ids})
    print(repr(p[:40]), "->", repr(text))
json.dump(out, open("reference.json", "w"), indent=1)

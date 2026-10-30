"""Pod check: continuous-batching greedy (bs 3, refills of 1, so rows join mid-stream with different cache lengths)
must reproduce esft.generate's greedy tokens. Prints the common-prefix share per prompt; ok if ~all are 1.0."""
import sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.join(_here, ".."), os.path.join(_here, "..", "..", "model")]  # train/dpo, train, model
from jinja2 import Environment
from tokenizers import Tokenizer
import esft, flint_torch as ft
from rollout import sample_stream

M = "/root/flint"
tmpl = Environment().from_string(open(f"{M}/chat_template.jinja").read())
tok = Tokenizer.from_file(f"{M}/tokenizer.json")
qs = ["Write a Python function that reverses a string.", "What is 17*23? Answer briefly.",
      "Explain in two sentences what a hash map is, then give a one-line Python example of a dict comprehension.",
      "Read n then n integers from stdin and print their sum. Write the Python program.", "Name three sorting algorithms.",
      "Write a Python one-liner that prints the first 10 squares."]
P = [tok.encode(tmpl.render(messages=[{"role": "user", "content": q}], add_generation_prompt=True), add_special_tokens=False).ids for q in qs]
m = ft.load(M, esft.DEV).eval()
ref, _ = esft.generate(m, P, 96)
got = {k: o for k, o, _ in sample_stream(m, list(enumerate(P)), 96, 0.0, 3, 1)}
for i, r in enumerate(ref):
    g = got[i]
    n = next((j for j, (a, b) in enumerate(zip(r, g)) if a != b), min(len(r), len(g)))
    print(i, len(r), len(g), "prefix share", round(n / max(len(r), 1), 3))
s = list(sample_stream(m, [(0, P[0])] * 4, 64, 0.8, 4, 1))
print("sampled distinct", len({tuple(o) for _, o, _ in s}), "of 4")

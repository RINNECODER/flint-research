"""python humaneval.py OUT_DIR [PORT]: 40 fixed HumanEval+ items (every 4th task), temperature 0, full reasoning,
against a Locals server; writes OUT_DIR/samples.jsonl (+ raw.jsonl, resumable). Score with evalplus.evaluate."""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from evalplus.data import get_human_eval_plus
from evalplus.sanitize import sanitize

out = Path(sys.argv[1])
port = sys.argv[2] if len(sys.argv) > 2 else "8322"
out.mkdir(parents=True, exist_ok=True)
probs = get_human_eval_plus()
ids = sorted(probs, key=lambda t: int(t.split("/")[1]))[::4][:40]
raw = out / "raw.jsonl"
done = {json.loads(l)["task_id"] for l in raw.open()} if raw.exists() else set()
for tid in ids:
    if tid in done:
        continue
    p = probs[tid]
    msg = ("Complete the following Python function. Reply with the complete function (including the imports and "
           "signature) in one ```python code block.\n\n```python\n" + p["prompt"] + "```")
    req = {"messages": [{"role": "user", "content": msg}], "temperature": 0, "max_tokens": 16384}
    t0 = time.time()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(req).encode(),
                               {"Content-Type": "application/json"})
    try:
        resp = json.load(urllib.request.urlopen(r, timeout=3600))
    except urllib.error.HTTPError as e:  # a server error counts as a failed item, recorded
        resp = {"choices": [{"message": {"content": ""}, "finish_reason": f"http {e.code}"}], "usage": {}}
    ch = resp["choices"][0]
    text = ch["message"].get("content") or ""
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    code = blocks[-1] if blocks else text
    sol = sanitize(code, entrypoint=p["entry_point"])
    with raw.open("a") as f:
        f.write(json.dumps({"task_id": tid, "content": text, "finish": ch.get("finish_reason"),
                            "usage": resp.get("usage"), "secs": time.time() - t0, "solution": sol}) + "\n")
    print(tid, ch.get("finish_reason"), resp.get("usage", {}).get("completion_tokens"), f"{time.time() - t0:.0f}s", flush=True)
with (out / "samples.jsonl").open("w") as f:
    for l in raw.open():
        d = json.loads(l)
        f.write(json.dumps({"task_id": d["task_id"], "solution": d["solution"]}) + "\n")

"""python score.py RESULTS_DIR/he: score our 40 items with evalplus (pads the other HumanEval+ tasks with an empty
solution, since evalplus.evaluate requires all of them, then counts only ours). Prints base and plus pass@1, n."""
import json
import subprocess
import sys
from pathlib import Path

from evalplus.data import get_human_eval_plus

d = Path(sys.argv[1])
ours = [json.loads(l) for l in (d / "samples.jsonl").open()]
ids = {s["task_id"] for s in ours}
padded = d / "padded.jsonl"
with padded.open("w") as f:
    for s in ours:
        f.write(json.dumps(s) + "\n")
    for tid in get_human_eval_plus():
        if tid not in ids:
            f.write(json.dumps({"task_id": tid, "solution": ""}) + "\n")
res = d / "padded_eval_results.json"
res.unlink(missing_ok=True)
subprocess.run([str(Path(sys.executable).parent / "evalplus.evaluate"), "--dataset", "humaneval", "--samples", str(padded),
                "--i-just-wanna-run"], env={**__import__("os").environ, "EVALPLUS_MAX_MEMORY_BYTES": "-1"}, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
r = json.load(res.open())["eval"]
base = sum(r[t][0]["base_status"] == "pass" for t in ids)
plus = sum(r[t][0]["base_status"] == "pass" and r[t][0]["plus_status"] == "pass" for t in ids)
fails = sorted(t for t in ids if not (r[t][0]["base_status"] == "pass" and r[t][0]["plus_status"] == "pass"))
print(json.dumps({"n": len(ids), "base_pass": base, "plus_pass": plus, "plus_pass@1": plus / len(ids), "plus_fails": fails}))

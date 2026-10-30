"""Fetch calibration and held-out text (public HF datasets, via datasets-server rows API) to data/{calib,heldout}.jsonl.
Each line: {"source": ..., "messages": [...]} (chat) or {"source": ..., "text": ...} (raw code). Disjoint row ranges."""
import json
import time
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows?dataset={d}&config=default&split={s}&offset={o}&length={n}"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def rows(d, s, offset, count):
    got = []
    while len(got) < count:
        n = min(100, count - len(got))
        for _ in range(5):
            try:
                with urllib.request.urlopen(API.format(d=d, s=s, o=offset + len(got), n=n), timeout=60) as r:
                    batch = [x["row"] for x in json.load(r)["rows"]]
                break
            except Exception as e:  # rate limits: retry
                print("retry", d, e, flush=True)
                time.sleep(10)
        else:
            raise SystemExit(f"failed {d}")
        if not batch:
            break
        got += batch
    return got


def swe(r):
    m = [{"role": {"ai": "assistant"}.get(t["role"], t["role"]), "content": t.get("system_prompt") or t["text"]}
         for t in r["trajectory"]]
    return {"source": "swe-agent", "messages": [x for x in m if x["content"]]}


SPECS = {  # (dataset, split, calib offset, calib n, heldout offset, heldout n, to-record)
    "code": ("codeparrot/codeparrot-clean-valid", "train", 0, 400, 30000, 60, lambda r: {"source": "code", "text": r["content"]}),
    "swe-agent": ("nebius/SWE-agent-trajectories", "train", 0, 60, 60000, 20, swe),
    "openhands": ("SWE-Gym/OpenHands-SFT-Trajectories", "train.success.oss", 0, 40, 420, 20,
                  lambda r: {"source": "openhands", "messages": r["messages"]}),
}
for split, (oi, ni) in {"calib": (2, 3), "heldout": (4, 5)}.items():
    with open(OUT / f"{split}.jsonl", "w") as f:
        for name, spec in SPECS.items():
            for r in rows(spec[0], spec[1], spec[oi], spec[ni]):
                f.write(json.dumps(spec[6](r)) + "\n")
            print(split, name, "done", flush=True)

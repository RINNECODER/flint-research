"""Execute a reply's last ```python block against stdin/stdout tests (line-wise, trailing space ignored), 5 s CPU each.
  python3 score.py <problems.jsonl> <rollouts.jsonl>   -> adds "passed","n_tests" to each rollout (writes <rollouts>.scored.jsonl)"""
import json, os, re, resource, subprocess, sys, tempfile
from multiprocessing import Pool

CODE = re.compile(r"```(?:python|py|python3)?\n(.*?)```", re.S)
PY = os.environ.get("SCORE_PY", sys.executable)


def extract(text):
    m = CODE.findall(text.split("</think>")[-1]) if "</think>" in text else []
    return m[-1] if m else ""


def _lim():
    resource.setrlimit(resource.RLIMIT_CPU, (6, 6))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))  # not supported on macOS
    except (ValueError, OSError):
        pass


norm = lambda s: [l.rstrip() for l in s.strip().splitlines()]


def run_tests(code, tests):
    """Number of tests passed; stops at the first failure."""
    if not code:
        return 0
    n = 0
    with tempfile.TemporaryDirectory() as d:
        for t in tests:
            try:
                r = subprocess.run([PY, "-I", "-c", code], input=t["input"], capture_output=True, text=True, timeout=10, cwd=d, preexec_fn=_lim)
            except Exception:
                return n
            if r.returncode != 0 or norm(r.stdout) != norm(t["output"]):
                return n
            n += 1
    return n


def _score(a):
    r, tests = a
    r["n_tests"] = len(tests)
    r["passed"] = run_tests(extract(r["text"]), tests) == len(tests)
    return r


if __name__ == "__main__":
    probs = {p["id"]: p["tests"] for p in map(json.loads, open(sys.argv[1]))}
    rows = [json.loads(l) for l in open(sys.argv[2])]
    with Pool(os.cpu_count()) as pool, open(sys.argv[2].replace(".jsonl", ".scored.jsonl"), "w") as f:
        for r in pool.imap(_score, [(r, probs[r["id"]]) for r in rows], chunksize=4):
            f.write(json.dumps(r) + "\n")
    print("scored", len(rows))

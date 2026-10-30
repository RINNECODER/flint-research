"""Rollout problem set from open-r1/codeforces-cots solutions_py_decontaminated (stdin/stdout, 'diff' checker only).
Keeps problems whose own R1 reference code passes every kept test (so the tests are runnable and correct).
  python3 build_problems.py <parquet dir> <out problems.jsonl> [n=600]"""
import glob, json, random, re, sys
from multiprocessing import Pool
import pyarrow.parquet as pq
from score import run_tests

CODE = re.compile(r"```(?:python|py|python3)?\n(.*?)```", re.S)


def cands(d):
    seen = set()
    for f in sorted(glob.glob(f"{d}/*.parquet")):
        for r in pq.read_table(f, columns=["id", "prompt", "generation", "finish_reason", "problem_type", "public_tests", "private_tests"]).to_pylist():
            if r["id"] in seen or r["problem_type"] != "diff" or r["finish_reason"] != "stop":
                continue
            pub, priv = r["public_tests"] or {}, r["private_tests"] or {}
            tests = [{"input": i, "output": o} for i, o in zip(pub.get("input") or [], pub.get("output") or [])]
            tests += [{"input": i, "output": o} for i, o in zip((priv.get("input") or [])[:6], (priv.get("output") or [])[:6])]
            code = CODE.findall((r["generation"] or "").split("</think>")[-1])
            if len(tests) < 3 or not code or max(len(t["input"]) for t in tests) > 200_000:
                continue
            seen.add(r["id"])
            yield {"id": r["id"], "prompt": r["prompt"], "tests": tests, "ref": code[-1]}


def ok(c):
    return c if run_tests(c["ref"], c["tests"]) == len(c["tests"]) else None


if __name__ == "__main__":
    cs = list(cands(sys.argv[1]))
    random.Random(0).shuffle(cs)
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    out = []
    with Pool(8) as p:
        for c in p.imap(ok, cs[: n * 2], chunksize=4):
            if c:
                out.append({k: v for k, v in c.items() if k != "ref"})
            if len(out) >= n:
                break
    with open(sys.argv[2], "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    print("candidates", len(cs), "kept", len(out))

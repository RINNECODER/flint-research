"""HumanEval+ / MBPP+ subset for Flint, greedy with thinking, real test execution (evalplus).

  python eval_code.py gen   <model> <delta|none> <out.jsonl> [--he N] [--mbpp N] [--max-new T] [--bs B]
  python eval_code.py score <out.jsonl>     -> pass@1 base / plus on the subset

Same items for every model (fixed seed). A reply cut at --max-new counts as whatever code it holds (usually none).
"""
import json
import os
import random
import re
import subprocess
import sys
import time

TEMPLATE = None


def problems(n_he, n_mbpp):
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    he, mb = get_human_eval_plus(), get_mbpp_plus()
    pick = lambda d, n: sorted(random.Random(0).sample(sorted(d), n))
    return [(t, he[t], "humaneval") for t in pick(he, n_he)] + [(t, mb[t], "mbpp") for t in pick(mb, n_mbpp)]


def user_msg(p, ds):
    if ds == "humaneval":
        return ("Complete the following Python function. Reply with the complete function (with any imports it needs) "
                f"in a single ```python code block.\n\n```python\n{p['prompt'].strip()}\n```")
    return ("Solve this task in Python. Reply with the complete solution (with any imports it needs) in a single ```python "
            f"code block. Your function must pass this test:\n\n{p['prompt'].strip()}")


def extract(text):
    text = text.split("</think>")[-1]
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else ""


def gen(a):
    import torch
    _root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    sys.path[:0] = [os.path.join(_root, "model"), os.path.join(_root, "train")]
    import flint_torch as ft
    import esft
    from jinja2 import Environment
    from tokenizers import Tokenizer
    model_dir, delta, out = a[0], a[1], a[2]
    opt = dict(zip(a[3::2], a[4::2]))
    tmpl = Environment().from_string(open(f"{model_dir}/chat_template.jinja").read())
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    m = ft.load(model_dir, esft.DEV).eval()
    if delta != "none":
        from safetensors.torch import load_file
        dl = load_file(delta)
        for k in [k for k in dl if k.startswith("model.") and k.endswith(".weight")]:  # sft.py delta: checkpoint layout
            q = m.get_submodule(k[len("model."):-len(".weight")])
            p = k[:-len(".weight")]
            q.load(dl[k].to(q.w.device), dl[p + ".scales"].to(q.w.device), dl[p + ".biases"].to(q.w.device))
            q.bits = dl[k].shape[-1] * 8 // (dl[p + ".scales"].shape[-1] * ft.GROUP)  # 4 or 8
        for i, moe in esft.moes(m):
            if f"{i}.experts" not in dl:
                continue
            es = dl[f"{i}.experts"].tolist()
            for n in ("gate", "up", "down"):
                dev = getattr(moe, f"{n}_w").device
                getattr(moe, f"{n}_w")[es] = dl[f"{i}.{n}_proj.weight"].to(dev)
                getattr(moe, f"{n}_s")[es] = dl[f"{i}.{n}_proj.scales"].to(dev, ft.CDT)
                getattr(moe, f"{n}_b")[es] = dl[f"{i}.{n}_proj.biases"].to(dev, ft.CDT)
        esft.log("applied delta", delta)
    done = {json.loads(l)["task_id"] for l in open(out)} if os.path.exists(out) else set()
    items = [x for x in problems(int(opt.get("--he", 48)), int(opt.get("--mbpp", 48))) if x[0] not in done]
    prompts = [tok.encode(tmpl.render(messages=[{"role": "user", "content": user_msg(p, ds)}], add_generation_prompt=True),
                          add_special_tokens=False).ids for _, p, ds in items]
    order = sorted(range(len(items)), key=lambda i: len(prompts[i]))
    bs, max_new = int(opt.get("--bs", 32)), int(opt.get("--max-new", 4096))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        t = time.time()
        outs, capped = esft.generate(m, [prompts[i] for i in idx], max_new)
        with open(out, "a") as f:
            for i, o, c in zip(idx, outs, capped):
                tid, p, ds = items[i]
                text = tok.decode(o, skip_special_tokens=False)
                f.write(json.dumps({"task_id": tid, "dataset": ds, "tokens": len(o), "capped": c, "text": text,
                                    "solution": extract(text)}) + "\n")
        esft.log(f"batch {s // bs}: {len(idx)} items, {sum(map(len, outs))} tokens, {time.time() - t:.0f}s, capped {sum(capped)}")


def score(path):
    """Run evalplus' own checker; tasks outside the subset get an empty solution and are ignored in the tally."""
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    rows = [json.loads(l) for l in open(path)]
    res = {}
    for ds, allp in (("humaneval", get_human_eval_plus()), ("mbpp", get_mbpp_plus())):
        mine = {r["task_id"]: r for r in rows if r["dataset"] == ds}
        if not mine:
            continue
        sp = f"{path}.{ds}.samples.jsonl"
        with open(sp, "w") as f:
            for t in allp:
                f.write(json.dumps({"task_id": t, "solution": mine[t]["solution"] if t in mine else ""}) + "\n")
        rp = sp.replace(".jsonl", "_eval_results.json")
        if os.path.exists(rp):
            os.remove(rp)
        subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", ds, "--samples", sp, "--parallel", "16"],
                       check=True, stdout=subprocess.DEVNULL, stderr=open(f"{sp}.log", "w"))
        ev = json.load(open(rp))["eval"]
        for t, r in mine.items():
            e = ev[t][0]
            res[t] = {"base": e["base_status"] == "pass", "plus": e["base_status"] == "pass" and e["plus_status"] == "pass",
                      "tokens": r["tokens"], "capped": r["capped"]}
    json.dump(res, open(path + ".scores.json", "w"))
    n = len(res)
    print(json.dumps({"n": n, "base": sum(r["base"] for r in res.values()) / n, "plus": sum(r["plus"] for r in res.values()) / n,
                      "capped": sum(r["capped"] for r in res.values()), "mean_tokens": sum(r["tokens"] for r in res.values()) / n}))


if __name__ == "__main__":
    gen(sys.argv[2:]) if sys.argv[1] == "gen" else score(sys.argv[2])

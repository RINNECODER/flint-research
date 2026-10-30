"""Build the paper's figures from results/ in this repo.

Run from the repo root:  python paper/figs/make_figs.py
Writes paper/figs/speed_path.pdf and paper/figs/pruning.pdf.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
P = ROOT / "results" / "prune"

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, ACCENT = "#222222", "#9a9a9a", "#1f5fa8"


def speed(variant, mode):
    """decode_tps_mean from the JSON summary line of a speed log."""
    for line in (P / variant / f"speed_{mode}.log").read_text().splitlines():
        if line.startswith("{"):
            return json.loads(line)["decode_tps_mean"]
    raise ValueError(f"no summary in {variant}/speed_{mode}.log")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return c - h, c + h


# ---- Figure 1: the speed path -------------------------------------------------
base_on = [speed("base256", "on"), speed("base256-rerun", "on")]
reap_on = [speed("reap224", "on"), speed("reap224-rerun", "on")]
steps = [
    ("stock\nstreaming", 10, None, False),
    ("+ expert\npack", 22, None, False),
    ("+ 6 GB\nGPU cache", 28, None, False),
    ("+ DFlash 2\n(width 2)", sum(base_on) / 2, base_on, True),
    ("+ REAP-224", sum(reap_on) / 2, reap_on, True),
]
fig, ax = plt.subplots(figsize=(3.6, 2.3))
for i, (label, v, runs, logged) in enumerate(steps):
    ax.bar(i, v, width=0.62, color=ACCENT if logged else "white", edgecolor=ACCENT if logged else MUTED,
           hatch=None if logged else "///", linewidth=0.8)
    if runs:
        ax.plot([i, i], [min(runs), max(runs)], color=INK, lw=0.8)
        ax.scatter([i] * len(runs), runs, s=6, color=INK, zorder=3)
    ax.text(i, (max(runs) if runs else v) + 1.2, f"{v:.1f}" if logged else f"~{v:.0f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=6.5)
ax.set_ylabel("decode tok/s")
ax.set_ylim(0, 45)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="white", ec=MUTED, hatch="///", label="from notes (no log)"),
                   plt.Rectangle((0, 0), 1, 1, fc=ACCENT, label="logged: mean of 2 sessions")],
          fontsize=6, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "speed_path.pdf")

# ---- Figure 2: KL vs experts kept, and HumanEval+ pass@1 ----------------------
kl = {}
for block in (P / "kl_vs_256.txt").read_text().split("== ")[1:]:
    name = block.split()[0]
    kl[name] = float(re.search(r"KL\(ref\|\|variant\) ([\d.]+) CI \[([\d.]+), ([\d.]+)\]", block).group(1))
reap_kept = [224, 192, 160]
reap_kl = [kl["reap224"], kl["reap192"], kl["reap160"]]
he = {256: 36}
for n, v in zip(reap_kept, ["reap224", "reap192", "reap160"]):
    he[n] = json.loads((P / v / "score.log").read_text().splitlines()[0])["plus_pass"]

fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.2))
a.axhline(kl["256chunk"], color=MUTED, ls="--", lw=0.8)
a.text(157, kl["256chunk"] + 0.04, "noise floor (256, chunked prefill)", ha="right", fontsize=6, color=MUTED)
a.plot(reap_kept, reap_kl, "-o", color=ACCENT, ms=3, label="REAP")
for key, lab, mk in [("freq224", "frequency", "s"), ("rand224", "random", "x"), ("224", "RESMP RCO", "^")]:
    a.scatter([224], [kl[key]], marker=mk, s=18, color=INK, label=lab, zorder=3)
a.set_xlim(258, 155)
a.set_xlabel("routed experts kept per layer (of 256)")
a.set_ylabel("KL(256 || variant), nats/token")
a.set_ylim(0, 1.7)
a.legend(fontsize=6, frameon=False, loc="lower right", bbox_to_anchor=(1, 0.3))

ks = [256, 224, 192, 160]
vals = [he[k] / 40 for k in ks]
lo_hi = [wilson(he[k], 40) for k in ks]
b.errorbar(ks, vals, yerr=[[v - l for v, (l, _) in zip(vals, lo_hi)], [h - v for v, (_, h) in zip(vals, lo_hi)]],
           fmt="-o", color=ACCENT, ms=3, capsize=2, lw=0.9)
for k, v in zip(ks, vals):
    b.text(k, v + 0.03, f"{he[k]}/40", ha="center", fontsize=6)
b.set_xlim(262, 154)
b.set_ylim(0.5, 1.05)
b.set_xlabel("routed experts kept per layer (of 256)")
b.set_ylabel("HE+ plus pass@1 (n=40)")
fig.tight_layout()
fig.savefig(OUT / "pruning.pdf")
print("wrote", OUT / "speed_path.pdf", OUT / "pruning.pdf")

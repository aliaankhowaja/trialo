#!/usr/bin/env python3
"""Build all report figures from the real result artifacts in the repository."""
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "src" / "trailo" / "results" / "final_a100"
CKPT = REPO / "src" / "trailo" / "checkpoints" / "final_a100_seed42"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "lines.linewidth": 1.6, "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ---------------------------------------------------------------- training loss
rows = []
for name in ["best", "epoch-1", "epoch-2", "epoch-3"]:
    meta = json.loads((CKPT / name / "metadata.json").read_text())
    rows.append((meta["step"], meta["epoch"], meta["train_loss"], meta["val_loss"]))
rows.sort()
steps = [r[0] for r in rows]
tr = [r[2] for r in rows]
va = [r[3] for r in rows]

fig, ax = plt.subplots(figsize=(3.6, 2.3))
ax.plot(steps, tr, marker="o", label="train loss", color="#1e3a8a")
ax.plot(steps, va, marker="s", label="val loss (loss-masked)", color="#b45309")
best = min(rows, key=lambda r: r[3])
ax.annotate(f"best ckpt: epoch {best[1]}\nval {best[3]:.3f}",
            xy=(best[0], best[3]), xytext=(steps[0] + 18, best[3] + 0.006),
            fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("optimizer step"); ax.set_ylabel("loss")
ax.legend(fontsize=7.5, loc="center right")
fig.tight_layout(); fig.savefig(OUT / "train.png"); plt.close(fig)

# ------------------------------------------------------------- attention timing/memory
t = json.loads((RES / "task4" / "tiled_attention_benchmark.json").read_text())["timing"]
fp16 = [r for r in t if r["dtype"] == "fp16" and r["dim"] == 64]
seqs = sorted({r["seq"] for r in fp16})
series = {}
for impl in ["naive", "tiled", "sdpa_auto"]:
    series[impl] = {r["seq"]: r for r in fp16 if r["impl"] == impl}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
colors = {"naive": "#b91c1c", "tiled": "#1e3a8a", "sdpa_auto": "#15803d"}
ax = axes[0]
for impl, pts in series.items():
    ax.plot(seqs, [pts[s]["seconds_median"] * 1e3 for s in seqs],
            marker="o", label=impl, color=colors[impl])
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(seqs, [str(s) for s in seqs])
ax.set_xlabel("sequence length S"); ax.set_ylabel("median time per call (ms)")
ax.set_title("(a) speed", fontsize=9); ax.legend(fontsize=7.5)

ax = axes[1]
for impl, pts in series.items():
    ax.plot(seqs, [pts[s]["peak_delta_gb"] for s in seqs],
            marker="o", label=impl, color=colors[impl])
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(seqs, [str(s) for s in seqs])
ax.set_xlabel("sequence length S"); ax.set_ylabel("peak extra memory (GB)")
ax.set_title("(b) peak memory", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "attn.png"); plt.close(fig)

# ------------------------------------------------------------ speculative gamma sweep
spec = json.loads((RES / "task5" / "speculative_benchmark.json").read_text())
lat = spec["timing"]
gammas = [2, 3, 4, 6]
by_gamma_speedup, by_gamma_acc, by_gamma_tok = {}, {}, {}
for row in lat:
    if row.get("impl") != "spec":
        continue
    g = row.get("gamma"); x = row.get("speedup_vs_baseline")
    if g in gammas and isinstance(x, (int, float)) and x:
        by_gamma_speedup.setdefault(g, []).append(x)
    a = row.get("acceptance_rate")
    if g in gammas and isinstance(a, (int, float)):
        by_gamma_acc.setdefault(g, []).append(a)
    tok = row.get("tokens_per_target_forward")
    if g in gammas and isinstance(tok, (int, float)):
        by_gamma_tok.setdefault(g, []).append(tok)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
ax = axes[0]
bp = ax.boxplot([by_gamma_speedup.get(g, [0]) for g in gammas],
                tick_labels=[f"$\\gamma$={g}" for g in gammas], patch_artist=True)
for box in bp["boxes"]:
    box.set(facecolor="#dbeafe", edgecolor="#1e3a8a")
ax.axhline(1.0, color="#b91c1c", lw=1, ls="--")
ax.text(0.02, 1.02, "parity", color="#b91c1c", fontsize=7, transform=ax.get_yaxis_transform())
ax.set_ylabel("speedup vs.\ntarget-only greedy")
ax = axes[1]
ax.plot(gammas, [sum(v) / len(v) for v in by_gamma_tok.values()], marker="o", color="#1e3a8a", label="tokens per target forward")
ax.axhline(0.885, color="#b91c1c", ls="--", lw=1)
ax.text(4.1, 0.90, "draft cost share 0.885", color="#b91c1c", fontsize=7, ha="right")
ax.set_xlabel("draft length $\\gamma$"); ax.set_ylabel("tokens / target forward")
ax.legend(fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "spec.png"); plt.close(fig)
print("figures written to", OUT)

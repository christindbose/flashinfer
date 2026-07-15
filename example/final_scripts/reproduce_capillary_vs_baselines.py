#!/usr/bin/env python3
"""
Self-contained reproduction of capillary_vs_baselines.pdf from freshly measured
data. Reads a PAT kernel-benchmark JSON (pat / FastTree / fa / cascade
latencies) and the Capillary variant CSVs (geo_mean_ms), and renders the
PAT-normalized comparison (bars = pat_ms / method_ms; higher = faster).

Unlike plot_capillary_vs_baselines.py (which reads a pre-normalized April CSV
with hard-coded paths), this takes every input as an argument so it works on a
fresh H100/H200 run produced by reproduce_capillary_vs_baselines.sh.

Usage:
  python reproduce_capillary_vs_baselines.py \
      --pat-json  run/baselines.json \
      --cap-dir   run/ \
      --out       run/capillary_vs_baselines
The --cap-dir must contain the capillary sweep CSVs named exactly:
  bench_mech2_gqa_sweep_kvsplit_cluster{2,4,8}.csv, bench_pack_on.csv,
  bench_pack_kvsplit_cluster{2,4}.csv
Pass --pat-json multiple times to merge head configs measured in separate runs.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PANELS = [(32, 32), (32, 8), (16, 8)]
# baseline label -> latencies-dict key in the PAT JSON
BASELINES = [("PAT", "pat", "#444444"),
             ("FastTree", "FastTree", "#2a78d6"),
             ("FlashAttention3", "fa", "#e34948"),
             ("Cascade", "cascade", "#1baf7a")]
# capillary label -> csv basename, color
CAP = [
    ("kvsplit c2", "bench_mech2_gqa_sweep_kvsplit_cluster2.csv", "#eb6834"),
    ("kvsplit c4", "bench_mech2_gqa_sweep_kvsplit_cluster4.csv", "#eda100"),
    ("kvsplit c8", "bench_mech2_gqa_sweep_kvsplit_cluster8.csv", "#8c564b"),
    ("pack_gqa", "bench_pack_on.csv", "#008300"),
    ("kvsplit c2 + pack", "bench_pack_kvsplit_cluster2.csv", "#e87ba4"),
    ("kvsplit c4 + pack", "bench_pack_kvsplit_cluster4.csv", "#4a3aa7"),
]


def load_pat(paths):
    out = {}  # (hq,hkv,tree) -> {pat,FastTree,fa,cascade}
    for p in paths:
        for e in json.load(open(p)):
            out[(e["nheads_q"], e["nheads_kv"], e["tree"])] = e["latencies"]
    return out


def load_cap(path):
    out = {}
    if not path.exists():
        return out
    for r in csv.DictReader(open(path)):
        out[(int(r["num_qo_heads"]), int(r["num_kv_heads"]), r["tree_spec"])] = float(
            r["geo_mean_ms"])
    return out


def tree_key(t):
    nodes, rest = t.split("_")
    root, leaf = rest.split(",")
    return (int(nodes.split(",")[1]), int(root), int(leaf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pat-json", type=Path, nargs="+", required=True)
    ap.add_argument("--cap-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-fa", action="store_true", help="omit FlashAttention3")
    args = ap.parse_args()

    pat = load_pat(args.pat_json)
    caps = {label: load_cap(args.cap_dir / fn) for label, fn, _ in CAP}
    trees = sorted({k[2] for k in pat}, key=tree_key)

    baselines = [b for b in BASELINES if not (args.no_fa and b[1] == "fa")]
    series = ([(lab, col) for lab, _, col in baselines]
              + [(lab, col) for lab, _, col in CAP])
    n = len(series)
    width = 0.8 / n

    def speedup(qo, kv, t, label):
        base = pat.get((qo, kv, t))
        if base is None or base.get("pat") is None:
            return np.nan
        p = base["pat"]
        for lab, key, _ in BASELINES:
            if lab == label:
                v = base.get(key)
                return (p / v) if v else np.nan
        v = caps.get(label, {}).get((qo, kv, t))
        return (p / v) if v else np.nan

    color = {lab: col for lab, _, col in BASELINES}
    color.update({lab: col for lab, _, col in CAP})

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(15.5, 8.2), sharex=True)
    x = np.arange(len(trees))
    ymax = 1.08 * np.nanmax([speedup(qo, kv, t, lab)
                             for qo, kv in PANELS for t in trees for lab, _ in series] + [1.0])

    for ax, (qo, kv) in zip(axes, PANELS):
        for i, (lab, _) in enumerate(series):
            vals = [speedup(qo, kv, t, lab) for t in trees]
            ax.bar(x + (i - (n - 1) / 2) * width, vals, width * 0.92,
                   color=color[lab], zorder=3, label=lab if ax is axes[0] else None)
        ax.axhline(1.0, color="#555555", linestyle=(0, (4, 3)), linewidth=1.0, zorder=4)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.annotate(f"{qo} QO / {kv} KV heads (GQA={qo // kv})", (0.995, 0.94),
                    xycoords="axes fraction", ha="right", va="top", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left", ncol=n)
    axes[1].set_ylabel("speedup over PAT (PAT ms / method ms)")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(trees, rotation=60, ha="right", fontsize=7.5)
    axes[-1].set_xlabel("tree spec (nodes_root-KV,leaf-ctx)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = args.out.with_suffix(f".{ext}")
        fig.savefig(out, dpi=200)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

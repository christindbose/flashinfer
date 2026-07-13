"""
Per-tree comparison of capillary kvsplit cluster sizes on H100.

Three vertically stacked panels (one per head config, matching the
plot_normalized_vs_pat_combined.py layout), x = tree spec, bars = speedup of
kvsplit-on over kvsplit-off for cluster sizes 2 / 4 / 8 (higher = faster).

Inputs:
  bench_mech2_gqa_sweep_kvsplit_cluster{2,4,8}.csv
  bench_mech2_gqa_sweep.csv                        (kvsplit off)

Outputs:
  final_plots/kvsplit_cluster_tree_bars.pdf / .png
  final_plots/kvsplit_cluster_tree_bars.csv        (raw ms + speedups)
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "final_plots"
OUT_DIR.mkdir(exist_ok=True)

CLUSTERS = [2, 4, 8]
# (label, on-csv, baseline-csv). All series normalize against the pack-build
# no-split no-pack baseline (bench_pack_off.csv); it reproduces the old-build
# baseline within ~1% and the old 72-row CSV was overwritten by smoke runs.
SERIES = (
    [(f"kvsplit c{c}", f"bench_mech2_gqa_sweep_kvsplit_cluster{c}.csv",
      "bench_pack_off.csv") for c in CLUSTERS]
    + [("pack_gqa", "bench_pack_on.csv", "bench_pack_off.csv")]
    + [(f"kvsplit c{c} + pack", f"bench_pack_kvsplit_cluster{c}.csv",
        "bench_pack_off.csv") for c in CLUSTERS]
)
COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4"]
PANELS = [(32, 32), (32, 8), (16, 8)]  # (num_qo_heads, num_kv_heads), top -> bottom


def load(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (int(r["num_qo_heads"]), int(r["num_kv_heads"]), r["tree_spec"])
            out[key] = float(r["geo_mean_ms"])
    return out


def tree_sort_key(spec):
    nodes, rest = spec.split("_")
    root_kv, leaf_ctx = rest.split(",")
    return (int(nodes.split(",")[1]), int(root_kv), int(leaf_ctx))


def main():
    data = [(label, load(SCRIPT_DIR / on_csv), load(SCRIPT_DIR / base_csv))
            for label, on_csv, base_csv in SERIES]

    tree_sets = [set(k[2] for k in on) for _, on, _ in data]
    trees = sorted(set.intersection(*tree_sets), key=tree_sort_key)
    x = np.arange(len(trees))
    n_series = len(data)
    width = 0.8 / n_series

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(16.5, 8.2), sharex=True)

    y_max = 1.05 * max(base[(qo, kv, t)] / on[(qo, kv, t)]
                       for qo, kv in PANELS for t in trees for _, on, base in data)

    for ax, (qo, kv) in zip(axes, PANELS):
        for i, (label, on, base) in enumerate(data):
            speedups = [base[(qo, kv, t)] / on[(qo, kv, t)] for t in trees]
            ax.bar(x + (i - (n_series - 1) / 2) * width, speedups, width * 0.92,
                   color=COLORS[i], zorder=3, label=label if ax is axes[0] else None)
        ax.axhline(1.0, color="#888888", linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.annotate(f"{qo} QO / {kv} KV heads (GQA={qo // kv})", (0.995, 0.94),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=9, color="#333333")

    axes[0].legend(fontsize=9, frameon=False, loc="upper left", ncol=7)
    axes[1].set_ylabel("speedup over baseline (no split, no pack)")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(trees, rotation=60, ha="right", fontsize=7.5)
    axes[-1].set_xlabel("tree spec (nodes_root-KV,leaf-ctx)")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"kvsplit_cluster_tree_bars.{ext}"
        fig.savefig(out, dpi=300)
        print(f"wrote {out}")

    summary = OUT_DIR / "kvsplit_cluster_tree_bars.csv"
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        labels = [label for label, _, _ in data]
        w.writerow(["num_qo_heads", "num_kv_heads", "tree"]
                   + [f"{l}_ms" for l in labels] + [f"{l}_speedup" for l in labels])
        for qo, kv in PANELS:
            for t in trees:
                k = (qo, kv, t)
                w.writerow([qo, kv, t]
                           + [f"{on[k]:.6f}" for _, on, _ in data]
                           + [f"{base[k] / on[k]:.4f}" for _, on, base in data])
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()

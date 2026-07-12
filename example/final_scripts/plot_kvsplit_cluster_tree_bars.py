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
COLORS = {2: "#2a78d6", 4: "#1baf7a", 8: "#eda100"}
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
    off = load(SCRIPT_DIR / "bench_mech2_gqa_sweep.csv")
    on = {c: load(SCRIPT_DIR / f"bench_mech2_gqa_sweep_kvsplit_cluster{c}.csv") for c in CLUSTERS}

    trees = sorted({k[2] for k in off}, key=tree_sort_key)
    x = np.arange(len(trees))
    width = 0.26

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(11.5, 7.6), sharex=True)

    y_max = 1.05 * max(off[(qo, kv, t)] / on[c][(qo, kv, t)]
                       for qo, kv in PANELS for t in trees for c in CLUSTERS)

    for ax, (qo, kv) in zip(axes, PANELS):
        for i, c in enumerate(CLUSTERS):
            speedups = [off[(qo, kv, t)] / on[c][(qo, kv, t)] for t in trees]
            ax.bar(x + (i - 1) * width, speedups, width * 0.92, color=COLORS[c],
                   zorder=3, label=f"cluster {c}" if ax is axes[0] else None)
        ax.axhline(1.0, color="#888888", linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.annotate(f"{qo} QO / {kv} KV heads (GQA={qo // kv})", (0.995, 0.94),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=9, color="#333333")

    axes[0].legend(fontsize=9, frameon=False, loc="upper left", ncol=3)
    axes[1].set_ylabel("speedup over no split")
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
        w.writerow(["num_qo_heads", "num_kv_heads", "tree", "off_ms",
                    "c2_ms", "c4_ms", "c8_ms", "c2_speedup", "c4_speedup", "c8_speedup"])
        for qo, kv in PANELS:
            for t in trees:
                k = (qo, kv, t)
                w.writerow([qo, kv, t, f"{off[k]:.6f}",
                            *[f"{on[c][k]:.6f}" for c in CLUSTERS],
                            *[f"{off[k] / on[c][k]:.4f}" for c in CLUSTERS]])
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()

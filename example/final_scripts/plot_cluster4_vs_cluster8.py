"""
Side-by-side runtime comparison of Capillary at cluster size 4 vs cluster size 8.

Reads:
  bench_mech2_gqa_sweep_cluster4.csv
  bench_mech2_gqa_sweep_cluster8.csv

Writes to final_plots/:
  cluster4_vs_cluster8_runtime.pdf / .png    raw geo_mean_ms bars, side by side
  cluster4_vs_cluster8_ratio.pdf   / .png    per-tree speedup (c4 / c8), >1 means c8 wins
  cluster4_vs_cluster8.csv                   joined table
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
C4_CSV = SCRIPT_DIR / "bench_mech2_gqa_sweep_cluster4.csv"
C8_CSV = SCRIPT_DIR / "bench_mech2_gqa_sweep_cluster8.csv"
OUT_DIR = SCRIPT_DIR / "final_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_ORDER = [(32, 32, 1), (32, 8, 4), (16, 8, 2)]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 18,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
})


def load_sweep(path):
    """Return dict: (qo, kv, tree_spec) -> geo_mean_ms."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["num_qo_heads"]), int(row["num_kv_heads"]), row["tree_spec"])
            out[key] = float(row["geo_mean_ms"])
    return out


def tree_sort_key(t):
    nodes_part, ctx_part = t.split("_", 1)
    leaf_count = int(nodes_part.split(",")[1])
    prefix_ctx = int(ctx_part.split(",")[0])
    leaf_ctx = int(ctx_part.split(",")[1])
    return (leaf_count, prefix_ctx, leaf_ctx)


def fmt_tree(t):
    nodes_part, ctx_part = t.split("_", 1)
    return f"B=[{nodes_part}] L=[{ctx_part}]"


def collect_panel(c4, c8, qo, kv):
    trees = sorted({t for (q, k, t) in c4.keys() if (q, k) == (qo, kv)} |
                   {t for (q, k, t) in c8.keys() if (q, k) == (qo, kv)},
                   key=tree_sort_key)
    v4 = [c4.get((qo, kv, t), np.nan) for t in trees]
    v8 = [c8.get((qo, kv, t), np.nan) for t in trees]
    return trees, np.array(v4), np.array(v8)


def render_runtime(panels_data):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10), dpi=200)
    bar_w = 0.38
    for ax_idx, ((qo, kv, gqa), (trees, v4, v8)) in enumerate(zip(PANEL_ORDER, panels_data)):
        ax = axes[ax_idx]
        x = np.arange(len(trees))
        ax.bar(x - bar_w / 2, v4, bar_w, color="#2850AA", edgecolor="black", linewidth=0.4,
               label="Cluster=4")
        ax.bar(x + bar_w / 2, v8, bar_w, color="#DC3232", edgecolor="black", linewidth=0.4,
               label="Cluster=8")
        ax.tick_params(axis="y", labelsize=16)
        ax.grid(axis="y", linestyle=":", alpha=0.5, color="lightgray")
        ax.set_axisbelow(True)
        ax.set_title(f"qo={qo}, kv={kv} (GQA={gqa})", fontsize=16, loc="left")
        ax.set_ylabel("geo_mean_ms", fontsize=16)
        ax.set_xticks(x)
        if ax_idx == 2:
            ax.set_xticklabels([fmt_tree(t) for t in trees], rotation=55, ha="right", fontsize=12)
        else:
            ax.set_xticklabels([])
        if ax_idx == 0:
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97),
               ncol=2, fontsize=16, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"cluster4_vs_cluster8_runtime.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")
    plt.close(fig)


def render_ratio(panels_data):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10), dpi=200)
    for ax_idx, ((qo, kv, gqa), (trees, v4, v8)) in enumerate(zip(PANEL_ORDER, panels_data)):
        ax = axes[ax_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = v4 / v8  # >1 => cluster=8 is faster
        x = np.arange(len(trees))
        colors = np.where(ratio > 1.0, "#2BA02B", "#DC3232")
        ax.bar(x, ratio, 0.7, color=colors, edgecolor="black", linewidth=0.4)
        ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.tick_params(axis="y", labelsize=16)
        ax.grid(axis="y", linestyle=":", alpha=0.5, color="lightgray")
        ax.set_axisbelow(True)
        ax.set_title(f"qo={qo}, kv={kv} (GQA={gqa})   green=c8 faster, red=c4 faster",
                     fontsize=15, loc="left")
        ax.set_ylabel("c4 / c8", fontsize=16)
        ymax = float(np.nanmax(ratio)) if np.any(~np.isnan(ratio)) else 1.5
        ax.set_ylim(0, max(1.2, ymax * 1.15))
        ax.set_xticks(x)
        if ax_idx == 2:
            ax.set_xticklabels([fmt_tree(t) for t in trees], rotation=55, ha="right", fontsize=12)
        else:
            ax.set_xticklabels([])

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"cluster4_vs_cluster8_ratio.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")
    plt.close(fig)


def main():
    c4 = load_sweep(C4_CSV)
    c8 = load_sweep(C8_CSV)

    panels_data = [collect_panel(c4, c8, qo, kv) for (qo, kv, _) in PANEL_ORDER]

    render_runtime(panels_data)
    render_ratio(panels_data)

    # Joined table
    summary_csv = OUT_DIR / "cluster4_vs_cluster8.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_qo_heads", "num_kv_heads", "gqa", "tree_spec",
                    "cluster4_ms", "cluster8_ms", "ratio_c4_over_c8", "c8_faster"])
        wins = defaultdict(lambda: [0, 0])
        for (qo, kv, gqa), (trees, v4, v8) in zip(PANEL_ORDER, panels_data):
            for i, t in enumerate(trees):
                a, b = v4[i], v8[i]
                if np.isnan(a) or np.isnan(b):
                    continue
                r = a / b
                c8_faster = r > 1.0
                w.writerow([qo, kv, gqa, t, f"{a:.6f}", f"{b:.6f}", f"{r:.4f}",
                            "yes" if c8_faster else "no"])
                wins[(qo, kv)][0 if c8_faster else 1] += 1
    print(f"wrote {summary_csv}")

    print("\nPer-config win counts (c8_faster, c4_faster):")
    for (qo, kv), (w8, w4) in wins.items():
        print(f"  qo={qo:2} kv={kv:2}:  c8_faster={w8:3}  c4_faster={w4:3}  "
              f"({100.0 * w8 / (w8 + w4):.1f}% c8)")


if __name__ == "__main__":
    main()

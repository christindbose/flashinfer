"""
Capillary variants vs external baselines, PAT-normalized.

Bars = PAT_ms / method_ms (higher = faster than PAT), three stacked panels
(one per head config), x = tree spec. Baselines (PAT, FastTree,
FlashAttention3, Cascade) come from the April measurement set
(results/normalized_vs_pat_combined.csv); capillary variants (kvsplit c4,
pack_gqa, kvsplit c4 + pack) from final_plots/kvsplit_cluster_tree_bars.csv.
Cross-set ms comparison is valid: today's kvsplit c4 reproduces the April
capillary latencies within ~1%.

Outputs:
  final_plots/capillary_vs_baselines.pdf / .png          (with FA3)
  final_plots/capillary_vs_baselines_no_fa.pdf / .png    (FA3 omitted)
  final_plots/capillary_vs_baselines.csv
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "final_plots"
APRIL_CSV = Path(
    "/home/chris241/myflashinfer_old/flashinfer/results/normalized_vs_pat_combined.csv"
)
# Each capillary variant reads its own sweep CSV (geo_mean_ms per config).
OURS_CSVS = {
    "kvsplit c2": SCRIPT_DIR / "bench_mech2_gqa_sweep_kvsplit_cluster2.csv",
    "kvsplit c4": SCRIPT_DIR / "bench_mech2_gqa_sweep_kvsplit_cluster4.csv",
    "kvsplit c8": SCRIPT_DIR / "bench_mech2_gqa_sweep_kvsplit_cluster8.csv",
    "pack_gqa": SCRIPT_DIR / "bench_pack_on.csv",
    "kvsplit c2 + pack": SCRIPT_DIR / "bench_pack_kvsplit_cluster2.csv",
    "kvsplit c4 + pack": SCRIPT_DIR / "bench_pack_kvsplit_cluster4.csv",
}

PANELS = [(32, 32), (32, 8), (16, 8)]

# (label, color); PAT bars are exactly 1.0 by construction.
BASELINES = [
    ("PAT", "#444444"),
    ("FastTree", "#2a78d6"),
    ("FlashAttention3", "#e34948"),
    ("Cascade", "#1baf7a"),
]
OURS = [
    ("kvsplit c2", "#eb6834"),
    ("kvsplit c4", "#eda100"),
    ("kvsplit c8", "#8c564b"),
    ("pack_gqa", "#008300"),
    ("kvsplit c2 + pack", "#e87ba4"),
    ("kvsplit c4 + pack", "#4a3aa7"),
]
APRIL_COL = {"PAT": "pat_ms", "FastTree": "FastTree_ms", "FlashAttention3": "fa_ms",
             "Cascade": "cascade_ms"}


def load_april():
    out = {}
    with open(APRIL_CSV) as f:
        for r in csv.DictReader(f):
            key = (int(r["panel_qo"]), int(r["panel_kv"]), r["tree"])
            out[key] = r
    return out


def load_ours():
    out = {}  # (qo, kv, tree) -> {label: ms}
    for label, path in OURS_CSVS.items():
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (int(r["num_qo_heads"]), int(r["num_kv_heads"]), r["tree_spec"])
                out.setdefault(key, {})[f"{label}_ms"] = r["geo_mean_ms"]
    return out


def tree_sort_key(spec):
    nodes, rest = spec.split("_")
    root_kv, leaf_ctx = rest.split(",")
    return (int(nodes.split(",")[1]), int(root_kv), int(leaf_ctx))


def make_plot(april, ours, trees, include_fa, suffix):
    baselines = [b for b in BASELINES if include_fa or b[0] != "FlashAttention3"]
    series = baselines + OURS
    n_series = len(series)
    width = 0.8 / n_series

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(15.5, 8.2), sharex=True)
    x = np.arange(len(trees))

    def speedup(qo, kv, t, label):
        k = (qo, kv, t)
        try:
            pat = float(april[k]["pat_ms"])
            ms = (float(april[k][APRIL_COL[label]]) if label in APRIL_COL
                  else float(ours[k][f"{label}_ms"]))
        except (ValueError, KeyError):
            return float("nan")  # missing April entry (X in the original plot)
        return pat / ms

    all_vals = [speedup(qo, kv, t, lab)
                for qo, kv in PANELS for t in trees for lab, _ in series]
    y_max = 1.08 * max(v for v in all_vals if v == v)

    for ax, (qo, kv) in zip(axes, PANELS):
        for i, (label, color) in enumerate(series):
            vals = [speedup(qo, kv, t, label) for t in trees]
            ax.bar(x + (i - (n_series - 1) / 2) * width, vals, width * 0.92,
                   color=color, zorder=3, label=label if ax is axes[0] else None)
        ax.axhline(1.0, color="#555555", linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.annotate(f"{qo} QO / {kv} KV heads (GQA={qo // kv})", (0.995, 0.94),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=9, color="#333333")

    axes[0].legend(fontsize=9, frameon=False, loc="upper left", ncol=n_series)
    axes[1].set_ylabel("speedup over PAT (PAT ms / method ms)")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(trees, rotation=60, ha="right", fontsize=7.5)
    axes[-1].set_xlabel("tree spec (nodes_root-KV,leaf-ctx)")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"capillary_vs_baselines{suffix}.{ext}"
        fig.savefig(out, dpi=300)
        print(f"wrote {out}")
    plt.close(fig)


def main():
    april = load_april()
    ours = load_ours()
    trees = sorted({k[2] for k in april} & {k[2] for k in ours}, key=tree_sort_key)

    make_plot(april, ours, trees, include_fa=True, suffix="")
    make_plot(april, ours, trees, include_fa=False, suffix="_no_fa")

    summary = OUT_DIR / "capillary_vs_baselines.csv"
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        labels = [b[0] for b in BASELINES] + [o[0] for o in OURS]
        w.writerow(["num_qo_heads", "num_kv_heads", "tree", "pat_ms"]
                   + [f"{l}_speedup_vs_pat" for l in labels])
        for qo, kv in PANELS:
            for t in trees:
                k = (qo, kv, t)
                pat = float(april[k]["pat_ms"])
                row = [qo, kv, t, f"{pat:.6f}"]
                for l in labels:
                    try:
                        ms = (float(april[k][APRIL_COL[l]]) if l in APRIL_COL
                              else float(ours[k][f"{l}_ms"]))
                        row.append(f"{pat / ms:.4f}")
                    except (ValueError, KeyError):
                        row.append("")
                w.writerow(row)
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()

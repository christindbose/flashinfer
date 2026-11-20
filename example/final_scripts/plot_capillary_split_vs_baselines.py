"""
Capillary kvsplit+pack variants vs external baselines, PAT-normalized.

Same layout as plot_capillary_best_vs_baselines.py, but instead of a single
combined "Capillary" (best-of) series, this plots "Capillary_2" (kvsplit c2 +
pack) and "Capillary_4" (kvsplit c4 + pack) individually so you can see which
cluster size wins per point.

Baselines (PAT, FastTree, FlashAttention3, Cascade) come from the April
measurement set (results/normalized_vs_pat_combined.csv); the capillary
variants come from the per-variant sweep CSVs. Cross-set ms comparison is
valid: today's kvsplit reproduces the April capillary latencies within ~1%.

Outputs:
  final_plots/capillary_split_vs_baselines.pdf / .png          (with FA3)
  final_plots/capillary_split_vs_baselines_no_fa.pdf / .png    (FA3 omitted)
  final_plots/capillary_split_vs_baselines.csv
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "final_plots"
APRIL_CSV = Path(
    "/home/chris241/myflashinfer_old/flashinfer/results/normalized_vs_pat_combined.csv"
)

VARIANT_CSVS = {
    "kvsplit c2 + pack": SCRIPT_DIR / "bench_pack_kvsplit_cluster2.csv",
    "kvsplit c4 + pack": SCRIPT_DIR / "bench_pack_kvsplit_cluster4.csv",
}

PANELS = [(32, 32), (32, 8), (16, 8)]

# (label, color); PAT bars are exactly 1.0 by construction. Palette matches
# results/plot_normalized_vs_pat_combined.py. Hatches make each series tell
# apart in grayscale (e.g. printed) in addition to color.
BASELINES = [
    ("PAT", "#444444", ""),
    ("FastTree", "#1f77b4", "////"),
    ("FlashAttention3", "#d62728", "...."),
    ("Cascade", "#2ca02c", "\\\\\\\\"),
]
SPLIT_SERIES = [
    ("Capillary_2", "kvsplit c2 + pack", "#e08214", "xxxx"),
    ("Capillary_4", "kvsplit c4 + pack", "#8073ac", "oooo"),
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


def load_variant(path):
    """(qo, kv, tree) -> geo_mean_ms for a single variant CSV."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (int(r["num_qo_heads"]), int(r["num_kv_heads"]), r["tree_spec"])
            try:
                out[key] = float(r["geo_mean_ms"])
            except ValueError:
                continue
    return out


def tree_sort_key(spec):
    nodes, rest = spec.split("_")
    root_kv, leaf_ctx = rest.split(",")
    return (int(nodes.split(",")[1]), int(root_kv), int(leaf_ctx))


def make_plot(april, variants, trees, include_fa, suffix):
    plt.rcParams.update({"font.size": 20})
    baselines = [b for b in BASELINES if include_fa or b[0] != "FlashAttention3"]
    split_series = [(label, color, hatch) for label, _, color, hatch in SPLIT_SERIES]
    series = baselines + split_series
    n_series = len(series)
    bar_w = 0.8 / n_series
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * bar_w

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(17, 10), dpi=200, sharex=True)
    x = np.arange(len(trees))

    variant_ms = {label: variants[csv_key] for label, csv_key, _, _ in SPLIT_SERIES}

    def speedup(qo, kv, t, label):
        k = (qo, kv, t)
        try:
            pat = float(april[k]["pat_ms"])
            if label in variant_ms:
                ms = variant_ms[label][k]
            else:
                ms = float(april[k][APRIL_COL[label]])
        except (ValueError, KeyError):
            return float("nan")
        return pat / ms

    # Per-panel y limit: GQA=1 panel tops out below 2; others use their own max.
    def panel_ymax(pi, qo, kv):
        if pi == 0:
            return 2.0
        vals = [speedup(qo, kv, t, lab) for t in trees for lab, _, _ in series]
        m = max([v for v in vals if v == v], default=1.0)
        return m * 1.12

    legend_handles = None
    for pi, (ax, (qo, kv)) in enumerate(zip(axes, PANELS)):
        for off, (label, color, hatch) in zip(offsets, series):
            vals = np.array([speedup(qo, kv, t, label) for t in trees], dtype=float)
            ax.bar(x + off, np.nan_to_num(vals, nan=0.0), bar_w,
                   color=color, hatch=hatch, edgecolor="black", linewidth=0.4)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.tick_params(axis="y", labelsize=20)
        ax.set_ylim(0, panel_ymax(pi, qo, kv))
        # Bottom two panels: horizontal gridlines at 1, 2, 3.
        if pi in (1, 2):
            ax.set_yticks([1, 2, 3])
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.annotate(f"{qo} Q / {kv} KV heads (GQA={qo // kv})", (0.995, 0.93),
                    xycoords="axes fraction", ha="right", va="top", fontsize=16)
        if pi == 0:
            legend_handles = ([Patch(facecolor=c, edgecolor="black", hatch=h, linewidth=0.4)
                                for _, c, h in series],
                               [lbl for lbl, _, _ in series])

    fig.supylabel("Speedup normalized to PAT", fontsize=20, x=0.075, y=0.58)

    def fmt_tree(spec):
        nodes_part, ctx_part = spec.split("_", 1)
        return f"B=[{nodes_part}] L=[{ctx_part}]"
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([fmt_tree(t) for t in trees],
                             rotation=55, ha="right", fontsize=14)

    if legend_handles is not None:
        handles, labels = legend_handles
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.57, 0.90),
                   ncol=len(labels), fontsize=18, frameon=False, borderaxespad=0.0,
                   handlelength=1.4, handletextpad=0.45, columnspacing=1.1)

    fig.tight_layout(rect=(0.02, 0, 1.0, 0.90))
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"capillary_split_vs_baselines{suffix}.{ext}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.1)
        print(f"wrote {out}")
    plt.close(fig)


def main():
    april = load_april()
    variants = {csv_key: load_variant(path)
                for csv_key, path in VARIANT_CSVS.items()}
    common_keys = set.intersection(*(set(v) for v in variants.values()))
    trees = sorted({k[2] for k in april} & {k[2] for k in common_keys},
                   key=tree_sort_key)

    make_plot(april, variants, trees, include_fa=True, suffix="")
    make_plot(april, variants, trees, include_fa=False, suffix="_no_fa")

    summary = OUT_DIR / "capillary_split_vs_baselines.csv"
    labels = [b[0] for b in BASELINES] + [label for label, _, _, _ in SPLIT_SERIES]
    variant_ms = {label: variants[csv_key] for label, csv_key, _, _ in SPLIT_SERIES}
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_qo_heads", "num_kv_heads", "tree", "pat_ms"]
                   + [f"{l}_speedup_vs_pat" for l in labels])
        for qo, kv in PANELS:
            for t in trees:
                k = (qo, kv, t)
                pat = float(april[k]["pat_ms"])
                row = [qo, kv, t, f"{pat:.6f}"]
                for l in labels:
                    try:
                        ms = (variant_ms[l][k] if l in variant_ms
                              else float(april[k][APRIL_COL[l]]))
                        row.append(f"{pat / ms:.4f}")
                    except (ValueError, KeyError):
                        row.append("")
                w.writerow(row)
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()

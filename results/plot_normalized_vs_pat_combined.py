"""
Combined view: three GQA configs in one figure, vertically stacked, sharing
the x-axis (tree spec) and a single y-axis label.

Panels (top -> bottom):
  1) num_qo_heads=32, num_kv_heads=32, GQA=1   (PAT JSON 20260428_170148)
  2) num_qo_heads=32, num_kv_heads=8,  GQA=4   (PAT JSON 20260428_200506)
  3) num_qo_heads=16, num_kv_heads=8,  GQA=2   (PAT JSON 20260428_200506)

Capillary lookup keys on (num_qo_heads, num_kv_heads, tree_spec). Missing
entries get an X marker. Bars are PAT_lat / method_lat (higher = faster).

Output:
  bars_normalized_vs_pat_combined.png         (with FA)
  bars_normalized_vs_pat_combined_no_fa.png   (FA omitted)
  normalized_vs_pat_combined.csv              (raw + normalized)
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAT_JSON_RUN1 = Path(
    "/home/chris241/myflashinfer_old/baselines/PAT_workspace/PAT/benchmark/"
    "final_results/kernel_perf_20260428_170148/kernel_perf_20260428_170148.json"
)
PAT_JSON_RUN2 = Path(
    "/home/chris241/myflashinfer_old/baselines/PAT_workspace/PAT/benchmark/"
    "final_results/kernel_perf_20260428_200506/kernel_perf_20260428_200506.json"
)
CAPILLARY_CSV = Path("/home/chris241/myflashinfer_old/flashinfer/bench_mech2_gqa_sweep.csv")
OUT_DIR = Path("/home/chris241/myflashinfer_old/flashinfer/results")

METHODS = ["pat", "FastTree", "fa", "cascade", "capillary"]
LABELS = {
    "pat": "PAT",
    "FastTree": "FastTree",
    "fa": "FlashAttention3",
    "cascade": "Cascade",
    "capillary": "Capillary",
}
COLORS = {
    "pat": "#444444",
    "FastTree": "#1f77b4",
    "fa": "#d62728",
    "cascade": "#2ca02c",
    "capillary": "#ff7f0e",
}


def load_pat(path):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for entry in data:
        key = (entry["nheads_q"], entry["nheads_kv"])
        out.setdefault(key, {})[entry["tree"]] = entry["latencies"]
    return out


def load_capillary(path):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["num_qo_heads"]), int(row["num_kv_heads"]), row["tree_spec"])
            out[key] = float(row["geo_mean_ms"])
    return out


def compute_norm(pat_for_cfg, capillary, hq, hkv, trees):
    norm = {m: [] for m in METHODS}
    for t in trees:
        pat_lat = pat_for_cfg[t]["pat"]
        for m in METHODS:
            if m == "pat":
                v = 1.0
            elif m == "capillary":
                cap_lat = capillary.get((hq, hkv, t))
                v = (pat_lat / cap_lat) if cap_lat is not None else np.nan
            else:
                lat = pat_for_cfg[t].get(m)
                v = (pat_lat / lat) if lat is not None else np.nan
            norm[m].append(v)
    return norm


def main():
    pat_run1 = load_pat(PAT_JSON_RUN1)
    pat_run2 = load_pat(PAT_JSON_RUN2)
    capillary = load_capillary(CAPILLARY_CSV)

    panels = [
        # (hq, hkv, gqa, pat_for_cfg)
        (32, 32, 1, pat_run1[(32, 32)]),
        (32, 8,  4, pat_run2[(32, 8)]),
        (16, 8,  2, pat_run2[(16, 8)]),
    ]

    # Tree axis: union of trees that appear in any panel.
    all_trees_set = set()
    for _, _, _, p in panels:
        all_trees_set.update(p.keys())
    trees = sorted(
        all_trees_set,
        key=lambda t: (
            int(t.split("_")[0].split(",")[1]),     # leaf count
            int(t.split("_")[1].split(",")[0]),     # prefix context
            int(t.split("_")[1].split(",")[1]),     # leaf context
        ),
    )

    panel_norms = []
    for hq, hkv, gqa, pat_for_cfg in panels:
        # Use NaN for trees missing in this panel's PAT data.
        per_tree = []
        for t in trees:
            if t in pat_for_cfg:
                per_tree.append(t)
            else:
                per_tree.append(None)
        norm = {m: [] for m in METHODS}
        for t in per_tree:
            if t is None:
                for m in METHODS:
                    norm[m].append(np.nan)
                continue
            pat_lat = pat_for_cfg[t]["pat"]
            for m in METHODS:
                if m == "pat":
                    v = 1.0
                elif m == "capillary":
                    cap_lat = capillary.get((hq, hkv, t))
                    v = (pat_lat / cap_lat) if cap_lat is not None else np.nan
                else:
                    lat = pat_for_cfg[t].get(m)
                    v = (pat_lat / lat) if lat is not None else np.nan
                norm[m].append(v)
        panel_norms.append(norm)

    n_trees = len(trees)
    bar_w = 0.16
    x = np.arange(n_trees)

    def render(include_fa, fname, title_suffix):
        plt.rcParams.update({"font.size": 20})
        # Total figure: (15, 10) at dpi=200 -> 3000 x 2000 px.
        fig, axes = plt.subplots(
            nrows=3, ncols=1,
            figsize=(15, 10), dpi=200,
            sharex=True,
        )
        active = [m for m in METHODS if include_fa or m != "fa"]
        offsets = (np.arange(len(active)) - (len(active) - 1) / 2) * bar_w

        # Per-panel y limit: panel 0 -> 2.0, panel 1 -> 2.2, panel 2 -> dynamic.
        per_panel_ylim = []
        for ax_idx, norm in enumerate(panel_norms):
            if ax_idx == 0:
                per_panel_ylim.append(2.0)
            elif ax_idx == 1:
                per_panel_ylim.append(2.2)
            else:
                ymax_local = 1.0
                for m in active:
                    vals = np.array(norm[m], dtype=float)
                    if np.any(~np.isnan(vals)):
                        ymax_local = max(ymax_local, float(np.nanmax(vals)))
                per_panel_ylim.append(ymax_local * 1.18)

        legend_handles = None
        for ax_idx, ((hq, hkv, gqa, _), norm) in enumerate(zip(panels, panel_norms)):
            ax = axes[ax_idx]
            for offset, m in zip(offsets, active):
                vals = np.array(norm[m], dtype=float)
                mask = np.isnan(vals)
                plot_vals = np.where(mask, 0.0, vals)
                ax.bar(
                    x + offset, plot_vals, bar_w,
                    color=COLORS[m], label=LABELS[m],
                    edgecolor="black", linewidth=0.4,
                )

            ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.tick_params(axis="y", labelsize=20)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            ax.set_ylim(0, per_panel_ylim[ax_idx])
            if ax_idx == 2:
                ax.set_yticks([1, 2, 3])
            if ax_idx == 0:
                legend_handles = ax.get_legend_handles_labels()

        # Single shared y-axis label on the figure
        fig.supylabel("Speedup normalized to PAT", fontsize=20,x=0.085,y=0.58)

        # x-axis labels only on bottom panel (sharex hides the others); no xlabel.
        # Decompose "n1,n2_c1,c2" into "B=[n1,n2] L=[c1,c2]".
        def fmt_tree(spec):
            nodes_part, ctx_part = spec.split("_", 1)
            return f"B=[{nodes_part}] L=[{ctx_part}]"
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([fmt_tree(t) for t in trees],
                                 rotation=55, ha="right", fontsize=14)

        # Single horizontal legend in one row above the entire plot
        if legend_handles is not None:
            handles, labels = legend_handles
            fig.legend(
                handles, labels,
                loc="upper center", bbox_to_anchor=(0.57, 0.95),
                ncol=len(labels),
                fontsize=20, frameon=False, borderaxespad=0.0,
            )

        # Leave room at the top for the horizontal legend
        fig.tight_layout(rect=(0.02, 0, 1.0, 0.93))
        out_path = OUT_DIR / fname
        fig.savefig(out_path)
        print(f"Wrote {out_path}")
        plt.close(fig)

    render(True,  "bars_normalized_vs_pat_combined.pdf",       "")
    render(False, "bars_normalized_vs_pat_combined_no_fa.pdf", " (FA omitted)")

    # Combined raw+normalized CSV
    summary_csv = OUT_DIR / "normalized_vs_pat_combined.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["panel_qo", "panel_kv", "panel_gqa", "tree", "pat_ms"]
            + [f"{m}_ms" for m in METHODS if m != "pat"]
            + [f"{m}_norm" for m in METHODS]
        )
        for (hq, hkv, gqa, pat_for_cfg), norm in zip(panels, panel_norms):
            for i, t in enumerate(trees):
                if t not in pat_for_cfg:
                    continue
                row = [hq, hkv, gqa, t, pat_for_cfg[t]["pat"]]
                for m in METHODS:
                    if m == "pat":
                        continue
                    if m == "capillary":
                        row.append(capillary.get((hq, hkv, t), ""))
                    else:
                        row.append(pat_for_cfg[t].get(m, ""))
                for m in METHODS:
                    v = norm[m][i]
                    row.append("" if (isinstance(v, float) and np.isnan(v))
                               else f"{v:.4f}")
                w.writerow(row)
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()

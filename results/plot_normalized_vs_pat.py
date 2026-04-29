"""
Build a grouped-bar chart of attention-kernel latencies, normalized to PAT=1.0.

Inputs:
  - PAT JSON: per-tree latencies for pat, FastTree, fa, cascade
  - Mech2-off CSV: per-tree geo_mean_ms for the modified flashinfer cascade
    (test/bench with mech2_mode=False)

Trees are matched by tree_spec (e.g. "1,64_16384,32"). All bars are
latency_method / latency_pat. PAT is shown explicitly at 1.0.

Output: bars_normalized_vs_pat.png + bars_normalized_vs_pat_no_fa.png
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAT_JSON = Path(
    "/home/chris241/myflashinfer_old/baselines/PAT_workspace/PAT/benchmark/"
    "final_results/kernel_perf_20260428_170148/kernel_perf_20260428_170148.json"
)
MECH2OFF_CSV = Path("/home/chris241/myflashinfer_old/flashinfer/bench_mech2_gqa_sweep.csv")
OUT_DIR = Path("/home/chris241/myflashinfer_old/flashinfer/results")


def load_pat(path):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for entry in data:
        out[entry["tree"]] = entry["latencies"]
    return out


def load_mech2off(path):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["tree_spec"]] = float(row["geo_mean_ms"])
    return out


def main():
    pat_data = load_pat(PAT_JSON)
    mech2off = load_mech2off(MECH2OFF_CSV)

    trees = [t for t in pat_data if t in mech2off]

    methods = ["pat", "FastTree", "fa", "cascade", "capillary"]
    labels = {
        "pat": "PAT (baseline)",
        "FastTree": "FastTree",
        "fa": "FlashAttention",
        "cascade": "Cascade",
        "capillary": "Capillary",
    }
    colors = {
        "pat": "#444444",
        "FastTree": "#1f77b4",
        "fa": "#d62728",
        "cascade": "#2ca02c",
        "capillary": "#ff7f0e",
    }

    # Speedup over PAT: pat_latency / method_latency. Higher = faster than PAT.
    norm = {m: [] for m in methods}
    for t in trees:
        pat_lat = pat_data[t]["pat"]
        for m in methods:
            if m == "pat":
                v = 1.0
            elif m == "capillary":
                v = pat_lat / mech2off[t]
            else:
                lat = pat_data[t].get(m)
                v = (pat_lat / lat) if lat is not None else np.nan
            norm[m].append(v)

    n_trees = len(trees)
    n_methods = len(methods)
    bar_w = 0.16
    x = np.arange(n_trees)

    def render(include_fa, fname, title_suffix):
        plt.rcParams.update({"font.size": 16})
        fig, ax = plt.subplots(figsize=(max(22, 1.2 * n_trees), 9))
        active = [m for m in methods if include_fa or m != "fa"]
        offsets = (np.arange(len(active)) - (len(active) - 1) / 2) * bar_w
        for offset, m in zip(offsets, active):
            vals = np.array(norm[m], dtype=float)
            mask = np.isnan(vals)
            plot_vals = np.where(mask, 0.0, vals)
            ax.bar(
                x + offset,
                plot_vals,
                bar_w,
                color=colors[m],
                label=labels[m],
                edgecolor="black",
                linewidth=0.4,
            )
            if np.any(mask):
                miss_x = (x + offset)[mask]
                ax.scatter(miss_x, np.full_like(miss_x, 0.15, dtype=float),
                           marker="x", s=300, color=colors[m],
                           linewidths=4.0, zorder=5)

        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(trees, rotation=55, ha="right", fontsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_xlabel("Tree structure <num nodes>_<KV context length>", fontsize=16)
        ax.set_ylabel("Speedup over PAT (higher = faster)",
                      fontsize=16)
        fig.suptitle("num_qo_heads=32  |  num_kv_heads=32  |  GQA ratio=1",
                     fontsize=20, fontweight="bold", y=0.995)
        ax.set_title("head_dim=128", fontsize=14)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                  fontsize=16, frameon=True, borderaxespad=0.0)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ymax = 1.0
        for m in active:
            vals = np.array(norm[m], dtype=float)
            if np.any(~np.isnan(vals)):
                ymax = max(ymax, np.nanmax(vals))
        ax.set_ylim(0, ymax * 1.18)
        fig.tight_layout(rect=(0, 0, 0.92, 1))
        out_path = OUT_DIR / fname
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Wrote {out_path}")
        plt.close(fig)

    render(include_fa=True,  fname="bars_normalized_vs_pat.png",
           title_suffix=" (incl. FlashAttention)")
    render(include_fa=False, fname="bars_normalized_vs_pat_no_fa.png",
           title_suffix=" (FlashAttention omitted)")

    summary_csv = OUT_DIR / "normalized_vs_pat.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tree", "pat_ms"] + [f"{m}_ms" for m in methods if m != "pat"]
                   + [f"{m}_norm" for m in methods])
        for i, t in enumerate(trees):
            row = [t, pat_data[t]["pat"]]
            for m in methods:
                if m == "pat":
                    continue
                if m == "capillary":
                    row.append(mech2off[t])
                else:
                    row.append(pat_data[t].get(m, ""))
            for m in methods:
                v = norm[m][i]
                row.append("" if (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}")
            w.writerow(row)
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()

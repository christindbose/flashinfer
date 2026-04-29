"""
Per-GQA-config grouped-bar charts of attention-kernel speedup vs PAT, for the
20260428_200506 PAT run which sweeps two GQA configs:

  (nheads_q=16, nheads_kv=8)  — gqa=2  — only pat + cascade collected
  (nheads_q=32, nheads_kv=8)  — gqa=4  — pat + FastTree + fa + cascade

Capillary numbers come from bench_mech2_gqa_sweep.csv (qo=32, kv=32, gqa=1).
The capillary head config differs from both JSON GQA groups, so it appears on
each plot as a cross-config reference point.

Outputs (one per GQA config), in the same format as bars_normalized_vs_pat*:
  bars_normalized_vs_pat_qo<HQ>_kv<HKV>.png
  bars_normalized_vs_pat_qo<HQ>_kv<HKV>_no_fa.png
  normalized_vs_pat_qo<HQ>_kv<HKV>.csv
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAT_JSON = Path(
    "/home/chris241/myflashinfer_old/baselines/PAT_workspace/PAT/benchmark/"
    "final_results/kernel_perf_20260428_200506/kernel_perf_20260428_200506.json"
)
CAPILLARY_CSV = Path("/home/chris241/myflashinfer_old/flashinfer/bench_mech2_gqa_sweep.csv")
OUT_DIR = Path("/home/chris241/myflashinfer_old/flashinfer/results")


def load_pat_grouped(path):
    with open(path) as f:
        data = json.load(f)
    grouped = {}
    for entry in data:
        key = (entry["nheads_q"], entry["nheads_kv"])
        grouped.setdefault(key, {})[entry["tree"]] = entry["latencies"]
    return grouped


def load_capillary(path):
    """Key on (num_qo_heads, num_kv_heads, tree_spec) so each plot can pick
    the head-matched Capillary row. Missing combinations -> not in dict ->
    NaN downstream -> X marker on the plot."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["num_qo_heads"]), int(row["num_kv_heads"]), row["tree_spec"])
            out[key] = float(row["geo_mean_ms"])
    return out


METHODS = ["pat", "FastTree", "fa", "cascade", "capillary"]
LABELS = {
    "pat": "PAT (baseline)",
    "FastTree": "FastTree",
    "fa": "FlashAttention",
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


def render_for_config(hq, hkv, pat_for_cfg, capillary, suffix):
    trees = list(pat_for_cfg.keys())
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

    n_trees = len(trees)
    bar_w = 0.16
    x = np.arange(n_trees)

    def render(include_fa, fname, title_suffix):
        plt.rcParams.update({"font.size": 16})
        fig, ax = plt.subplots(figsize=(max(22, 1.2 * n_trees), 9))
        active = [m for m in METHODS if include_fa or m != "fa"]
        offsets = (np.arange(len(active)) - (len(active) - 1) / 2) * bar_w
        for offset, m in zip(offsets, active):
            vals = np.array(norm[m], dtype=float)
            mask = np.isnan(vals)
            plot_vals = np.where(mask, 0.0, vals)
            ax.bar(
                x + offset, plot_vals, bar_w,
                color=COLORS[m], label=LABELS[m],
                edgecolor="black", linewidth=0.4,
            )
            if np.any(mask):
                miss_x = (x + offset)[mask]
                ax.scatter(miss_x, np.full_like(miss_x, 0.15, dtype=float),
                           marker="x", s=300, color=COLORS[m],
                           linewidths=4.0, zorder=5)

        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(trees, rotation=55, ha="right", fontsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_xlabel("Tree structure <num nodes>_<KV context length>", fontsize=16)
        ax.set_ylabel("Speedup over PAT (higher = faster)", fontsize=16)
        gqa_ratio = hq // hkv
        fig.suptitle(
            f"num_qo_heads={hq}  |  num_kv_heads={hkv}  |  GQA ratio={gqa_ratio}",
            fontsize=20, fontweight="bold", y=0.995,
        )
        ax.set_title(f"head_dim=128{title_suffix}", fontsize=14)
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

    render(True,  f"bars_normalized_vs_pat_{suffix}.png",      "")
    render(False, f"bars_normalized_vs_pat_{suffix}_no_fa.png", " (FA omitted)")

    summary_csv = OUT_DIR / f"normalized_vs_pat_{suffix}.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["tree", "pat_ms"] + [f"{m}_ms" for m in METHODS if m != "pat"]
            + [f"{m}_norm" for m in METHODS]
        )
        for i, t in enumerate(trees):
            row = [t, pat_for_cfg[t]["pat"]]
            for m in METHODS:
                if m == "pat":
                    continue
                if m == "capillary":
                    row.append(capillary.get((hq, hkv, t), ""))
                else:
                    row.append(pat_for_cfg[t].get(m, ""))
            for m in METHODS:
                v = norm[m][i]
                row.append(""
                           if (isinstance(v, float) and np.isnan(v))
                           else f"{v:.4f}")
            w.writerow(row)
    print(f"Wrote {summary_csv}")


def main():
    grouped = load_pat_grouped(PAT_JSON)
    capillary = load_capillary(CAPILLARY_CSV)
    for (hq, hkv), pat_for_cfg in sorted(grouped.items()):
        suffix = f"qo{hq}_kv{hkv}"
        render_for_config(hq, hkv, pat_for_cfg, capillary, suffix)


if __name__ == "__main__":
    main()

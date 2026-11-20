"""
Bars normalized to PAT, extending plot_normalized_vs_pat_combined.py with a
new Capillary_8 series read from the current bench_mech2_gqa_sweep.csv.

Inputs:
  results/normalized_vs_pat_combined.csv    frozen numbers from the April run:
                                            PAT / FastTree / FA / Cascade / Capillary(cluster=4)
  bench_mech2_gqa_sweep.csv                 fresh sweep with cluster=8 (Capillary_8)

Output:
  results/bars_normalized_vs_pat_capillary8.pdf
  results/bars_normalized_vs_pat_capillary8.png
  results/bars_normalized_vs_pat_capillary8_no_fa.pdf
  results/bars_normalized_vs_pat_capillary8_no_fa.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FROZEN_CSV = Path("/home/chris241/myflashinfer_old/flashinfer/results/normalized_vs_pat_combined.csv")
CAP8_CSV   = Path("/home/chris241/myflashinfer_old/flashinfer/bench_mech2_gqa_sweep.csv")
OUT_DIR    = Path("/home/chris241/myflashinfer_old/flashinfer/results")

METHODS = ["pat", "FastTree", "fa", "cascade", "capillary", "capillary_8"]
LABELS = {
    "pat":         "PAT",
    "FastTree":    "FastTree",
    "fa":          "FlashAttention3",
    "cascade":     "Cascade",
    "capillary":   "Capillary (c=4)",
    "capillary_8": "Capillary (c=8)",
}
COLORS = {
    "pat":         "#444444",
    "FastTree":    "#1f77b4",
    "fa":          "#d62728",
    "cascade":     "#2ca02c",
    "capillary":   "#ff7f0e",
    "capillary_8": "#9467bd",
}

# Panel order matches the original plot: (qo, kv, gqa).
PANEL_ORDER = [(32, 32, 1), (32, 8, 4), (16, 8, 2)]


def _f(v):
    return float(v) if v not in ("", None) else None


def load_frozen(path):
    """Return dict: (qo, kv) -> {tree -> {pat_ms, FastTree_ms, fa_ms, cascade_ms, capillary_ms}}."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["panel_qo"]), int(row["panel_kv"]))
            entry = {
                "pat_ms":       _f(row["pat_ms"]),
                "FastTree_ms":  _f(row["FastTree_ms"]),
                "fa_ms":        _f(row["fa_ms"]),
                "cascade_ms":   _f(row["cascade_ms"]),
                "capillary_ms": _f(row["capillary_ms"]),
            }
            out.setdefault(key, {})[row["tree"]] = entry
    return out


def load_cap8(path):
    """Return dict: (qo, kv, tree_spec) -> geo_mean_ms (float)."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["num_qo_heads"]), int(row["num_kv_heads"]), row["tree_spec"])
            out[key] = float(row["geo_mean_ms"])
    return out


def main():
    frozen = load_frozen(FROZEN_CSV)
    cap8   = load_cap8(CAP8_CSV)

    # Union of trees across the three panels.
    all_trees = set()
    for (qo, kv, _) in PANEL_ORDER:
        all_trees.update(frozen.get((qo, kv), {}).keys())

    trees = sorted(
        all_trees,
        key=lambda t: (
            int(t.split("_")[0].split(",")[1]),   # leaf count
            int(t.split("_")[1].split(",")[0]),   # prefix ctx
            int(t.split("_")[1].split(",")[1]),   # leaf ctx
        ),
    )

    # Compute per-panel normalized values.
    panel_norms = []
    for (qo, kv, _) in PANEL_ORDER:
        rows = frozen.get((qo, kv), {})
        norm = {m: [] for m in METHODS}
        for t in trees:
            entry = rows.get(t)
            if entry is None:
                for m in METHODS:
                    norm[m].append(np.nan)
                continue
            pat_lat = entry["pat_ms"]
            norm["pat"].append(1.0)
            for m in ["FastTree", "fa", "cascade", "capillary"]:
                lat = entry[f"{m}_ms"]
                norm[m].append((pat_lat / lat) if lat else np.nan)
            cap8_lat = cap8.get((qo, kv, t))
            norm["capillary_8"].append((pat_lat / cap8_lat) if cap8_lat else np.nan)
        panel_norms.append(norm)

    n_trees = len(trees)
    bar_w = 0.13  # tighter than the 0.16 in the 5-method plot to fit a 6th bar
    x = np.arange(n_trees)

    def render(include_fa, fname):
        plt.rcParams.update({"font.size": 20})
        fig, axes = plt.subplots(
            nrows=3, ncols=1,
            figsize=(15, 10), dpi=200,
            sharex=True,
        )
        active = [m for m in METHODS if include_fa or m != "fa"]
        offsets = (np.arange(len(active)) - (len(active) - 1) / 2) * bar_w

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
        for ax_idx, ((qo, kv, gqa), norm) in enumerate(zip(PANEL_ORDER, panel_norms)):
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

        fig.supylabel("Speedup normalized to PAT", fontsize=20, x=0.085, y=0.58)

        def fmt_tree(spec):
            nodes_part, ctx_part = spec.split("_", 1)
            return f"B=[{nodes_part}] L=[{ctx_part}]"

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([fmt_tree(t) for t in trees],
                                 rotation=55, ha="right", fontsize=14)

        if legend_handles is not None:
            handles, labels = legend_handles
            fig.legend(
                handles, labels,
                loc="upper center", bbox_to_anchor=(0.57, 0.96),
                ncol=len(labels),
                fontsize=16, frameon=False, borderaxespad=0.0,
            )

        fig.tight_layout(rect=(0.02, 0, 1.0, 0.92))
        pdf_path = OUT_DIR / fname
        png_path = pdf_path.with_suffix(".png")
        fig.savefig(pdf_path)
        fig.savefig(png_path)
        print(f"Wrote {pdf_path}")
        print(f"Wrote {png_path}")
        plt.close(fig)

    render(True,  "bars_normalized_vs_pat_capillary8.pdf")
    render(False, "bars_normalized_vs_pat_capillary8_no_fa.pdf")

    # Extended CSV with capillary_8 columns.
    ext_csv = OUT_DIR / "normalized_vs_pat_capillary8.csv"
    with open(ext_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["panel_qo", "panel_kv", "panel_gqa", "tree", "pat_ms"]
            + [f"{m}_ms" for m in METHODS if m != "pat"]
            + [f"{m}_norm" for m in METHODS]
        )
        for (qo, kv, gqa), norm in zip(PANEL_ORDER, panel_norms):
            rows = frozen.get((qo, kv), {})
            for i, t in enumerate(trees):
                if t not in rows:
                    continue
                entry = rows[t]
                pat_ms = entry["pat_ms"]
                cap8_ms = cap8.get((qo, kv, t))
                row = [qo, kv, gqa, t, pat_ms,
                       entry["FastTree_ms"], entry["fa_ms"], entry["cascade_ms"],
                       entry["capillary_ms"] if entry["capillary_ms"] is not None else "",
                       cap8_ms if cap8_ms is not None else ""]
                for m in METHODS:
                    v = norm[m][i]
                    row.append("" if (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}")
                w.writerow(row)
    print(f"Wrote {ext_csv}")


if __name__ == "__main__":
    main()

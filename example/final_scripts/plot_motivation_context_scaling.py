"""
Motivation: PAT and FastTree scale poorly with context length; capillary's
advantage widens as the root KV grows.

Absolute latency (ms) vs root KV length, geomean over the trees sharing each
root, one panel per GQA config. PAT/FastTree/Cascade from the April
measurement set (results/normalized_vs_pat_combined.csv); capillary variants
from the current sweeps (per-variant ms backed out is unnecessary here: we
read ms directly from the source CSVs). Cross-set ms comparison is valid:
today's kvsplit c4 reproduces April capillary within ~1%.

Outputs:
  final_plots/motivation_context_scaling.pdf / .png
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "final_plots"
APRIL_CSV = Path(
    "/home/chris241/myflashinfer_old/flashinfer/results/normalized_vs_pat_combined.csv"
)

# label -> (source, column/csv, color, marker)
APRIL = {"pat_ms": "PAT", "FastTree_ms": "FastTree", "cascade_ms": "Cascade"}
OURS_CSVS = {
    "capillary (kvsplit c4)": SCRIPT_DIR / "bench_mech2_gqa_sweep_kvsplit_cluster4.csv",
    "capillary (c4 + pack_gqa)": SCRIPT_DIR / "bench_pack_kvsplit_cluster4.csv",
}
STYLE = {
    "PAT": ("#444444", "o", "-"),
    "FastTree": ("#2a78d6", "s", "-"),
    "Cascade": ("#1baf7a", "^", "-"),
    "capillary (kvsplit c4)": ("#eda100", "D", "--"),
    "capillary (c4 + pack_gqa)": ("#4a3aa7", "*", "--"),
}
PANELS = [(32, 8), (16, 8)]


def root_kv(tree):
    return int(tree.split("_")[1].split(",")[0])


def gm(xs):
    xs = [x for x in xs if x == x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def load_april():
    out = {}
    with open(APRIL_CSV) as f:
        for r in csv.DictReader(f):
            out[(int(r["panel_qo"]), int(r["panel_kv"]), r["tree"])] = r
    return out


def load_ours(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(int(r["num_qo_heads"]), int(r["num_kv_heads"]), r["tree_spec"])] = float(
                r["geo_mean_ms"]
            )
    return out


def main():
    april = load_april()
    ours = {label: load_ours(p) for label, p in OURS_CSVS.items()}

    fig, axes = plt.subplots(1, len(PANELS), figsize=(11, 4.2), sharey=True)

    for ax, (qo, kv) in zip(axes, PANELS):
        keys = [k for k in april if k[0] == qo and k[1] == kv]
        roots = sorted({root_kv(k[2]) for k in keys})

        def series_ms(getter):
            out = []
            for rk in roots:
                vals = []
                for k in keys:
                    if root_kv(k[2]) == rk:
                        v = getter(k)
                        if v is not None:
                            vals.append(v)
                out.append(gm(vals))
            return out

        for col, label in APRIL.items():
            def get(k, col=col):
                try:
                    return float(april[k][col])
                except (ValueError, KeyError):
                    return None
            ys = series_ms(get)
            c, m, ls = STYLE[label]
            ax.plot(roots, ys, ls, color=c, marker=m, markersize=6, linewidth=1.8, label=label)

        for label, tbl in ours.items():
            ys = series_ms(lambda k, tbl=tbl: tbl.get(k))
            c, m, ls = STYLE[label]
            ax.plot(roots, ys, ls, color=c, marker=m, markersize=8, linewidth=1.8, label=label)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(roots)
        ax.set_xticklabels([f"{r // 1024}K" for r in roots])
        ax.set_xlabel("root KV length (tokens)")
        ax.set_title(f"{qo} QO / {kv} KV heads (GQA={qo // kv})", fontsize=10)
        ax.grid(True, which="both", color="#dddddd", linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("attention latency (ms, geomean)")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"motivation_context_scaling.{ext}"
        fig.savefig(out, dpi=300)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""
Cluster-size crossover for capillary kvsplit (mech1) on H100.

For each KV length, plots the geomean speedup of kvsplit-on over kvsplit-off
(same tree suite, all head configs) for cluster sizes 2 / 4 / 8. Faint dots
show the per-config spread behind each geomean.

Inputs (produced by bench_mech2_GQA_only.py --kvsplit_mode at each cluster
size, plus the kvsplit-off sweep):
  bench_mech2_gqa_sweep_kvsplit_cluster{2,4,8}.csv
  bench_mech2_gqa_sweep.csv                        (kvsplit off)

Outputs:
  final_plots/kvsplit_cluster_crossover.pdf / .png
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "final_plots"
OUT_DIR.mkdir(exist_ok=True)

CLUSTERS = [2, 4, 8]
COLORS = {2: "#2a78d6", 4: "#1baf7a", 8: "#eda100"}
MARKERS = {2: "o", 4: "s", 8: "^"}


def load(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (int(r["num_qo_heads"]), int(r["num_kv_heads"]), r["tree_spec"])
            out[key] = float(r["geo_mean_ms"])
    return out


def kv_len_of(key):
    return int(key[2].split("_")[1].split(",")[0])


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    off = load(SCRIPT_DIR / "bench_mech2_gqa_sweep.csv")
    on = {c: load(SCRIPT_DIR / f"bench_mech2_gqa_sweep_kvsplit_cluster{c}.csv") for c in CLUSTERS}

    common = set(off)
    for c in CLUSTERS:
        common &= set(on[c])
    kv_lens = sorted({kv_len_of(k) for k in common})

    fig, ax = plt.subplots(figsize=(4.6, 3.2))

    for c in CLUSTERS:
        geo = []
        for kv in kv_lens:
            keys = [k for k in common if kv_len_of(k) == kv]
            speedups = [off[k] / on[c][k] for k in keys]
            geo.append(gmean(speedups))
            ax.scatter([kv] * len(speedups), speedups, s=6, color=COLORS[c],
                       alpha=0.25, linewidths=0, zorder=2)
        ax.plot(kv_lens, geo, color=COLORS[c], marker=MARKERS[c], markersize=5,
                linewidth=1.8, zorder=3)
        dy = {2: 6, 4: 6, 8: -11}[c]
        ax.annotate(f"cluster {c}", (kv_lens[-1], geo[-1]),
                    xytext=(8, dy), textcoords="offset points",
                    fontsize=8.5, color="#333333", va="center")

    ax.axhline(1.0, color="#888888", linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("no split", (kv_lens[0], 1.0), xytext=(0, -11),
                textcoords="offset points", fontsize=8, color="#666666")

    # crossover: cluster 4 overtakes cluster 2 between 32K and 64K
    ax.axvspan(32768, 65536, color="#000000", alpha=0.045, zorder=0)
    ax.annotate("4 overtakes 2", (math.sqrt(32768 * 65536), 0.62),
                fontsize=8, color="#666666", ha="center")

    ax.set_xscale("log", base=2)
    ax.set_xticks(kv_lens)
    ax.set_xticklabels([f"{kv // 1024}K" for kv in kv_lens])
    ax.minorticks_off()
    ax.set_xlabel("root KV length (tokens)")
    ax.set_ylabel("speedup over no split (geomean)")
    ax.set_ylim(bottom=0.5)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(handles=[
        plt.Line2D([], [], color=COLORS[c], marker=MARKERS[c], markersize=5,
                   linewidth=1.8, label=f"cluster {c}") for c in CLUSTERS
    ], fontsize=8, frameon=False, loc="upper left")
    ax.margins(x=0.14)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"kvsplit_cluster_crossover.{ext}"
        fig.savefig(out, dpi=300)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

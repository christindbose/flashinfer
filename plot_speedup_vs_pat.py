"""Plot speedup vs PAT across trees. Combines PAT/FastTree/cascade latencies
from a kernel_perf JSON with Capillary latencies from a CSV.

speedup = pat_ms / baseline_ms  (higher = faster than PAT; PAT itself = 1.0)

Usage:
    python plot_speedup_vs_pat.py <kernel_perf.json> <capillary.csv> \
        [--out results/speedup_vs_pat.png]
"""
import argparse
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path):
    """Return ({tree: {baseline: ms}}, {tree: set_of_FAILED_baselines})."""
    with open(path) as f:
        data = json.load(f)
    lats = {}
    failed = {}
    for e in data:
        lats.setdefault(e["tree"], {}).update(e["latencies"])
        f_set = failed.setdefault(e["tree"], set())
        for k, v in (e.get("correctness") or {}).items():
            if v.get("status") != "PASSED":
                f_set.add(k)
    return lats, failed


def load_capillary_csv(path):
    """Return {tree_spec: geo_mean_ms}."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["tree_spec"]] = float(r["geo_mean_ms"])
    return out


TREE_RE = re.compile(r"^1,(\d+)_(\d+),(\d+)$")


def tree_key(t):
    m = TREE_RE.match(t)
    if not m:
        return (float("inf"), float("inf"), float("inf"), t)
    leaves, ctx, leaf_ctx = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return (leaf_ctx, ctx, leaves, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("csv")
    ap.add_argument("--out", default="/home/chris241/myflashinfer_old/flashinfer/results/speedup_vs_pat.png")
    ap.add_argument("--width", type=int, default=1800)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--font-size", type=int, default=14)
    args = ap.parse_args()

    latencies_json, failed_by_tree = load_json(args.json)
    capillary = load_capillary_csv(args.csv)

    trees = sorted(set(latencies_json) & set(capillary), key=tree_key)
    missing_j = set(capillary) - set(latencies_json)
    missing_c = set(latencies_json) - set(capillary)
    if missing_j:
        print(f"[warn] in CSV but not JSON: {sorted(missing_j)}")
    if missing_c:
        print(f"[warn] in JSON but not CSV: {sorted(missing_c)}")

    baseline_order = ["pat", "Capillary", "FastTree", "cascade"]

    per_baseline = {b: [] for b in baseline_order}
    dropped_ft = []
    for t in trees:
        pat = latencies_json[t].get("pat")
        fails = failed_by_tree.get(t, set())
        if pat is None:
            for b in baseline_order:
                per_baseline[b].append(float("nan"))
            continue
        per_baseline["pat"].append(1.0)
        cap_ms = capillary.get(t)
        per_baseline["Capillary"].append(pat / cap_ms if cap_ms else float("nan"))
        ft = latencies_json[t].get("FastTree")
        if "FastTree" in fails:
            per_baseline["FastTree"].append(float("nan"))
            dropped_ft.append(t)
        else:
            per_baseline["FastTree"].append(pat / ft if ft else float("nan"))
        cs = latencies_json[t].get("cascade")
        per_baseline["cascade"].append(pat / cs if cs else float("nan"))
    if dropped_ft:
        print(f"[info] FastTree FAILED correctness — dropped from plot for: {dropped_ft}")

    plt.rcParams.update({"font.size": args.font_size})
    xs = list(range(len(trees)))
    n = len(baseline_order)
    group_width = 0.8
    bar_w = group_width / n

    fig, ax = plt.subplots(
        figsize=(args.width / args.dpi, args.height / args.dpi),
        dpi=args.dpi,
    )
    display_name = {"pat": "PAT", "cascade": "Cascade"}
    for i, b in enumerate(baseline_order):
        offsets = [x - group_width / 2 + bar_w * (i + 0.5) for x in xs]
        ax.bar(offsets, per_baseline[b], width=bar_w,
               label=display_name.get(b, b))

    # Mark missing FastTree with 'x' at the bar position.
    ft_idx = baseline_order.index("FastTree")
    miss_xs = []
    for i, v in enumerate(per_baseline["FastTree"]):
        if v != v:  # NaN check
            miss_xs.append(xs[i] - group_width / 2 + bar_w * (ft_idx + 0.5))
    if miss_xs:
        ax.scatter(miss_xs, [0.08] * len(miss_xs), marker="x", color="red",
                   s=140, linewidths=2.5, zorder=5,
                   label="x = missing data (FastTree failed correctness)")

    ax.axhline(1.0, linestyle="--", color="gray", alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(trees, rotation=35, ha="right")
    ax.set_xlabel("Tree structure")
    ax.set_ylabel("Speedup relative to PAT")
    ax.grid(True, axis="y", alpha=0.3)
    # Legend outside the axes (top-horizontal) so it doesn't occlude bars.
    legend_entries = len(baseline_order) + (1 if miss_xs else 0)
    ax.legend(
        fontsize=max(args.font_size, 14),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(legend_entries, 5),
        frameon=False,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.out}")

    print()
    print(f"{'tree':<22}  " + "  ".join(f"{b:>10}" for b in baseline_order))
    for i, t in enumerate(trees):
        row = "  ".join(f"{per_baseline[b][i]:>10.3f}" for b in baseline_order)
        print(f"{t:<22}  {row}")


if __name__ == "__main__":
    main()

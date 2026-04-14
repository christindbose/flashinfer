#!/usr/bin/env python3
"""
Plot Cascade (FlashInfer MultiLevelCascadeAttention), PAT, and capillary runtimes.

Joins on tree id: PAT `tree_pat` / capillary `tree_spec` ==
  `{node_num_per_level}_{node_seqlen_per_level}` from cascade_summary.csv.

Panels: raw latency (ms), and normalized to cascade (= 1.0 per configuration).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_pat_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_capillary_csv(path: Path) -> dict[str, float]:
    """tree_spec -> geo_mean_ms"""
    out: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row["tree_spec"].strip()
            out[key] = float(row["geo_mean_ms"])
    return out


def read_cascade_csv(path: Path) -> dict[str, float]:
    """tree key -> latency_ms_geo_mean"""
    out: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nodes = row["node_num_per_level"].strip().strip('"')
            seqlen = row["node_seqlen_per_level"].strip().strip('"')
            key = f"{nodes}_{seqlen}"
            out[key] = float(row["latency_ms_geo_mean"])
    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    default_pat = (
        root.parent.parent
        / "baselines"
        / "PAT_workspace"
        / "PAT"
        / "benchmark"
        / "pat_mech2_trees_20260413_114503.csv"
    )
    default_cap = root / "capillary.csv"
    default_cascade = (
        root.parent.parent
        / "baselines"
        / "example"
        / "outputs"
        / "cascade_summary.csv"
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pat",
        type=Path,
        default=default_pat,
        help="PAT benchmark CSV (tree_pat, pat_latency_ms_median)",
    )
    p.add_argument(
        "--capillary",
        type=Path,
        default=default_cap,
        help="Capillary benchmark CSV (tree_spec, geo_mean_ms)",
    )
    p.add_argument(
        "--cascade",
        type=Path,
        default=default_cascade,
        help="Cascade benchmark CSV (node_*_per_level, latency_ms_geo_mean)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save figure to this path. If omitted, show GUI.",
    )
    args = p.parse_args()

    pat_rows = read_pat_csv(args.pat)
    cap_by_spec = read_capillary_csv(args.capillary)
    cascade_by_key = read_cascade_csv(args.cascade)

    labels: list[str] = []
    cascade_ms: list[float] = []
    pat_ms: list[float] = []
    cap_ms: list[float] = []

    for row in pat_rows:
        key = row["tree_pat"].strip()
        if key not in cap_by_spec:
            raise KeyError(
                f"tree_pat not in capillary CSV: {key!r} "
                f"(check --pat / --capillary paths)"
            )
        if key not in cascade_by_key:
            raise KeyError(
                f"tree_pat not in cascade CSV: {key!r} "
                f"(check --cascade path; expect nodes_seqlen key)"
            )
        labels.append(key)
        cascade_ms.append(cascade_by_key[key])
        pat_ms.append(float(row["pat_latency_ms_median"]))
        cap_ms.append(cap_by_spec[key])

    n = len(labels)
    if n == 0:
        raise SystemExit("No rows to plot.")

    cas_arr = np.array(cascade_ms, dtype=np.float64)
    pat_arr = np.array(pat_ms, dtype=np.float64)
    cap_arr = np.array(cap_ms, dtype=np.float64)
    norm_cas = np.ones_like(cas_arr)
    norm_pat = pat_arr / cas_arr
    norm_cap = cap_arr / cas_arr

    x = np.arange(n)
    width = 0.25

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(max(10, n * 0.85), 9), sharex=True)
    fig.suptitle("Cascade vs PAT vs capillary (same tree per bar group)")

    ax0.bar(x - width, cascade_ms, width, label="Cascade")
    ax0.bar(x, pat_ms, width, label="PAT")
    ax0.bar(x + width, cap_ms, width, label="Capillary")
    ax0.set_ylabel("Latency (ms)")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.set_title("Raw")
    ax0.grid(axis="y", alpha=0.3)

    ax1.bar(x - width, norm_cas, width, label="Cascade (= 1)")
    ax1.bar(x, norm_pat, width, label="PAT / Cascade")
    ax1.bar(x + width, norm_cap, width, label="Capillary / Cascade")
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Relative latency (Cascade = 1)")
    ax1.set_title("Normalized to cascade per configuration")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Wrote {args.output.resolve()}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

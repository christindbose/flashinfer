#!/usr/bin/env python3
"""
Plot Cascade (MultiLevelCascade / bench_cascade), PAT, and FlashInfer mech2 (GQA sweep)
latencies from CSVs.

Join key: tree_key = {node_num_per_level}_{node_seqlen_per_level} plus num_qo_heads,
num_kv_heads.

Outputs:
  - *_raw.png — raw latency (ms)
  - *_norm.png — relative latency (method / cascade), cascade = 1
  - *_speedup.png — cascade_ms / method_ms (>1 means cascade is faster)

Deduplication:
  - cascade: first row per join key
  - mech2: drop non-finite geo_mean_ms; keep latest row per key by timestamp
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def strip_field(s: str) -> str:
    s = str(s).strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return s


def load_cascade(path: Path) -> dict[tuple[str, int, int], float]:
    """(tree_key, num_qo, num_kv) -> cascade_ms (first wins)"""
    out: dict[tuple[str, int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nodes = strip_field(row["node_num_per_level"])
            seqlen = strip_field(row["node_seqlen_per_level"])
            tree_key = f"{nodes}_{seqlen}"
            nq = int(row["num_qo_heads"])
            nk = int(row["num_kv_heads"])
            ms = float(row["latency_ms_geo_mean"])
            k = (tree_key, nq, nk)
            if k not in out:
                out[k] = ms
    return out


def load_pat(path: Path) -> dict[tuple[str, int, int], float]:
    out: dict[tuple[str, int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tree_key = strip_field(row["tree_pat"])
            nq = int(row["nheads_q"])
            nk = int(row["nheads_kv"])
            out[(tree_key, nq, nk)] = float(row["pat_latency_ms_median"])
    return out


def load_mech2(path: Path) -> tuple[dict[tuple[str, int, int], float], int]:
    """Latest timestamp per key. Returns (dict, num_rows_skipped_nan)."""
    rows: list[tuple[datetime, tuple[str, int, int], float]] = []
    skipped_nan = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tree_key = strip_field(row["tree_spec"])
            nq = int(row["num_qo_heads"])
            nk = int(row["num_kv_heads"])
            g = float(row["geo_mean_ms"])
            if not np.isfinite(g):
                skipped_nan += 1
                continue
            ts_s = row["timestamp"].strip()
            try:
                ts = datetime.fromisoformat(ts_s.replace("Z", "+00:00"))
            except ValueError:
                continue
            rows.append((ts, (tree_key, nq, nk), g))
    rows.sort(key=lambda x: x[0])
    out: dict[tuple[str, int, int], float] = {}
    for _ts, k, g in rows:
        out[k] = g
    return out, skipped_nan


def merge_keys(
    cascade: dict[tuple[str, int, int], float],
    pat: dict[tuple[str, int, int], float],
    mech2: dict[tuple[str, int, int], float],
) -> tuple[list[tuple[str, int, int]], list[str], list[tuple]]:
    c_set = set(cascade.keys())
    p_set = set(pat.keys())
    m_set = set(mech2.keys())
    notes: list[str] = []

    def report(name: str, missing: set[tuple]) -> None:
        if not missing:
            return
        sample = sorted(missing)[:12]
        more = f" … (+{len(missing) - len(sample)} more)" if len(missing) > 12 else ""
        notes.append(f"{name} ({len(missing)}): {sample}{more}")

    report("Cascade keys not in PAT", c_set - p_set)
    report("Cascade keys not in mech2", c_set - m_set)
    report("PAT keys not in cascade", p_set - c_set)
    report("mech2 keys not in cascade", m_set - c_set)

    inner = sorted(k for k in c_set if k in p_set and k in m_set)
    return inner, notes


def pat_tree_order(pat_path: Path) -> list[str]:
    """First-seen tree_pat order."""
    seen: list[str] = []
    got: set[str] = set()
    with pat_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = strip_field(row["tree_pat"])
            if t not in got:
                got.add(t)
                seen.append(t)
    return seen


def main() -> None:
    root = Path(__file__).resolve().parent
    default_cascade = (
        root.parent.parent / "baselines" / "example" / "outputs" / "cascade_summary_qk.csv"
    )
    default_pat = (
        root.parent.parent
        / "baselines"
        / "PAT_workspace"
        / "PAT"
        / "benchmark"
        / "pat_mech2_trees_20260414_164811.csv"
    )
    default_mech2 = root.parent / "bench_mech2_gqa_sweep.csv"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cascade", type=Path, default=default_cascade)
    p.add_argument("--pat", type=Path, default=default_pat)
    p.add_argument("--mech2", type=Path, default=default_mech2)
    p.add_argument(
        "-o",
        "--output-prefix",
        type=Path,
        default=root / "cascade_pat_mech2_compare",
        help="Write {prefix}_raw.png, _norm.png, _speedup.png",
    )
    args = p.parse_args()

    cascade = load_cascade(args.cascade)
    pat = load_pat(args.pat)
    mech2, mech2_skipped_nan = load_mech2(args.mech2)
    keys_merged, notes = merge_keys(cascade, pat, mech2)

    print("=== Merge summary ===")
    print(f"Inner-joined configurations: {len(keys_merged)}")
    if mech2_skipped_nan:
        print(
            f"[missing] mech2 rows skipped (non-finite geo_mean_ms): {mech2_skipped_nan}"
        )
    for line in notes:
        print(f"[missing] {line}")
    if not notes:
        print("(No key mismatches between cascade / PAT / mech2 sets.)")

    if not keys_merged:
        raise SystemExit("No rows after inner join — check CSV paths.")

    tree_order = pat_tree_order(args.pat)
    tree_to_x = {t: i for i, t in enumerate(tree_order)}
    head_groups = [(1, 1), (32, 8), (32, 32)]

    def plot_block(ax_raw, ax_rel, ax_sup, nq: int, nk: int, title_suffix: str) -> None:
        sub_keys = [k for k in keys_merged if k[1] == nq and k[2] == nk]
        sub_keys.sort(key=lambda k: (tree_to_x.get(k[0], 999), k[0]))
        if not sub_keys:
            for ax in (ax_raw, ax_rel, ax_sup):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            return

        x = np.array([tree_to_x[k[0]] for k in sub_keys], dtype=np.float64)
        cas = np.array([cascade[k] for k in sub_keys])
        pm = np.array([pat[k] for k in sub_keys])
        mm = np.array([mech2[k] for k in sub_keys])
        w = 0.22
        ax_raw.bar(x - w, cas, w, label="Cascade", color="#1f77b4")
        ax_raw.bar(x, pm, w, label="PAT", color="#ff7f0e")
        ax_raw.bar(x + w, mm, w, label="mech2", color="#2ca02c")
        ax_raw.set_ylabel("ms")
        ax_raw.set_title(f"Raw latency — {title_suffix}")
        ax_raw.legend(fontsize=7, loc="upper left")
        ax_raw.grid(axis="y", alpha=0.3)

        rel_c = np.ones_like(cas)
        rel_p = pm / cas
        rel_m = mm / cas
        ax_rel.bar(x - w, rel_c, w, label="Cascade (=1)", color="#1f77b4")
        ax_rel.bar(x, rel_p, w, label="PAT / Cascade", color="#ff7f0e")
        ax_rel.bar(x + w, rel_m, w, label="mech2 / Cascade", color="#2ca02c")
        ax_rel.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax_rel.set_ylabel("Relative latency (Cascade = 1)")
        ax_rel.set_title(f"Normalized to cascade — {title_suffix}")
        ax_rel.legend(fontsize=7, loc="upper left")
        ax_rel.grid(axis="y", alpha=0.3)

        sp_p = cas / pm
        sp_m = cas / mm
        ax_sup.bar(x - w, np.ones_like(cas), w, label="Cascade (ref)", color="#1f77b4")
        ax_sup.bar(x, sp_p, w, label="Cascade / PAT", color="#ff7f0e")
        ax_sup.bar(x + w, sp_m, w, label="Cascade / mech2", color="#2ca02c")
        ax_sup.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax_sup.set_ylabel("Cascade / method (>1 ⇒ cascade faster)")
        ax_sup.set_title(f"Speedup vs cascade — {title_suffix}")
        ax_sup.legend(fontsize=7, loc="upper left")
        ax_sup.grid(axis="y", alpha=0.3)

        xt = np.arange(len(tree_order))
        labs = tree_order
        for ax in (ax_raw, ax_rel, ax_sup):
            ax.set_xticks(xt)
            ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=6)

    n_trees = len(tree_order)
    fig1, axes1 = plt.subplots(3, 1, figsize=(max(12, n_trees * 1.0), 12), sharex=True)
    fig1.suptitle("Cascade vs PAT vs mech2 — raw latency (ms)")
    fig2, axes2 = plt.subplots(3, 1, figsize=(max(12, n_trees * 1.0), 12), sharex=True)
    fig2.suptitle("Cascade vs PAT vs mech2 — normalized to cascade")
    fig3, axes3 = plt.subplots(3, 1, figsize=(max(12, n_trees * 1.0), 12), sharex=True)
    fig3.suptitle("Cascade vs PAT vs mech2 — cascade speedup factor")

    for i, (nq, nk) in enumerate(head_groups):
        suf = f"Qo={nq}  Kv={nk}"
        plot_block(axes1[i], axes2[i], axes3[i], nq, nk, suf)

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    out_raw = Path(str(args.output_prefix) + "_raw.png")
    out_norm = Path(str(args.output_prefix) + "_norm.png")
    out_sup = Path(str(args.output_prefix) + "_speedup.png")
    fig1.savefig(out_raw, dpi=150, bbox_inches="tight")
    fig2.savefig(out_norm, dpi=150, bbox_inches="tight")
    fig3.savefig(out_sup, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_raw.resolve()}")
    print(f"Wrote {out_norm.resolve()}")
    print(f"Wrote {out_sup.resolve()}")
    plt.close("all")


if __name__ == "__main__":
    main()

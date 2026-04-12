"""
Benchmark Mech2 Cascade Attention (tree-walk scheduling)

Configuration: use_tree_walk_scheduling=True, mech2_mode=True, merge_every_n_levels=2
(default merge step overridable via --merge_every_n_levels).

Tree specs use the same encoding as PAT benchmarks: <nodes_csv>_<contexts_csv>
(single underscore between the last node count and the first context length).

With no --nodes/--contexts and no --tree, runs the default suite (same trees as
baselines/PAT_workspace/PAT/benchmark/run_pat_bench_mech2_trees.sh).
"""

import argparse
import csv
import math
import sys
from datetime import datetime

import torch
import flashinfer


# Same lines as run_pat_bench_mech2_trees.sh TREES (underscore between nodes and contexts)
DEFAULT_MECH2_PAT_TREES = [
    "1,3,9,27,81_256,32,16,16,16",
    "1,3,9,27,81_512,32,16,16,16",
    "1,3,9,27,81_1024,32,16,16,16",
    "1,3,9,27,81_256,64,16,16,16",
    "1,3,9,27,81_256,64,32,16,16",
    "1,2,4,8,16,32_32,16,16,16,16,16",
    "1,2,4,8,16,32_64,16,16,16,16,16",
    "1,2,4,8,16,32,64,128_64,32,32,16,32,16,16,16",
    "1,2,4,8,16,32,64,128_64,32,16,16,16,16,16,16",
    "1,2,4,8,16,32,64,128_64,32,32,16,16,16,16,16",
]


def _geom_mean_ms(times_ms):
    n = len(times_ms)
    if n == 0:
        raise ValueError("times_ms must be non-empty")
    return math.exp(sum(math.log(t) for t in times_ms) / n)


def parse_pat_tree(spec: str):
    """Parse 'n1,n2,..._c1,c2,...' into (nodes, contexts) lists of int."""
    if "_" not in spec:
        raise ValueError(
            f"Expected PAT tree spec with underscore, e.g. '1,3,9_32,16,8', got {spec!r}"
        )
    nodes_str, contexts_str = spec.split("_", 1)
    nodes = [int(x) for x in nodes_str.split(",")]
    contexts = [int(x) for x in contexts_str.split(",")]
    return nodes, contexts


def benchmark_mech2_one_tree(
    nodes,
    contexts,
    *,
    merge_every,
    page_size,
    num_kv_heads,
    head_dim,
    gqa,
    warmup_iters,
    bench_iters,
    verbose,
):
    num_levels = len(nodes)
    batch_size = nodes[-1]
    assert len(nodes) == len(contexts), "nodes and contexts must have same length"
    assert all(batch_size % n == 0 for n in nodes), "Each level must evenly divide batch_size"

    num_qo_heads = num_kv_heads * gqa
    pages_per_node = [(ctx + page_size - 1) // page_size for ctx in contexts]
    total_kv_groups = sum(nodes)
    total_pages = sum(nodes[level] * pages_per_node[level] for level in range(num_levels))
    total_queries = batch_size * num_levels

    if verbose:
        print(f"Nodes per level:    {nodes}")
        print(f"Contexts per level: {contexts}")
        print(f"Batch size (leaves): {batch_size}")
        print(f"Num levels:         {num_levels}")
        print(f"Total KV groups:    {total_kv_groups}")
        print(f"Total queries:      {total_queries}")
        print(f"Total pages:        {total_pages}")
        print(f"Page size:          {page_size}")
        print(f"Num QO heads:       {num_qo_heads}")
        print(f"Num KV heads:       {num_kv_heads}")
        print(f"Head dim:           {head_dim}")
        print(f"merge_every_n_levels: {merge_every}")
        print("\nTree visualization [Mech2]:")
        for level in range(num_levels):
            seqs_per_node = batch_size // nodes[level]
            indent = "  " * level
            will_merge = (level > 0 and (level - 1) % merge_every == 0) or level == 0
            merge_indicator = " [BASE]" if level == 0 else (" [MERGE]" if will_merge else " [SKIP]")
            print(
                f"{indent}Level {level}: {nodes[level]} node(s), {contexts[level]} tokens, "
                f"{seqs_per_node} seqs/node{merge_indicator}"
            )

    kv_page_indices_list = []
    kv_page_indptr_list = [0]
    kv_last_page_len_list = []
    page_offset = 0

    for level in range(num_levels):
        last_page_len = contexts[level] % page_size
        if last_page_len == 0:
            last_page_len = page_size

        for _ in range(nodes[level]):
            node_pages = list(range(page_offset, page_offset + pages_per_node[level]))
            kv_page_indices_list.extend(node_pages)
            kv_page_indptr_list.append(kv_page_indptr_list[-1] + pages_per_node[level])
            kv_last_page_len_list.append(last_page_len)
            page_offset += pages_per_node[level]

    kv_page_indices = torch.tensor(kv_page_indices_list, dtype=torch.int32, device="cuda:0")
    kv_page_indptr = torch.tensor(kv_page_indptr_list, dtype=torch.int32, device="cuda:0")
    kv_last_page_len = torch.tensor(kv_last_page_len_list, dtype=torch.int32, device="cuda:0")

    qo_indptr_list = [0]
    for level in range(num_levels):
        seqs_per_node = batch_size // nodes[level]
        for _ in range(nodes[level]):
            qo_indptr_list.append(qo_indptr_list[-1] + seqs_per_node)

    qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device="cuda:0")

    if verbose:
        print(f"\nqo_indptr: {qo_indptr}")
        print(f"kv_page_indptr: {kv_page_indptr}")
        print(f"kv_page_indices: {kv_page_indices}")
        print(f"kv_last_page_len: {kv_last_page_len}")

    kv_cache = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=torch.float16, device="cuda:0",
    )
    q = torch.randn(total_queries, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")

    if verbose:
        print("\n" + "=" * 70)
        print("RUNNING: Mech2")
        print("  use_tree_walk_scheduling = True")
        print("  mech2_mode               = True")
        print(f"  merge_every_n_levels     = {merge_every}")
        print("=" * 70)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(1, workspace_buffer, "NHD")

    wrapper.plan(
        [qo_indptr],
        [kv_page_indptr],
        [kv_page_indices],
        [kv_last_page_len],
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        use_tree_walk_scheduling=True,
        kvsplit_mode=False,
        mech2_mode=True,
    )

    for _ in range(warmup_iters):
        out = wrapper.run(q, kv_cache, tree_nodes=nodes, merge_every_n_levels=merge_every)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]

    for i in range(bench_iters):
        start_events[i].record()
        out = wrapper.run(q, kv_cache, tree_nodes=nodes, merge_every_n_levels=merge_every)
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    geo_mean_ms = _geom_mean_ms(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    if verbose:
        print(f"\n  Output shape: {out.shape}")
        print(f"  Expected:     [{batch_size}, {num_qo_heads}, {head_dim}]")
        print(f"  Output mean:  {out.mean().item():.6f}")
        print(f"  Output std:   {out.std().item():.6f}")
        print(f"\n  Latency (ms): geo_mean={geo_mean_ms:.4f}  min={min_ms:.4f}  max={max_ms:.4f}")
        print(f"\n  Sample outputs (first 8 sequences):")
        for seq_id in range(min(batch_size, 8)):
            out_val = out[seq_id, 0, 0].item()
            print(f"    Seq {seq_id}: out[{seq_id}][0,0] = {out_val:+.6f}")
        if batch_size > 8:
            print(f"    ... ({batch_size - 8} more sequences)")

    return {
        "geo_mean_ms": geo_mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "total_pages": total_pages,
        "total_queries": total_queries,
        "batch_size": batch_size,
        "num_levels": num_levels,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mech2 cascade attention")
    parser.add_argument(
        "--tree",
        action="append",
        dest="trees",
        default=None,
        metavar="PAT_SPEC",
        help="PAT-style tree n1,n2,..._c1,c2,... (repeat for multiple). "
        "If omitted and --nodes/--contexts omitted, runs the default mech2 tree suite.",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default=None,
        help='Comma-separated nodes per level (single run; use with --contexts)',
    )
    parser.add_argument(
        "--contexts",
        type=str,
        default=None,
        help='Comma-separated context lengths per level (single run; use with --nodes)',
    )
    parser.add_argument("--page_size", type=int, default=16, help="KV page size (default: 16)")
    parser.add_argument("--num_kv_heads", type=int, default=1, help="KV heads (default: 1)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dim (default: 128)")
    parser.add_argument("--gqa", type=int, default=1, help="GQA ratio (default: 1)")
    parser.add_argument("--warmup_iters", type=int, default=5, help="Warmup iterations (default: 5)")
    parser.add_argument("--bench_iters", type=int, default=100, help="Benchmark iterations (default: 100)")
    parser.add_argument("--merge_every_n_levels", type=int, default=2, help="default: 2")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Per-tree detailed prints (qo_indptr, samples, …)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Append one row per tree (creates file with header if missing)",
    )
    args = parser.parse_args()

    if args.nodes is not None and args.contexts is not None:
        if args.trees:
            print("error: use either --tree / default suite or --nodes with --contexts, not both", file=sys.stderr)
            sys.exit(2)
        run_list = [
            ([int(x) for x in args.nodes.split(",")], [int(x) for x in args.contexts.split(",")])
        ]
        labels = [f"{args.nodes}_{args.contexts}"]
    elif args.nodes is not None or args.contexts is not None:
        print("error: supply both --nodes and --contexts for a single explicit tree", file=sys.stderr)
        sys.exit(2)
    elif args.trees:
        run_list = []
        labels = []
        for spec in args.trees:
            n, c = parse_pat_tree(spec)
            run_list.append((n, c))
            labels.append(spec)
    else:
        run_list = [parse_pat_tree(s) for s in DEFAULT_MECH2_PAT_TREES]
        labels = list(DEFAULT_MECH2_PAT_TREES)

    merge_every = args.merge_every_n_levels
    results = []

    print("=" * 70)
    print("BENCHMARK: MECH2 CASCADE ATTENTION")
    print(f"Trees to run: {len(run_list)}")
    print("=" * 70)

    for idx, ((nodes, contexts), label) in enumerate(zip(run_list, labels), start=1):
        if not args.verbose and len(run_list) > 1:
            print(f"\n[{idx}/{len(run_list)}] {label}")
        elif args.verbose and len(run_list) > 1:
            print("\n" + "=" * 70)
            print(f"[{idx}/{len(run_list)}] {label}")
            print("=" * 70)
        elif len(run_list) == 1 and not args.verbose:
            print(f"\nSingle tree: {label}")

        stats = benchmark_mech2_one_tree(
            nodes,
            contexts,
            merge_every=merge_every,
            page_size=args.page_size,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            gqa=args.gqa,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
            verbose=args.verbose or len(run_list) == 1,
        )
        stats["tree_spec"] = label
        results.append(stats)

        if not args.verbose and len(run_list) > 1:
            print(
                f"  geo_mean={stats['geo_mean_ms']:.4f} ms  "
                f"min={stats['min_ms']:.4f}  max={stats['max_ms']:.4f}  "
                f"pages={stats['total_pages']}  queries={stats['total_queries']}"
            )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'tree_spec':<52} {'geo_ms':>10} {'min_ms':>10} {'max_ms':>10}")
    print("-" * 86)
    for r in results:
        spec = r["tree_spec"]
        if len(spec) > 50:
            spec = spec[:47] + "..."
        print(
            f"{spec:<52} {r['geo_mean_ms']:>10.4f} {r['min_ms']:>10.4f} {r['max_ms']:>10.4f}"
        )
    print("=" * 70)

    if args.output_csv:
        file_exists = False
        try:
            with open(args.output_csv, "r", encoding="utf-8"):
                file_exists = True
        except FileNotFoundError:
            pass
        fieldnames = [
            "timestamp",
            "tree_spec",
            "geo_mean_ms",
            "min_ms",
            "max_ms",
            "merge_every_n_levels",
            "page_size",
            "num_kv_heads",
            "head_dim",
            "gqa",
            "warmup_iters",
            "bench_iters",
            "total_pages",
            "total_queries",
            "batch_size",
            "num_levels",
        ]
        row_base = {
            "merge_every_n_levels": merge_every,
            "page_size": args.page_size,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "gqa": args.gqa,
            "warmup_iters": args.warmup_iters,
            "bench_iters": args.bench_iters,
        }
        ts = datetime.utcnow().isoformat() + "Z"
        with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            for r in results:
                w.writerow({**row_base, **r, "timestamp": ts})
        print(f"Wrote CSV: {args.output_csv}")

    print("DONE")


if __name__ == "__main__":
    main()

"""
Benchmark PAT prefix-tree scheduling (debug: prints fused tree only).

Same tree/KV setup as bench_mech2_only.py but with use_pat_scheduling=True.
Currently only prints the PAT fused tree structure via FLASHINFER_DEBUG_SCHEDULER;
the actual CTA assignment still uses the existing mech2 path.

Default tree: 1,2,4,8,16,32_32,16,16,16,16,16
"""

import argparse
import math
import sys

import torch
import flashinfer


def _geom_mean_ms(times_ms):
    n = len(times_ms)
    if n == 0:
        raise ValueError("times_ms must be non-empty")
    return math.exp(sum(math.log(t) for t in times_ms) / n)


def parse_pat_tree(spec: str):
    if "_" not in spec:
        raise ValueError(f"Expected PAT tree spec with underscore, got {spec!r}")
    nodes_str, contexts_str = spec.split("_", 1)
    nodes = [int(x) for x in nodes_str.split(",")]
    contexts = [int(x) for x in contexts_str.split(",")]
    return nodes, contexts


def main():
    parser = argparse.ArgumentParser(description="Benchmark PAT prefix-tree scheduling")
    parser.add_argument(
        "--tree",
        type=str,
        default="1,3,9,27,81,243_256,32,16,16,16,16",
        help="PAT-style tree spec n1,n2,..._c1,c2,... (default: 1,3,9,27,81,243_256,32,16,16,16,16)",
    )
    parser.add_argument("--page_size", type=int, default=16, help="KV page size (default: 16)")
    parser.add_argument("--num_kv_heads", type=int, default=1, help="KV heads (default: 1)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dim (default: 128)")
    parser.add_argument("--gqa", type=int, default=1, help="GQA ratio (default: 1)")
    parser.add_argument("--warmup_iters", type=int, default=5, help="Warmup iterations (default: 0)")
    parser.add_argument("--bench_iters", type=int, default=100, help="Bench iterations (default: 1)")
    parser.add_argument("--merge_every_n_levels", type=int, default=2, help="default: 2")
    args = parser.parse_args()

    nodes, contexts = parse_pat_tree(args.tree)
    num_levels = len(nodes)
    batch_size = nodes[-1]

    assert len(nodes) == len(contexts), "nodes and contexts must have same length"
    assert all(batch_size % n == 0 for n in nodes), "Each level must evenly divide batch_size"

    num_kv_heads = args.num_kv_heads
    num_qo_heads = num_kv_heads * args.gqa
    head_dim = args.head_dim
    page_size = args.page_size

    pages_per_node = [(ctx + page_size - 1) // page_size for ctx in contexts]
    total_kv_groups = sum(nodes)
    total_pages = sum(nodes[level] * pages_per_node[level] for level in range(num_levels))
    total_queries = batch_size * num_levels

    print("=" * 70)
    print(f"PAT SCHEDULING BENCHMARK: {args.tree}")
    print("=" * 70)
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

    print("\nTree visualization:")
    for level in range(num_levels):
        seqs_per_node = batch_size // nodes[level]
        indent = "  " * level
        print(
            f"{indent}Level {level}: {nodes[level]} node(s), {contexts[level]} tokens, "
            f"{seqs_per_node} seqs/node"
        )

    # Build per-node KV pages
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

    # Build qo_indptr (per-node, same as mech2)
    qo_indptr_list = [0]
    for level in range(num_levels):
        seqs_per_node = batch_size // nodes[level]
        for _ in range(nodes[level]):
            qo_indptr_list.append(qo_indptr_list[-1] + seqs_per_node)

    qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device="cuda:0")

    print(f"\nqo_indptr: {qo_indptr}")
    print(f"kv_page_indptr: {kv_page_indptr}")
    print(f"kv_page_indices: {kv_page_indices}")
    print(f"kv_last_page_len: {kv_last_page_len}")

    kv_cache = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=torch.float16, device="cuda:0",
    )
    q = torch.randn(total_queries, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    # For PAT: leaf-only Q (one row per leaf sequence)
    q_leaf = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")

    print("\n" + "=" * 70)
    print("RUNNING: PAT scheduling (fused tree printed via FLASHINFER_DEBUG_SCHEDULER)")
    print("  use_tree_walk_scheduling = True")
    print("  mech2_mode               = True")
    print("  use_pat_scheduling       = True")
    print(f"  merge_every_n_levels     = {args.merge_every_n_levels}")
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
        use_pat_scheduling=True,
        tree_nodes=nodes,
    )

    # Warmup
    for _ in range(args.warmup_iters):
        out = wrapper.run(q_leaf, kv_cache, tree_nodes=nodes, merge_every_n_levels=args.merge_every_n_levels)
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.bench_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.bench_iters)]

    for i in range(args.bench_iters):
        start_events[i].record()
        out = wrapper.run(q_leaf, kv_cache, tree_nodes=nodes, merge_every_n_levels=args.merge_every_n_levels)
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    geo_mean_ms = _geom_mean_ms(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    print(f"\n  Output shape: {out.shape}")
    print(f"  Latency (ms): geo_mean={geo_mean_ms:.4f}  min={min_ms:.4f}  max={max_ms:.4f}")
    print("\nDONE")


if __name__ == "__main__":
    main()

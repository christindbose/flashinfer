"""
Single-iteration smoke test for Mech2 Cascade Attention (tree-walk scheduling).

Stripped-down version of bench_mech2_GQA_only.py: no warmup, no timing, runs
one tree exactly once. Tree specs use the same encoding as the benchmark:
<nodes_csv>_<contexts_csv>.
"""

import argparse
import sys

import torch
import flashinfer


DEFAULT_MECH2_PAT_TREES = [
    "1,64_128,32",
]

GQA_CONFIGS = [
    # (num_qo_heads, num_kv_heads)
    (32, 32),
]


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


def run_mech2_one_tree(
    nodes,
    contexts,
    *,
    merge_every,
    page_size,
    num_kv_heads,
    head_dim,
    gqa,
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
        print("RUNNING: Mech2 (single iteration, no timing)")
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

    out = wrapper.run(q, kv_cache, tree_nodes=nodes, merge_every_n_levels=merge_every)
    torch.cuda.synchronize()

    print(f"\n  Output shape: {out.shape}")
    print(f"  Expected:     [{batch_size}, {num_qo_heads}, {head_dim}]")
    print(f"  Output mean:  {out.mean().item():.6f}")
    print(f"  Output std:   {out.std().item():.6f}")
    print(f"\n  Sample outputs (first 8 sequences):")
    for seq_id in range(min(batch_size, 8)):
        out_val = out[seq_id, 0, 0].item()
        print(f"    Seq {seq_id}: out[{seq_id}][0,0] = {out_val:+.6f}")
    if batch_size > 8:
        print(f"    ... ({batch_size - 8} more sequences)")


def main():
    parser = argparse.ArgumentParser(description="Single-iteration test for Mech2 cascade attention")
    parser.add_argument(
        "--tree",
        type=str,
        default=None,
        metavar="PAT_SPEC",
        help="PAT-style tree n1,n2,..._c1,c2,... (single tree). "
        "If omitted and --nodes/--contexts omitted, runs the default tree.",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default=None,
        help='Comma-separated nodes per level (use with --contexts)',
    )
    parser.add_argument(
        "--contexts",
        type=str,
        default=None,
        help='Comma-separated context lengths per level (use with --nodes)',
    )
    parser.add_argument("--page_size", type=int, default=16, help="KV page size (default: 16)")
    parser.add_argument("--num_kv_heads", type=int, default=1, help="KV heads (default: 1)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dim (default: 128)")
    parser.add_argument("--gqa", type=int, default=1, help="GQA ratio (default: 1)")
    parser.add_argument("--merge_every_n_levels", type=int, default=2, help="default: 2")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Detailed prints (qo_indptr, samples, …)"
    )
    args = parser.parse_args()

    if args.nodes is not None and args.contexts is not None:
        if args.tree:
            print("error: use either --tree or --nodes with --contexts, not both", file=sys.stderr)
            sys.exit(2)
        nodes = [int(x) for x in args.nodes.split(",")]
        contexts = [int(x) for x in args.contexts.split(",")]
        label = f"{args.nodes}_{args.contexts}"
    elif args.nodes is not None or args.contexts is not None:
        print("error: supply both --nodes and --contexts", file=sys.stderr)
        sys.exit(2)
    elif args.tree:
        nodes, contexts = parse_pat_tree(args.tree)
        label = args.tree
    else:
        label = DEFAULT_MECH2_PAT_TREES[0]
        nodes, contexts = parse_pat_tree(label)

    num_qo_heads, num_kv_heads = GQA_CONFIGS[0]
    gqa = num_qo_heads // num_kv_heads

    print("=" * 70)
    print("TEST: MECH2 CASCADE ATTENTION (single iteration)")
    print(f"Tree: {label}  (qo={num_qo_heads}, kv={num_kv_heads}, gqa={gqa})")
    print("=" * 70)

    run_mech2_one_tree(
        nodes,
        contexts,
        merge_every=args.merge_every_n_levels,
        page_size=args.page_size,
        num_kv_heads=num_kv_heads,
        head_dim=args.head_dim,
        gqa=gqa,
        verbose=args.verbose,
    )

    print("\nDONE")


if __name__ == "__main__":
    main()

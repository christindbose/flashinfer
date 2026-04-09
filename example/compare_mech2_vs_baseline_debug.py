"""
Compare Mech2 vs Baseline Cascade Attention

Compares two settings on the same tree structure and KV data:

  Setting 1 (Mech2):
    use_tree_walk_scheduling=True, mech2_mode=True, merge_every_n_levels=2

  Setting 2 (Baseline):
    use_tree_walk_scheduling=False, mech2_mode=False, merge_every_n_levels=1

Tree structure: nodes=[1,3,9,27], contexts=[32,16,8,8]
            [Root: 32 tokens]                        <- Level 0: 1 node, all 27 seqs
            /        |        \
      [A: 16]    [B: 16]    [C: 16]                 <- Level 1: 3 nodes, 9 seqs each
      / | \      / | \      / | \
   (9 nodes, 8 tokens each)                          <- Level 2: 9 nodes, 3 seqs each
   / | \  ...
  (27 leaf nodes, 8 tokens each)                     <- Level 3: 27 nodes, 1 seq each
"""

import math
import torch
import flashinfer
import argparse


def _geom_mean_ms(times_ms):
    n = len(times_ms)
    if n == 0:
        raise ValueError("times_ms must be non-empty")
    return math.exp(sum(math.log(t) for t in times_ms) / n)


# Parse command line arguments
parser = argparse.ArgumentParser(description='Compare Mech2 vs Baseline Cascade Attention')
parser.add_argument('--nodes', type=str, default='1,3,9,27',
                    help='Nodes at each level, comma-separated (default: "1,3,9,27")')
parser.add_argument('--contexts', type=str, default='32,16,8,8',
                    help='Context tokens at each level, comma-separated (default: "32,16,8,8")')
parser.add_argument('--page_size', type=int, default=128,
                    help='Page size for KV cache (default: 128)')
parser.add_argument('--num_kv_heads', type=int, default=1,
                    help='Number of KV heads (default: 1)')
parser.add_argument('--head_dim', type=int, default=128,
                    help='Head dimension (default: 128)')
parser.add_argument('--gqa', type=int, default=1,
                    help='GQA ratio (default: 1)')
parser.add_argument('--warmup_iters', type=int, default=1,
                    help='Warmup iterations (default: 5)')
parser.add_argument('--bench_iters', type=int, default=1,
                    help='Benchmark iterations (default: 20)')
args = parser.parse_args()

# Parse tree structure
nodes = [int(x) for x in args.nodes.split(',')]
contexts = [int(x) for x in args.contexts.split(',')]
num_levels = len(nodes)
batch_size = nodes[-1]  # Number of leaf sequences

assert len(nodes) == len(contexts), "nodes and contexts must have same length"
assert all(nodes[-1] % n == 0 for n in nodes), "Each level must evenly divide batch_size"

# Model configuration
num_kv_heads = args.num_kv_heads
num_qo_heads = num_kv_heads * args.gqa
head_dim = args.head_dim
page_size = args.page_size

# Calculate pages per node at each level (ceiling division)
pages_per_node = [(ctx + page_size - 1) // page_size for ctx in contexts]

# Total statistics
total_kv_groups = sum(nodes)
total_pages = sum(nodes[level] * pages_per_node[level] for level in range(num_levels))
total_queries = batch_size * num_levels

print("=" * 70)
print("COMPARE: MECH2 vs BASELINE CASCADE ATTENTION")
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

# Visualize tree structure for both settings
for setting_name, merge_every in [("Mech2 (merge_every=2)", 2), ("Baseline (merge_every=1)", 1)]:
    print(f"\nTree visualization [{setting_name}]:")
    for level in range(num_levels):
        seqs_per_node = batch_size // nodes[level]
        indent = "  " * level
        will_merge = (level > 0 and (level - 1) % merge_every == 0) or level == 0
        merge_indicator = " [BASE]" if level == 0 else (" [MERGE]" if will_merge else " [SKIP]")
        print(f"{indent}Level {level}: {nodes[level]} node(s), {contexts[level]} tokens, "
              f"{seqs_per_node} seqs/node{merge_indicator}")

# ============================================================
# BUILD KV PAGE STRUCTURE (shared between both settings)
# ============================================================

kv_page_indices_list = []
kv_page_indptr_list = [0]
kv_last_page_len_list = []
page_offset = 0

for level in range(num_levels):
    last_page_len = contexts[level] % page_size
    if last_page_len == 0:
        last_page_len = page_size

    for node_id in range(nodes[level]):
        node_pages = list(range(page_offset, page_offset + pages_per_node[level]))
        kv_page_indices_list.extend(node_pages)
        kv_page_indptr_list.append(kv_page_indptr_list[-1] + pages_per_node[level])
        kv_last_page_len_list.append(last_page_len)
        page_offset += pages_per_node[level]

kv_page_indices = torch.tensor(kv_page_indices_list, dtype=torch.int32, device="cuda:0")
kv_page_indptr = torch.tensor(kv_page_indptr_list, dtype=torch.int32, device="cuda:0")
kv_last_page_len = torch.tensor(kv_last_page_len_list, dtype=torch.int32, device="cuda:0")

# ============================================================
# BUILD QO_INDPTR (shared between both settings)
# ============================================================

qo_indptr_list = [0]
for level in range(num_levels):
    seqs_per_node = batch_size // nodes[level]
    for node_id in range(nodes[level]):
        qo_indptr_list.append(qo_indptr_list[-1] + seqs_per_node)

qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device="cuda:0")

print(f"\nqo_indptr: {qo_indptr}")
print(f"kv_page_indptr: {kv_page_indptr}")
print(f"kv_page_indices: {kv_page_indices}")
print(f"kv_last_page_len: {kv_last_page_len}")

# ============================================================
# SHARED DATA
# ============================================================

# Allocate KV cache (shared)
kv_cache = torch.randn(
    total_pages, 2, page_size, num_kv_heads, head_dim,
    dtype=torch.float16, device="cuda:0"
)

# Create query tensor (shared)
q = torch.randn(total_queries, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")

# ============================================================
# DEFINE THE TWO SETTINGS
# ============================================================

settings = {
    "Mech2": {
        "use_tree_walk_scheduling": True,
        "mech2_mode": True,
        "kvsplit_mode": False,
        "merge_every_n_levels": 2,
    },
    "Baseline": {
        "use_tree_walk_scheduling": False,
        "mech2_mode": False,
        "kvsplit_mode": False,
        "merge_every_n_levels": 1,
    },
}

results = {}

for name, cfg in settings.items():
    print("\n" + "=" * 70)
    print(f"RUNNING: {name}")
    print(f"  use_tree_walk_scheduling = {cfg['use_tree_walk_scheduling']}")
    print(f"  mech2_mode               = {cfg['mech2_mode']}")
    print(f"  merge_every_n_levels     = {cfg['merge_every_n_levels']}")
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
        use_tree_walk_scheduling=cfg["use_tree_walk_scheduling"],
        kvsplit_mode=cfg["kvsplit_mode"],
        mech2_mode=cfg["mech2_mode"],
    )

    # Warmup
    for _ in range(args.warmup_iters):
        out = wrapper.run(q, kv_cache, tree_nodes=nodes,
                          merge_every_n_levels=cfg["merge_every_n_levels"])
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.bench_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.bench_iters)]

    for i in range(args.bench_iters):
        start_events[i].record()
        out = wrapper.run(q, kv_cache, tree_nodes=nodes,
                          merge_every_n_levels=cfg["merge_every_n_levels"])
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    geo_mean_ms = _geom_mean_ms(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    results[name] = {
        "output": out.clone(),
        "geo_mean_ms": geo_mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
    }

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

# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

out_mech2 = results["Mech2"]["output"]
out_baseline = results["Baseline"]["output"]

# Numerical difference
diff = (out_mech2 - out_baseline).abs()
max_abs_diff = diff.max().item()
mean_abs_diff = diff.mean().item()
# Relative diff (avoid division by zero)
denom = out_baseline.abs().clamp(min=1e-6)
rel_diff = (diff / denom)
max_rel_diff = rel_diff.max().item()
mean_rel_diff = rel_diff.mean().item()

print(f"\n{'Metric':<30} {'Mech2':>15} {'Baseline':>15}")
print("-" * 62)
print(f"{'Geo-mean latency (ms)':<30} {results['Mech2']['geo_mean_ms']:>15.4f} {results['Baseline']['geo_mean_ms']:>15.4f}")
print(f"{'Min latency (ms)':<30} {results['Mech2']['min_ms']:>15.4f} {results['Baseline']['min_ms']:>15.4f}")
print(f"{'Max latency (ms)':<30} {results['Mech2']['max_ms']:>15.4f} {results['Baseline']['max_ms']:>15.4f}")
print(f"{'Output mean':<30} {out_mech2.mean().item():>15.6f} {out_baseline.mean().item():>15.6f}")
print(f"{'Output std':<30} {out_mech2.std().item():>15.6f} {out_baseline.std().item():>15.6f}")

speedup = results["Baseline"]["geo_mean_ms"] / results["Mech2"]["geo_mean_ms"] if results["Mech2"]["geo_mean_ms"] > 0 else float('inf')

print(f"\n{'Numerical Difference':<30}")
print("-" * 62)
print(f"{'Max absolute diff':<30} {max_abs_diff:>15.8f}")
print(f"{'Mean absolute diff':<30} {mean_abs_diff:>15.8f}")
print(f"{'Max relative diff':<30} {max_rel_diff:>15.8f}")
print(f"{'Mean relative diff':<30} {mean_rel_diff:>15.8f}")

print(f"\n{'Speedup (Mech2 vs Baseline)':<30} {speedup:>15.4f}x")

# Per-sequence comparison
print(f"\nPer-sequence comparison (first 8):")
print(f"  {'Seq':<6} {'Mech2[0,0]':>14} {'Baseline[0,0]':>14} {'AbsDiff':>12}")
print("  " + "-" * 48)
for seq_id in range(min(batch_size, 8)):
    v1 = out_mech2[seq_id, 0, 0].item()
    v2 = out_baseline[seq_id, 0, 0].item()
    d = abs(v1 - v2)
    print(f"  {seq_id:<6} {v1:>+14.6f} {v2:>+14.6f} {d:>12.8f}")
if batch_size > 8:
    print(f"  ... ({batch_size - 8} more sequences)")

print("\n" + "=" * 70)
print("DONE - Comparison complete!")
print("=" * 70)

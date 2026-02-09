"""
Fused Cascade Attention for Arbitrary Tree Structures

This script demonstrates how to fuse all levels of a tree-structured cascade attention
into a single kernel call by carefully constructing qo_indptr and kv_page_indptr.

Tree structure is specified as:
  --nodes: comma-separated node counts at each level (e.g., "1,2,64")
  --contexts: comma-separated context lengths at each level (e.g., "8,256,32")

Example: "1,2,64" with "8,256,32" means:
  - Level 0: 1 root node with 8 tokens (all 64 seqs share)
  - Level 1: 2 nodes with 256 tokens each (32 seqs share each)
  - Level 2: 64 leaf nodes with 32 tokens each (1 seq each)

The tree looks like:
            [Root: 8 tokens]                <- Level 0: 1 node, all 64 seqs attend
            /              \
     [Node A: 256]    [Node B: 256]         <- Level 1: 2 nodes, 32 seqs each
     /    ...    \    /    ...    \
    S0   ...    S31  S32   ...   S63        <- Level 2: 64 unique nodes
"""

import torch
import flashinfer
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fused Cascade Attention for Tree Structures')
parser.add_argument('--nodes', type=str, default='1,2,64',
                    help='Nodes at each level, comma-separated (e.g., "1,2,64")')
parser.add_argument('--contexts', type=str, default='8,256,32',
                    help='Context tokens at each level, comma-separated (e.g., "8,256,32")')
parser.add_argument('--page_size', type=int, default=128,
                    help='Page size for KV cache (default: 128)')
parser.add_argument('--num_kv_heads', type=int, default=1,
                    help='Number of KV heads (default: 1)')
parser.add_argument('--head_dim', type=int, default=128,
                    help='Head dimension (default: 128)')
parser.add_argument('--gqa', type=int, default=1,
                    help='GQA ratio (default: 1)')
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

print("=" * 60)
print("FUSED CASCADE ATTENTION - TREE STRUCTURE")
print("=" * 60)
print(f"Nodes per level:    {nodes}")
print(f"Contexts per level: {contexts}")
print(f"Batch size (leaves): {batch_size}")
print(f"Total KV groups:    {total_kv_groups}")
print(f"Total queries:      {total_queries}")

# Visualize tree structure
print("\nTree visualization:")
for level in range(num_levels):
    seqs_per_node = batch_size // nodes[level]
    indent = "  " * level
    print(f"{indent}Level {level}: {nodes[level]} node(s), {contexts[level]} tokens, {seqs_per_node} seqs/node")

# ============================================================
# BUILD KV PAGE STRUCTURE
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
# BUILD QO_INDPTR (Query-to-KV-Group mapping)
# ============================================================

qo_indptr_list = [0]
for level in range(num_levels):
    seqs_per_node = batch_size // nodes[level]
    for node_id in range(nodes[level]):
        qo_indptr_list.append(qo_indptr_list[-1] + seqs_per_node)

qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device="cuda:0")

# ============================================================
# ALLOCATE AND RUN ATTENTION
# ============================================================

print("\n" + "=" * 60)
print("RUNNING FUSED CASCADE ATTENTION")
print("=" * 60)

# Allocate workspace buffer
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")

# Create wrapper with 1 level (we fuse everything into single kernel)
wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(1, workspace_buffer, "NHD")

# Allocate KV cache
kv_cache = torch.randn(
    total_pages, 2, page_size, num_kv_heads, head_dim,
    dtype=torch.float16, device="cuda:0"
)

# Plan the attention
wrapper.plan(
    [qo_indptr],
    [kv_page_indptr],
    [kv_page_indices],
    [kv_last_page_len],
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
)

# Create query tensor
q = torch.randn(total_queries, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")

# Run fused cascade attention with hierarchical merge
merged_out = wrapper.run(q, kv_cache, tree_nodes=nodes)

# ============================================================
# DISPLAY MERGED RESULTS
# ============================================================

print(f"\nMerged output shape: {merged_out.shape}")
print(f"Expected shape: [{batch_size}, {num_qo_heads}, {head_dim}]")

print("\nSample merged outputs (first 8 sequences):")
for seq_id in range(min(batch_size, 8)):
    out_val = merged_out[seq_id, 0, 0].item()
    print(f"  Seq {seq_id}: merged_out[{seq_id}][0,0] = {out_val:+.6f}")

if batch_size > 8:
    print(f"  ... ({batch_size - 8} more sequences)")

print(f"\nMerged output stats:")
print(f"  mean = {merged_out.mean().item():.6f}")
print(f"  std  = {merged_out.std().item():.6f}")

print("\n" + "=" * 60)
print("DONE - Fused cascade attention complete!")
print("=" * 60)

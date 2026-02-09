"""
NCU Profiling: Mech2 vs Baseline Cascade Attention

Runs each setting exactly once for clean ncu profiling.

Usage:
  ncu --set full --nvtx --nvtx-include "mech2/,baseline/" -o mech2_vs_baseline python ncu_profile_mech2_vs_baseline.py

NVTX ranges label each setting's kernels so they are easy to identify in the ncu report.

Settings:
  Setting 1 (Mech2):    use_tree_walk_scheduling=True,  mech2_mode=True,  merge_every_n_levels=2
  Setting 2 (Baseline): use_tree_walk_scheduling=False, mech2_mode=False, merge_every_n_levels=1

Tree structure: nodes=[1,3,9,27], contexts=[32,16,8,8]
"""

import torch
import flashinfer
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='NCU Profile: Mech2 vs Baseline Cascade Attention')
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
args = parser.parse_args()

# Parse tree structure
nodes = [int(x) for x in args.nodes.split(',')]
contexts = [int(x) for x in args.contexts.split(',')]
num_levels = len(nodes)
batch_size = nodes[-1]

assert len(nodes) == len(contexts), "nodes and contexts must have same length"
assert all(nodes[-1] % n == 0 for n in nodes), "Each level must evenly divide batch_size"

# Model configuration
num_kv_heads = args.num_kv_heads
num_qo_heads = num_kv_heads * args.gqa
head_dim = args.head_dim
page_size = args.page_size

pages_per_node = [(ctx + page_size - 1) // page_size for ctx in contexts]
total_pages = sum(nodes[level] * pages_per_node[level] for level in range(num_levels))
total_queries = batch_size * num_levels

print(f"nodes={nodes}  contexts={contexts}  batch_size={batch_size}")
print(f"total_pages={total_pages}  total_queries={total_queries}")

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
# BUILD QO_INDPTR
# ============================================================

qo_indptr_list = [0]
for level in range(num_levels):
    seqs_per_node = batch_size // nodes[level]
    for node_id in range(nodes[level]):
        qo_indptr_list.append(qo_indptr_list[-1] + seqs_per_node)

qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device="cuda:0")

# ============================================================
# SHARED DATA
# ============================================================

kv_cache = torch.randn(
    total_pages, 2, page_size, num_kv_heads, head_dim,
    dtype=torch.float16, device="cuda:0"
)
q = torch.randn(total_queries, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")

# ============================================================
# SETTINGS
# ============================================================

settings = {
    "mech2": {
        "use_tree_walk_scheduling": True,
        "mech2_mode": True,
        "kvsplit_mode": False,
        "merge_every_n_levels": 2,
    },
    "baseline": {
        "use_tree_walk_scheduling": False,
        "mech2_mode": False,
        "kvsplit_mode": False,
        "merge_every_n_levels": 1,
    },
}

# ============================================================
# PLAN BOTH SETTINGS (outside profiled region)
# ============================================================

wrappers = {}
for name, cfg in settings.items():
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
    wrappers[name] = wrapper
    print(f"Planned: {name}  tree_walk={cfg['use_tree_walk_scheduling']}  mech2={cfg['mech2_mode']}  merge_every={cfg['merge_every_n_levels']}")

torch.cuda.synchronize()

# ============================================================
# RUN (single invocation per setting, NVTX-annotated for ncu)
# ============================================================

for name, cfg in settings.items():
    print(f"\n--- {name.upper()} ---")
    torch.cuda.nvtx.range_push(name)
    out = wrappers[name].run(q, kv_cache, tree_nodes=nodes,
                             merge_every_n_levels=cfg["merge_every_n_levels"])
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    #print(f"  output shape: {out.shape}  mean={out.mean().item():.6f}  std={out.std().item():.6f}")

print("\nDone.")

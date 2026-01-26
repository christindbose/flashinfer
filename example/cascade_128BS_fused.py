import torch
import flashinfer
import ctypes
import os
import triton
import argparse

os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
_cudart = ctypes.CDLL("libcudart.so")

def cu_prof_start():
    """Start CUDA profiler."""
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise RuntimeError(f"cudaProfilerStart() returned {ret}")


def cu_prof_stop():
    """Stop CUDA profiler."""
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise RuntimeError(f"cudaProfilerStop() returned {ret}")
    

# Parse command line arguments
parser = argparse.ArgumentParser(description='FlashInfer Fused Cascade Attention (Combined Level)')
parser.add_argument('--shared_kv_num_pages', type=int, default=512, 
                    help='Number of shared KV pages (default: 512)')
parser.add_argument('--unique_kv_pages_per_seq', type=int, default=1,
                    help='Number of unique KV pages per sequence (default: 1)')
parser.add_argument('--gqa', type=int, default=1, 
                    help='GQA ratio, number of KV heads per Q head (default: 1)')
args = parser.parse_args()

num_layers = 1
num_kv_heads = 1
num_qo_heads = num_kv_heads * args.gqa
head_dim = 128
page_size = 128
batch_size = 128

# KV cache configuration
shared_kv_num_pages = args.shared_kv_num_pages
unique_kv_pages_per_seq = args.unique_kv_pages_per_seq
total_unique_pages = batch_size * unique_kv_pages_per_seq
total_num_pages = shared_kv_num_pages + total_unique_pages

# Get GPU SM info
device = torch.cuda.current_device()
num_sms = torch.cuda.get_device_properties(device).multi_processor_count
gpu_name = torch.cuda.get_device_properties(device).name

print(f"GPU Info:")
print(f"  Device: {gpu_name}")
print(f"  Total SMs: {num_sms}")

print(f"\nConfiguration:")
print(f"  Batch size: {batch_size}")
print(f"  Shared KV pages: {shared_kv_num_pages} ({shared_kv_num_pages * page_size} tokens)")
print(f"  Unique KV pages per seq: {unique_kv_pages_per_seq} ({unique_kv_pages_per_seq * page_size} tokens)")
print(f"  Total pages: {total_num_pages}")

# Allocate workspace buffer
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")

# Use MultiLevelCascadeAttentionWrapper with 1 level (combined shared + unique)
wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
    1, workspace_buffer, "NHD",
)

# Allocate KV cache
kv_cache_at_layer = [
    torch.randn(
        total_num_pages, 2, page_size, num_kv_heads, head_dim, 
        dtype=torch.float16, device="cuda:0"
    ) for _ in range(num_layers)
]

# ============================================================
# Single Level with shared pages stored ONCE:
# - KV Group 0: shared pages (all batch queries attend to this)
# - KV Groups 1 to batch_size: unique pages per sequence
# ============================================================

# combined_kv_page_indices: [shared_pages..., unique_seq0..., unique_seq1..., ...]
# Shared pages stored exactly once
shared_page_indices = torch.arange(shared_kv_num_pages, dtype=torch.int32, device="cuda:0")
# Unique pages for all sequences
unique_page_indices = torch.arange(shared_kv_num_pages, total_num_pages, dtype=torch.int32, device="cuda:0")
# Concatenate: shared once + all unique
combined_kv_page_indices = torch.cat([shared_page_indices, unique_page_indices])
print(f"  combined_kv_page_indices : {combined_kv_page_indices}")

# combined_kv_page_indptr: 
# [0, shared_kv_num_pages, shared_kv_num_pages + unique_per_seq, shared_kv_num_pages + 2*unique_per_seq, ...]
# - indptr[0] to indptr[1]: shared pages (group 0)
# - indptr[1] to indptr[2]: unique pages for seq 0 (group 1)
# - indptr[2] to indptr[3]: unique pages for seq 1 (group 2)
# - etc.
# Total: batch_size + 2 entries (1 shared group + batch_size unique groups + 1 for end)
combined_kv_page_indptr = torch.cat([
    torch.tensor([0, shared_kv_num_pages], dtype=torch.int32, device="cuda:0"),
    torch.arange(
        shared_kv_num_pages + unique_kv_pages_per_seq,
        shared_kv_num_pages + (batch_size + 1) * unique_kv_pages_per_seq,
        unique_kv_pages_per_seq,
        dtype=torch.int32,
        device="cuda:0"
    )
])
print(f"  combined_kv_page_indptr : {combined_kv_page_indptr}")

# combined_kv_last_page_len: batch_size + 1 entries (1 for shared, batch_size for unique)
combined_kv_last_page_len = torch.full(
    (batch_size + 1,), 
    page_size,
    dtype=torch.int32, 
    device="cuda:0"
)

# qo_indptr: 
# [0, batch_size, batch_size+1, batch_size+2, ..., 2*batch_size]
# - Queries 0 to batch_size-1 attend to KV group 0 (shared)
# - Query batch_size attends to KV group 1 (unique for seq 0)
# - Query batch_size+1 attends to KV group 2 (unique for seq 1)
# - etc.
qo_indptr = torch.cat([
    torch.tensor([0, batch_size], dtype=torch.int32, device="cuda:0"),
    torch.arange(batch_size + 1, 2 * batch_size + 1, dtype=torch.int32, device="cuda:0")
])
print(f"  qo_indptr : {qo_indptr}")

print(f"\nCombined KV structure (single level, shared stored once):")
print(f"  combined_kv_page_indices shape: {combined_kv_page_indices.shape}")
print(f"  combined_kv_page_indptr shape: {combined_kv_page_indptr.shape} (batch_size + 2 = {batch_size + 2})")
print(f"  qo_indptr shape: {qo_indptr.shape} (batch_size + 2 = {batch_size + 2})")
print(f"  Number of KV groups: {len(combined_kv_page_indptr) - 1} (1 shared + {batch_size} unique)")

# SM utilization analysis
num_kv_groups = len(combined_kv_page_indptr) - 1  # batch_size + 1 groups
total_queries = 2 * batch_size
num_blocks_per_sm = 2  # typical for prefill kernels

# For prefill with paged KV: grid is typically (num_sm, num_qo_heads) or work-based
max_grid_size = num_blocks_per_sm * num_sms

# Estimate work items (depends on scheduler, this is approximate)
# Each KV group with queries creates work
estimated_work_items = num_kv_groups * num_qo_heads

print(f"\nSM Utilization Analysis:")
print(f"  Total SMs available: {num_sms}")
print(f"  Max grid size (blocks_per_sm * num_sm): {max_grid_size}")
print(f"  Number of KV groups: {num_kv_groups}")
print(f"  Total queries: {total_queries}")
print(f"  Estimated work items: {estimated_work_items}")
print(f"  Work items per SM: {estimated_work_items / num_sms:.2f}")
print(f"  SM saturation: {'YES' if estimated_work_items >= num_sms else 'NO (may need split KV)'}")

# Plan the attention (single level with combined KV)
wrapper.plan(
    [qo_indptr],
    [combined_kv_page_indptr],
    [combined_kv_page_indices],
    [combined_kv_last_page_len],
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
)
print("Finished planning fused cascade attention (combined level)")

# Run attention
# Note: q needs 2*batch_size queries (batch_size for shared, batch_size for unique)
outputs = []
for i in range(num_layers):
    q = torch.randn(2 * batch_size, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    
    # Single level processes: shared attention + unique attention
    o = wrapper.run(q, kv_cache_at_layer[i])
    outputs.append(o)

print(f"\nOutput shape: {outputs[0].shape}")
print(f"Output[0,0,0] (shared attention result): {outputs[0][0, 0, 0].item():.6f}")
print(f"Output[{batch_size},0,0] (unique attention for seq 0): {outputs[0][batch_size, 0, 0].item():.6f}")
print("\nDone! Fused cascade attention (combined level, shared stored once) completed.")

# Note: This stores shared pages once, but outputs are separate:
# - outputs[0:batch_size]: attention over shared KV only
# - outputs[batch_size:2*batch_size]: attention over unique KV per sequence
# To get final cascade result, you need to manually merge these using cascade reduction.

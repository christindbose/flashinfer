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
parser = argparse.ArgumentParser(description='FlashInfer Fused Cascade Attention (Option 1: Concatenate KV)')
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
batch_size = 256

# KV cache configuration
shared_kv_num_pages = args.shared_kv_num_pages
unique_kv_pages_per_seq = args.unique_kv_pages_per_seq
total_unique_pages = batch_size * unique_kv_pages_per_seq
total_num_pages = shared_kv_num_pages + total_unique_pages

print(f"Configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Shared KV pages: {shared_kv_num_pages} ({shared_kv_num_pages * page_size} tokens)")
print(f"  Unique KV pages per seq: {unique_kv_pages_per_seq} ({unique_kv_pages_per_seq * page_size} tokens)")
print(f"  Total pages: {total_num_pages}")

# Allocate workspace buffer
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")

# Use single-level BatchPrefillWithPagedKVCacheWrapper (not multi-level)
wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")

# Allocate KV cache
kv_cache_at_layer = [
    torch.randn(
        total_num_pages, 2, page_size, num_kv_heads, head_dim, 
        dtype=torch.float16, device="cuda:0"
    ) for _ in range(num_layers)
]

# ============================================================
# Option 1: Concatenate shared + unique KV indices per sequence
# Each sequence sees: [shared_page_0, ..., shared_page_N, unique_page_0, ..., unique_page_M]
# ============================================================

# Build combined page indices for all sequences
# For each sequence i: indices = [shared_pages..., unique_pages_for_seq_i...]
combined_kv_page_indices_list = []
for seq_idx in range(batch_size):
    # Shared pages (same for all sequences)
    shared_indices = torch.arange(shared_kv_num_pages, dtype=torch.int32, device="cuda:0")
    
    # Unique pages for this sequence
    unique_start = shared_kv_num_pages + seq_idx * unique_kv_pages_per_seq
    unique_indices = torch.arange(
        unique_start, 
        unique_start + unique_kv_pages_per_seq, 
        dtype=torch.int32, 
        device="cuda:0"
    )
    
    # Concatenate: [shared..., unique...]
    seq_indices = torch.cat([shared_indices, unique_indices])
    combined_kv_page_indices_list.append(seq_indices)

# Flatten all indices into single tensor
combined_kv_page_indices = torch.cat(combined_kv_page_indices_list)

# Build combined indptr
# Each sequence has (shared_kv_num_pages + unique_kv_pages_per_seq) pages
pages_per_seq = shared_kv_num_pages + unique_kv_pages_per_seq
combined_kv_page_indptr = torch.arange(
    0, 
    (batch_size + 1) * pages_per_seq, 
    pages_per_seq,
    dtype=torch.int32, 
    device="cuda:0"
)

# Build combined last_page_len (last page of unique portion)
# Assuming unique pages are fully filled
combined_kv_last_page_len = torch.full(
    (batch_size,), 
    page_size,  # or set to actual last page length if partial
    dtype=torch.int32, 
    device="cuda:0"
)

# qo_indptr: one query per sequence (decode scenario)
qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda:0")

print(f"\nCombined KV structure:")
print(f"  combined_kv_page_indices shape: {combined_kv_page_indices.shape}")
print(f"  combined_kv_page_indptr shape: {combined_kv_page_indptr.shape}")
print(f"  Pages per sequence: {pages_per_seq}")

print(f"  combined_kv_page_indices : {combined_kv_page_indices}")
print(f"  combined_kv_page_indptr : {combined_kv_page_indptr}")


# Plan the attention (single kernel)
wrapper.plan(
    qo_indptr,
    combined_kv_page_indptr,
    combined_kv_page_indices,
    combined_kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
)
print("Finished planning fused cascade attention")

# Run attention
outputs = []
for i in range(num_layers):
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    
    # Single kernel processes both shared + unique KV
    o = wrapper.run(q, kv_cache_at_layer[i])
    outputs.append(o)

print(f"\nOutput shape: {outputs[0].shape}")
print(f"Output[0,0,0]: {outputs[0][0, 0, 0].item():.6f}")
print("\nDone! Fused cascade attention completed in single kernel.")

# Note: This approach does NOT exploit sharing - shared KV is read batch_size times.
# For true sharing benefits, need kernel modifications (Option 2 or 3).

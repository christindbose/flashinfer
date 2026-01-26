import torch
import flashinfer
import ctypes
import os
import csv
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
parser = argparse.ArgumentParser(description='FlashInfer Multi-Level Cascade Attention Sweep')
parser.add_argument('--gqa', type=int, default=1, 
                    help='GQA ratio, number of KV heads per Q head (default: 1)')
args = parser.parse_args()

# Sweep configuration
shared_kv_num_pages_list = [32, 64, 128, 256, 512, 768, 1024]

num_layers = 1
num_kv_heads = 1
num_qo_heads = num_kv_heads * args.gqa
head_dim = 128
page_size = 128
batch_size = 256
unique_kv_num_pages = 256

# allocate 128MB workspace buffer
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")

# Store results for tabulation
results = []

print(f"Sweeping shared_kv_num_pages from {shared_kv_num_pages_list[0]} to {shared_kv_num_pages_list[-1]}")
print(f"Configuration: batch_size={batch_size}, num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, page_size={page_size}")
print("-" * 80)

for shared_kv_num_pages in shared_kv_num_pages_list:
    # Calculate total context length for this configuration
    shared_kv_len = shared_kv_num_pages * page_size
    unique_kv_len = unique_kv_num_pages * page_size
    total_kv_len = shared_kv_len + unique_kv_len
    
    print(f"\nTesting shared_kv_num_pages={shared_kv_num_pages} (shared_kv_len={shared_kv_len}, total_kv_len={total_kv_len})...")
    
    wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        2, workspace_buffer, "NHD",
    )
    
    total_num_pages = shared_kv_num_pages + unique_kv_num_pages
    shared_kv_page_indices = torch.arange(shared_kv_num_pages).int().to("cuda:0")
    shared_kv_page_indptr = torch.tensor([0, shared_kv_num_pages], dtype=torch.int32, device="cuda:0")
    unique_kv_page_indices = torch.arange(shared_kv_num_pages, total_num_pages).int().to("cuda:0")
    unique_kv_page_indptr = torch.arange(0, 257, 1, dtype=torch.int32, device="cuda:0")
    
    shared_kv_last_page_len = torch.tensor([page_size], dtype=torch.int32, device="cuda:0")
    unique_kv_last_page_len = torch.ones(256, dtype=torch.int32, device="cuda:0")
    
    kv_cache_at_layer = [
        torch.randn(
            total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
        ) for _ in range(num_layers)
    ]
    qo_indptr_arr = [
        torch.tensor([0, batch_size], dtype=torch.int32, device="cuda:0"),
        torch.arange(batch_size + 1, dtype=torch.int32, device="cuda:0")
    ]
    
    wrapper.plan(
        qo_indptr_arr,
        [shared_kv_page_indptr, unique_kv_page_indptr],
        [shared_kv_page_indices, unique_kv_page_indices],
        [shared_kv_last_page_len, unique_kv_last_page_len],
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )
    
    q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    
    # Warmup run
    _ = wrapper.run(q, kv_cache_at_layer[0])
    
    # Benchmark
    t_median, t_min, t_max = triton.testing.do_bench(
        lambda: wrapper.run(q, kv_cache_at_layer[0]),
        quantiles=[0.5, 0.2, 0.8],
        warmup=10,
        rep=100
    )
    
    results.append({
        'shared_kv_num_pages': shared_kv_num_pages,
        'shared_kv_len': shared_kv_len,
        'total_kv_len': total_kv_len,
        'time_ms': t_median,
        'time_min_ms': t_min,
        'time_max_ms': t_max,
    })
    
    print(f"  Time: {t_median:.4f} ms (min: {t_min:.4f}, max: {t_max:.4f})")

# Print tabulated results
print("\n" + "=" * 100)
print("RESULTS SUMMARY")
print("=" * 100)
print(f"{'shared_kv_pages':>16} | {'shared_kv_len':>14} | {'total_kv_len':>13} | {'time (ms)':>12} | {'min (ms)':>10} | {'max (ms)':>10}")
print("-" * 100)
for r in results:
    print(f"{r['shared_kv_num_pages']:>16} | {r['shared_kv_len']:>14} | {r['total_kv_len']:>13} | {r['time_ms']:>12.4f} | {r['time_min_ms']:>10.4f} | {r['time_max_ms']:>10.4f}")
print("=" * 100)

# Write results to CSV
csv_filename = f"cascade_sweep_bs{batch_size}_gqa{args.gqa}.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['shared_kv_num_pages', 'shared_kv_len', 'total_kv_len', 'time_ms', 'time_min_ms', 'time_max_ms']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {csv_filename}")

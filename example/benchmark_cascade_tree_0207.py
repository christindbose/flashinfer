"""
Benchmark Fused Cascade Attention for Various Tree Structures

Benchmarks tree-structured cascade attention and outputs results to CSV.
"""

import torch
import flashinfer
import triton
import argparse
import csv
from datetime import datetime

# Tree structures to benchmark (nodes, contexts)
# Format: "nodes contexts" where nodes and contexts are comma-separated
"""
TREE_CONFIGS = [
    "1,2,64 8,256,32",
    "1,4,256 8,256,32",
    "1,4,8,256 8,256,256,32",
    "1,256 256,32",
    "1,1024 2048,32",
    "1,16,64 1024,256,32",
    "1,4,16,512 1024,256,128,32",
    "1,4,16,64,256,1024 256,8,256,64,32,256",
    "1,10 4000,400",

    "1,2,4,8,16,32,64,128,1024 16,16,16,16,16,16,16,16,16",
    "1,2,4,8,16,32,64,128,1024 256,128,64,16,16,16,16,16,16",
    "1,8,16,32,64,128,1024 256,128,64,16,16,16,16",
    "1,8,16,32,64,256,1024 256,128,64,16,16,16,16",
    "1,16,32,64,128,1024 256,128,64,16,16,16",
    "1,16,32,64,256,1024 256,128,64,16,16,16",
]
"""

TREE_CONFIGS = [
  "1,256 256,32",
  "1,1024 2048,32",
  "1,16,64 1024,256,32",
  "1,4,16,512 1024,256,128,32",
  "1,10 4000,400",
  "1,2,4,8,16,32,64,128,1024 16,16,16,16,16,16,16,16,16",
  "1,2,4,8,16,32,64,128,1024 256,128,64,16,16,16,16,16,16",
  "1,8,16,32,64,128,1024 256,128,64,16,16,16,16",
  "1,8,16,32,64,256,1024 256,128,64,16,16,16,16",
  "1,16,32,64,128,1024 256,128,64,16,16,16",
  "1,16,32,64,256,1024 256,128,64,16,16,16",
]
  #  "1,256_256,32"
  #  "1,1024_2048,32"
  #  "1,16,64_1024,256,32"
  #  "1,4,16,512_1024,256,128,32"
  #  "1,10_4000,400"
  #  "1,2,4,8,16,32,64,128,1024_16,16,16,16,16,16,16,16,16"
  #  "1,2,4,8,16,32,64,128,1024_256,128,64,16,16,16,16,16,16"
  #  "1,8,16,32,64,128,1024_256,128,64,16,16,16,16"
  #  "1,8,16,32,64,256,1024_256,128,64,16,16,16,16"
  #  "1,16,32,64,128,1024_256,128,64,16,16,16"
  #  "1,16,32,64,256,1024_256,128,64,16,16,16"


def benchmark_tree_config(nodes_str, contexts_str, page_size=16, num_kv_heads=1, 
                          head_dim=128, gqa=1, warmup=10, rep=100):
    """Benchmark a single tree configuration."""
    
    # Parse tree structure
    nodes = [int(x) for x in nodes_str.split(',')]
    contexts = [int(x) for x in contexts_str.split(',')]
    num_levels = len(nodes)
    batch_size = nodes[-1]
    
    # Model configuration
    num_qo_heads = num_kv_heads * gqa
    
    # Calculate pages per node at each level
    pages_per_node = [(ctx + page_size - 1) // page_size for ctx in contexts]
    
    # Total statistics
    total_kv_groups = sum(nodes)
    total_pages = sum(nodes[level] * pages_per_node[level] for level in range(num_levels))
    total_queries = batch_size * num_levels
    total_kv_tokens = sum(nodes[level] * contexts[level] for level in range(num_levels))
    
    # Build KV page structure
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
    
    # Build qo_indptr
    qo_indptr_list = [0]
    for level in range(num_levels):
        seqs_per_node = batch_size // nodes[level]
        for node_id in range(nodes[level]):
            qo_indptr_list.append(qo_indptr_list[-1] + seqs_per_node)
    
    qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device="cuda:0")
    
    # Allocate workspace and wrapper
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
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
    
    # Benchmark
    t_median, t_min, t_max = triton.testing.do_bench(
        lambda: wrapper.run(q, kv_cache, tree_nodes=nodes),
        quantiles=[0.5, 0.2, 0.8],
        warmup=warmup,
        rep=rep
    )
    
    return {
        'nodes': nodes_str,
        'contexts': contexts_str,
        'num_levels': num_levels,
        'batch_size': batch_size,
        'total_kv_groups': total_kv_groups,
        'total_queries': total_queries,
        'total_pages': total_pages,
        'total_kv_tokens': total_kv_tokens,
        't_median_ms': t_median,
        't_min_ms': t_min,
        't_max_ms': t_max,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark Fused Cascade Attention')
    parser.add_argument('--output', type=str, default='cascade_benchmark_results.csv',
                        help='Output CSV file (default: cascade_benchmark_results.csv)')
    parser.add_argument('--page_size', type=int, default=128,
                        help='Page size for KV cache (default: 128)')
    parser.add_argument('--num_kv_heads', type=int, default=1,
                        help='Number of KV heads (default: 1)')
    parser.add_argument('--head_dim', type=int, default=128,
                        help='Head dimension (default: 128)')
    parser.add_argument('--gqa', type=int, default=1,
                        help='GQA ratio (default: 1)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations (default: 10)')
    parser.add_argument('--rep', type=int, default=100,
                        help='Benchmark repetitions (default: 100)')
    parser.add_argument('--configs', type=str, default=None,
                        help='Specific configs to run, semicolon-separated (e.g., "1,2,64 8,256,32;1,1024 2048,32")')
    args = parser.parse_args()
    
    # Get GPU info
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_properties(device).name
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    
    print("=" * 70)
    print("CASCADE TREE ATTENTION BENCHMARK")
    print("=" * 70)
    print(f"GPU: {gpu_name} ({num_sms} SMs)")
    print(f"Page size: {args.page_size}")
    print(f"KV heads: {args.num_kv_heads}, Head dim: {args.head_dim}, GQA: {args.gqa}")
    print(f"Warmup: {args.warmup}, Repetitions: {args.rep}")
    print(f"Output: {args.output}")
    print("=" * 70)
    
    # Select configs to run
    if args.configs:
        configs = args.configs.split(';')
    else:
        configs = TREE_CONFIGS
    
    results = []
    
    for i, config in enumerate(configs):
        parts = config.strip().split()
        if len(parts) != 2:
            print(f"Skipping invalid config: {config}")
            continue
        
        nodes_str, contexts_str = parts
        
        print(f"\n[{i+1}/{len(configs)}] Benchmarking: nodes={nodes_str}, contexts={contexts_str}")
        
        try:
            result = benchmark_tree_config(
                nodes_str, contexts_str,
                page_size=args.page_size,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                gqa=args.gqa,
                warmup=args.warmup,
                rep=args.rep
            )
            results.append(result)
            
            print(f"  Levels: {result['num_levels']}, Batch: {result['batch_size']}, "
                  f"KV groups: {result['total_kv_groups']}, Queries: {result['total_queries']}")
            print(f"  Total KV tokens: {result['total_kv_tokens']}, Pages: {result['total_pages']}")
            print(f"  Time: {result['t_median_ms']:.3f} ms (median), "
                  f"{result['t_min_ms']:.3f} ms (min), {result['t_max_ms']:.3f} ms (max)")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Write results to CSV
    if results:
        fieldnames = ['nodes', 'contexts', 'num_levels', 'batch_size', 'total_kv_groups',
                      'total_queries', 'total_pages', 'total_kv_tokens', 
                      't_median_ms', 't_min_ms', 't_max_ms']
        
        with open(args.output, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n{'=' * 70}")
        print(f"Results written to {args.output}")
        print(f"{'=' * 70}")
        
        # Print summary table
        print("\nSUMMARY:")
        print(f"{'Nodes':<40} {'Batch':>6} {'Levels':>6} {'KV Tok':>10} {'Time(ms)':>10}")
        print("-" * 80)
        for r in results:
            nodes_short = r['nodes'][:37] + '...' if len(r['nodes']) > 40 else r['nodes']
            print(f"{nodes_short:<40} {r['batch_size']:>6} {r['num_levels']:>6} "
                  f"{r['total_kv_tokens']:>10} {r['t_median_ms']:>10.3f}")
    else:
        print("\nNo successful benchmarks to write.")


if __name__ == "__main__":
    main()

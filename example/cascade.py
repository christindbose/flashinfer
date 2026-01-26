import torch
import flashinfer
import ctypes
import os
import triton

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
    

num_layers = 1 #32
num_qo_heads = 64
num_kv_heads = 8
head_dim = 128
page_size = 128
# allocate 128MB workspace buffer
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
    2, workspace_buffer, "NHD",
)
batch_size = 7
shared_kv_num_pages = 512
unique_kv_num_pages = 128 #256 #128
total_num_pages = shared_kv_num_pages + unique_kv_num_pages
shared_kv_page_indices = torch.arange(shared_kv_num_pages).int().to("cuda:0")
shared_kv_page_indptr = torch.tensor([0, shared_kv_num_pages], dtype=torch.int32, device="cuda:0")
unique_kv_page_indices = torch.arange(shared_kv_num_pages, total_num_pages).int().to("cuda:0")
unique_kv_page_indptr = torch.tensor(
    [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
)
shared_kv_last_page_len = torch.tensor([page_size], dtype=torch.int32, device="cuda:0")
# 1 <= kv_last_page_len <= page_size
unique_kv_last_page_len = torch.tensor(
    [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
)
kv_cache_at_layer = [
    torch.randn(
        total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ) for _ in range(num_layers)
]
qo_indptr_arr = [
    torch.tensor([0, batch_size], dtype=torch.int32, device="cuda:0"),  # top-level for shared KV-Cache
    torch.arange(batch_size + 1, dtype=torch.int32, device="cuda:0")    # bottom-level for unique KV-Cache
]

print("finished setting up multi-level cascade attention wrapper")
# create auxiliary data structures for batch decode attention
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
print("finished planning multi-level cascade attention wrapper")
outputs = []
for i in range(num_layers):
    #print(f"running layer {i}")
    q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    # compute batch decode attention, reuse auxiliary data structures for all layers
    o = wrapper.run(q, kv_cache_at_layer[i])
    outputs.append(o)

print(outputs[0].shape)




t, _, _ = triton.testing.do_bench(
    lambda: wrapper.run(q, kv_cache_at_layer[0]),
    quantiles=[0.5, 0.2, 0.8],
    warmup=1
)
print("FlashInfer MutliLevelCascadeAttn time:", t)


"""

print("Beginning timing")
outputs2 = []

# get start time
#start_time = torch.cuda.Event(enable_timing=True)
#end_time = torch.cuda.Event(enable_timing=True)
#start_time.record()
for i in range(num_layers):
    #print(f"running layer {i}")
    q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    # compute batch decode attention, reuse auxiliary data structures for all layers
    o = wrapper.run(q, kv_cache_at_layer[i])
    outputs2.append(o)

# get end time
#end_time.record()
#end_time.synchronize()
print(outputs2[0].shape)
# print net times
#print(f"Total time: {start_time.elapsed_time(end_time)} ms")

print("All done")
"""
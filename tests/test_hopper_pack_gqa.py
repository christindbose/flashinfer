"""
Correctness tests for pack_gqa (KV-head-major CTA mapping) on the SM90 paged
batch prefill path. Requires an H100 (sm90a).

Run: pytest tests/test_hopper_pack_gqa.py -x -q
"""

import math

import pytest
import torch

import flashinfer


def torch_paged_ref(q, kv_cache, qo_indptr, kv_indptr, kv_last_page_len, page_size,
                    num_qo_heads, num_kv_heads, causal, sm_scale):
    """Reference over the paged cache; returns (out, lse_log2e)."""
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[0], num_qo_heads, dtype=torch.float32, device=q.device)
    group = num_qo_heads // num_kv_heads
    batch = qo_indptr.numel() - 1
    for b in range(batch):
        qs, qe = int(qo_indptr[b]), int(qo_indptr[b + 1])
        ps, pe = int(kv_indptr[b]), int(kv_indptr[b + 1])
        last = int(kv_last_page_len[b])
        k_pages = kv_cache[ps:pe, 0]  # (pages, page_size, H_kv, D)
        v_pages = kv_cache[ps:pe, 1]
        kv_len = (pe - ps - 1) * page_size + last
        k = k_pages.reshape(-1, num_kv_heads, k_pages.shape[-1])[:kv_len]
        v = v_pages.reshape(-1, num_kv_heads, v_pages.shape[-1])[:kv_len]
        qo_len = qe - qs
        for h in range(num_qo_heads):
            hk = h // group
            s = (q[qs:qe, h].float() @ k[:, hk].float().T) * sm_scale
            if causal:
                qi = torch.arange(qo_len, device=q.device)[:, None]
                ki = torch.arange(kv_len, device=q.device)[None, :]
                s.masked_fill_(ki > qi + (kv_len - qo_len), float("-inf"))
            m = s.max(dim=-1, keepdim=True).values
            p = torch.exp(s - m)
            d = p.sum(dim=-1, keepdim=True)
            out[qs:qe, h] = (p / d @ v[:, hk].float()).to(q.dtype)
            lse[qs:qe, h] = (m.squeeze(-1) + torch.log(d.squeeze(-1))) / math.log(2.0)
    return out, lse


def run_case(qo_lens, kv_lens, num_qo_heads, num_kv_heads, causal, page_size,
             head_dim=128, dtype=torch.float16, seed=0):
    torch.manual_seed(seed)
    dev = "cuda:0"
    batch = len(qo_lens)
    qo_indptr = torch.tensor([0] + list(torch.tensor(qo_lens).cumsum(0)),
                             dtype=torch.int32, device=dev)
    pages_per = [(kv + page_size - 1) // page_size for kv in kv_lens]
    kv_indptr = torch.tensor([0] + list(torch.tensor(pages_per).cumsum(0)),
                             dtype=torch.int32, device=dev)
    total_pages = int(kv_indptr[-1])
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=dev)
    kv_last_page_len = torch.tensor(
        [kv - (p - 1) * page_size for kv, p in zip(kv_lens, pages_per)],
        dtype=torch.int32, device=dev)
    q = torch.randn(sum(qo_lens), num_qo_heads, head_dim, dtype=dtype, device=dev)
    kv_cache = torch.randn(total_pages, 2, page_size, num_kv_heads, head_dim,
                           dtype=dtype, device=dev)
    sm_scale = 1.0 / math.sqrt(head_dim)

    outs = {}
    for pack in (False, True):
        wb = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(wb, "NHD", backend="fa3")
        w.plan(qo_indptr, kv_indptr, kv_indices, kv_last_page_len, num_qo_heads,
               num_kv_heads, head_dim, page_size, causal=causal,
               q_data_type=dtype, sm_scale=sm_scale, pack_gqa=pack)
        outs[pack] = w.run(q, kv_cache, return_lse=True)
    torch.cuda.synchronize()
    (o_off, lse_off), (o_on, lse_on) = outs[False], outs[True]

    ref_o, ref_lse = torch_paged_ref(q, kv_cache, qo_indptr, kv_indptr,
                                     kv_last_page_len, page_size, num_qo_heads,
                                     num_kv_heads, causal, sm_scale)
    # Both arms are held to the torch reference (the multi-tile causal
    # masking fix in mainloop_mma.cuh makes the baseline exact as well),
    # and pack on must match pack off.
    torch.testing.assert_close(o_on, ref_o, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse_on, ref_lse, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(o_off, ref_o, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(o_on, o_off, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse_on, lse_off, atol=2e-2, rtol=2e-2)


HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (9, 0)


@pytest.mark.skipif(not HOPPER, reason="requires sm90")
@pytest.mark.parametrize("heads", [(8, 8), (16, 8), (32, 8)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "qo_lens,kv_lens",
    [
        ([128], [2048]),
        ([37], [300]),          # ragged, tile-partial rows
        ([200, 75], [1024, 517]),
        ([1, 1, 1, 1], [4096, 100, 2048, 16]),  # decode-like
    ],
)
def test_pack_gqa_paged(heads, causal, qo_lens, kv_lens):
    run_case(qo_lens, kv_lens, heads[0], heads[1], causal, page_size=16)


@pytest.mark.skipif(not HOPPER, reason="requires sm90")
def test_pack_gqa_page_size_1():
    run_case([64, 33], [777, 64], 16, 8, False, page_size=1)


@pytest.mark.skipif(not HOPPER, reason="requires sm90")
def test_pack_gqa_kvsplit_compose():
    # mech1 kvsplit (kv_len >= 4096 triggers replication) combined with pack
    torch.manual_seed(3)
    dev = "cuda:0"
    num_qo, num_kv, hd, ps = 32, 8, 128, 16
    kv_len, qo_len = 8192, 64
    pages = kv_len // ps
    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=dev)
    kv_indptr = torch.tensor([0, pages], dtype=torch.int32, device=dev)
    kv_indices = torch.arange(pages, dtype=torch.int32, device=dev)
    last = torch.tensor([ps], dtype=torch.int32, device=dev)
    q = torch.randn(qo_len, num_qo, hd, dtype=torch.float16, device=dev)
    kv = torch.randn(pages, 2, ps, num_kv, hd, dtype=torch.float16, device=dev)
    sm_scale = 1.0 / math.sqrt(hd)
    outs = {}
    for pack in (False, True):
        wb = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(wb, "NHD", backend="fa3")
        w.plan(qo_indptr, kv_indptr, kv_indices, last, num_qo, num_kv, hd, ps,
               causal=False, q_data_type=torch.float16, sm_scale=sm_scale,
               use_tree_walk_scheduling=True, kvsplit_mode=True, pack_gqa=pack)
        outs[pack] = w.run(q, kv)
    torch.cuda.synchronize()
    torch.testing.assert_close(outs[True], outs[False], atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-x", "-q"]))

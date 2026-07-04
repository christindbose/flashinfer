"""
Tests for SM100 mech1 (cluster/DSM split-KV) on the cutlass fwd-prefill path.

Requires a Blackwell GPU (sm100a). Run stages in order on first bring-up:

  Stage A (cluster launch, no behavior change):
      pytest tests/test_blackwell_fmha.py -x -q            # cluster 1
      FLASHINFER_FMHA_FORCE_CLUSTER=2 pytest tests/test_blackwell_fmha.py -x -q
  Stage B..D:
      pytest tests/test_blackwell_fmha_kvsplit.py -x -q

Copyright (c) 2026 by FlashInfer team.
Licensed under the Apache License, Version 2.0.
"""

import math

import pytest
import torch

from flashinfer.prefill import fmha_varlen
from flashinfer.utils import is_sm100a_supported


def attention_varlen_ref(q, k, v, qo_offsets, kv_offsets, sm_scale):
    """Per-segment torch reference; returns (out, lse_log2)."""
    out = torch.empty_like(q, dtype=q.dtype)
    lse = torch.empty(q.shape[0], q.shape[1], dtype=torch.float32, device=q.device)
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    group = num_qo_heads // num_kv_heads
    for b in range(qo_offsets.numel() - 1):
        qs, qe = int(qo_offsets[b]), int(qo_offsets[b + 1])
        ks, ke = int(kv_offsets[b]), int(kv_offsets[b + 1])
        for h in range(num_qo_heads):
            hk = h // group
            qi = q[qs:qe, h].float()
            ki = k[ks:ke, hk].float()
            vi = v[ks:ke, hk].float()
            s = (qi @ ki.T) * sm_scale
            m = s.max(dim=-1, keepdim=True).values
            p = torch.exp(s - m)
            denom = p.sum(dim=-1, keepdim=True)
            out[qs:qe, h] = (p / denom @ vi).to(q.dtype)
            # kernel reports log2-space LSE
            lse[qs:qe, h] = (m.squeeze(-1) + torch.log(denom.squeeze(-1))) / math.log(
                2.0
            )
    return out, lse


def make_inputs(qo_lens, kv_lens, num_qo_heads, num_kv_heads, head_dim, dtype):
    device = "cuda:0"
    qo_offsets = torch.tensor(
        [0] + list(torch.tensor(qo_lens).cumsum(0)), dtype=torch.int32, device=device
    )
    kv_offsets = torch.tensor(
        [0] + list(torch.tensor(kv_lens).cumsum(0)), dtype=torch.int32, device=device
    )
    q = torch.randn(sum(qo_lens), num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(sum(kv_lens), num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(sum(kv_lens), num_kv_heads, head_dim, dtype=dtype, device=device)
    return q, k, v, qo_offsets, kv_offsets


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda:0")),
    reason="requires sm100a",
)
@pytest.mark.parametrize(
    "qo_lens,kv_lens",
    [
        ([256], [4096]),
        ([512], [8192]),
        ([1024], [12345]),
        ([256, 512], [16384, 4096]),
        ([256, 512, 1024, 256], [8192, 12345, 4096, 16384]),
        # mixed: items below the 4096 threshold take the vanilla path per item
        ([256, 256], [8192, 300]),
    ],
)
@pytest.mark.parametrize("heads", [(32, 32), (32, 8)])
@pytest.mark.parametrize("head_dim", [128])
def test_kvsplit_vs_baseline(qo_lens, kv_lens, heads, head_dim):
    torch.manual_seed(42)
    num_qo_heads, num_kv_heads = heads
    q, k, v, qo_offsets, kv_offsets = make_inputs(
        qo_lens, kv_lens, num_qo_heads, num_kv_heads, head_dim, torch.bfloat16
    )
    sm_scale = 1.0 / math.sqrt(head_dim)

    out_ref, lse_ref = fmha_varlen(
        q, k, v, qo_offsets, kv_offsets, causal=False, sm_scale=sm_scale,
        return_lse=True,
    )
    out_ks, lse_ks = fmha_varlen(
        q, k, v, qo_offsets, kv_offsets, causal=False, sm_scale=sm_scale,
        return_lse=True, kvsplit_mode=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out_ks, out_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse_ks, lse_ref, atol=2e-2, rtol=2e-2)

    out_t, lse_t = attention_varlen_ref(q, k, v, qo_offsets, kv_offsets, sm_scale)
    torch.testing.assert_close(out_ks, out_t, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse_ks, lse_t, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda:0")),
    reason="requires sm100a",
)
def test_kvsplit_rejects_causal():
    q, k, v, qo_offsets, kv_offsets = make_inputs(
        [256], [8192], 8, 8, 128, torch.bfloat16
    )
    with pytest.raises(ValueError):
        fmha_varlen(
            q, k, v, qo_offsets, kv_offsets, causal=True, kvsplit_mode=True
        )

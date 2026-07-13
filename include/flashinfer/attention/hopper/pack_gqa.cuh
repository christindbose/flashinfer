/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * PackGQA support for the SM90 batch prefill kernel: with a KV-head-major
 * grid, the h_r = num_qo_heads / num_kv_heads QO heads of one GQA group are
 * packed into the Q-tile row dimension (packed row idx: head-in-group =
 * idx % h_r, token = idx / h_r), so K/V is loaded once per group.
 *
 * The packed Q gmem view cannot be expressed as a 2-D TMA box for arbitrary
 * h_r, so Q is loaded with cp.async by the producer warp. Ported from FA3's
 * PackGQAManager (flash-attention/hopper/pack_gqa.h).
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_PACK_GQA_CUH_
#define FLASHINFER_ATTENTION_HOPPER_PACK_GQA_CUH_

#include <cute/tensor.hpp>
#include <cutlass/fast_math.h>

namespace flashinfer {

using namespace cute;

// Decode a packed Q-tile row index into (token, head-in-group).
struct PackedQoCoord {
  int token;
  int head_offset;  // qo head = kv_head_idx * h_r + head_offset
};

CUTLASS_DEVICE PackedQoCoord unpack_qo_row(int packed_idx,
                                           cutlass::FastDivmod const& group_size_fastdiv) {
  int head_offset;
  int token = group_size_fastdiv.divmod(head_offset, packed_idx);
  return {token, head_offset};
}

template <int CTA_Q, int HEAD_DIM, int NumThreads, typename Element>
struct PackGQAManager {
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(HEAD_DIM % kGmemElemsPerLoad == 0,
                "Headdim must be a multiple of kGmemElemsPerLoad");
  // Each "row pass" covers one cache line worth of columns so the number of
  // divmod/pointer computations is minimized (FA3 comment).
  static constexpr int kBytePerRow = HEAD_DIM * sizeof(Element);
  static constexpr int kBlockKGmem =
      (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
  static_assert(NumThreads % kGmemThreadsPerRow == 0,
                "NumThreads must be a multiple of kGmemThreadsPerRow");
  static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0,
                "kGmemThreadsPerRow must divide NumThreadsPerWarp");

  using GmemCopyAtomCpAsync = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>;
  using GmemLayoutAtom = Layout<Shape<Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopyQCpAsync =
      decltype(make_tiled_copy(GmemCopyAtomCpAsync{}, GmemLayoutAtom{},
                               Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));

  // Per-row gmem pointers are expensive under packing (divmod each); split the
  // work across the threads of a row and share via __shfl_sync (FA3 pattern).
  template <int NumThreadsPerRow = kGmemThreadsPerRow, typename Engine, typename Layout,
            typename TensorC>
  CUTLASS_DEVICE static auto compute_ptr(Tensor<Engine, Layout>& tensor, TensorC const& tRows,
                                         cutlass::FastDivmod const& group_size_fastdiv,
                                         int const thread_idx, int const q_tile_idx) {
    static constexpr int NumPtrPerThread =
        cute::ceil_div(CUTE_STATIC_V(cute::size(tRows)), NumThreadsPerRow);
    using TensorType = typename Engine::value_type;
    Tensor tPrPtr = make_tensor<TensorType const*>(Shape<Int<NumPtrPerThread>>{});
#pragma unroll
    for (int i = 0; i < NumPtrPerThread; ++i) {
      int const row = i * NumThreads + get<0>(tRows(thread_idx % NumThreadsPerRow));
      int const idx = q_tile_idx * CTA_Q + row;
      int m_idx, h_idx;
      m_idx = group_size_fastdiv.divmod(h_idx, idx);
      tPrPtr[i] = &tensor(make_coord(make_coord(h_idx, m_idx)));
    }
    return tPrPtr;
  }

  // mQ: packed gmem view ((h_r, qo_len), HEAD_DIM); sQ: (CTA_Q, HEAD_DIM)
  // position-independent-swizzle smem tensor. All NumThreads threads must call.
  template <typename TensormQ, typename TensorsQ>
  CUTLASS_DEVICE static void load_Q(TensormQ const& mQ, TensorsQ&& sQ,
                                    cutlass::FastDivmod const& group_size_fastdiv,
                                    int const thread_idx, int const qo_len,
                                    int const q_tile_idx) {
    GmemTiledCopyQCpAsync gmem_tiled_copy_Q_cp_async;
    auto gmem_thr_copy_Q_cp_async = gmem_tiled_copy_Q_cp_async.get_thread_slice(thread_idx);
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor tQcQ = gmem_thr_copy_Q_cp_async.partition_S(cQ);
    Tensor tQsQ = gmem_thr_copy_Q_cp_async.partition_D(sQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
#pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < size<1>(mQ);
    }

    Tensor mQ_0 = mQ(_, _0{});
    Tensor tQcQ_row = tQcQ(_0{}, _, _0{});
    Tensor tPrQPtr = compute_ptr(mQ_0, tQcQ_row, group_size_fastdiv, thread_idx, q_tile_idx);
    int const group_size = group_size_fastdiv.divisor;
#pragma unroll
    for (int m = 0; m < size<1>(tQsQ); ++m) {
      int idx = q_tile_idx * CTA_Q + get<0>(tQcQ(_0{}, m, _0{}));
      Element const* q_ptr = reinterpret_cast<Element const*>(
          __shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrQPtr(m / kGmemThreadsPerRow)),
                      m % kGmemThreadsPerRow, kGmemThreadsPerRow));
      if (idx < qo_len * group_size) {
        Tensor mQ_cur = make_tensor(make_gmem_ptr(q_ptr), Shape<Int<HEAD_DIM>>{});
        Tensor mQ_cur_copy = cute::tiled_divide(mQ_cur, Shape<Int<kGmemElemsPerLoad>>{});
#pragma unroll
        for (int k = 0; k < size<2>(tQsQ); ++k) {
          int ki = get<1>(tQcQ(_0{}, _0{}, k)) / kGmemElemsPerLoad;
          cute::copy(gmem_tiled_copy_Q_cp_async.with(tQpQ(k)), mQ_cur_copy(_, ki), tQsQ(_, m, k));
        }
      }
      // No zero-fill for OOB rows: they are masked in softmax and never stored.
    }
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_PACK_GQA_CUH_

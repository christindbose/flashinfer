/*
 * Copyright (c) 2023 by FlashInfer team.
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
 */
#ifndef FLASHINFER_ATTENTION_SCHEDULER_CUH_
#define FLASHINFER_ATTENTION_SCHEDULER_CUH_

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#include "../allocator.h"
#include "../exception.h"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "heap.h"

namespace flashinfer {

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params);

template <uint32_t num_stages_smem, uint32_t vec_size_ckv, uint32_t vec_size_kpe, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, uint32_t tile_size_qo_heads, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernelMLA(Params params);

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, uint32_t QO_TILE_LEN, typename DTypeKV>
std::tuple<uint32_t, uint32_t, uint32_t> LaunchSpecForDecodeKernelMlaCuteSM80(
    const uint32_t num_qo_heads);

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, uint32_t QO_TILE_LEN, typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernelMlaCuteSM80(Params params);

template <typename DType>
inline void CopyToPageLockedBuffer(void* page_locked_int_buffer, int64_t offset,
                                   const std::vector<DType>& vec) {
  DType* ptr = GetPtrFromBaseOffset<DType>(page_locked_int_buffer, offset);
  std::copy(vec.begin(), vec.end(), ptr);
}

/*!
 * \brief Compute the maximum number of pages per batch and the new batch size
 *   after we partition Paged KV-Cache into multiple chunks on KV sequence length
 *   dimension.
 * \tparam IdType A template type indicates the index data type
 * \param max_grid_size The maximum grid size of the kernel
 * \param gdy gridDim.y
 * \param num_pages The number of pages per request in the batch
 * \param max_num_pages_per_batch_lb The pre-set lower bound of maximum number of
 *   pages per batch, default to 1
 * \return (max_num_pages_per_batch, new_batch_size) The number of pages per batch and
 *   the new batch size after the partition.
 */
template <typename IdType>
inline auto PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(
    const uint32_t max_grid_size, const uint32_t gdy, const std::vector<IdType>& num_pages,
    const uint32_t min_num_pages_per_batch = 1) {
  uint32_t low = min_num_pages_per_batch, high = 0;
  for (const IdType& elem : num_pages) {
    high = max(high, elem);
  }
  uint32_t new_batch_size;
  while (low < high) {
    uint32_t mid = (low + high) / 2;
    new_batch_size = 0;
    for (const IdType& elem : num_pages) {
      new_batch_size += ceil_div(elem, mid);
    }
    if (new_batch_size * gdy > max_grid_size) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  new_batch_size = 0;
  for (const IdType& elem : num_pages) {
    new_batch_size += ceil_div(std::max(elem, 1), low);
  }
  return std::make_tuple(low, new_batch_size);
}

inline auto PrefillBinarySearchKVChunkSize(const bool enable_cuda_graph,
                                           const uint32_t max_batch_size_if_split,
                                           const std::vector<int64_t>& packed_qo_len_arr,
                                           const std::vector<int64_t>& kv_len_arr,
                                           const uint32_t qo_chunk_size,
                                           const uint32_t min_kv_chunk_size = 1) {
  const int64_t batch_size = packed_qo_len_arr.size();
  int64_t max_kv_len = 1;
  for (const int64_t& kv_len : kv_len_arr) {
    max_kv_len = std::max(max_kv_len, kv_len);
  }

  int64_t low = min_kv_chunk_size;
  int64_t high = max_kv_len;
  constexpr int64_t min_kv_len = 1;
  while (low < high) {
    const int64_t mid = (low + high) / 2;
    int64_t new_batch_size = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      new_batch_size += ceil_div(packed_qo_len_arr[i], qo_chunk_size) *
                        ceil_div(std::max(kv_len_arr[i], min_kv_len), mid);
    }
    if (new_batch_size > max_batch_size_if_split) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return std::make_tuple(enable_cuda_graph || low < max_kv_len, low);
}

/*!
 * \brief Estimate the temporary buffer size and the maximum grid size for the
 *   partition-kv BatchDecodeWithPagedKVCache kernel
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param split_kv Whether to split the KV cache into multiple chunks
 * \param max_grid_size The maximum grid size that can be used in a partiton-kv kernel
 * \param max_num_pages_per_batch The maximum number of pages per batch
 * \param new_batch_size The new batch size after the partition
 * \param paged_kv The paged kv cache data structure
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param pos_encoding_mode The positional encoding mode
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          typename AttentionVariant, typename Params>
inline cudaError_t BatchDecodeWithPagedKVCacheWorkEstimationDispatched(
    bool& split_kv, uint32_t& max_grid_size, uint32_t& max_num_pages_per_batch,
    uint32_t& new_batch_size, uint32_t& gdy, uint32_t batch_size,
    typename Params::IdType* kv_indptr_h, const uint32_t num_qo_heads, const uint32_t page_size,
    bool enable_cuda_graph, cudaStream_t stream) {
  using DTypeKV = typename Params::DTypeKV;
  using IdType = typename Params::IdType;
  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    static_assert(bdx <= 32);
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
    const uint32_t num_kv_heads = num_qo_heads / GROUP_SIZE;
    gdy = num_kv_heads;
    const uint32_t smem_size =
        2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
        std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*), 2 * bdy * bdz * sizeof(float));

    auto kernel =
        BatchDecodeWithPagedKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                          vec_size, bdx, bdy, bdz, AttentionVariant, Params>;
    int num_blocks_per_sm = 0;
    int num_sm = 0;
    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                       num_threads, smem_size));
    max_grid_size = num_blocks_per_sm * num_sm;
    if (batch_size * gdy >= max_grid_size) {
      split_kv = false;
      max_num_pages_per_batch = 1;
      for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        max_num_pages_per_batch = std::max<uint32_t>(
            max_num_pages_per_batch, kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx]);
      }
      new_batch_size = batch_size;
    } else {
      // compute max_num_pages_per_batch and new_batch_size
      std::vector<IdType> num_pages(batch_size);
      for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        num_pages[batch_idx] = kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx];
      }
      std::tie(max_num_pages_per_batch, new_batch_size) =
          PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(max_grid_size, gdy, num_pages,
                                                              std::max(128 / page_size, 1U));
      if (new_batch_size == batch_size && !enable_cuda_graph) {
        // do not use partition-kv kernel for short sequence, when not using CUDAGraph
        split_kv = false;
      } else {
        // when using CUDAGraph, we always use partition-kv kernel
        split_kv = true;
      }
    }
    return cudaSuccess;
  })
}

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename AttentionVariant, typename Params>
inline cudaError_t BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMLA(
    bool& split_kv, uint32_t& max_grid_size, uint32_t& max_num_pages_per_batch,
    uint32_t& new_batch_size, uint32_t& gdy, uint32_t batch_size,
    typename Params::IdType* kv_indptr_h, const uint32_t num_qo_heads, const uint32_t page_size,
    bool enable_cuda_graph, cudaStream_t stream) {
  using DTypeKV = typename Params::DTypeKV;
  using IdType = typename Params::IdType;

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
    constexpr uint32_t vec_size_ckv = std::max(16UL / sizeof(DTypeKV), HEAD_DIM_CKV / 32UL);
    constexpr uint32_t bdx = HEAD_DIM_CKV / vec_size_ckv;
    constexpr uint32_t vec_size_kpe = HEAD_DIM_KPE / bdx;

    constexpr uint32_t bdy = 8;
    constexpr uint32_t tile_size_qo_heads = 2;
    constexpr uint32_t qo_heads_per_block = bdy * tile_size_qo_heads;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    gdy = ceil_div(num_qo_heads, qo_heads_per_block);

    const uint32_t smem_size =
        NUM_STAGES_SMEM * bdy * bdz * (HEAD_DIM_CKV + HEAD_DIM_KPE) * sizeof(DTypeKV) +
        std::max(num_threads * sizeof(size_t) * 2, 2 * bdy * bdz * sizeof(float));

    auto kernel =
        BatchDecodeWithPagedKVCacheKernelMLA<NUM_STAGES_SMEM, vec_size_ckv, vec_size_kpe, bdx, bdy,
                                             bdz, tile_size_qo_heads, AttentionVariant, Params>;
    int num_blocks_per_sm = 0;
    int num_sm = 0;
    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                       num_threads, smem_size));
    max_grid_size = num_blocks_per_sm * num_sm;
    if (batch_size * gdy >= max_grid_size) {
      split_kv = false;
      max_num_pages_per_batch = 1;
      for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        max_num_pages_per_batch = std::max<uint32_t>(
            max_num_pages_per_batch, kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx]);
      }
      new_batch_size = batch_size;
    } else {
      // compute max_num_pages_per_batch and new_batch_size
      std::vector<IdType> num_pages(batch_size);
      for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        num_pages[batch_idx] = kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx];
      }
      std::tie(max_num_pages_per_batch, new_batch_size) =
          PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(max_grid_size, gdy, num_pages,
                                                              std::max(128 / page_size, 1U));
      if (new_batch_size == batch_size && !enable_cuda_graph) {
        // do not use partition-kv kernel for short sequence, when not using CUDAGraph
        split_kv = false;
      } else {
        // when using CUDAGraph, we always use partition-kv kernel
        split_kv = true;
      }
    }

    return cudaSuccess;
  });
}

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, uint32_t QO_TILE_LEN,
          typename AttentionVariant, typename Params>
inline cudaError_t BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMlaCuteSM80(
    bool& split_kv, uint32_t& max_grid_size, uint32_t& max_num_pages_per_batch,
    uint32_t& new_batch_size, uint32_t& gdy_, uint32_t batch_size,
    typename Params::IdType* kv_indptr_h, const uint32_t num_qo_heads, const uint32_t page_size,
    bool enable_cuda_graph, cudaStream_t stream) {
  using DTypeKV = typename Params::DTypeKV;
  using IdType = typename Params::IdType;

  auto [smem_size, gdy, k_warps] =
      LaunchSpecForDecodeKernelMlaCuteSM80<HEAD_DIM_CKV, HEAD_DIM_KPE, QO_TILE_LEN, DTypeKV>(
          num_qo_heads);
  gdy_ = gdy;
  const uint32_t num_threads = k_warps * 32;
  auto kernel =
      BatchDecodeWithPagedKVCacheKernelMlaCuteSM80<HEAD_DIM_CKV, HEAD_DIM_KPE, QO_TILE_LEN, Params>;
  int num_blocks_per_sm;
  int num_sm = 0;
  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));

  // FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
  //                                   num_threads, smem_size));
  // fixme: num_blocks_per_sm is 0 derived from cudaOccupancyMaxActiveBlocksPerMultiprocessor at
  // times, and we fill smem with q-heads as many as possible, so num_blocks_per_sm should be 1
  num_blocks_per_sm = 1;

  max_grid_size = num_blocks_per_sm * num_sm;
  if (batch_size * gdy >= max_grid_size) {
    split_kv = false;
    max_num_pages_per_batch = 1;
    for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      max_num_pages_per_batch = std::max<uint32_t>(
          max_num_pages_per_batch, kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx]);
    }
    new_batch_size = batch_size;
  } else {
    // compute max_num_pages_per_batch and new_batch_size
    std::vector<IdType> num_pages(batch_size);
    for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      num_pages[batch_idx] = kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx];
    }
    std::tie(max_num_pages_per_batch, new_batch_size) =
        PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(max_grid_size, gdy, num_pages,
                                                            std::max(128 / page_size, 1U));
    if (new_batch_size == batch_size && !enable_cuda_graph) {
      // do not use partition-kv kernel for short sequence, when not using CUDAGraph
      split_kv = false;
    } else {
      // when using CUDAGraph, we always use partition-kv kernel
      split_kv = true;
    }
  }

  return cudaSuccess;
}

/*!
 * \brief Partition Paged KV-Cache into multiple chunks on KV sequence length
 * \tparam IdType A template type indicates the index data type
 * \param old_batch_size The batch size of the old Paged KV-Cache
 * \param old_page_indptr_h The host-side page indptr of the old Paged KV-Cache
 * \param max_num_pages_per_batch The maximum number of pages per batch
 * \param new_paged_kv_d The device-side new Paged KV-Cache
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <typename IdType>
inline auto DecodeSplitKVIndptr(IdType* indptr_h, uint32_t batch_size, uint32_t kv_chunk_size) {
  std::vector<IdType> request_indices, kv_tile_indices, o_indptr;
  o_indptr.push_back(0);

  for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    uint32_t num_tiles_kv = ceil_div(
        std::max<uint32_t>(indptr_h[batch_idx + 1] - indptr_h[batch_idx], 1U), kv_chunk_size);
    for (uint32_t kv_tile_idx = 0; kv_tile_idx < num_tiles_kv; ++kv_tile_idx) {
      request_indices.push_back(batch_idx);
      kv_tile_indices.push_back(kv_tile_idx);
    }
    o_indptr.push_back(o_indptr.back() + num_tiles_kv);
  }

  return std::make_tuple(request_indices, kv_tile_indices, o_indptr);
}

struct DecodePlanInfo {
  int64_t padded_batch_size;
  int64_t v_offset;
  int64_t s_offset;
  int64_t request_indices_offset;
  int64_t kv_tile_indices_offset;
  int64_t o_indptr_offset;
  int64_t block_valid_mask_offset;
  int64_t kv_chunk_size_ptr_offset;
  bool enable_cuda_graph;
  bool split_kv;

  DecodePlanInfo()
      : padded_batch_size(0),
        v_offset(0),
        s_offset(0),
        request_indices_offset(0),
        kv_tile_indices_offset(0),
        o_indptr_offset(0),
        block_valid_mask_offset(0),
        kv_chunk_size_ptr_offset(0),
        enable_cuda_graph(false),
        split_kv(false) {}

  // convert DecodePlanInfo to std::vector<int64_t>
  std::vector<int64_t> ToVector() const {
    return {padded_batch_size,
            v_offset,
            s_offset,
            request_indices_offset,
            kv_tile_indices_offset,
            o_indptr_offset,
            block_valid_mask_offset,
            kv_chunk_size_ptr_offset,
            enable_cuda_graph,
            split_kv};
  }

  // From std::vector<int64_t> to DecodePlanInfo
  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 10) {
      std::ostringstream err_msg;
      err_msg << "DecodePlanInfo::FromVector: vec.size() should be 10, but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    padded_batch_size = vec[0];
    v_offset = vec[1];
    s_offset = vec[2];
    request_indices_offset = vec[3];
    kv_tile_indices_offset = vec[4];
    o_indptr_offset = vec[5];
    block_valid_mask_offset = vec[6];
    kv_chunk_size_ptr_offset = vec[7];
    enable_cuda_graph = vec[8];
    split_kv = vec[9];
  }
};

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params, typename WorkEstimationFunc>
inline cudaError_t DecodePlan(void* float_buffer, size_t float_workspace_size_in_bytes,
                              void* int_buffer, void* page_locked_int_buffer,
                              size_t int_workspace_size_in_bytes, DecodePlanInfo& plan_info,
                              typename Params::IdType* indptr_h, uint32_t batch_size,
                              uint32_t num_qo_heads, uint32_t page_size, bool enable_cuda_graph,
                              cudaStream_t stream, WorkEstimationFunc work_estimation_func) {
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  bool split_kv;
  uint32_t max_grid_size, kv_chunk_size_in_pages, new_batch_size, gdy;

  FLASHINFER_CUDA_CALL(work_estimation_func(split_kv, max_grid_size, kv_chunk_size_in_pages,
                                            new_batch_size, gdy, batch_size, indptr_h, num_qo_heads,
                                            page_size, enable_cuda_graph, stream));
  size_t padded_batch_size;
  plan_info.enable_cuda_graph = enable_cuda_graph;
  plan_info.split_kv = split_kv;
  padded_batch_size =
      (enable_cuda_graph) ? (split_kv ? max_grid_size / gdy : batch_size) : new_batch_size;
  plan_info.padded_batch_size = padded_batch_size;

  auto [request_indices_vec, kv_tile_indices_vec, o_indptr_vec] =
      DecodeSplitKVIndptr(indptr_h, batch_size, kv_chunk_size_in_pages);

  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
  plan_info.request_indices_offset = int_allocator.aligned_alloc_offset(
      padded_batch_size * sizeof(IdType), 16, "batch_decode_request_indices");
  plan_info.kv_tile_indices_offset = int_allocator.aligned_alloc_offset(
      padded_batch_size * sizeof(IdType), 16, "batch_decode_kv_tile_indices");
  plan_info.o_indptr_offset = int_allocator.aligned_alloc_offset(
      (padded_batch_size + 1) * sizeof(IdType), 16, "batch_decode_o_indptr");
  plan_info.kv_chunk_size_ptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType), 1, "batch_decode_kv_chunk_size_ptr");
  IdType* request_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.request_indices_offset);
  IdType* kv_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_tile_indices_offset);
  IdType* o_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.o_indptr_offset);
  IdType* kv_chunk_size_ptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_chunk_size_ptr_offset);
  std::copy(request_indices_vec.begin(), request_indices_vec.end(), request_indices_h);
  std::copy(kv_tile_indices_vec.begin(), kv_tile_indices_vec.end(), kv_tile_indices_h);
  std::copy(o_indptr_vec.begin(), o_indptr_vec.end(), o_indptr_h);
  kv_chunk_size_ptr_h[0] = kv_chunk_size_in_pages * page_size;

  if (split_kv) {
    AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
    plan_info.v_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * HEAD_DIM * sizeof(float), 16, "batch_decode_tmp_v");
    plan_info.s_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * sizeof(float), 16, "batch_decode_tmp_s");

    plan_info.block_valid_mask_offset = int_allocator.aligned_alloc_offset(
        padded_batch_size * sizeof(bool), 16, "batch_decode_block_valid_mask");
    bool* block_valid_mask_h =
        GetPtrFromBaseOffset<bool>(page_locked_int_buffer, plan_info.block_valid_mask_offset);
    for (uint32_t i = 0; i < padded_batch_size; ++i) {
      block_valid_mask_h[i] = i < new_batch_size;
    }
  }

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();

  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));
  return cudaSuccess;
}

template <typename IdType>
inline auto PrefillSplitQOKVIndptr(IdType* qo_indptr_h, IdType* kv_indptr_h,
                                   uint32_t total_num_rows, uint32_t batch_size,
                                   uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
                                   uint32_t page_size, uint32_t max_batch_size_if_split,
                                   bool enable_cuda_graph) {
  std::vector<IdType> request_indices, qo_tile_indices, kv_tile_indices, merge_indptr, o_indptr;
  merge_indptr.push_back(0);
  o_indptr.push_back(0);

  const uint32_t gqa_group_size = num_qo_heads / num_kv_heads;

  // step 1: determine packed_qo_len_arr and verify qo_indptr contents.
  std::vector<int64_t> packed_qo_len_arr(batch_size), kv_len_arr(batch_size);
  for (uint32_t i = 0; i < batch_size; ++i) {
    packed_qo_len_arr[i] = int64_t(qo_indptr_h[i + 1] - qo_indptr_h[i]) * int64_t(gqa_group_size);
    if (packed_qo_len_arr[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "qo_indptr[" << i + 1 << "]" << qo_indptr_h[i + 1] << " - qo_indptr[" << i << "]"
              << qo_indptr_h[i] << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }
    kv_len_arr[i] = int64_t(kv_indptr_h[i + 1] - kv_indptr_h[i]);
    if (kv_len_arr[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "kv_indptr[" << i + 1 << "]" << kv_indptr_h[i + 1] << " - kv_indptr[" << i << "]"
              << kv_indptr_h[i] << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }
  }

  // step 2: determine cta_tile_q, kv_chunk_size and total_num_tiles_q
  const uint32_t min_kv_chunk_size = std::max((128 / page_size), 1U);
  uint32_t cta_tile_q;
  uint32_t total_num_tiles_q;
  if (enable_cuda_graph) {
    // When CUDA graphs are enabled, the lengths of sequences determined by
    // qo_indptr_h can vary. We assume that the dummy data based on which
    // the CUDA graph is created fixes the maximum number of tokens.
    const uint64_t max_seq_len = total_num_rows - batch_size + 1;
    uint64_t max_qo_len = uint64_t(max_seq_len) * gqa_group_size;
    cta_tile_q = FA2DetermineCtaTileQ(max_qo_len, head_dim);

    // Find an upper bound for the number of tiles, derived from the total
    // number of rows and the batch size.  The sum of qo lengths rounded
    // up to cta_tile_q will not exceed this number derived from the total
    // number of rows.
    total_num_tiles_q = ceil_div(total_num_rows * gqa_group_size, cta_tile_q) + batch_size - 1;
  } else {
    int64_t sum_packed_qo_len = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      sum_packed_qo_len += packed_qo_len_arr[i];
    }
    const int64_t avg_packed_qo_len = sum_packed_qo_len / batch_size;
    cta_tile_q = FA2DetermineCtaTileQ(avg_packed_qo_len, head_dim);

    total_num_tiles_q = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      total_num_tiles_q += ceil_div(packed_qo_len_arr[i], cta_tile_q);
    }
  }

  auto [split_kv, kv_chunk_size] =
      PrefillBinarySearchKVChunkSize(enable_cuda_graph, max_batch_size_if_split, packed_qo_len_arr,
                                     kv_len_arr, cta_tile_q, min_kv_chunk_size);

  // step 3: split qo_indptr and kv_indptr
  uint32_t new_batch_size = 0;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    const int64_t packed_qo_len = packed_qo_len_arr[request_idx];
    const int64_t kv_len = std::max(int(kv_len_arr[request_idx]), 1);
    const int64_t num_tiles_q = ceil_div(packed_qo_len, cta_tile_q);
    const int64_t num_tiles_kv = ceil_div(kv_len, kv_chunk_size);

    for (uint32_t q_tile_idx = 0; q_tile_idx < num_tiles_q; ++q_tile_idx) {
      for (uint32_t kv_tile_idx = 0; kv_tile_idx < num_tiles_kv; ++kv_tile_idx) {
        new_batch_size += 1;
        request_indices.push_back(request_idx);
        qo_tile_indices.push_back(q_tile_idx);
        kv_tile_indices.push_back(kv_tile_idx);
      }
    }

    int64_t qo_len = packed_qo_len / gqa_group_size;
    for (uint32_t row = 0; row < qo_len; ++row) {
      merge_indptr.push_back(merge_indptr.back() + num_tiles_kv);
    }
    o_indptr.push_back(o_indptr.back() + qo_len * num_tiles_kv);
  }

  const size_t padded_batch_size =
      enable_cuda_graph ? std::max(max_batch_size_if_split, total_num_tiles_q) : new_batch_size;
  FLASHINFER_CHECK(new_batch_size <= padded_batch_size,
                   "new batch size should not exceed padded batch size");

  // step 4: multiply kv_chunk_size by page_size
  kv_chunk_size *= page_size;

  return std::make_tuple(split_kv, new_batch_size, padded_batch_size, cta_tile_q, kv_chunk_size,
                         std::move(request_indices), std::move(qo_tile_indices),
                         std::move(kv_tile_indices), std::move(merge_indptr), std::move(o_indptr));
}

struct PrefillPlanInfo {
  int64_t padded_batch_size;
  int64_t total_num_rows;
  int64_t total_num_rows_offset;
  int64_t cta_tile_q;
  int64_t request_indices_offset;
  int64_t qo_tile_indices_offset;
  int64_t kv_tile_indices_offset;
  int64_t merge_indptr_offset;
  int64_t o_indptr_offset;
  int64_t kv_chunk_size_ptr_offset;
  int64_t v_offset;
  int64_t s_offset;
  int64_t block_valid_mask_offset;
  bool enable_cuda_graph;
  bool split_kv;

  PrefillPlanInfo()
      : padded_batch_size(0),
        total_num_rows(0),
        total_num_rows_offset(0),
        cta_tile_q(0),
        request_indices_offset(0),
        qo_tile_indices_offset(0),
        kv_tile_indices_offset(0),
        merge_indptr_offset(0),
        o_indptr_offset(0),
        kv_chunk_size_ptr_offset(0),
        v_offset(0),
        s_offset(0),
        block_valid_mask_offset(0),
        enable_cuda_graph(false),
        split_kv(false) {}

  // convert PrefillPlanInfo to std::vector<int64_t>
  std::vector<int64_t> ToVector() const {
    return {padded_batch_size,
            total_num_rows,
            total_num_rows_offset,
            cta_tile_q,
            request_indices_offset,
            qo_tile_indices_offset,
            kv_tile_indices_offset,
            merge_indptr_offset,
            o_indptr_offset,
            kv_chunk_size_ptr_offset,
            v_offset,
            s_offset,
            block_valid_mask_offset,
            enable_cuda_graph,
            split_kv};
  }

  // From std::vector<int64_t> to PrefillPlanInfo
  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 15) {
      std::ostringstream err_msg;
      err_msg << "PrefillPlanInfo::FromVector: vec.size() should be 15, but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    padded_batch_size = vec[0];
    total_num_rows = vec[1];
    total_num_rows_offset = vec[2];
    cta_tile_q = vec[3];
    request_indices_offset = vec[4];
    qo_tile_indices_offset = vec[5];
    kv_tile_indices_offset = vec[6];
    merge_indptr_offset = vec[7];
    o_indptr_offset = vec[8];
    kv_chunk_size_ptr_offset = vec[9];
    v_offset = vec[10];
    s_offset = vec[11];
    block_valid_mask_offset = vec[12];
    enable_cuda_graph = vec[13];
    split_kv = vec[14];
  }
};

template <typename IdType>
inline cudaError_t PrefillPlan(void* float_buffer, size_t float_workspace_size_in_bytes,
                               void* int_buffer, void* page_locked_int_buffer,
                               size_t int_workspace_size_in_bytes, PrefillPlanInfo& plan_info,
                               IdType* qo_indptr_h, IdType* kv_indptr_h, uint32_t total_num_rows,
                               uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
                               uint32_t head_dim_qk, uint32_t head_dim_vo, uint32_t page_size,
                               bool enable_cuda_graph, uint32_t sizeof_dtype_o,
                               cudaStream_t stream) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
            << num_kv_heads;
    FLASHINFER_ERROR(err_msg.str());
  }

  // step 0: get the number of SMs
  int num_sm = 0;
  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
  int num_blocks_per_sm = 2;
  int max_grid_size = num_blocks_per_sm * num_sm;
  uint32_t max_batch_size_if_split = max_grid_size / num_kv_heads;

  // step 2: determine kv_chunk_size
  auto [split_kv, new_batch_size, padded_batch_size, cta_tile_q, kv_chunk_size, request_indices_vec,
        qo_tile_indices_vec, kv_tile_indices_vec, merge_indptr_vec, o_indptr_vec] =
      PrefillSplitQOKVIndptr(qo_indptr_h, kv_indptr_h, total_num_rows, batch_size, num_qo_heads,
                             num_kv_heads, head_dim_vo, page_size, max_batch_size_if_split,
                             enable_cuda_graph);

  plan_info.cta_tile_q = cta_tile_q;
  plan_info.total_num_rows = total_num_rows;
  plan_info.enable_cuda_graph = enable_cuda_graph;
  plan_info.padded_batch_size = padded_batch_size;
  plan_info.split_kv = split_kv;

  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
  plan_info.request_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * padded_batch_size, 16, "batch_prefill_request_indices");
  plan_info.qo_tile_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * padded_batch_size, 16, "batch_prefill_qo_tile_indices");
  plan_info.kv_tile_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * padded_batch_size, 16, "batch_prefill_kv_tile_indices");
  plan_info.o_indptr_offset = int_allocator.aligned_alloc_offset(sizeof(IdType) * (batch_size + 1),
                                                                 16, "batch_prefill_o_indptr");
  plan_info.kv_chunk_size_ptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType), 1, "batch_prefill_kv_chunk_size_ptr");

  if (plan_info.enable_cuda_graph) {
    plan_info.total_num_rows_offset =
        int_allocator.aligned_alloc_offset(sizeof(uint32_t), 16, "batch_prefill_total_num_rows");
    uint32_t* total_num_rows_h =
        GetPtrFromBaseOffset<uint32_t>(page_locked_int_buffer, plan_info.total_num_rows_offset);
    *total_num_rows_h = qo_indptr_h[batch_size];
  }

  IdType* request_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.request_indices_offset);
  IdType* qo_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.qo_tile_indices_offset);
  IdType* kv_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_tile_indices_offset);
  IdType* o_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.o_indptr_offset);
  IdType* kv_chunk_size_ptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_chunk_size_ptr_offset);
  std::copy(request_indices_vec.begin(), request_indices_vec.end(), request_indices_h);
  std::copy(qo_tile_indices_vec.begin(), qo_tile_indices_vec.end(), qo_tile_indices_h);
  std::copy(kv_tile_indices_vec.begin(), kv_tile_indices_vec.end(), kv_tile_indices_h);
  std::copy(o_indptr_vec.begin(), o_indptr_vec.end(), o_indptr_h);
  kv_chunk_size_ptr_h[0] = kv_chunk_size;

  if (split_kv) {
    AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
    plan_info.v_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * cta_tile_q * head_dim_vo * sizeof(float), 16,
        "batch_prefill_tmp_v");
    plan_info.s_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * cta_tile_q * sizeof(float), 16, "batch_prefill_tmp_s");
    plan_info.merge_indptr_offset = int_allocator.aligned_alloc_offset(
        sizeof(IdType) * (plan_info.total_num_rows + 1), 16, "batch_prefill_merge_indptr");
    plan_info.block_valid_mask_offset = int_allocator.aligned_alloc_offset(
        sizeof(bool) * padded_batch_size, 16, "batch_prefill_block_valid_mask");

    IdType* merge_indptr_h =
        GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.merge_indptr_offset);
    bool* block_valid_mask_h =
        GetPtrFromBaseOffset<bool>(page_locked_int_buffer, plan_info.block_valid_mask_offset);
    std::copy(merge_indptr_vec.begin(), merge_indptr_vec.end(), merge_indptr_h);
    for (uint32_t i = 0; i < padded_batch_size; ++i) {
      block_valid_mask_h[i] = i < new_batch_size;
    }
  }

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));

  return cudaSuccess;
}

inline float cost_function(int qo_len, int kv_len) { return 2 * float(qo_len) + kv_len; }

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& vec, int size_after_flatten) {
  std::vector<T> result;
  result.reserve(size_after_flatten);
  for (const auto& inner_vec : vec) {
    result.insert(result.end(), inner_vec.begin(), inner_vec.end());
  }
  return result;
}

inline int packed_causal_kv_end(int qo_len, int kv_len, int qo_tile_idx, int cluster_tile_q,
                                int num_qo_tiles, int group_size) {
  if (qo_tile_idx + 1 == num_qo_tiles) {
    return kv_len;
  }
  int kv_len_init = kv_len - qo_len;  // right aligned
  return max(min(kv_len_init + ceil_div((qo_tile_idx + 1) * cluster_tile_q, group_size), kv_len),
             0);
}

struct PrefillPlanSM90Info {
  int64_t qo_tile_indices_offset;
  int64_t qo_indptr_offset;
  int64_t kv_indptr_offset;
  int64_t qo_len_offset;
  int64_t kv_len_offset;
  int64_t head_indices_offset;
  int64_t work_indptr_offset;
  int64_t batch_indices_offset;
  int64_t cta_mech_mode_offset;
  int64_t cta_valid_work_offset;  // per-CTA: 1 = has valid work (any kv_len > 0), 0 = not
  int64_t cta_is_dummy_offset;    // per-CTA: 1 = dummy CTA for mech2 cluster padding, 0 = real
  int num_ctas_launched;           // actual number of CTAs to launch (cluster-aligned)
  bool same_schedule_for_all_heads;
  bool kvsplit_mode;
  bool mech2_mode;

  PrefillPlanSM90Info()
      : qo_tile_indices_offset(0),
        qo_indptr_offset(0),
        kv_indptr_offset(0),
        qo_len_offset(0),
        kv_len_offset(0),
        head_indices_offset(0),
        work_indptr_offset(0),
        batch_indices_offset(0),
        cta_mech_mode_offset(0),
        cta_valid_work_offset(0),
        cta_is_dummy_offset(0),
        num_ctas_launched(0),
        same_schedule_for_all_heads(false),
        kvsplit_mode(false),
        mech2_mode(false) {}

  // convert PrefillPlanSM90Info to std::vector<int64_t>
  std::vector<int64_t> ToVector() const {
    return {qo_tile_indices_offset, qo_indptr_offset,     kv_indptr_offset,
            qo_len_offset,          kv_len_offset,        head_indices_offset,
            work_indptr_offset,     batch_indices_offset, cta_mech_mode_offset,
            cta_valid_work_offset,  cta_is_dummy_offset,  static_cast<int64_t>(num_ctas_launched),
            same_schedule_for_all_heads,
            static_cast<int64_t>(kvsplit_mode), static_cast<int64_t>(mech2_mode)};
  }

  // From std::vector<int64_t> to PrefillPlanSM90Info
  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() == 9) {
      // Backward compatibility: old format without flags
      qo_tile_indices_offset = vec[0];
      qo_indptr_offset = vec[1];
      kv_indptr_offset = vec[2];
      qo_len_offset = vec[3];
      kv_len_offset = vec[4];
      head_indices_offset = vec[5];
      work_indptr_offset = vec[6];
      batch_indices_offset = vec[7];
      same_schedule_for_all_heads = vec[8];
      cta_mech_mode_offset = 0;
      cta_valid_work_offset = 0;
      kvsplit_mode = false;
      mech2_mode = false;
    } else if (vec.size() == 11) {
      // Format with kvsplit/mech2 flags, no per-CTA mech
      qo_tile_indices_offset = vec[0];
      qo_indptr_offset = vec[1];
      kv_indptr_offset = vec[2];
      qo_len_offset = vec[3];
      kv_len_offset = vec[4];
      head_indices_offset = vec[5];
      work_indptr_offset = vec[6];
      batch_indices_offset = vec[7];
      same_schedule_for_all_heads = vec[8];
      kvsplit_mode = static_cast<bool>(vec[9]);
      mech2_mode = static_cast<bool>(vec[10]);
      cta_mech_mode_offset = 0;
      cta_valid_work_offset = 0;
    } else if (vec.size() == 12) {
      // New format with per-CTA mech array offset
      qo_tile_indices_offset = vec[0];
      qo_indptr_offset = vec[1];
      kv_indptr_offset = vec[2];
      qo_len_offset = vec[3];
      kv_len_offset = vec[4];
      head_indices_offset = vec[5];
      work_indptr_offset = vec[6];
      batch_indices_offset = vec[7];
      cta_mech_mode_offset = vec[8];
      cta_valid_work_offset = 0;
      same_schedule_for_all_heads = vec[9];
      kvsplit_mode = static_cast<bool>(vec[10]);
      mech2_mode = static_cast<bool>(vec[11]);
    } else if (vec.size() == 13) {
      // Format with per-CTA mech and per-CTA valid-work array offsets
      qo_tile_indices_offset = vec[0];
      qo_indptr_offset = vec[1];
      kv_indptr_offset = vec[2];
      qo_len_offset = vec[3];
      kv_len_offset = vec[4];
      head_indices_offset = vec[5];
      work_indptr_offset = vec[6];
      batch_indices_offset = vec[7];
      cta_mech_mode_offset = vec[8];
      cta_valid_work_offset = vec[9];
      cta_is_dummy_offset = 0;
      same_schedule_for_all_heads = vec[10];
      kvsplit_mode = static_cast<bool>(vec[11]);
      mech2_mode = static_cast<bool>(vec[12]);
    } else if (vec.size() == 14) {
      // Format with per-CTA mech, valid-work, and dummy array offsets (no num_ctas_launched)
      qo_tile_indices_offset = vec[0];
      qo_indptr_offset = vec[1];
      kv_indptr_offset = vec[2];
      qo_len_offset = vec[3];
      kv_len_offset = vec[4];
      head_indices_offset = vec[5];
      work_indptr_offset = vec[6];
      batch_indices_offset = vec[7];
      cta_mech_mode_offset = vec[8];
      cta_valid_work_offset = vec[9];
      cta_is_dummy_offset = vec[10];
      num_ctas_launched = 0;
      same_schedule_for_all_heads = vec[11];
      kvsplit_mode = static_cast<bool>(vec[12]);
      mech2_mode = static_cast<bool>(vec[13]);
    } else if (vec.size() == 15) {
      // Format with per-CTA mech, valid-work, dummy, and num_ctas_launched
      qo_tile_indices_offset = vec[0];
      qo_indptr_offset = vec[1];
      kv_indptr_offset = vec[2];
      qo_len_offset = vec[3];
      kv_len_offset = vec[4];
      head_indices_offset = vec[5];
      work_indptr_offset = vec[6];
      batch_indices_offset = vec[7];
      cta_mech_mode_offset = vec[8];
      cta_valid_work_offset = vec[9];
      cta_is_dummy_offset = vec[10];
      num_ctas_launched = static_cast<int>(vec[11]);
      same_schedule_for_all_heads = vec[12];
      kvsplit_mode = static_cast<bool>(vec[13]);
      mech2_mode = static_cast<bool>(vec[14]);
    } else {
      std::ostringstream err_msg;
      err_msg << "PrefillPlanSM90Info::FromVector: vec.size() should be 9, 11, 12, 13, 14 or 15, but got "
              << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// PAT prefix tree: radix tree construction + merge/split heuristic
////////////////////////////////////////////////////////////////////////////////////////////////////

struct PATNode {
  int parent = -1;
  int s_value = 0;
  int length = 0;
  const int32_t* block_ptr = nullptr;
  std::vector<int> seq_indices;
  std::unordered_map<int, int> children;

  PATNode(int p, int s, int l, const std::vector<int>& seq, const int32_t* bp)
      : parent(p), s_value(s), length(l), seq_indices(seq), block_ptr(bp) {}
};

struct PATPackedBox {
  std::vector<int> q_table;
  int num_seqs_per_CTA = 0;
  int kv_in_CTA = 0;
};

class PATPrefixTree {
 public:
  int block_size;
  std::vector<PATNode> nodes;
  int root;
  int num_nodes = 0;

  PATPrefixTree(int bs) : block_size(bs) {
    add_node(-1, 0, 0, {}, nullptr);
    root = 0;
  }

  int add_node(int parent, int s_value, int length,
               const std::vector<int>& seq_indices, const int32_t* block_ptr) {
    nodes.emplace_back(parent, s_value, length, seq_indices, block_ptr);
    return num_nodes++;
  }

  void build_radix_tree(const int* seq_lens_ptr, const int32_t* flat_table_ptr,
                         int num_seqs, int max_blocks) {
    nodes.reserve(num_seqs * 2);
    for (int i = 0; i < num_seqs; ++i) {
      const int32_t* row_ptr = flat_table_ptr + (i * max_blocks);
      int seq_len = seq_lens_ptr[i];
      int block_count = (seq_len + block_size - 1) / block_size;
      insert(i, seq_len, row_ptr, block_count);
    }
  }

  void insert(int sId, int seq_len, const int32_t* input_blocks_ptr, int input_block_count) {
    int node_idx = root;
    int res_block = input_block_count;
    int current_offset = 0;

    while (res_block > 0) {
      int first_block = input_blocks_ptr[current_offset];
      auto it = nodes[node_idx].children.find(first_block);

      if (it != nodes[node_idx].children.end()) {
        int child_id = it->second;
        int child_num_blocks = (nodes[child_id].length + block_size - 1) / block_size;
        const int32_t* child_ptr = nodes[child_id].block_ptr;

        int limit = std::min(child_num_blocks, res_block);
        int common_len = 0;
        for (int i = 0; i < limit; ++i) {
          if (input_blocks_ptr[current_offset + i] == child_ptr[i]) {
            common_len++;
          } else {
            break;
          }
        }

        if (common_len == child_num_blocks) {
          nodes[child_id].s_value += 1;
          nodes[child_id].seq_indices.push_back(sId);
          node_idx = child_id;
          res_block -= common_len;
          seq_len -= common_len * block_size;
          current_offset += common_len;
        } else {
          std::vector<int> mid_seq = nodes[child_id].seq_indices;
          const int32_t* mid_block_ptr = nodes[child_id].block_ptr;
          int split_block_id = child_ptr[common_len];
          int original_head_block = child_ptr[0];

          int mid = add_node(node_idx, nodes[child_id].s_value + 1,
                             common_len * block_size, mid_seq, mid_block_ptr);
          nodes[mid].children[split_block_id] = child_id;
          nodes[node_idx].children[original_head_block] = mid;
          nodes[child_id].parent = mid;
          nodes[child_id].block_ptr += common_len;
          nodes[child_id].length -= common_len * block_size;

          if (common_len == res_block) {
            nodes[mid].seq_indices.push_back(sId);
            break;
          }

          const int32_t* new_leaf_ptr = input_blocks_ptr + current_offset + common_len;
          int new_len = seq_len - common_len * block_size;
          int new_node_id = add_node(mid, 1, new_len, {sId}, new_leaf_ptr);
          nodes[mid].seq_indices.push_back(sId);
          nodes[mid].children[new_leaf_ptr[0]] = new_node_id;
          break;
        }
      } else {
        const int32_t* new_leaf_ptr = input_blocks_ptr + current_offset;
        int new_node = add_node(node_idx, 1, seq_len, {sId}, new_leaf_ptr);
        nodes[node_idx].children[first_block] = new_node;
        break;
      }
    }
  }

  // Recursive merge/split heuristic.
  // mm = max sequences per CTA.
  // inherited_kv_len = accumulated KV tokens from merged ancestors.
  std::vector<PATPackedBox> tree_heuristics(int node_id, int mm, int inherited_kv_len) {
    std::vector<PATPackedBox> res;
    PATNode& node = nodes[node_id];
    int current_kv_len = inherited_kv_len + node.length;

    if (node.children.empty()) {
      int S = node.s_value;
      for (int s = 0; s < S; s += mm) {
        PATPackedBox box;
        int end = std::min(s + mm, S);
        box.q_table.assign(node.seq_indices.begin() + s, node.seq_indices.begin() + end);
        box.num_seqs_per_CTA = box.q_table.size();
        box.kv_in_CTA = current_kv_len;
        res.push_back(std::move(box));
      }
    } else {
      int S = node.s_value;
      std::vector<int> ops(node.children.size(), 0);
      std::vector<int> split_child_ids, merge_child_ids;

      int idx = 0;
      for (auto& kv : node.children) {
        int child_id = kv.second;
        int child_s = nodes[child_id].s_value;

        // PAT profit model
        if (S == child_s ||
            (pat_ceil_div(S, mm) - pat_ceil_div(S - child_s, mm) -
             pat_ceil_div(child_s, mm)) * current_kv_len + 4 * child_s >= 0) {
          ops[idx] = 1;
          S -= child_s;
          merge_child_ids.push_back(child_id);
        } else {
          split_child_ids.push_back(child_id);
        }
        idx++;
      }

      if (S != 0) {
        size_t total_seqs = node.seq_indices.size();
        std::vector<uint8_t> is_merged(total_seqs > 0 ? *std::max_element(
            node.seq_indices.begin(), node.seq_indices.end()) + 1 : 0, 0);
        int c_idx = 0;
        for (auto& kv : node.children) {
          if (ops[c_idx] == 1) {
            for (int sId : nodes[kv.second].seq_indices) {
              if (sId < (int)is_merged.size()) is_merged[sId] = 1;
            }
          }
          c_idx++;
        }

        std::vector<int> remaining_seqs;
        for (int sId : node.seq_indices) {
          if (sId >= (int)is_merged.size() || is_merged[sId] == 0)
            remaining_seqs.push_back(sId);
        }

        for (size_t s = 0; s < remaining_seqs.size(); s += mm) {
          PATPackedBox box;
          size_t end = std::min(s + (size_t)mm, remaining_seqs.size());
          box.q_table.assign(remaining_seqs.begin() + s, remaining_seqs.begin() + end);
          box.num_seqs_per_CTA = (int)box.q_table.size();
          box.kv_in_CTA = current_kv_len;
          res.push_back(std::move(box));
        }
      }

      for (int child_id : split_child_ids) {
        auto child_boxes = tree_heuristics(child_id, mm, 0);
        res.insert(res.end(), std::make_move_iterator(child_boxes.begin()),
                   std::make_move_iterator(child_boxes.end()));
      }
      for (int child_id : merge_child_ids) {
        auto child_boxes = tree_heuristics(child_id, mm, current_kv_len);
        res.insert(res.end(), std::make_move_iterator(child_boxes.begin()),
                   std::make_move_iterator(child_boxes.end()));
      }
    }
    return res;
  }

 private:
  static inline int pat_ceil_div(int a, int b) { return (a + b - 1) / b; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename IdType>
inline cudaError_t PrefillSM90Plan(
    void* float_buffer, size_t float_workspace_size_in_bytes, void* int_buffer,
    void* page_locked_int_buffer, size_t int_workspace_size_in_bytes,
    PrefillPlanSM90Info& plan_info, IdType* qo_indptr_h, IdType* kv_indptr_h, IdType* kv_len_arr_h,
    uint32_t total_num_rows, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t head_dim_qk, uint32_t head_dim_vo, uint32_t page_size, bool causal,
    bool enable_cuda_graph, uint32_t sizeof_dtype_o, cudaStream_t stream,
    bool use_tree_walk_scheduling = false, bool kvsplit_mode = false, bool mech2_mode = false,
    bool use_pat_scheduling = false,
    const int32_t* block_tables_ptr = nullptr, int max_blocks_per_seq = 0,
    int num_pat_seqs = 0,
    int32_t* q_perm_out = nullptr, int* q_perm_size_out = nullptr) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
            << num_kv_heads;
    FLASHINFER_ERROR(err_msg.str());
  }

  std::vector<std::tuple<int, int, int>> idx_qo_kv_len_vec;
  for (uint32_t i = 0; i < batch_size; ++i) {
    int qo_len = qo_indptr_h[i + 1] - qo_indptr_h[i];
    int kv_len = kv_len_arr_h[i];
    if (kv_len < 0) {
      std::ostringstream err_msg;
      err_msg << "kv_len[" << i << "]" << kv_len << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }
    if (qo_len < 0) {
      std::ostringstream err_msg;
      err_msg << "qo_indptr[" << i + 1 << "]" << qo_indptr_h[i + 1] << " - qo_indptr[" << i << "]"
              << qo_indptr_h[i] << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }
    idx_qo_kv_len_vec.push_back({i, qo_len, kv_len});
  }

#ifdef FLASHINFER_DEBUG_SCHEDULER
  printf("\n--- idx_qo_kv_len_vec (BEFORE sorting) ---\n");
  printf("batch_size=%zu\n", batch_size);
  for (const auto& [idx, qo_len, kv_len] : idx_qo_kv_len_vec) {
    printf("  [batch_idx=%d, qo_len=%d, kv_len=%d]\n", idx, qo_len, kv_len);
  }
#endif

  std::vector<bool> is_dummy_cta_vec;
  // Per-entry cluster group ID: entries with same group_id >= 0 must be assigned to
  // cluster-aligned CTAs. -1 means no cluster constraint (use normal min-heap assignment).
  std::vector<int> cluster_group_id_vec;
  int next_cluster_group_id = 0;

  if (use_tree_walk_scheduling) {
    // Tree-based 2-level walk scheduling (assumes uniform/balanced trees):
    // Infer tree structure from qo_indptr by grouping nodes with same qo_len
    // Group nodes by qo_len to infer tree levels
    //printf("Using tree-based 2-level walk scheduling\n");
    std::map<int, std::vector<int>> nodes_by_qo_len;
    for (uint32_t i = 0; i < batch_size; ++i) {
      int qo_len = qo_indptr_h[i + 1] - qo_indptr_h[i];
      nodes_by_qo_len[qo_len].push_back(i);
    }
    
    // Build tree structure: levels are ordered by decreasing qo_len (more sequences = higher level)
    std::vector<std::vector<int>> tree_levels;
    for (auto it = nodes_by_qo_len.rbegin(); it != nodes_by_qo_len.rend(); ++it) {
      tree_levels.push_back(it->second);
    }
    
    // Reorder idx_qo_kv_len_vec: walk levels from root (0).
    // If any node at level L has kv_len >= 128: add all nodes at L and move on.
    // If all nodes at L have kv_len < 128: add one node of L, then its children; then next node of L, then its children; and so on.
    std::vector<std::tuple<int, int, int>> reordered_idx_qo_kv_len_vec;
    std::set<size_t> already_output;

    if (tree_levels.size() >= 2) {
      for (size_t L = 0; L < tree_levels.size(); ++L) {
        if (already_output.count(L) != 0) continue;

        bool level_has_long_kv = false;
        for (int node_idx : tree_levels[L]) {
          if (std::get<2>(idx_qo_kv_len_vec[node_idx]) >= 128) {
            level_has_long_kv = true;
            break;
          }
        }

        if (level_has_long_kv) {
          for (int node_idx : tree_levels[L]) {
            reordered_idx_qo_kv_len_vec.push_back(idx_qo_kv_len_vec[node_idx]);
            is_dummy_cta_vec.push_back(false);
            cluster_group_id_vec.push_back(-1);  // mech1: no cluster constraint
          }
          already_output.insert(L);
        } else {
          constexpr int kMech2ClusterSize = 4;
          for (size_t parent_idx = 0; parent_idx < tree_levels[L].size(); ++parent_idx) {
            int group_id = next_cluster_group_id++;
            int parent_node_idx = tree_levels[L][parent_idx];
            reordered_idx_qo_kv_len_vec.push_back(idx_qo_kv_len_vec[parent_node_idx]);
            is_dummy_cta_vec.push_back(false);
            cluster_group_id_vec.push_back(group_id);
            if (L + 1 < tree_levels.size()) {
              int children_per_parent = tree_levels[L + 1].size() / tree_levels[L].size();
              int child_start = parent_idx * children_per_parent;
              int child_end = (parent_idx + 1) * children_per_parent;
              int num_children = 0;
              for (int child_idx = child_start; child_idx < child_end &&
                   child_idx < (int)tree_levels[L + 1].size(); ++child_idx) {
                int child_node_idx = tree_levels[L + 1][child_idx];
                reordered_idx_qo_kv_len_vec.push_back(idx_qo_kv_len_vec[child_node_idx]);
                is_dummy_cta_vec.push_back(false);
                cluster_group_id_vec.push_back(group_id);
                num_children++;
              }
              // Pad with dummy CTAs if children < kMech2ClusterSize for mech2 cluster alignment
              int total_in_group = 1 + num_children;  // parent + children
              int remainder = total_in_group % kMech2ClusterSize;
              if (remainder != 0) {
                int num_dummies = kMech2ClusterSize - remainder;
                // Use parent's entry as template for dummy (same qo/kv metadata, marked invalid)
                for (int d = 0; d < num_dummies; ++d) {
                  reordered_idx_qo_kv_len_vec.push_back(idx_qo_kv_len_vec[parent_node_idx]);
                  is_dummy_cta_vec.push_back(true);
                  cluster_group_id_vec.push_back(group_id);
                }
              }
            }
          }
          already_output.insert(L);
          if (L + 1 < tree_levels.size()) already_output.insert(L + 1);
        }
      }

      idx_qo_kv_len_vec = reordered_idx_qo_kv_len_vec;
    }
  }

  // Build per-entry dummy flag and cluster group ID aligned with idx_qo_kv_len_vec.
  // Non-tree-walk paths have no dummies and no cluster constraints.
  std::vector<bool> idx_is_dummy(idx_qo_kv_len_vec.size(), false);
  std::vector<int> idx_cluster_group(idx_qo_kv_len_vec.size(), -1);
  if (use_tree_walk_scheduling && !is_dummy_cta_vec.empty()) {
    for (size_t i = 0; i < is_dummy_cta_vec.size() && i < idx_is_dummy.size(); ++i) {
      idx_is_dummy[i] = is_dummy_cta_vec[i];
    }
    for (size_t i = 0; i < cluster_group_id_vec.size() && i < idx_cluster_group.size(); ++i) {
      idx_cluster_group[i] = cluster_group_id_vec[i];
    }
  }

  if (!use_tree_walk_scheduling) {
    // Original scheduling: sort by kv_len descending, then batch_idx ascending
    std::sort(idx_qo_kv_len_vec.begin(), idx_qo_kv_len_vec.end(),
              [](const auto& a, const auto& b) {
                if (std::get<2>(a) != std::get<2>(b))
                  return std::get<2>(a) > std::get<2>(b);  // primary: kv_len descending
                return std::get<0>(a) < std::get<0>(b);    // secondary: batch_idx ascending
              });
  }

#ifdef FLASHINFER_DEBUG_SCHEDULER
  printf("\n--- idx_qo_kv_len_vec (AFTER sorting by kv_len descending) ---\n");
  for (const auto& [idx, qo_len, kv_len] : idx_qo_kv_len_vec) {
    printf("  [batch_idx=%d, qo_len=%d, kv_len=%d]\n", idx, qo_len, kv_len);
  }
#endif

  // Collect all tiles with their costs for debugging
  std::vector<std::tuple<int, int, int, int, int, int, float>> all_tiles_with_cost;
  // tuple: (batch_idx, qo_tile_idx, qo_len, kv_len, qo_indptr, kv_indptr, cost)
  for (const auto& [idx, qo_len, kv_len] : idx_qo_kv_len_vec) {
    int num_qo_tiles = ceil_div(qo_len, 128);  // Using cta_tile_q = 128 temporarily
    for (int qo_tile_idx = 0; qo_tile_idx < num_qo_tiles; ++qo_tile_idx) {
      // For non-causal or when causal, calculate effective kv_len
      int effective_kv_len = causal 
        ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx, 128, num_qo_tiles, 1)
        : kv_len;
      float tile_cost = cost_function(128, effective_kv_len);
      all_tiles_with_cost.push_back({idx, qo_tile_idx, qo_len, kv_len, 
                                      int(qo_indptr_h[idx]), int(kv_indptr_h[idx]), tile_cost});
    }
  }

  // Sort by cost descending
  std::sort(all_tiles_with_cost.begin(), all_tiles_with_cost.end(),
            [](const auto& a, const auto& b) {
              return std::get<6>(a) > std::get<6>(b);  // cost descending
            });

#ifdef FLASHINFER_DEBUG_SCHEDULER
  printf("\n--- ALL TILES sorted by COST (descending) ---\n");
  printf("Total tiles: %zu\n", all_tiles_with_cost.size());
  for (const auto& [batch_idx, qo_tile_idx, qo_len, kv_len, qo_indptr, kv_indptr, cost] : all_tiles_with_cost) {
    printf("  Batch%d-Tile%d: qo_len=%d, kv_len=%d, qo_indptr=%d, kv_indptr=%d, cost=%.1f\n",
           batch_idx, qo_tile_idx, qo_len, kv_len, qo_indptr, kv_indptr, cost);
  }
  printf("================================================\n\n");
#endif

  int cta_tile_q = 128;
  if (head_dim_vo == 64) {
    cta_tile_q = 192;
  }

  int device = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int num_sm90_ctas = 0;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&num_sm90_ctas, cudaDevAttrMultiProcessorCount, device));

  // Compute total CTAs needed: cluster groups each reserve kClusterSize CTAs,
  // plus enough for non-cluster work (up to num_sm90_ctas).
  constexpr int kClusterSizeForAlloc = 4;
  int num_cluster_ctas_needed = next_cluster_group_id * kClusterSizeForAlloc;
  int num_total_ctas = std::max(num_sm90_ctas, num_cluster_ctas_needed);
  // PAT scheduling: need at least one CTA per batch entry (1:1 mapping)
  if (use_pat_scheduling) {
    num_total_ctas = std::max(num_total_ctas, static_cast<int>(batch_size));
  }

  MinHeap cta_cost_heap(num_total_ctas);
  std::vector<std::vector<IdType>> cta_qo_tile_indices(num_total_ctas, std::vector<IdType>()),
      cta_qo_indptr(num_total_ctas, std::vector<IdType>()),
      cta_kv_indptr(num_total_ctas, std::vector<IdType>()),
      cta_qo_len(num_total_ctas, std::vector<IdType>()),
      cta_kv_len(num_total_ctas, std::vector<IdType>()),
      cta_head_indices(num_total_ctas, std::vector<IdType>()),
      cta_batch_indices(num_total_ctas, std::vector<IdType>());

  int max_num_works_per_head = ceil_div(total_num_rows, cta_tile_q) + batch_size - 1;
  plan_info.same_schedule_for_all_heads = max_num_works_per_head > 4096;
  plan_info.kvsplit_mode = use_pat_scheduling ? false : kvsplit_mode;
  plan_info.mech2_mode = use_pat_scheduling ? false : mech2_mode;

#ifdef FLASHINFER_DEBUG_SCHEDULER
  printf("\n========== FLASHINFER LOAD BALANCING ==========\n");
  printf("num_sms=%d, batch_size=%zu, cta_tile_q=%d\n", num_sm90_ctas, batch_size, cta_tile_q);
  printf("Processing order (sorted by kv_len descending):\n");
  for (auto& [i, qo_len, kv_len] : idx_qo_kv_len_vec) {
    printf("  Batch %d: qo_len=%d, kv_len=%d, num_tiles=%d\n", i, qo_len, kv_len,
           ceil_div(qo_len, cta_tile_q));
  }
  printf("\n--- Tile Assignment ---\n");
#endif

  // Track which CTAs received real vs dummy work items
  constexpr int kMech1NumReplicas = 4;
  constexpr int kClusterSize = 4;
  std::vector<bool> cta_has_real_work(num_total_ctas, false);
  std::vector<bool> cta_has_dummy_work(num_total_ctas, false);
  std::vector<float> cta_cost(num_total_ctas, 0.0f);
  std::unordered_map<int, int> cluster_group_base;

  // ========== PAT scheduling: build fused tree from block tables ==========
  if (use_pat_scheduling && block_tables_ptr != nullptr && num_pat_seqs > 0) {
    const int pat_batch = num_pat_seqs;
    const uint32_t group_size = num_qo_heads / num_kv_heads;
    int mm = cta_tile_q / static_cast<int>(group_size);
    if (mm < 1) mm = 1;

    // Build radix tree from per-leaf block tables.
    // kv_len_arr_h has batch_size entries (all tree nodes), but we only need
    // the first pat_batch entries' total KV lengths. Since the block tables
    // are per-leaf with full root-to-leaf pages, compute seq_len from page count.
    std::vector<int> seq_lens_int(pat_batch);
    for (int i = 0; i < pat_batch; ++i) {
      // Count non-zero pages in this leaf's block table row
      int num_pages = 0;
      for (int j = 0; j < max_blocks_per_seq; ++j) {
        if (block_tables_ptr[i * max_blocks_per_seq + j] != 0 || j == 0) {
          num_pages = j + 1;
        }
      }
      seq_lens_int[i] = num_pages * page_size;
    }

    PATPrefixTree tree(page_size);
    tree.build_radix_tree(seq_lens_int.data(), block_tables_ptr, pat_batch, max_blocks_per_seq);

    // Debug: print radix tree structure
    {
      printf("\n========== PAT RADIX TREE ==========\n");
      printf("num_nodes=%d, block_size=%d\n", tree.num_nodes, tree.block_size);
      for (int n = 0; n < tree.num_nodes; ++n) {
        const auto& nd = tree.nodes[n];
        printf("  Node[%d]: parent=%d, s=%d, len=%d, seqs=[", n, nd.parent, nd.s_value, nd.length);
        for (size_t j = 0; j < nd.seq_indices.size(); ++j) {
          printf("%d%s", nd.seq_indices[j], j + 1 < nd.seq_indices.size() ? "," : "");
        }
        printf("], blocks=[");
        if (nd.block_ptr) {
          int num_blks = (nd.length + tree.block_size - 1) / tree.block_size;
          for (int j = 0; j < num_blks; ++j) {
            printf("%d%s", nd.block_ptr[j], j + 1 < num_blks ? "," : "");
          }
        }
        printf("], children={");
        bool first = true;
        for (const auto& ch : nd.children) {
          if (!first) printf(", ");
          printf("%d->%d", ch.first, ch.second);
          first = false;
        }
        printf("}\n");
      }
      printf("====================================\n\n");
    }

    std::vector<PATPackedBox> packed_boxes;
    for (const auto& kv : tree.nodes[tree.root].children) {
      auto new_boxes = tree.tree_heuristics(kv.second, mm, 0);
      packed_boxes.insert(packed_boxes.end(),
                          std::make_move_iterator(new_boxes.begin()),
                          std::make_move_iterator(new_boxes.end()));
    }

    // Sort boxes in BFS order: by kv_in_CTA descending (root first), then by first seq index
    std::sort(packed_boxes.begin(), packed_boxes.end(),
              [](const PATPackedBox& a, const PATPackedBox& b) {
                if (a.kv_in_CTA != b.kv_in_CTA) return a.kv_in_CTA > b.kv_in_CTA;
                int a_first = a.q_table.empty() ? 0 : a.q_table.front();
                int b_first = b.q_table.empty() ? 0 : b.q_table.front();
                if (a_first != b_first) return a_first < b_first;
                return a.num_seqs_per_CTA > b.num_seqs_per_CTA;
              });

    // Print PAT fused tree summary (runs once during plan, not in hot path)
    {
      std::map<int, int> boxes_by_kv;
      for (const auto& box : packed_boxes) boxes_by_kv[box.kv_in_CTA] += 1;

      printf("\n========== PAT FUSED TREE ==========\n");
      printf("pat_batch=%d, mm=%d, page_size=%d, total_boxes=%zu\n",
             pat_batch, mm, page_size, packed_boxes.size());
      printf("Fused levels (CTAs@kv): ");
      std::vector<std::pair<int,int>> levels(boxes_by_kv.rbegin(), boxes_by_kv.rend());
      for (size_t i = 0; i < levels.size(); ++i) {
        printf("%d@kv=%d%s", levels[i].second, levels[i].first,
               i + 1 < levels.size() ? ", " : "\n");
      }
      for (size_t i = 0; i < packed_boxes.size(); ++i) {
        const auto& box = packed_boxes[i];
        printf("  Box[%zu]: %d seqs, kv=%d, seqs=[",
               i, box.num_seqs_per_CTA, box.kv_in_CTA);
        for (size_t j = 0; j < box.q_table.size(); ++j) {
          printf("%d%s", box.q_table[j], j + 1 < box.q_table.size() ? "," : "");
        }
        printf("]\n");
      }
      printf("====================================\n\n");
    }

    // Step 1: Build Q permutation — walk boxes, collect leaf indices in order
    std::vector<int32_t> q_perm;
    q_perm.reserve(pat_batch);
    for (const auto& box : packed_boxes) {
      for (int leaf_id : box.q_table) {
        q_perm.push_back(leaf_id);
      }
    }
    if (q_perm_out != nullptr) {
      std::copy(q_perm.begin(), q_perm.end(), q_perm_out);
    }
    if (q_perm_size_out != nullptr) {
      *q_perm_size_out = static_cast<int>(q_perm.size());
    }

    // Step 2+3: Build per-box work items, box[i] → CTA[i]
    // Compute per-leaf token-level KV start offsets (cumulative sum of seq_lens)
    // kv_indptr_h is indexed by original batch entries (all tree nodes), not leaves.
    // We need per-leaf offsets computed from seq_lens_int.
    std::vector<int> leaf_kv_start(pat_batch, 0);
    for (int i = 1; i < pat_batch; ++i) {
      leaf_kv_start[i] = leaf_kv_start[i - 1] + seq_lens_int[i - 1];
    }

    // Track per-leaf accumulated KV offset for split boxes
    std::vector<int> leaf_kv_offset(pat_batch, 0);
    int perm_qo_offset = 0;

    for (int qo_head_idx = 0;
         qo_head_idx < (plan_info.same_schedule_for_all_heads ? 1 : num_qo_heads);
         ++qo_head_idx) {
      perm_qo_offset = 0;
      std::fill(leaf_kv_offset.begin(), leaf_kv_offset.end(), 0);

      for (size_t box_idx = 0; box_idx < packed_boxes.size(); ++box_idx) {
        const auto& box = packed_boxes[box_idx];
        if (box.q_table.empty()) continue;

        int target_cta = static_cast<int>(box_idx);
        int fused_qo_len = box.num_seqs_per_CTA;
        int fused_kv_len = box.kv_in_CTA;

        // kv_indptr: first leaf's token-level start + accumulated offset from prior boxes
        int first_leaf = box.q_table.front();
        int fused_kv_indptr = leaf_kv_start[first_leaf] + leaf_kv_offset[first_leaf];

        // Tile the fused Q range
        int packed_qo_len = fused_qo_len * static_cast<int>(group_size);
        int num_qo_tiles = ceil_div(packed_qo_len, cta_tile_q);
        if (num_qo_tiles < 1) num_qo_tiles = 1;

        for (int qo_tile_idx = 0; qo_tile_idx < num_qo_tiles; ++qo_tile_idx) {
          cta_qo_tile_indices[target_cta].push_back(qo_tile_idx);
          cta_qo_indptr[target_cta].push_back(perm_qo_offset);
          cta_qo_len[target_cta].push_back(fused_qo_len);
          cta_kv_indptr[target_cta].push_back(fused_kv_indptr);
          cta_kv_len[target_cta].push_back(fused_kv_len);
          cta_head_indices[target_cta].push_back(qo_head_idx);
          cta_batch_indices[target_cta].push_back(first_leaf);
          cta_has_real_work[target_cta] = true;
        }

        // Advance per-leaf KV offset for all leaves in this box
        for (int leaf_id : box.q_table) {
          leaf_kv_offset[leaf_id] += fused_kv_len;
        }

        perm_qo_offset += fused_qo_len;
      }
    }

    // Print PAT CTA assignment
    {
      printf("\n========== PAT CTA ASSIGNMENT ==========\n");
      printf("total_boxes=%zu, perm_qo_offset=%d\n", packed_boxes.size(), perm_qo_offset);
      perm_qo_offset = 0;
      std::fill(leaf_kv_offset.begin(), leaf_kv_offset.end(), 0);
      for (size_t box_idx = 0; box_idx < packed_boxes.size(); ++box_idx) {
        const auto& box = packed_boxes[box_idx];
        if (box.q_table.empty()) continue;
        int first_leaf = box.q_table.front();
        int kv_off = leaf_kv_offset[first_leaf];
        printf("  CTA%zu: qo_indptr=%d, qo_len=%d, kv_start=%d+%d=%d, kv_len=%d, seqs=[",
               box_idx, perm_qo_offset, box.num_seqs_per_CTA,
               leaf_kv_start[first_leaf], kv_off,
               leaf_kv_start[first_leaf] + kv_off,
               box.kv_in_CTA);
        for (size_t j = 0; j < box.q_table.size(); ++j) {
          printf("%d%s", box.q_table[j], j + 1 < box.q_table.size() ? "," : "");
        }
        printf("]\n");
        for (int leaf_id : box.q_table) {
          leaf_kv_offset[leaf_id] += box.kv_in_CTA;
        }
        perm_qo_offset += box.num_seqs_per_CTA;
      }
      printf("=========================================\n\n");
    }
  } else if (use_pat_scheduling) {
    // PAT scheduling without block tables: 1:1 assignment of batch entries to CTAs.
    // Used when Python already restructured the data (each "batch entry" = one box).
    const uint32_t group_size = num_qo_heads / num_kv_heads;
    for (int qo_head_idx = 0;
         qo_head_idx < (plan_info.same_schedule_for_all_heads ? 1 : num_qo_heads);
         ++qo_head_idx) {
      for (uint32_t entry = 0; entry < batch_size; ++entry) {
        int target_cta = static_cast<int>(entry);
        int qo_start = static_cast<int>(qo_indptr_h[entry]);
        int qo_len = static_cast<int>(qo_indptr_h[entry + 1]) - qo_start;
        int kv_len = static_cast<int>(kv_len_arr_h[entry]);
        int packed_qo = qo_len * static_cast<int>(group_size);
        int num_qo_tiles = ceil_div(packed_qo, cta_tile_q);
        if (num_qo_tiles < 1) num_qo_tiles = 1;

        for (int qo_tile_idx = 0; qo_tile_idx < num_qo_tiles; ++qo_tile_idx) {
          cta_qo_tile_indices[target_cta].push_back(qo_tile_idx);
          cta_qo_indptr[target_cta].push_back(qo_indptr_h[entry]);
          cta_qo_len[target_cta].push_back(qo_len);
          cta_kv_indptr[target_cta].push_back(kv_indptr_h[entry]);
          cta_kv_len[target_cta].push_back(kv_len);
          cta_head_indices[target_cta].push_back(qo_head_idx);
          cta_batch_indices[target_cta].push_back(entry);
          cta_has_real_work[target_cta] = true;
        }
      }
    }
  } else {

  // When mech1 (effective_kv_len >= 128), assign the same Q tile to 4 CTAs so each CTA
  // works on a subset of KV; the kernel handles kv_start/num_kv_tiles internally.

  // Helper: find the least-loaded cluster-aligned base CTA
  auto find_best_cluster_base = [&]() -> int {
    int num_clusters = num_total_ctas / kClusterSize;
    int best_base = 0;
    float best_cost = std::numeric_limits<float>::max();
    for (int c = 0; c < num_clusters; ++c) {
      int base = c * kClusterSize;
      float max_cost_in_cluster = 0.0f;
      for (int j = 0; j < kClusterSize; ++j) {
        max_cost_in_cluster = std::max(max_cost_in_cluster, cta_cost[base + j]);
      }
      if (max_cost_in_cluster < best_cost) {
        best_cost = max_cost_in_cluster;
        best_base = base;
      }
    }
    return best_base;
  };

  for (int qo_head_idx = 0;
       qo_head_idx < (plan_info.same_schedule_for_all_heads ? 1 : num_qo_heads); ++qo_head_idx) {
    size_t vec_idx = 0;
    while (vec_idx < idx_qo_kv_len_vec.size()) {
      int group_id = idx_cluster_group[vec_idx];

      if (group_id >= 0) {
        // Cluster-aligned group: collect all entries with same group_id
        size_t group_start = vec_idx;
        while (vec_idx < idx_qo_kv_len_vec.size() && idx_cluster_group[vec_idx] == group_id) {
          vec_idx++;
        }
        size_t group_size = vec_idx - group_start;

        // Find or reuse cluster-aligned base CTA for this group
        int base_cta;
        auto it = cluster_group_base.find(group_id);
        if (it != cluster_group_base.end()) {
          base_cta = it->second;
        } else {
          base_cta = find_best_cluster_base();
          cluster_group_base[group_id] = base_cta;
        }

        // Assign each entry in the group to consecutive CTAs within the cluster
        for (size_t g = 0; g < group_size; ++g) {
          size_t entry_idx = group_start + g;
          auto& [i, qo_len, kv_len] = idx_qo_kv_len_vec[entry_idx];
          bool is_dummy = idx_is_dummy[entry_idx];
          int cta_idx = base_cta + static_cast<int>(g % kClusterSize);
          int num_qo_tiles = ceil_div(qo_len, cta_tile_q);

          for (int qo_tile_idx = num_qo_tiles - 1; qo_tile_idx >= 0; --qo_tile_idx) {
            int effective_kv_len =
                causal ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx, cta_tile_q, num_qo_tiles, 1)
                       : kv_len;
            float tile_cost = is_dummy ? 0.0f : cost_function(cta_tile_q, effective_kv_len);

#ifdef FLASHINFER_DEBUG_SCHEDULER
            printf("  Batch%d-Tile%d (qo_head=%d, eff_kv=%d, cost=%.1f, dummy=%d, cluster_group=%d) -> SM%d (cluster base=%d)\n",
                   i, qo_tile_idx, qo_head_idx, effective_kv_len, tile_cost,
                   is_dummy, group_id, cta_idx, base_cta);
#endif
            cta_cost[cta_idx] += tile_cost;
            cta_qo_tile_indices[cta_idx].push_back(qo_tile_idx);
            cta_qo_indptr[cta_idx].push_back(qo_indptr_h[i]);
            cta_qo_len[cta_idx].push_back(qo_len);
            cta_kv_indptr[cta_idx].push_back(kv_indptr_h[i]);
            cta_kv_len[cta_idx].push_back(is_dummy ? 0 : kv_len);
            cta_head_indices[cta_idx].push_back(qo_head_idx);
            cta_batch_indices[cta_idx].push_back(i);
            if (!is_dummy) {
              cta_has_real_work[cta_idx] = true;
            } else {
              cta_has_dummy_work[cta_idx] = true;
            }
          }
        }

        // Rebuild the min-heap to reflect updated costs after cluster assignment.
        // MinHeap constructor initializes all entries with cost 0, so we pop all
        // and re-insert with actual costs.
        cta_cost_heap = MinHeap(num_total_ctas);
        // Pop all default-initialized entries
        for (uint32_t c = 0; c < num_total_ctas; ++c) {
          cta_cost_heap.pop();
        }
        // Re-insert with actual costs
        for (uint32_t c = 0; c < num_total_ctas; ++c) {
          cta_cost_heap.insert({static_cast<int>(c), cta_cost[c]});
        }
      } else {
        // Normal min-heap assignment (mech1 or unconstrained)
        auto& [i, qo_len, kv_len] = idx_qo_kv_len_vec[vec_idx];
        bool is_dummy = idx_is_dummy[vec_idx];
        int num_qo_tiles = ceil_div(qo_len, cta_tile_q);
        for (int qo_tile_idx = num_qo_tiles - 1; qo_tile_idx >= 0; --qo_tile_idx) {
          int effective_kv_len =
              causal ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx, cta_tile_q, num_qo_tiles, 1)
                     : kv_len;
          bool is_mech1 = (effective_kv_len >= 128);
          int num_replicas = is_mech1 ? kMech1NumReplicas : 1;
          float tile_cost = is_dummy ? 0.0f : cost_function(cta_tile_q, effective_kv_len);

          for (int replica = 0; replica < num_replicas; ++replica) {
            auto [cta_idx, accum_cost] = cta_cost_heap.pop();
#ifdef FLASHINFER_DEBUG_SCHEDULER
            printf("  Batch%d-Tile%d (qo_head=%d, eff_kv=%d, cost=%.1f, replica=%d/%d, dummy=%d) -> SM%d (prev_cost=%.1f, new_cost=%.1f)\n",
                   i, qo_tile_idx, qo_head_idx, effective_kv_len, tile_cost, replica, num_replicas,
                   is_dummy, cta_idx, accum_cost, accum_cost + tile_cost);
#endif
            cta_cost[cta_idx] = accum_cost + tile_cost;
            cta_cost_heap.insert({cta_idx, cta_cost[cta_idx]});
            cta_qo_tile_indices[cta_idx].push_back(qo_tile_idx);
            cta_qo_indptr[cta_idx].push_back(qo_indptr_h[i]);
            cta_qo_len[cta_idx].push_back(qo_len);
            cta_kv_indptr[cta_idx].push_back(kv_indptr_h[i]);
            cta_kv_len[cta_idx].push_back(is_dummy ? 0 : kv_len);
            cta_head_indices[cta_idx].push_back(qo_head_idx);
            cta_batch_indices[cta_idx].push_back(i);
            if (!is_dummy) {
              cta_has_real_work[cta_idx] = true;
            } else {
              cta_has_dummy_work[cta_idx] = true;
            }
          }
        }
        vec_idx++;
      }
    }
  }
  } // end else (non-PAT scheduling)

  std::vector<IdType> work_indptr_vec(num_total_ctas + 1, 0);
  for (uint32_t i = 0; i < num_total_ctas; ++i) {
    work_indptr_vec[i + 1] = work_indptr_vec[i] + cta_qo_tile_indices[i].size();
  }
  int total_num_works = work_indptr_vec.back();

  // Compute actual number of CTAs needed: find highest CTA with work, round up to cluster size
  {
    // PAT mode uses cluster size 1 (no DSM); normal mode uses cluster size 4
    int kClusterSizeForLaunch = (use_pat_scheduling) ? 1 : 4;
    int max_active_cta = 0;
    for (uint32_t i = 0; i < num_total_ctas; ++i) {
      if (!cta_qo_tile_indices[i].empty()) {
        max_active_cta = i + 1;
      }
    }
    // Round up to nearest cluster multiple
    int num_ctas = ((max_active_cta + kClusterSizeForLaunch - 1) / kClusterSizeForLaunch) * kClusterSizeForLaunch;
    //printf("max_active_cta: %d, num_ctas: %d\n", max_active_cta, num_ctas);
    plan_info.num_ctas_launched = num_ctas;
    // Extend arrays to cover cluster-aligned CTAs (empty CTAs need valid entries)
    while ((int)work_indptr_vec.size() <= num_ctas) {
      work_indptr_vec.push_back(total_num_works);
    }
    // Update num_total_ctas to match launched CTAs so all per-CTA arrays are sized correctly
    if (num_ctas > num_total_ctas) {
      cta_qo_tile_indices.resize(num_ctas);
      cta_qo_indptr.resize(num_ctas);
      cta_kv_indptr.resize(num_ctas);
      cta_qo_len.resize(num_ctas);
      cta_kv_len.resize(num_ctas);
      cta_head_indices.resize(num_ctas);
      cta_batch_indices.resize(num_ctas);
      cta_has_real_work.resize(num_ctas, false);
      cta_has_dummy_work.resize(num_ctas, false);
      cta_cost.resize(num_ctas, 0.0f);
      num_total_ctas = num_ctas;
    }
  }

#ifdef FLASHINFER_DEBUG_SCHEDULER
  printf("\n--- Final SM Assignment Summary ---\n");
  printf("num_ctas_launched=%d (num_total_ctas=%d, num_sms=%d)\n", plan_info.num_ctas_launched, num_total_ctas, num_sm90_ctas);
  printf("total_num_works=%d\n", total_num_works);
  printf("work_indptr = [");
  for (uint32_t i = 0; i <= num_total_ctas; ++i) {
    printf("%d%s", work_indptr_vec[i], i < num_total_ctas ? ", " : "]\n");
  }
  
  // Print final accumulated costs per SM from the heap
  printf("\n--- Final Accumulated Cost Per SM ---\n");
  float max_cost = 0.0f, min_cost = 1e9f, total_cost = 0.0f;
  auto heap_state = cta_cost_heap.getHeap();
  for (const auto& [sm_idx, cost] : heap_state) {
    printf("SM%d: cost=%.1f (%d tiles)\n", sm_idx, cost, (int)cta_qo_tile_indices[sm_idx].size());
    max_cost = std::max(max_cost, cost);
    min_cost = std::min(min_cost, cost);
    total_cost += cost;
  }
  printf("Cost stats: min=%.1f, max=%.1f, avg=%.1f, imbalance=%.2f%%\n",
         min_cost, max_cost, total_cost / num_total_ctas,
         (max_cost - min_cost) / max_cost * 100.0f);
  
  printf("\n--- CTA ID -> Work (per-CTA assignment) ---\n");
  for (uint32_t cta_id = 0; cta_id < num_total_ctas; ++cta_id) {
    int num_works = static_cast<int>(cta_qo_tile_indices[cta_id].size());
    if (num_works == 0) continue;
    int work_start = work_indptr_vec[cta_id];
    int work_end = work_indptr_vec[cta_id + 1];
    printf("CTA%d: work_indptr[%d..%d] (%d works)\n", cta_id, work_start, work_end - 1, num_works);
    for (int t = 0; t < num_works; ++t) {
      int batch = static_cast<int>(cta_batch_indices[cta_id][t]);
      int qo_tile = static_cast<int>(cta_qo_tile_indices[cta_id][t]);
      IdType qo_ptr = cta_qo_indptr[cta_id][t];
      IdType kv_ptr = cta_kv_indptr[cta_id][t];
      IdType qlen = cta_qo_len[cta_id][t];
      IdType kvlen = cta_kv_len[cta_id][t];
      int head = static_cast<int>(cta_head_indices[cta_id][t]);
      printf("  work[%d]: batch=%d qo_tile=%d qo_indptr=%lld kv_indptr=%lld qo_len=%lld kv_len=%lld head=%d\n",
             work_start + t, batch, qo_tile, static_cast<long long>(qo_ptr),
             static_cast<long long>(kv_ptr), static_cast<long long>(qlen),
             static_cast<long long>(kvlen), head);
    }
  }
  printf("==============================================\n\n");
#endif
  auto qo_tile_indices_vec = flatten(cta_qo_tile_indices, total_num_works);
  auto qo_indptr_vec = flatten(cta_qo_indptr, total_num_works);
  auto kv_indptr_vec = flatten(cta_kv_indptr, total_num_works);
  auto qo_len_vec = flatten(cta_qo_len, total_num_works);
  auto kv_len_vec = flatten(cta_kv_len, total_num_works);
  auto head_indices_vec = flatten(cta_head_indices, total_num_works);
  auto batch_indices_vec = flatten(cta_batch_indices, total_num_works);

  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
  int max_total_num_works;

  if (enable_cuda_graph) {
    max_total_num_works = plan_info.same_schedule_for_all_heads
                              ? max_num_works_per_head
                              : max_num_works_per_head * num_qo_heads;
  } else {
    max_total_num_works = total_num_works;
  }

  plan_info.qo_tile_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "batch_prefill_sm90_qo_tile_indices");
  plan_info.qo_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "batch_prefill_sm90_qo_offset");
  plan_info.kv_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "batch_prefill_sm90_kv_offset");
  plan_info.qo_len_offset = int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works,
                                                               16, "batch_prefill_sm90_qo_len");
  plan_info.kv_len_offset = int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works,
                                                               16, "batch_prefill_sm90_kv_len");
  plan_info.head_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "batch_prefill_sm90_head_indices");
  plan_info.work_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * (num_total_ctas + 1), 16, "batch_prefill_sm90_work_indptr");
  plan_info.batch_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "batch_prefill_sm90_batch_indices");

  // Per-CTA mech array: mode=0 (mech1) if max KV length for that CTA >= 128, else mode=1 (mech2)
  std::vector<uint8_t> cta_mech_mode_vec(num_total_ctas, 0);
  for (uint32_t cta_idx = 0; cta_idx < num_total_ctas; ++cta_idx) {
    IdType max_kv = 0;
    for (IdType kv_len_val : cta_kv_len[cta_idx]) {
      max_kv = std::max(max_kv, kv_len_val);
    }
    cta_mech_mode_vec[cta_idx] = (max_kv >= 128) ? 0 : 1;
  }

  // Per-CTA dummy flag: 1 only if this CTA has dummy work and no real work (cluster padding).
  // CTAs with no work at all (neither real nor dummy) get dummy=0, valid_work=0.
  std::vector<uint8_t> cta_is_dummy_vec(num_total_ctas, 0);
  for (uint32_t cta_idx = 0; cta_idx < num_total_ctas; ++cta_idx) {
    cta_is_dummy_vec[cta_idx] = (!cta_has_real_work[cta_idx] && cta_has_dummy_work[cta_idx]) ? 1 : 0;
  }

  // Per-CTA valid work:
  // - Real CTAs: 1 if has any work item with kv_len > 0, else 0
  // - Dummy CTAs: 1 (must participate in cluster barriers, kernel uses dummy flag to skip compute)
  std::vector<uint8_t> cta_valid_work_vec(num_total_ctas, 0);
  for (uint32_t cta_idx = 0; cta_idx < num_total_ctas; ++cta_idx) {
    if (cta_is_dummy_vec[cta_idx]) {
      cta_valid_work_vec[cta_idx] = 1;  // dummy must enter kernel for cluster.sync()
      continue;
    }
    bool has_valid = false;
    for (IdType kv_len_val : cta_kv_len[cta_idx]) {
      if (kv_len_val > 0) {
        has_valid = true;
        break;
      }
    }
    cta_valid_work_vec[cta_idx] = has_valid ? 1 : 0;
  }

  // Per-CTA mech mode: dummy CTAs are forced to mech2 (mode=1)
  for (uint32_t cta_idx = 0; cta_idx < num_total_ctas; ++cta_idx) {
    if (cta_is_dummy_vec[cta_idx]) {
      cta_mech_mode_vec[cta_idx] = 1;  // mech2
    }
  }

#ifdef FLASHINFER_DEBUG_SCHEDULER
  printf("\n--- Per-CTA mech mode (mech1=KV>=128, mech2=KV<128) ---\n");
  printf("num_ctas_launched=%d, num_total_ctas=%d, num_sm90_ctas=%d\n",
         plan_info.num_ctas_launched, num_total_ctas, num_sm90_ctas);
  for (uint32_t cta_idx = 0; cta_idx < std::min((uint32_t)plan_info.num_ctas_launched, (uint32_t)num_total_ctas); ++cta_idx) {
    IdType max_kv = 0;
    for (IdType kv_len_val : cta_kv_len[cta_idx]) {
      max_kv = std::max(max_kv, kv_len_val);
    }
    uint8_t mode = cta_mech_mode_vec[cta_idx];
    uint8_t valid = cta_valid_work_vec[cta_idx];
    uint8_t dummy = cta_is_dummy_vec[cta_idx];
    printf("  CTA%u: max_kv=%lld -> %s (mode=%u) valid_work=%u dummy=%u\n", cta_idx,
           static_cast<long long>(max_kv), mode == 0 ? "mech1" : "mech2", mode, valid, dummy);
  }
  printf("==============================================\n\n");
#endif

  plan_info.cta_mech_mode_offset = int_allocator.aligned_alloc_offset(
      sizeof(uint8_t) * num_total_ctas, 16, "batch_prefill_sm90_cta_mech_mode");
  if (use_pat_scheduling) {
    // PAT scheduling: set cta_valid_work to 0 so the kernel skips
    // the per-CTA mech mode logic and uses global kvsplit_mode/mech2_mode (both false)
    plan_info.cta_valid_work_offset = 0;
  } else {
    plan_info.cta_valid_work_offset = int_allocator.aligned_alloc_offset(
        sizeof(uint8_t) * num_total_ctas, 16, "batch_prefill_sm90_cta_valid_work");
  }
  plan_info.cta_is_dummy_offset = int_allocator.aligned_alloc_offset(
      sizeof(uint8_t) * num_total_ctas, 16, "batch_prefill_sm90_cta_is_dummy");

  IdType* qo_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.qo_tile_indices_offset);
  IdType* qo_offset_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.qo_indptr_offset);
  IdType* kv_offset_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_indptr_offset);
  IdType* qo_len_h = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.qo_len_offset);
  IdType* kv_len_h = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_len_offset);
  IdType* head_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.head_indices_offset);
  IdType* work_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.work_indptr_offset);
  IdType* batch_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.batch_indices_offset);

  std::copy(qo_tile_indices_vec.begin(), qo_tile_indices_vec.end(), qo_tile_indices_h);
  std::copy(qo_indptr_vec.begin(), qo_indptr_vec.end(), qo_offset_h);
  std::copy(kv_indptr_vec.begin(), kv_indptr_vec.end(), kv_offset_h);
  std::copy(qo_len_vec.begin(), qo_len_vec.end(), qo_len_h);
  std::copy(kv_len_vec.begin(), kv_len_vec.end(), kv_len_h);
  std::copy(head_indices_vec.begin(), head_indices_vec.end(), head_indices_h);
  std::copy(work_indptr_vec.begin(), work_indptr_vec.end(), work_indptr_h);
  std::copy(batch_indices_vec.begin(), batch_indices_vec.end(), batch_indices_h);
  uint8_t* cta_mech_mode_h =
      GetPtrFromBaseOffset<uint8_t>(page_locked_int_buffer, plan_info.cta_mech_mode_offset);
  std::copy(cta_mech_mode_vec.begin(), cta_mech_mode_vec.end(), cta_mech_mode_h);
  uint8_t* cta_valid_work_h =
      GetPtrFromBaseOffset<uint8_t>(page_locked_int_buffer, plan_info.cta_valid_work_offset);
  std::copy(cta_valid_work_vec.begin(), cta_valid_work_vec.end(), cta_valid_work_h);

  uint8_t* cta_is_dummy_h =
      GetPtrFromBaseOffset<uint8_t>(page_locked_int_buffer, plan_info.cta_is_dummy_offset);
  std::copy(cta_is_dummy_vec.begin(), cta_is_dummy_vec.end(), cta_is_dummy_h);

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));
  return cudaSuccess;
}

template <uint32_t NUM_TASKS>
struct HolisticPlanInfo {
  int64_t num_blks_x;
  int64_t num_blks_y;
  struct {
    int64_t q_indptr_offset;
    int64_t kv_indptr_offset;
    int64_t partial_indptr_offset;
    int64_t q_len_offset;
    int64_t kv_len_offset;
    int64_t q_start_offset;
    int64_t kv_start_offset;
    int64_t kv_end_offset;
    int64_t kv_head_idx_offset;
    int64_t work_indptr_offset;
  } tasks[NUM_TASKS];
  int64_t len_kv_chunk_offset;
  int64_t partial_o_offset;
  int64_t partial_lse_offset;
  int64_t merge_indptr_offset;
  int64_t merge_o_indices_offset;
  int64_t num_qo_len_offset;

  static constexpr uint32_t NUM_TASK_ARGS = 10;
  static constexpr uint32_t NUM_SHARED_ARGS = 8;

  std::vector<int64_t> ToVector() const {
    std::vector<int64_t> vec;
    vec.push_back(num_blks_x);
    vec.push_back(num_blks_y);
    for (uint32_t i = 0; i < NUM_TASKS; ++i) {
      vec.push_back(tasks[i].q_indptr_offset);
      vec.push_back(tasks[i].kv_indptr_offset);
      vec.push_back(tasks[i].partial_indptr_offset);
      vec.push_back(tasks[i].q_len_offset);
      vec.push_back(tasks[i].kv_len_offset);
      vec.push_back(tasks[i].q_start_offset);
      vec.push_back(tasks[i].kv_start_offset);
      vec.push_back(tasks[i].kv_end_offset);
      vec.push_back(tasks[i].kv_head_idx_offset);
      vec.push_back(tasks[i].work_indptr_offset);
    }
    vec.push_back(len_kv_chunk_offset);
    vec.push_back(partial_o_offset);
    vec.push_back(partial_lse_offset);
    vec.push_back(merge_indptr_offset);
    vec.push_back(merge_o_indices_offset);
    vec.push_back(num_qo_len_offset);
    return vec;
  }

  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != NUM_SHARED_ARGS + NUM_TASKS * NUM_TASK_ARGS) {
      std::ostringstream err_msg;
      err_msg << "HolisticPlanInfo::FromVector: vec.size() should be "
              << NUM_SHARED_ARGS + NUM_TASKS * NUM_TASK_ARGS << ", but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    num_blks_x = vec[0];
    num_blks_y = vec[1];
    for (uint32_t i = 0; i < NUM_TASKS; ++i) {
      tasks[i].q_indptr_offset = vec[2 + i * NUM_TASK_ARGS + 0];
      tasks[i].kv_indptr_offset = vec[2 + i * NUM_TASK_ARGS + 1];
      tasks[i].partial_indptr_offset = vec[2 + i * NUM_TASK_ARGS + 2];
      tasks[i].q_len_offset = vec[2 + i * NUM_TASK_ARGS + 3];
      tasks[i].kv_len_offset = vec[2 + i * NUM_TASK_ARGS + 4];
      tasks[i].q_start_offset = vec[2 + i * NUM_TASK_ARGS + 5];
      tasks[i].kv_start_offset = vec[2 + i * NUM_TASK_ARGS + 6];
      tasks[i].kv_end_offset = vec[2 + i * NUM_TASK_ARGS + 7];
      tasks[i].kv_head_idx_offset = vec[2 + i * NUM_TASK_ARGS + 8];
      tasks[i].work_indptr_offset = vec[2 + i * NUM_TASK_ARGS + 9];
    }
    len_kv_chunk_offset = vec[2 + NUM_TASKS * NUM_TASK_ARGS];
    partial_o_offset = vec[3 + NUM_TASKS * NUM_TASK_ARGS];
    partial_lse_offset = vec[4 + NUM_TASKS * NUM_TASK_ARGS];
    merge_indptr_offset = vec[5 + NUM_TASKS * NUM_TASK_ARGS];
    merge_o_indices_offset = vec[6 + NUM_TASKS * NUM_TASK_ARGS];
    num_qo_len_offset = vec[7 + NUM_TASKS * NUM_TASK_ARGS];
  }
};

template <typename IdType>
inline cudaError_t TwoStageHolisticPlan(void* float_buffer, size_t float_workspace_size_in_bytes,
                                        void* int_buffer, void* page_locked_int_buffer,
                                        size_t int_workspace_size_in_bytes,
                                        HolisticPlanInfo<2>& plan_info, IdType* qo_indptr_h,
                                        IdType* kv_indptr_h, IdType* kv_len_arr_h,
                                        uint32_t batch_size, uint32_t num_qo_heads,
                                        uint32_t num_kv_heads, uint32_t head_dim, bool causal,
                                        cudaStream_t stream) {
  constexpr uint32_t NUM_TASKS = 2;
  const uint32_t CTA_TILE_Q_SIZES[NUM_TASKS] = {128, 16};
  int num_sm = 0;
  int dev_id = 0;

  uint32_t gqa_group_size = num_qo_heads / num_kv_heads;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));

  if (head_dim >= 256) {
    // NOTE (Yilong): optimize this code path
    // constraint gridDim due to cooperative group
    num_sm *= 1;
  } else {
    // NOTE(Zihao): two cta per sm
    num_sm *= 2;
  }

  // step 0. determine the number of blocks in x and y dimensions
  std::vector<std::tuple<int, int, int>> idx_qo_kv_len_vec[NUM_TASKS];
  for (uint32_t i = 0; i < batch_size; ++i) {
    if (qo_indptr_h[i + 1] - qo_indptr_h[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "qo_indptr[" << i + 1 << "]" << qo_indptr_h[i + 1] << " - qo_indptr[" << i << "]"
              << qo_indptr_h[i] << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }

    int qo_len = qo_indptr_h[i + 1] - qo_indptr_h[i];
    int packed_qo_len = qo_len * gqa_group_size;
    int kv_len = kv_len_arr_h[i];

    if (packed_qo_len > CTA_TILE_Q_SIZES[1]) {
      idx_qo_kv_len_vec[0].push_back({i, qo_len, kv_len});
    } else {
      idx_qo_kv_len_vec[1].push_back({i, qo_len, kv_len});
    }
  }

  int cluster_size = 1;
  int num_clusters = num_sm / cluster_size;
  plan_info.num_blks_x = cluster_size;
  plan_info.num_blks_y = num_clusters;

  auto f = [](int x) {
    if (x <= 128) {
      // This aligns with CTA_TILE_KV in persistent mainloop
      // NOTE (Yilong): Optimize here for smaller batch/seqlen scenarios
      return 128;
    }
    return ceil_div(x, 256) * 256;
  };

  MinHeap cluster_cost_heap(num_clusters);
  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);

  // NOTE(Zihao): adjust it later
  const int max_total_num_works = 65536;
  const int max_num_kv_splits =
      4 * num_clusters * cluster_size * (CTA_TILE_Q_SIZES[0] + CTA_TILE_Q_SIZES[1]);

  // calculate kv_len_limit first, considering all workloads
  int64_t total_kv_lens = 0;
  for (uint32_t task = 0; task < NUM_TASKS; ++task) {
    int cluster_tile_q = CTA_TILE_Q_SIZES[task] * cluster_size;
    for (auto& [_, qo_len, kv_len] : idx_qo_kv_len_vec[task]) {
      int packed_qo_len = qo_len * gqa_group_size;
      int num_qo_tiles = ceil_div(packed_qo_len, cluster_tile_q);
      for (int qo_tile_idx = num_qo_tiles - 1; qo_tile_idx >= 0; --qo_tile_idx) {
        int effective_kv_len =
            causal ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx, cluster_tile_q, num_qo_tiles,
                                          gqa_group_size)
                   : kv_len;
        total_kv_lens += effective_kv_len;
      }
    }
  }

  // used for remapping the output offsets
  // layout [packed_qo_len x num_kv_tiles, num_kv_heads, head_dim]
  int partial_o_nnz = 0;
  std::vector<IdType> merge_indptr, merge_o_indices, num_expand_qo_len_vec;
  std::vector<IdType> cluster_len_kv_chunk(NUM_TASKS, 0);
  merge_indptr.push_back(partial_o_nnz);
  for (uint32_t task = 0; task < NUM_TASKS; ++task) {
    int cluster_tile_q = CTA_TILE_Q_SIZES[task] * cluster_size;
    int kv_len_limit = f(std::max(ceil_div(total_kv_lens * num_kv_heads, num_clusters), 1L));
    if (cluster_tile_q >= 64) {
      // chunked-prefill workloads are much more expensive than decode
      // so we use a smaller kv_len_limit for chunked-prefill workloads
      kv_len_limit /= std::min(num_kv_heads, 2U);
    }
    cluster_len_kv_chunk[task] = kv_len_limit;
    std::vector<std::vector<IdType>> cluster_q_indptr(num_clusters, std::vector<IdType>()),
        cluster_kv_indptr(num_clusters, std::vector<IdType>()),
        cluster_q_len(num_clusters, std::vector<IdType>()),
        cluster_kv_len(num_clusters, std::vector<IdType>()),
        cluster_q_start(num_clusters, std::vector<IdType>()),
        cluster_kv_start(num_clusters, std::vector<IdType>()),
        cluster_kv_end(num_clusters, std::vector<IdType>()),
        cluster_kv_head_idx(num_clusters, std::vector<IdType>()),
        cluster_partial_indptr(num_clusters, std::vector<IdType>());

    for (auto& [i, qo_len, kv_len] : idx_qo_kv_len_vec[task]) {
      int packed_qo_len = qo_len * gqa_group_size;
      int num_qo_tiles = ceil_div(packed_qo_len, cluster_tile_q);
      // NOTE (Yilong): this ordering correspoinds to the layout of reduction kernel
      for (int qo_tile_idx = 0; qo_tile_idx < num_qo_tiles; ++qo_tile_idx) {
        int remaining_len = causal
                                ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx, cluster_tile_q,
                                                       num_qo_tiles, gqa_group_size)
                                : kv_len;
        int kv_start = 0;
        bool split_kv = remaining_len > kv_len_limit;
        int num_kv_tiles = split_kv ? ceil_div(remaining_len, kv_len_limit) : 1;
        int row_tile_size = std::min(cluster_tile_q, packed_qo_len - qo_tile_idx * cluster_tile_q);
        bool zero_kv_len = (remaining_len == 0);
        while (remaining_len > 0 || zero_kv_len) {
          int actual_len = std::min(remaining_len, kv_len_limit);
          for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
            auto [cluster_idx, accum_cost] = cluster_cost_heap.pop();
            cluster_cost_heap.insert(
                {cluster_idx, accum_cost + cost_function(cluster_tile_q, actual_len)});
            cluster_q_len[cluster_idx].push_back(qo_len);
            cluster_kv_len[cluster_idx].push_back(kv_len);
            cluster_q_indptr[cluster_idx].push_back(qo_indptr_h[i]);
            cluster_kv_indptr[cluster_idx].push_back(kv_indptr_h[i]);

            // use kv_chunk to rematerize num_kv_tiles and kv_tile_idx
            cluster_partial_indptr[cluster_idx].push_back(partial_o_nnz);

            cluster_q_start[cluster_idx].push_back(qo_tile_idx * cluster_tile_q);
            cluster_kv_start[cluster_idx].push_back(kv_start);
            cluster_kv_end[cluster_idx].push_back(kv_start + actual_len);
            cluster_kv_head_idx[cluster_idx].push_back(kv_head_idx);
          }
          remaining_len -= actual_len;
          zero_kv_len = (remaining_len == 0);
          kv_start += actual_len;
          if (zero_kv_len) {
            break;
          }
        }
        if (split_kv) {
          // non-split kv is directly written through
          for (int row = 0; row < row_tile_size; ++row) {
            merge_indptr.push_back(merge_indptr.back() + num_kv_tiles);
            merge_o_indices.push_back(qo_indptr_h[i] +
                                      (qo_tile_idx * cluster_tile_q + row) / gqa_group_size);
          }
          partial_o_nnz += row_tile_size * num_kv_tiles;
        }
      }
    }

    std::vector<IdType> work_indptr_vec(num_clusters + 1, 0);
    for (uint32_t i = 0; i < num_clusters; ++i) {
      work_indptr_vec[i + 1] = work_indptr_vec[i] + cluster_q_indptr[i].size();
    }
    int total_num_works = work_indptr_vec.back();
    if (total_num_works > max_total_num_works) {
      std::ostringstream err_msg;
      err_msg << "total_num_works (#q tiles * #kv tiles) " << total_num_works
              << " exceeds max_total_num_works " << max_total_num_works;
      FLASHINFER_ERROR(err_msg.str());
    }
    auto q_indptr_vec = flatten(cluster_q_indptr, total_num_works);
    auto kv_indptr_vec = flatten(cluster_kv_indptr, total_num_works);
    auto partial_indptr_vec = flatten(cluster_partial_indptr, total_num_works);
    auto q_len_vec = flatten(cluster_q_len, total_num_works);
    auto kv_len_vec = flatten(cluster_kv_len, total_num_works);
    auto q_start_vec = flatten(cluster_q_start, total_num_works);
    auto kv_start_vec = flatten(cluster_kv_start, total_num_works);
    auto kv_end_vec = flatten(cluster_kv_end, total_num_works);
    auto kv_head_idx_vec = flatten(cluster_kv_head_idx, total_num_works);

    plan_info.tasks[task].q_indptr_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "q_indptr");
    plan_info.tasks[task].kv_indptr_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "kv_indptr");
    plan_info.tasks[task].partial_indptr_offset = int_allocator.aligned_alloc_offset(
        sizeof(IdType) * max_total_num_works, 16, "partial_indptr");
    plan_info.tasks[task].q_len_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "q_len");
    plan_info.tasks[task].kv_len_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "kv_len");
    plan_info.tasks[task].q_start_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "q_start");
    plan_info.tasks[task].kv_start_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "kv_start");
    plan_info.tasks[task].kv_end_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "kv_end");
    plan_info.tasks[task].kv_head_idx_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "kv_head_idx");
    plan_info.tasks[task].work_indptr_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "work_indptr");

    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].q_indptr_offset,
                           q_indptr_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].kv_indptr_offset,
                           kv_indptr_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].partial_indptr_offset,
                           partial_indptr_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].q_len_offset, q_len_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].kv_len_offset, kv_len_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].q_start_offset,
                           q_start_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].kv_start_offset,
                           kv_start_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].kv_end_offset, kv_end_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].kv_head_idx_offset,
                           kv_head_idx_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].work_indptr_offset,
                           work_indptr_vec);
  }
  plan_info.len_kv_chunk_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * NUM_TASKS, 16, "len_kv_chunk");
  CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.len_kv_chunk_offset,
                         cluster_len_kv_chunk);

  if (merge_indptr.size() > max_num_kv_splits) {
    std::ostringstream err_msg;
    err_msg << "Number of kv splits " << merge_indptr.size() << " exceeds max buffer size "
            << max_num_kv_splits << ". Please increase the threshold.";
    FLASHINFER_ERROR(err_msg.str());
  }

  // update num_qo_len_vec
  num_expand_qo_len_vec.push_back(merge_indptr.size() - 1);
  // allocate buffer for state merge function
  plan_info.merge_indptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_num_kv_splits, 16, "merge_indptr");
  plan_info.merge_o_indices_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_num_kv_splits, 16, "merge_o_indices");
  plan_info.num_qo_len_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType), 16, "num_qo_len_offset");
  // copy data to paged cpu buffer
  CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.merge_indptr_offset, merge_indptr);
  CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.merge_o_indices_offset, merge_o_indices);
  CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.num_qo_len_offset,
                         num_expand_qo_len_vec);

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));
  constexpr size_t sizeof_dtype_o = 2;  // NOTE (Yilong): assume fp16

  // Note(Yilong): times num_kv_heads as it is not counted in partial_o_nnz
  AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
  plan_info.partial_o_offset = float_allocator.aligned_alloc_offset(
      max_num_kv_splits * sizeof_dtype_o * head_dim * num_kv_heads, 16, "holistic_partial_o");
  plan_info.partial_lse_offset = float_allocator.aligned_alloc_offset(
      max_num_kv_splits * sizeof(float) * num_kv_heads, 16, "holistic_partial_lse");

  return cudaSuccess;
}

struct MLAPlanInfo {
  int64_t num_blks_x;
  int64_t num_blks_y;
  int64_t q_indptr_offset;
  int64_t kv_indptr_offset;
  int64_t partial_indptr_offset;
  int64_t merge_packed_offset_start_offset;
  int64_t merge_packed_offset_end_offset;
  int64_t merge_partial_packed_offset_start_offset;
  int64_t merge_partial_packed_offset_end_offset;
  int64_t merge_partial_stride_offset;
  int64_t q_len_offset;
  int64_t kv_len_offset;
  int64_t q_start_offset;
  int64_t kv_start_offset;
  int64_t kv_end_offset;
  int64_t work_indptr_offset;
  int64_t partial_o_offset;
  int64_t partial_lse_offset;

  std::vector<int64_t> ToVector() const {
    return {num_blks_x,
            num_blks_y,
            q_indptr_offset,
            kv_indptr_offset,
            partial_indptr_offset,
            merge_packed_offset_start_offset,
            merge_packed_offset_end_offset,
            merge_partial_packed_offset_start_offset,
            merge_partial_packed_offset_end_offset,
            merge_partial_stride_offset,
            q_len_offset,
            kv_len_offset,
            q_start_offset,
            kv_start_offset,
            kv_end_offset,
            work_indptr_offset,
            partial_o_offset,
            partial_lse_offset};
  }

  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 18) {
      std::ostringstream err_msg;
      err_msg << "MLAPlanInfo::FromVector: vec.size() should be 18, but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    num_blks_x = vec[0];
    num_blks_y = vec[1];
    q_indptr_offset = vec[2];
    kv_indptr_offset = vec[3];
    partial_indptr_offset = vec[4];
    merge_packed_offset_start_offset = vec[5];
    merge_packed_offset_end_offset = vec[6];
    merge_partial_packed_offset_start_offset = vec[7];
    merge_partial_packed_offset_end_offset = vec[8];
    merge_partial_stride_offset = vec[9];
    q_len_offset = vec[10];
    kv_len_offset = vec[11];
    q_start_offset = vec[12];
    kv_start_offset = vec[13];
    kv_end_offset = vec[14];
    work_indptr_offset = vec[15];
    partial_o_offset = vec[16];
    partial_lse_offset = vec[17];
  }
};

template <typename IdType>
inline cudaError_t MLAPlan(void* float_buffer, size_t float_workspace_size_in_bytes,
                           void* int_buffer, void* page_locked_int_buffer,
                           size_t int_workspace_size_in_bytes, MLAPlanInfo& plan_info,
                           IdType* qo_indptr_h, IdType* kv_indptr_h, IdType* kv_len_arr_h,
                           uint32_t batch_size, uint32_t num_heads, uint32_t head_dim_o,
                           bool causal, cudaStream_t stream) {
  int num_sm = 0;
  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));

  // step 0. determine the number of blocks in x and y dimensions
  int accum_packed_qo_len = 0;
  std::vector<std::tuple<int, int, int>> idx_qo_kv_len_vec;
  for (uint32_t i = 0; i < batch_size; ++i) {
    if (qo_indptr_h[i + 1] - qo_indptr_h[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "qo_indptr[" << i + 1 << "]" << qo_indptr_h[i + 1] << " - qo_indptr[" << i << "]"
              << qo_indptr_h[i] << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }

    int qo_len = qo_indptr_h[i + 1] - qo_indptr_h[i];
    int packed_qo_len = qo_len * num_heads;
    accum_packed_qo_len += packed_qo_len;

    int kv_len = kv_len_arr_h[i];
    idx_qo_kv_len_vec.push_back({i, qo_len, kv_len});
  }
  int avg_packed_qo_len = accum_packed_qo_len / batch_size;

  int cluster_size;
  if (avg_packed_qo_len > 64) {
    cluster_size = 2;  // two ctas in a cluster
  } else {
    cluster_size = 1;  // one cta in a cluster
  }
  uint32_t num_clusters = num_sm / cluster_size;
  plan_info.num_blks_x = cluster_size;
  plan_info.num_blks_y = num_clusters;
  const int cta_tile_q = 64;
  int cluster_tile_q = cluster_size * cta_tile_q;

  int64_t total_kv_lens = 0;
  for (auto& [_, qo_len, kv_len] : idx_qo_kv_len_vec) {
    int packed_qo_len = qo_len * num_heads;
    int num_qo_tiles = ceil_div(packed_qo_len, cluster_tile_q);
    for (int qo_tile_idx = num_qo_tiles - 1; qo_tile_idx >= 0; --qo_tile_idx) {
      int effective_kv_len = causal ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx,
                                                           cluster_tile_q, num_qo_tiles, num_heads)
                                    : kv_len;
      total_kv_lens += effective_kv_len;
    }
  }

  auto f = [](int x) {
    if (x <= 8) {
      return 32;
    } else if (x <= 16) {
      return 64;
    } else if (x <= 32) {
      return 128;
    } else if (x <= 64) {
      return 192;
    }
    return ceil_div(x, 256) * 256;
  };

  int kv_len_limit = f(std::max(ceil_div(total_kv_lens, num_clusters), 1L));

  // step 1. load-balancing scheduling algorithm
  MinHeap cluster_cost_heap(num_clusters);
  std::vector<std::vector<IdType>> cluster_q_indptr(num_clusters, std::vector<IdType>()),
      cluster_kv_indptr(num_clusters, std::vector<IdType>()),
      cluster_q_len(num_clusters, std::vector<IdType>()),
      cluster_kv_len(num_clusters, std::vector<IdType>()),
      cluster_q_start(num_clusters, std::vector<IdType>()),
      cluster_kv_start(num_clusters, std::vector<IdType>()),
      cluster_kv_end(num_clusters, std::vector<IdType>()),
      cluster_partial_indptr(num_clusters, std::vector<IdType>());

  std::vector<IdType> merge_packed_offset_start(num_sm, 0), merge_packed_offset_end(num_sm, 0),
      merge_partial_packed_offset_start(num_sm, 0), merge_partial_packed_offset_end(num_sm, 0),
      merge_partial_stride(num_sm, 0);

  int merge_cta_counter = 0;
  int partial_o_nnz = 0;

  for (auto& [i, qo_len, kv_len] : idx_qo_kv_len_vec) {
    int packed_qo_len = qo_len * num_heads;
    int num_qo_tiles = ceil_div(packed_qo_len, cluster_tile_q);
    for (int qo_tile_idx = num_qo_tiles - 1; qo_tile_idx >= 0; --qo_tile_idx) {
      int remaining_len = causal ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx, cluster_tile_q,
                                                        num_qo_tiles, num_heads)
                                 : kv_len;
      int kv_start = 0;
      bool split_kv = remaining_len > kv_len_limit;
      int row_tile_size = std::min(cluster_tile_q, packed_qo_len - qo_tile_idx * cluster_tile_q);
      if (split_kv) {
        /*
         * Proof(Zihao): merge_cta_counter <= num_sm (num_sm == num_clusters * cluster_size)
         *
         * Precondition:
         * 1. kv_len_limit * num_clusters >= total_kv_lens == sum(remaining_len)
         * 2. num_qo_chunks <= max((remaining_len * cluster_size) // kv_len_limit, 1)
         * 3. num_qo_tiles_requires_split <= num_clusters

         * Implication:
         * 1. sum(num_qo_chunks) <= max(sum(remaining_len) * cluster_size / kv_len_limit,
         num_qo_tiles_requires_split)
         * 2. sum(num_qo_chunks) <= max(cluster_size * num_clusters, num_qo_tiles_requires_split)
         */
        int num_qo_chunks = std::max(remaining_len * cluster_size / kv_len_limit, 1);
        // row_chunk_size * num_qo_chunks >= row_tile_size
        int row_chunk_size = ceil_div(row_tile_size, num_qo_chunks);
        int current_q_tile_end =
            std::min(cluster_tile_q, packed_qo_len - qo_tile_idx * cluster_tile_q);
        for (int offset_start = 0; offset_start < row_tile_size; offset_start += row_chunk_size) {
          merge_packed_offset_start[merge_cta_counter] =
              qo_indptr_h[i] * num_heads + qo_tile_idx * cluster_tile_q + offset_start;
          merge_packed_offset_end[merge_cta_counter] =
              qo_indptr_h[i] * num_heads + qo_tile_idx * cluster_tile_q +
              std::min(offset_start + row_chunk_size, current_q_tile_end);
          merge_partial_packed_offset_start[merge_cta_counter] = partial_o_nnz + offset_start;
          merge_partial_packed_offset_end[merge_cta_counter] =
              partial_o_nnz + ceil_div(remaining_len, kv_len_limit) * row_tile_size;
          merge_partial_stride[merge_cta_counter] = row_tile_size;
          merge_cta_counter++;
        }
      }
      bool zero_kv_len = (remaining_len == 0);
      while (remaining_len > 0 || zero_kv_len) {
        auto [cluster_idx, accum_cost] = cluster_cost_heap.pop();
        int actual_len = std::min(remaining_len, kv_len_limit);
        cluster_cost_heap.insert(
            {cluster_idx, accum_cost + cost_function(cluster_tile_q, actual_len)});
        cluster_q_len[cluster_idx].push_back(qo_len);
        cluster_kv_len[cluster_idx].push_back(kv_len);
        cluster_q_indptr[cluster_idx].push_back(qo_indptr_h[i]);
        cluster_kv_indptr[cluster_idx].push_back(kv_indptr_h[i]);
        if (split_kv) {
          cluster_partial_indptr[cluster_idx].push_back(partial_o_nnz);
          partial_o_nnz += row_tile_size;
        } else {
          cluster_partial_indptr[cluster_idx].push_back(-1);
        }
        cluster_q_start[cluster_idx].push_back(qo_tile_idx * cluster_tile_q);
        cluster_kv_start[cluster_idx].push_back(kv_start);
        cluster_kv_end[cluster_idx].push_back(kv_start + actual_len);
        remaining_len -= actual_len;
        kv_start += actual_len;
        if (zero_kv_len) break;
      }
    }
  }

  FLASHINFER_CHECK(merge_cta_counter <= num_sm,
                   "Internal Error: merge_cta_counter should be less than or equal to num_sm, "
                   "please report this bug to the developers");

  int max_total_num_works = 16384;  // NOTE(Zihao): adjust it later

  std::vector<IdType> work_indptr_vec(num_clusters + 1, 0);
  for (uint32_t i = 0; i < num_clusters; ++i) {
    work_indptr_vec[i + 1] = work_indptr_vec[i] + cluster_q_indptr[i].size();
  }
  int total_num_works = work_indptr_vec.back();
  auto q_indptr_vec = flatten(cluster_q_indptr, total_num_works);
  auto kv_indptr_vec = flatten(cluster_kv_indptr, total_num_works);
  auto partial_indptr_vec = flatten(cluster_partial_indptr, total_num_works);
  auto q_len_vec = flatten(cluster_q_len, total_num_works);
  auto kv_len_vec = flatten(cluster_kv_len, total_num_works);
  auto q_start_vec = flatten(cluster_q_start, total_num_works);
  auto kv_start_vec = flatten(cluster_kv_start, total_num_works);
  auto kv_end_vec = flatten(cluster_kv_end, total_num_works);

  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
  plan_info.q_indptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_q_indptr");
  plan_info.kv_indptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_kv_indptr");
  plan_info.partial_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "mla_partial_indptr");
  plan_info.merge_packed_offset_start_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * num_sm, 16, "mla_merge_packed_offset_start");
  plan_info.merge_packed_offset_end_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * num_sm, 16, "mla_merge_packed_offset_end");
  plan_info.merge_partial_packed_offset_start_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * num_sm, 16, "mla_merge_partial_packed_offset_start");
  plan_info.merge_partial_packed_offset_end_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * num_sm, 16, "mla_merge_partial_packed_offset_end");
  plan_info.merge_partial_stride_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * num_sm, 16, "mla_merge_partial_stride");
  plan_info.q_len_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_q_len");
  plan_info.kv_len_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_kv_len");
  plan_info.q_start_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_q_start");
  plan_info.kv_start_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_kv_start");
  plan_info.kv_end_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_kv_end");
  plan_info.work_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "mla_work_indptr");

  IdType* cluster_q_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.q_indptr_offset);
  IdType* cluster_kv_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_indptr_offset);
  IdType* cluster_partial_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.partial_indptr_offset);
  IdType* cluster_merge_packed_offset_start_h = GetPtrFromBaseOffset<IdType>(
      page_locked_int_buffer, plan_info.merge_packed_offset_start_offset);
  IdType* cluster_merge_packed_offset_end_h = GetPtrFromBaseOffset<IdType>(
      page_locked_int_buffer, plan_info.merge_packed_offset_end_offset);
  IdType* cluster_merge_partial_packed_offset_start_h = GetPtrFromBaseOffset<IdType>(
      page_locked_int_buffer, plan_info.merge_partial_packed_offset_start_offset);
  IdType* cluster_merge_partial_packed_offset_end_h = GetPtrFromBaseOffset<IdType>(
      page_locked_int_buffer, plan_info.merge_partial_packed_offset_end_offset);
  IdType* cluster_merge_partial_stride_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.merge_partial_stride_offset);
  IdType* cluster_q_len_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.q_len_offset);
  IdType* cluster_kv_len_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_len_offset);
  IdType* cluster_q_start_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.q_start_offset);
  IdType* cluster_kv_start_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_start_offset);
  IdType* cluster_kv_end_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_end_offset);
  IdType* cluster_work_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.work_indptr_offset);

  std::copy(q_indptr_vec.begin(), q_indptr_vec.end(), cluster_q_indptr_h);
  std::copy(kv_indptr_vec.begin(), kv_indptr_vec.end(), cluster_kv_indptr_h);
  std::copy(partial_indptr_vec.begin(), partial_indptr_vec.end(), cluster_partial_indptr_h);
  std::copy(merge_packed_offset_start.begin(), merge_packed_offset_start.end(),
            cluster_merge_packed_offset_start_h);
  std::copy(merge_packed_offset_end.begin(), merge_packed_offset_end.end(),
            cluster_merge_packed_offset_end_h);
  std::copy(merge_partial_packed_offset_start.begin(), merge_partial_packed_offset_start.end(),
            cluster_merge_partial_packed_offset_start_h);
  std::copy(merge_partial_packed_offset_end.begin(), merge_partial_packed_offset_end.end(),
            cluster_merge_partial_packed_offset_end_h);
  std::copy(merge_partial_stride.begin(), merge_partial_stride.end(),
            cluster_merge_partial_stride_h);
  std::copy(q_len_vec.begin(), q_len_vec.end(), cluster_q_len_h);
  std::copy(kv_len_vec.begin(), kv_len_vec.end(), cluster_kv_len_h);
  std::copy(q_start_vec.begin(), q_start_vec.end(), cluster_q_start_h);
  std::copy(kv_start_vec.begin(), kv_start_vec.end(), cluster_kv_start_h);
  std::copy(kv_end_vec.begin(), kv_end_vec.end(), cluster_kv_end_h);
  std::copy(work_indptr_vec.begin(), work_indptr_vec.end(), cluster_work_indptr_h);

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));

  constexpr size_t sizeof_dtype_o = 2;
  AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
  plan_info.partial_o_offset = float_allocator.aligned_alloc_offset(
      2 * num_clusters * cluster_tile_q * sizeof_dtype_o * head_dim_o, 16, "mla_partial_o");
  plan_info.partial_lse_offset = float_allocator.aligned_alloc_offset(
      2 * num_clusters * cluster_tile_q * sizeof(float), 16, "mla_partial_lse");

  return cudaSuccess;
}

}  // namespace flashinfer
#endif  // FLASHINFER_ATTENTION_SCHEDULER_CUH_

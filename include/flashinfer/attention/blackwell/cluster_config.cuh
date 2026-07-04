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
 */
#ifndef FLASHINFER_ATTENTION_BLACKWELL_CLUSTER_CONFIG_CUH_
#define FLASHINFER_ATTENTION_BLACKWELL_CLUSTER_CONFIG_CUH_

namespace flashinfer {

// Single source of truth for the SM100 mech1 (DSM split-KV) cluster size.
// Shared by the host planner (flashinfer/prefill.py mirrors this value), the
// csrc dispatch, and the kernel launch so cluster grouping, grid rounding, and
// clusterDim always agree. Bring-up value is 2; target is 8 (portable max).
constexpr int kSm100ClusterSize = 2;

// Minimum kv_len for a work item to activate mech1 KV splitting; below this
// the replicated ranks skip the item and rank 0 runs the vanilla path.
constexpr int kMech1KvLenThreshold = 4096;

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_BLACKWELL_CLUSTER_CONFIG_CUH_

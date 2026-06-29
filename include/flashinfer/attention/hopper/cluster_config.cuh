/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_CLUSTER_CONFIG_CUH_
#define FLASHINFER_ATTENTION_HOPPER_CLUSTER_CONFIG_CUH_

namespace flashinfer {

// Physical thread-block cluster size (cudaLaunchAttributeClusterDimension.x).
// Single source of truth: must match the value used both in the host-side
// scheduler (PrefillSM90Plan in attention/scheduler.cuh) and in the launch
// site that sets cudaLaunchAttributeClusterDimension (attention/hopper/prefill_sm90.cuh).
constexpr int kClusterSize = 8;

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_CLUSTER_CONFIG_CUH_

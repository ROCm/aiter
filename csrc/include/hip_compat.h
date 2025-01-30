#pragma once
/*
 * Copyright © Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#endif

#ifndef USE_ROCM
#define WARP_SIZE 32
#else
#define WARP_SIZE warpSize
#endif

#ifndef USE_ROCM
#define VLLM_LDG(arg) __ldg(arg)
#else
#define VLLM_LDG(arg) *(arg)
#endif

#ifndef USE_ROCM
#define VLLM_SHFL_XOR_SYNC(var, lane_mask) \
  __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
  __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)
#else
#define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
#define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
  __shfl_xor(var, lane_mask, width)
#endif

#ifndef USE_ROCM
#define VLLM_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)
#else
#define VLLM_SHFL_SYNC(var, src_lane) __shfl(var, src_lane)
#endif

#ifndef USE_ROCM
#define VLLM_SHFL_DOWN_SYNC(var, lane_delta) \
  __shfl_down_sync(uint32_t(-1), var, lane_delta)
#else
#define VLLM_SHFL_DOWN_SYNC(var, lane_delta) __shfl_down(var, lane_delta)
#endif

#ifndef USE_ROCM
#define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
  hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)
#else
#define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
  hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)
#endif

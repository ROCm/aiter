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
#include "attention_generic.cuh"

#include <stdint.h>
#ifdef ENABLE_FP8
#ifndef USE_ROCM
#include <cuda_fp8.h>
#endif // USE_ROCM
#endif // ENABLE_FP8

namespace vllm
{

  enum class Fp8KVCacheDataType
  {
    kAuto = 0,
    kFp8E4M3 = 1,
    kFp8E5M2 = 2,
  };

  // fp8 vector types for quantization of kv cache
  template <>
  struct Vec<uint8_t, 1>
  {
    using Type = uint8_t;
  };

  template <>
  struct Vec<uint8_t, 2>
  {
    using Type = uint16_t;
  };

  template <>
  struct Vec<uint8_t, 4>
  {
    using Type = uint32_t;
  };

  template <>
  struct Vec<uint8_t, 8>
  {
    using Type = uint2;
  };

} // namespace vllm

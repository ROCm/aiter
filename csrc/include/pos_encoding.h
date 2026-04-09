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
#include "aiter_tensor.h"

void rotary_embedding(const aiter_tensor_t &positions, const aiter_tensor_t &query,
                      const aiter_tensor_t &key, int64_t head_size,
                      const aiter_tensor_t &cos_cache, const aiter_tensor_t &sin_cache, bool is_neox, bool is_nope_first);

void batched_rotary_embedding(const aiter_tensor_t &positions, const aiter_tensor_t &query,
                              const aiter_tensor_t &key, int64_t head_size,
                              const aiter_tensor_t &cos_cache, const aiter_tensor_t &sin_cache, bool is_neox, bool is_nope_first,
                              int64_t rot_dim,
                              const aiter_tensor_t &cos_sin_cache_offsets);

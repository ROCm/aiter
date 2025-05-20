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
#include <torch/extension.h>

// all reduce
using fptr_t = int64_t;
fptr_t init_custom_ar(torch::Tensor &meta, torch::Tensor &rank_data,
                      const std::vector<std::string> &handles,
                      const std::vector<int64_t> &offsets, int64_t rank,
                      bool full_nvlink);
void all_reduce_reg(fptr_t _fa, torch::Tensor &inp, torch::Tensor &out, bool open_fp8_quant);
void all_reduce_unreg(fptr_t _fa, torch::Tensor &inp, torch::Tensor &reg_buffer,
                      torch::Tensor &out);

void dispose(fptr_t _fa);
int64_t meta_size();
void register_buffer(fptr_t _fa, torch::Tensor &t,
                     const std::vector<std::string> &handles,
                     const std::vector<int64_t> &offsets);
std::tuple<torch::Tensor, std::vector<int64_t>> get_graph_buffer_ipc_meta(
    fptr_t _fa);
void register_graph_buffers(fptr_t _fa, const std::vector<std::string> &handles,
                            const std::vector<std::vector<int64_t>> &offsets);
#ifdef USE_ROCM
torch::Tensor allocate_meta_buffer(int64_t size);
torch::Tensor get_meta_buffer_ipc_handle(torch::Tensor &inp);
#endif

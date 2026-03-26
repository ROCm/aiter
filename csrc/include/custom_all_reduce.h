#pragma once
/*
 * Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (C) 2024-2026, The vLLM team.
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
#include <cstdint>
#include <vector>
#include "aiter_tensor.h"

// all reduce
using fptr_t = int64_t;

namespace aiter {

fptr_t init_custom_ar(int64_t meta_ptr,
                      int64_t rank_data_ptr,
                      int64_t rank_data_sz,
                      const std::vector<int64_t>& ipc_handle_ptrs,
                      const std::vector<int64_t>& offsets,
                      int64_t rank,
                      bool fully_connected);
void all_reduce(fptr_t _fa,
                const aiter_tensor_t& inp,
                const aiter_tensor_t& out,
                bool use_new,
                bool open_fp8_quant,
                int64_t reg_inp_ptr,
                int64_t reg_inp_bytes,
                int64_t reg_out_ptr,
                int64_t reg_out_bytes,
                int64_t stream);
void reduce_scatter(fptr_t _fa,
                    const aiter_tensor_t& inp,
                    const aiter_tensor_t& out,
                    int64_t reg_ptr,
                    int64_t reg_bytes,
                    int64_t stream);
void all_gather_reg(fptr_t _fa,
                    const aiter_tensor_t& inp,
                    const aiter_tensor_t& out,
                    int64_t dim,
                    int64_t stream);
void all_gather_unreg(fptr_t _fa,
                      const aiter_tensor_t& inp,
                      int64_t reg_buffer,
                      const aiter_tensor_t& out,
                      int64_t reg_bytes,
                      int64_t dim,
                      int64_t stream);
void fused_allreduce_rmsnorm(fptr_t _fa,
                             const aiter_tensor_t& inp,
                             const aiter_tensor_t& res_inp,
                             const aiter_tensor_t& res_out,
                             const aiter_tensor_t& out,
                             const aiter_tensor_t& w,
                             double eps,
                             int64_t reg_ptr,
                             int64_t reg_bytes,
                             bool use_1stage,
                             int64_t stream);
void fused_allreduce_rmsnorm_quant(fptr_t _fa,
                                   const aiter_tensor_t& inp,
                                   const aiter_tensor_t& res_inp,
                                   const aiter_tensor_t& res_out,
                                   const aiter_tensor_t& out,
                                   const aiter_tensor_t& scale_out,
                                   const aiter_tensor_t& w,
                                   double eps,
                                   int64_t reg_ptr,
                                   int64_t reg_bytes,
                                   bool use_1stage,
                                   int64_t stream);
void dispose(fptr_t _fa);
int64_t meta_size();
void register_input_buffer(fptr_t _fa,
                           int64_t self_ptr,
                           const std::vector<int64_t>& ipc_handle_ptrs,
                           const std::vector<int64_t>& offsets);
void register_output_buffer(fptr_t _fa,
                            int64_t self_ptr,
                            const std::vector<int64_t>& ipc_handle_ptrs,
                            const std::vector<int64_t>& offsets);
int64_t get_graph_buffer_count(fptr_t _fa);
void get_graph_buffer_ipc_meta(fptr_t _fa,
                               int64_t handle_out,
                               int64_t offset_out);
void register_graph_buffers(fptr_t _fa,
                            const std::vector<int64_t>& handle_ptrs,
                            const std::vector<int64_t>& offset_ptrs);
#ifdef USE_ROCM
int64_t allocate_meta_buffer(int64_t size, int64_t stream);
void free_meta_buffer(int64_t ptr);
void get_meta_buffer_ipc_handle(int64_t inp_ptr, int64_t out_handle_ptr);
#endif

} // namespace aiter

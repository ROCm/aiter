#pragma once
/*
 * Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
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
 *
 * gfx1250 (MI450) custom allreduce — pointer-based API (no hipIpc).
 */
#include <cstdint>
#include <vector>
#include "aiter_tensor.h"

using fptr_t = int64_t;

namespace aiter {

// init_custom_ar receives direct device pointers for each rank's meta buffer
// (already mapped by torch's cross-process sharing), not IPC handles.
fptr_t init_custom_ar(int64_t meta_ptr,
                      int64_t rank_data_ptr,
                      int64_t rank_data_sz,
                      const std::vector<int64_t>& all_meta_ptrs,
                      int64_t rank,
                      bool fully_connected);
// IPC transport variant (ROCm >= 7.15): peer buffers shared via hipIpc
// handles+offsets instead of direct pointers. Same gfx1250 kernel.
fptr_t init_custom_ar_ipc(int64_t meta_ptr,
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
                int64_t reg_inp_bytes);
void all_gather(fptr_t _fa,
                const aiter_tensor_t& inp,
                const aiter_tensor_t& out,
                int64_t dim,
                int64_t reg_inp_ptr,
                int64_t reg_inp_bytes);
void p2p_bw_test(fptr_t _fa,
                  const aiter_tensor_t& inp,
                  const aiter_tensor_t& out,
                  int64_t unroll,
                  int64_t threads,
                  int64_t blocks,
                  int64_t reg_inp_ptr,
                  int64_t reg_inp_bytes);
void reduce_scatter(fptr_t _fa,
                    const aiter_tensor_t& inp,
                    const aiter_tensor_t& out,
                    int64_t m, int64_t n, int64_t k,
                    int64_t split_dim,
                    int64_t reg_ptr, int64_t reg_bytes);
void dispose(fptr_t _fa);
int64_t meta_size();
// register_input/output_buffer receive direct device pointers per rank.
void register_input_buffer(fptr_t _fa,
                           int64_t self_ptr,
                           const std::vector<int64_t>& all_ptrs);
void register_output_buffer(fptr_t _fa,
                            int64_t self_ptr,
                            const std::vector<int64_t>& all_ptrs);
// IPC transport variants (ROCm >= 7.15): peer buffers via handles+offsets.
void register_input_buffer_ipc(fptr_t _fa,
                               int64_t self_ptr,
                               const std::vector<int64_t>& ipc_handle_ptrs,
                               const std::vector<int64_t>& offsets);
void register_output_buffer_ipc(fptr_t _fa,
                                int64_t self_ptr,
                                const std::vector<int64_t>& ipc_handle_ptrs,
                                const std::vector<int64_t>& offsets);
int64_t get_graph_buffer_count(fptr_t _fa);
void get_graph_buffer_ptrs(fptr_t _fa, int64_t ptrs_out);
void register_graph_buffers(fptr_t _fa,
                            const std::vector<int64_t>& ptrs_per_rank);
void start_sync_latency(fptr_t _fa, int64_t blocks);
void end_sync_latency(fptr_t _fa, int64_t blocks);
void two_sync_latency(fptr_t _fa, int64_t blocks);

} // namespace aiter

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# Copyright (C) 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

from typing import Optional
import functools
import json
import triton
import triton.language as tl
import torch
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd
from aiter import dtypes

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# @gluon.jit
# def _fwd_grouped_kernel_stage1_n16x4_prefetch_k_paged_64(
#     Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
#     K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
#     V_buffer,  # Holds [KV], b*s x (c)
#     sm_scale,
#     kv_indptr,
#     kv_indices,
#     Att_Out,  # b x h x NUM_KV_SPLITS x (kv_lora_rank)
#     Att_Lse,  # b x h x NUM_KV_SPLITS x (1)
#     stride_qb,
#     stride_qh,
#     stride_buf_kbs,
#     stride_buf_kh,
#     stride_mid_ob,
#     stride_mid_oh,
#     stride_mid_os,
#     stride_mid_lse_b,
#     stride_mid_lse_h,
#     stride_mid_lse_s,
#     stride_b_block_table,
#     kv_lora_rank: gl.constexpr,
#     qk_rope_head_dim: gl.constexpr,
#     kv_group_num: gl.constexpr,
#     q_head_num: gl.constexpr,
#     batch: gl.constexpr,
#     logit_cap: gl.constexpr,
#     max_qo_len: gl.constexpr,
#     BLOCK_C: gl.constexpr,
#     BLOCK_R: gl.constexpr,
#     BLOCK_N: gl.constexpr,
#     BLOCK_H: gl.constexpr,
#     NUM_KV_SPLITS: gl.constexpr,
#     PAGE_BLOCK_SIZE: gl.constexpr,
# ):
#     pid = gl.program_id(0)
#     num_q_head_blk = gl.cdiv(q_head_num, BLOCK_H)
#
#     pid_head_kv_split = pid % (num_q_head_blk * NUM_KV_SPLITS)
#     pid_head_kv_split = remap_xcd(pid_head_kv_split, (num_q_head_blk * NUM_KV_SPLITS))
#
#     cur_head_id = pid_head_kv_split % num_q_head_blk
#     split_kv_id = (pid_head_kv_split // num_q_head_blk) % NUM_KV_SPLITS
#
#     log2e: gl.constexpr = 1.4426950408889634
#
#     cur_batch = (pid // (num_q_head_blk * NUM_KV_SPLITS)) % batch
#
#     cur_batch_kv = cur_batch # // (max_qo_len // 2)
#     cur_batch_kv_start_idx = gl.load(kv_indptr + cur_batch_kv)
#     cur_batch_kv_end_idx = gl.load(kv_indptr + cur_batch_kv + 1)
#
#     blocked_q_nope_mk: gl.constexpr = gl.BlockedLayout(  # max 64 * 512 in
#         size_per_thread=[1, 16],  # 64 * 512
#         threads_per_warp=[16, 4],
#         warps_per_cta=[1, 4],
#         order=[1, 0],
#     )
#     blocked_q_rope_mk: gl.constexpr = gl.BlockedLayout(  # max 64 * 64 in
#         size_per_thread=[1, 4],  # 64 * 576
#         threads_per_warp=[16, 4],
#         warps_per_cta=[1, 4],
#         order=[1, 0],
#     )
#
#     blocked_ld_k_nope_kn: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
#         size_per_thread=[4, 16],  # 64 * 512
#         threads_per_warp=[16, 4],
#         warps_per_cta=[1, 4],
#         order=[1, 0],
#     )
#
#     blocked_ld_k_rope_kn: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
#         size_per_thread=[1, 16],  # 64 * 576
#         threads_per_warp=[64, 1],
#         warps_per_cta=[1, 4],
#         order=[1, 0],
#     )
#
#     shared_q: gl.constexpr = gl.SwizzledSharedLayout(
#         vec=8, per_phase=1, max_phase=8, order=[1, 0]
#     )
#     shared_k: gl.constexpr = gl.SwizzledSharedLayout(
#         vec=16, per_phase=1, max_phase=16, order=[0, 1]
#     )
#     shared_k_rope: gl.constexpr = gl.SwizzledSharedLayout(
#         vec=16, per_phase=1, max_phase=16, order=[0, 1]
#     )
#     shared_p: gl.constexpr = gl.SwizzledSharedLayout(
#         vec=8, per_phase=1, max_phase=16, order=[1, 0]
#     )
#     shared_v: gl.constexpr = gl.SwizzledSharedLayout(
#         vec=16, per_phase=1, max_phase=16, order=[0, 1]
#     )
#
#     smem_q0_nope = gl.allocate_shared_memory(
#         Q.type.element_ty, [BLOCK_H, kv_lora_rank // 2], layout=shared_q
#     )
#     smem_q1_nope = gl.allocate_shared_memory(
#         Q.type.element_ty, [BLOCK_H, kv_lora_rank // 2], layout=shared_q
#     )
#     smem_q_rope = gl.allocate_shared_memory(
#         Q.type.element_ty, [BLOCK_H, qk_rope_head_dim], layout=shared_q
#     )
#
#     smem_p = gl.allocate_shared_memory(
#         Q.type.element_ty, [BLOCK_H, BLOCK_N], layout=shared_p
#     )
#
#     mfma_layout_qk: gl.constexpr = gl.amd.AMDMFMALayout(
#         version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
#     )
#     mfma_layout_kv: gl.constexpr = gl.amd.AMDMFMALayout(
#         version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
#     )
#
#     dot_q_layout: gl.constexpr = gl.DotOperandLayout(
#         operand_index=0, parent=mfma_layout_qk, k_width=16
#     )
#     dot_k_layout: gl.constexpr = gl.DotOperandLayout(
#         operand_index=1, parent=mfma_layout_qk, k_width=16
#     )
#
#     dot_p_layout: gl.constexpr = gl.DotOperandLayout(
#         operand_index=0, parent=mfma_layout_kv, k_width=16
#     )
#     dot_v_layout: gl.constexpr = gl.DotOperandLayout(
#         operand_index=1, parent=mfma_layout_kv, k_width=16
#     )
#
#     if BLOCK_H < kv_group_num:
#         VALID_BLOCK_H: gl.constexpr = BLOCK_H
#     else:
#         VALID_BLOCK_H: gl.constexpr = kv_group_num
#
#     cur_head = cur_head_id * VALID_BLOCK_H + gl.arange(
#         0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_nope_mk)
#     )
#     mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
#     mask_h = mask_h & (cur_head < q_head_num)
#
#     cur_N = gl.arange(
#         0, BLOCK_N, layout=gl.SliceLayout(1, blocked_ld_k_nope_kn)
#     )
#
#     cur_N_pe = gl.arange(
#         0, BLOCK_N, layout=gl.SliceLayout(1, blocked_ld_k_rope_kn)
#     )
#
#     cur_head_pe = gl.arange(
#         0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_rope_mk)
#     )
#     mask_h_pe = cur_head_pe < VALID_BLOCK_H
#     mask_h_pe = mask_h_pe & (cur_head_pe < q_head_num)
#
#     offs_c = gl.arange(
#         0, BLOCK_C, layout=gl.SliceLayout(0, blocked_q_nope_mk)
#     )
#
#     offs_om = gl.arange(
#         0, BLOCK_H, layout=gl.SliceLayout(1, mfma_layout_kv)
#     )
#     mask_om = offs_om < VALID_BLOCK_H
#     mask_om = mask_om & (offs_om < q_head_num)
#     offs_on = gl.arange(
#         0, BLOCK_C, layout=gl.SliceLayout(0, mfma_layout_kv)
#     )
#     mask_on = offs_on < kv_lora_rank
#
#     offs_qk_r = gl.arange(
#         kv_lora_rank,
#         kv_lora_rank + BLOCK_R,
#         layout=gl.SliceLayout(0, blocked_q_rope_mk)
#     )  # to get the k_pe
#
#     # offs_k_c = gl.arange(
#     #     0, BLOCK_C, layout=gl.SliceLayout(0, blocked_ld_k_nope_kn)
#     # )
#     offs_k_c = gl.arange(
#         0, 256, layout=gl.SliceLayout(0, blocked_ld_k_nope_kn)
#     )
#     offs_k_r = gl.arange(
#         kv_lora_rank,
#         kv_lora_rank + BLOCK_R,
#         layout=gl.SliceLayout(0, blocked_ld_k_rope_kn)
#     )  # to get the k_pe
#
#     cur_batch_seq_len = cur_batch_kv_end_idx - cur_batch_kv_start_idx
#     cur_batch_block_nums = gl.cdiv(cur_batch_seq_len, PAGE_BLOCK_SIZE)
#
#     off_q_pe = (
#         cur_batch * stride_qb + cur_head_pe[:, None] * stride_qh + offs_qk_r[None, :]
#     )
#     offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]
#
#     mask_c = offs_c < kv_lora_rank
#     mask_k_c = offs_k_c < kv_lora_rank
#     mask_qk_r = offs_qk_r < (kv_lora_rank + qk_rope_head_dim)
#     mask_k_r = offs_k_r < (kv_lora_rank + qk_rope_head_dim)
#
#     blocks_per_split = gl.cdiv(cur_batch_block_nums, NUM_KV_SPLITS)
#     split_kv_start = blocks_per_split * split_kv_id
#     
#     # if pid == 4:
#     #     tl.device_print("split_kv_start   ", split_kv_start)
#     #     tl.device_print("cur_batch_seq_len   ", cur_batch_seq_len)
#
#     if split_kv_start * PAGE_BLOCK_SIZE > cur_batch_seq_len:
#         return
#
#     split_kv_end = gl.minimum(split_kv_start + blocks_per_split, cur_batch_block_nums)
#
#     q = gl.amd.cdna3.buffer_load(
#         ptr=Q,
#         offsets=offs_q,
#         mask=(mask_h[:, None]) & (mask_c[None, :]),
#     )
#     q_pe = gl.amd.cdna3.buffer_load(
#         ptr=Q,
#         offsets=off_q_pe,
#         mask=(mask_h_pe[:, None]) & (mask_qk_r[None, :]),
#     )
#     kv_loc = gl.load(
#         kv_indices + split_kv_start + cur_batch * stride_b_block_table,
#     )
#
#     # if pid == 4:
#     #     tl.device_print("kv_loc ", kv_loc)
#
#     q = gl.reshape(q, [BLOCK_H, 2, BLOCK_C // 2])
#     q = gl.permute(q, (0, 2, 1))
#     q0, q1 = gl.split(q)
#
#     smem_q0_nope.store(q0)
#     smem_q1_nope.store(q1)
#     q0 = smem_q0_nope.load(layout=dot_q_layout)
#     q1 = smem_q1_nope.load(layout=dot_q_layout)
#
#     smem_q_rope.store(q_pe)
#     q_pe = smem_q_rope.load(layout=dot_q_layout)
#
#     smem_q0_nope._keep_alive()
#     smem_q1_nope._keep_alive()
#     smem_q_rope._keep_alive()
#
#     # smem_v = gl.allocate_shared_memory(
#     #     K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_k
#     # )
#     smem_k_rope = gl.allocate_shared_memory(
#         K_Buffer.type.element_ty, [qk_rope_head_dim, BLOCK_N], layout=shared_k_rope
#     )
#
#     acc1 = gl.zeros([BLOCK_H, BLOCK_C // 2], dtype=gl.float32,
#         layout=mfma_layout_kv,
#     )
#     acc2 = gl.zeros([BLOCK_H, BLOCK_C // 2], dtype=gl.float32,
#         layout=mfma_layout_kv,
#     )
#
#     zeros = gl.zeros(
#         (BLOCK_H, BLOCK_N), dtype=gl.float32, layout=mfma_layout_qk,
#     )
#     k_id = kv_loc * PAGE_BLOCK_SIZE + cur_N
#     mask_k_id = split_kv_start * PAGE_BLOCK_SIZE + cur_N
#     mask_k = mask_k_id < cur_batch_seq_len
#     offs_buf_kv = (k_id[:, None]) * stride_buf_kh + offs_k_c[None, :]
#     # if pid == 2:
#     #     tl.device_print("k_id  ", mask_k_id)
#     #     tl.device_print("mask_k   ", mask_k)
#
#     kv1 = gl.amd.cdna3.buffer_load(
#         ptr=K_Buffer,
#         offsets=offs_buf_kv,
#         mask=mask_k[:, None] & mask_k_c[None, :]
#     )  # the shared latent tensor for keys and values
#
#     kv2 = gl.amd.cdna3.buffer_load(
#         ptr=K_Buffer,
#         offsets=offs_buf_kv + 256,
#         mask=mask_k[:, None] & mask_k_c[None, :]
#     )  # the shared latent tensor for keys and values
#
#     # if pid == 1:
#     #     tl.device_print("kv2 before ", kv2.to(gl.float32))
#
#     smem_kv1 = gl.allocate_shared_memory(
#         K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k
#     )
#     smem_kv2 = gl.allocate_shared_memory(
#         K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k
#     )
#
#     gl.amd.cdna3.sched_barrier(0x0)
#
#     smem_kv1.store(kv1.T)
#     smem_kv2.store(kv2.T)
#
#     k_id_pe = kv_loc * PAGE_BLOCK_SIZE + cur_N_pe
#     offs_buf_k_pe = k_id_pe[:, None] * stride_buf_kh + offs_k_r[None, :]
#     mask_k_id = split_kv_start * PAGE_BLOCK_SIZE + cur_N_pe
#     mask_k_pe = mask_k_id < cur_batch_seq_len 
#
#     # if pid == 2:
#     #     tl.device_print("cur_batch_kv_end_idx before ", cur_batch_kv_end_idx)
#
#     k_pe = gl.amd.cdna3.buffer_load(
#         ptr=K_Buffer,
#         offsets=offs_buf_k_pe,
#         mask=mask_k_pe[:, None] & mask_k_r[None, :]
#     )  # positional embedding part of keys
#
#     # if pid == 2:
#     #     tl.device_print("k_pe before ", k_pe)
#
#     e_sum = gl.arange(0, BLOCK_H, layout = gl.SliceLayout(1, mfma_layout_qk)).to(gl.float32) * float(0)
#     e_max = e_sum - float("inf")
#
#     # cur_k1 = gl.convert_layout(kv1.T, layout=dot_k_layout)
#     # cur_k2 = gl.convert_layout(kv2.T, layout=dot_k_layout)
#
#     cur_k1 = smem_kv1.load(layout=dot_k_layout)
#     cur_k2 = smem_kv2.load(layout=dot_k_layout)
#
#     gl.amd.cdna3.sched_barrier(0x0)
#     smem_kv1 = smem_kv1._reinterpret(
#         K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)
#     kv1_transpose = tl.trans(kv1)
#     gl.amd.cdna3.sched_barrier(0x0)
#
#     smem_kv1.store(kv1_transpose.T)
#     smem_kv2 = smem_kv2._reinterpret(
#         K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)
#     kv2_transpose = tl.trans(kv2)
#     gl.amd.cdna3.sched_barrier(0x0)
#
#     smem_kv2.store(kv2_transpose.T)
#
#     smem_k_rope.store(k_pe.T)
#     gl.amd.cdna3.sched_barrier(0x0)
#     split_kv_start += 1 
#
#     mask_qk_h = gl.arange(0, BLOCK_H, gl.SliceLayout(1, mfma_layout_qk))
#     # if pid == 1:
#     #     tl.device_print("split_kv_start  ", split_kv_start)
#     #     tl.device_print("split_kv_end  ", split_kv_end)
#
#     for start_n in range(split_kv_start, split_kv_end):
#         kv_loc = gl.load(
#             kv_indices + start_n + cur_batch * stride_b_block_table,
#         )
#
#         cur_k_pe = smem_k_rope.load(layout=dot_k_layout)
#
#         gl.amd.cdna3.sched_barrier(0x0)
#         k_id = kv_loc * PAGE_BLOCK_SIZE + cur_N
#         offs_buf_kv = k_id[:, None] * stride_buf_kh + offs_k_c[None, :]
#         mask_k_id = start_n * PAGE_BLOCK_SIZE + cur_N
#         mask_k = mask_k_id < cur_batch_seq_len
#         gl.amd.cdna3.sched_barrier(0x0)
#
#         qk = gl.amd.cdna3.mfma(q0, cur_k1, zeros)
#         kv1 = gl.amd.cdna3.buffer_load(
#             ptr=K_Buffer,
#             offsets=offs_buf_kv,
#             mask=mask_k[:, None] & mask_k_c[None, :]
#         )
#
#         gl.amd.cdna3.sched_barrier(0x0)
#
#         qk = gl.amd.cdna3.mfma(q1, cur_k2, qk)
#         kv2 = gl.amd.cdna3.buffer_load(
#             ptr=K_Buffer,
#             offsets=offs_buf_kv + 256,
#             mask=mask_k[:, None] & mask_k_c[None, :]
#         )
#
#         # smem_kv1 = smem_kv1.permute([1, 0])
#         # smem_kv2 = smem_kv2.permute([1, 0])
#
#         cur_k1 = smem_kv1.load(layout=dot_v_layout)
#         cur_k2 = smem_kv2.load(layout=dot_v_layout)
#
#         qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, qk)
#
#         qk *= sm_scale
#         # if pid == 0:
#         #     tl.device_print("qk before ", qk)
#         # mask_qk_n = (start_n - 1) * PAGE_BLOCK_SIZE + gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma_layout_qk))
#         # qk = tl.where(
#         #     (mask_qk_n[None, :] < cur_batch_seq_len) & (mask_qk_h[:, None] >= 0), qk, float("-inf")
#         # )
#         # if pid == 0:
#         #     tl.device_print("qk after ", qk)
#         
#
#         k_id_pe = kv_loc * PAGE_BLOCK_SIZE + cur_N_pe
#         offs_buf_k_pe = k_id_pe[:, None] * stride_buf_kh + offs_k_r[None, :]
#         mask_k_id = start_n * PAGE_BLOCK_SIZE + cur_N_pe
#         mask_k_pe = mask_k_id < cur_batch_seq_len
#
#         gl.amd.cdna3.sched_barrier(0x0)
#         k_pe = gl.amd.cdna3.buffer_load(
#             ptr=K_Buffer,
#             offsets=offs_buf_k_pe,
#             mask=mask_k_pe[:, None] & mask_k_r[None, :]
#         )  # positional embedding part of keys
#
#
#         # smem_kv1._keep_alive()
#         # smem_kv2._keep_alive()
#
#         n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))
#         n_e_max = gl.maximum(n_e_max, e_max)
#
#         re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
#         p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)
#         smem_p.store(p.to(q0.dtype))
#         gl.amd.cdna3.sched_barrier(0x0)
#
#         cur_p = smem_p.load(layout=dot_p_layout)
#         smem_kv1 = smem_kv1._reinterpret(
#             K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k)
#         smem_kv2 = smem_kv2._reinterpret(
#             K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k)
#
#         smem_kv1.store(kv1.T)
#         acc1 = acc1 * re_scale[:, None]
#         acc1 = gl.amd.cdna3.mfma(cur_p, cur_k1, acc1)
#         e_sum = e_sum * re_scale + gl.sum(p, 1)
#
#         smem_kv2.store(kv2.T)
#         acc2 = acc2 * re_scale[:, None]
#         acc2 = gl.amd.cdna3.mfma(cur_p, cur_k2, acc2)
#         smem_k_rope.store(k_pe.T)
#         e_max = n_e_max
#
#         cur_k1 = smem_kv1.load(layout=dot_k_layout)
#         kv1_transpose = tl.trans(kv1)
#         gl.amd.cdna3.sched_barrier(0x0)
#         smem_kv1 = smem_kv1._reinterpret(
#             K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)
#
#         smem_kv1.store(kv1_transpose.T)
#         cur_k2 = smem_kv2.load(layout=dot_k_layout)
#
#         kv2_transpose = tl.trans(kv2)
#         gl.amd.cdna3.sched_barrier(0x0)
#         smem_kv2 = smem_kv2._reinterpret(
#             K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)
#         smem_kv2.store(kv2_transpose.T)
#
#     # smem_kv1 = smem_kv1._reinterpret(
#     #     K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)
#     # smem_kv2 = smem_kv2._reinterpret(
#     #     K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)
#     #
#     # # kv1_transpose = tl.trans(kv1)
#     # # smem_kv1.store(kv1_transpose.T)
#     # # kv2_transpose = tl.trans(kv2)
#     # # smem_kv2.store(kv2_transpose.T)
#
#     qk = gl.amd.cdna3.mfma(q0, cur_k1, zeros)
#     qk = gl.amd.cdna3.mfma(q1, cur_k2, qk)
#
#     cur_k_pe = smem_k_rope.load(layout=dot_k_layout)
#
#     qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, qk)
#
#     # if pid == 5:
#     #     tl.device_print("qk", qk)
#
#     cur_k1 = smem_kv1.load(layout=dot_v_layout)
#     cur_k2 = smem_kv2.load(layout=dot_v_layout)
#
#     qk *= sm_scale
#     # if pid == 1:
#     #     tl.device_print("qk before ", (split_kv_end - 1) * PAGE_BLOCK_SIZE)
#     mask_qk_n = (split_kv_end - 1) * PAGE_BLOCK_SIZE + gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma_layout_qk))
#
#     qk = tl.where(
#         (mask_qk_n[None, :] < cur_batch_seq_len) & (mask_qk_h[:, None] >= 0), qk, float("-inf")
#     )
#     # if pid == 1:
#     #     tl.device_print("qk after", qk)
#
#     n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))
#
#     n_e_max = gl.maximum(n_e_max, e_max)
#     re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
#     p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)
#
#     # if pid == 5:
#     #     tl.device_print("p", p)
#
#     smem_p.store(p.to(Q.dtype.element_ty))
#
#     acc1 = acc1 * re_scale[:, None]
#     acc2 = acc2 * re_scale[:, None]
#     e_sum = e_sum * re_scale + gl.sum(p, 1)
#     gl.amd.cdna3.sched_barrier(0x0)
#     cur_p = smem_p.load(layout=dot_p_layout)
#     e_max = n_e_max
#
#     acc1 = gl.amd.cdna3.mfma(cur_p, cur_k1, acc1)
#     acc2 = gl.amd.cdna3.mfma(cur_p, cur_k2, acc2)
#
#     smem_kv1._keep_alive()
#     smem_kv2._keep_alive()
#
#     acc = gl.join(acc1, acc2)
#     acc  = gl.permute(acc, (0, 2, 1))
#     acc = gl.reshape(acc, (BLOCK_H, BLOCK_C))
#     acc = gl.convert_layout(acc, mfma_layout_kv)
#
#     # if pid == 5:
#     #     tl.device_print("acc", acc)
#
#     smem_o = gl.allocate_shared_memory(
#         Att_Out.type.element_ty, [BLOCK_H, kv_lora_rank], layout=shared_q
#     )
#
#     e_sum = 1 / e_sum
#     smem_o.store(acc * e_sum[:, None])
#     # smem_o.store(acc)
#     cur_o = smem_o.load(layout=blocked_q_nope_mk)
#
#     offs_mid_o = (
#         cur_batch * stride_mid_ob
#         + cur_head[:, None] * stride_mid_oh
#         + split_kv_id * stride_mid_os
#         + offs_c[None, :]
#     )
#     # offs_mid_o = (
#     #     cur_batch * stride_mid_ob
#     #     + offs_om[:, None] * stride_mid_oh
#     #     + split_kv_id * stride_mid_os
#     #     + offs_on[None, :]
#     # )
#
#     gl.amd.cdna3.buffer_store(
#         stored_value=cur_o,
#         ptr=Att_Out,
#         offsets=offs_mid_o,
#         mask=(mask_h[:, None]) & (mask_c[None, :]),
#     )
#
#     offs_mid_lse = (
#         cur_batch * stride_mid_lse_b
#         + offs_om * stride_mid_lse_h
#         + split_kv_id * stride_mid_lse_s 
#     )
#
#     mask_lse = offs_om < (cur_head_id + 1) * VALID_BLOCK_H
#     gl.amd.cdna3.buffer_store(
#         stored_value=e_max - gl.log(e_sum),
#         ptr=Att_Lse,
#         offsets=offs_mid_lse,
#         mask=mask_lse,
#     )


@gluon.jit
def _fwd_grouped_kernel_stage1_n16x4_prefetch_k_paged_64_async(
    Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
    K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
    V_buffer,  # Holds [KV], b*s x (c)
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,  # b x h x NUM_KV_SPLITS x (kv_lora_rank)
    Att_Lse,  # b x h x NUM_KV_SPLITS x (1)
    stride_qb,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_lse_b,
    stride_mid_lse_h,
    stride_mid_lse_s,
    stride_b_block_table,
    kv_lora_rank: gl.constexpr,
    qk_rope_head_dim: gl.constexpr,
    kv_group_num: gl.constexpr,
    q_head_num: gl.constexpr,
    batch: gl.constexpr,
    logit_cap: gl.constexpr,
    max_qo_len: gl.constexpr,
    BLOCK_C: gl.constexpr,
    BLOCK_R: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_H: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    PAGE_BLOCK_SIZE: gl.constexpr,
):
    pid = gl.program_id(0)
    num_q_head_blk = gl.cdiv(q_head_num, BLOCK_H)

    pid_head_kv_split = pid % (num_q_head_blk * NUM_KV_SPLITS)
    pid_head_kv_split = remap_xcd(pid_head_kv_split, (num_q_head_blk * NUM_KV_SPLITS))

    cur_head_id = pid_head_kv_split % num_q_head_blk
    split_kv_id = (pid_head_kv_split // num_q_head_blk) % NUM_KV_SPLITS

    log2e: gl.constexpr = 1.4426950408889634

    cur_batch = (pid // (num_q_head_blk * NUM_KV_SPLITS)) % batch

    cur_batch_kv = cur_batch # // (max_qo_len // 2)
    cur_batch_kv_start_idx = gl.load(kv_indptr + cur_batch_kv)
    cur_batch_kv_end_idx = gl.load(kv_indptr + cur_batch_kv + 1)

    blocked_q_nope_mk: gl.constexpr = gl.BlockedLayout(  # max 64 * 512 in
        size_per_thread=[2, 8],  # 64 * 512
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )
    blocked_q_rope_mk: gl.constexpr = gl.BlockedLayout(  # max 64 * 64 in
        size_per_thread=[1, 4],  # 64 * 576
        threads_per_warp=[16, 4],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )

    # blocked_ld_k_nope_kn: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
    #     size_per_thread=[8, 8],  # 64 * 512
    #     threads_per_warp=[8, 8],
    #     warps_per_cta=[1, 4],
    #     order=[1, 0],
    # )
    blocked_ld_k_nope_kn_async: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
        size_per_thread=[1, 8],  # 64 * 512
        threads_per_warp=[2, 32],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    # blocked_ld_k_rope_kn: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
    blocked_ld_k_rope_kn_async: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
        size_per_thread=[1, 8],  # 64 * 576
        threads_per_warp=[32, 2],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )

    shared_q: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[1, 0]
    )
    shared_k_store: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0]
    )
    shared_k: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[0, 1]
    )
    shared_k_rope: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[0, 1]
    )
    shared_p: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0]
    )
    shared_v: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[0, 1]
    )

    smem_q0_nope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, kv_lora_rank // 2], layout=shared_q
    )
    smem_q1_nope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, kv_lora_rank // 2], layout=shared_q
    )
    smem_q_rope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, qk_rope_head_dim], layout=shared_q
    )

    smem_p = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, BLOCK_N], layout=shared_p
    )

    mfma_layout_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    mfma_layout_kv: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[1, 4]
    )

    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_qk, k_width=16
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_qk, k_width=16
    )

    dot_p_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_kv, k_width=16
    )
    dot_v_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_kv, k_width=16
    )

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: gl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: gl.constexpr = kv_group_num

    cur_head = cur_head_id * VALID_BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_nope_mk)
    )
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    TILE_N: gl.constexpr = BLOCK_H // 8

    cur_N = gl.arange(
        0, TILE_N, layout=gl.SliceLayout(1, blocked_ld_k_nope_kn_async)
    )

    cur_N_pe = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(1, blocked_ld_k_rope_kn_async)
    )

    cur_head_pe = gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_rope_mk)
    )
    mask_h_pe = cur_head_pe < VALID_BLOCK_H
    mask_h_pe = mask_h_pe & (cur_head_pe < q_head_num)

    offs_c = gl.arange(
        0, BLOCK_C, layout=gl.SliceLayout(0, blocked_q_nope_mk)
    )

    offs_om = gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, mfma_layout_kv)
    )
    mask_om = offs_om < VALID_BLOCK_H
    mask_om = mask_om & (offs_om < q_head_num)
    offs_on = gl.arange(
        0, BLOCK_C, layout=gl.SliceLayout(0, mfma_layout_kv)
    )
    mask_on = offs_on < kv_lora_rank

    offs_qk_r = gl.arange(
        kv_lora_rank,
        kv_lora_rank + BLOCK_R,
        layout=gl.SliceLayout(0, blocked_q_rope_mk)
    )  # to get the k_pe

    # offs_k_c = gl.arange(
    #     0, BLOCK_C, layout=gl.SliceLayout(0, blocked_ld_k_nope_kn)
    # )
    offs_k_c = gl.arange(
        0, 256, layout=gl.SliceLayout(0, blocked_ld_k_nope_kn_async)
    )
    offs_k_r = gl.arange(
        kv_lora_rank,
        kv_lora_rank + BLOCK_R,
        layout=gl.SliceLayout(0, blocked_ld_k_rope_kn_async)
    )  # to get the k_pe

    cur_batch_seq_len = cur_batch_kv_end_idx - cur_batch_kv_start_idx
    cur_batch_block_nums = gl.cdiv(cur_batch_seq_len, PAGE_BLOCK_SIZE)

    off_q_pe = (
        cur_batch * stride_qb + cur_head_pe[:, None] * stride_qh + offs_qk_r[None, :]
    )
    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]

    mask_c = offs_c < kv_lora_rank
    mask_k_c = offs_k_c < kv_lora_rank
    mask_qk_r = offs_qk_r < (kv_lora_rank + qk_rope_head_dim)
    mask_k_r = offs_k_r < (kv_lora_rank + qk_rope_head_dim)

    blocks_per_split = gl.cdiv(cur_batch_block_nums, NUM_KV_SPLITS)
    split_kv_start = blocks_per_split * split_kv_id
    
    if split_kv_start * PAGE_BLOCK_SIZE > cur_batch_seq_len:
        return

    split_kv_end = gl.minimum(split_kv_start + blocks_per_split, cur_batch_block_nums)

    q = gl.amd.cdna3.buffer_load(
        ptr=Q,
        offsets=offs_q,
        mask=(mask_h[:, None]) & (mask_c[None, :]),
    )
    q_pe = gl.amd.cdna3.buffer_load(
        ptr=Q,
        offsets=off_q_pe,
        mask=(mask_h_pe[:, None]) & (mask_qk_r[None, :]),
    )
    kv_loc = gl.load(
        kv_indices + split_kv_start + cur_batch * stride_b_block_table,
    )

    # if pid == 4:
    #     tl.device_print("kv_loc ", kv_loc)

    q = gl.reshape(q, [BLOCK_H, 2, BLOCK_C // 2])
    q = gl.permute(q, (0, 2, 1))
    q0, q1 = gl.split(q)

    smem_q0_nope.store(q0)
    smem_q1_nope.store(q1)
    q0 = smem_q0_nope.load(layout=dot_q_layout)
    q1 = smem_q1_nope.load(layout=dot_q_layout)

    smem_q_rope.store(q_pe)
    q_pe = smem_q_rope.load(layout=dot_q_layout)

    smem_q0_nope._keep_alive()
    smem_q1_nope._keep_alive()
    smem_q_rope._keep_alive()

    # smem_v = gl.allocate_shared_memory(
    #     K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_k
    # )
    smem_k_rope = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [qk_rope_head_dim, BLOCK_N], layout=shared_k_rope
    )

    acc1 = gl.zeros([BLOCK_H, BLOCK_C // 2], dtype=gl.float32,
        layout=mfma_layout_kv,
    )
    acc2 = gl.zeros([BLOCK_H, BLOCK_C // 2], dtype=gl.float32,
        layout=mfma_layout_kv,
    )

    zeros = gl.zeros(
        (BLOCK_H, BLOCK_N), dtype=gl.float32, layout=mfma_layout_qk,
    )
    k_id = kv_loc * PAGE_BLOCK_SIZE + cur_N
    mask_k_id = split_kv_start * PAGE_BLOCK_SIZE + cur_N
    mask_k = mask_k_id < cur_batch_seq_len
    offs_buf_kv = (k_id[:, None]) * stride_buf_kh + offs_k_c[None, :]


    # smem_kv1 = gl.allocate_shared_memory(
    #     K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k
    # )
    # smem_kv2 = gl.allocate_shared_memory(
    #     K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k
    # )
    smem_kv1 = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_k_store
    )
    smem_kv2 = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_k_store
    )
    smem_kv10 = smem_kv1.slice(0,          TILE_N)
    smem_kv11 = smem_kv1.slice(TILE_N,     TILE_N)
    smem_kv12 = smem_kv1.slice(TILE_N * 2, TILE_N)
    smem_kv13 = smem_kv1.slice(TILE_N * 3, TILE_N)
    smem_kv14 = smem_kv1.slice(TILE_N * 4, TILE_N)
    smem_kv15 = smem_kv1.slice(TILE_N * 5, TILE_N)
    smem_kv16 = smem_kv1.slice(TILE_N * 6, TILE_N)
    smem_kv17 = smem_kv1.slice(TILE_N * 7, TILE_N)

    smem_kv20 = smem_kv2.slice(0,          TILE_N)
    smem_kv21 = smem_kv2.slice(TILE_N,     TILE_N)
    smem_kv22 = smem_kv2.slice(TILE_N * 2, TILE_N)
    smem_kv23 = smem_kv2.slice(TILE_N * 3, TILE_N)
    smem_kv24 = smem_kv1.slice(TILE_N * 4, TILE_N)
    smem_kv25 = smem_kv1.slice(TILE_N * 5, TILE_N)
    smem_kv26 = smem_kv1.slice(TILE_N * 6, TILE_N)
    smem_kv27 = smem_kv1.slice(TILE_N * 7, TILE_N)

    ### =======   load kv1 ====== ###
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv10,
        K_Buffer,
        offsets=offs_buf_kv,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv11,
        K_Buffer,
        offsets=offs_buf_kv + TILE_N * kv_lora_rank,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv12,
        K_Buffer,
        offsets=offs_buf_kv + TILE_N * kv_lora_rank * 2,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv13,
        K_Buffer,
        offsets=offs_buf_kv + TILE_N * kv_lora_rank * 3,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv14,
        K_Buffer,
        offsets=offs_buf_kv + TILE_N * kv_lora_rank * 4,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv15,
        K_Buffer,
        offsets=offs_buf_kv + TILE_N * kv_lora_rank * 5,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv16,
        K_Buffer,
        offsets=offs_buf_kv + TILE_N * kv_lora_rank * 6,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv17,
        K_Buffer,
        offsets=offs_buf_kv + TILE_N * kv_lora_rank * 7,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    ### =======   load kv1 end  ====== ###

    ### =======   load kv2  ====== ###
    # gl.amd.cdna3.sched_barrier(0x0)
    #
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv20,
        K_Buffer,
        offsets=offs_buf_kv + 256,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv21,
        K_Buffer,
        offsets=offs_buf_kv + TILE_N * kv_lora_rank + 256,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv22,
        K_Buffer,
        offsets=offs_buf_kv + TILE_N * kv_lora_rank * 2 + 256,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv23,
        K_Buffer,
        offsets=offs_buf_kv + TILE_N * kv_lora_rank * 3 + 256,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv24,
        K_Buffer,
        offsets=offs_buf_kv + TILE_N * kv_lora_rank * 4 + 256,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv25,
        K_Buffer,
        offsets=offs_buf_kv + TILE_N * kv_lora_rank * 5 + 256,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv26,
        K_Buffer,
        offsets=offs_buf_kv + TILE_N * kv_lora_rank * 6 + 256,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv27,
        K_Buffer,
        offsets=offs_buf_kv + TILE_N * kv_lora_rank * 7 + 256,
        mask=mask_k[:, None] & mask_k_c[None, :],
    )
    ### =======   load kv2 end  ====== ###

    k_id_pe = kv_loc * PAGE_BLOCK_SIZE + cur_N_pe
    offs_buf_k_pe = k_id_pe[:, None] * stride_buf_kh + offs_k_r[None, :]
    mask_k_id = split_kv_start * PAGE_BLOCK_SIZE + cur_N_pe
    mask_k_pe = mask_k_id < cur_batch_seq_len 

    k_pe = gl.amd.cdna4.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_k_pe,
        mask=mask_k_pe[:, None] & mask_k_r[None, :]
    )  # positional embedding part of keys

    # if pid == 2:
    #     tl.device_print("k_pe before ", k_pe)

    e_sum = gl.arange(0, BLOCK_H, layout = gl.SliceLayout(1, mfma_layout_qk)).to(gl.float32) * float(0)
    e_max = e_sum - float("inf")

    # cur_k1 = gl.convert_layout(kv1.T, layout=dot_k_layout)
    # cur_k2 = gl.convert_layout(kv2.T, layout=dot_k_layout)
    smem_kv1 = smem_kv1._reinterpret(
        K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k)
    smem_kv2 = smem_kv2._reinterpret(
        K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k)

    cur_k1 = smem_kv1.load(layout=dot_k_layout)
    cur_k2 = smem_kv2.load(layout=dot_k_layout)

    smem_k_rope.store(k_pe.T)
    gl.amd.cdna3.sched_barrier(0x0)
    split_kv_start += 1 

    mask_qk_h = gl.arange(0, BLOCK_H, gl.SliceLayout(1, mfma_layout_qk))
    # if pid == 1:
    #     tl.device_print("split_kv_start  ", split_kv_start)
    #     tl.device_print("split_kv_end  ", split_kv_end)

    for start_n in range(split_kv_start, split_kv_end):
        kv_loc = gl.load(
            kv_indices + start_n + cur_batch * stride_b_block_table,
        )

        cur_k_pe = smem_k_rope.load(layout=dot_k_layout)

        gl.amd.cdna3.sched_barrier(0x0)
        k_id = kv_loc * PAGE_BLOCK_SIZE + cur_N
        offs_buf_kv = k_id[:, None] * stride_buf_kh + offs_k_c[None, :]
        mask_k_id = start_n * PAGE_BLOCK_SIZE + cur_N
        mask_k = mask_k_id < cur_batch_seq_len
        gl.amd.cdna3.sched_barrier(0x0)

        qk = gl.amd.cdna3.mfma(q0, cur_k1, zeros)
        qk = gl.amd.cdna3.mfma(q1, cur_k2, qk)

        smem_kv1 = smem_kv1.permute([1, 0])
        smem_kv2 = smem_kv2.permute([1, 0])

        cur_k1 = smem_kv1.load(layout=dot_v_layout)
        cur_k2 = smem_kv2.load(layout=dot_v_layout)
        gl.amd.cdna3.sched_barrier(0x0)

        smem_kv1 = smem_kv1._reinterpret(
            K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank], layout=shared_k_store)
        smem_kv2 = smem_kv2._reinterpret(
            K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank], layout=shared_k_store)

        ### =======   load kv1 ====== ###
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv10,
            K_Buffer,
            offsets=offs_buf_kv,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv11,
            K_Buffer,
            offsets=offs_buf_kv + TILE_N * kv_lora_rank,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv12,
            K_Buffer,
            offsets=offs_buf_kv + TILE_N * kv_lora_rank * 2,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv13,
            K_Buffer,
            offsets=offs_buf_kv + TILE_N * kv_lora_rank * 3,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv14,
            K_Buffer,
            offsets=offs_buf_kv + TILE_N * kv_lora_rank * 4,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv15,
            K_Buffer,
            offsets=offs_buf_kv + TILE_N * kv_lora_rank * 5,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv16,
            K_Buffer,
            offsets=offs_buf_kv + TILE_N * kv_lora_rank * 6,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv17,
            K_Buffer,
            offsets=offs_buf_kv + TILE_N * kv_lora_rank * 7,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        ### =======   load kv1 end  ====== ###

        gl.amd.cdna3.sched_barrier(0x0)

        ### =======   load kv2  ====== ###
        # gl.amd.cdna3.sched_barrier(0x0)
        #
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv20,
            K_Buffer,
            offsets=offs_buf_kv + 256,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv21,
            K_Buffer,
            offsets=offs_buf_kv + TILE_N * kv_lora_rank + 256,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv22,
            K_Buffer,
            offsets=offs_buf_kv + TILE_N * kv_lora_rank * 2 + 256,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv23,
            K_Buffer,
            offsets=offs_buf_kv + TILE_N * kv_lora_rank * 3 + 256,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv24,
            K_Buffer,
            offsets=offs_buf_kv + TILE_N * kv_lora_rank * 4 + 256,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv25,
            K_Buffer,
            offsets=offs_buf_kv + TILE_N * kv_lora_rank * 5 + 256,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv26,
            K_Buffer,
            offsets=offs_buf_kv + TILE_N * kv_lora_rank * 6 + 256,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv27,
            K_Buffer,
            offsets=offs_buf_kv + TILE_N * kv_lora_rank * 7 + 256,
            mask=mask_k[:, None] & mask_k_c[None, :],
        )
        ### =======   load kv2 end  ====== ###
        qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, qk)

        qk *= sm_scale
        # if pid == 0:
        #     tl.device_print("qk before ", qk)
        # mask_qk_n = (start_n - 1) * PAGE_BLOCK_SIZE + gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma_layout_qk))
        # qk = tl.where(
        #     (mask_qk_n[None, :] < cur_batch_seq_len) & (mask_qk_h[:, None] >= 0), qk, float("-inf")
        # )
        # if pid == 0:
        #     tl.device_print("qk after ", qk)
        

        k_id_pe = kv_loc * PAGE_BLOCK_SIZE + cur_N_pe
        offs_buf_k_pe = k_id_pe[:, None] * stride_buf_kh + offs_k_r[None, :]
        mask_k_id = start_n * PAGE_BLOCK_SIZE + cur_N_pe
        mask_k_pe = mask_k_id < cur_batch_seq_len

        gl.amd.cdna3.sched_barrier(0x0)
        k_pe = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_k_pe,
            mask=mask_k_pe[:, None] & mask_k_r[None, :]
        )  # positional embedding part of keys


        # smem_kv1._keep_alive()
        # smem_kv2._keep_alive()

        n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))
        n_e_max = gl.maximum(n_e_max, e_max)

        re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
        p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)
        smem_p.store(p.to(q0.dtype))
        gl.amd.cdna3.sched_barrier(0x0)

        cur_p = smem_p.load(layout=dot_p_layout)

        acc1 = acc1 * re_scale[:, None]
        acc1 = gl.amd.cdna3.mfma(cur_p, cur_k1, acc1)
        e_sum = e_sum * re_scale + gl.sum(p, 1)

        acc2 = acc2 * re_scale[:, None]
        acc2 = gl.amd.cdna3.mfma(cur_p, cur_k2, acc2)
        smem_k_rope.store(k_pe.T)
        e_max = n_e_max
        smem_kv1 = smem_kv1._reinterpret(
            K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k)
        smem_kv2 = smem_kv2._reinterpret(
            K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k)

        cur_k1 = smem_kv1.load(layout=dot_k_layout)
        cur_k2 = smem_kv2.load(layout=dot_k_layout)

    qk = gl.amd.cdna3.mfma(q0, cur_k1, zeros)
    qk = gl.amd.cdna3.mfma(q1, cur_k2, qk)

    cur_k_pe = smem_k_rope.load(layout=dot_k_layout)

    qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, qk)

    # if pid == 5:
    #     tl.device_print("qk", qk)
    smem_kv1 = smem_kv1.permute([1, 0])
    smem_kv2 = smem_kv2.permute([1, 0])

    cur_k1 = smem_kv1.load(layout=dot_v_layout)
    cur_k2 = smem_kv2.load(layout=dot_v_layout)

    qk *= sm_scale
    # if pid == 1:
    #     tl.device_print("qk before ", (split_kv_end - 1) * PAGE_BLOCK_SIZE)
    mask_qk_n = (split_kv_end - 1) * PAGE_BLOCK_SIZE + gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma_layout_qk))

    qk = tl.where(
        (mask_qk_n[None, :] < cur_batch_seq_len) & (mask_qk_h[:, None] >= 0), qk, float("-inf")
    )
    # if pid == 1:
    #     tl.device_print("qk after", qk)

    n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))

    n_e_max = gl.maximum(n_e_max, e_max)
    re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
    p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)

    # if pid == 5:
    #     tl.device_print("p", p)

    smem_p.store(p.to(Q.dtype.element_ty))

    acc1 = acc1 * re_scale[:, None]
    acc2 = acc2 * re_scale[:, None]
    e_sum = e_sum * re_scale + gl.sum(p, 1)
    gl.amd.cdna3.sched_barrier(0x0)
    cur_p = smem_p.load(layout=dot_p_layout)
    e_max = n_e_max

    acc1 = gl.amd.cdna3.mfma(cur_p, cur_k1, acc1)
    acc2 = gl.amd.cdna3.mfma(cur_p, cur_k2, acc2)

    smem_kv1._keep_alive()
    smem_kv2._keep_alive()

    acc = gl.join(acc1, acc2)
    acc  = gl.permute(acc, (0, 2, 1))
    acc = gl.reshape(acc, (BLOCK_H, BLOCK_C))
    acc = gl.convert_layout(acc, mfma_layout_kv)

    # if pid == 5:
    #     tl.device_print("acc", acc)

    smem_o = gl.allocate_shared_memory(
        Att_Out.type.element_ty, [BLOCK_H, kv_lora_rank], layout=shared_q
    )

    e_sum = 1 / e_sum
    smem_o.store(acc * e_sum[:, None])
    # smem_o.store(acc)
    cur_o = smem_o.load(layout=blocked_q_nope_mk)

    offs_mid_o = (
        cur_batch * stride_mid_ob
        + cur_head[:, None] * stride_mid_oh
        + split_kv_id * stride_mid_os
        + offs_c[None, :]
    )
    # offs_mid_o = (
    #     cur_batch * stride_mid_ob
    #     + offs_om[:, None] * stride_mid_oh
    #     + split_kv_id * stride_mid_os
    #     + offs_on[None, :]
    # )

    gl.amd.cdna3.buffer_store(
        stored_value=cur_o,
        ptr=Att_Out,
        offsets=offs_mid_o,
        mask=(mask_h[:, None]) & (mask_c[None, :]),
    )

    offs_mid_lse = (
        cur_batch * stride_mid_lse_b
        + offs_om * stride_mid_lse_h
        + split_kv_id * stride_mid_lse_s 
    )

    mask_lse = offs_om < (cur_head_id + 1) * VALID_BLOCK_H
    gl.amd.cdna3.buffer_store(
        stored_value=e_max - gl.log(e_sum),
        ptr=Att_Lse,
        offsets=offs_mid_lse,
        mask=mask_lse,
    )

# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MLA decode kernel (nhead=128, fp8 Q, fp8 KV, bf16 output).

Transplanted from csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h128_fp8_fp8.cuh.

NOTE: Do NOT use ``from __future__ import annotations`` here -- it breaks
``fx.Constexpr`` detection in the FlyDSL AST rewriter.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl.expr import gpu, buffer_ops, vector
from flydsl.expr.typing import T


# ---------------------------------------------------------------------------
# Compile-time constants (mirroring HkMlaDecodeFwdTraits)
# ---------------------------------------------------------------------------
NUM_QO_HEADS: int = 128
NUM_KV_HEADS: int = 1
KV_LORA_RANK: int = 512
QK_NOPE_HEAD_DIM: int = KV_LORA_RANK  # 512
QK_ROPE_HEAD_DIM: int = 64
QK_HEAD_DIM: int = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM: int = KV_LORA_RANK  # 512
PAGE_SIZE: int = 1
NUM_WARPS: int = 8
WARP_SIZE: int = 64
NUM_THREADS: int = NUM_WARPS * WARP_SIZE  # 512
BLOCK_M: int = 128  # == NUM_QO_HEADS
BLOCK_N: int = 32
BLOCK_K: int = 32
TILE_M: int = BLOCK_M // NUM_WARPS  # 16
OCCUPANCY: int = 1

SIZE_MLA_WORK_INFO_IN_DW: int = 8


@flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
def kn_mla_fwd_decode_h128_fp8_fp8(
    # --- inputs ---
    query: fx.Tensor,  # [num_seqs * num_heads, qk_head_dim]  (fp8)
    kv_buffer: fx.Tensor,  # [num_pages, qk_head_dim]  (fp8)
    kv_page_indices: fx.Tensor,  # [num_page_used]  (i32)
    # --- metadata ---
    work_indptr: fx.Tensor,  # [num_workers + 1]  (i32)
    work_info_set: fx.Tensor,  # [num_work_items * 8]  (i32)
    # --- outputs ---
    final_output: fx.Tensor,  # [num_seqs * num_heads, v_head_dim]  (bf16)
    split_output: fx.Tensor,  # [num_partial_slots * num_heads, v_head_dim]  (f32)
    split_lse: fx.Tensor,  # [num_partial_slots * num_heads]  (f32)
    # --- parameters ---
    softmax_scale: fx.Constexpr,
):
    """MLA decode forward kernel (nhead=128, fp8/fp8 -> bf16).

    This is a persistent-thread kernel: each workgroup picks up work items
    from ``work_indptr`` / ``work_info_set`` and processes them sequentially.
    """
    pass

    kv_page_indices_rsrc = buffer_ops.create_buffer_resource(kv_page_indices)
    work_indptr_rsrc = buffer_ops.create_buffer_resource(work_indptr)
    work_info_set_rsrc = buffer_ops.create_buffer_resource(work_info_set)

    worker_idx = gpu.block_idx.x
    work_range = buffer_ops.buffer_load(
        work_indptr_rsrc, worker_idx, vec_width=2, dtype=T.i32
    )
    work_start_idx = vector.extract(work_range, [0])
    work_end_idx = vector.extract(work_range, [1])

    for work_idx in range(work_start_idx, work_end_idx):
        # Load MlaWorkInfo dw1..5 (partial_qo_loc, qo_start, qo_end, kv_start, kv_end)
        wi_base = work_idx * SIZE_MLA_WORK_INFO_IN_DW
        wi_dw1_4 = buffer_ops.buffer_load(
            work_info_set_rsrc, wi_base + 1, vec_width=4, dtype=T.i32
        )
        wi_dw5 = buffer_ops.buffer_load(
            work_info_set_rsrc, wi_base + 5, vec_width=1, dtype=T.i32
        )
        partial_qo_loc = vector.extract(wi_dw1_4, [0])
        qo_start = vector.extract(wi_dw1_4, [1])
        qo_end = vector.extract(wi_dw1_4, [2])
        kv_start = vector.extract(wi_dw1_4, [3])
        kv_end = wi_dw5

        # tid = gpu.thread_idx.x
        # if arith.cmpi(CmpIPredicate.eq, tid, arith.constant(5, type=T.i32)):
        #     fx.printf(
        #         "work_idx={} partial_qo_loc={} qo_start={} qo_end={} kv_start={} kv_end={}",
        #         work_idx, partial_qo_loc, qo_start, qo_end, kv_start, kv_end,
        #     )


@flyc.jit
def launch_mla_fwd_decode_h128_fp8_fp8(
    query: fx.Tensor,
    kv_buffer: fx.Tensor,
    kv_page_indices: fx.Tensor,
    work_indptr: fx.Tensor,
    work_info_set: fx.Tensor,
    final_output: fx.Tensor,
    split_output: fx.Tensor,
    split_lse: fx.Tensor,
    softmax_scale: fx.Constexpr,
    num_cus: fx.Constexpr,
    lds_size: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """JIT host function: configures grid/block and launches the kernel."""
    kn_mla_fwd_decode_h128_fp8_fp8(
        query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        final_output,
        split_output,
        split_lse,
        softmax_scale,
    ).launch(
        grid=(num_cus, 1, 1),
        block=(NUM_THREADS, 1, 1),
        # smem=lds_size,
        stream=stream,
    )

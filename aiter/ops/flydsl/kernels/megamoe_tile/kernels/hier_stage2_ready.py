# SPDX-License-Identifier: MIT
"""Persistent weighted GMM2 core producing rank-local source-token partials."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int8, T

from .. import comm_ops
from ..gemm_common import _udiv
from aiter.ops.flydsl.kernels.mxmoe_gemm_v2 import (
    gemm2_body_v2,
    global_typed_ptr,
    issue_a_load_lds_dt,
    kStages,
)

from .hier_sync import (
    increment_i32_system,
    publish_generation_system,
    wait_i64_at_least_system,
)


def compile_hier_stage2_partial_a4w4(
    *,
    D_HIDDEN: int,
    D_INTER: int,
    NE: int,
    TOPK: int,
    BM: int = 32,
    BN: int = 128,
    BK: int = 128,
    waves_per_eu_hint: int = 3,
):
    """Compile compute-only GMM2.

    ``arg_stids`` must encode ``source_flat`` in its low 24 bits and top-k slot
    in its high 8 bits. The atomic epilogue applies routing weight once and
    accumulates into ``arg_out[source_flat, D_HIDDEN]``.
    """

    if BM not in (16, 32, 64, 128):
        raise ValueError("BM must be one of 16,32,64,128")
    if BN not in (128, 256) or BK not in (128, 256):
        raise ValueError("BN/BK must be supported GMM2 tile sizes")
    if D_HIDDEN % BN or D_INTER % BK:
        raise ValueError("H2 dimensions must divide BN/BK")
    if waves_per_eu_hint not in (1, 2, 3, 4):
        raise ValueError("waves_per_eu_hint must be one of 1,2,3,4")

    kh_tile_a = BK // 2
    slot_bytes = BM * kh_tile_a
    c_lds_bytes = BM * BN * 2
    a_stages = 3 if 3 * slot_bytes <= c_lds_bytes else 2
    lds_bytes = max(c_lds_bytes, a_stages * slot_bytes)
    a_slot_alias = a_stages <= kStages
    num_n_blocks = D_HIDDEN // BN

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    name = (
        f"megamoe_tile_h2_ready_a4w4_h{D_HIDDEN}_i{D_INTER}_"
        f"e{NE}_k{TOPK}_bm{BM}_bn{BN}_bk{BK}_wpe{waves_per_eu_hint}_sync2"
    )

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        arg_h1_ready: fx.Int64,
        arg_h2_done: fx.Int64,
        arg_h2_ready: fx.Int64,
        generation: fx.Int64,
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_source_capacity: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))
        k_bytes = fx.Int32(D_INTER // 2)
        aq_num = fx.Int64(i32_max_m_blocks) * fx.Int64(BM) * fx.Int64(k_bytes)
        total_m_blocks = _udiv(global_typed_ptr(arg_cumsum, T.i32)[0], BM)
        bound = total_m_blocks * fx.Int32(num_n_blocks)

        for unit in range(bx, bound, fx.Int32(gpu.grid_dim.x)):
            unit_bx = fx.Int32(unit)
            m_block = _udiv(unit_bx, fx.Int32(num_n_blocks))
            if tx == fx.Int32(0):
                wait_i64_at_least_system(
                    arg_h1_ready + fx.Int64(m_block) * fx.Int64(8), generation
                )
            gpu.barrier()
            comm_ops.fence_system_acquire()

            for slot in range_constexpr(kStages):
                issue_a_load_lds_dt(
                    arg_aq,
                    aq_num,
                    lds_base_i32,
                    slot,
                    slot,
                    m_block * fx.Int32(BM),
                    wave,
                    lane,
                    False,
                    kh_tile_a,
                    k_bytes,
                    BM=BM,
                )
            rocdl.sched_barrier(0)
            gemm2_body_v2(
                lds_base_i32,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                i32_source_capacity,
                i32_max_m_blocks,
                arg_out,
                unit_bx,
                lane,
                wave,
                arg_aq,
                fx.Int32(D_INTER),
                fx.Int32(D_HIDDEN),
                fx.Int32(0),
                fx.Int32(0),
                BM=BM,
                BN=BN,
                BK=BK,
                use_nt=False,
                INTER_MAX=D_INTER,
                g2_kstatic=True,
                aStages=a_stages,
                a_slot_alias=a_slot_alias,
                a_dtype="fp4",
                use_reduce=False,
                topk=TOPK,
                has_pad=False,
                SBM=BM,
                g2_bhoist=True,
                g2_ascale_pf=True,
                g2_bf16_lds=True,
                g2_defer_weight=False,
                g2_out_pitch_align=0,
                g2_scale_blk=8,
                route_out_fp8=False,
                mn_idx=None,
            )

            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tx == fx.Int32(0):
                comm_ops.fence_system_release()
                old = increment_i32_system(arg_h2_done, m_block)
                if old + fx.Int32(1) == fx.Int32(num_n_blocks):
                    publish_generation_system(
                        arg_h2_ready + fx.Int64(m_block) * fx.Int64(8), generation
                    )
            gpu.barrier()

    @flyc.jit
    def launch_sync2(
        arg_h1_ready: fx.Int64,
        arg_h2_done: fx.Int64,
        arg_h2_ready: fx.Int64,
        generation: fx.Int64,
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_source_capacity: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        worker_blocks: fx.Int32,
        arg_out: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            arg_h1_ready,
            arg_h2_done,
            arg_h2_ready,
            generation,
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_source_capacity,
            i32_max_m_blocks,
            arg_out,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(worker_blocks, 1, 1), block=(256, 1, 1), stream=stream)

    launch_sync2.kernel_name = name
    launch_sync2.lds_bytes = lds_bytes
    launch_sync2.num_n_blocks = num_n_blocks
    launch_sync2.output_contract = "weighted-rank-local-source-partial"
    launch_sync2.requires_zeroed_output = True
    return launch_sync2

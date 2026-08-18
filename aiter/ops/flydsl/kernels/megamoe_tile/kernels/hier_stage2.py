# SPDX-License-Identifier: MIT
"""H2 prototype: node-partial copy/return role plus A4W4 GMM2."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int8, T

from aiter.ops.flydsl.kernels import buffer_ops
from .. import comm_ops
from ..gemm_common import _udiv
from aiter.ops.flydsl.kernels.mxmoe_gemm_v2 import (
    gemm2_body_v2,
    global_typed_ptr,
    issue_a_load_lds_dt,
    kStages,
)


def compile_hier_stage2_a4w4(
    *,
    D_HIDDEN: int,
    D_INTER: int,
    NE: int,
    TOPK: int,
    COMM_BLOCKS: int = 1,
    BM: int = 32,
    BN: int = 128,
    BK: int = 128,
):
    """Compile non-persistent linear GMM2 plus one low-ticket comm role."""

    if D_HIDDEN % BN or D_INTER % BK:
        raise ValueError("H2 dimensions must be divisible by BN/BK")
    if COMM_BLOCKS < 1:
        raise ValueError("COMM_BLOCKS must be positive")

    kh_tile_a = BK // 2
    slot_bytes = BM * kh_tile_a
    c_lds_bytes = BM * BN * 2  # bf16 C-shuffle LDS
    a_stages = 3 if 3 * slot_bytes <= c_lds_bytes else 2
    lds_bytes = max(c_lds_bytes, a_stages * slot_bytes)
    a_slot_alias = a_stages <= kStages

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    name = (
        f"megamoe_tile_h2_a4w4_h{D_HIDDEN}_i{D_INTER}_"
        f"e{NE}_k{TOPK}_bm{BM}_bn{BN}_bk{BK}_cb{COMM_BLOCKS}"
    )

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def h2_kernel(
        copy_src: fx.Int64,
        copy_dst: fx.Int64,
        copy_signal: fx.Int64,
        copy_nbytes: fx.Int32,
        generation: fx.Int64,
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))

        if bx < fx.Int32(COMM_BLOCKS):
            src = buffer_ops.create_buffer_resource_from_addr(copy_src)
            dst = buffer_ops.create_buffer_resource_from_addr(copy_dst)
            units = fx.Int32(copy_nbytes) // fx.Int32(16)
            for unit in range(tx, units, fx.Int32(256)):
                elem = unit * fx.Int32(4)
                value = buffer_ops.buffer_load(src, elem, vec_width=4, dtype=T.i32)
                buffer_ops.buffer_store(value, dst, elem)
            tail = units * fx.Int32(16) + tx
            if tail < fx.Int32(copy_nbytes):
                value = buffer_ops.buffer_load(src, tail, vec_width=1, dtype=T.i8)
                buffer_ops.buffer_store(value, dst, tail)
            gpu.barrier()
            if tx == fx.Int32(0):
                comm_ops.store_i64_global_system(copy_signal, generation)
        else:
            unit_bx = bx - fx.Int32(COMM_BLOCKS)
            num_n_blocks = _udiv(fx.Int32(D_HIDDEN), BN)
            m_block = _udiv(unit_bx, num_n_blocks)
            k_bytes = fx.Int32(D_INTER // 2)
            aq_num = fx.Int64(i32_max_m_blocks) * fx.Int64(BM) * fx.Int64(k_bytes)
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
            total_m_blocks = _udiv(global_typed_ptr(arg_cumsum, T.i32)[0], BM)
            bound = total_m_blocks * num_n_blocks
            if unit_bx < bound:
                gemm2_body_v2(
                    lds_base_i32,
                    arg_ascale,
                    arg_bq,
                    arg_bscale,
                    arg_eids,
                    arg_stids,
                    arg_sweights,
                    i32_M,
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

    @flyc.jit
    def launch_h2(
        copy_src: fx.Int64,
        copy_dst: fx.Int64,
        copy_signal: fx.Int64,
        copy_nbytes: fx.Int32,
        generation: fx.Int64,
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_gemm_grid: fx.Int32,
        arg_out: fx.Int64,
        stream: fx.Stream,
    ):
        h2_kernel(
            copy_src,
            copy_dst,
            copy_signal,
            copy_nbytes,
            generation,
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            i32_max_m_blocks,
            arg_out,
        ).launch(
            grid=(fx.Int32(COMM_BLOCKS) + i32_gemm_grid, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    launch_h2.kernel_name = name
    launch_h2.lds_bytes = lds_bytes
    return launch_h2

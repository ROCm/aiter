# SPDX-License-Identifier: MIT
"""Persistent GMM1 core with external plan/tile readiness.

Communication is intentionally absent from this code object. A CCO or SHMEM
sidecar fills the registered arena and publishes readiness; this kernel only
waits, computes GMM1+activation+A4 requant, and publishes H1 output tiles.

Target-shape profiling is sensitive to the occupancy hint: WPE1/2 remain
spill-free, while WPE3 forces register pressure and spills. WPE2 is therefore
the production default; :mod:`hier_stage1_queue` remains an A/B scheduler.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import T

from .. import comm_ops
from ..gemm1 import (
    MXFP4_SCALE_LAYOUT_TAG,
    _bm_constants,
    _gemm1_body,
    _global_i32_at,
    k_tiles_total_for,
    n_out_for,
    num_n_blocks_for,
)

from ..activation import normalize_activation, validate_activation_parameters
from .hier_sync import (
    increment_i32_system,
    publish_generation_system,
    wait_i32_count_system,
    wait_i64_at_least_system,
)


def _tag(value: float) -> str:
    return (
        str(float(value))
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
    )


def compile_hier_stage1_ready_a4w4(
    *,
    D_HIDDEN: int,
    D_INTER: int,
    NE: int,
    TOPK: int,
    BM: int = 32,
    BN: int = 256,
    BK: int = 256,
    use_nt: bool = True,
    waves_per_eu_hint: int = 2,
    wait_plan: bool = False,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
):
    activation = normalize_activation(activation)
    validate_activation_parameters(
        activation=activation,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    if BN != 256 or BK != 256:
        raise ValueError("ready H1 currently requires BN=BK=256")
    if (BM, use_nt) not in {
        (32, True),
        (32, False),
        (64, False),
        (128, False),
    }:
        raise ValueError(f"unsupported ready H1 variant BM={BM}, use_nt={use_nt}")
    if D_HIDDEN % BK or (2 * D_INTER) % BN:
        raise ValueError("ready H1 dimensions must divide BK/BN")
    if waves_per_eu_hint not in (1, 2, 3, 4):
        raise ValueError("waves_per_eu_hint must be one of 1,2,3,4")

    kh_tile = BK // 2
    k_tiles_total = k_tiles_total_for(D_HIDDEN, BK)
    _, _, _, lds_bytes = _bm_constants(BM, BN, kh_tile, k_tiles_total)
    num_n_blocks = num_n_blocks_for(n_out_for(D_INTER), BN)

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    act_tag = activation
    if activation == "swiglu":
        limit = 7.0 if swiglu_limit is None else float(swiglu_limit)
        act_tag += f"_l{_tag(limit)}"
    elif activation == "situv2":
        act_tag += f"_b{_tag(situ_beta)}_lb{_tag(situ_linear_beta)}"
    elif swiglu_limit is not None:
        act_tag += f"_l{_tag(swiglu_limit)}"
    name = (
        f"megamoe_tile_h1_ready_a4w4_{act_tag}_h{D_HIDDEN}_i{D_INTER}_"
        f"e{NE}_k{TOPK}_bm{BM}_wpe{waves_per_eu_hint}_nt{int(use_nt)}_"
        f"{MXFP4_SCALE_LAYOUT_TAG}_pw{int(wait_plan)}"
    )

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        arg_plan_ready: fx.Int64,
        arg_input_ready: fx.Int64,
        arg_input_expected: fx.Int64,
        arg_output_done: fx.Int64,
        arg_output_ready: fx.Int64,
        generation: fx.Int64,
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_raw_ptr = lds.raw.ptr

        # Production launches after the route/count/cumsum sidecar event, so
        # making every CTA poll one plan word would serialize on a hot address.
        # The optional gate is retained for standalone/debug launchers only.
        if const_expr(wait_plan):
            if tx == fx.Int32(0):
                wait_i64_at_least_system(arg_plan_ready, generation)
            gpu.barrier()
            comm_ops.fence_system_acquire()

        total_m_blocks = _global_i32_at(arg_cumsum, fx.Int32(0)) // fx.Int32(BM)
        total_work = total_m_blocks * fx.Int32(num_n_blocks)
        for flat in range(bx, total_work, fx.Int32(gpu.grid_dim.x)):
            tile = fx.Int32(flat)
            m_tile = tile // fx.Int32(num_n_blocks)
            if tx == fx.Int32(0):
                wait_i32_count_system(
                    arg_input_ready, arg_input_expected, m_tile
                )
            gpu.barrier()
            comm_ops.fence_system_acquire()

            _gemm1_body(
                lds_raw_ptr,
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_mind,
                arg_aqout,
                arg_ascaleout,
                arg_hidden,
                tile,
                lane,
                wave,
                use_nt,
                i32_ntok,
                total_m_blocks,
                BM=BM,
                BN=BN,
                BK=BK,
                inline_quant=False,
                K=D_HIDDEN,
                N_OUT=2 * D_INTER,
                NE=NE,
                interleave=False,
                act=activation,
                swiglu_limit=swiglu_limit,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )

            # All N tiles must finish before GMM2 may consume this M tile.
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tx == fx.Int32(0):
                comm_ops.fence_system_release()
                old = increment_i32_system(arg_output_done, m_tile)
                if old + fx.Int32(1) == fx.Int32(num_n_blocks):
                    publish_generation_system(
                        arg_output_ready + fx.Int64(m_tile) * fx.Int64(8),
                        generation,
                    )
            gpu.barrier()

    @flyc.jit
    def launch_sc2_plan_gate(
        arg_plan_ready: fx.Int64,
        arg_input_ready: fx.Int64,
        arg_input_expected: fx.Int64,
        arg_output_done: fx.Int64,
        arg_output_ready: fx.Int64,
        generation: fx.Int64,
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        worker_blocks: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            arg_plan_ready,
            arg_input_ready,
            arg_input_expected,
            arg_output_done,
            arg_output_ready,
            generation,
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_mind,
            i32_ntok,
            arg_aqout,
            arg_ascaleout,
            arg_hidden,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(worker_blocks, 1, 1), block=(256, 1, 1), stream=stream)

    launch_sc2_plan_gate.kernel_name = name
    launch_sc2_plan_gate.lds_bytes = lds_bytes
    launch_sc2_plan_gate.num_n_blocks = num_n_blocks
    launch_sc2_plan_gate.input_ready_kind = "route-count"
    launch_sc2_plan_gate.output_ready_kind = "absolute-generation"
    launch_sc2_plan_gate.wait_plan = wait_plan
    launch_sc2_plan_gate.resource_note = (
        "WPE1/2 profiled spill-free; WPE3 spills at H3584/I384"
    )
    return launch_sc2_plan_gate

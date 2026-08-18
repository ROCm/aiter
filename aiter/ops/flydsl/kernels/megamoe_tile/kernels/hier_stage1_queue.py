# SPDX-License-Identifier: MIT
"""Spill-oriented H1 queue publisher and compute-only consumer.

Exactly one communication sidecar appends ready flat tile IDs and release-stores
the queue tail. Compute CTAs own deterministic queue positions
``block_id + n*grid``; there is no atomic work head and no CCO state in the
MFMA code object.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
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
from .hier_sync import wait_i64_at_least_system


_EPOCH_OFF = 0
_TOTAL_OFF = 8
_TAIL_OFF = 16
_DONE_OFF = 24


def build_h1_ready_queue_publisher(*, num_n_blocks: int, max_work: int):
    """Build queue init/append/finish launchers for one sidecar publisher."""

    if num_n_blocks <= 0 or max_work <= 0:
        raise ValueError("queue geometry must be positive")
    geometry = f"n{int(num_n_blocks)}_w{int(max_work)}"

    @flyc.kernel(
        name=f"megamoe_h1_queue_init_{geometry}", known_block_size=[1, 1, 1]
    )
    def init_kernel(header: fx.Int64, generation: fx.Int64, total_work: fx.Int32):
        total = (total_work < fx.Int32(max_work)).select(
            total_work, fx.Int32(max_work)
        )
        comm_ops.store_i64_global_system(
            header + fx.Int64(_TOTAL_OFF), fx.Int64(total)
        )
        comm_ops.store_i64_global_system(
            header + fx.Int64(_TAIL_OFF), fx.Int64(0)
        )
        comm_ops.store_i64_global_system(
            header + fx.Int64(_DONE_OFF), fx.Int64(0)
        )
        # Epoch is the final publication; consumers may read total/tail after it.
        comm_ops.store_i64_global_system(
            header + fx.Int64(_EPOCH_OFF), generation
        )

    @flyc.kernel(
        name=f"megamoe_h1_queue_publish_{geometry}", known_block_size=[1, 1, 1]
    )
    def publish_kernel(header: fx.Int64, queue: fx.Int64, m_tile: fx.Int32):
        tail = fx.Int64(
            comm_ops.atomic_add_system(
                header + fx.Int64(_TAIL_OFF), fx.Int64(0)
            )
        )
        total = fx.Int64(
            comm_ops.atomic_add_system(
                header + fx.Int64(_TOTAL_OFF), fx.Int64(0)
            )
        )
        queue_rsrc = buffer_ops.create_buffer_resource_from_addr(queue)
        for n in range_constexpr(num_n_blocks):
            pos = tail + fx.Int64(n)
            if (pos < total) & (pos < fx.Int64(max_work)):
                flat = m_tile * fx.Int32(num_n_blocks) + fx.Int32(n)
                buffer_ops.buffer_store(flat, queue_rsrc, fx.Int32(pos))
        rocdl.s_waitcnt(0)
        comm_ops.fence_system_release()
        candidate = tail + fx.Int64(num_n_blocks)
        new_tail = (candidate < total).select(candidate, total)
        comm_ops.store_i64_global_system(
            header + fx.Int64(_TAIL_OFF), new_tail
        )

    @flyc.kernel(
        name=f"megamoe_h1_queue_finish_{geometry}", known_block_size=[1, 1, 1]
    )
    def finish_kernel(header: fx.Int64, generation: fx.Int64):
        comm_ops.store_i64_global_system(
            header + fx.Int64(_DONE_OFF), generation
        )

    @flyc.jit
    def launch_init(
        header: fx.Int64,
        generation: fx.Int64,
        total_work: fx.Int32,
        stream: fx.Stream,
    ):
        init_kernel(header, generation, total_work).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_publish(
        header: fx.Int64,
        queue: fx.Int64,
        m_tile: fx.Int32,
        stream: fx.Stream,
    ):
        publish_kernel(header, queue, m_tile).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_finish(
        header: fx.Int64, generation: fx.Int64, stream: fx.Stream
    ):
        finish_kernel(header, generation).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    return launch_init, launch_publish, launch_finish


def _tag(value: float) -> str:
    return (
        str(float(value))
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
    )


def compile_hier_stage1_queue_a4w4(
    *,
    D_HIDDEN: int,
    D_INTER: int,
    NE: int,
    TOPK: int,
    max_work: int,
    BM: int = 32,
    BN: int = 256,
    BK: int = 256,
    use_nt: bool = True,
    waves_per_eu_hint: int = 2,
    wait_epoch: bool = False,
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
        raise ValueError("queue H1 currently requires BN=BK=256")
    if (BM, use_nt) not in {
        (32, True),
        (32, False),
        (64, False),
        (128, False),
    }:
        raise ValueError(f"unsupported queue H1 variant BM={BM}, use_nt={use_nt}")
    if D_HIDDEN % BK or (2 * D_INTER) % BN:
        raise ValueError("queue H1 dimensions must divide BK/BN")
    if max_work <= 0:
        raise ValueError("max_work must be positive")

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
        f"megamoe_tile_h1_queue_a4w4_{act_tag}_h{D_HIDDEN}_i{D_INTER}_"
        f"e{NE}_k{TOPK}_bm{BM}_mw{max_work}_wpe{waves_per_eu_hint}_"
        f"nt{int(use_nt)}_{MXFP4_SCALE_LAYOUT_TAG}_ew{int(wait_epoch)}"
    )

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        queue_header: fx.Int64,
        queue_entries: fx.Int64,
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
        work_scratch = fx.recast_iter(fx.Int32, lds_raw_ptr)
        work_scratch_view = fx.make_view(work_scratch, fx.make_layout(1, 1))

        # Header init is ordered before the production launch by a stream
        # event. Keep the shared epoch gate only for debug/standalone use.
        if const_expr(wait_epoch):
            if tx == fx.Int32(0):
                wait_i64_at_least_system(
                    queue_header + fx.Int64(_EPOCH_OFF), generation
                )
            gpu.barrier()
            comm_ops.fence_system_acquire()
        total = fx.Int32(
            comm_ops.load_i64_global(queue_header + fx.Int64(_TOTAL_OFF))
        )
        total = (total < fx.Int32(max_work)).select(total, fx.Int32(max_work))
        total_m_blocks = _global_i32_at(arg_cumsum, fx.Int32(0)) // fx.Int32(BM)
        queue_rsrc = buffer_ops.create_buffer_resource_from_addr(queue_entries)

        for qidx in range(bx, total, gpu.grid_dim.x):
            if tx == fx.Int32(0):
                tail = fx.Int64(
                    comm_ops.load_i64_global_system(
                        queue_header + fx.Int64(_TAIL_OFF)
                    )
                )
                while tail <= fx.Int64(qidx):
                    tail = fx.Int64(
                        comm_ops.load_i64_global_system(
                            queue_header + fx.Int64(_TAIL_OFF)
                        )
                    )
                comm_ops.fence_system_acquire()
                flat = buffer_ops.buffer_load(
                    queue_rsrc, fx.Int32(qidx), vec_width=1, dtype=T.i32
                )
                fx.ptr_store(Vec.from_elements([flat], fx.Int32), work_scratch)
            gpu.barrier()
            flat = fx.Int32(Vec(work_scratch_view.load())[0])

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
                flat,
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
            gpu.barrier()

    @flyc.jit
    def launch_sc2_epoch_gate(
        queue_header: fx.Int64,
        queue_entries: fx.Int64,
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
            queue_header,
            queue_entries,
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

    launch_sc2_epoch_gate.kernel_name = name
    launch_sc2_epoch_gate.lds_bytes = lds_bytes
    launch_sc2_epoch_gate.num_n_blocks = num_n_blocks
    launch_sc2_epoch_gate.max_work = max_work
    launch_sc2_epoch_gate.scheduler = "single-publisher-ready-queue"
    launch_sc2_epoch_gate.wait_epoch = wait_epoch
    return launch_sc2_epoch_gate

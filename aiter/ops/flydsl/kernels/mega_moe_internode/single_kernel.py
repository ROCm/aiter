"""Single fused MegaMoEInternode kernel: Stage1 and the fused Stage2 in one launch.

Every CTA of the persistent grid (one 8-wave CTA per CU) runs the Stage1 body
(dispatch, rail, fanout, GEMM1), then the fused Stage2 body (GEMM2 + node
push, then kernel2's node reduce / rail return / combine on the first
``k2_ctas`` CTAs).  The two bodies share one LDS allocation (the larger).

Hand-off instead of the old kernel boundary: a CTA leaving Stage1 waits for
Stage1's tables-final gate (h1_queue_eos == this generation: expert counts,
tile maps, row metadata are published), drops stale L1 lines (L2 once per XCD), and
starts claiming GEMM2 units.  GMM1 runs with ``group_ready``: h1 is stored
write-through and the last of a tile group's GMM1 jobs writes the generation
into that group's ready word, which each GEMM2 unit polls before loading its
A tile.  So GEMM2 starts on finished groups while other CTAs are still in
GMM1 (all CTAs are resident, and a CTA only leaves GMM1 once every job has
been claimed, so every awaited group is in progress).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .. import communication_ops_utils as comm
from . import communication as comm2
from .buffer_addressing import region_local_address
from .stage2 import SK_POLL  # ready-word poll back-off (s_sleep units)
from .stage2_pipeline import compile_stage2_pipeline

# The GMM1-skipping CTAs reach Stage2 early and plan the SDMA layout while
# GMM1 runs; the compute CTAs then find it done (plan_ctx below).
# The hand-off EOS word is polled relaxed (an acquire load invalidates the XCD L2
# on every poll); the per-XCD invalidate below is the acquire. Stage1 then skips
# its per-CTA acquire after the EOS spin.


def compile_single_kernel(stage1_launch, k1_kw: dict, k2_kw: dict, k2_ctas: int):
    s1_body = stage1_launch.body
    pipeline = compile_stage2_pipeline(k1_kw, k2_kw, k2_ctas)
    k1, k2 = pipeline.k1, pipeline.k2
    threads = int(stage1_launch.threads)
    grid = int(stage1_launch.grid)
    if pipeline.threads != threads:
        raise ValueError(
            f"CTA widths differ: stage1 {threads}, stage2 {pipeline.threads}"
        )
    if pipeline.grid != grid:
        raise ValueError(f"grids differ: stage1 {grid}, stage2 {pipeline.grid}")
    gsync = stage1_launch.group_sync
    if gsync is None:
        raise ValueError(
            "the single kernel needs Stage1 compiled with group_ready=True"
        )
    xcd_inv_off = int(stage1_launch.xcd_inv_off)
    buffer_regions = stage1_launch.buffer_regions
    region_indices = {(stage, name): index for stage, name, index in buffer_regions}
    remote_first = pipeline.remote_first
    lds_bytes = max(int(stage1_launch.lds_bytes), int(k1.lds_bytes))
    if lds_bytes > 160 * 1024:
        raise ValueError(f"single kernel LDS {lds_bytes} exceeds 160 KiB")
    kernel_name = (
        "megamoe_one_"
        + stage1_launch.kernel_name[len("megamoe_") :]
        + "__"
        + k1.kernel_name[len("megamoe_") :]
        + "__"
        + k2.kernel_name[len("megamoe_") :]
        + ("_p1q" if remote_first else "")
        + f"_ps{SK_POLL}_xi_epl_rp"
        + (f"_k2c{int(k2_ctas)}" if int(k2_ctas) != 96 else "")
    )

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    # fmt: off
    @flyc.kernel(name=kernel_name, known_block_size=[threads, 1, 1])
    def one(dev_comm: fx.Int64, arena_win: fx.Int64, arena_ptr: fx.Int64,
            x_q: fx.Int64, input_scale: fx.Int64, route_weights: fx.Int64,
            topk_ids: fx.Int64, w1q: fx.Int64, w1scale: fx.Int64, ntokens: fx.Int32,
            arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
            arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64,
            arg_trb: fx.Int64, arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
            i32_inter: fx.Int32, i32_hidden: fx.Int32, arg_arrival_count: fx.Int64,
            i64_generation: fx.Int64, arg_a_tile_map: fx.Int64, i32_local_rank: fx.Int32,
            arg_output_bf16: fx.Int64, arg_scratch: fx.Int64, local_tokens: fx.Int32,
            arg_expert_count: fx.Int64, arg_sd_send: fx.Int64, arg_sd_tb: fx.Int64,
            arg_sd_plan: fx.Int64, arg_sd_cdone: fx.Int64, arg_tr_mcnt: fx.Int64, arg_tr_sdprog: fx.Int64):
    # fmt: on
        lds = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        s1_body(dev_comm, arena_win, arena_ptr, x_q, input_scale, route_weights,
                topk_ids, w1q, w1scale, ntokens, lds_raw=lds)
        # Stage1 left (generation, EOS passed) in LDS words 0/1. Reloading the
        # global epoch/EOS words here queued behind other traffic.
        gpu.barrier()
        gl = Vec(fx.make_view(fx.recast_iter(fx.Int64, lds), fx.make_layout(2, 1)).load())
        gen = fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(gl[0]).ir_value()))
        eos_seen = fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(gl[1]).ir_value()))
        pw = Vec(fx.make_view(fx.add_offset(fx.recast_iter(fx.Int32, lds), 4),
                              fx.make_layout(2, 1)).load())
        plan_ctx = (fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(pw[0]).ir_value())),
                    fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(pw[1]).ir_value())))
        if fx.Int32(gpu.thread_id("x")) == fx.Int32(0):
            # Hand-off (see the module docstring): CTAs that did not pass
            # Stage1's EOS spin wait until its tables are final.
            if eos_seen == fx.Int64(0):
                eos = (arena_ptr + fx.Int64(gsync["eos_off"])
                       + (gen & fx.Int64(1)) * fx.Int64(gsync["eos_plane"]))
                if const_expr(bool(buffer_regions)):
                    eos = region_local_address(arena_ptr, region_indices[1, "h1_queue_eos"], gen & fx.Int64(1))
                comm2.spin_until_ge_i64_sleep_relaxed(eos, gen, SK_POLL)
            # One L2 invalidate per XCD and generation (MegaMoEV2 style): a
            # per-CTA buffer_inv sc1 wipes the XCD's shared L2 32 times and
            # stalls this CTA's first load behind it. The word holds
            # the generation once an invalidate issued after EOS completed.
            xa = arena_ptr + fx.Int64(xcd_inv_off)
            if const_expr(bool(buffer_regions)):
                xa = region_local_address(arena_ptr, region_indices[1, "xcd_l2_inv"], fx.Int64(0))
            xa = xa + fx.Int64(fx.Int32(comm.hw_xcc_id()) & fx.Int32(7)) * fx.Int64(4)
            xv = fx.Int32(comm2.load_i32_global_system_relaxed(xa))
            if xv != fx.Int32(gen):
                comm.fence_agent_acquire()
                comm2.store_i32_global_system_relaxed(xa, fx.Int32(gen))
            comm.invalidate_l1()
        gpu.barrier()
        pipeline.body(arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum,
                      arg_stids, arg_sweights, arg_trb, arg_p2p_comb_inp, i32_max_m_blocks,
                      i32_inter, i32_hidden, arg_arrival_count, i64_generation, arg_a_tile_map,
                      i32_local_rank, dev_comm, arena_win, arena_ptr, arg_output_bf16,
                      arg_scratch, local_tokens, arg_expert_count, arg_sd_send, arg_sd_tb,
                      arg_sd_plan, arg_sd_cdone, arg_tr_mcnt, arg_tr_sdprog,
                      lds_ptr=lds, gen_value=gen,
                      grp_sync=(gsync["sync_off"], gsync["sync_plane"], gsync["tile_group"]),
                      plan_ctx=plan_ctx)

    # fmt: off
    @flyc.jit
    def launch(dev_comm: fx.Int64, arena_win: fx.Int64, arena_ptr: fx.Int64,
               x_q: fx.Int64, input_scale: fx.Int64, route_weights: fx.Int64,
               topk_ids: fx.Int64, w1q: fx.Int64, w1scale: fx.Int64, ntokens: fx.Int32,
               arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
               arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64,
               arg_trb: fx.Int64, arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
               i32_inter: fx.Int32, i32_hidden: fx.Int32, arg_arrival_count: fx.Int64,
               i64_generation: fx.Int64, arg_a_tile_map: fx.Int64, i32_local_rank: fx.Int32,
               arg_output_bf16: fx.Int64, arg_scratch: fx.Int64, local_tokens: fx.Int32,
               arg_expert_count: fx.Int64, arg_sd_send: fx.Int64, arg_sd_tb: fx.Int64,
               arg_sd_plan: fx.Int64, arg_sd_cdone: fx.Int64, arg_tr_mcnt: fx.Int64,
               arg_tr_sdprog: fx.Int64,
               stream: fx.Stream):
    # fmt: on
        one(dev_comm, arena_win, arena_ptr, x_q, input_scale, route_weights, topk_ids,
            w1q, w1scale, ntokens, arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids,
            arg_cumsum, arg_stids, arg_sweights, arg_trb, arg_p2p_comb_inp,
            i32_max_m_blocks, i32_inter, i32_hidden, arg_arrival_count, i64_generation,
            arg_a_tile_map, i32_local_rank, arg_output_bf16, arg_scratch, local_tokens,
            arg_expert_count, arg_sd_send, arg_sd_tb, arg_sd_plan, arg_sd_cdone, arg_tr_mcnt, arg_tr_sdprog,
            value_attrs={"rocdl.waves_per_eu": threads // 256,
                         "rocdl.flat_work_group_size": f"{threads},{threads}"},
            ).launch(grid=(grid, 1, 1), block=(threads, 1, 1), stream=stream)

    launch.kernel_name = kernel_name
    return launch

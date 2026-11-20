"""Single fused MegaMoEInternode kernel: Stage1 and the fused Stage2 in one launch.

Every CTA of the persistent grid (one 8-wave CTA per CU) runs the Stage1 body
(dispatch, rail, fanout, GEMM1), then the fused Stage2 body (GEMM2 + node
push, then kernel2's node reduce / rail return / combine on the first
``k2_ctas`` CTAs).  The two bodies share one LDS allocation (the larger).

Hand-off instead of the old kernel boundary: a CTA leaving Stage1 waits for
Stage1's tables-final gate (h1_queue_eos == this generation: expert counts,
tile maps, row metadata are published), drops stale L1/L2 lines once, and
starts claiming GEMM2 units.  GMM1 runs with ``group_ready``: h1 is stored
write-through and the last of a tile group's GMM1 jobs writes the generation
into that group's ready word, which each GEMM2 unit polls before loading its
A tile.  So GEMM2 starts on finished groups while other CTAs are still in
GMM1 (all CTAs are resident, and a CTA only leaves GMM1 once every job has
been claimed, so every awaited group is in progress).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl

from .. import communication_ops_utils as comm
from . import communication as comm2
from .mega_moe_internode_config import STAGE2_PART1_PROBE
from .stage2_pipeline import compile_stage2_pipeline

_CACHE = {}


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
    epoch_off = int(stage1_launch.epoch_off)
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
        + ("_p1q" if remote_first and STAGE2_PART1_PROBE else "")
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
        # This generation: Stage1 published it in epoch_gate before any CTA
        # left its entry (every CTA spun on it there).
        gen = fx.Int64(comm2.load_i64_global_system(arena_ptr + fx.Int64(epoch_off)))
        # Hand-off (see the module docstring): wait until Stage1's tables are
        # final, drop stale lines once, then GEMM2 waits per tile group.
        if fx.Int32(gpu.thread_id("x")) == fx.Int32(0):
            eos = (arena_ptr + fx.Int64(gsync["eos_off"])
                   + (gen & fx.Int64(1)) * fx.Int64(gsync["eos_plane"]))
            ev = fx.Int64(comm2.load_i64_global_system(eos))
            while ev < gen:
                rocdl.s_sleep(1)
                ev = fx.Int64(comm2.load_i64_global_system(eos))
        gpu.barrier()
        comm.fence_agent_acquire()
        pipeline.body(arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum,
                      arg_stids, arg_sweights, arg_trb, arg_p2p_comb_inp, i32_max_m_blocks,
                      i32_inter, i32_hidden, arg_arrival_count, i64_generation, arg_a_tile_map,
                      i32_local_rank, dev_comm, arena_win, arena_ptr, arg_output_bf16,
                      arg_scratch, local_tokens, arg_expert_count, arg_sd_send, arg_sd_tb,
                      arg_sd_plan, arg_sd_cdone, arg_tr_mcnt, arg_tr_sdprog,
                      lds_ptr=lds,
                      grp_sync=(gsync["sync_off"], gsync["sync_plane"], gsync["tile_group"]))

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


def run_single_kernel(
    *,
    stage1_launch,
    s1_args: tuple,
    k1_ptrs: dict,
    k1_compile: dict,
    k2_compile: dict,
    k2_ctas: int,
    arg_output_bf16,
    arg_scratch,
    local_tokens,
    stream,
    run_compiled,
):
    """Compile (once per configuration) and launch the single fused kernel.

    ``s1_args`` = (dev_comm, arena_win, arena_ptr, x_q, input_scale,
    route_weights, topk_ids, w1q, w1scale, ntokens) as Stage1 takes them.
    """
    key = (
        stage1_launch.kernel_name,
        tuple(sorted(k1_compile.items())),
        tuple(sorted(k2_compile.items())),
        int(k2_ctas),
    )
    launch = _CACHE.get(key)
    if launch is None:
        launch = _CACHE[key] = compile_single_kernel(
            stage1_launch, k1_compile, k2_compile, k2_ctas
        )
    p = k1_ptrs
    from .stage2 import BM

    run_compiled(
        launch,
        *(fx.Int64(int(v)) for v in s1_args[:9]),
        fx.Int32(int(s1_args[9])),
        *(
            fx.Int64(int(p[k]))
            for k in (
                "arg_aq",
                "arg_ascale",
                "arg_bq",
                "arg_bscale",
                "arg_eids",
                "arg_cumsum",
                "arg_stids",
                "arg_sweights",
                "arg_trb",
                "arg_p2p",
            )
        ),
        fx.Int32((p["row_capacity"] + BM - 1) // BM),
        fx.Int32(k1_compile["inter_dim"]),
        fx.Int32(k1_compile["model_dim"]),
        fx.Int64(int(p["arg_arrival_count"])),
        fx.Int64(int(p["arg_generation"])),
        fx.Int64(int(p["arg_a_tile_map"])),
        fx.Int32(int(p["local_rank"])),
        fx.Int64(int(arg_output_bf16)),
        fx.Int64(int(arg_scratch)),
        fx.Int32(int(local_tokens)),
        fx.Int64(int(p.get("arg_expert_count", 0))),
        *(
            fx.Int64(int(p.get(k, 0)))
            for k in (
                "arg_sd_send",
                "arg_sd_tb",
                "arg_sd_plan",
                "arg_sd_cdone",
                "arg_tr_mcnt",
                "arg_tr_sdprog",
            )
        ),
        stream,
    )

"""Shared GEMM2/reduce/RAIL pipeline for one- and two-kernel launches."""

from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu

from .mega_moe_internode_config import STAGE2_PART1_PROBE
from .stage2 import compile_mega_moe_stage2, compile_stage2_node_combine


def compile_stage2_pipeline(k1_kw: dict, k2_kw: dict, k2_ctas: int):
    k1 = compile_mega_moe_stage2(**k1_kw, fused=True)
    k2 = compile_stage2_node_combine(**k2_kw)
    k1_body, k2_body = k1.body, k2.body
    grid = int(k1.grid_ctas)
    threads = int(k1.threads)
    if int(k2.threads) != threads:
        raise ValueError(f"Stage2 CTA widths differ: {threads}, {k2.threads}")
    sdma = bool(k1_kw.get("sdma", False))
    if sdma != bool(k2_kw.get("sdma", False)):
        raise ValueError("kernel1 and kernel2 must agree on sdma")
    k2_ctas = int(k2_ctas)
    if sdma:
        # Leave the plan CTAs outside the blocking combine group.
        k2_ctas = min(k2_ctas, int(k1.sd_plan_bx))
    if not 0 < k2_ctas <= grid:
        raise ValueError(f"invalid combine CTA count {k2_ctas} for grid {grid}")
    remote_first = bool(k1.remote_first)
    tr = bool(k1.token_ready)
    if tr != bool(k2.token_ready):
        raise ValueError("kernel1 and kernel2 must agree on token_ready")

    # fmt: off
    @flyc.jit
    def body(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
             arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64,
             arg_trb: fx.Int64, arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
             i32_inter: fx.Int32, i32_hidden: fx.Int32, arg_arrival_count: fx.Int64,
             i64_generation: fx.Int64, arg_a_tile_map: fx.Int64, i32_local_rank: fx.Int32,
             dev_comm: fx.Int64, arena_win: fx.Int64, arena_ptr: fx.Int64,
             arg_output_bf16: fx.Int64, arg_scratch: fx.Int64, local_tokens: fx.Int32,
             arg_expert_count: fx.Int64, arg_sd_send: fx.Int64, arg_sd_tb: fx.Int64,
             arg_sd_plan: fx.Int64, arg_sd_cdone: fx.Int64, arg_tr_mcnt: fx.Int64,
             arg_tr_sdprog: fx.Int64, lds_ptr=None, grp_sync=None):
    # fmt: on
        bx = fx.Int32(fx.block_idx.x)
        if const_expr(tr):
            k2.tr_init(arena_ptr, local_tokens, bx, fx.Int32(grid))

        def tr_unit(lw, pre, post):
            if const_expr(k1_kw.get("sdma", False)):
                k2.tr_step(dev_comm, arena_win, arena_ptr, local_tokens, bx, lw, False, pre, post,
                           owner_grid=grid)
            else:
                k2.tr_step(dev_comm, arena_win, arena_ptr, local_tokens, bx, lw, False, pre, post)

        def tr_tail(lw):
            # After its GEMM2 tickets: claim and reduce until every position
            # of the remote plane has been claimed.
            more = fx.Int32(1)
            while more != fx.Int32(0):
                if const_expr(k1_kw.get("sdma", False)):
                    more = fx.Int32(k2.tr_step(dev_comm, arena_win, arena_ptr, local_tokens, bx,
                                               lw, True, owner_grid=grid))
                else:
                    more = fx.Int32(k2.tr_step(dev_comm, arena_win, arena_ptr, local_tokens, bx,
                                               lw, True))

        def remote_half_done():
            if bx < fx.Int32(k2_ctas):
                k2_body(dev_comm, arena_win, arena_ptr, arg_output_bf16, arg_scratch,
                        local_tokens, bx, fx.Int32(k2_ctas), part=1)

        def remote_half_probe(word_ptr):
            # LDS word := 1 when part 1 would not wait (always, past kernel2's CTAs).
            if bx < fx.Int32(k2_ctas):
                k2.remote_ready(dev_comm, arena_ptr, arg_scratch, word_ptr)
            else:
                if fx.Int32(gpu.thread_id("x")) == fx.Int32(0):
                    fx.ptr_store(fx.Int32(1), word_ptr)

        k1_body(arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_stids,
                arg_sweights, arg_trb, arg_p2p_comb_inp, i32_max_m_blocks, i32_inter,
                i32_hidden, arg_arrival_count, i64_generation, arg_a_tile_map, i32_local_rank,
                fx.block_idx.x, arg_expert_count=arg_expert_count,
                boundary_hook=remote_half_done if remote_first and not tr else None,
                boundary_probe=(remote_half_probe if remote_first and STAGE2_PART1_PROBE
                                and not tr else None),
                arg_tr_mcnt=arg_tr_mcnt, arg_tr_sdprog=arg_tr_sdprog, unit_hook=tr_unit if tr else None,
                tail_hook=tr_tail if tr else None,
                dev_comm=dev_comm, arg_sd_send=arg_sd_send, arg_sd_tb=arg_sd_tb,
                arg_sd_plan=arg_sd_plan, arg_sd_cdone=arg_sd_cdone,
                arena_win=arena_win, arena_ptr=arena_ptr, lds_ptr=lds_ptr,
                grp_sync=grp_sync)
        if bx < fx.Int32(k2_ctas):
            k2_body(dev_comm, arena_win, arena_ptr, arg_output_bf16, arg_scratch,
                    local_tokens, bx, fx.Int32(k2_ctas), part=2 if remote_first else 0)

    return SimpleNamespace(body=body, k1=k1, k2=k2, grid=grid, threads=threads,
                           k2_ctas=k2_ctas, lds_bytes=int(k1.lds_bytes),
                           remote_first=remote_first, token_ready=tr)

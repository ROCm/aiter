# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Token-id GEMM1 with per-expert payload wait (TP incremental stage1).

Step 2: host pre-fills ``payload_ready == expected`` so the wait returns
immediately. Step 4 fuses min-expert producers into the same grid: consumer
CTAs wait only on ``received[e] == expected[e]`` while producers continue.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr
from flydsl.expr.typing import Vector as Vec

from .. import communication_ops_utils as comm_ops
from ..tensor_shim import _run_compiled
from .gemm1 import _LdsF32View
from .gemm_util import _make_buffer
from .tp_incremental_payload import emit_tp_incremental_payload, wait_expert_payload
from .tp_incremental_push import run_tp_incremental_push
from .tp_incremental_schedule import publish_order_from_topk
from .tp_token_gemm1 import build_token_gemm1, gemm1_token_kernel

WAIT_ON_READY = True


# fmt: off
@functools.cache
def compile_tp_incremental_stage1(
    *, model_dim: int, inter_dim: int, expert_offset: int = 0, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, pipe_weights: bool = True,
    mfma_amajor: bool = False, swizzle_a: bool = True, use_tile_resource: bool = True,
    waves_per_eu_hint: int = 2, b_cache_modifier: int = 0, swiglu_limit: float = 0.0,
):
    # fmt: on
    num_waves = int(num_waves)
    assert num_waves > 1
    assert 1 <= waves_per_eu_hint <= 4
    assert tile_n % num_waves == 0
    assert (2 * inter_dim) % tile_n == 0
    assert tile_k == 256 and model_dim % tile_k == 0

    n_per_wave = tile_n // num_waves
    n_tiles = (2 * inter_dim) // tile_n
    m_repeat = sort_block_m // 16
    num_acc_n = n_per_wave // 16
    assert num_acc_n % 2 == 0 and m_repeat % 2 == 0

    a_k_step_bytes = tile_k
    k_iters = model_dim // tile_k
    total_threads = num_waves * 64
    a_lds_size = sort_block_m * a_k_step_bytes
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n // 2
    lds_pool_bytes = max(2 * a_lds_size, sort_block_m * cs_tile_n * 4)
    n_scale_bytes = sort_block_m * (model_dim // 32)
    swiglu_suffix = (
        "" if swiglu_limit <= 0 else f"_sl{str(float(swiglu_limit)).replace('.', 'p')}"
    )
    kernel_name = (
        f"tp_incr_s1_t{sort_block_m}x{tile_n}x{tile_k}"
        f"_w{num_waves}_pw{int(pipe_weights)}ma{int(mfma_amajor)}"
        f"sw{int(swizzle_a)}_tr{int(use_tile_resource)}wpe{waves_per_eu_hint}"
        f"{swiglu_suffix}"
    )

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[total_threads, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, sorted_ids: fx.Tensor,
        out_scale: fx.Tensor, payload_ready: fx.Tensor, expected: fx.Tensor,
        num_valid: fx.Int32, tokens: fx.Int32, grid_x: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))

        w_rsrc = _make_buffer(w, fx.Int32, 4)
        sw_rsrc = _make_buffer(scale_w, fx.Int32)
        trb_rsrc = _make_buffer(tile_row_base, fx.Int32)
        expert_rsrc = _make_buffer(expert_ids, fx.Int32)
        expected_rsrc = _make_buffer(expected, fx.Int32)
        ready_base = fx.Int64(fx.ptrtoint(fx.get_iter(payload_ready)))
        if const_expr(use_tile_resource):
            out_rsrc = None
        else:
            out_rsrc = _make_buffer(
                out, fx.Int16, max_size=False,
                num_records_bytes=num_valid * fx.Int32(inter_dim),
            )
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_rsrc = _make_buffer(
            out_scale,
            fx.Int8,
            max_size=False,
            num_records_bytes=num_valid * fx.Int32(scale_cols) + fx.Int32(8192),
        )
        wave_id = tid // 64

        expert_of_flat, run_tile = build_token_gemm1(
            x_tensor=x, w_rsrc=w_rsrc, sw_rsrc=sw_rsrc, scale_x=scale_x,
            sorted_ids=sorted_ids, tokens=tokens, out_rsrc=out_rsrc, os_rsrc=os_rsrc,
            trb_rsrc=trb_rsrc, expert_rsrc=expert_rsrc, out_tensor=out, a_buf=a_buf,
            a_scale_lds=a_scale_lds, c_tile=c_tile, model_dim=model_dim, inter_dim=inter_dim,
            sort_block_m=sort_block_m, tile_n=tile_n, num_waves=num_waves, n_per_wave=n_per_wave,
            wave_id=wave_id, m_repeat=m_repeat, num_acc_n=num_acc_n, a_k_step_bytes=a_k_step_bytes,
            total_threads=total_threads, k_iters=k_iters, a_lds_i32=a_lds_i32, n_tiles=n_tiles,
            expert_offset=expert_offset, b_cache_modifier=b_cache_modifier, swizzle_a=swizzle_a,
            pipe_weights=pipe_weights, mfma_amajor=mfma_amajor,
            use_tile_resource=use_tile_resource, swiglu_limit=swiglu_limit,
        )
        total_work = (num_valid // fx.Int32(sort_block_m)) * fx.Int32(n_tiles)
        for flat in range(fx.block_idx.x, total_work, grid_x):
            if const_expr(WAIT_ON_READY):
                if tid == fx.Int32(0):
                    wait_expert_payload(ready_base, expected_rsrc, expert_of_flat(flat))
                fx.barrier()
            run_tile(flat)

    @flyc.jit
    def launch(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, sorted_ids: fx.Tensor,
        out_scale: fx.Tensor, payload_ready: fx.Tensor, expected: fx.Tensor,
        num_valid: fx.Int32, tokens: fx.Int32, grid_x: fx.Int32, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, tile_row_base, expert_ids, sorted_ids, out_scale,
            payload_ready, expected, num_valid, tokens, grid_x,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{total_threads},{total_threads}",
            },
        ).launch(grid=(fx.Int64(grid_x), 1, 1), block=(total_threads, 1, 1), stream=stream)

    return launch


# fmt: off
def gemm1_incremental_kernel(
    out, x, w, scale_x, scale_w, tile_row_base, expert_ids, sorted_ids, out_scale,
    payload_ready, expected, num_valid, tokens, stream, *, model_dim: int, inter_dim: int,
    expert_offset: int = 0, sort_block_m: int = 32, tile_n: int = 256, tile_k: int = 256,
    num_waves: int = 4, grid_mult: int = 4, pipe_weights: bool = True,
    mfma_amajor: bool = False, swizzle_a: bool = True, use_tile_resource: bool = True,
    waves_per_eu_hint: int = 2, num_cu: int = 256, b_cache_modifier: int = 0,
    swiglu_limit: float = 0.0,
):
    # fmt: on
    """GEMM1 with per-expert ready wait. Host must pre-fill ready for Step 2."""
    num_valid = int(num_valid)
    tokens = int(tokens)
    if num_valid < 0 or num_valid % int(sort_block_m):
        raise ValueError("num_valid must be a non-negative multiple of sort_block_m")
    if num_valid == 0:
        return out, out_scale
    n_tiles = (2 * int(inter_dim)) // int(tile_n)
    total_work = (num_valid // int(sort_block_m)) * n_tiles
    grid_x = min(total_work, int(num_cu) * int(grid_mult))
    launch = compile_tp_incremental_stage1(
        model_dim=model_dim, inter_dim=inter_dim, expert_offset=expert_offset,
        sort_block_m=sort_block_m, tile_n=tile_n, tile_k=tile_k, num_waves=num_waves,
        pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, swizzle_a=swizzle_a,
        use_tile_resource=use_tile_resource, waves_per_eu_hint=waves_per_eu_hint,
        b_cache_modifier=b_cache_modifier, swiglu_limit=swiglu_limit,
    )
    _run_compiled(
        launch, out, x, w.view(torch.uint8), scale_x, scale_w.view(torch.uint8),
        tile_row_base, expert_ids, sorted_ids, out_scale, payload_ready, expected,
        fx.Int32(num_valid), fx.Int32(tokens), fx.Int32(grid_x), stream,
    )
    return out, out_scale


# fmt: off
@functools.cache
def compile_tp_incremental_fused(
    *, model_dim: int, inter_dim: int, npes: int, topk: int, num_producers: int,
    row_major: bool = False, expert_offset: int = 0, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, pipe_weights: bool = True,
    mfma_amajor: bool = False, swizzle_a: bool = True, use_tile_resource: bool = True,
    waves_per_eu_hint: int = 2, b_cache_modifier: int = 0, swiglu_limit: float = 0.0,
):
    # fmt: on
    """Producer CTAs push; every CTA then steals GEMM tiles waiting per expert."""
    num_waves = int(num_waves)
    num_producers = int(num_producers)
    assert num_waves > 1
    assert num_producers > 0
    assert 1 <= waves_per_eu_hint <= 4
    assert tile_n % num_waves == 0
    assert (2 * inter_dim) % tile_n == 0
    assert tile_k == 256 and model_dim % tile_k == 0

    n_per_wave = tile_n // num_waves
    n_tiles = (2 * inter_dim) // tile_n
    m_repeat = sort_block_m // 16
    num_acc_n = n_per_wave // 16
    assert num_acc_n % 2 == 0 and m_repeat % 2 == 0

    a_k_step_bytes = tile_k
    k_iters = model_dim // tile_k
    total_threads = num_waves * 64
    a_lds_size = sort_block_m * a_k_step_bytes
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n // 2
    lds_pool_bytes = max(2 * a_lds_size, sort_block_m * cs_tile_n * 4)
    n_scale_bytes = sort_block_m * (model_dim // 32)
    swiglu_suffix = (
        "" if swiglu_limit <= 0 else f"_sl{str(float(swiglu_limit)).replace('.', 'p')}"
    )
    kernel_name = (
        f"tp_incr_fused_t{sort_block_m}x{tile_n}x{tile_k}"
        f"_p{npes}_k{topk}_np{num_producers}_rm{int(row_major)}"
        f"_w{num_waves}_pw{int(pipe_weights)}ma{int(mfma_amajor)}"
        f"sw{int(swizzle_a)}_tr{int(use_tile_resource)}wpe{waves_per_eu_hint}"
        f"{swiglu_suffix}"
    )

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[total_threads, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, sorted_ids: fx.Tensor,
        out_scale: fx.Tensor, payload_ready: fx.Tensor, expected: fx.Tensor,
        local_x: fx.Tensor, local_scale: fx.Tensor, local_ids: fx.Tensor,
        publish_order: fx.Tensor, p2p_rx: fx.Tensor, p2p_scale: fx.Tensor,
        p2p_received: fx.Tensor, p2p_ranks_done: fx.Tensor, ranks_done: fx.Tensor,
        local_prod_done: fx.Tensor, work_cursor: fx.Tensor, expert0_done: fx.Tensor,
        overlap: fx.Tensor, num_valid: fx.Int32, tokens: fx.Int32, rank: fx.Int32,
        m_local: fx.Int32, skew_rank: fx.Int32, skew_split: fx.Int32,
        skew_sleeps: fx.Int32, grid_x: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))
        work_scratch = fx.recast_iter(fx.Int32, a_buf.ptr)
        work_scratch_view = fx.make_view(work_scratch, fx.make_layout(1, 1))

        w_rsrc = _make_buffer(w, fx.Int32, 4)
        sw_rsrc = _make_buffer(scale_w, fx.Int32)
        trb_rsrc = _make_buffer(tile_row_base, fx.Int32)
        expert_rsrc = _make_buffer(expert_ids, fx.Int32)
        expected_rsrc = _make_buffer(expected, fx.Int32)
        ready_base = fx.Int64(fx.ptrtoint(fx.get_iter(payload_ready)))
        work_addr = fx.Int64(fx.ptrtoint(fx.get_iter(work_cursor)))
        expert0_addr = fx.Int64(fx.ptrtoint(fx.get_iter(expert0_done)))
        if const_expr(use_tile_resource):
            out_rsrc = None
        else:
            out_rsrc = _make_buffer(
                out, fx.Int16, max_size=False,
                num_records_bytes=num_valid * fx.Int32(inter_dim),
            )
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_rsrc = _make_buffer(
            out_scale,
            fx.Int8,
            max_size=False,
            num_records_bytes=num_valid * fx.Int32(scale_cols) + fx.Int32(8192),
        )
        wave_id = tid // 64

        expert_of_flat, run_tile = build_token_gemm1(
            x_tensor=x, w_rsrc=w_rsrc, sw_rsrc=sw_rsrc, scale_x=scale_x,
            sorted_ids=sorted_ids, tokens=tokens, out_rsrc=out_rsrc, os_rsrc=os_rsrc,
            trb_rsrc=trb_rsrc, expert_rsrc=expert_rsrc, out_tensor=out, a_buf=a_buf,
            a_scale_lds=a_scale_lds, c_tile=c_tile, model_dim=model_dim, inter_dim=inter_dim,
            sort_block_m=sort_block_m, tile_n=tile_n, num_waves=num_waves, n_per_wave=n_per_wave,
            wave_id=wave_id, m_repeat=m_repeat, num_acc_n=num_acc_n, a_k_step_bytes=a_k_step_bytes,
            total_threads=total_threads, k_iters=k_iters, a_lds_i32=a_lds_i32, n_tiles=n_tiles,
            expert_offset=expert_offset, b_cache_modifier=b_cache_modifier, swizzle_a=swizzle_a,
            pipe_weights=pipe_weights, mfma_amajor=mfma_amajor,
            use_tile_resource=use_tile_resource, swiglu_limit=swiglu_limit,
        )
        total_work = (num_valid // fx.Int32(sort_block_m)) * fx.Int32(n_tiles)

        if fx.Int32(fx.block_idx.x) < fx.Int32(num_producers):
            emit_tp_incremental_payload(
                npes=npes,
                rank=rank,
                topk=topk,
                model_dim=model_dim,
                row_major=row_major,
                producer_slot=fx.Int32(fx.block_idx.x),
                num_producers=num_producers,
                m_local=m_local,
                addr_in_tok=fx.Int64(fx.ptrtoint(fx.get_iter(local_x))),
                addr_in_sc=fx.Int64(fx.ptrtoint(fx.get_iter(local_scale))),
                addr_ids=fx.Int64(fx.ptrtoint(fx.get_iter(local_ids))),
                addr_order=fx.Int64(fx.ptrtoint(fx.get_iter(publish_order))),
                addr_p2p_rx=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_rx))),
                addr_p2p_sc=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_scale))),
                addr_p2p_received=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_received))),
                addr_p2p_ranks_done=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_ranks_done))),
                addr_local_prod_done=fx.Int64(fx.ptrtoint(fx.get_iter(local_prod_done))),
                skew_rank=skew_rank,
                skew_split=skew_split,
                skew_sleeps=skew_sleeps,
                addr_expert0_done=expert0_addr,
                addr_overlap=fx.Int64(fx.ptrtoint(fx.get_iter(overlap))),
            )

        consumer_active = fx.Int32(1)
        while consumer_active != fx.Int32(0):
            if tid == fx.Int32(0):
                work = fx.Int32(comm_ops.atomic_add_agent(work_addr, fx.Int32(1)))
                fx.ptr_store(Vec.from_elements([work], fx.Int32), work_scratch)
            fx.barrier()
            work = Vec(work_scratch_view.load())[0]
            if tid == fx.Int32(0):
                has_work = (work < total_work).select(fx.Int32(1), fx.Int32(0))
                if has_work != fx.Int32(0):
                    wait_expert_payload(ready_base, expected_rsrc, expert_of_flat(work))
                fx.ptr_store(Vec.from_elements([has_work], fx.Int32), work_scratch)
            fx.barrier()
            has_work = Vec(work_scratch_view.load())[0]
            if has_work != fx.Int32(0):
                comm_ops.fence_system_acquire()
                run_tile(work)
                if (tid == fx.Int32(0)) & (expert_of_flat(work) == fx.Int32(0)):
                    comm_ops.store_i32_system(
                        expert0_addr, fx.Int32(0), fx.Int32(1)
                    )
            consumer_active = has_work

    @flyc.jit
    def launch(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, sorted_ids: fx.Tensor,
        out_scale: fx.Tensor, payload_ready: fx.Tensor, expected: fx.Tensor,
        local_x: fx.Tensor, local_scale: fx.Tensor, local_ids: fx.Tensor,
        publish_order: fx.Tensor, p2p_rx: fx.Tensor, p2p_scale: fx.Tensor,
        p2p_received: fx.Tensor, p2p_ranks_done: fx.Tensor, ranks_done: fx.Tensor,
        local_prod_done: fx.Tensor, work_cursor: fx.Tensor, expert0_done: fx.Tensor,
        overlap: fx.Tensor, num_valid: fx.Int32, tokens: fx.Int32, rank: fx.Int32,
        m_local: fx.Int32, skew_rank: fx.Int32, skew_split: fx.Int32,
        skew_sleeps: fx.Int32, grid_x: fx.Int32, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, tile_row_base, expert_ids, sorted_ids, out_scale,
            payload_ready, expected, local_x, local_scale, local_ids, publish_order,
            p2p_rx, p2p_scale, p2p_received, p2p_ranks_done, ranks_done, local_prod_done,
            work_cursor, expert0_done, overlap, num_valid, tokens, rank, m_local,
            skew_rank, skew_split, skew_sleeps, grid_x,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{total_threads},{total_threads}",
            },
        ).launch(grid=(fx.Int64(grid_x), 1, 1), block=(total_threads, 1, 1), stream=stream)

    return launch


# fmt: off
def run_tp_incremental_fused(
    workspace, local_x, local_scale, local_ids, out, w, scale_w, tile_row_base,
    expert_ids, sorted_ids, out_scale, expected, num_valid, tokens, m_local, stream,
    *, model_dim: int, inter_dim: int, expert_offset: int = 0, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, grid_mult: int = 4,
    pipe_weights: bool = True, mfma_amajor: bool = False, swizzle_a: bool = True,
    use_tile_resource: bool = True, waves_per_eu_hint: int = 2, num_cu: int = 256,
    b_cache_modifier: int = 0, swiglu_limit: float = 0.0, row_major: bool = False,
    num_producers: int | None = None, skew_rank: int = -1, skew_split: int = 0,
    skew_sleeps: int = 0, publish_order: torch.Tensor | None = None,
):
    # fmt: on
    """Fused min-expert push + per-expert GEMM1. Overlap is the Step 4 contract."""
    num_valid = int(num_valid)
    tokens = int(tokens)
    m_local = int(m_local)
    producers = int(num_producers or 8)
    if num_valid < 0 or num_valid % int(sort_block_m):
        raise ValueError("num_valid must be a non-negative multiple of sort_block_m")
    if num_valid == 0 or m_local <= 0:
        return out, out_scale
    workspace.work_cursor.zero_()
    workspace.expert0_done.zero_()
    workspace.overlap.zero_()
    workspace.local_prod_done.zero_()
    n_tiles = (2 * int(inter_dim)) // int(tile_n)
    total_work = (num_valid // int(sort_block_m)) * n_tiles
    grid_x = max(producers, min(total_work, int(num_cu) * int(grid_mult)))
    order = (
        publish_order
        if publish_order is not None
        else publish_order_from_topk(local_ids).to(torch.int32).contiguous()
    )
    launch = compile_tp_incremental_fused(
        model_dim=model_dim, inter_dim=inter_dim, npes=workspace.npes,
        topk=int(local_ids.shape[1]), num_producers=producers, row_major=bool(row_major),
        expert_offset=expert_offset, sort_block_m=sort_block_m, tile_n=tile_n,
        tile_k=tile_k, num_waves=num_waves, pipe_weights=pipe_weights,
        mfma_amajor=mfma_amajor, swizzle_a=swizzle_a, use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, b_cache_modifier=b_cache_modifier,
        swiglu_limit=swiglu_limit,
    )
    rx = workspace.rx[: tokens + 1]
    rx_scale = workspace.rx_scale[: tokens + 1]
    _run_compiled(
        launch, out, rx, w.view(torch.uint8), rx_scale, scale_w.view(torch.uint8),
        tile_row_base, expert_ids, sorted_ids, out_scale, workspace.received, expected,
        local_x.contiguous().view(torch.uint8),
        local_scale.contiguous().view(torch.uint8),
        local_ids.to(torch.int32).contiguous(), order, workspace.p2p_rx,
        workspace.p2p_scale, workspace.p2p_received, workspace.p2p_ranks_done,
        workspace.ranks_done, workspace.local_prod_done, workspace.work_cursor,
        workspace.expert0_done, workspace.overlap, fx.Int32(num_valid), fx.Int32(tokens),
        fx.Int32(workspace.rank), fx.Int32(m_local), fx.Int32(int(skew_rank)),
        fx.Int32(int(skew_split)), fx.Int32(int(skew_sleeps)), fx.Int32(grid_x), stream,
    )
    return out, out_scale


def run_tp_two_launch_stage1(
    workspace,
    local_x,
    local_scale,
    local_ids,
    out,
    w,
    scale_w,
    tile_row_base,
    expert_ids,
    sorted_ids,
    out_scale,
    num_valid,
    tokens,
    m_local,
    stream,
    *,
    row_major: bool = False,
    num_producers: int | None = None,
    num_waves: int | None = None,
    **gemm_kwargs,
):
    """Bulk P2P gather then token-id GEMM1. Product stage1 for Step B."""
    run_tp_incremental_push(
        workspace,
        local_x,
        local_scale,
        local_ids,
        m_local,
        stream,
        row_major=row_major,
        num_producers=num_producers,
        num_waves=num_waves,
    )
    tokens = int(tokens)
    return gemm1_token_kernel(
        out,
        workspace.rx[: tokens + 1],
        w,
        workspace.rx_scale[: tokens + 1],
        scale_w,
        tile_row_base,
        expert_ids,
        sorted_ids,
        out_scale,
        num_valid,
        tokens,
        stream,
        **gemm_kwargs,
    )

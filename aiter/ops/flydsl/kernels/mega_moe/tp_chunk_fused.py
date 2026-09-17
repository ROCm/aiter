# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Fused chunked TP all-gather + token-id GEMM1 (first-send first-compute).

Producers emit ordered dense chunks like EP dispatch. Consumers wait on
``chunk_ready[chunk(m_tile)] == epoch * npes`` and run the same token-id GEMM1.
Epoch / ``launch_ready`` live in-kernel so CUDA Graph replay needs no host zero.
``early_compute=True`` maps early M-tiles onto early chunks (overlap upper
bound; A contents for those tiles may be incomplete). ``early_compute=False``
waits for the last chunk so GEMM1 matches bulk all-gather.
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr
from flydsl.expr.typing import Vector as Vec

from .. import communication_ops_utils as comm_ops
from ..tensor_shim import _run_compiled
from .gemm1 import _LdsF32View
from .gemm_util import _make_buffer
from .tp_chunk_payload import begin_tp_chunk_epoch, emit_tp_chunk_payload, wait_chunk_payload
from .tp_incremental_schedule import num_publish_chunks
from .tp_token_gemm1 import build_token_gemm1

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHUNK_ROWS = 32
DEFAULT_NUM_PRODUCERS = 32


def _register_chunk_fused_source_dir():
    try:
        from flydsl.compiler.jit_function import EXTRA_SOURCE_DIRS
    except ImportError:
        EXTRA_SOURCE_DIRS = None
    if EXTRA_SOURCE_DIRS is not None and _HERE not in EXTRA_SOURCE_DIRS:
        EXTRA_SOURCE_DIRS.append(_HERE)
    cur = os.environ.get("FLYDSL_EXTRA_SOURCE_DIRS", "")
    parts = [p for p in cur.split(":") if p]
    if _HERE not in parts:
        parts.append(_HERE)
        os.environ["FLYDSL_EXTRA_SOURCE_DIRS"] = ":".join(parts)


_register_chunk_fused_source_dir()


def _chunk_fused_grid(total_work, producers, num_cu, grid_mult):
    """Keep dedicated consumer CTAs while producers send, like MegaMoE stage1."""
    producers = int(producers)
    total_work = int(total_work)
    cu_grid = int(num_cu) * int(grid_mult)
    spare = max(cu_grid - producers, 1)
    consumers = min(total_work, spare) if total_work else 0
    return producers + consumers


# fmt: off
@functools.cache
def compile_tp_chunk_fused(
    *, model_dim: int, inter_dim: int, npes: int, num_producers: int,
    chunk_rows: int, early_compute: bool = True, row_major: bool = False,
    expert_offset: int = 0, sort_block_m: int = 32, tile_n: int = 256,
    tile_k: int = 256, num_waves: int = 4, pipe_weights: bool = True,
    mfma_amajor: bool = False, swizzle_a: bool = True, use_tile_resource: bool = True,
    waves_per_eu_hint: int = 2, b_cache_modifier: int = 0, swiglu_limit: float = 0.0,
):
    # fmt: on
    num_waves = int(num_waves)
    num_producers = int(num_producers)
    chunk_rows = int(chunk_rows)
    npes = int(npes)
    assert num_waves > 1
    assert num_producers > 0 and num_producers % npes == 0
    assert chunk_rows > 0
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
        f"tp_chunk_fused_t{sort_block_m}x{tile_n}x{tile_k}"
        f"_p{npes}_np{num_producers}_cr{chunk_rows}_ec{int(early_compute)}"
        f"_rm{int(row_major)}_w{num_waves}_pw{int(pipe_weights)}ma{int(mfma_amajor)}"
        f"sw{int(swizzle_a)}_tr{int(use_tile_resource)}wpe{waves_per_eu_hint}"
        f"{swiglu_suffix}_epoch"
    )

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[total_threads, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, sorted_ids: fx.Tensor,
        out_scale: fx.Tensor, chunk_ready: fx.Tensor, local_x: fx.Tensor,
        local_scale: fx.Tensor, p2p_rx: fx.Tensor, p2p_scale: fx.Tensor,
        p2p_chunk_ready: fx.Tensor, p2p_launch_ready: fx.Tensor,
        local_chunk_done: fx.Tensor, work_cursor: fx.Tensor, launch_ready: fx.Tensor,
        epoch_gate: fx.Tensor, entry_count: fx.Tensor, num_valid: fx.Int32,
        tokens: fx.Int32, rank: fx.Int32, m_local: fx.Int32, grid_x: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))
        work_scratch = fx.recast_iter(fx.Int32, a_buf.ptr)
        work_scratch_view = fx.make_view(work_scratch, fx.make_layout(1, 1))
        ticket_scratch = fx.recast_iter(fx.Int64, a_buf.ptr)
        ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))

        w_rsrc = _make_buffer(w, fx.Int32, 4)
        sw_rsrc = _make_buffer(scale_w, fx.Int32)
        trb_rsrc = _make_buffer(tile_row_base, fx.Int32)
        expert_rsrc = _make_buffer(expert_ids, fx.Int32)
        ready_base = fx.Int64(fx.ptrtoint(fx.get_iter(chunk_ready)))
        work_addr = fx.Int64(fx.ptrtoint(fx.get_iter(work_cursor)))
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

        _, run_tile = build_token_gemm1(
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
        n_m = num_valid // fx.Int32(sort_block_m)
        chunk_h = fx.Int32(chunk_rows)
        num_chunks = (m_local + chunk_h - fx.Int32(1)) // chunk_h
        last_chunk = num_chunks - fx.Int32(1)

        if tid == fx.Int32(0):
            ticket64 = fx.Int64(
                comm_ops.atomic_add_agent(
                    fx.Int64(fx.ptrtoint(fx.get_iter(entry_count))), fx.Int64(1)
                )
            )
            fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
        fx.barrier()
        ticket64 = Vec(ticket_view.load())[0]
        gate_epoch = fx.Int32(ticket64 // fx.Int64(grid_x) + fx.Int64(1))
        payload_expected = gate_epoch * fx.Int32(npes)
        begin_tp_chunk_epoch(
            npes=npes,
            rank=rank,
            tid=tid,
            is_owner=(fx.Int32(fx.block_idx.x) == fx.Int32(0)).select(
                fx.Int32(1), fx.Int32(0)
            ),
            gate_epoch=gate_epoch,
            addr_epoch_gate=fx.Int64(fx.ptrtoint(fx.get_iter(epoch_gate))),
            addr_launch_ready=fx.Int64(fx.ptrtoint(fx.get_iter(launch_ready))),
            addr_p2p_launch_ready=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_launch_ready))),
            addr_work_cursor=work_addr,
            zero_work_cursor=True,
            addr_local_chunk_done=fx.Int64(fx.ptrtoint(fx.get_iter(local_chunk_done))),
            num_done_flags=fx.Int32(npes) * num_chunks,
            total_threads=total_threads,
        )

        if fx.Int32(fx.block_idx.x) < fx.Int32(num_producers):
            emit_tp_chunk_payload(
                npes=npes,
                rank=rank,
                model_dim=model_dim,
                row_major=row_major,
                producer_slot=fx.Int32(fx.block_idx.x),
                num_producers=num_producers,
                num_waves=num_waves,
                chunk_rows=chunk_rows,
                m_local=m_local,
                addr_in_tok=fx.Int64(fx.ptrtoint(fx.get_iter(local_x))),
                addr_in_sc=fx.Int64(fx.ptrtoint(fx.get_iter(local_scale))),
                addr_p2p_rx=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_rx))),
                addr_p2p_sc=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_scale))),
                addr_p2p_chunk_ready=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_chunk_ready))),
                addr_local_chunk_done=fx.Int64(fx.ptrtoint(fx.get_iter(local_chunk_done))),
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
                    m_tile = work // fx.Int32(n_tiles)
                    if const_expr(early_compute):
                        ch = (m_tile * num_chunks) // n_m
                        ch = (ch < last_chunk).select(ch, last_chunk)
                    else:
                        ch = last_chunk
                    wait_chunk_payload(ready_base, ch, payload_expected)
                fx.ptr_store(Vec.from_elements([has_work], fx.Int32), work_scratch)
            fx.barrier()
            has_work = Vec(work_scratch_view.load())[0]
            if has_work != fx.Int32(0):
                comm_ops.fence_system_acquire()
                run_tile(work)
            consumer_active = has_work

    @flyc.jit
    def launch(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, sorted_ids: fx.Tensor,
        out_scale: fx.Tensor, chunk_ready: fx.Tensor, local_x: fx.Tensor,
        local_scale: fx.Tensor, p2p_rx: fx.Tensor, p2p_scale: fx.Tensor,
        p2p_chunk_ready: fx.Tensor, p2p_launch_ready: fx.Tensor,
        local_chunk_done: fx.Tensor, work_cursor: fx.Tensor, launch_ready: fx.Tensor,
        epoch_gate: fx.Tensor, entry_count: fx.Tensor, num_valid: fx.Int32,
        tokens: fx.Int32, rank: fx.Int32, m_local: fx.Int32, grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, tile_row_base, expert_ids, sorted_ids, out_scale,
            chunk_ready, local_x, local_scale, p2p_rx, p2p_scale, p2p_chunk_ready,
            p2p_launch_ready, local_chunk_done, work_cursor, launch_ready, epoch_gate,
            entry_count, num_valid, tokens, rank, m_local, grid_x,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{total_threads},{total_threads}",
            },
        ).launch(grid=(fx.Int64(grid_x), 1, 1), block=(total_threads, 1, 1), stream=stream)

    return launch


# fmt: off
def run_tp_chunk_fused(
    workspace, local_x, local_scale, out, w, scale_w, tile_row_base, expert_ids,
    sorted_ids, out_scale, num_valid, tokens, m_local, stream, *, model_dim: int,
    inter_dim: int, expert_offset: int = 0, sort_block_m: int = 32, tile_n: int = 256,
    tile_k: int = 256, num_waves: int = 4, grid_mult: int = 4, pipe_weights: bool = True,
    mfma_amajor: bool = False, swizzle_a: bool = True, use_tile_resource: bool = True,
    waves_per_eu_hint: int = 2, num_cu: int = 256, b_cache_modifier: int = 0,
    swiglu_limit: float = 0.0, row_major: bool = False, num_producers: int | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS, early_compute: bool = True,
):
    # fmt: on
    """Fused dest-sharded chunked push + GEMM1. ``early_compute`` is the overlap bound.

    CUDA Graph safe: epoch / launch_ready live in the kernel. Host should not
    zero ``chunk_ready`` between replays.
    """
    num_valid = int(num_valid)
    tokens = int(tokens)
    m_local = int(m_local)
    producers = int(num_producers or DEFAULT_NUM_PRODUCERS)
    chunk_rows = int(chunk_rows)
    if num_valid < 0 or num_valid % int(sort_block_m):
        raise ValueError("num_valid must be a non-negative multiple of sort_block_m")
    if num_valid == 0 or m_local <= 0:
        return out, out_scale
    n_chunks = num_publish_chunks(m_local, chunk_rows)
    if n_chunks > workspace.max_chunks:
        raise ValueError(
            f"num_chunks={n_chunks} exceeds workspace max_chunks={workspace.max_chunks}"
        )
    n_tiles = (2 * int(inter_dim)) // int(tile_n)
    total_work = (num_valid // int(sort_block_m)) * n_tiles
    grid_x = _chunk_fused_grid(total_work, producers, num_cu, grid_mult)
    launch = compile_tp_chunk_fused(
        model_dim=model_dim, inter_dim=inter_dim, npes=workspace.npes,
        num_producers=producers, chunk_rows=chunk_rows, early_compute=bool(early_compute),
        row_major=bool(row_major), expert_offset=expert_offset, sort_block_m=sort_block_m,
        tile_n=tile_n, tile_k=tile_k, num_waves=num_waves, pipe_weights=pipe_weights,
        mfma_amajor=mfma_amajor, swizzle_a=swizzle_a, use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, b_cache_modifier=b_cache_modifier,
        swiglu_limit=swiglu_limit,
    )
    rx = workspace.rx[: tokens + 1]
    rx_scale = workspace.rx_scale[: tokens + 1]
    _run_compiled(
        launch, out, rx, w.view(torch.uint8), rx_scale, scale_w.view(torch.uint8),
        tile_row_base, expert_ids, sorted_ids, out_scale, workspace.chunk_ready,
        local_x.contiguous().view(torch.uint8),
        local_scale.contiguous().view(torch.uint8),
        workspace.p2p_rx, workspace.p2p_scale, workspace.p2p_chunk_ready,
        workspace.p2p_launch_ready, workspace.local_chunk_done, workspace.work_cursor,
        workspace.launch_ready, workspace.epoch_gate, workspace.entry_count,
        fx.Int32(num_valid),
        fx.Int32(tokens), fx.Int32(workspace.rank), fx.Int32(m_local),
        fx.Int32(grid_x), stream,
    )
    return out, out_scale

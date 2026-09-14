# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Token-id GEMM1 with per-expert payload_ready wait (TP incremental stage1).

Step 2: host pre-fills ``payload_ready == expected`` so the wait returns
immediately. Producers that increment ready land in later steps.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr

from ..tensor_shim import _run_compiled
from .gemm1 import _LdsF32View
from .gemm_util import _make_buffer
from .tp_incremental_payload import wait_expert_payload
from .tp_token_gemm1 import build_token_gemm1

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

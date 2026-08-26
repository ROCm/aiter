# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone TP MoE GEMM1: dense rank-major A gathered by sorted_token_ids.

Forks only the builder half of ``gemm1.build_fused_gemm1``. The 230-line
``do_tile`` MFMA pipeline is imported and reused unmodified, because it takes
the loaders as parameters and never touches their internals.

``gather=False`` selects the original contiguous loaders. That mode exists so
the test can run the identical kernel over host-permuted data and require the
two to agree bit for bit; it is not a production path.

Two things that look adjustable but are not:

* ``always_valid=True`` on the epilogue must stay. ``SiluQuantEpilogue`` reads
  its ``sorted_rsrc`` as a token table indexed by row slot (gemm_util.py:662),
  but what we hand it is ``trb_rsrc``, a row-base table indexed by tile. The
  two meanings are incompatible and only ``always_valid=True`` keeps that read
  dead. Flipping it to ``False`` indexes a length-``num_m_tiles`` table with a
  row slot and reads out of bounds.
* ``trb_rsrc`` is built with ``max_size=True``, so the hardware applies no
  bounds clamp to it. The host-allocated ``tile_row_base`` must therefore hold
  at least ``num_m_tiles`` int32 entries.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.runtime.device import get_rocm_arch

from ..tensor_shim import _run_compiled
from .gemm1 import _LdsF32View, do_tile
from .gemm_util import (
    AS2RLoader,
    AScaleLoader,
    ATileLoader,
    BScaleLoader,
    BWeightLoader,
    MfmaScaleGU,
    SiluQuantEpilogue,
    TileScheduler,
    _buffer_load,
    _make_buffer,
)
from .tp_gemm_util import TPAScaleLoader, TPATileLoader

# fmt: off
@functools.cache
def _compile_tp_gemm1(
    *, model_dim: int, inter_dim: int, experts: int, total_rows: int,
    gather: bool = True, sort_block_m: int = 32, tile_n: int = 256, tile_k: int = 256,
    num_waves: int = 4, num_cu: int = 256, grid_mult: int = 4, swizzle_a: bool = True,
    pipe_weights: bool = True, mfma_amajor: bool = False, async_a_copy: bool = False,
    waves_per_eu_hint: int = 2, swiglu_limit: float = 0.0,
):
    # fmt: on
    arch = get_rocm_arch()
    if not str(arch).startswith("gfx95"):
        raise RuntimeError(f"tp_gemm1 targets gfx95x, got {arch}")

    assert num_waves > 1
    assert 1 <= waves_per_eu_hint <= 4
    assert tile_n % num_waves == 0
    assert (2 * inter_dim) % tile_n == 0
    assert tile_k == 256 and model_dim % tile_k == 0

    NUM_WAVES = num_waves
    TOTAL_THREADS = NUM_WAVES * 64
    n_per_wave = tile_n // NUM_WAVES
    N_TILES = (2 * inter_dim) // tile_n
    M_REPEAT = sort_block_m // 16
    NUM_ACC_N = n_per_wave // 16
    assert NUM_ACC_N % 2 == 0 and M_REPEAT % 2 == 0
    A_K_STEP_BYTES = tile_k
    K_ITERS = model_dim // tile_k
    a_lds_size = sort_block_m * A_K_STEP_BYTES
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n // 2
    lds_pool_bytes = max(2 * a_lds_size, sort_block_m * cs_tile_n * 4)
    n_scale_bytes = sort_block_m * (model_dim // 32)
    GRID_X = num_cu * grid_mult
    PAD_ROW = total_rows - 1  # last row of A is the zeroed padding row

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    kernel_name = (
        f"tp_gemm1_{'gather' if gather else 'contig'}"
        f"_t{sort_block_m}x{tile_n}x{tile_k}_w{NUM_WAVES}_d{model_dim}_i{inter_dim}"
    )

    # fmt: off
    @flyc.kernel(name=kernel_name, known_block_size=[TOTAL_THREADS, 1, 1])
    def tp_gemm1_kernel(
        out: fx.Tensor, out_scale: fx.Tensor, x: fx.Tensor, scale_x: fx.Tensor,
        w: fx.Tensor, scale_w: fx.Tensor, tile_row_base: fx.Tensor,
        expert_ids: fx.Tensor, sorted_token_ids: fx.Tensor, num_valid_ids: fx.Tensor,
        tokens: fx.Int32,
    ):
        # fmt: on
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))
        wave_id = fx.thread_idx.x // 64

        w_rsrc = _make_buffer(w, fx.Int32, 4)
        sw_rsrc = _make_buffer(scale_w, fx.Int32)
        sx_rsrc = _make_buffer(scale_x, fx.Int32, 4)
        trb_rsrc = _make_buffer(tile_row_base, fx.Int32)
        expert_rsrc = _make_buffer(expert_ids, fx.Int32)
        tok_rsrc = _make_buffer(sorted_token_ids, fx.Int32)
        nv_rsrc = _make_buffer(num_valid_ids, fx.Int32)
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_nbytes = tokens * fx.Int32(scale_cols) + fx.Int32(8192)
        os_rsrc = _make_buffer(
            out_scale, fx.Int8, max_size=False, num_records_bytes=os_nbytes
        )

        sched = TileScheduler(
            expert_rsrc=expert_rsrc, inter_dim=inter_dim, expert_offset=0
        )
        n_wave_base = wave_id * fx.Int32(n_per_wave)

        # fmt: off
        if const_expr(gather):
            a_gather = TPATileLoader(row_bytes=model_dim, sort_block_m=sort_block_m,
                k_step_bytes=A_K_STEP_BYTES, total_threads=TOTAL_THREADS,
                swizzle=swizzle_a, x_tensor=x, tok_rsrc=tok_rsrc, pad_row=PAD_ROW,
                total_rows=total_rows, async_copy=async_a_copy)
            a_scale = TPAScaleLoader(scale_rsrc=sx_rsrc, m_repeat=M_REPEAT,
                model_dim=model_dim, sort_block_m=sort_block_m,
                total_threads=TOTAL_THREADS, tok_rsrc=tok_rsrc, pad_row=PAD_ROW)
        else:
            a_gather = ATileLoader(row_bytes=model_dim, sort_block_m=sort_block_m,
                k_step_bytes=A_K_STEP_BYTES, total_threads=TOTAL_THREADS,
                swizzle=swizzle_a, x_tensor=x, async_copy=async_a_copy)
            a_scale = AScaleLoader(scale_rsrc=sx_rsrc, m_repeat=M_REPEAT,
                model_dim=model_dim, sort_block_m=sort_block_m,
                total_threads=TOTAL_THREADS)
        # fmt: on

        a_s2r = AS2RLoader(k_step_bytes=A_K_STEP_BYTES, swizzle=swizzle_a)
        b_loader = BWeightLoader(
            w_rsrc=w_rsrc, num_acc_n=NUM_ACC_N, model_dim=model_dim, cache_modifier=0
        )
        b_scale = BScaleLoader(
            scale_rsrc=sw_rsrc, num_acc_n=NUM_ACC_N, model_dim=model_dim
        )
        mfma = MfmaScaleGU(m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N)
        # fmt: off
        epi = SiluQuantEpilogue(out_rsrc=None, out_scale_rsrc=os_rsrc, sorted_rsrc=trb_rsrc,
            tokens=0, inter_dim=inter_dim, m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N,
            sort_block_m=sort_block_m, tile_n=tile_n, num_waves=NUM_WAVES, lds_out=c_tile,
            swiglu_limit=swiglu_limit, always_valid=True, out_tensor=out)
        # fmt: on

        # A closure, not an inline body: the scf.for rewriter treats every name
        # assigned in a loop body as loop-carried state, and the loader/scheduler
        # objects have no IR representation.
        def do_scheduled_tile(flat):
            m_tile = flat // fx.Int32(N_TILES)
            n_tile = flat - m_tile * fx.Int32(N_TILES)
            n_tile_base = n_wave_base + n_tile * fx.Int32(tile_n)
            expert = sched.expert_of(m_tile)
            # fmt: off
            do_tile(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader,
                b_scale, a_scale, mfma, epi, a_buf, a_scale_lds, a_lds_i32,
                K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, pipe_weights,
                mfma_amajor, async_a_copy, trb_rsrc)
            # fmt: on

        num_valid = _buffer_load(nv_rsrc, fx.Int32(0), fx.Int32)
        num_m_tiles = (num_valid + fx.Int32(sort_block_m - 1)) // fx.Int32(sort_block_m)
        total_work = num_m_tiles * fx.Int32(N_TILES)

        for flat in range(fx.block_idx.x, total_work, fx.Int32(GRID_X)):
            do_scheduled_tile(flat)

    # fmt: off
    @flyc.jit
    def launch(
        out: fx.Tensor, out_scale: fx.Tensor, x: fx.Tensor, scale_x: fx.Tensor,
        w: fx.Tensor, scale_w: fx.Tensor, tile_row_base: fx.Tensor,
        expert_ids: fx.Tensor, sorted_token_ids: fx.Tensor, num_valid_ids: fx.Tensor,
        tokens: fx.Int32, stream: fx.Stream,
    ):
        # fmt: on
        tp_gemm1_kernel(
            out, out_scale, x, scale_x, w, scale_w, tile_row_base, expert_ids,
            sorted_token_ids, num_valid_ids, tokens,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}",
            },
        ).launch(
            grid=(GRID_X, 1, 1), block=(TOTAL_THREADS, 1, 1), stream=stream
        )

    return launch


def compile_tp_gemm1(*, gather: bool = True, total_rows: int, **cfg):
    """Compile (and cache) one TP GEMM1 kernel.

    ``total_rows`` only reaches the gathering loaders, so it is normalised away
    for ``gather=False``. Without that, the contiguous reference would recompile
    once per test shape purely because of an argument it never reads.
    """
    return _compile_tp_gemm1(
        gather=bool(gather), total_rows=int(total_rows) if gather else 0, **cfg
    )


# fmt: off
def run_tp_gemm1(
    *, x, scale_x, w, scale_w, tile_row_base, expert_ids, sorted_token_ids,
    num_valid_ids, max_sorted, model_dim, inter_dim, experts, total_rows,
    gather=True, sort_block_m=32, swiglu_limit=0.0, stream=None, **cfg,
):
    # fmt: on
    """Allocate outputs and launch. Returns ``(payload_fp8, packed_mx_scale)``."""
    import torch

    dev = x.device
    out = torch.empty((max_sorted, inter_dim), dtype=torch.float8_e4m3fn, device=dev)
    prows = ((max_sorted + 255) // 256) * 256
    pcols = (((inter_dim // 32) + 7) // 8) * 8
    # +8192 mirrors the num_records slack the kernel gives os_rsrc, so the buffer
    # descriptor can never describe more memory than actually exists.
    out_scale = torch.zeros(prows * pcols + 8192, dtype=torch.uint8, device=dev)
    launch = compile_tp_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        total_rows=total_rows,
        gather=gather,
        sort_block_m=sort_block_m,
        swiglu_limit=swiglu_limit,
        **cfg,
    )
    if stream is None:
        stream = fx.Stream(torch.cuda.current_stream())
    # fmt: off
    _run_compiled(
        launch, out, out_scale, x.view(torch.uint8), scale_x.view(torch.uint8),
        w.view(torch.uint8), scale_w.view(torch.uint8), tile_row_base, expert_ids,
        sorted_token_ids, num_valid_ids, fx.Int32(int(max_sorted)), stream)
    # fmt: on
    return out, out_scale

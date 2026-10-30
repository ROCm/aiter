# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Building blocks of the gfx942 / CDNA3 bf16 GEMM main loop.

The standalone GEMM (``gemm_a16w16_gfx942``) and fused kernels share this
layer: they own the launch, the LDS allocation and the epilogue, and get the
tuned A/B loading and MFMA pipelines from here.

Inside a kernel:
- size the A/B LDS buffers with ``gfx942_ab_lds_elems(param)`` (plus
  ``gfx942_red_lds_elems(param)`` when ``k_waves > 1``);
- ``ml = make_gfx942_mainloop(...)`` on this workgroup's tile and K range;
- ``run_gfx942_mainloop(ml)``: ``ml.frag_C`` then holds the fp32 tile, in
  K wave 0 when ``k_waves > 1``.

``transform_a`` lets a fused kernel rewrite A fragments between the LDS read
and the MFMAs (e.g. normalize the activations); ``gfx942_a_coords`` gives the
tile coordinates of their elements.

Config fields that only concern the standalone epilogue (``direct_store``,
``split_k``, ``split_k_reduce``, ``out_dtype_id``) are ignored here.
"""

import functools
from dataclasses import dataclass
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

from .gemm_a16w16_gfx950 import AsyncLoadTile, async_load_to_lds
from .gemm_a16w16_gfx950_utils import GFX950_WAVE_SIZE, get_wave_lds_offset

# Unstable rocdl calls kept on purpose (no stable equivalent in FlyDSL 0.4):
# - rocdl.s_barrier: gpu.barrier adds workgroup fences, which drain the
#   in-flight global loads the pipelines overlap with the barrier.
# - rocdl.sched_barrier(0): pins the hand-scheduled MFMA/DS/VMEM interleave;
#   the stable sched_* helpers only emit sched_group_barrier.
# - rocdl.readfirstlane: makes k_wave_idx wave-uniform (SGPR), so the
#   per-K-wave LDS offsets and branches stay scalar.
# async_load_to_lds (gfx950 helper) adds its own rocdl.sched_barrier.

WAVE_SIZE = GFX950_WAVE_SIZE  # 64 lanes on CDNA3
SMEM_CAPACITY_GFX942 = 65536  # 64 KB LDS / CU

GEMM_A16W16_DTYPE_BF16 = 2
GEMM_A16W16_DTYPE_FP32 = 1


def ab_k_perm(mma_k: int, *, dtype_width: int = 16, copy_bits: int = 128):
    """K-axis permutation pairing a ``copy_bits`` A/B load with the MFMA K fragment.

    One MFMA consumes ``k_group = mma_k // 4`` elements per lane. With bf16
    ``mma_k=16`` a 128-bit load spans ``num_frgv = 2`` K-groups; the extra axis
    interleaves them so the tiled-copy partition derived from the tiled MMA
    matches the wide load. Collapses to ``(k_group, 4):(1, k_group)`` when a load
    is a single K-group.
    """
    k_group = mma_k // 4
    num_frgv = copy_bits // (k_group * dtype_width)
    if num_frgv <= 1:
        return fx.make_layout((k_group, 4), (1, k_group))
    num_elems = copy_bits // dtype_width
    return fx.make_layout((k_group, 4, num_frgv), (1, num_elems, k_group))


@fx.struct
class GemmA16W16Gfx942Param:
    in_dtype_id: fx.Constexpr[int]
    out_dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    stages: fx.Constexpr[int]
    is_split_k: fx.Constexpr[bool]
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    k_waves: fx.Constexpr[int]
    group_m: fx.Constexpr[int]
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    buffered_load: fx.Constexpr[bool]
    direct_store: fx.Constexpr[bool]
    pingpong: fx.Constexpr[bool]
    split_k_reduce: fx.Constexpr[bool]
    async_load_bytes: fx.Constexpr[int]
    # derived
    in_data_bytes: fx.Constexpr[int]
    out_data_bytes: fx.Constexpr[int]
    cshuffle_r2g_vec_size: fx.Constexpr[int]
    c_chunks: fx.Constexpr[int]
    ldg_x_threads: fx.Constexpr[int]
    block_threads: fx.Constexpr[int]
    ldg_a_iters: fx.Constexpr[int]
    ldg_b_iters: fx.Constexpr[int]


@dataclass(slots=True, kw_only=True, eq=False)
class Gfx942LoadContext:
    wave_offset: Any
    tid: Any
    ks_begin: Any
    param: GemmA16W16Gfx942Param
    async_g2s_copy_atom: Any
    buffer_copy_atom: Any
    uni_copy_atom: Any


# Cached: building the frozen param struct costs ~90 us of host time per launch.
@functools.cache
def make_gemm_a16w16_gfx942_param(
    in_dtype_id: int = GEMM_A16W16_DTYPE_BF16,
    out_dtype_id: int = GEMM_A16W16_DTYPE_BF16,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    stages: int = 2,
    split_k: int = 1,
    m_waves: int = 2,
    n_waves: int = 2,
    k_waves: int = 1,
    group_m: int = 0,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 16,
    async_load_bytes: int = 4,
    buffered_load: bool = False,
    direct_store: bool = False,
    pingpong: bool = False,
    split_k_reduce: bool = False,
) -> GemmA16W16Gfx942Param:
    if in_dtype_id != GEMM_A16W16_DTYPE_BF16:
        raise ValueError(f"gfx942 kernel is bf16-only (got in_dtype_id={in_dtype_id})")
    if out_dtype_id not in (in_dtype_id, GEMM_A16W16_DTYPE_FP32):
        raise ValueError(f"unsupported out_dtype_id={out_dtype_id}")
    if (mma_m, mma_n, mma_k) != (16, 16, 16):
        raise ValueError("gfx942 bf16 MFMA is 16x16x16")
    if not buffered_load:
        if stages < 2:
            raise ValueError(
                "the async pipeline needs stages >= 2; stages=1 needs buffered_load"
            )
        if async_load_bytes != 4:
            raise ValueError(
                "gfx942 async global->LDS loads are 32-bit only (async_load_bytes=4); "
                "128-bit loads need buffered_load"
            )
    if buffered_load:
        if async_load_bytes != 16:
            raise ValueError(
                "buffered_load requires async_load_bytes=16 (128-bit buffer_load)"
            )
        if stages not in (1, 2):
            raise ValueError("buffered_load supports stages=1 (single LDS buffer) or 2")
    if k_waves < 1:
        raise ValueError("k_waves must be positive")
    if pingpong:
        if not (buffered_load and stages == 1):
            raise ValueError("pingpong requires buffered_load with stages=1")
        # Two equal halves of whole 128-bit (32-element) K chunks per K wave.
        if block_k % (k_waves * 64) != 0:
            raise ValueError("pingpong needs block_k // k_waves to be a multiple of 64")
    if k_waves > 1:
        if k_waves & (k_waves - 1):
            raise ValueError("k_waves must be a power of two")
        if not pingpong:
            raise ValueError("k_waves > 1 requires the pingpong pipeline")
    # Each K wave reads LDS in 128-bit (32-element) K chunks of two MFMA K steps.
    if block_k % (k_waves * 32) != 0:
        raise ValueError("block_k // k_waves must be a multiple of 32")
    # The async loader recovers a lane's global K column as its swizzled LDS
    # offset modulo block_k. The swizzle XORs index bits 6-8 into bits 3-5, so
    # that stays within the row only if a row spans whole 64-element segments.
    if not buffered_load and block_k % 64 != 0:
        raise ValueError("the async pipeline needs block_k to be a multiple of 64")
    if split_k < 1:
        raise ValueError("split_k must be >= 1")
    if split_k_reduce and split_k == 1:
        raise ValueError("split_k_reduce requires split_k > 1")
    if direct_store and split_k != 1 and not split_k_reduce:
        raise ValueError("direct_store with split_k > 1 requires split_k_reduce")
    if m_waves <= 0 or n_waves <= 0:
        raise ValueError("m_waves and n_waves must be positive")
    if m_waves * n_waves * k_waves > 16:
        raise ValueError("the workgroup cannot contain more than 16 waves")

    in_dbytes = 2
    out_dbytes = 4 if out_dtype_id == GEMM_A16W16_DTYPE_FP32 else 2
    block_threads = m_waves * n_waves * k_waves * WAVE_SIZE

    atomic_split_k = split_k > 1 and not split_k_reduce
    # CDNA3 has no packed-bf16 buffer atomic add; gemm_a16w16 gives bf16
    # output an fp32 scratch.
    if atomic_split_k and out_dtype_id != GEMM_A16W16_DTYPE_FP32:
        raise ValueError("atomic split-K accumulates in fp32 output only")
    # LDS->global vector width of the cshuffle epilogue: 128 bits.
    cshuffle_r2g_vec_size = 16 // out_dbytes
    if block_n % cshuffle_r2g_vec_size != 0:
        raise ValueError("block_n must be divisible by the cshuffle store vector size")
    # SplitKProtocol.zero_c clears the C tile in whole vectors, the same
    # number per thread.
    if atomic_split_k and (block_m * block_n) % (block_threads * cshuffle_r2g_vec_size):
        raise ValueError(
            "atomic split-K needs block_m * block_n to be a whole number of "
            f"{cshuffle_r2g_vec_size}-wide vectors per thread"
        )

    # LDS budget: A tile + B tile, per stage.
    smem_bytes = stages * (block_m + block_n) * block_k * in_dbytes
    if smem_bytes > SMEM_CAPACITY_GFX942:
        raise ValueError(
            f"staged LDS buffers exceed 64 KB: stages={stages}, block_m={block_m}, "
            f"block_n={block_n}, block_k={block_k}, smem_bytes={smem_bytes}"
        )
    red_smem_bytes = (k_waves // 2) * block_m * block_n * 4
    if red_smem_bytes > SMEM_CAPACITY_GFX942:
        raise ValueError(
            f"k_waves reduction buffer exceeds 64 KB: k_waves={k_waves}, "
            f"block_m={block_m}, block_n={block_n}"
        )

    def c_chunk_bytes(chunks):
        return block_m // chunks * block_n * out_dbytes

    # A C tile over 64 KB goes through LDS in row chunks of whole MMA M
    # repeats, sized to fit the A/B buffers when possible so LDS does not grow.
    c_smem_bytes = c_chunk_bytes(1)
    c_chunks = 1
    if not direct_store and c_smem_bytes > SMEM_CAPACITY_GFX942:
        reps_m = block_m // (mma_m * m_waves)
        fits = [
            d
            for d in range(2, reps_m + 1)
            if reps_m % d == 0 and c_chunk_bytes(d) <= SMEM_CAPACITY_GFX942
        ]
        if not fits:
            raise ValueError(
                f"cshuffle tile exceeds 64 KB: block_m={block_m}, block_n={block_n}, "
                f"smem_bytes={c_smem_bytes}; use direct_store for aligned output tiles"
            )
        c_chunks = next((d for d in fits if c_chunk_bytes(d) <= smem_bytes), fits[0])

    # Global->LDS bytes per lane: 4 for async DMA, 16 for buffered_load.
    async_load_vec_size = async_load_bytes // in_dbytes
    ldg_x_threads = block_k // async_load_vec_size
    if ldg_x_threads * async_load_vec_size != block_k:
        raise ValueError(
            f"block_k must be divisible by async_load_vec_size={async_load_vec_size}"
        )
    ldg_y_threads = block_threads // ldg_x_threads
    if ldg_y_threads * ldg_x_threads != block_threads:
        raise ValueError("ldg thread layout must exactly cover the workgroup")
    ldg_a_iters = (block_m * block_k) // (block_threads * async_load_vec_size)
    ldg_b_iters = (block_n * block_k) // (block_threads * async_load_vec_size)
    if ldg_a_iters * block_threads * async_load_vec_size != block_m * block_k:
        raise ValueError("A tile not covered by whole-thread vector loads")
    if ldg_b_iters * block_threads * async_load_vec_size != block_n * block_k:
        raise ValueError("B tile not covered by whole-thread vector loads")
    # vmcnt tracks at most 63 outstanding loads; the async pipelines keep
    # stages-2 tiles of loads in flight.
    if (stages - 2) * (ldg_a_iters + ldg_b_iters) >= 63:
        raise ValueError("too many in-flight async loads for the vmcnt budget")

    mma_m_repeat = block_m // m_waves // mma_m
    mma_n_repeat = block_n // n_waves // mma_n
    mma_k_repeat = block_k // k_waves // mma_k
    if mma_m_repeat * m_waves * mma_m != block_m:
        raise ValueError("block_m must be divisible by m_waves * mma_m")
    if mma_n_repeat * n_waves * mma_n != block_n:
        raise ValueError("block_n must be divisible by n_waves * mma_n")
    if mma_k_repeat * k_waves * mma_k != block_k:
        raise ValueError("block_k must be divisible by k_waves * mma_k")

    return GemmA16W16Gfx942Param(
        in_dtype_id=in_dtype_id,
        out_dtype_id=out_dtype_id,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        is_split_k=split_k > 1,
        m_waves=m_waves,
        n_waves=n_waves,
        k_waves=k_waves,
        group_m=group_m,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
        buffered_load=buffered_load,
        direct_store=direct_store,
        pingpong=pingpong,
        split_k_reduce=split_k_reduce,
        async_load_bytes=async_load_bytes,
        in_data_bytes=in_dbytes,
        out_data_bytes=out_dbytes,
        cshuffle_r2g_vec_size=cshuffle_r2g_vec_size,
        c_chunks=c_chunks,
        ldg_x_threads=ldg_x_threads,
        block_threads=block_threads,
        ldg_a_iters=ldg_a_iters,
        ldg_b_iters=ldg_b_iters,
    )


def make_k_sliced_lds_layout(rows, block_k, k_waves):
    """Row-major ``(rows, block_k)`` LDS tile stored as ``k_waves`` K sub-tiles.

    Each ``(rows, block_k // k_waves)`` sub-tile is contiguous and carries the
    swizzle of ``make_lds_layout`` (gemm_a16w16_gfx950_utils); a K wave reads
    its sub-tile by slicing the ``k_waves`` mode. The 3-bit swizzle never
    touches the sub-tile index bits because every sub-tile spans a multiple of
    512 elements. With ``k_waves=1`` this is ``make_lds_layout``.
    """
    block_k_w = block_k // k_waves
    if (rows * block_k_w) % 512:
        raise ValueError(
            "each K-sliced LDS sub-tile must span a multiple of 512 elements"
        )
    swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))
    return fx.make_composed_layout(
        swizzle,
        fx.make_layout(
            (rows, (block_k_w, k_waves)), (block_k_w, (1, rows * block_k_w))
        ),
    )


def make_gfx942_load_context(
    elem_dtype, load_tid, ks_begin, param: GemmA16W16Gfx942Param
):
    """Copy atoms and per-thread offsets of the A/B global->LDS loads."""
    uni_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
    buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    if const_expr(param.buffered_load):
        # gfx942 has no 128-bit async-to-LDS; the buffered path stages the
        # 128-bit load through registers (buffer_load) then ds_write to LDS.
        async_g2s_copy_atom = None
    else:
        # The 32-bit form is CDNA3's buffer_load_dword ... lds. Its async atom
        # lives under cdna4 but runs on gfx942; the synchronous
        # fx.rocdl.BufferCopyLDS32b would make the compiler wait on vmcnt
        # before every LDS read and serialize the asyncmark-tracked stages.
        async_g2s_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.BufferLoadAsyncLDS32b(), 32
        )

    return Gfx942LoadContext(
        wave_offset=get_wave_lds_offset(load_tid, param.async_load_bytes),
        tid=load_tid,
        ks_begin=ks_begin,
        param=param,
        async_g2s_copy_atom=async_g2s_copy_atom,
        buffer_copy_atom=buffer_copy_atom,
        uni_copy_atom=uni_copy_atom,
    )


def gfx942_k_wave_split(param: GemmA16W16Gfx942Param, tid):
    """(tid_mn, k_wave_idx): a thread's index within its K wave, and its K wave
    (None when k_waves == 1).

    k_waves groups of m_waves x n_waves waves share the cooperative
    global->LDS copies, but each group reads and multiplies only its own
    block_k // k_waves slice of every LDS tile. Call it first in the kernel:
    issued later, the readfirstlane shifts the prologue schedule.
    """
    k_waves = param.k_waves
    if const_expr(k_waves == 1):
        return tid, None
    threads_per_k_slice = param.block_threads // k_waves
    k_wave_idx = fx.Int32(
        rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(tid // threads_per_k_slice))
    )
    return tid % threads_per_k_slice, k_wave_idx


def make_gfx942_tiled_mma(param: GemmA16W16Gfx942Param):
    """The bf16 16x16x16 tiled MMA over one K wave's m_waves x n_waves waves."""
    mma_atom = fx.make_mma_atom(
        fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, fx.BFloat16)
    )
    return fx.make_tiled_mma(
        mma_atom,
        fx.make_layout((param.m_waves, param.n_waves, 1), (param.n_waves, 1, 0)),
        fx.make_tile(None, None, ab_k_perm(param.mma_k)),
    )


def gfx942_ab_lds_elems(param: GemmA16W16Gfx942Param) -> tuple[int, int]:
    """bf16 elements of the A and B LDS buffers over all stages."""
    return (
        param.stages * param.block_m * param.block_k,
        param.stages * param.block_n * param.block_k,
    )


def gfx942_red_lds_elems(param: GemmA16W16Gfx942Param) -> int:
    """fp32 elements of the K-wave reduction buffer (0 when k_waves == 1).

    It is used only after the main loop, so it may alias the A/B buffers.
    """
    return (param.k_waves // 2) * param.block_m * param.block_n


@dataclass(slots=True, kw_only=True, eq=False)
class Gfx942Mainloop:
    """One workgroup's main loop, built by ``make_gfx942_mainloop``.

    For the caller: ``frag_C`` (fp32 accumulator), ``thr_mma`` and ``tid_mn``
    (the MMA slice and thread index within this thread's K wave),
    ``k_wave_idx`` and ``threads_per_k_slice``. The other fields are shared
    with the pipeline functions.
    """

    param: GemmA16W16Gfx942Param
    tiled_mma: Any
    thr_mma: Any
    k_tiles: Any
    tid_mn: Any
    k_wave_idx: Any
    threads_per_k_slice: int
    k_slice: Any
    smem_red: Any
    # Compile-time K chunking: 128-bit LDS reads per K wave tile, MFMA K steps
    # per tile, and MFMA K steps fed by one 128-bit read.
    copy_chunks: int
    mma_k_iters: int
    num_frgv: int
    frag_A: Any
    frag_B: Any
    frag_A_ret: Any
    frag_B_ret: Any
    frag_pf_A: Any
    frag_pf_B: Any
    frag_pf_A_ret: Any
    frag_pf_B_ret: Any
    frag_C: Any
    read_tile_into: Any
    mma_tile: Any
    compute_stage: Any
    load_ab_to_reg: Any
    commit_ab_to_lds: Any
    async_load_a_to_lds: Any
    async_load_b_to_lds: Any
    # (frag, k_tile, g_begin, g_end) -> None, or None without transform_a.
    transform_a: Any


# Both entry points are @flyc.jit: FlyDSL's cache key follows plain helpers
# only within the caller's directory, so fused kernels elsewhere would keep
# stale binaries after a change here.
@flyc.jit
def make_gfx942_mainloop(
    param: GemmA16W16Gfx942Param,
    tiled_mma,
    *,
    tid,
    k_wave_split,
    a: fx.Tensor,
    b: fx.Tensor,
    m,
    n,
    k,
    a_leading_stride,
    b_leading_stride,
    bid_m,
    bid_n,
    ks_begin,
    k_tiles,
    c_tile,
    smem_a,
    smem_b,
    smem_red=None,
    transform_a=None,
) -> Gfx942Mainloop:
    """Build the main loop of output tile (bid_m, bid_n) over K tiles
    [ks_begin / block_k, ks_begin / block_k + k_tiles).

    ``a`` is row-major [M, K] and ``b`` the [K, N] view of a row-major [N, K]
    weight (``nt``), with row strides ``a_leading_stride`` and
    ``b_leading_stride``. Edge tiles are handled: rows past M or N only feed
    outputs the caller must not store. ``c_tile`` is the (block_m, block_n)
    output tile; it sets the accumulator layout and is not written.
    ``smem_a``/``smem_b`` hold ``gfx942_ab_lds_elems(param)`` elements;
    ``smem_red`` (``gfx942_red_lds_elems``) is needed when ``k_waves > 1``.
    ``k_wave_split`` is ``gfx942_k_wave_split(param, tid)``.

    ``transform_a(frag, k_col, g_begin, g_end)`` is called once MFMA K groups
    ``frag[None, None, g]``, g in [g_begin, g_end), of an A fragment hold
    LDS data and before any MFMA reads them. ``k_col`` is the global K index
    of the fragment's column 0; ``gfx942_a_coords`` gives each element's
    (row, column) in the tile. The transform must keep the fragment bf16 and
    is outside the hand-tuned schedule, so its cost adds to the main loop.
    """
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    stages = param.stages
    block_threads = param.block_threads
    ldg_a_iters = param.ldg_a_iters
    ldg_b_iters = param.ldg_b_iters
    elem_dtype = fx.BFloat16

    k_waves = param.k_waves
    block_k_w = block_k // k_waves
    tid_mn, k_wave_idx = k_wave_split
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)

    ctx = make_gfx942_load_context(
        elem_dtype, load_tid=tid, ks_begin=ks_begin, param=param
    )
    s2r_copy_atom = ctx.uni_copy_atom

    # The LDS writers address a stage's whole (rows, block_k) tile; each K wave
    # reads its (rows, block_k_w) slice of the k_waves mode. A stage is a base
    # pointer, not a layout mode: the swizzle would cover a runtime stage index
    # too, and the compiler then emits an XOR per LDS access.
    a_lds_layout = make_k_sliced_lds_layout(block_m, block_k, k_waves)
    b_lds_layout = make_k_sliced_lds_layout(block_n, block_k, k_waves)
    k_slice = k_wave_idx if const_expr(k_waves > 1) else 0

    def a_stage_base(stage):
        return smem_a + stage * block_m * block_k

    def b_stage_base(stage):
        return smem_b + stage * block_n * block_k

    def k_wave_views(stage):
        return (
            fx.make_view(a_stage_base(stage), a_lds_layout)[None, (None, k_slice)],
            fx.make_view(b_stage_base(stage), b_lds_layout)[None, (None, k_slice)],
        )

    thr_mma = tiled_mma.thr_slice(tid_mn)
    thr_copy_A = fx.make_tiled_copy_A(ctx.buffer_copy_atom, tiled_mma).get_slice(tid_mn)
    thr_copy_B = fx.make_tiled_copy_B(ctx.buffer_copy_atom, tiled_mma).get_slice(tid_mn)

    sA0, sB0 = k_wave_views(0)
    frag_A = thr_mma.make_fragment_A(sA0)
    frag_B = thr_mma.make_fragment_B(sB0)
    frag_C = thr_mma.make_fragment_C(c_tile)
    frag_A_ret = thr_copy_A.retile(frag_A)
    frag_B_ret = thr_copy_B.retile(frag_B)
    # Register double buffer of the async pipeline with stages >= 3: the next
    # K-tile's LDS->reg read lands here while frag_A/frag_B feed the MFMAs.
    frag_pf_A = frag_pf_B = frag_pf_A_ret = frag_pf_B_ret = None
    if const_expr(stages >= 3):
        frag_pf_A = thr_mma.make_fragment_A(sA0)
        frag_pf_B = thr_mma.make_fragment_B(sB0)
        frag_pf_A_ret = thr_copy_A.retile(frag_pf_A)
        frag_pf_B_ret = thr_copy_B.retile(frag_pf_B)

    # Buffered-load tiled copies: 128-bit buffer_load global->reg, then a
    # layout-aware reg->LDS copy into the swizzled LDS tile.
    load_ab_to_reg = commit_ab_to_lds = None
    if const_expr(param.buffered_load):
        vec_per_thread = param.async_load_bytes // param.in_data_bytes
        buffer_atom = ctx.buffer_copy_atom
        uni_atom = ctx.uni_copy_atom

        def make_g2s_tv(rows):
            thrs_k = block_k // vec_per_thread
            thrs_r = block_threads // thrs_k
            tv = fx.make_layout(
                ((thrs_k, thrs_r), (1, vec_per_thread)),
                ((thrs_r * vec_per_thread, 1), (1, thrs_r)),
            )
            return tv, thrs_r

        # The descriptors span each operand's real extent, so edge tiles never
        # read past the allocation; rows past M/N only feed unstored outputs.
        # A: tile (block_m, block_k), global row-block bid_m, all K-tiles.
        a_tv, a_thrs_r = make_g2s_tv(block_m)
        a_bounded = fx.rocdl.make_buffer_tensor(a, max_size=False)
        gA_all = fx.flat_divide(a_bounded, (block_m, block_k))[None, None, bid_m, None]
        a_g2r = fx.make_tiled_copy(
            buffer_atom, a_tv, fx.make_tile(a_thrs_r, block_k)
        ).get_slice(tid)
        a_r2s = fx.make_tiled_copy(
            uni_atom, a_tv, fx.make_tile(a_thrs_r, block_k)
        ).get_slice(tid)
        gA_thr = a_g2r.partition_S(gA_all)
        a_cp_frag = fx.make_fragment_like(gA_thr[None, None, None, 0])
        a_cp_frag_ret = a_r2s.retile(a_cp_frag)

        # B: logical (N, K) view of the [K, N] weight (strides (b_leading_stride, 1)),
        # tile (block_n, block_k), global row-block bid_n, all K-tiles.
        b_nk_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(b), fx.make_layout((n, k), (b_leading_stride, 1))),
            max_size=False,
        )
        b_tv, b_thrs_r = make_g2s_tv(block_n)
        gB_all = fx.flat_divide(b_nk_buf, (block_n, block_k))[None, None, bid_n, None]
        b_g2r = fx.make_tiled_copy(
            buffer_atom, b_tv, fx.make_tile(b_thrs_r, block_k)
        ).get_slice(tid)
        b_r2s = fx.make_tiled_copy(
            uni_atom, b_tv, fx.make_tile(b_thrs_r, block_k)
        ).get_slice(tid)
        gB_thr = b_g2r.partition_S(gB_all)
        b_cp_frag = fx.make_fragment_like(gB_thr[None, None, None, 0])
        b_cp_frag_ret = b_r2s.retile(b_cp_frag)
        buffered_k_base = ks_begin // block_k

        def load_ab_to_reg(k_tile):
            global_k_tile = buffered_k_base + k_tile
            fx.copy(buffer_atom, gA_thr[None, None, None, global_k_tile], a_cp_frag)
            fx.copy(buffer_atom, gB_thr[None, None, None, global_k_tile], b_cp_frag)

        def commit_ab_to_lds(lds_stage):
            sA_w = a_r2s.partition_D(
                fx.make_view(a_stage_base(lds_stage), a_lds_layout)
            )
            sB_w = b_r2s.partition_D(
                fx.make_view(b_stage_base(lds_stage), b_lds_layout)
            )
            fx.copy(uni_atom, a_cp_frag_ret, sA_w)
            fx.copy(uni_atom, b_cp_frag_ret, sB_w)

    def async_load_a_to_lds(k_tile, stage):
        async_load_to_lds(
            tile=AsyncLoadTile(
                lds_base=a_stage_base(stage),
                src_base=fx.get_iter(a_buf),
                lds_layout=a_lds_layout,
                outer_tile_size=block_m,
                outer_bound=m,
                global_outer_offset=block_m_offset,
                leading_stride=a_leading_stride,
                k_tile=k_tile,
            ),
            context=ctx,
            load_iters=ldg_a_iters,
            is_k_major=False,
        )

    def async_load_b_to_lds(k_tile, stage):
        async_load_to_lds(
            tile=AsyncLoadTile(
                lds_base=b_stage_base(stage),
                src_base=fx.get_iter(b_buf),
                lds_layout=b_lds_layout,
                outer_tile_size=block_n,
                outer_bound=n,
                global_outer_offset=block_n_offset,
                leading_stride=b_leading_stride,
                k_tile=k_tile,
            ),
            context=ctx,
            load_iters=ldg_b_iters,
            is_k_major=False,
        )

    # K-chunking for the software-pipelined compute. One 128-bit LDS->reg read
    # (``ab_k_perm``) feeds ``num_frgv`` MFMA K-groups, so we interleave copy and
    # MFMA at the 128-bit-chunk granularity: issuing chunk i+1's ds_read between
    # chunk i's MFMAs lets the scheduler hide the LDS->reg (lgkmcnt) latency.
    mma_k = param.mma_k
    k_group = mma_k // 4
    num_frgv = const_expr(128 // (k_group * 16))
    mma_k_iters = const_expr(block_k_w // mma_k)
    copy_chunks = const_expr(mma_k_iters // num_frgv)

    transform_a_tile = None
    if const_expr(transform_a is not None):

        def transform_a_tile(frag, k_tile, g_begin, g_end):
            k_col = (ks_begin // block_k + k_tile) * block_k + k_slice * block_k_w
            transform_a(frag, k_col, g_begin, g_end)

    def read_tile_into(read_stage, a_ret, b_ret, c_begin=0, c_end=copy_chunks):
        """ds_read this K wave's slice of LDS stage ``read_stage`` into registers.

        Only 128-bit K chunks ``[c_begin, c_end)`` are read.
        """
        sA_k, sB_k = k_wave_views(read_stage)
        thr_sA = thr_copy_A.partition_S(sA_k)
        thr_sB = thr_copy_B.partition_S(sB_k)
        for c in range_constexpr(c_begin, c_end):
            fx.copy(s2r_copy_atom, thr_sA[None, None, c], a_ret[None, None, c])
            fx.copy(s2r_copy_atom, thr_sB[None, None, c], b_ret[None, None, c])

    def mma_tile(a_frag, b_frag):
        for g in range_constexpr(mma_k_iters):
            fx.gemm(
                tiled_mma,
                frag_C,
                a_frag[None, None, g],
                b_frag[None, None, g],
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )

    def compute_stage(read_stage, k_tile):
        """Compute one LDS stage, issuing chunk c+1's ds_read before chunk c's MFMAs."""
        sA_k, sB_k = k_wave_views(read_stage)
        thr_sA = thr_copy_A.partition_S(sA_k)
        thr_sB = thr_copy_B.partition_S(sB_k)

        def rd(c):
            fx.copy(s2r_copy_atom, thr_sA[None, None, c], frag_A_ret[None, None, c])
            fx.copy(s2r_copy_atom, thr_sB[None, None, c], frag_B_ret[None, None, c])

        rd(0)
        for c in range_constexpr(copy_chunks):
            if const_expr(c + 1 < copy_chunks):
                rd(c + 1)
            if const_expr(transform_a_tile is not None):
                transform_a_tile(frag_A, k_tile, c * num_frgv, (c + 1) * num_frgv)
            for j in range_constexpr(num_frgv):
                g = c * num_frgv + j
                fx.gemm(
                    tiled_mma,
                    frag_C,
                    frag_A[None, None, g],
                    frag_B[None, None, g],
                    frag_C,
                    traversal_order=fx.GemmTraversalOrder.KNM,
                )

    return Gfx942Mainloop(
        param=param,
        tiled_mma=tiled_mma,
        thr_mma=thr_mma,
        k_tiles=k_tiles,
        tid_mn=tid_mn,
        k_wave_idx=k_wave_idx,
        threads_per_k_slice=block_threads // k_waves,
        k_slice=k_slice,
        smem_red=smem_red,
        copy_chunks=copy_chunks,
        mma_k_iters=mma_k_iters,
        num_frgv=num_frgv,
        frag_A=frag_A,
        frag_B=frag_B,
        frag_A_ret=frag_A_ret,
        frag_B_ret=frag_B_ret,
        frag_pf_A=frag_pf_A,
        frag_pf_B=frag_pf_B,
        frag_pf_A_ret=frag_pf_A_ret,
        frag_pf_B_ret=frag_pf_B_ret,
        frag_C=frag_C,
        read_tile_into=read_tile_into,
        mma_tile=mma_tile,
        compute_stage=compute_stage,
        load_ab_to_reg=load_ab_to_reg,
        commit_ab_to_lds=commit_ab_to_lds,
        async_load_a_to_lds=async_load_a_to_lds,
        async_load_b_to_lds=async_load_b_to_lds,
        transform_a=transform_a_tile,
    )


def gfx942_a_coords(param: GemmA16W16Gfx942Param, tiled_mma, k_wave_split):
    """(rows, cols) tensors laid out like the A fragments ``transform_a`` gets:
    each element's row in the block tile and column relative to ``k_col``.
    Compile-time constants, read with ``fx.get_scalar``."""
    thr_mma = tiled_mma.thr_slice(k_wave_split[0])
    tile = (param.block_m, param.block_k // param.k_waves)
    rows = thr_mma.partition_A(fx.make_view(0, fx.make_layout(tile, (1, 0))))
    cols = thr_mma.partition_A(fx.make_view(0, fx.make_layout(tile, (0, 1))))
    return rows, cols


@flyc.jit
def run_gfx942_mainloop(ml: Gfx942Mainloop):
    """Zero ``ml.frag_C`` and accumulate the tile over all K tiles.

    With ``k_waves > 1`` the K waves' partials are then summed into K wave 0.
    """
    param = ml.param
    ml.frag_C.fill(0.0)
    if const_expr(param.pingpong):
        _main_loop_pingpong(ml)
    elif const_expr(param.buffered_load):
        _main_loop_buffered(ml)
    elif const_expr(param.stages < 3):
        _main_loop_async(ml)
    else:
        _main_loop_async_reg_db(ml)
    if const_expr(param.k_waves > 1):
        _reduce_k_waves(
            ml.frag_C,
            param.k_waves,
            ml.k_wave_idx,
            ml.tid_mn,
            ml.threads_per_k_slice,
            ml.smem_red,
        )


def _mma_groups(s: Gfx942Mainloop, a_frag, b_frag, g_begin, g_end):
    for g in range_constexpr(g_begin, g_end):
        fx.gemm(
            s.tiled_mma,
            s.frag_C,
            a_frag[None, None, g],
            b_frag[None, None, g],
            s.frag_C,
            traversal_order=fx.GemmTraversalOrder.KNM,
        )


def _schedule_phase(n_mfma, n_rd, n_wr, n_ld):
    """Spread one phase's memory ops evenly across its MFMAs.

    A next-tile load refills the staging registers being written to LDS, so
    each load follows the LDS write of the same slot.
    """
    n_mem = n_rd + n_wr + n_ld
    used_mfma = 0
    rd = 0
    wr = 0
    ld = 0
    for i in range(n_mem):
        if rd < n_rd:
            rocdl.sched_dsrd(1)
            rd += 1
        elif ld < wr and ld < n_ld:
            rocdl.sched_vmem(1)
            ld += 1
        elif wr < n_wr:
            rocdl.sched_dswr(1)
            wr += 1
        else:
            rocdl.sched_vmem(1)
            ld += 1
        target = (i + 1) * n_mfma // n_mem
        if target > used_mfma:
            rocdl.sched_mfma(target - used_mfma)
            used_mfma = target
    if used_mfma < n_mfma:
        rocdl.sched_mfma(n_mfma - used_mfma)


@flyc.jit
def _main_loop_pingpong(s: Gfx942Mainloop):
    """One LDS buffer, with the MMA operand set streamed in two K halves.

    One operand set and one staging set, so a large tile fits without
    spilling. At the top of iteration k the registers hold half 0 of tile k,
    LDS holds tile k and tile k+1 is in flight from global memory:
      1: ds_read half 1 of tile k         | MFMAs on half 0   | barrier
      2: ds_write tile k+1, reload the    | MFMAs on both     | barrier
         staging set with tile k+2        |   halves          |
      3: ds_read half 0 of tile k+1       | MFMAs on half 1
    Needs k_tiles >= 2 (checked on the host).
    """
    param = s.param
    k_tiles = s.k_tiles
    frag_A = s.frag_A
    frag_B = s.frag_B
    frag_A_ret = s.frag_A_ret
    frag_B_ret = s.frag_B_ret
    read_tile_into = s.read_tile_into
    load_ab_to_reg = s.load_ab_to_reg
    commit_ab_to_lds = s.commit_ab_to_lds
    transform_a = s.transform_a
    copy_chunks = s.copy_chunks
    mma_k_iters = s.mma_k_iters
    m_rep = const_expr(param.block_m // param.m_waves // param.mma_m)
    n_rep = const_expr(param.block_n // param.n_waves // param.mma_n)
    n_dswr = const_expr(param.ldg_a_iters + param.ldg_b_iters)
    half_c = const_expr(copy_chunks // 2)
    half_g = const_expr(half_c * s.num_frgv)
    p1_end = const_expr(half_g // 2)
    p2_end = const_expr(half_g + (mma_k_iters - half_g) // 2)
    mfma_per_g = const_expr(m_rep * n_rep)
    n_dsrd_half = const_expr(half_c * (m_rep + n_rep))

    def pipelined_iter(k_tile, commit, prefetch):
        read_tile_into(0, frag_A_ret, frag_B_ret, half_c, copy_chunks)
        _mma_groups(s, frag_A, frag_B, 0, p1_end)
        _schedule_phase(p1_end * mfma_per_g, n_dsrd_half, 0, 0)
        rocdl.sched_barrier(0)
        if const_expr(transform_a is not None):
            transform_a(frag_A, k_tile, half_g, mma_k_iters)
        if const_expr(commit):
            rocdl.s_waitcnt(lgkmcnt=0)  # every wave is done reading tile k
            rocdl.s_barrier()
            commit_ab_to_lds(0)
            if const_expr(prefetch):
                load_ab_to_reg(k_tile + 2)
        _mma_groups(s, frag_A, frag_B, p1_end, p2_end)
        _schedule_phase(
            (p2_end - p1_end) * mfma_per_g,
            0,
            n_dswr if commit else 0,
            n_dswr if prefetch else 0,
        )
        rocdl.sched_barrier(0)
        if const_expr(commit):
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()  # tile k+1 visible before reading it
            read_tile_into(0, frag_A_ret, frag_B_ret, 0, half_c)
        _mma_groups(s, frag_A, frag_B, p2_end, mma_k_iters)
        _schedule_phase(
            (mma_k_iters - p2_end) * mfma_per_g,
            n_dsrd_half if commit else 0,
            0,
            0,
        )
        rocdl.sched_barrier(0)
        if const_expr(commit and transform_a is not None):
            transform_a(frag_A, k_tile + 1, 0, half_g)

    load_ab_to_reg(fx.Int32(0))
    rocdl.s_waitcnt(vmcnt=0)
    commit_ab_to_lds(0)
    load_ab_to_reg(fx.Int32(1))
    rocdl.s_waitcnt(lgkmcnt=0)
    rocdl.s_barrier()
    read_tile_into(0, frag_A_ret, frag_B_ret, 0, half_c)
    if const_expr(transform_a is not None):
        transform_a(frag_A, 0, 0, half_g)
    rocdl.sched_barrier(0)
    for k_tile in range(0, k_tiles - 2, 1):
        pipelined_iter(k_tile, True, True)
    pipelined_iter(k_tiles - 2, True, False)
    pipelined_iter(k_tiles - 1, False, False)


def _schedule_buffered_single_stage(s: Gfx942Mainloop):
    """Scheduler directives for one stages=1 buffered iteration.

    Forces each ds_write of the next tile into the tail of the MFMA run.
    """
    param = s.param
    m_rep = param.block_m // param.m_waves // param.mma_m
    n_rep = param.block_n // param.n_waves // param.mma_n
    n_mfma = m_rep * n_rep * (param.block_k // param.mma_k)
    n_dswr = param.ldg_a_iters + param.ldg_b_iters
    if n_mfma > n_dswr:
        rocdl.sched_mfma(n_mfma - n_dswr)
        for _ in range(n_dswr):
            rocdl.sched_dswr(1)
            rocdl.sched_mfma(1)


@flyc.jit
def _main_loop_buffered(s: Gfx942Mainloop):
    """128-bit buffer_load (global->reg) + ds_write (reg->LDS), depth-1 prefetch.

    The next tile's vmem loads are in flight during the current tile's MFMAs;
    the ds_write commits them once vmcnt drains. stages=1 overwrites the single
    LDS buffer once the current tile is in registers; stages=2 double-buffers.
    """
    stages = s.param.stages
    k_tiles = s.k_tiles
    frag_A = s.frag_A
    frag_B = s.frag_B
    read_tile_into = s.read_tile_into
    load_ab_to_reg = s.load_ab_to_reg
    commit_ab_to_lds = s.commit_ab_to_lds
    mma_tile = s.mma_tile
    transform_a = s.transform_a
    mma_k_iters = s.mma_k_iters
    load_ab_to_reg(fx.Int32(0))
    rocdl.s_waitcnt(vmcnt=0)
    commit_ab_to_lds(0)
    rocdl.s_waitcnt(lgkmcnt=0)
    rocdl.s_barrier()
    for k_tile in range(0, k_tiles - 1, 1):
        write_stage = (k_tile + 1) % stages
        # 128-bit buffer_load of tile k+1 (vmem, overlaps the MFMAs below)
        load_ab_to_reg(k_tile + 1)
        read_tile_into(k_tile % stages, s.frag_A_ret, s.frag_B_ret)
        if const_expr(stages == 1):
            # Single LDS buffer, but tile k's operands are already live
            # in registers. Once those ds_reads land, overwrite LDS with
            # tile k+1 while mma_tile consumes only registers.
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()
            rocdl.s_waitcnt(vmcnt=0)
            commit_ab_to_lds(write_stage)
            if const_expr(transform_a is not None):
                transform_a(frag_A, k_tile, 0, mma_k_iters)
            mma_tile(frag_A, frag_B)
            _schedule_buffered_single_stage(s)
            rocdl.s_waitcnt(lgkmcnt=0)  # tile k+1 writes landed
            rocdl.s_barrier()  # writes visible before next read
        else:
            if const_expr(transform_a is not None):
                transform_a(frag_A, k_tile, 0, mma_k_iters)
            mma_tile(frag_A, frag_B)
            rocdl.s_waitcnt(vmcnt=0)
            # ds_write tile k+1 into the other buffer (double-buffered)
            commit_ab_to_lds(write_stage)
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()
        rocdl.sched_barrier(0)
    read_tile_into((k_tiles - 1) % stages, s.frag_A_ret, s.frag_B_ret)
    if const_expr(transform_a is not None):
        transform_a(frag_A, k_tiles - 1, 0, mma_k_iters)
    mma_tile(frag_A, frag_B)


def _prime_async_stages(s: Gfx942Mainloop):
    """Preload the first stages-1 K tiles with async global->LDS copies."""
    for stage in range(s.param.stages - 1):
        s.async_load_b_to_lds(stage, stage)
        s.async_load_a_to_lds(stage, stage)
        rocdl.asyncmark()
    rocdl.sched_barrier(0)


@flyc.jit
def _main_loop_async(s: Gfx942Mainloop):
    """Async global->LDS DMA over ``stages`` buffers, reading the current tile inline.

    Needs k_tiles >= stages - 1 (checked on the host).
    """
    # Method calls on a name inside a dynamic loop mark it loop-carried, so
    # every helper the loops call is bound to a local first.
    stages = s.param.stages
    compute_stage = s.compute_stage
    async_load_a_to_lds = s.async_load_a_to_lds
    async_load_b_to_lds = s.async_load_b_to_lds
    _prime_async_stages(s)
    main_loop_end = s.k_tiles - (stages - 1)
    for k_tile in range(0, main_loop_end, 1):
        current_stage = k_tile % stages
        write_stage = (current_stage + stages - 1) % stages
        rocdl.wait_asyncmark(stages - 2)
        rocdl.s_barrier()
        async_load_b_to_lds(k_tile + (stages - 1), write_stage)
        async_load_a_to_lds(k_tile + (stages - 1), write_stage)
        rocdl.asyncmark()
        compute_stage(current_stage, k_tile)
        rocdl.sched_barrier(0)

    current_stage = main_loop_end % stages
    for i in range_constexpr(0, stages - 1):
        rocdl.wait_asyncmark(stages - 2 - i)
        rocdl.s_barrier()
        compute_stage(current_stage, main_loop_end + i)
        current_stage = (current_stage + 1) % stages


@flyc.jit
def _main_loop_async_reg_db(s: Gfx942Mainloop):
    """Async DMA with a register double buffer (stages >= 3).

    Tile k+1's LDS->reg read goes into the prefetch fragments *before* tile k's
    MFMAs, so the read stays in flight during the MFMAs and its lgkmcnt wait
    lands on the next iteration's work. stages >= 3 keeps the stage being read
    (k+1) apart from the one being async-written (k+stages-1).
    Needs k_tiles >= stages - 1 (checked on the host).
    """
    stages = s.param.stages
    frag_A = s.frag_A
    frag_B = s.frag_B
    frag_pf_A = s.frag_pf_A
    frag_pf_B = s.frag_pf_B
    read_tile_into = s.read_tile_into
    mma_tile = s.mma_tile
    async_load_a_to_lds = s.async_load_a_to_lds
    async_load_b_to_lds = s.async_load_b_to_lds
    transform_a = s.transform_a
    mma_k_iters = s.mma_k_iters
    _prime_async_stages(s)
    main_loop_end = s.k_tiles - (stages - 1)
    rocdl.wait_asyncmark(stages - 2)
    rocdl.s_barrier()
    read_tile_into(0, s.frag_A_ret, s.frag_B_ret)
    if const_expr(transform_a is not None):
        transform_a(frag_A, 0, 0, mma_k_iters)
    for k_tile in range(0, main_loop_end, 1):
        write_stage = (k_tile + stages - 1) % stages
        async_load_b_to_lds(k_tile + (stages - 1), write_stage)
        async_load_a_to_lds(k_tile + (stages - 1), write_stage)
        rocdl.asyncmark()
        rocdl.wait_asyncmark(stages - 2)
        rocdl.s_barrier()
        read_tile_into((k_tile + 1) % stages, s.frag_pf_A_ret, s.frag_pf_B_ret)
        mma_tile(frag_A, frag_B)
        if const_expr(transform_a is not None):
            transform_a(frag_pf_A, k_tile + 1, 0, mma_k_iters)
        # Rotate the just-read next tile into the MFMA fragments. The ds_read
        # above is only consumed here (after the MFMAs), so its lgkmcnt wait
        # is hidden behind the current tile's MFMAs.
        frag_A.store(frag_pf_A.load())
        frag_B.store(frag_pf_B.load())
        rocdl.sched_barrier(0)

    # Drain: compute the final stages-1 resident tiles, still prefetching one
    # tile ahead (no new async loads; progressively relax the wait count).
    for i in range_constexpr(0, stages - 1):
        if const_expr(i < stages - 2):
            rocdl.wait_asyncmark(stages - 2 - (i + 1))
            rocdl.s_barrier()
            read_tile_into(
                (main_loop_end + (i + 1)) % stages,
                s.frag_pf_A_ret,
                s.frag_pf_B_ret,
            )
        mma_tile(frag_A, frag_B)
        if const_expr(i < stages - 2):
            if const_expr(transform_a is not None):
                transform_a(frag_pf_A, main_loop_end + i + 1, 0, mma_k_iters)
            frag_A.store(frag_pf_A.load())
            frag_B.store(frag_pf_B.load())
        rocdl.sched_barrier(0)


@flyc.jit
def _reduce_k_waves(frag_C, k_waves, k_wave_idx, tid_mn, threads_per_k_slice, smem_red):
    """Tree-reduce the K waves' fp32 partial tiles into K wave 0.

    The exchange goes through k_waves // 2 LDS slots. Peer waves with the same
    tid_mn hold identically laid out fragments, so it is fragment-ordered in
    128-bit lane-interleaved chunks (ds_write_b128 / ds_read_b128). Waves that
    are already folded keep adding a peer slot; their sums are never read again.
    """
    n_vec4 = const_expr(fx.size(frag_C.shape).unpack() // 4)
    red = fx.make_view(
        smem_red,
        fx.make_layout(
            (4, threads_per_k_slice, n_vec4, k_waves // 2),
            (1, 4, 4 * threads_per_k_slice, 4 * threads_per_k_slice * n_vec4),
        ),
    )
    lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    frag_chunks = fx.make_layout((4, n_vec4), (1, 4))
    acc_chunks = fx.make_view(fx.get_iter(frag_C), frag_chunks)
    peer = fx.make_fragment_like(frag_C)
    peer_chunks = fx.make_view(fx.get_iter(peer), frag_chunks)

    for level in range_constexpr(k_waves.bit_length() - 1):
        active = const_expr(k_waves >> level)
        half = const_expr(active // 2)
        gpu.barrier()
        if (k_wave_idx >= half) & (k_wave_idx < active):
            fx.copy(lds_atom, acc_chunks, red[None, tid_mn, None, k_wave_idx - half])
        gpu.barrier()
        fx.copy(lds_atom, red[None, tid_mn, None, k_wave_idx % half], peer_chunks)
        frag_C.store(frag_C.load() + peer.load())

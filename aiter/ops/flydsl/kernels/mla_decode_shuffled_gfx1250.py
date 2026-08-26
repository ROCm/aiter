# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    gpu,
    range_constexpr,
    rocdl,
)
from flydsl.expr import math as fmath
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import ReductionOp, T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from aiter.ops.flydsl.kernels import buffer_ops


def _build_v4i32_buffer_rsrc(tensor, num_records_bytes=0xFFFFFFFF, arch=None):
    """Build a ``<4 x i32>`` V# (buffer resource descriptor) for ``s_buffer_load``.

    ``s_buffer_load`` intrinsics take the legacy ``<4 x i32>`` descriptor;
    ``rocdl.make.buffer.rsrc`` (the modern op used by
    ``buffer_ops.create_buffer_resource``) only produces ``!llvm.ptr<8>``, so
    we assemble the V# manually here.

    AMDGPU V# layout (low to high):
      word0: base[31:0]
      word1: base[47:32] (low 16) | stride<<16 (high 16)
      word2: num_records (bytes)
      word3: flags (DATA_FORMAT, NUM_FORMAT, OOB_SELECT, etc.)
    """
    base_i64 = fx.Int64(buffer_ops.extract_base_index(tensor, address_space=1))
    w0 = base_i64.to(fx.Int32)
    # word 1: only base[47:32] is meaningful for addresses, and stride=0 leaves
    # the high 16 bits zero.
    w1 = (base_i64 >> 32).to(fx.Int32)
    w2 = fx.Int32(num_records_bytes)
    w3 = fx.Int32(buffer_ops._get_buffer_flags(arch))
    return fx.Vector.from_elements([w0, w1, w2, w3], fx.Int32)


def _s_buffer_load_b32(rsrc_v4i32, byte_offset_i32):
    """Emit ``s_buffer_load_b32`` — scalar K$ load, result lands in an SGPR.

    Bypasses the VGPR → ``v_readfirstlane`` round-trip that the vmem
    ``buffer_load`` path requires, and uses the ``s_wait_kmcnt`` counter
    (separate from vmem ``s_wait_loadcnt``). No FlyDSL wrapper exposes the
    ``s_buffer_load`` intrinsics, so this drops to ``llvm.call_intrinsic``.

    Args:
        rsrc_v4i32: buffer descriptor as ``vector<4xi32>``
                    (from ``_build_v4i32_buffer_rsrc``).
        byte_offset_i32: byte offset (i32 SGPR value).

    Returns: i32 (scalar — uniform across the wave).
    """
    return _llvm.call_intrinsic(
        T.i32,
        "llvm.amdgcn.s.buffer.load.i32",
        [rsrc_v4i32.ir_value(), byte_offset_i32.ir_value(), fx.Int32(0).ir_value()],
        [],
        [],
    )


def _s_buffer_load_vec(rsrc_v4i32, byte_offset_i32, width):
    """Width-dispatched ``s_buffer_load_b{64,128}`` returning vector<widthxi32>.

    All lanes see the same vector; per-element extracts stay uniform.
    Module-level so the dispatch runs at Python trace time and the kernel's
    AST rewriter never sees the ``if/elif`` (which it would otherwise lift
    into ``scf.if`` branches and scope away the assigned value).

    Supports width ∈ {2, 4}. Width=1 should use ``_s_buffer_load_b32`` directly.
    """
    if width not in (2, 4):
        raise ValueError(
            f"_s_buffer_load_vec width must be 2 or 4 (got {width}); "
            "use _s_buffer_load_b32 for width=1."
        )
    return _llvm.call_intrinsic(
        T.vec(width, T.i32),
        f"llvm.amdgcn.s.buffer.load.v{width}i32",
        [rsrc_v4i32.ir_value(), byte_offset_i32.ir_value(), fx.Int32(0).ir_value()],
        [],
        [],
    )


# fmath.exp2 lowers to llvm.exp2.f32, which adds extra instructions to guard
# x < -126 underflow. The softmax sum's ULP is big enough that the guard buys
# nothing here, so use the bare hardware exp2.
def _amdgcn_exp2_f32(x):
    return _llvm.call_intrinsic(
        T.f32, "llvm.amdgcn.exp2.f32", [fx.Float32(x).ir_value()], [], []
    )


WAVE_SIZE = 32
WMMA_M = 16
WMMA_N = 16
WMMA_K = 32

# Q NOT pre-shuffled needs padding
Q_LORA_PAD = 8
Q_ROPE_PAD = 8


def _decompose_bt_widths(n):
    """Greedy-split a block count `n` into s_buffer_load-able vector widths {4,2,1}."""
    widths = []
    while n >= 4:
        widths.append(4)
        n -= 4
    if n >= 2:
        widths.append(2)
        n -= 2
    if n == 1:
        widths.append(1)
    return widths


@functools.lru_cache(maxsize=1024)
def compile_mla_decode_main(
    *,
    KV_LORA_RANK: int = 512,
    QK_ROPE_HEAD_DIM: int = 64,
    KV_BLOCK_SIZE: int = 64,
    NUM_Q_HEADS: int = 16,
    NUM_SEGS: int = 2,
    KV_COMPUTE_BLOCK_SIZE: int = 64,
    NUM_WARPS: int = 2,
    dtype: str = "bf16",
    waves_per_eu: int = 1,
    WARP_TOKEN_SPLIT: bool = True,
    WARP_HEAD_SPLIT: bool = False,
    WRITE_FINAL_OUTPUT: bool = False,
):

    # Warps split exactly one axis: Q head tiles, tokens, or d_c.
    if WARP_HEAD_SPLIT and WARP_TOKEN_SPLIT:
        raise ValueError("WARP_HEAD_SPLIT and WARP_TOKEN_SPLIT are mutually exclusive")

    if NUM_WARPS & (NUM_WARPS - 1):
        raise ValueError(f"NUM_WARPS must be a power of 2, got {NUM_WARPS}")

    SPLIT = NUM_WARPS if WARP_TOKEN_SPLIT else 1
    NUM_PARTITIONS = NUM_SEGS * SPLIT

    if WRITE_FINAL_OUTPUT and NUM_PARTITIONS != 1:
        raise ValueError(
            f"WRITE_FINAL_OUTPUT requires NUM_PARTITIONS == 1, got {NUM_PARTITIONS} "
            f"(NUM_SEGS={NUM_SEGS}, SPLIT={SPLIT})"
        )

    QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # K width for QK (lora + rope)
    V_HEAD_DIM = KV_LORA_RANK  # V width for PV (lora only)

    # ---- tile-shape constraints (16x16x32 ) ----
    if KV_LORA_RANK % WMMA_K != 0:
        raise ValueError(
            f"KV_LORA_RANK (={KV_LORA_RANK}) must be a multiple of WMMA_K={WMMA_K}"
        )
    if QK_ROPE_HEAD_DIM % WMMA_K != 0:
        raise ValueError(
            f"QK_ROPE_HEAD_DIM (={QK_ROPE_HEAD_DIM}) must be a multiple of WMMA_K={WMMA_K}"
        )
    if V_HEAD_DIM % WMMA_N != 0:
        raise ValueError(
            f"V_HEAD_DIM (={V_HEAD_DIM}) must be a multiple of WMMA_N={WMMA_N}"
        )
    if KV_BLOCK_SIZE % WMMA_M != 0:
        raise ValueError(
            f"KV_BLOCK_SIZE must be multiple of {WMMA_M}, got {KV_BLOCK_SIZE}"
        )
    if KV_COMPUTE_BLOCK_SIZE % WMMA_K != 0:
        raise ValueError(
            f"KV_COMPUTE_BLOCK_SIZE must be multiple of WMMA_K={WMMA_K}, got {KV_COMPUTE_BLOCK_SIZE}"
        )

    if KV_COMPUTE_BLOCK_SIZE % KV_BLOCK_SIZE and KV_BLOCK_SIZE % KV_COMPUTE_BLOCK_SIZE:
        raise ValueError(
            f"KV_COMPUTE_BLOCK_SIZE {KV_COMPUTE_BLOCK_SIZE} and KV_BLOCK_SIZE "
            f"{KV_BLOCK_SIZE} must divide one another"
        )
    N_PV_TILES = V_HEAD_DIM // WMMA_N
    WARP_DC_SPLIT = not (WARP_HEAD_SPLIT or WARP_TOKEN_SPLIT)
    if WARP_DC_SPLIT and N_PV_TILES % NUM_WARPS != 0:
        raise ValueError(
            f"NUM_WARPS ({NUM_WARPS}) must divide V_HEAD_DIM/WMMA_N ({N_PV_TILES})"
        )
    if dtype not in ("bf16", "f16"):
        raise ValueError(f"dtype must be 'bf16' or 'f16', got {dtype!r}")

    # One compute tile gathers BLOCKS_PER_COMPUTE physical pages
    TOKENS_PER_LOAD = min(KV_BLOCK_SIZE, KV_COMPUTE_BLOCK_SIZE)
    BLOCKS_PER_COMPUTE = KV_COMPUTE_BLOCK_SIZE // TOKENS_PER_LOAD
    TILES_PER_BLOCK = KV_BLOCK_SIZE // TOKENS_PER_LOAD
    BT_LOAD_WIDTHS = _decompose_bt_widths(BLOCKS_PER_COMPUTE)
    BT_LOAD_OFFSETS = [sum(BT_LOAD_WIDTHS[:i]) for i in range(len(BT_LOAD_WIDTHS))]

    # Each physical block brought in by 2 async TDM loads (one lora blob, one rope blob)
    K_OPS_PER_WAVE = 2 * BLOCKS_PER_COMPUTE

    NUM_QGSP_TILES = (NUM_Q_HEADS + WMMA_M - 1) // WMMA_M
    QGSP_PADDED = NUM_QGSP_TILES * WMMA_M
    if WARP_HEAD_SPLIT and NUM_QGSP_TILES % NUM_WARPS:
        raise ValueError(
            f"WARP_HEAD_SPLIT needs NUM_WARPS ({NUM_WARPS}) to divide the "
            f"{NUM_QGSP_TILES} 16-head tiles"
        )
    NQGSP_LOCAL = (NUM_QGSP_TILES // NUM_WARPS) if WARP_HEAD_SPLIT else NUM_QGSP_TILES

    K_QK_LORA_TILES = KV_LORA_RANK // WMMA_K
    K_QK_ROPE_TILES = QK_ROPE_HEAD_DIM // WMMA_K
    N_QK_TILES = KV_COMPUTE_BLOCK_SIZE // WMMA_N
    K_PV_TILES = KV_COMPUTE_BLOCK_SIZE // WMMA_K
    N_PV_TILES_PER_WARP = N_PV_TILES // NUM_WARPS

    # without WARP_TOKEN_SPLIT each warp redundantly does QK to avoid LDS roundtrip for PV.
    # with WARP_TOKEN_SPLIT we split QK along token dimension among warps. each warp owns
    # N_QK_TILES/NUM_WARPS 16 token QK tiles. for PV each warp pairs two QK tiles to make a
    # 32 token tile (needed since WMMA_K=32). So KV_COMPUTE_BLOCK_SIZE needs to be a multiple
    # of 32*NUM_WARPS.
    if WARP_TOKEN_SPLIT:
        if N_QK_TILES % NUM_WARPS != 0:
            raise ValueError(
                f"WARP_TOKEN_SPLIT requires N_QK_TILES ({N_QK_TILES}) divisible by "
                f"NUM_WARPS ({NUM_WARPS}); KV_COMPUTE_BLOCK_SIZE={KV_COMPUTE_BLOCK_SIZE}."
            )
        _nqk_local = N_QK_TILES // NUM_WARPS
        if _nqk_local < 2 or _nqk_local % 2 != 0:
            raise ValueError(
                f"WARP_TOKEN_SPLIT requires each warp to own an even number (>=2) of "
                f"16-token score-tiles so they pair into 32-token PV K-steps; got "
                f"NQK_LOCAL={_nqk_local} (KV_COMPUTE_BLOCK_SIZE={KV_COMPUTE_BLOCK_SIZE}, "
                f"NUM_WARPS={NUM_WARPS}). Require KV_COMPUTE_BLOCK_SIZE to be a multiple "
                f"of 32*NUM_WARPS={32 * NUM_WARPS}."
            )

    block_threads = NUM_WARPS * WAVE_SIZE
    NUM_KV_STAGES = 2
    elem_bytes = 2  # bf16 / f16

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    # Shuffled-layout strides in ELEMENTS, to step one 16-token group within a
    # block's blob.
    LORA_BSG_STRIDE = (KV_LORA_RANK // WMMA_M) * 256
    ROPE_BSG_STRIDE = (QK_ROPE_HEAD_DIM // WMMA_M) * 256

    # Padded Q LDS
    Q_LORA_ROW = KV_LORA_RANK + Q_LORA_PAD
    Q_ROPE_ROW = QK_ROPE_HEAD_DIM + Q_ROPE_PAD

    lora_compute_block_elems = TOKENS_PER_LOAD * KV_LORA_RANK
    rope_compute_block_elems = TOKENS_PER_LOAD * QK_ROPE_HEAD_DIM
    # lora part of a page starts after rope part
    page_lora_elems = KV_BLOCK_SIZE * KV_LORA_RANK

    # KV LDS slabs
    kv_lora_elems = BLOCKS_PER_COMPUTE * lora_compute_block_elems
    kv_rope_elems = BLOCKS_PER_COMPUTE * rope_compute_block_elems
    kv_lora_bytes = kv_lora_elems * elem_bytes
    kv_rope_bytes = kv_rope_elems * elem_bytes

    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="mla_decode_shuf_lds"
    )
    q_lora_elems = QGSP_PADDED * Q_LORA_ROW
    q_rope_elems = QGSP_PADDED * Q_ROPE_ROW

    kv_lora_off = allocator._align(allocator.ptr, 16)
    kv_rope_off = allocator._align(kv_lora_off + NUM_KV_STAGES * kv_lora_bytes, 16)
    kv_end = kv_rope_off + NUM_KV_STAGES * kv_rope_bytes

    # Q is read from LDS exactly once (prologue -> registers) and never touched again, so its
    # LDS bytes are dead for the whole main loop. So it is aliased over the KV slabs
    q_bytes = (q_lora_elems + q_rope_elems) * elem_bytes
    Q_ON_STAGE0 = q_bytes > kv_lora_bytes

    # if Q can fit within one kv lora slab we can just alias over second kv lora buffer
    # which lets us issue the first kv load with the Q load instead of waiting for Q LDS
    # to drain
    q_lora_off = kv_lora_off if Q_ON_STAGE0 else kv_lora_off + kv_lora_bytes
    q_rope_off = q_lora_off + q_lora_elems * elem_bytes
    assert q_lora_off % 16 == 0 and q_rope_off % 16 == 0, "aliased Q offsets misaligned"
    allocator.ptr = max(kv_end, q_rope_off + q_rope_elems * elem_bytes)

    @flyc.kernel
    def kernel_mla_decode_main(
        arg_out: fx.Tensor,
        arg_max_logits: fx.Tensor,
        arg_exp_sums: fx.Tensor,
        arg_query: fx.Tensor,
        arg_kv_cache: fx.Tensor,
        arg_block_tables: fx.Tensor,
        arg_seq_lens: fx.Tensor,
        i32_qk_scale: fx.Int32,
        i32_num_seqs: fx.Int32,
        i32_max_blocks_per_seq: fx.Int32,
    ):
        # Grid = (num_seqs, NUM_SEGS): this block owns one sequence and one KV segment.
        seq_idx = gpu.block_id("x")
        seg_idx = gpu.block_id("y")
        tid = gpu.thread_id("x")
        wave_id = tid / WAVE_SIZE
        lane_id = tid % WAVE_SIZE
        # Lane decomposition
        lane_kgrp = lane_id / WMMA_M
        lane16 = lane_id % WMMA_M
        # with head split each warp owns NQGSP_LOCAL consecutive head tiles.
        head_row_off = (
            wave_id * (NQGSP_LOCAL * WMMA_M) if WARP_HEAD_SPLIT else fx.Index(0)
        )

        sl_rsrc = buffer_ops.create_buffer_resource(arg_seq_lens, max_size=True)
        seq_len_i32 = fx.Int32(
            buffer_ops.buffer_load(sl_rsrc, seq_idx, vec_width=1, dtype=T.i32)
        )
        seq_len = fx.Index(seq_len_i32)

        elem_ty = T.bf16 if dtype == "bf16" else T.f16
        elem_dtype = fx.BFloat16 if dtype == "bf16" else fx.Float16

        # Arch-dispatched WMMA atom: lowers to wmma_f32_16x16x32_{bf16,f16}
        # with modC=0 and no operand reuse.
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_dtype))

        def _wmma(a_frag, b_frag, acc):
            return fx.Vector(
                _fly.mma_atom_call_ssa(
                    [T.vec(8, T.f32)],
                    mma_atom,
                    a_frag.ir_value(),
                    b_frag.ir_value(),
                    acc.ir_value(),
                )
            )

        # ---- this segment's tile range [tile_start, tile_end) ----
        num_tiles = (seq_len + (KV_COMPUTE_BLOCK_SIZE - 1)) / KV_COMPUTE_BLOCK_SIZE
        tiles_per_seg = (num_tiles + (NUM_SEGS - 1)) / NUM_SEGS
        tile_start = seg_idx * tiles_per_seg
        tile_end_raw = (seg_idx + 1) * tiles_per_seg
        tile_end = (tile_end_raw < num_tiles).select(tile_end_raw, num_tiles)
        is_live = tile_start < tile_end
        iters_this_seg = is_live.select(tile_end - tile_start, fx.Index(0))

        stride_o_seq = NUM_PARTITIONS * NUM_Q_HEADS * V_HEAD_DIM
        stride_o_part = NUM_Q_HEADS * V_HEAD_DIM
        stride_o_row = V_HEAD_DIM
        stride_lse_seq = NUM_PARTITIONS * NUM_Q_HEADS
        stride_lse_part = NUM_Q_HEADS
        stride_bt_seq = fx.Index(i32_max_blocks_per_seq)

        bt_rsrc_v4i32 = _build_v4i32_buffer_rsrc(arg_block_tables, arch=gpu_arch)

        base = allocator.get_base()
        q_lora_lds = SmemPtr(base, q_lora_off, elem_ty, shape=(q_lora_elems,))
        q_rope_lds = SmemPtr(base, q_rope_off, elem_ty, shape=(q_rope_elems,))
        kv_lora_lds = SmemPtr(
            base, kv_lora_off, elem_ty, shape=(NUM_KV_STAGES * kv_lora_elems,)
        )
        kv_rope_lds = SmemPtr(
            base, kv_rope_off, elem_ty, shape=(NUM_KV_STAGES * kv_rope_elems,)
        )
        q_lora_lds.get()
        q_rope_lds.get()
        kv_lora_lds.get()
        kv_rope_lds.get()

        # We multiply scale by log2e so we can use exp2 later for softmax (exp2 op faster than exp)
        LOG2E = 1.4426950408889634
        qk_scale_log2_scalar = i32_qk_scale.bitcast(fx.Float32) * LOG2E

        neg_inf_f32 = fx.Float32(float("-inf"))
        zero_f32 = fx.Float32(0.0)
        # Padded rows are masked to a large FINITE negative, not -inf, so the
        # softmax of an all-masked row stays finite (-inf - -inf would be NaN).
        NEG_FINITE_MAX = -3.4e38
        neg_finite_max_vec8 = fx.Vector.filled(8, NEG_FINITE_MAX, fx.Float32)
        neg_finite_max_f32 = fx.Float32(NEG_FINITE_MAX)

        # Loads one 16-row x 32-K WMMA fragment for Q from Q LDS.
        def _load_q_frag(lds_ptr, row_base_idx, k_base_elem, row_stride):
            lds_mem = lds_ptr.get()
            chunks = []
            for k0 in range_constexpr(2):
                kk_base = (lane_kgrp + k0 * 2) * 8 + k_base_elem
                elem_off = row_base_idx * row_stride + kk_base
                chunks.append(fx.Vector.load(T.vec(8, elem_ty), lds_mem, [elem_off]))
            return chunks[0].shuffle(chunks[1], list(range(16)))

        # ---- shuffled K fragment loader (straight ds_load_b128) ----
        def _load_shuf_K(lds_ptr, n_tile, ks, bsg_stride, stage_off):
            lds_mem = lds_ptr.get()
            base_t = stage_off + n_tile * bsg_stride
            o0 = base_t + (2 * ks) * 256 + lane_id * 8
            o1 = base_t + (2 * ks + 1) * 256 + lane_id * 8
            c0 = fx.Vector.load(T.vec(8, elem_ty), lds_mem, [o0])
            c1 = fx.Vector.load(T.vec(8, elem_ty), lds_mem, [o1])
            return c0.shuffle(c1, list(range(16)))

        # ---- shuffled V fragment loader (transpose ds_load_tr16_b128) ----
        lane8 = lane16 % 8
        lane_ngrp = lane16 / 8

        def _load_shuf_V_tr(pv_n_global, ks, stage_off):
            lds_mem = kv_lora_lds.get()
            base = pv_n_global * 256 + lane_ngrp * 128 + (lane_kgrp * 8 + lane8) * 8
            o0 = stage_off + (2 * ks) * LORA_BSG_STRIDE + base
            o1 = stage_off + (2 * ks + 1) * LORA_BSG_STRIDE + base
            v0 = fx.Vector(
                rocdl.lds_transpose_load(T.vec(8, elem_ty), lds_mem, o0, elem_bytes)
            )
            v1 = fx.Vector(
                rocdl.lds_transpose_load(T.vec(8, elem_ty), lds_mem, o1, elem_bytes)
            )
            return v0.shuffle(v1, list(range(16)))

        # ---- block table -> physical page IDs ----
        live_blocks = (
            seq_len + (KV_BLOCK_SIZE - 1)
        ) / KV_BLOCK_SIZE  # pages this seq uses

        # Returns the BLOCKS_PER_COMPUTE physical page IDs
        # Any logical page past live_blocks is forced to page 0
        def _phys_blks_for_compute(tile_global_idx):
            base_logical = (
                tile_global_idx / TILES_PER_BLOCK
                if TILES_PER_BLOCK > 1
                else tile_global_idx * BLOCKS_PER_COMPUTE
            )
            bt_base = seq_idx * stride_bt_seq + base_logical
            out = []
            for ldi in range_constexpr(len(BT_LOAD_WIDTHS)):
                this_width = BT_LOAD_WIDTHS[ldi]
                this_offset = BT_LOAD_OFFSETS[ldi]
                bt_off_bytes = fx.Int32((bt_base + this_offset) * 4)
                if this_width == 1:
                    phys_i32 = _s_buffer_load_b32(bt_rsrc_v4i32, bt_off_bytes)
                    in_range = (base_logical + this_offset) < live_blocks
                    out.append(fx.Index(in_range.select(phys_i32, 0)))
                else:
                    phys_vec = fx.Vector(
                        _s_buffer_load_vec(bt_rsrc_v4i32, bt_off_bytes, this_width)
                    )
                    for b in range_constexpr(this_width):
                        in_range = (base_logical + (this_offset + b)) < live_blocks
                        out.append(fx.Index(in_range.select(phys_vec[b], 0)))
            return out

        BLOCK_STRIDE = KV_BLOCK_SIZE * QK_HEAD_DIM

        # for cases where KVC < KVB sub_tok tells the start token of the subtile we are loading from the page/block
        def _sub_tok(tile_global_idx):
            return (
                (tile_global_idx % TILES_PER_BLOCK) * TOKENS_PER_LOAD
                if TILES_PER_BLOCK > 1
                else fx.Index(0)
            )

        def _issue_kv_load_single_block(
            phys_blk, lora_byte_off, rope_byte_off, sub_tok
        ):
            lora_desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_kv_cache,
                lds_memref=kv_lora_lds.get(),
                global_offset=(phys_blk, sub_tok * KV_LORA_RANK),
                tensor_shape=(1, lora_compute_block_elems),
                strides=(BLOCK_STRIDE, 1),
                tile_shape=(1, lora_compute_block_elems),
                elem_bytes=elem_bytes,
                num_warps=NUM_WARPS,
                lds_byte_offset=lora_byte_off,
            )
            tdm_ops.tensor_load_2d(lora_desc)
            rope_desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_kv_cache,
                lds_memref=kv_rope_lds.get(),
                global_offset=(
                    phys_blk,
                    sub_tok * QK_ROPE_HEAD_DIM + page_lora_elems,
                ),
                tensor_shape=(1, rope_compute_block_elems),
                strides=(BLOCK_STRIDE, 1),
                tile_shape=(1, rope_compute_block_elems),
                elem_bytes=elem_bytes,
                num_warps=NUM_WARPS,
                lds_byte_offset=rope_byte_off,
            )
            tdm_ops.tensor_load_2d(rope_desc)

        # Issue loads for a whole compute tile
        def _issue_kv_tile_loads(
            phys_blks_list, lora_stage_off, rope_stage_off, sub_tok
        ):
            lora_block_bytes = lora_compute_block_elems * elem_bytes
            rope_block_bytes = rope_compute_block_elems * elem_bytes
            for b in range_constexpr(BLOCKS_PER_COMPUTE):
                _issue_kv_load_single_block(
                    phys_blks_list[b],
                    lora_stage_off + b * lora_block_bytes,
                    rope_stage_off + b * rope_block_bytes,
                    sub_tok,
                )

        def _issue_q_load():
            q_outer_off = seq_idx * NUM_Q_HEADS
            lora_desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_query,
                lds_memref=q_lora_lds.get(),
                global_offset=(q_outer_off, fx.Index(0)),
                tensor_shape=(NUM_Q_HEADS, KV_LORA_RANK),
                strides=(QK_HEAD_DIM, 1),
                tile_shape=(NUM_Q_HEADS, KV_LORA_RANK),
                elem_bytes=elem_bytes,
                pad_interval=KV_LORA_RANK,
                pad_amount=Q_LORA_PAD,
                num_warps=NUM_WARPS,
            )
            tdm_ops.tensor_load_2d(lora_desc)
            rope_desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_query,
                lds_memref=q_rope_lds.get(),
                global_offset=(q_outer_off, fx.Index(KV_LORA_RANK)),
                tensor_shape=(NUM_Q_HEADS, QK_ROPE_HEAD_DIM),
                strides=(QK_HEAD_DIM, 1),
                tile_shape=(NUM_Q_HEADS, QK_ROPE_HEAD_DIM),
                elem_bytes=elem_bytes,
                pad_interval=QK_ROPE_HEAD_DIM,
                pad_amount=Q_ROPE_PAD,
                num_warps=NUM_WARPS,
            )
            tdm_ops.tensor_load_2d(rope_desc)

        # ---- prologue ----
        # we issue Q load first if it is small enough to fit within second kv buffer we issue
        # fist kv load as well. otherwise if Q is big it will also need first kv buffer as well
        # so we issue Q load drain it and only then issue the first kv load
        _issue_q_load()
        phys_blks_first = _phys_blks_for_compute(tile_start)
        if Q_ON_STAGE0:
            tdm_ops.tensor_wait(0)  # nothing else in flight; drain Q
        else:
            _issue_kv_tile_loads(
                phys_blks_first, fx.Index(0), fx.Index(0), _sub_tok(tile_start)
            )
            tdm_ops.tensor_wait(K_OPS_PER_WAVE)  # only waiting for Q
        gpu.barrier()

        # Read ALL of Q's WMMA fragments into registers
        q_lora_frags = [[None] * K_QK_LORA_TILES for _ in range(NQGSP_LOCAL)]
        q_rope_frags = [[None] * K_QK_ROPE_TILES for _ in range(NQGSP_LOCAL)]
        for qt in range_constexpr(NQGSP_LOCAL):
            q_row = head_row_off + qt * WMMA_M + lane16
            for ks in range_constexpr(K_QK_LORA_TILES):
                q_lora_frags[qt][ks] = _load_q_frag(
                    q_lora_lds, q_row, ks * WMMA_K, Q_LORA_ROW
                )
            for ks in range_constexpr(K_QK_ROPE_TILES):
                q_rope_frags[qt][ks] = _load_q_frag(
                    q_rope_lds, q_row, ks * WMMA_K, Q_ROPE_ROW
                )

        rocdl.s_wait_dscnt(0)
        gpu.barrier()

        # Q is dead now: safe to overwrite stage 0.
        if Q_ON_STAGE0:
            _issue_kv_tile_loads(
                phys_blks_first, fx.Index(0), fx.Index(0), _sub_tok(tile_start)
            )

        # if WARP_TOKEN_SPLIT on then we don't split PV accross warps
        # each warp owns the full PV  accumulator with their subset of tokens and then combined in reduce kernel
        P = N_PV_TILES_PER_WARP if WARP_DC_SPLIT else N_PV_TILES

        def _pack(m_list, l_list, pv_list, cur_stage):
            out = []
            for qt in range_constexpr(NQGSP_LOCAL):
                out.append(m_list[qt].ir_value())
                out.append(l_list[qt].ir_value())
                out.extend(pv_list[qt])
            out.append(cur_stage)
            return out

        # Initial state: m = -inf, l = 0, PV accumulators = 0, double-buffer stage = 0.
        def _init_state():
            m_list = [neg_inf_f32 for _ in range(NQGSP_LOCAL)]
            l_list = [zero_f32 for _ in range(NQGSP_LOCAL)]
            pv_list = [
                [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(P)]
                for _ in range(NQGSP_LOCAL)
            ]
            return _pack(m_list, l_list, pv_list, fx.Index(0))

        init_state = _init_state()

        # ===== main loop: one KV compute tile (KVC tokens) per iteration =====
        for iv, state in range(
            fx.Index(0), iters_this_seg, fx.Index(1), init=init_state
        ):
            m_list, l_list, pv_list = [], [], []
            si = 0
            for qt in range_constexpr(NQGSP_LOCAL):
                m_list.append(fx.Float32(state[si]))
                si += 1
                l_list.append(fx.Float32(state[si]))
                si += 1
                pv_list.append([fx.Vector(v) for v in state[si : si + P]])
                si += P

            cur_stage = fx.Index(state[si])
            nxt_stage = fx.Index(1) - cur_stage
            nxt_lora_byte_off = nxt_stage * kv_lora_bytes
            nxt_rope_byte_off = nxt_stage * kv_rope_bytes
            cur_lora_elem_off = cur_stage * kv_lora_elems
            cur_rope_elem_off = cur_stage * kv_rope_elems

            NQK_LOCAL = (N_QK_TILES // NUM_WARPS) if WARP_TOKEN_SPLIT else N_QK_TILES
            qk_lora_off = (
                cur_lora_elem_off + wave_id * (NQK_LOCAL * LORA_BSG_STRIDE)
                if WARP_TOKEN_SPLIT
                else cur_lora_elem_off
            )
            qk_rope_off = (
                cur_rope_elem_off + wave_id * (NQK_LOCAL * ROPE_BSG_STRIDE)
                if WARP_TOKEN_SPLIT
                else cur_rope_elem_off
            )
            pv_lora_off = qk_lora_off

            g = tile_start + iv  # global tile index in this sequence
            tile_first_tok = g * KV_COMPUTE_BLOCK_SIZE
            is_not_last = iv < iters_this_seg - 1

            # Prefetch the NEXT tile
            if is_not_last:
                next_phys = _phys_blks_for_compute(g + 1)
                _issue_kv_tile_loads(
                    next_phys,
                    nxt_lora_byte_off,
                    nxt_rope_byte_off,
                    _sub_tok(g + 1),
                )
                tdm_ops.tensor_wait(K_OPS_PER_WAVE)
            else:
                tdm_ops.tensor_wait(0)
            gpu.barrier()

            qk_accs = [
                [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(NQK_LOCAL)]
                for _ in range(NQGSP_LOCAL)
            ]
            for ks in range_constexpr(K_QK_LORA_TILES):
                for n_tile in range_constexpr(NQK_LOCAL):
                    k_frag = _load_shuf_K(
                        kv_lora_lds, n_tile, ks, LORA_BSG_STRIDE, qk_lora_off
                    )
                    for qt in range_constexpr(NQGSP_LOCAL):
                        qk_accs[qt][n_tile] = _wmma(
                            k_frag, q_lora_frags[qt][ks], qk_accs[qt][n_tile]
                        )
            for ks in range_constexpr(K_QK_ROPE_TILES):
                for n_tile in range_constexpr(NQK_LOCAL):
                    k_frag = _load_shuf_K(
                        kv_rope_lds, n_tile, ks, ROPE_BSG_STRIDE, qk_rope_off
                    )
                    for qt in range_constexpr(NQGSP_LOCAL):
                        qk_accs[qt][n_tile] = _wmma(
                            k_frag, q_rope_frags[qt][ks], qk_accs[qt][n_tile]
                        )

            # ---- online-softmax update ----
            for qt in range_constexpr(NQGSP_LOCAL):
                # Apply scale*log2e so we can use exp2
                for n_tile in range_constexpr(NQK_LOCAL):
                    qk_accs[qt][n_tile] = qk_accs[qt][n_tile] * qk_scale_log2_scalar

                tile_n_base = (
                    wave_id * (NQK_LOCAL * WMMA_M) if WARP_TOKEN_SPLIT else fx.Index(0)
                )

                # Token mask: set any tile token at/after seq_len to a large FINITE negative
                for n_tile in range_constexpr(NQK_LOCAL):
                    new_vals = []
                    for mi in range_constexpr(8):
                        tok_abs = fx.Int32(
                            tile_first_tok
                            + tile_n_base
                            + n_tile * WMMA_M
                            + lane_kgrp * 8
                            + mi
                        )
                        in_range = tok_abs < seq_len_i32
                        new_vals.append(
                            in_range.select(qk_accs[qt][n_tile][mi], neg_finite_max_f32)
                        )
                    qk_accs[qt][n_tile] = fx.Vector.from_elements(new_vals, fx.Float32)

                # row mask when NUM_Q_HEADS isn't a multiple of 16:
                if QGSP_PADDED != NUM_Q_HEADS:
                    is_row_valid = (head_row_off + qt * WMMA_M + lane16) < NUM_Q_HEADS
                    for n_tile in range_constexpr(NQK_LOCAL):
                        qk_accs[qt][n_tile] = fx.Vector(
                            is_row_valid.select(
                                qk_accs[qt][n_tile], neg_finite_max_vec8
                            )
                        )

                # Row max
                m_state = m_list[qt]
                l_state = l_list[qt]
                local_max = qk_accs[qt][0].reduce(ReductionOp.MAX)
                for n_tile in range_constexpr(1, NQK_LOCAL):
                    local_max = arith.maximumf(
                        local_max, qk_accs[qt][n_tile].reduce(ReductionOp.MAX)
                    )
                peer = gpu.shuffle_xor(local_max, 16, WAVE_SIZE)
                row_max = arith.maximumf(local_max, peer)

                new_m = fx.Float32(arith.maximumf(m_state, row_max))

                alpha = fx.Float32(_amdgcn_exp2_f32(m_state - new_m))

                # Probabilities p = exp2(score - new_m)
                row_sum_partial = zero_f32
                for n_tile in range_constexpr(NQK_LOCAL):
                    diff = qk_accs[qt][n_tile] - new_m
                    p_vec = fx.Vector.from_elements(
                        [
                            fx.Float32(_amdgcn_exp2_f32(diff[mi]))
                            for mi in range_constexpr(8)
                        ],
                        fx.Float32,
                    )
                    qk_accs[qt][n_tile] = p_vec
                    row_sum_partial = row_sum_partial + p_vec.reduce(
                        ReductionOp.ADD, fastmath=arith.FastMathFlags.fast
                    )

                # Complete the row sum across the two half-waves, then update running l and m.
                peer = gpu.shuffle_xor(row_sum_partial, 16, WAVE_SIZE)
                row_sum = row_sum_partial + peer

                l_list[qt] = fx.Float32(alpha * l_state + row_sum)
                m_list[qt] = new_m

                # rescale pv_accs by alpha
                for pv_n in range_constexpr(P):
                    pv_list[qt][pv_n] = pv_list[qt][pv_n] * alpha

            # ---- PV ----
            # if WARP_TOKEN_SPLIT each warp responsible for it's share of token's PV and PV not split accross warps
            # if not WARP_TOEKN_SPLIT then each warp has a copy of full P and PV is split accross warps in N dim
            K_PV_LOCAL = (NQK_LOCAL // 2) if WARP_TOKEN_SPLIT else K_PV_TILES
            pv_v_off = pv_lora_off if WARP_TOKEN_SPLIT else cur_lora_elem_off
            for qt in range_constexpr(NQGSP_LOCAL):
                for pv_n in range_constexpr(P):
                    pv_n_global = wave_id * P + pv_n if WARP_DC_SPLIT else pv_n
                    for ks in range_constexpr(K_PV_LOCAL):
                        p_f32 = qk_accs[qt][2 * ks].shuffle(
                            qk_accs[qt][2 * ks + 1], list(range(16))
                        )
                        p_frag = p_f32.to(elem_dtype)
                        v_frag = _load_shuf_V_tr(pv_n_global, ks, pv_v_off)
                        pv_list[qt][pv_n] = _wmma(v_frag, p_frag, pv_list[qt][pv_n])

            # Barrier so every warp is done reading this stage's LDS before the next
            # iteration prefetches into (and overwrites) it.
            gpu.barrier()
            results = yield _pack(m_list, l_list, pv_list, nxt_stage)

        # ---- epilogue ----
        m_final, l_final, pv_final = [], [], []
        si = 0
        for qt in range_constexpr(NQGSP_LOCAL):
            m_final.append(fx.Float32(results[si]))
            si += 1
            l_final.append(fx.Float32(results[si]))
            si += 1
            pv_final.append([fx.Vector(v) for v in results[si : si + P]])
            si += P

        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)

        if WRITE_FINAL_OUTPUT:
            # since only one partition we normalize here so no need to call reduce kernel
            out_seq_base = seq_idx * stride_o_seq
            is_empty = seq_len == 0
            zero_vec_half = fx.Vector.filled(8, 0.0, elem_dtype)

            for qt in range_constexpr(NQGSP_LOCAL):
                row = head_row_off + qt * WMMA_M + lane16
                row_valid = row < NUM_Q_HEADS  # skip padded head rows
                inv_l = fx.Float32(rocdl.rcp(T.f32, l_final[qt]))
                row_base = out_seq_base + row * stride_o_row

                for pv_n in range_constexpr(P):
                    pv_n_global = wave_id * P + pv_n if WARP_DC_SPLIT else pv_n
                    head_col_base = pv_n_global * WMMA_M + lane_kgrp * 8
                    vec_half = (pv_final[qt][pv_n] * inv_l).to(elem_dtype)
                    vec_half = fx.Vector(is_empty.select(zero_vec_half, vec_half))
                    if row_valid:
                        buffer_ops.buffer_store(
                            vec_half, out_rsrc, row_base + head_col_base
                        )
        else:
            ml_rsrc = buffer_ops.create_buffer_resource(arg_max_logits, max_size=True)
            es_rsrc = buffer_ops.create_buffer_resource(arg_exp_sums, max_size=True)

            part_idx = seg_idx * SPLIT + (wave_id if WARP_TOKEN_SPLIT else 0)
            out_base = seq_idx * stride_o_seq + part_idx * stride_o_part
            lse_base = seq_idx * stride_lse_seq + part_idx * stride_lse_part

            for qt in range_constexpr(NQGSP_LOCAL):
                row = head_row_off + qt * WMMA_M + lane16
                row_valid = row < NUM_Q_HEADS  # skip padded head rows

                for pv_n in range_constexpr(P):
                    pv_n_global = wave_id * P + pv_n if WARP_DC_SPLIT else pv_n
                    head_col_base = pv_n_global * WMMA_M + lane_kgrp * 8
                    off_lo = out_base + row * stride_o_row + head_col_base
                    off_hi = off_lo + 4
                    lo = pv_final[qt][pv_n].shuffle(pv_final[qt][pv_n], [0, 1, 2, 3])
                    hi = pv_final[qt][pv_n].shuffle(pv_final[qt][pv_n], [4, 5, 6, 7])
                    if row_valid:
                        buffer_ops.buffer_store(lo, out_rsrc, off_lo)
                        buffer_ops.buffer_store(hi, out_rsrc, off_hi)

                off_lse = lse_base + row
                if row_valid:
                    buffer_ops.buffer_store(m_final[qt], ml_rsrc, off_lse)
                    buffer_ops.buffer_store(l_final[qt], es_rsrc, off_lse)

    cache_tag = (
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        KV_BLOCK_SIZE,
        NUM_Q_HEADS,
        NUM_SEGS,
        KV_COMPUTE_BLOCK_SIZE,
        NUM_WARPS,
        dtype,
        waves_per_eu,
        WARP_TOKEN_SPLIT,
        WARP_HEAD_SPLIT,
        WRITE_FINAL_OUTPUT,
    )

    @flyc.jit
    def launch_mla_decode_main(
        arg_out: fx.Tensor,
        arg_max_logits: fx.Tensor,
        arg_exp_sums: fx.Tensor,
        arg_query: fx.Tensor,
        arg_kv_cache: fx.Tensor,
        arg_block_tables: fx.Tensor,
        arg_seq_lens: fx.Tensor,
        i32_qk_scale: fx.Int32,
        i32_num_seqs: fx.Int32,
        i32_max_blocks_per_seq: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalized = False
            allocator.finalize()

        launcher = kernel_mla_decode_main(
            arg_out,
            arg_max_logits,
            arg_exp_sums,
            arg_query,
            arg_kv_cache,
            arg_block_tables,
            arg_seq_lens,
            i32_qk_scale,
            i32_num_seqs,
            i32_max_blocks_per_seq,
        )
        # One block per (sequence, segment). The static block dims pin
        # min == max flat_work_group_size, so the backend doesn't budget VGPRs
        # for a larger WG.
        launcher.launch(
            grid=(i32_num_seqs, NUM_SEGS, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    # Occupancy hint: target this many waves resident per execution unit.
    if waves_per_eu is not None and int(waves_per_eu) >= 1:
        launch_mla_decode_main.compile_hints = {"waves_per_eu": int(waves_per_eu)}

    return launch_mla_decode_main


# ============================================================================
# Reduce kernel
# ============================================================================


@functools.lru_cache(maxsize=1024)
def compile_mla_decode_reduce(
    *,
    KV_LORA_RANK: int = 512,
    NUM_Q_HEADS: int = 16,
    NUM_SEGS: int = 8,
    KV_COMPUTE_BLOCK_SIZE: int = 64,
    dtype: str = "bf16",
    ATTN_NUM_WARPS: int = 2,
    WARP_TOKEN_SPLIT: bool = True,
):
    """
    Merges the partials written by the main kernel into the final output.
    Grid = (num_seqs,). With WARP_TOKEN_SPLIT there are NUM_SEGS*ATTN_NUM_WARPS paritions.
    """
    SPLIT = ATTN_NUM_WARPS if WARP_TOKEN_SPLIT else 1
    NUM_PARTITIONS = NUM_SEGS * SPLIT
    V_HEAD_DIM = KV_LORA_RANK
    VEC = 4  # f32 cols per lane per chunk -> 128-bit buffer load/store
    if V_HEAD_DIM % (WAVE_SIZE * VEC) != 0:
        raise ValueError(
            f"V_HEAD_DIM ({V_HEAD_DIM}) must be a multiple of WAVE_SIZE*VEC "
            f"({WAVE_SIZE * VEC})"
        )
    # Each lane owns N_COL_CHUNKS of VEC cols
    N_COL_CHUNKS = V_HEAD_DIM // (WAVE_SIZE * VEC)

    # Use up to 4 warps, as many as evenly divide the head count; each warp reduces
    # ROWS_PER_WARP heads.
    NUM_WARPS = 1
    for w in (4, 2):
        if NUM_Q_HEADS % w == 0:
            NUM_WARPS = w
            break
    ROWS_PER_WARP = NUM_Q_HEADS // NUM_WARPS
    block_threads = NUM_WARPS * WAVE_SIZE

    f32_bytes = 4
    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    @flyc.kernel
    def kernel_mla_decode_reduce(
        arg_out: fx.Tensor,
        arg_tmp_out: fx.Tensor,
        arg_max_logits: fx.Tensor,
        arg_exp_sums: fx.Tensor,
        arg_seq_lens: fx.Tensor,
        i32_num_seqs: fx.Int32,
    ):
        elem_dtype = fx.BFloat16 if dtype == "bf16" else fx.Float16

        tx = gpu.thread_id("x")
        seq_idx = gpu.block_id("x")  # grid = (num_seqs,): one block per sequence
        warp_id = tx / WAVE_SIZE
        lane_id = tx % WAVE_SIZE

        sl_rsrc = buffer_ops.create_buffer_resource(arg_seq_lens, max_size=True)
        seq_len = fx.Index(
            buffer_ops.buffer_load(sl_rsrc, seq_idx, vec_width=1, dtype=T.i32)
        )

        # Re-derive the main kernel's tiling
        num_tiles = (seq_len + (KV_COMPUTE_BLOCK_SIZE - 1)) // KV_COMPUTE_BLOCK_SIZE
        tiles_per_seg = (num_tiles + (NUM_SEGS - 1)) // NUM_SEGS
        tiles_per_seg = (tiles_per_seg > 0).select(tiles_per_seg, fx.Index(1))
        num_segs_actual = (num_tiles + tiles_per_seg - 1) // tiles_per_seg
        num_parts_actual = num_segs_actual * SPLIT

        stride_tmp_seq = NUM_PARTITIONS * NUM_Q_HEADS * V_HEAD_DIM
        stride_tmp_seg = NUM_Q_HEADS * V_HEAD_DIM
        stride_tmp_row = V_HEAD_DIM
        stride_lse_seq = NUM_PARTITIONS * NUM_Q_HEADS
        stride_lse_seg = NUM_Q_HEADS
        stride_out_seq = NUM_Q_HEADS * V_HEAD_DIM
        stride_out_row = V_HEAD_DIM

        tmp_rsrc = buffer_ops.create_buffer_resource(arg_tmp_out, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
        ml_rsrc_v4i32 = _build_v4i32_buffer_rsrc(arg_max_logits, arch=gpu_arch)
        es_rsrc_v4i32 = _build_v4i32_buffer_rsrc(arg_exp_sums, arch=gpu_arch)

        zero_f32 = fx.Float32(0.0)
        neg_inf_f32 = fx.Float32(float("-inf"))

        warp_first_row = warp_id * ROWS_PER_WARP

        def _lane_col(c):
            # block-cyclic: lane owns VEC cols in each of N_COL_CHUNKS blocks
            return lane_id * VEC + c * (WAVE_SIZE * VEC)

        # Each warp reduces ROWS_PER_WARP head rows.
        for r_local in range_constexpr(ROWS_PER_WARP):
            r = warp_first_row + r_local
            lse_row_base = seq_idx * stride_lse_seq + r
            tmp_row_base = seq_idx * stride_tmp_seq + r * stride_tmp_row

            # Load one segment's contribution to head row r. The B023 waivers below
            # are safe: this closure is called within the same loop iteration that
            # defines it (trace time), so it never sees a later row's bases.
            def _prefetch_partition(p_idx):
                lse_off_idx = lse_row_base + p_idx * stride_lse_seg  # noqa: B023
                lse_off_sgpr = fx.Int32(
                    rocdl.readfirstlane(T.i32, fx.Int32(lse_off_idx * f32_bytes))
                )
                m_i32 = fx.Int32(_s_buffer_load_b32(ml_rsrc_v4i32, lse_off_sgpr))
                l_i32 = fx.Int32(_s_buffer_load_b32(es_rsrc_v4i32, lse_off_sgpr))
                m_f = m_i32.bitcast(fx.Float32)
                l_f = l_i32.bitcast(fx.Float32)
                v_chunks = []
                for c in range_constexpr(N_COL_CHUNKS):
                    col = _lane_col(c)
                    tmp_off = tmp_row_base + p_idx * stride_tmp_seg + col  # noqa: B023
                    v_chunks.append(
                        fx.Vector(
                            buffer_ops.buffer_load(
                                tmp_rsrc, tmp_off, vec_width=VEC, dtype=T.f32
                            )
                        )
                    )
                return m_f, l_f, v_chunks

            nonzero_n = num_parts_actual > 0
            last_p = nonzero_n.select(num_parts_actual - 1, fx.Index(0))

            def _prefetch_clamped(p_py):
                p_const = fx.Index(p_py)
                is_valid = p_const < num_parts_actual
                p_safe = is_valid.select(p_const, last_p)  # noqa: B023
                m_p, l_p, v_chunks = _prefetch_partition(p_safe)
                return m_p, l_p, v_chunks, is_valid

            # Prefetch partition 0, init the merge state (running max/sum + accumulator).
            m_p, l_p, v_p_chunks, valid = _prefetch_clamped(0)
            m_state = neg_inf_f32
            l_state = zero_f32
            acc_chunks = [
                fx.Vector.filled(VEC, 0.0, fx.Float32) for _ in range(N_COL_CHUNKS)
            ]

            # Software-pipelined merge: combine the current partition while the next one's
            # loads are already in flight.
            for p in range_constexpr(NUM_PARTITIONS):
                next_p_py = min(p + 1, NUM_PARTITIONS - 1)
                m_p_next, l_p_next, v_p_next, valid_next = _prefetch_clamped(next_p_py)

                new_m = arith.maximumf(m_state, m_p)
                alpha_old = fmath.exp2(
                    m_state - new_m, fastmath=arith.FastMathFlags.fast
                )
                alpha_this_raw = fmath.exp2(
                    m_p - new_m, fastmath=arith.FastMathFlags.fast
                )
                alpha_this = fx.Float32(valid.select(alpha_this_raw, zero_f32))
                new_l = alpha_old * l_state + alpha_this * l_p

                new_acc = []
                for c in range_constexpr(N_COL_CHUNKS):
                    new_acc.append(
                        acc_chunks[c] * alpha_old + v_p_chunks[c] * alpha_this
                    )

                m_state = new_m
                l_state = new_l
                acc_chunks = new_acc
                m_p = m_p_next
                l_p = l_p_next
                v_p_chunks = v_p_next
                valid = valid_next

            inv_l = fx.Float32(rocdl.rcp(T.f32, l_state))
            is_empty = num_parts_actual == 0
            zero_vec_half = fx.Vector.filled(VEC, 0.0, elem_dtype)

            for c in range_constexpr(N_COL_CHUNKS):
                out_vec_f32 = acc_chunks[c] * inv_l
                out_vec_half = out_vec_f32.to(elem_dtype)
                out_vec_half = fx.Vector(is_empty.select(zero_vec_half, out_vec_half))
                out_off = seq_idx * stride_out_seq + r * stride_out_row + _lane_col(c)
                buffer_ops.buffer_store(out_vec_half, out_rsrc, out_off)

    cache_tag = (
        KV_LORA_RANK,
        NUM_Q_HEADS,
        NUM_SEGS,
        KV_COMPUTE_BLOCK_SIZE,
        dtype,
        NUM_WARPS,
        ATTN_NUM_WARPS,
        WARP_TOKEN_SPLIT,
    )

    @flyc.jit
    def launch_mla_decode_reduce(
        arg_out: fx.Tensor,
        arg_tmp_out: fx.Tensor,
        arg_max_logits: fx.Tensor,
        arg_exp_sums: fx.Tensor,
        arg_seq_lens: fx.Tensor,
        i32_num_seqs: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        launcher = kernel_mla_decode_reduce(
            arg_out,
            arg_tmp_out,
            arg_max_logits,
            arg_exp_sums,
            arg_seq_lens,
            i32_num_seqs,
        )
        # One block per sequence (the reduce merges all NUM_SEGS segments
        # internally). Static block dims pin min == max flat_work_group_size.
        launcher.launch(
            grid=(i32_num_seqs, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_mla_decode_reduce


__all__ = ["compile_mla_decode_main", "compile_mla_decode_reduce"]

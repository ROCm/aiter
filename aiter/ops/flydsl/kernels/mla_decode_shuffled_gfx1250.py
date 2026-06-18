# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA (Multi-head Latent Attention) decode kernel for gfx1250 — PRE-SHUFFLED KV cache.

Computes, per (sequence, query head):
    out = Softmax(Q @ K^T * scale) @ V              (decode, query_len == 1)

KV cache:
    cache[token] = [ lora (d_c = KV_LORA_RANK) | rope (d_rope = QK_ROPE_HEAD_DIM) ]
  * QK uses the whole row   (K = lora ++ rope, width QK_HEAD_DIM = d_c + d_rope)
  * PV uses only the lora part (V = lora,       width V_HEAD_DIM = d_c)

This is the "main" of a 2-kernel split:
  The KV sequence is cut into NUM_SEGS segments; grid = (num_seqs, NUM_SEGS). Each
  (seq, seg) block runs an online-softmax pass over its slice of KV tiles and writes
  the partials:
      tmp_out    [num_seqs, NUM_SEGS, num_q_heads, d_c]
      max_logits [num_seqs, NUM_SEGS, num_q_heads]
      exp_sums   [num_seqs, NUM_SEGS, num_q_heads]
  The companion reduce kernel merges the NUM_SEGS partials.

Work distribution accross warps:
  * QK is computed REDUNDANTLY by every warp so the full tile lives in each warp's 
    registers and an LDS roundtrip can be avoided.
  * PV is SPLIT across warps along the output's N dimension (d_c)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    gpu,
    math as fmath,
    range_constexpr,
    rocdl,
    tdm_ops,
)
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.vector import ReductionOp
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


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
    i32_t = ir.IntegerType.get_signless(32)

    base_idx = buffer_ops.extract_base_index(tensor, address_space=1)
    base_i64 = _raw(arith.index_cast(T.i64, base_idx))

    # word 0: base[31:0]
    w0 = arith.trunci(i32_t, base_i64)

    # word 1: base[63:32] truncated to i32 — only base[47:32] is meaningful
    # for addresses, and stride=0 leaves the high 16 bits zero.
    shift_amt = _raw(arith.constant(32, type=T.i64))
    base_hi_i64 = arith.shrui(base_i64, shift_amt)
    w1 = arith.trunci(i32_t, base_hi_i64)

    # word 2: num_records (bytes)
    w2 = _raw(arith.constant(num_records_bytes, type=T.i32))

    # word 3: flags (data format, OOB select, etc.)
    w3 = _raw(arith.constant(buffer_ops._get_buffer_flags(arch), type=T.i32))

    return fx.Vector.from_elements([w0, w1, w2, w3], fx.Int32)


def _s_buffer_load_b32(rsrc_v4i32, byte_offset_i32):
    """Emit ``s_buffer_load_b32`` — scalar K$ load, result lands in an SGPR.

    Bypasses the VGPR → ``v_readfirstlane`` round-trip that the vmem
    ``buffer_load`` path requires, and uses the ``s_wait_kmcnt`` counter
    (separate from vmem ``s_wait_loadcnt``).

    Args:
        rsrc_v4i32: buffer descriptor as ``vector<4xi32>``
                    (from ``_build_v4i32_buffer_rsrc``).
        byte_offset_i32: byte offset (i32 SGPR value).

    Returns: i32 (scalar — uniform across the wave).
    """
    cachepol = _raw(arith.constant(0, type=T.i32))
    return _llvm.call_intrinsic(
        T.i32,
        "llvm.amdgcn.s.buffer.load.i32",
        [_raw(rsrc_v4i32), _raw(byte_offset_i32), cachepol],
        [],
        [],
    )


def _s_buffer_load_v2i32(rsrc_v4i32, byte_offset_i32):
    """Emit ``s_buffer_load_b64`` — 2-dword scalar K$ load returning vector<2xi32>.

    All lanes see the same vector; per-element extracts stay uniform.
    """
    cachepol = _raw(arith.constant(0, type=T.i32))
    return _llvm.call_intrinsic(
        ir.Type.parse("vector<2xi32>"),
        "llvm.amdgcn.s.buffer.load.v2i32",
        [_raw(rsrc_v4i32), _raw(byte_offset_i32), cachepol],
        [],
        [],
    )


def _s_buffer_load_v4i32(rsrc_v4i32, byte_offset_i32):
    """Emit ``s_buffer_load_b128`` — 4-dword scalar K$ load returning vector<4xi32>."""
    cachepol = _raw(arith.constant(0, type=T.i32))
    return _llvm.call_intrinsic(
        ir.Type.parse("vector<4xi32>"),
        "llvm.amdgcn.s.buffer.load.v4i32",
        [_raw(rsrc_v4i32), _raw(byte_offset_i32), cachepol],
        [],
        [],
    )


def _s_buffer_load_vec(rsrc_v4i32, byte_offset_i32, width):
    """Width-dispatched ``s_buffer_load`` returning a vector<widthxi32>.

    Module-level so the dispatch runs at Python trace time and the kernel's
    AST rewriter never sees the ``if/elif`` (which it would otherwise lift
    into ``scf.if`` branches and scope away the assigned value).

    Supports width ∈ {2, 4}. Width=1 should use ``_s_buffer_load_b32`` directly.
    """
    if width == 2:
        return _s_buffer_load_v2i32(rsrc_v4i32, byte_offset_i32)
    if width == 4:
        return _s_buffer_load_v4i32(rsrc_v4i32, byte_offset_i32)
    raise ValueError(
        f"_s_buffer_load_vec width must be 2 or 4 (got {width}); "
        "use _s_buffer_load_b32 for width=1."
    )


WAVE_SIZE = 32
WMMA_M = 16
WMMA_N = 16
WMMA_K = 32

# Q NOT pre-shuffled needs padding
Q_LORA_PAD = 8
Q_ROPE_PAD = 8


def _decompose_bt_widths(n):
    """Greedy-split a block count `n` into s_buffer_load-able vector widths {4,2,1}.
    """
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
):

    QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # K width for QK (lora + rope)
    V_HEAD_DIM = KV_LORA_RANK                      # V width for PV (lora only)

    # ---- tile-shape constraints (16x16x32 ) ----
    if KV_LORA_RANK % WMMA_K != 0:
        raise ValueError(f"KV_LORA_RANK (={KV_LORA_RANK}) must be a multiple of WMMA_K={WMMA_K}")
    if QK_ROPE_HEAD_DIM % WMMA_K != 0:
        raise ValueError(
            f"QK_ROPE_HEAD_DIM (={QK_ROPE_HEAD_DIM}) must be a multiple of WMMA_K={WMMA_K}"
        )
    if V_HEAD_DIM % WMMA_N != 0:
        raise ValueError(f"V_HEAD_DIM (={V_HEAD_DIM}) must be a multiple of WMMA_N={WMMA_N}")
    if KV_BLOCK_SIZE % WMMA_M != 0:
        raise ValueError(f"KV_BLOCK_SIZE must be multiple of {WMMA_M}, got {KV_BLOCK_SIZE}")
    if KV_COMPUTE_BLOCK_SIZE % WMMA_K != 0:
        raise ValueError(
            f"KV_COMPUTE_BLOCK_SIZE must be multiple of WMMA_K={WMMA_K}, got {KV_COMPUTE_BLOCK_SIZE}"
        )
    if KV_COMPUTE_BLOCK_SIZE % KV_BLOCK_SIZE != 0:
        raise ValueError(
            f"KV_COMPUTE_BLOCK_SIZE {KV_COMPUTE_BLOCK_SIZE} must be multiple of "
            f"KV_BLOCK_SIZE {KV_BLOCK_SIZE}"
        )
    N_PV_TILES = V_HEAD_DIM // WMMA_N
    if N_PV_TILES % NUM_WARPS != 0:
        raise ValueError(
            f"NUM_WARPS ({NUM_WARPS}) must divide V_HEAD_DIM/WMMA_N ({N_PV_TILES})"
        )
    if dtype not in ("bf16", "f16"):
        raise ValueError(f"dtype must be 'bf16' or 'f16', got {dtype!r}")

    # One compute tile gathers BLOCKS_PER_COMPUTE physical pages
    BLOCKS_PER_COMPUTE = KV_COMPUTE_BLOCK_SIZE // KV_BLOCK_SIZE
    BT_LOAD_WIDTHS = _decompose_bt_widths(BLOCKS_PER_COMPUTE)
    BT_LOAD_OFFSETS = [sum(BT_LOAD_WIDTHS[:i]) for i in range(len(BT_LOAD_WIDTHS))]

    # Each physical block brought in by 2 async TDM loads (one lora blob, one rope blob)
    K_OPS_PER_WAVE = 2 * BLOCKS_PER_COMPUTE

    NUM_QGSP_TILES = (NUM_Q_HEADS + WMMA_M - 1) // WMMA_M
    QGSP_PADDED = NUM_QGSP_TILES * WMMA_M

    K_QK_LORA_TILES = KV_LORA_RANK // WMMA_K
    K_QK_ROPE_TILES = QK_ROPE_HEAD_DIM // WMMA_K
    N_QK_TILES = KV_COMPUTE_BLOCK_SIZE // WMMA_N
    K_PV_TILES = KV_COMPUTE_BLOCK_SIZE // WMMA_K
    N_PV_TILES_PER_WARP = N_PV_TILES // NUM_WARPS

    block_threads = NUM_WARPS * WAVE_SIZE
    NUM_KV_STAGES = 2
    elem_bytes = 2     # bf16 / f16

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    # Shuffled-layout strides in ELEMENTS, to step one 16-token group within a
    # block's blob.
    LORA_BSG_STRIDE = (KV_LORA_RANK // WMMA_M) * 256
    ROPE_BSG_STRIDE = (QK_ROPE_HEAD_DIM // WMMA_M) * 256

    # Padded Q LDS
    Q_LORA_ROW = KV_LORA_RANK + Q_LORA_PAD
    Q_ROPE_ROW = QK_ROPE_HEAD_DIM + Q_ROPE_PAD

    lora_compute_block_elems = KV_BLOCK_SIZE * KV_LORA_RANK
    rope_compute_block_elems = KV_BLOCK_SIZE * QK_ROPE_HEAD_DIM

    # KV LDS slabs
    kv_lora_elems = BLOCKS_PER_COMPUTE * lora_compute_block_elems
    kv_rope_elems = BLOCKS_PER_COMPUTE * rope_compute_block_elems
    kv_lora_bytes = kv_lora_elems * elem_bytes
    kv_rope_bytes = kv_rope_elems * elem_bytes

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="mla_decode_shuf_lds")
    q_lora_elems = QGSP_PADDED * Q_LORA_ROW
    q_rope_elems = QGSP_PADDED * Q_ROPE_ROW

    kv_lora_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = kv_lora_off + NUM_KV_STAGES * kv_lora_bytes
    kv_rope_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = kv_rope_off + NUM_KV_STAGES * kv_rope_bytes

    # Q is read from LDS exactly once (prologue -> registers) and never touched again, so
    # its LDS bytes are dead for the whole main loop. We place Q on top of kv_lora STAGE 1
    assert (q_lora_elems + q_rope_elems) * elem_bytes <= kv_lora_bytes, (
        "Q LDS does not fit within one kv_lora stage; cannot alias Q over stage 1"
    )
    q_lora_off = kv_lora_off + kv_lora_bytes          # base of kv_lora stage 1
    q_rope_off = q_lora_off + q_lora_elems * elem_bytes
    assert q_lora_off % 16 == 0 and q_rope_off % 16 == 0, "aliased Q offsets misaligned"

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
        wave_id = tid / fx.Index(WAVE_SIZE)
        lane_id = tid % fx.Index(WAVE_SIZE)
        # Lane decomposition
        lane_kgrp = lane_id / fx.Index(WMMA_M)
        lane16 = lane_id % fx.Index(WMMA_M)

        max_blocks = fx.Index(i32_max_blocks_per_seq)

        sl_rsrc = buffer_ops.create_buffer_resource(arg_seq_lens, max_size=True)
        seq_len_i32 = buffer_ops.buffer_load(sl_rsrc, seq_idx, vec_width=1, dtype=T.i32)
        seq_len = fx.Index(seq_len_i32)

        elem_ty = T.bf16 if dtype == "bf16" else T.f16
        elem_dtype = fx.BFloat16 if dtype == "bf16" else fx.Float16
        wmma_op = (
            rocdl.wmma_f32_16x16x32_bf16
            if dtype == "bf16"
            else rocdl.wmma_f32_16x16x32_f16
        )

        # ---- this segment's tile range [tile_start, tile_end) ----
        KVC = fx.Index(KV_COMPUTE_BLOCK_SIZE)
        num_tiles = (seq_len + KVC - fx.Index(1)) / KVC
        tiles_per_seg = (num_tiles + fx.Index(NUM_SEGS) - fx.Index(1)) / fx.Index(NUM_SEGS)
        tile_start = seg_idx * tiles_per_seg
        tile_end_raw = (seg_idx + fx.Index(1)) * tiles_per_seg
        tile_end = arith.select(tile_end_raw < num_tiles, tile_end_raw, num_tiles)
        is_live = tile_start < tile_end
        iters_this_seg = arith.select(is_live, tile_end - tile_start, fx.Index(0))

        num_q_heads_idx = fx.Index(NUM_Q_HEADS)
        stride_o_seq = fx.Index(NUM_SEGS * NUM_Q_HEADS * V_HEAD_DIM)
        stride_o_seg = fx.Index(NUM_Q_HEADS * V_HEAD_DIM)
        stride_o_row = fx.Index(V_HEAD_DIM)
        stride_lse_seq = fx.Index(NUM_SEGS * NUM_Q_HEADS)
        stride_lse_seg = fx.Index(NUM_Q_HEADS)
        stride_bt_seq = max_blocks

        bt_rsrc_v4i32 = _build_v4i32_buffer_rsrc(arg_block_tables, arch=gpu_arch)
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(arg_max_logits, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(arg_exp_sums, max_size=True)

        base = allocator.get_base()
        q_lora_lds = SmemPtr(base, q_lora_off, elem_ty, shape=(q_lora_elems,))
        q_rope_lds = SmemPtr(base, q_rope_off, elem_ty, shape=(q_rope_elems,))
        kv_lora_lds = SmemPtr(base, kv_lora_off, elem_ty, shape=(NUM_KV_STAGES * kv_lora_elems,))
        kv_rope_lds = SmemPtr(base, kv_rope_off, elem_ty, shape=(NUM_KV_STAGES * kv_rope_elems,))
        q_lora_lds.get()
        q_rope_lds.get()
        kv_lora_lds.get()
        kv_rope_lds.get()

        # We multiply scale by log2e so we can use exp2 later for softmax (exp2 op faster than exp)
        qk_scale_f32 = fx.Float32(arith.bitcast(T.f32, i32_qk_scale.ir_value()))
        LOG2E = fx.Float32(1.4426950408889634)
        qk_scale_log2_scalar = qk_scale_f32 * LOG2E

        neg_inf_f32 = fx.Float32(float("-inf"))
        zero_f32 = fx.Float32(0.0)
        # Padded (unused) head rows are masked to a large FINITE negative, not -inf, so the
        # softmax of an all-masked row stays finite (-inf - -inf would be NaN).
        NEG_FINITE_MAX = -3.4e38
        neg_finite_max_vec8 = fx.Vector.filled(8, NEG_FINITE_MAX, fx.Float32)
        zero_i32 = arith.constant(0, type=T.i32)

        # Loads one 16-row x 32-K WMMA fragment for Q from Q LDS.
        def _load_q_frag(lds_ptr, row_base_idx, k_base_elem, row_stride):
            lds_mem = lds_ptr.get()
            chunks = []
            for k0 in range_constexpr(2):
                kk_base = fx.Index(k_base_elem) + (fx.Index(k0 * 2) + lane_kgrp) * fx.Index(8)
                elem_off = row_base_idx * fx.Index(row_stride) + kk_base
                chunks.append(fx.Vector.load(T.vec(8, elem_ty), lds_mem, [elem_off]))
            return chunks[0].shuffle(chunks[1], list(range(16)))

        # ---- shuffled K fragment loader (straight ds_load_b128) ----
        def _load_shuf_K(lds_ptr, n_tile, ks, bsg_stride, stage_off):
            lds_mem = lds_ptr.get()
            base_t = stage_off + fx.Index(n_tile * bsg_stride)
            o0 = base_t + fx.Index((2 * ks) * 256) + lane_id * fx.Index(8)
            o1 = base_t + fx.Index((2 * ks + 1) * 256) + lane_id * fx.Index(8)
            c0 = fx.Vector.load(T.vec(8, elem_ty), lds_mem, [o0])
            c1 = fx.Vector.load(T.vec(8, elem_ty), lds_mem, [o1])
            return c0.shuffle(c1, list(range(16)))

        # ---- shuffled V fragment loader (transpose ds_load_tr16_b128) ----
        lane8 = lane16 % fx.Index(8)
        lane_ngrp = lane16 / fx.Index(8)

        def _load_shuf_V_tr(pv_n_global, ks, stage_off):
            lds_mem = kv_lora_lds.get()
            base = (
                pv_n_global * fx.Index(256)
                + lane_ngrp * fx.Index(128)
                + (lane_kgrp * fx.Index(8) + lane8) * fx.Index(8)
            )
            o0 = stage_off + fx.Index((2 * ks) * LORA_BSG_STRIDE) + base
            o1 = stage_off + fx.Index((2 * ks + 1) * LORA_BSG_STRIDE) + base
            v0 = fx.Vector(
                rocdl.lds_transpose_load(T.vec(8, elem_ty), lds_mem, o0, elem_bytes)
            )
            v1 = fx.Vector(
                rocdl.lds_transpose_load(T.vec(8, elem_ty), lds_mem, o1, elem_bytes)
            )
            return v0.shuffle(v1, list(range(16)))

        # ---- block table -> physical page IDs ----
        KVB_idx = fx.Index(KV_BLOCK_SIZE)
        live_blocks = (seq_len + KVB_idx - fx.Index(1)) / KVB_idx  # pages this seq uses

        # Returns the BLOCKS_PER_COMPUTE physical page IDs
        # Any logical page past live_blocks is forced to page 0
        def _phys_blks_for_compute(tile_global_idx):
            base_logical = tile_global_idx * fx.Index(BLOCKS_PER_COMPUTE)
            bt_base = seq_idx * stride_bt_seq + base_logical
            out = []
            for ldi in range_constexpr(len(BT_LOAD_WIDTHS)):
                this_width = BT_LOAD_WIDTHS[ldi]
                this_offset = BT_LOAD_OFFSETS[ldi]
                bt_off = bt_base + fx.Index(this_offset)
                bt_off_bytes_i32 = arith.index_cast(T.i32, bt_off * fx.Index(4))
                if this_width == 1:
                    phys_i32 = _s_buffer_load_b32(bt_rsrc_v4i32, bt_off_bytes_i32)
                    logical_idx = base_logical + fx.Index(this_offset)
                    in_range = logical_idx < live_blocks
                    phys_i32 = arith.select(in_range, phys_i32, zero_i32)
                    out.append(fx.Index(phys_i32))
                else:
                    phys_vec = fx.Vector(
                        _s_buffer_load_vec(bt_rsrc_v4i32, bt_off_bytes_i32, this_width)
                    )
                    for b in range_constexpr(this_width):
                        elem = phys_vec[b]
                        logical_idx = base_logical + fx.Index(this_offset + b)
                        in_range = logical_idx < live_blocks
                        elem = arith.select(in_range, elem, zero_i32)
                        out.append(fx.Index(elem))
            return out

        BLOCK_STRIDE = KV_BLOCK_SIZE * QK_HEAD_DIM

        def _issue_kv_load_single_block(phys_blk, lora_byte_off, rope_byte_off):
            lora_desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_kv_cache,
                lds_memref=kv_lora_lds.get(),
                global_offset=(phys_blk, fx.Index(0)),
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
                global_offset=(phys_blk, fx.Index(lora_compute_block_elems)),
                tensor_shape=(1, rope_compute_block_elems),
                strides=(BLOCK_STRIDE, 1),
                tile_shape=(1, rope_compute_block_elems),
                elem_bytes=elem_bytes,
                num_warps=NUM_WARPS,
                lds_byte_offset=rope_byte_off,
            )
            tdm_ops.tensor_load_2d(rope_desc)

        # Issue loads for a whole compute tile: each of the BLOCKS_PER_COMPUTE
        def _issue_kv_tile_loads(phys_blks_list, lora_stage_off, rope_stage_off):
            lora_block_bytes = lora_compute_block_elems * elem_bytes
            rope_block_bytes = rope_compute_block_elems * elem_bytes
            for b in range_constexpr(BLOCKS_PER_COMPUTE):
                _issue_kv_load_single_block(
                    phys_blks_list[b],
                    lora_stage_off + fx.Index(b * lora_block_bytes),
                    rope_stage_off + fx.Index(b * rope_block_bytes),
                )

        def _issue_q_load():
            q_outer_off = seq_idx * num_q_heads_idx
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
        # Kick off Q + the first KV tile
        _issue_q_load()
        phys_blks_first = _phys_blks_for_compute(tile_start)
        _issue_kv_tile_loads(phys_blks_first, fx.Index(0), fx.Index(0))
        tdm_ops.tensor_wait(K_OPS_PER_WAVE) # only waiting for Q
        gpu.barrier()

        # Read ALL of Q's WMMA fragments into registers
        q_lora_frags = [[None] * K_QK_LORA_TILES for _ in range(NUM_QGSP_TILES)]
        q_rope_frags = [[None] * K_QK_ROPE_TILES for _ in range(NUM_QGSP_TILES)]
        for qt in range_constexpr(NUM_QGSP_TILES):
            q_row = fx.Index(qt * WMMA_M) + lane16
            for ks in range_constexpr(K_QK_LORA_TILES):
                q_lora_frags[qt][ks] = _load_q_frag(q_lora_lds, q_row, ks * WMMA_K, Q_LORA_ROW)
            for ks in range_constexpr(K_QK_ROPE_TILES):
                q_rope_frags[qt][ks] = _load_q_frag(q_rope_lds, q_row, ks * WMMA_K, Q_ROPE_ROW)

        # Drain the LDS reads (dscnt) and barrier so that iter-0's prefetch into kv_lora stage 1
        # cannot overwrite Q while any wave is still reading it.
        rocdl.s_wait_dscnt(0)
        gpu.barrier()

        def _unwrap(v):
            return v.ir_value() if hasattr(v, "ir_value") else v

        P = N_PV_TILES_PER_WARP  # d_c output tiles this warp owns

        def _pack(m_list, l_list, pv_list, cur_stage):
            out = []
            for qt in range_constexpr(NUM_QGSP_TILES):
                out.append(_unwrap(m_list[qt]))
                out.append(_unwrap(l_list[qt]))
                out.extend(pv_list[qt])
            out.append(cur_stage)
            return out

        # Initial state: m = -inf, l = 0, PV accumulators = 0, double-buffer stage = 0.
        def _init_state():
            m_list = [neg_inf_f32 for _ in range(NUM_QGSP_TILES)]
            l_list = [zero_f32 for _ in range(NUM_QGSP_TILES)]
            pv_list = [
                [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(P)]
                for _ in range(NUM_QGSP_TILES)
            ]
            return _pack(m_list, l_list, pv_list, fx.Index(0))

        init_state = _init_state()

        # ===== main loop: one KV compute tile (KVC tokens) per iteration =====
        for iv, state in range(fx.Index(0), iters_this_seg, fx.Index(1), init=init_state):
            m_list, l_list, pv_list = [], [], []
            si = 0
            for qt in range_constexpr(NUM_QGSP_TILES):
                m_list.append(fx.Float32(state[si])); si += 1
                l_list.append(fx.Float32(state[si])); si += 1
                pv_list.append([fx.Vector(v) for v in state[si : si + P]]); si += P

            cur_stage = state[si]
            nxt_stage = fx.Index(1) - cur_stage
            nxt_lora_byte_off = nxt_stage * fx.Index(kv_lora_bytes)
            nxt_rope_byte_off = nxt_stage * fx.Index(kv_rope_bytes)
            cur_lora_elem_off = cur_stage * fx.Index(kv_lora_elems)
            cur_rope_elem_off = cur_stage * fx.Index(kv_rope_elems)

            g = tile_start + iv               # global tile index in this sequence
            tile_first_tok = g * KVC
            is_not_last = iv < (iters_this_seg - fx.Index(1))

            # Prefetch the NEXT tile
            if is_not_last:
                next_phys = _phys_blks_for_compute(g + fx.Index(1))
                _issue_kv_tile_loads(next_phys, nxt_lora_byte_off, nxt_rope_byte_off)
                tdm_ops.tensor_wait(K_OPS_PER_WAVE)
            else:
                tdm_ops.tensor_wait(0)
            gpu.barrier()

            # Every warp runs QK redundantly
            qk_accs = [
                [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(N_QK_TILES)]
                for _ in range(NUM_QGSP_TILES)
            ]
            for n_tile in range_constexpr(N_QK_TILES):
                for ks in range_constexpr(K_QK_LORA_TILES):
                    k_frag = _load_shuf_K(kv_lora_lds, n_tile, ks, LORA_BSG_STRIDE, cur_lora_elem_off)
                    for qt in range_constexpr(NUM_QGSP_TILES):
                        qk_accs[qt][n_tile] = fx.Vector(
                            wmma_op(
                                T.vec(8, T.f32), k_frag, q_lora_frags[qt][ks], qk_accs[qt][n_tile],
                                signA=False, signB=False, modC=0, reuseA=False, reuseB=False,
                            ).result
                        )
                for ks in range_constexpr(K_QK_ROPE_TILES):
                    k_frag = _load_shuf_K(kv_rope_lds, n_tile, ks, ROPE_BSG_STRIDE, cur_rope_elem_off)
                    for qt in range_constexpr(NUM_QGSP_TILES):
                        qk_accs[qt][n_tile] = fx.Vector(
                            wmma_op(
                                T.vec(8, T.f32), k_frag, q_rope_frags[qt][ks], qk_accs[qt][n_tile],
                                signA=False, signB=False, modC=0, reuseA=False, reuseB=False,
                            ).result
                        )

            # ---- online-softmax update ----
            for qt in range_constexpr(NUM_QGSP_TILES):
                # Apply scale*log2e so so we can use exp2
                for n_tile in range_constexpr(N_QK_TILES):
                    qk_accs[qt][n_tile] = qk_accs[qt][n_tile] * qk_scale_log2_scalar

                # Token mask
                for n_tile in range_constexpr(N_QK_TILES):
                    new_vals = []
                    for mi in range_constexpr(8):
                        v = qk_accs[qt][n_tile][mi]
                        tok_abs = (
                            tile_first_tok + fx.Index(n_tile * WMMA_M)
                            + lane_kgrp * fx.Index(8) + fx.Index(mi)
                        )
                        in_range = tok_abs < seq_len
                        new_vals.append(arith.select(in_range, v, neg_inf_f32))
                    qk_accs[qt][n_tile] = fx.Vector.from_elements(new_vals, fx.Float32)

                # row mask when NUM_Q_HEADS isn't a multiple of 16:
                if qt * WMMA_M + WMMA_M > NUM_Q_HEADS:
                    is_row_valid = (fx.Index(qt * WMMA_M) + lane16) < num_q_heads_idx
                    for n_tile in range_constexpr(N_QK_TILES):
                        qk_accs[qt][n_tile] = fx.Vector(
                            arith.select(
                                is_row_valid, qk_accs[qt][n_tile], neg_finite_max_vec8
                            )
                        )

                # Row max
                m_state = m_list[qt]
                l_state = l_list[qt]
                local_max = qk_accs[qt][0].reduce(ReductionOp.MAX)
                for n_tile in range_constexpr(1, N_QK_TILES):
                    local_max = local_max.maximumf(
                        qk_accs[qt][n_tile].reduce(ReductionOp.MAX)
                    )
                peer = local_max.shuffle_xor(fx.Int32(16), fx.Int32(WAVE_SIZE))
                row_max = local_max.maximumf(peer)

                new_m = m_state.maximumf(row_max)
                alpha = (m_state - new_m).exp2(fastmath=arith.FastMathFlags.fast)

                # Probabilities p = exp2(score - new_m), written back IN-PLACE into qk_accs so
                # PV can read them straight from these registers. Sum the lane-local part.
                row_sum_partial = zero_f32
                for n_tile in range_constexpr(N_QK_TILES):
                    p_vec = fx.Vector(
                        fmath.exp2(
                            qk_accs[qt][n_tile] - new_m, fastmath=arith.FastMathFlags.fast
                        )
                    )
                    qk_accs[qt][n_tile] = p_vec
                    row_sum_partial = row_sum_partial + p_vec.reduce(
                        ReductionOp.ADD, fastmath=arith.FastMathFlags.fast
                    )

                # Complete the row sum across the two half-waves, then update running l and m.
                peer = row_sum_partial.shuffle_xor(fx.Int32(16), fx.Int32(WAVE_SIZE))
                row_sum = row_sum_partial + peer

                l_list[qt] = alpha * l_state + row_sum
                m_list[qt] = new_m

                # rescale pv_accs by alpha
                for pv_n in range_constexpr(P):
                    pv_list[qt][pv_n] = pv_list[qt][pv_n] * alpha

            # ---- PV ----
            # Split across warps along d_c
            for qt in range_constexpr(NUM_QGSP_TILES):
                for pv_n in range_constexpr(P):
                    pv_n_global = wave_id * fx.Index(P) + fx.Index(pv_n)
                    for ks in range_constexpr(K_PV_TILES):
                        p_f32 = qk_accs[qt][2 * ks].shuffle(
                            qk_accs[qt][2 * ks + 1], list(range(16))
                        )
                        p_frag = p_f32.to(elem_dtype)
                        v_frag = _load_shuf_V_tr(pv_n_global, ks, cur_lora_elem_off)
                        pv_list[qt][pv_n] = fx.Vector(
                            wmma_op(
                                T.vec(8, T.f32), v_frag, p_frag, pv_list[qt][pv_n],
                                signA=False, signB=False, modC=0, reuseA=False, reuseB=False,
                            ).result
                        )

            # Barrier so every warp is done reading this stage's LDS before the next
            # iteration prefetches into (and overwrites) it.
            gpu.barrier()
            results = yield _pack(m_list, l_list, pv_list, nxt_stage)

        # ---- epilogue ----
        m_final, l_final, pv_final = [], [], []
        si = 0
        for qt in range_constexpr(NUM_QGSP_TILES):
            m_final.append(fx.Float32(results[si])); si += 1
            l_final.append(fx.Float32(results[si])); si += 1
            pv_final.append([fx.Vector(v) for v in results[si : si + P]]); si += P

        out_base = seq_idx * stride_o_seq + seg_idx * stride_o_seg
        lse_base = seq_idx * stride_lse_seq + seg_idx * stride_lse_seg

        for qt in range_constexpr(NUM_QGSP_TILES):
            row = fx.Index(qt * WMMA_M) + lane16    # output row = Q head = lane16
            row_valid = row < num_q_heads_idx       # skip padded head rows

            for pv_n in range_constexpr(P):
                pv_n_global = wave_id * fx.Index(P) + fx.Index(pv_n)
                head_col_base = pv_n_global * fx.Index(WMMA_M) + lane_kgrp * fx.Index(8)
                off_lo = out_base + row * stride_o_row + head_col_base
                off_hi = off_lo + fx.Index(4)
                lo = pv_final[qt][pv_n].shuffle(pv_final[qt][pv_n], [0, 1, 2, 3])
                hi = pv_final[qt][pv_n].shuffle(pv_final[qt][pv_n], [4, 5, 6, 7])
                if row_valid:
                    buffer_ops.buffer_store(lo, out_rsrc, off_lo)
                    buffer_ops.buffer_store(hi, out_rsrc, off_hi)

            # m and l are identical across warps (QK is redundant), so only wave 0 writes
            # the log-sum-exp stats, once per head row.
            if wave_id == fx.Index(0):
                off_lse = lse_base + row
                if row_valid:
                    buffer_ops.buffer_store(m_final[qt], ml_rsrc, off_lse)
                    buffer_ops.buffer_store(l_final[qt], es_rsrc, off_lse)

    cache_tag = (
        KV_LORA_RANK, QK_ROPE_HEAD_DIM, KV_BLOCK_SIZE, NUM_Q_HEADS, NUM_SEGS,
        KV_COMPUTE_BLOCK_SIZE, NUM_WARPS, dtype, waves_per_eu,
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
            arg_out, arg_max_logits, arg_exp_sums, arg_query, arg_kv_cache,
            arg_block_tables, arg_seq_lens, i32_qk_scale, i32_num_seqs,
            i32_max_blocks_per_seq,
        )
        # Occupancy hint: target this many waves resident per execution unit.
        if waves_per_eu is not None:
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    wpe = int(waves_per_eu)
                    if wpe >= 1:
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), wpe
                        )
        # Pin min == max block size so the backend doesn't budget VGPRs for a larger WG.
        flat_wg_attr = ir.StringAttr.get(f"{block_threads},{block_threads}")
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr
        # One block per (sequence, segment).
        launcher.launch(
            grid=(i32_num_seqs, NUM_SEGS, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_mla_decode_main


# ============================================================================
# Reduce kernel
# ============================================================================


def compile_mla_decode_reduce(
    *,
    KV_LORA_RANK: int = 512,
    NUM_Q_HEADS: int = 16,
    NUM_SEGS: int = 8,
    KV_COMPUTE_BLOCK_SIZE: int = 64,
    dtype: str = "bf16",
):
    """
    Merges the NUM_SEGS partials written by the main kernel into the final output.
    Grid = (num_seqs,).
    """
    V_HEAD_DIM = KV_LORA_RANK
    VEC = 4  # f32 cols per lane per chunk -> 128-bit buffer load/store
    if V_HEAD_DIM % (WAVE_SIZE * VEC) != 0:
        raise ValueError(
            f"V_HEAD_DIM ({V_HEAD_DIM}) must be a multiple of WAVE_SIZE*VEC "
            f"({WAVE_SIZE * VEC})"
        )
    # Each lane owns VEC d_c cols in each of N_COL_CHUNKS block-cyclic stripes.
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
        seq_idx = gpu.block_id("x")          # grid = (num_seqs,): one block per sequence
        warp_id = tx / fx.Index(WAVE_SIZE)
        lane_id = tx % fx.Index(WAVE_SIZE)

        sl_rsrc = buffer_ops.create_buffer_resource(arg_seq_lens, max_size=True)
        seq_len_i32 = buffer_ops.buffer_load(sl_rsrc, seq_idx, vec_width=1, dtype=T.i32)
        seq_len = fx.Index(seq_len_i32)

        # Re-derive the main kernel's tiling to learn how many segments actually produced partials.
        # num_segs_actual = ceil(num_tiles / tiles_per_seg) = number of live partition slots.
        KVC = fx.Index(KV_COMPUTE_BLOCK_SIZE)
        num_tiles = (seq_len + KVC - fx.Index(1)) / KVC
        tiles_per_seg = (num_tiles + fx.Index(NUM_SEGS) - fx.Index(1)) / fx.Index(NUM_SEGS)
        nonzero_tps = tiles_per_seg > fx.Index(0)
        tiles_per_seg = arith.select(nonzero_tps, tiles_per_seg, fx.Index(1))
        num_segs_actual = (num_tiles + tiles_per_seg - fx.Index(1)) / tiles_per_seg

        stride_tmp_seq = fx.Index(NUM_SEGS * NUM_Q_HEADS * V_HEAD_DIM)
        stride_tmp_seg = fx.Index(NUM_Q_HEADS * V_HEAD_DIM)
        stride_tmp_row = fx.Index(V_HEAD_DIM)
        stride_lse_seq = fx.Index(NUM_SEGS * NUM_Q_HEADS)
        stride_lse_seg = fx.Index(NUM_Q_HEADS)
        stride_out_seq = fx.Index(NUM_Q_HEADS * V_HEAD_DIM)
        stride_out_row = fx.Index(V_HEAD_DIM)

        tmp_rsrc = buffer_ops.create_buffer_resource(arg_tmp_out, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
        ml_rsrc_v4i32 = _build_v4i32_buffer_rsrc(arg_max_logits, arch=gpu_arch)
        es_rsrc_v4i32 = _build_v4i32_buffer_rsrc(arg_exp_sums, arch=gpu_arch)

        zero_f32 = fx.Float32(0.0)
        neg_inf_f32 = fx.Float32(float("-inf"))
        zero_idx = fx.Index(0)
        one_idx = fx.Index(1)
        f32_bytes_idx = fx.Index(f32_bytes)

        warp_first_row = warp_id * fx.Index(ROWS_PER_WARP)

        def _lane_col(c):
            # block-cyclic: lane owns VEC cols in each of N_COL_CHUNKS blocks
            return fx.Index(c * WAVE_SIZE * VEC) + lane_id * fx.Index(VEC)

        # Each warp reduces ROWS_PER_WARP head rows.
        for r_local in range_constexpr(ROWS_PER_WARP):
            r = warp_first_row + fx.Index(r_local)
            lse_row_base = seq_idx * stride_lse_seq + r
            tmp_row_base = seq_idx * stride_tmp_seq + r * stride_tmp_row

            # Load one segment's contribution to head row r
            def _prefetch_partition(p_idx):
                lse_off_idx = lse_row_base + p_idx * stride_lse_seg
                lse_off_bytes = arith.index_cast(T.i32, lse_off_idx * f32_bytes_idx)
                lse_off_sgpr = rocdl.readfirstlane(T.i32, lse_off_bytes)
                m_i32 = _s_buffer_load_b32(ml_rsrc_v4i32, lse_off_sgpr)
                l_i32 = _s_buffer_load_b32(es_rsrc_v4i32, lse_off_sgpr)
                m_f = fx.Float32(arith.bitcast(T.f32, m_i32))
                l_f = fx.Float32(arith.bitcast(T.f32, l_i32))
                v_chunks = []
                for c in range_constexpr(N_COL_CHUNKS):
                    tmp_off = tmp_row_base + p_idx * stride_tmp_seg + _lane_col(c)
                    v_chunks.append(
                        fx.Vector(
                            buffer_ops.buffer_load(
                                tmp_rsrc, tmp_off, vec_width=VEC, dtype=T.f32
                            )
                        )
                    )
                return m_f, l_f, v_chunks

            # The merge loop is unrolled over the compile-time NUM_SEGS, but only
            # num_segs_actual segments are live. Clamp any out-of-range index to the last
            # live one (so the load is always in-bounds) and return is_valid; invalid
            # partitions get weighted 0 in the merge below.
            last_p_raw = num_segs_actual - one_idx
            nonzero_n = num_segs_actual > zero_idx
            last_p = arith.select(nonzero_n, last_p_raw, zero_idx)

            def _prefetch_clamped(p_py):
                p_const = fx.Index(p_py)
                is_valid = p_const < num_segs_actual
                p_safe = arith.select(is_valid, p_const, last_p)
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
            for p in range_constexpr(NUM_SEGS):
                next_p_py = min(p + 1, NUM_SEGS - 1)
                m_p_next, l_p_next, v_p_next, valid_next = _prefetch_clamped(next_p_py)

                new_m = m_state.maximumf(m_p)
                alpha_old = (m_state - new_m).exp2(fastmath=arith.FastMathFlags.fast)
                alpha_this_raw = (m_p - new_m).exp2(fastmath=arith.FastMathFlags.fast)
                alpha_this = fx.Float32(arith.select(valid, alpha_this_raw, zero_f32))
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

            # Normalize (numerator / total l), cast to bf16/f16, store. An empty sequence has
            # no live segments -> write zeros instead of the 0/0 = NaN the merge would give.
            inv_l = fx.Float32(rocdl.rcp(T.f32, l_state.ir_value()))
            is_empty = num_segs_actual == zero_idx
            zero_vec_half = fx.Vector.filled(VEC, 0.0, elem_dtype)

            for c in range_constexpr(N_COL_CHUNKS):
                out_vec_f32 = acc_chunks[c] * inv_l
                out_vec_half = out_vec_f32.to(elem_dtype)
                out_vec_half = fx.Vector(
                    arith.select(is_empty, zero_vec_half, out_vec_half)
                )
                out_off = seq_idx * stride_out_seq + r * stride_out_row + _lane_col(c)
                buffer_ops.buffer_store(out_vec_half, out_rsrc, out_off)

    cache_tag = (KV_LORA_RANK, NUM_Q_HEADS, NUM_SEGS, KV_COMPUTE_BLOCK_SIZE, dtype, NUM_WARPS)

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
        ctx = CompilationContext.get_current()
        launcher = kernel_mla_decode_reduce(
            arg_out,
            arg_tmp_out,
            arg_max_logits,
            arg_exp_sums,
            arg_seq_lens,
            i32_num_seqs,
        )
        # Pin min == max block size so the backend doesn't over-budget VGPRs.
        flat_wg_attr = ir.StringAttr.get(f"{block_threads},{block_threads}")
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr
        # One block per sequence (the reduce merges all NUM_SEGS segments internally).
        launcher.launch(
            grid=(i32_num_seqs, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_mla_decode_reduce


__all__ = ["compile_mla_decode_main", "compile_mla_decode_reduce"]

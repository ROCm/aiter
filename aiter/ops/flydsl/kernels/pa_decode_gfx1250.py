# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged-attention decode kernel for gfx1250 (WMMA, 1-warp).

Layout:
  query        [num_seqs, num_q_heads, HEAD_SIZE]             bf16/f16
  key_cache    [num_blocks, num_kv_heads, KV_BLOCK_SIZE, HEAD_SIZE]
  value_cache  [num_blocks, num_kv_heads, KV_BLOCK_SIZE, HEAD_SIZE]
  block_tables [num_seqs, max_num_blocks_per_seq]             i32
  seq_lens     [num_seqs]                                     i32

Per workgroup (grid = num_seqs x num_kv_heads x num_partitions):
  - 32 threads = 1 wave32 warp.
  - The single warp does ALL QK M-tiles along KVB and ALL PV M-tiles along
    HEAD. Online softmax is a within-warp shfl_xor — no cross-warp LDS scratch.
  - QK→PV layout transform is register-only: with the swapped-WMMA layout, two
    consecutive QK output vec<8>'s already form the K-decomposition the PV
    K-fragment expects (positions 0..7 from M-tile 2*ks, 8..15 from M-tile
    2*ks+1, with lane_kgrp picking the right halves). One vector.shuffle
    replaces the P-LDS round-trip.

Requirements:
  HEAD_SIZE             multiple of WMMA_N = 16
  KV_BLOCK_SIZE         multiple of 16
  KV_COMPUTE_BLOCK_SIZE multiple of WMMA_K = 32, multiple of KV_BLOCK_SIZE
  PARTITION_SIZE        multiple of KV_COMPUTE_BLOCK_SIZE
  BLOCKS_PER_COMPUTE = KV_COMPUTE_BLOCK_SIZE / KV_BLOCK_SIZE  (any positive int;
                                                              BPC > 4 issues multiple buffer_loads)
  QUERY_GROUP_SIZE      in [1, WMMA_M=16] (padded internally to WMMA_M)

WMMA instruction: wmma_f32_16x16x32_bf16 (or f16), wave32.

Each outer loop iteration processes one "compute tile" of KV_COMPUTE_BLOCK_SIZE
tokens, built by gathering BLOCKS_PER_COMPUTE physical blocks from the paged KV
cache. One vectorized block_tables load fetches all phys-block IDs per tile;
BLOCKS_PER_COMPUTE TDM ops DMA each block into its sub-slice of the K/V LDS
stage. 2-stage double buffer overlaps the next tile's loads with current compute.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, tdm_ops, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T


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

    rsrc_type = ir.Type.parse('vector<4xi32>')
    return vector.from_elements(rsrc_type, [w0, w1, w2, w3])


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
        [], [],
    )


def _s_buffer_load_v2i32(rsrc_v4i32, byte_offset_i32):
    """Emit ``s_buffer_load_b64`` — 2-dword scalar K$ load returning vector<2xi32>.

    All lanes see the same vector; per-element extracts stay uniform.
    """
    cachepol = _raw(arith.constant(0, type=T.i32))
    return _llvm.call_intrinsic(
        ir.Type.parse('vector<2xi32>'),
        "llvm.amdgcn.s.buffer.load.v2i32",
        [_raw(rsrc_v4i32), _raw(byte_offset_i32), cachepol],
        [], [],
    )


def _s_buffer_load_v4i32(rsrc_v4i32, byte_offset_i32):
    """Emit ``s_buffer_load_b128`` — 4-dword scalar K$ load returning vector<4xi32>.
    """
    cachepol = _raw(arith.constant(0, type=T.i32))
    return _llvm.call_intrinsic(
        ir.Type.parse('vector<4xi32>'),
        "llvm.amdgcn.s.buffer.load.v4i32",
        [_raw(rsrc_v4i32), _raw(byte_offset_i32), cachepol],
        [], [],
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


from flydsl.expr.numeric import Float32 as fxFloat32, Int32 as fxInt32
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

WAVE_SIZE = 32      # gfx1250 wave32
NUM_WARPS = 1       # 1 warp = 32 threads
WMMA_M = 16
WMMA_N = 16
WMMA_K = 32

LDS_PAD_ELEMS = 8   # in bf16/f16 elements (= 16 bytes; encodes as 4 dwords)


def compile_pa_decode_main(
    *,
    HEAD_SIZE: int = 128,
    KV_BLOCK_SIZE: int = 32,
    QUERY_GROUP_SIZE: int = 16,
    PARTITION_SIZE: int = 256,
    KV_COMPUTE_BLOCK_SIZE: int = 64,
    dtype: str = "bf16",
    waves_per_eu: int = 1,
):
    if HEAD_SIZE % 32 != 0:
        raise ValueError(f"HEAD_SIZE must be multiple of 32, got {HEAD_SIZE}")
    # if KV_BLOCK_SIZE % 16 != 0:
    #     raise ValueError(f"KV_BLOCK_SIZE must be multiple of 16, got {KV_BLOCK_SIZE}")
    if KV_COMPUTE_BLOCK_SIZE % WMMA_K != 0:
        raise ValueError(
            f"KV_COMPUTE_BLOCK_SIZE must be multiple of WMMA_K={WMMA_K}, got {KV_COMPUTE_BLOCK_SIZE}"
        )
    if KV_COMPUTE_BLOCK_SIZE % KV_BLOCK_SIZE != 0:
        raise ValueError(
            f"KV_COMPUTE_BLOCK_SIZE {KV_COMPUTE_BLOCK_SIZE} must be multiple of "
            f"KV_BLOCK_SIZE {KV_BLOCK_SIZE}"
        )
    if PARTITION_SIZE % KV_COMPUTE_BLOCK_SIZE != 0:
        raise ValueError(
            f"PARTITION_SIZE {PARTITION_SIZE} must be multiple of "
            f"KV_COMPUTE_BLOCK_SIZE {KV_COMPUTE_BLOCK_SIZE}"
        )
    # Real query-group sizes ∈ [1, WMMA_M] are padded internally to WMMA_M (the
    # WMMA M-tile size). QGS > WMMA_M would need an M-axis warp split — not
    # currently supported.
    if not (1 <= QUERY_GROUP_SIZE <= WMMA_M):
        raise ValueError(
            f"QUERY_GROUP_SIZE must be in [1, {WMMA_M}], got {QUERY_GROUP_SIZE}"
        )
    if dtype not in ("bf16", "f16"):
        raise ValueError(f"dtype must be 'bf16' or 'f16', got {dtype!r}")
    # PV M-tiles along HEAD axis must tile cleanly.
    if HEAD_SIZE % WMMA_N != 0:
        raise ValueError(
            f"HEAD_SIZE must be multiple of WMMA_N={WMMA_N}, got {HEAD_SIZE}"
        )

    BLOCKS_PER_COMPUTE = KV_COMPUTE_BLOCK_SIZE // KV_BLOCK_SIZE
    # Physical block IDs are loaded via buffer_load with vec_width ∈ {1, 2, 4} on AMD.
    # For BPC > 4 we issue ceil(BPC/4) loads of size ≤ 4 each.
    BT_VEC_WIDTH = 4 if BLOCKS_PER_COMPUTE >= 4 else BLOCKS_PER_COMPUTE
    BT_NUM_LOADS = (BLOCKS_PER_COMPUTE + BT_VEC_WIDTH - 1) // BT_VEC_WIDTH
    BT_TAIL_WIDTH = BLOCKS_PER_COMPUTE - (BT_NUM_LOADS - 1) * BT_VEC_WIDTH

    K_OPS_PER_WAVE = BLOCKS_PER_COMPUTE
    KV_OPS_PER_WAVE = 2 * K_OPS_PER_WAVE

    block_threads = NUM_WARPS * WAVE_SIZE          # 32
    K_QK_TILES = HEAD_SIZE // WMMA_K               # HEAD/32
    N_QK_TILES = KV_COMPUTE_BLOCK_SIZE // WMMA_N   # compute/16
    K_PV_TILES = KV_COMPUTE_BLOCK_SIZE // WMMA_K   # compute/32
    N_PV_TILES = HEAD_SIZE // WMMA_N               # HEAD/16
    COMPUTES_PER_PARTITION = PARTITION_SIZE // KV_COMPUTE_BLOCK_SIZE

    elem_bytes = 2  # bf16/f16

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    # NUM_KV_STAGES = 2 enables a software-pipelined prefetch of the next compute
    # tile's K/V into a separate LDS stage while the current tile is being
    # consumed. With COMPUTES_PER_PARTITION == 1 there's no "next tile" to
    # overlap with, so we collapse to a single stage and save half the K/V LDS.
    NUM_KV_STAGES = 2 if COMPUTES_PER_PARTITION > 1 else 1

    # Padded LDS row stride (in elements).
    KV_LDS_ROW_STRIDE = HEAD_SIZE + LDS_PAD_ELEMS

    # LDS layout (NUM_KV_STAGES double-buffer for K and V; +LDS_PAD_ELEMS bf16
    # pad per row, applied via TDM `pad_interval`/`pad_amount`):
    #   Q     [WMMA_M=16,  HEAD+LDS_PAD]               dtype  (persistent until read into q_frags)
    #   K[i]  [KV_COMPUTE_BLOCK_SIZE, HEAD+LDS_PAD]    dtype  (stage i, holds BLOCKS_PER_COMPUTE blocks)
    #   V[i]  [KV_COMPUTE_BLOCK_SIZE, HEAD+LDS_PAD]    dtype  (stage i)
    #
    # P never goes through LDS: with the swapped-WMMA layout and 1 warp, the
    # QK output's K-decomposition matches PV's K-fragment layout exactly, so
    # we feed PV WMMAs directly from the QK accumulators via vector.shuffle.
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="pa_decode_main_lds")
    # Use WMMA_M (the *padded* M-tile size) for the in-LDS Q height.
    # When the real `QUERY_GROUP_SIZE` is < WMMA_M, the unused rows in LDS
    # are simply never written (Q TDM only loads real rows; the row-mask in
    # QK + masked stores keep their data from leaking out).
    q_lds_elems  = WMMA_M                * KV_LDS_ROW_STRIDE
    kv_lds_elems = KV_COMPUTE_BLOCK_SIZE * KV_LDS_ROW_STRIDE
    q_lds_bytes  = q_lds_elems * elem_bytes
    kv_one_bytes = kv_lds_elems * elem_bytes

    # Allocate each region with a 16-byte alignment.
    # Layout: Q | K-slab (NUM_KV_STAGES contiguous stages) | V-slab (NUM_KV_STAGES contiguous stages).
    # Keeping all K stages together (and all V stages together) lets us pick a
    # stage at runtime by adding `stage * kv_one_bytes` to the slab base —
    # required for the runtime (non-unrolled) compute loop.
    q_lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = q_lds_offset + q_lds_bytes
    k_slab_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = k_slab_offset + NUM_KV_STAGES * kv_one_bytes
    v_slab_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = v_slab_offset + NUM_KV_STAGES * kv_one_bytes

    @flyc.kernel
    def kernel_pa_decode_main(
        arg_out: fx.Tensor,
        arg_max_logits: fx.Tensor,
        arg_exp_sums: fx.Tensor,
        arg_query: fx.Tensor,
        arg_key_cache: fx.Tensor,
        arg_value_cache: fx.Tensor,
        arg_block_tables: fx.Tensor,
        arg_seq_lens: fx.Tensor,
        i32_qk_scale: fx.Int32,
        i32_num_seqs: fx.Int32,
        i32_num_kv_heads: fx.Int32,
        i32_num_parts: fx.Int32,
        i32_max_blocks_per_seq: fx.Int32,
    ):
        # ---- Minimal pre-check setup ----
        # Only what's needed to evaluate the early-exit guard. Anything else
        # (other GPU IDs, lane decomposition, strides, the rest of the buffer
        # resources, LDS setup, etc.) lives inside the `if` so we don't pay
        # for it on workgroups whose partition is past seq_len.
        seq_idx = gpu.block_id("x")
        part_idx = gpu.block_id("z")

        sl_rsrc = buffer_ops.create_buffer_resource(arg_seq_lens, max_size=True)
        seq_len_i32 = buffer_ops.buffer_load(sl_rsrc, seq_idx, vec_width=1, dtype=T.i32)
        seq_len = arith.index_cast(T.index, seq_len_i32)

        # Live-partition guard. We previously wrapped the whole body in a
        # dynamic `if arith.cmpi(...)`, but FlyDSL's AST rewriter turns that
        # into a helper function whose body must execute eagerly — and a body
        # containing `yield` (from the runtime scf.for compute loop) becomes a
        # generator function that the rewriter can't dispatch. So instead we
        # compute `is_live` here and clamp the runtime loop bound to 0 for
        # past-seq_len partitions (see `iters_to_run` below). The body always
        # runs but does no useful work for dead partitions; their initial
        # m_state=-inf / l_state=0 / pv_accs=0 match the host-side pre-init,
        # and the reduce kernel skips past-seq_len partitions anyway.
        part_first_tok = part_idx * arith.index(PARTITION_SIZE)
        is_live = arith.cmpi(
            arith.CmpIPredicate.slt, _raw(part_first_tok), _raw(seq_len)
        )
        if True:
            elem_ty = T.bf16 if dtype == "bf16" else T.f16
            wmma_op = (
                rocdl.wmma_f32_16x16x32_bf16
                if dtype == "bf16"
                else rocdl.wmma_f32_16x16x32_f16
            )

            tx = gpu.thread_id("x")
            kv_head = gpu.block_id("y")

            lane_id = tx  # NUM_WARPS=1, so threadIdx.x ∈ [0, WAVE_SIZE)
            lane_kgrp = lane_id / arith.index(WMMA_M)   # 0 or 1
            lane16 = lane_id % arith.index(WMMA_M)       # 0..15

            num_kv_heads = arith.index_cast(T.index, i32_num_kv_heads.ir_value())
            num_parts = arith.index_cast(T.index, i32_num_parts.ir_value())
            max_blocks = arith.index_cast(T.index, i32_max_blocks_per_seq.ir_value())

            # --- Runtime strides ---
            num_q_heads = num_kv_heads * arith.index(QUERY_GROUP_SIZE)

            stride_bt_seq = max_blocks

            stride_o_seq = num_kv_heads * num_parts * arith.index(QUERY_GROUP_SIZE * HEAD_SIZE)
            stride_o_head = num_parts * arith.index(QUERY_GROUP_SIZE * HEAD_SIZE)
            stride_o_part = arith.index(QUERY_GROUP_SIZE * HEAD_SIZE)
            stride_o_row = arith.index(HEAD_SIZE)

            stride_lse_seq = num_kv_heads * num_parts * arith.index(QUERY_GROUP_SIZE)
            stride_lse_head = num_parts * arith.index(QUERY_GROUP_SIZE)
            stride_lse_part = arith.index(QUERY_GROUP_SIZE)

            # --- Buffer resources (sl_rsrc already created above) ---
            bt_rsrc = buffer_ops.create_buffer_resource(arg_block_tables, max_size=True)
            # Parallel <4 x i32> descriptor for s_buffer_load.i32. The vmem
            # buffer_load path keeps using `bt_rsrc` (ptr<8>); the BPC=1 path
            # uses this one to drive the scalar K$ load.
            bt_rsrc_v4i32 = _build_v4i32_buffer_rsrc(arg_block_tables, arch=gpu_arch)
            out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
            ml_rsrc = buffer_ops.create_buffer_resource(arg_max_logits, max_size=True)
            es_rsrc = buffer_ops.create_buffer_resource(arg_exp_sums, max_size=True)
            # --- LDS setup ---
            # K-slab and V-slab are each a single SmemPtr that spans NUM_KV_STAGES
            # tiles back-to-back. Stage `s` lives at byte offset
            # `s * kv_one_bytes` (= elem offset `s * kv_lds_elems`) from the
            # slab base. Runtime stage selection is just an add on the offset.
            base = allocator.get_base()
            q_lds = SmemPtr(base, q_lds_offset, elem_ty, shape=(q_lds_elems,))
            k_lds = SmemPtr(base, k_slab_offset, elem_ty,
                            shape=(NUM_KV_STAGES * kv_lds_elems,))
            v_lds = SmemPtr(base, v_slab_offset, elem_ty,
                            shape=(NUM_KV_STAGES * kv_lds_elems,))
            q_lds.get()
            k_lds.get()
            v_lds.get()

            # --- qk_scale in log2 domain (so we can use exp2) ---
            qk_scale_f32 = arith.bitcast(T.f32, _raw(i32_qk_scale))
            LOG2E = arith.constant(1.4426950408889634, type=T.f32)
            qk_scale_log2_scalar = arith.mulf(qk_scale_f32, LOG2E)
            qk_scale_log2_vec = vector.broadcast(T.vec(8, T.f32), qk_scale_log2_scalar)

            # --- Running state ---
            # SWAPPED-WMMA LAYOUT: the QK and PV WMMAs are invoked with operands
            # swapped (e.g. wmma(K, Q, …) instead of wmma(Q, K, …)) so the
            # per-lane post-WMMA layout becomes:
            #   QK output: 8 mi (= KVB-tokens within M-tile) × 1 lane16 (= QGSP-row).
            #   PV output: 8 mi (= HEAD-cols within M-tile) × 1 lane16 (= QGSP-row).
            # Consequence: each lane owns ONE QGSP-row (= lane16). m_state and
            # l_state are scalars, not vec<8>. Padded-row mask is a single
            # scalar select. Output store is contiguous-cols → vectorized.
            neg_inf_f32 = arith.constant(float("-inf"), type=T.f32)
            zero_f32 = arith.constant(0.0, type=T.f32)

            m_state = neg_inf_f32   # per-lane scalar (for row = lane16)
            l_state = zero_f32      # per-lane scalar

            # PV accumulator: N_PV_TILES vec<8xf32>, one per HEAD M-tile.
            # Per-lane = 8 mi (HEAD-cols within M-tile) × 1 lane16 (QGSP-row).
            pv_accs = [
                arith.constant_vector(0.0, T.vec(8, T.f32))
                for _ in range(N_PV_TILES)
            ]

            # lane decomposition for ds_load_tr16_b128 (B-operand transposed read)
            lane8     = lane16 % arith.index(8)
            lane_ngrp = lane16 / arith.index(8)

            # ---------------- Row mask for padded query-group rows ----------------
            # In swapped layout, QGSP-row = lane16. So is_row_valid is a single
            # scalar i1 per lane: a vec-select with this mask covers the lane's
            # entire vec<8xf32>. Use NEG_FINITE_MAX (not -inf) to avoid
            # `exp2(-inf - -inf) = NaN` for fully-masked rows.
            NEG_FINITE_MAX = -3.4e38
            neg_finite_max_f32 = arith.constant(NEG_FINITE_MAX, type=T.f32)
            neg_finite_max_vec8 = arith.constant_vector(NEG_FINITE_MAX, T.vec(8, T.f32))
            qgs_idx = arith.index(QUERY_GROUP_SIZE)
            is_row_valid = arith.cmpi(
                arith.CmpIPredicate.slt, _raw(lane16), _raw(qgs_idx)
            )

            def _load_wmma_B_frag_tr(lds_ptr, n_col_base, k_base_elem, row_stride,
                                     stage_elem_off=None):
                """Load a vec<16xelem> WMMA B fragment via 2 × ds_load_tr16_b128.

                For tensors stored in LDS as [K-rows, N-cols] row-major (K = matmul-K
                = the contraction dim, N = matmul-N), where WMMA B-operand wants
                per-lane: 1 N-col × 16 K-rows. ds_load_tr16_b128 does the K↔N
                transpose in hardware while reading row-major LDS.

                Pre-transpose addressing per lane (lane_kgrp, lane16):
                  lane8 = lane16 % 8, lane_ngrp = lane16 / 8
                  LDS row = k_base_elem + lane_kgrp*8 + lane8     (K-row this lane reads)
                  LDS col = n_col_base + lane_ngrp * 8           (8-col chunk in N)
                One transposed load delivers vec<8xelem> = (8 K-rows × 1 N-col) per lane.
                Two calls (k_half = 0, 1) cover 16 K-rows per lane = full K=32 fragment.

                `stage_elem_off` (runtime, in elements) is added to every load offset
                to pick a stage out of the V-slab.
                """
                lds_mem = lds_ptr.get()
                k_row_in_lane = lane_kgrp * arith.index(8) + lane8
                n_col_in_lane = n_col_base + lane_ngrp * arith.index(8)
                base = (
                    (arith.index(k_base_elem) + k_row_in_lane) * arith.index(row_stride)
                    + n_col_in_lane
                )
                if stage_elem_off is not None:
                    base = stage_elem_off + base
                chunks = []
                for k_half in range_constexpr(2):
                    k_row_extra = arith.index(k_half * 16) * arith.index(row_stride)
                    elem_off = base + k_row_extra
                    v = rocdl.lds_transpose_load(
                        T.vec(8, elem_ty), lds_mem, elem_off, elem_bytes
                    )
                    chunks.append(v)
                return vector.shuffle(chunks[0], chunks[1], list(range(16)))

            def _load_wmma_A_frag(lds_ptr, row_base_idx, k_base_elem, row_stride,
                                  stage_elem_off=None):
                """Load a vec<16xelem> WMMA A fragment via 2 × ds_read_b128.

                Per-lane layout: 16 elements at (row = row_base_idx,
                K-col = k_base_elem + (k0*2 + lane_kgrp)*8 + k1), k0 ∈ [0,1], k1 ∈ [0,7].

                For each k0 the 8 K-elements (k1 = 0..7) are K-contiguous in LDS, so
                one vector.load_op of vec<8xelem> covers them — lowers to a single
                ds_read_b128 (16 bytes). Two such loads + a shuffle give the full
                vec<16xelem> WMMA A-operand fragment, replacing 16 scalar ds_read_u16's.

                `stage_elem_off` (runtime, in elements) is added to every load
                offset to pick a stage out of the K-slab. Q-frag loads (which
                go to a separate `q_lds` SmemPtr that has no stages) pass None.
                """
                lds_mem = lds_ptr.get()
                chunks = []
                for k0 in range_constexpr(2):
                    kk_base = (
                        arith.index(k_base_elem)
                        + (arith.index(k0 * 2) + lane_kgrp) * arith.index(8)
                    )
                    elem_off = row_base_idx * arith.index(row_stride) + kk_base
                    if stage_elem_off is not None:
                        elem_off = stage_elem_off + elem_off
                    chunk = vector.load_op(T.vec(8, elem_ty), lds_mem, [elem_off])
                    chunks.append(chunk)
                return vector.shuffle(chunks[0], chunks[1], list(range(16)))

            # ---------------- TDM async load helpers ----------------
            # Total logical blocks in a partition (used to index block_tables).
            BLOCKS_PER_PARTITION = COMPUTES_PER_PARTITION * BLOCKS_PER_COMPUTE

            # Number of physical blocks this seq actually uses. We use this to
            # mask any block-table read whose logical index lies past the
            # seq's live block count to phys_blk = 0 (a valid physical block —
            # block 0 always exists in the cache). This is gluon's defensive
            # block-level mask: even though `bt_rsrc.max_size=True` and the
            # vLLM zero-init contract already keep us safe, this guarantees
            # the TDM never sees an undefined `phys_blk` in adversarial cases
            # (uninitialized block_tables, custom allocators where 0 isn't
            # block 0, etc.).
            KVB_idx = arith.index(KV_BLOCK_SIZE)
            live_blocks = (seq_len + KVB_idx - arith.index(1)) / KVB_idx
            zero_i32 = arith.constant(0, type=T.i32)

            def _phys_blks_for_compute(compute_iter_idx):
                """Load BLOCKS_PER_COMPUTE physical block IDs via vectorized buffer_load(s).

                For compute iteration `compute_iter_idx`, we need physical block indices at
                logical positions [base, base + BLOCKS_PER_COMPUTE) where
                base = part_idx * BLOCKS_PER_PARTITION + compute_iter_idx * BLOCKS_PER_COMPUTE.

                AMD's buffer_load vec_width is capped at 4 (128 bits for i32). For
                BPC > 4 we issue BT_NUM_LOADS loads of width BT_VEC_WIDTH (with the
                last load possibly narrower = BT_TAIL_WIDTH).

                Loaded IDs whose logical index is past `live_blocks` are masked to 0.

                Returns: list of `T.index` values, one per block in the tile.
                """
                base_logical = (
                    part_idx * arith.index(BLOCKS_PER_PARTITION)
                    + compute_iter_idx * arith.index(BLOCKS_PER_COMPUTE)
                )
                bt_base = seq_idx * stride_bt_seq + base_logical

                out = []
                for ldi in range_constexpr(BT_NUM_LOADS):
                    # Last load uses BT_TAIL_WIDTH (may be < BT_VEC_WIDTH); all others use BT_VEC_WIDTH.
                    this_width = BT_TAIL_WIDTH if ldi == BT_NUM_LOADS - 1 else BT_VEC_WIDTH
                    bt_off = bt_base + arith.index(ldi * BT_VEC_WIDTH)
                    # s_buffer_load offset is in BYTES — i32 elements ⇒ ×4.
                    bt_off_bytes_i32 = arith.index_cast(
                        T.i32, bt_off * arith.index(4)
                    )
                    if this_width == 1:
                        phys_i32 = _s_buffer_load_b32(bt_rsrc_v4i32, bt_off_bytes_i32)
                        logical_idx = base_logical + arith.index(ldi * BT_VEC_WIDTH)
                        in_range = arith.cmpi(
                            arith.CmpIPredicate.slt, _raw(logical_idx), _raw(live_blocks)
                        )
                        phys_i32 = arith.select(in_range, _raw(phys_i32), _raw(zero_i32))
                        out.append(arith.index_cast(T.index, phys_i32))
                    else:
                        # Vectorized scalar K$ load. b64 (width 2) and b128
                        # (width 4) are the hardware-native sizes. Module-level
                        # dispatch keeps the if/elif out of the rewriter's path.
                        phys_vec = _s_buffer_load_vec(
                            bt_rsrc_v4i32, bt_off_bytes_i32, this_width
                        )
                        for b in range_constexpr(this_width):
                            elem = vector.extract(
                                phys_vec, static_position=[b], dynamic_position=[]
                            )
                            logical_idx = base_logical + arith.index(ldi * BT_VEC_WIDTH + b)
                            in_range = arith.cmpi(
                                arith.CmpIPredicate.slt, _raw(logical_idx), _raw(live_blocks)
                            )
                            elem = arith.select(in_range, _raw(elem), _raw(zero_i32))
                            out.append(arith.index_cast(T.index, elem))
                return out

            def _issue_kv_load_single_block(
                kv_tensor, lds_memref, phys_blk, lds_byte_off, tdm_num_warps
            ):
                """Issue one TDM async 2D load of [KVB, HEAD] into LDS at byte offset
                `lds_byte_off` relative to the base of `lds_memref`."""
                outer_off = (phys_blk * num_kv_heads + kv_head) * arith.index(KV_BLOCK_SIZE)
                desc = tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=kv_tensor,
                    lds_memref=lds_memref,
                    global_offset=(outer_off, arith.index(0)),
                    tensor_shape=(KV_BLOCK_SIZE, HEAD_SIZE),
                    strides=(HEAD_SIZE, 1),
                    tile_shape=(KV_BLOCK_SIZE, HEAD_SIZE),
                    elem_bytes=elem_bytes,
                    pad_interval=HEAD_SIZE, pad_amount=LDS_PAD_ELEMS,
                    num_warps=tdm_num_warps,
                    lds_byte_offset=lds_byte_off,
                )
                tdm_ops.tensor_load_2d(desc)

            def _issue_kv_tile_loads(k_lds_mem, v_lds_mem, phys_blks_list,
                                     stage_byte_off):
                """Issue TDM loads for one compute tile: K loads first, then V
                loads (K-then-V issue order so tensor_wait can drain just the
                K's via FIFO while V's DMA continues in background).

                `stage_byte_off` (runtime, in bytes) is added to every TDM
                lds_byte_offset so the load lands in the right stage of the
                K-slab / V-slab.
                """
                # Use the padded row stride: TDM inserts LDS_PAD_ELEMS of pad
                # after every HEAD_SIZE source elements, so each block in LDS
                # is KV_BLOCK_SIZE rows × KV_LDS_ROW_STRIDE elements wide.
                one_block_bytes = KV_BLOCK_SIZE * KV_LDS_ROW_STRIDE * elem_bytes

                # Single-warp cooperative issue: this warp issues every block.
                for b in range_constexpr(BLOCKS_PER_COMPUTE):
                    lds_sub_off = stage_byte_off + arith.index(b * one_block_bytes)
                    _issue_kv_load_single_block(
                        arg_key_cache, k_lds_mem,
                        phys_blks_list[b], lds_sub_off, tdm_num_warps=NUM_WARPS,
                    )
                for b in range_constexpr(BLOCKS_PER_COMPUTE):
                    lds_sub_off = stage_byte_off + arith.index(b * one_block_bytes)
                    _issue_kv_load_single_block(
                        arg_value_cache, v_lds_mem,
                        phys_blks_list[b], lds_sub_off, tdm_num_warps=NUM_WARPS,
                    )

            FENCE_OUTSTANDING = KV_OPS_PER_WAVE

            def _issue_q_load():
                """TDM load Q[seq, kv_head*QG : kv_head*QG + QG, :] into q_lds.

                Query is flat 1D view of [num_seqs, num_q_heads, HEAD_SIZE]. As a 2D
                view of HEAD-strided rows, the starting row for this (seq, kv_head)
                is seq_idx*num_q_heads + kv_head*QG.
                """
                q_outer_off = (
                    seq_idx * num_q_heads
                    + kv_head * arith.index(QUERY_GROUP_SIZE)
                )
                desc = tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_query,
                    lds_memref=q_lds.get(),
                    global_offset=(q_outer_off, arith.index(0)),
                    tensor_shape=(QUERY_GROUP_SIZE, HEAD_SIZE),
                    strides=(HEAD_SIZE, 1),
                    tile_shape=(QUERY_GROUP_SIZE, HEAD_SIZE),
                    elem_bytes=elem_bytes,
                    pad_interval=HEAD_SIZE, pad_amount=LDS_PAD_ELEMS,
                    num_warps=NUM_WARPS,
                )
                tdm_ops.tensor_load_2d(desc)

            # ---------------- Prologue: Q + compute tile 0 loads (concurrent) ----------------
            _issue_q_load()
            phys_blks_tile0 = _phys_blks_for_compute(arith.index(0)) # load physical blocks for compute iter 1
            _issue_kv_tile_loads(
                k_lds.get(), v_lds.get(), phys_blks_tile0,
                stage_byte_off=arith.index(0),  # tile 0 lands in stage 0
            )
            tdm_ops.tensor_wait(FENCE_OUTSTANDING)
            gpu.barrier()

            # Pre-load Q fragments from LDS (held in registers for the entire kernel).
            q_frags = []
            for ks in range_constexpr(K_QK_TILES):  # each warp has a complete copy of Q
                q_frags.append(_load_wmma_A_frag(q_lds, lane16, ks * WMMA_K, KV_LDS_ROW_STRIDE))

            # Pre-extract LDS memrefs once, outside the runtime loop. Keeping
            # `k_lds.get()` / `v_lds.get()` calls inside scf.if branches makes
            # FlyDSL's AST rewriter treat `k_lds` / `v_lds` as state variables
            # crossing the if (because of the visit_Call → invoked_args path)
            # and then fail with "state variable is SmemPtr, not an MLIR Value".
            k_lds_mem = k_lds.get()
            v_lds_mem = v_lds.get()

            # ---------------- Main loop over compute tiles in partition ----------------
            # Runtime scf.for (no constexpr unroll). Loop-carried state:
            #   [m_state, l_state, *pv_accs (length N_PV_TILES), cur_stage]
            # cur_stage toggles 0↔1 each iter and is used at runtime to pick the
            # K/V slab byte-offset (since the LDS layout is K-slab|V-slab with
            # both stages contiguous within each slab).
            #
            # iters_to_run replaces the old early-exit guard: live partitions
            # run COMPUTES_PER_PARTITION iters, dead partitions run 0 (and the
            # init_state values flow straight to the stores).
            iters_to_run = arith.select(
                is_live,
                _raw(arith.index(COMPUTES_PER_PARTITION)),
                _raw(arith.index(0)),
            )
            init_state = [m_state, l_state, *pv_accs, arith.index(0)]

            for iv, state in range(
                arith.index(0), iters_to_run, arith.index(1), init=init_state
            ):
                # Unpack loop-carried state.
                m_state = state[0]
                l_state = state[1]
                pv_accs = list(state[2:2 + N_PV_TILES])
                cur_stage = state[2 + N_PV_TILES]
                nxt_stage = arith.index(1) - cur_stage

                # Per-iter runtime stage offsets.
                cur_stage_byte_off = cur_stage * arith.index(kv_one_bytes)
                nxt_stage_byte_off = nxt_stage * arith.index(kv_one_bytes)
                cur_stage_elem_off = cur_stage * arith.index(kv_lds_elems)

                tile_first_tok = (
                    part_idx * arith.index(PARTITION_SIZE)
                    + iv * arith.index(KV_COMPUTE_BLOCK_SIZE)
                )

                # is_not_last → (iv < COMPUTES_PER_PARTITION - 1)
                is_not_last = arith.cmpi(
                    arith.CmpIPredicate.slt,
                    _raw(iv),
                    _raw(arith.index(COMPUTES_PER_PARTITION - 1)),
                )

                # ---------- scf.if #1: prefetch next tile + drain current K ----------
                # Both branches must use a constexpr literal for tensor_wait, so
                # the last-vs-not-last split has to live inside scf.if branches.
                # Note: we pass the pre-extracted `k_lds_mem` / `v_lds_mem`
                # rather than calling `.get()` inside the branch, see comment
                # at their definition.
                if is_not_last:
                    next_iter_idx = iv + arith.index(1)
                    next_phys_blks = _phys_blks_for_compute(next_iter_idx)
                    _issue_kv_tile_loads(
                        k_lds_mem, v_lds_mem, next_phys_blks,
                        stage_byte_off=nxt_stage_byte_off,
                    )
                    # In flight before issue: 2*K_OPS (cur tile's K+V).
                    # +2*K_OPS prefetched = 4*K_OPS total. Drain 1*K_OPS (= cur K).
                    tdm_ops.tensor_wait(3 * K_OPS_PER_WAVE)
                else:
                    # In flight: 2*K_OPS (cur K+V, prefetched in prev iter or prologue).
                    # No new prefetch. Drain 1*K_OPS (= cur K), leave V in flight.
                    tdm_ops.tensor_wait(K_OPS_PER_WAVE)
                gpu.barrier()

                # ------------------------ QK WMMA (SWAPPED operands) ------------------------
                # `wmma(K, Q, …)` instead of `wmma(Q, K, …)` → output is S^T in
                # WMMA C-layout. Per-lane post-WMMA: 8 mi (= KVB-tokens within
                # M-tile) × 1 lane16 (= QGSP-row).
                # The single warp does all N_QK_TILES M-tiles along KVB.
                qk_accs = [
                    arith.constant_vector(0.0, T.vec(8, T.f32))
                    for _ in range(N_QK_TILES)
                ]
                for n_tile in range_constexpr(N_QK_TILES):
                    n_row_lds = arith.index(n_tile * WMMA_M) + lane16
                    for ks in range_constexpr(K_QK_TILES):
                        k_frag = _load_wmma_A_frag(
                            k_lds, n_row_lds, ks * WMMA_K, KV_LDS_ROW_STRIDE,
                            stage_elem_off=cur_stage_elem_off,
                        )
                        # SWAPPED: K is A operand (M=KVB), Q is B operand (N=QGSP).
                        qk_accs[n_tile] = wmma_op(
                            T.vec(8, T.f32),
                            k_frag,
                            q_frags[ks],
                            qk_accs[n_tile],
                            signA=False,
                            signB=False,
                            modC=0,
                            reuseA=False,
                            reuseB=False,
                        ).result

                # Scale by qk_scale * log2e so we can use exp2 later.
                for n_tile in range_constexpr(N_QK_TILES):
                    qk_accs[n_tile] = arith.mulf(qk_accs[n_tile], qk_scale_log2_vec)

                # Col mask (token < seq_len). In swapped layout, KVB-token varies
                # per mi (not per lane16). Per-element mask:
                #   tok_abs[mi] = tile_first_tok + n_tile*WMMA_M + lane_kgrp*8 + mi
                neg_inf_f32_local = arith.constant(float("-inf"), type=T.f32)
                for n_tile in range_constexpr(N_QK_TILES):
                    new_vals = []
                    for mi in range_constexpr(8):
                        v = vector.extract(
                            qk_accs[n_tile], static_position=[mi], dynamic_position=[]
                        )
                        tok_abs_mi = (
                            tile_first_tok
                            + arith.index(n_tile * WMMA_M)
                            + lane_kgrp * arith.index(8)
                            + arith.index(mi)
                        )
                        in_range = arith.cmpi(
                            arith.CmpIPredicate.slt, _raw(tok_abs_mi), _raw(seq_len)
                        )
                        v = arith.select(in_range, _raw(v), _raw(neg_inf_f32_local))
                        new_vals.append(v)
                    qk_accs[n_tile] = vector.from_elements(T.vec(8, T.f32), new_vals)

                # Row mask for QGSP padding (when QGS_real < WMMA_M). In swapped
                # layout: row = lane16 (single per lane). One vec-wide select.
                if QUERY_GROUP_SIZE < WMMA_M:
                    for n_tile in range_constexpr(N_QK_TILES):
                        qk_accs[n_tile] = arith.select(
                            is_row_valid,
                            _raw(qk_accs[n_tile]),
                            _raw(neg_finite_max_vec8),
                        )

                # ------------------------ Online softmax (1-warp) ------------------------
                # Per-lane = 8 KVB-tokens × 1 QGSP-row. Reduction is along KVB.
                # m_state, l_state, alpha are SCALAR per lane (lane16 = row).
                # No cross-warp scratch — reduction collapses to local + shfl_xor.

                # Step 1: per-lane local max across all (n_tile, mi) → scalar.
                local_max = vector.extract(
                    qk_accs[0], static_position=[0], dynamic_position=[]
                )
                for mi in range_constexpr(1, 8):
                    v = vector.extract(
                        qk_accs[0], static_position=[mi], dynamic_position=[]
                    )
                    local_max = arith.maximumf(local_max, v)
                if N_QK_TILES > 1:
                    for n_tile in range_constexpr(1, N_QK_TILES):
                        for mi in range_constexpr(8):
                            v = vector.extract(
                                qk_accs[n_tile], static_position=[mi], dynamic_position=[]
                            )
                            local_max = arith.maximumf(local_max, v)

                # Step 2: within-warp max via shfl_xor(by 16) — swaps lane_kgrp halves.
                # After shfl, both lanes (kgrp=0/1, lane16=R) have the row-max for row R.
                x = fxFloat32(local_max)
                peer = x.shuffle_xor(fxInt32(16), fxInt32(WAVE_SIZE))
                row_max = x.maximumf(peer).ir_value()

                # Step 3: update m_state, compute alpha (scalars).
                new_m = arith.maximumf(m_state, row_max)
                alpha = rocdl.exp2(T.f32, arith.subf(m_state, new_m))

                # Step 4: compute p = exp2(qk - new_m) IN-PLACE into qk_accs (now
                # they hold p), and accumulate row_sum (scalar). Keeping P in the
                # qk_accs registers avoids a separate p_accs allocation; the PV
                # WMMA reads from these same registers via vector.shuffle.
                row_sum_partial = zero_f32
                for n_tile in range_constexpr(N_QK_TILES):
                    vals_new = []
                    for mi in range_constexpr(8):
                        q_ij = vector.extract(
                            qk_accs[n_tile], static_position=[mi], dynamic_position=[]
                        )
                        p_ij = rocdl.exp2(T.f32, arith.subf(q_ij, new_m))
                        vals_new.append(p_ij)
                        row_sum_partial = arith.addf(row_sum_partial, p_ij)
                    qk_accs[n_tile] = vector.from_elements(T.vec(8, T.f32), vals_new)

                # Step 5: within-warp sum via shfl_xor(by 16).
                x = fxFloat32(row_sum_partial)
                peer = x.shuffle_xor(fxInt32(16), fxInt32(WAVE_SIZE))
                row_sum = (x + peer).ir_value()

                # Step 6: update l_state and m_state (scalars).
                l_state = arith.addf(arith.mulf(alpha, l_state), row_sum)
                m_state = new_m

                # Step 7: rescale pv_accs by alpha (vec<8> *= scalar via broadcast).
                alpha_vec = vector.broadcast(T.vec(8, T.f32), alpha)
                for pv_n in range_constexpr(N_PV_TILES):
                    pv_accs[pv_n] = arith.mulf(pv_accs[pv_n], alpha_vec)

                # ---------- scf.if #2: drain current V ----------
                # Same is_not_last condition, different constexpr wait counts.
                if is_not_last:
                    # 3*K_OPS in flight (V_cur + K_next + V_next). Drain V_cur.
                    tdm_ops.tensor_wait(2 * K_OPS_PER_WAVE)
                else:
                    # K_OPS in flight (just V_cur). Drain everything.
                    tdm_ops.tensor_wait(0)
                gpu.barrier()

                # ------------------------ PV WMMA (SWAPPED operands) ------------------------
                # `wmma(V, P, …)` → output is O^T in WMMA C-layout. Per-lane
                # post-WMMA: 8 mi (= HEAD-cols within M-tile) × 1 lane16
                # (= QGSP-row). M=HEAD, N=QGSP, K=KVB.
                #
                # P-fragment is built register-only from qk_accs. Per PV K-tile
                # ks (K=WMMA_K=32 KVB tokens), the B-frag wants 16 K-elements
                # per lane decomposed as positions[0..7]={K[0..7] for kgrp=0,
                # K[8..15] for kgrp=1} and positions[8..15]={K[16..23] for kgrp=0,
                # K[24..31] for kgrp=1}. Each lane's qk_accs[2*ks] holds 8 KVB
                # tokens of QK M-tile 2*ks at this lane's kgrp-half (= positions
                # 0..7 of the desired layout); qk_accs[2*ks+1] holds the next 8
                # at the same kgrp-half (= positions 8..15). So a plain concat
                # via vector.shuffle followed by trunc_f produces the bf16/f16
                # B-fragment with no LDS round-trip and no cross-lane shuffles.
                for pv_n in range_constexpr(N_PV_TILES):
                    for ks in range_constexpr(K_PV_TILES):
                        p_f32 = vector.shuffle(
                            qk_accs[2 * ks], qk_accs[2 * ks + 1], list(range(16))
                        )
                        p_frag = arith.trunc_f(T.vec(16, elem_ty), p_f32)
                        v_frag = _load_wmma_B_frag_tr(
                            v_lds,
                            arith.index(pv_n * WMMA_M),  # base in HEAD dim
                            ks * WMMA_K,                 # base in KVB dim
                            KV_LDS_ROW_STRIDE,           # row stride (V LDS [KVB, HEAD+pad])
                            stage_elem_off=cur_stage_elem_off,
                        )
                        # SWAPPED: V is A (M=HEAD), P is B (N=QGSP).
                        pv_accs[pv_n] = wmma_op(
                            T.vec(8, T.f32),
                            v_frag,
                            p_frag,
                            pv_accs[pv_n],
                            signA=False,
                            signB=False,
                            modC=0,
                            reuseA=False,
                            reuseB=False,
                        ).result

                gpu.barrier()

                # Yield updated state for next iter. cur_stage toggles to nxt_stage.
                results = yield [m_state, l_state, *pv_accs, nxt_stage]

            # After the loop: pull final state out of `results` (last yielded values).
            m_state = results[0]
            l_state = results[1]
            pv_accs = list(results[2:2 + N_PV_TILES])

            # ---------------- Write per-partition results ----------------
            out_base = (
                seq_idx * stride_o_seq
                + kv_head * stride_o_head
                + part_idx * stride_o_part
            )
            lse_base = (
                seq_idx * stride_lse_seq
                + kv_head * stride_lse_head
                + part_idx * stride_lse_part
            )

            # Accumulator: the warp writes all N_PV_TILES × vec<8xf32>.
            # Swapped layout: per-lane = 8 mi (HEAD-cols within M-tile) × 1
            # lane16 (QGSP-row). HEAD-cols are stride-1 in the row-major output
            # buffer → vector stores. RDNA4's largest buffer_store is b128 (16
            # bytes / vec<4xf32>), so split each vec<8xf32> into two vec<4>
            # stores at consecutive offsets.
            # Predicated by is_row_valid (= lane16 < QGSP_real) for QGS-padding.
            for pv_n in range_constexpr(N_PV_TILES):
                head_col_base = (
                    arith.index(pv_n * WMMA_M)
                    + lane_kgrp * arith.index(8)
                )
                off_lo = out_base + lane16 * stride_o_row + head_col_base
                off_hi = off_lo + arith.index(4)
                lo = vector.shuffle(pv_accs[pv_n], pv_accs[pv_n], [0, 1, 2, 3])
                hi = vector.shuffle(pv_accs[pv_n], pv_accs[pv_n], [4, 5, 6, 7])
                # cmpi inlined into `if` test (FlyDSL's AST rewriter only
                # converts to scf.if when the test contains a Call).
                if arith.cmpi(
                    arith.CmpIPredicate.slt, _raw(lane16), _raw(qgs_idx)
                ):
                    buffer_ops.buffer_store(lo, out_rsrc, off_lo)
                    buffer_ops.buffer_store(hi, out_rsrc, off_hi)

            # max_logits / exp_sums: per-lane SCALAR (one row's m/l for row=lane16).
            # Two lanes per row (kgrp=0/1) hold the same value (after shfl_xor in
            # softmax) → benign-race redundant writes. Predicated by is_row_valid.
            off_lse = lse_base + lane16
            if arith.cmpi(
                arith.CmpIPredicate.slt, _raw(lane16), _raw(qgs_idx)
            ):
                buffer_ops.buffer_store(m_state, ml_rsrc, off_lse)
                buffer_ops.buffer_store(l_state, es_rsrc, off_lse)

    cache_tag = (HEAD_SIZE, KV_BLOCK_SIZE, QUERY_GROUP_SIZE, PARTITION_SIZE,
                 KV_COMPUTE_BLOCK_SIZE, dtype, NUM_WARPS, waves_per_eu)

    @flyc.jit
    def launch_pa_decode_main(
        arg_out: fx.Tensor,
        arg_max_logits: fx.Tensor,
        arg_exp_sums: fx.Tensor,
        arg_query: fx.Tensor,
        arg_key_cache: fx.Tensor,
        arg_value_cache: fx.Tensor,
        arg_block_tables: fx.Tensor,
        arg_seq_lens: fx.Tensor,
        i32_qk_scale: fx.Int32,
        i32_num_seqs: fx.Int32,
        i32_num_kv_heads: fx.Int32,
        i32_num_parts: fx.Int32,
        i32_max_blocks_per_seq: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalized = False
            allocator.finalize()

        gx = _raw(arith.index_cast(T.index, i32_num_seqs.ir_value()))
        gy = _raw(arith.index_cast(T.index, i32_num_kv_heads.ir_value()))
        gz = _raw(arith.index_cast(T.index, i32_num_parts.ir_value()))

        # Trace the kernel into the gpu module first so we can attach the
        # `rocdl.waves_per_eu` attribute to its gpu.func before launching.
        # Capping waves_per_eu lets the LLVM backend allocate more VGPRs per
        # wave (avoiding spills on register-heavy configs like HEAD=128 +
        # KVB>=128 where qk_accs + pv_accs together can exceed 200 VGPRs).
        launcher = kernel_pa_decode_main(
            arg_out, arg_max_logits, arg_exp_sums,
            arg_query, arg_key_cache, arg_value_cache,
            arg_block_tables, arg_seq_lens,
            i32_qk_scale, i32_num_seqs, i32_num_kv_heads,
            i32_num_parts, i32_max_blocks_per_seq,
        )

        if waves_per_eu is not None:
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    wpe = int(waves_per_eu)
                    if wpe >= 1:
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), wpe
                        )

        # Pin min == max == block_threads so the LLVM backend knows the
        # workgroup is exactly this size (no need to budget registers for a
        # larger upper bound). Launch always uses block=(block_threads,1,1).
        flat_wg_attr = ir.StringAttr.get(f"{block_threads},{block_threads}")
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

        launcher.launch(
            grid=(gx, gy, gz),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_pa_decode_main


# ============================================================================
# Reduce kernel (unchanged; 1 warp, scalar)
# ============================================================================

def compile_pa_decode_reduce(
    *,
    HEAD_SIZE: int = 128,
    QUERY_GROUP_SIZE: int = 16,
    PARTITION_SIZE: int = 256,
    dtype: str = "bf16",
):
    if HEAD_SIZE % 16 != 0:
        raise ValueError("HEAD_SIZE must be multiple of 16")
    if dtype not in ("bf16", "f16"):
        raise ValueError(f"dtype must be 'bf16' or 'f16', got {dtype!r}")
    if HEAD_SIZE % WAVE_SIZE != 0:
        raise ValueError(f"HEAD_SIZE must be multiple of {WAVE_SIZE}")

    block_threads = WAVE_SIZE
    COLS_PER_THREAD = HEAD_SIZE // WAVE_SIZE

    allocator = SmemAllocator(None, arch="gfx1250", global_sym_name="pa_decode_red_lds")
    allocator.ptr = 4

    @flyc.kernel
    def kernel_pa_decode_reduce(
        arg_out: fx.Tensor,
        arg_tmp_out: fx.Tensor,
        arg_max_logits: fx.Tensor,
        arg_exp_sums: fx.Tensor,
        arg_seq_lens: fx.Tensor,
        i32_num_seqs: fx.Int32,
        i32_num_kv_heads: fx.Int32,
        i32_num_parts: fx.Int32,
    ):
        elem_ty = T.bf16 if dtype == "bf16" else T.f16

        tx = gpu.thread_id("x")
        seq_idx = gpu.block_id("x")
        kv_head = gpu.block_id("y")

        num_kv_heads = arith.index_cast(T.index, i32_num_kv_heads.ir_value())
        num_parts = arith.index_cast(T.index, i32_num_parts.ir_value())
        num_q_heads = num_kv_heads * arith.index(QUERY_GROUP_SIZE)

        sl_rsrc = buffer_ops.create_buffer_resource(arg_seq_lens, max_size=True)
        seq_len_i32 = buffer_ops.buffer_load(sl_rsrc, seq_idx, vec_width=1, dtype=T.i32)
        seq_len = arith.index_cast(T.index, seq_len_i32)
        part_size_idx = arith.index(PARTITION_SIZE)
        num_parts_actual = (seq_len + part_size_idx - arith.index(1)) / part_size_idx

        stride_tmp_seq = num_kv_heads * num_parts * arith.index(QUERY_GROUP_SIZE * HEAD_SIZE)
        stride_tmp_head = num_parts * arith.index(QUERY_GROUP_SIZE * HEAD_SIZE)
        stride_tmp_part = arith.index(QUERY_GROUP_SIZE * HEAD_SIZE)
        stride_tmp_row = arith.index(HEAD_SIZE)

        stride_lse_seq = num_kv_heads * num_parts * arith.index(QUERY_GROUP_SIZE)
        stride_lse_head = num_parts * arith.index(QUERY_GROUP_SIZE)
        stride_lse_part = arith.index(QUERY_GROUP_SIZE)

        stride_out_seq = num_q_heads * arith.index(HEAD_SIZE)
        stride_out_head = arith.index(HEAD_SIZE)

        tmp_rsrc = buffer_ops.create_buffer_resource(arg_tmp_out, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(arg_max_logits, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(arg_exp_sums, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)

        zero_f32 = arith.constant(0.0, type=T.f32)
        neg_inf_f32 = arith.constant(float("-inf"), type=T.f32)
        zero_idx = arith.index(0)
        one_idx = arith.index(1)

        for r in range_constexpr(QUERY_GROUP_SIZE):
            lse_row_base = (
                seq_idx * stride_lse_seq
                + kv_head * stride_lse_head
                + arith.index(r)
            )
            tmp_row_base = (
                seq_idx * stride_tmp_seq
                + kv_head * stride_tmp_head
                + arith.index(r) * stride_tmp_row
            )

            # Pass 1: global max + online exp_sum
            init_state = [neg_inf_f32, zero_f32]
            for p, state in range(zero_idx, num_parts_actual, one_idx, init=init_state):
                old_m = state[0]
                old_l = state[1]
                lse_off = lse_row_base + p * stride_lse_part
                this_m = buffer_ops.buffer_load(ml_rsrc, lse_off, vec_width=1, dtype=T.f32)
                this_l = buffer_ops.buffer_load(es_rsrc, lse_off, vec_width=1, dtype=T.f32)
                new_m = arith.maximumf(old_m, this_m)
                alpha_old = rocdl.exp2(T.f32, arith.subf(old_m, new_m))
                alpha_this = rocdl.exp2(T.f32, arith.subf(this_m, new_m))
                new_l = arith.addf(
                    arith.mulf(alpha_old, old_l), arith.mulf(alpha_this, this_l)
                )
                results1 = yield [new_m, new_l]
            g_max = results1[0]
            g_exp_sum = results1[1]

            # Pass 2: weighted sum
            for c in range_constexpr(COLS_PER_THREAD):
                col = tx + arith.index(c * WAVE_SIZE)

                init_acc = [zero_f32, zero_f32]
                for p, state in range(zero_idx, num_parts_actual, one_idx, init=init_acc):
                    old_acc = state[0]
                    lse_off = lse_row_base + p * stride_lse_part
                    this_m = buffer_ops.buffer_load(ml_rsrc, lse_off, vec_width=1, dtype=T.f32)
                    tmp_off = tmp_row_base + p * stride_tmp_part + col
                    this_v = buffer_ops.buffer_load(tmp_rsrc, tmp_off, vec_width=1, dtype=T.f32)
                    alpha = rocdl.exp2(T.f32, arith.subf(this_m, g_max))
                    new_acc = arith.addf(old_acc, arith.mulf(alpha, this_v))
                    results2 = yield [new_acc, state[1]]
                final_acc = arith.divf(results2[0], g_exp_sum)

                out_off = (
                    seq_idx * stride_out_seq
                    + (kv_head * arith.index(QUERY_GROUP_SIZE) + arith.index(r)) * stride_out_head
                    + col
                )
                final_half = arith.trunc_f(elem_ty, final_acc)
                final_i16 = arith.bitcast(T.i16, final_half)
                buffer_ops.buffer_store(final_i16, out_rsrc, out_off)

    cache_tag = (HEAD_SIZE, QUERY_GROUP_SIZE, PARTITION_SIZE, dtype)

    @flyc.jit
    def launch_pa_decode_reduce(
        arg_out: fx.Tensor,
        arg_tmp_out: fx.Tensor,
        arg_max_logits: fx.Tensor,
        arg_exp_sums: fx.Tensor,
        arg_seq_lens: fx.Tensor,
        i32_num_seqs: fx.Int32,
        i32_num_kv_heads: fx.Int32,
        i32_num_parts: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalized = False
            allocator.finalize()

        gx = _raw(arith.index_cast(T.index, i32_num_seqs.ir_value()))
        gy = _raw(arith.index_cast(T.index, i32_num_kv_heads.ir_value()))

        kernel_pa_decode_reduce(
            arg_out, arg_tmp_out, arg_max_logits, arg_exp_sums,
            arg_seq_lens, i32_num_seqs, i32_num_kv_heads, i32_num_parts,
        ).launch(
            grid=(gx, gy, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_pa_decode_reduce


__all__ = ["compile_pa_decode_main", "compile_pa_decode_reduce"]

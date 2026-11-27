# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Flash Attention varlen forward for gfx11 (RDNA3 / RDNA3.5), bf16/f16.

The packed-THD (``[total, heads, head_dim]`` + ``cu_seqlens``) counterpart of
``flash_attn_func_gfx11.py``, with the same core: computed transposed so that
every lane owns one query row,

  S^T[key, q] = K[key, :] . Q[q, :]          (A = K from LDS, B = Q in VGPRs)
  O^T[d, q]  += V^T[d, key] . P^T[key, q]    (A = V^T from LDS, B = P^T in VGPRs)

gfx11 WMMA takes v16 A/B operands (lanes 16-31 replicate lanes 0-15) and puts
lane L's accumulator element ``si`` at D[2*si + L/16][L%16]. The query is
therefore L%16 in both lane halves: the row max needs one xor-16 exchange and
the row sum stays per lane until the epilogue.

Sequence ``b`` is the token range ``cu_seqlens[b]`` to ``cu_seqlens[b + 1]``,
read from the two tables at kernel start with scalar loads (the lengths stay in
SGPRs, so loop bounds and descriptors are uniform). Every sequence gets its own
bounds-checked buffer descriptors based at its first token, so loads past its
end return zero and stores past it are dropped. The grid covers the longest
sequence; a query block past a shorter sequence's end runs no KV tile and its
stores fall outside its descriptor. Keys at or past a sequence's end are masked
to -inf, on the last KV tile only (the tiles before it are full). A sequence
without keys gets O = 0 and LSE = -inf. LSE is ``[heads, total_q]`` as in
flash-attn's varlen interface. ``head_dim`` only needs to be a multiple of 8: it
is zero-padded to the WMMA K of 16 inside the kernel.

``rope`` builds the vision-encoder form: Q and K come straight from the packed
qkv projection and are rotated (NeoX RoPE over the whole head) in the kernel,
Q in registers and K while it is staged into LDS, so no separate rotary pass
writes and re-reads them.
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from . import buffer_ops
from .flash_attn_func_gfx11 import (
    LDS_LIMIT_BYTES,
    LDS_PAD,
    WAVE_SIZE,
    WMMA_MNK,
    lds_bytes,
)
from .kernels_common import LOG2E as _LOG2E
from .tensor_shim import (
    _run_compiled,
    buf_base_i64,
    buf_copy_load,
    buf_copy_store,
    ptr_arg,
    ptr_buf_tensor,
)

KERNEL_NAME = "flash_attn_varlen_func_gfx11_kernel"

NEG_BIG = -1.0e30  # finite: NEG_BIG - m stays finite, exp2 of it is exactly 0
POS_BIG = 1.0e30  # initial m of a causal row with no visible key: all its p are 0
_LN2 = host_math.log(2.0)


def build_flash_attn_varlen_func_module_gfx11(
    num_heads,
    head_dim,
    causal=False,
    dtype_str="bf16",
    sm_scale=None,
    num_waves=16,
    block_n=32,
    vt_rows=2,
    double_buffer=True,
    head_group=None,
    waves_per_eu=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    q_token_stride=None,
    k_token_stride=None,
    v_token_stride=None,
    num_kv_heads=None,
    return_lse=False,
    rope=None,
):
    """Build the gfx11 Flash Attention varlen forward kernel.

    Args:
        num_heads, head_dim: compile-time head geometry (``head_dim % 8 == 0``).
        causal: bottom-right aligned causal mask, as in flash-attn: query row r
            of a sequence attends keys ``<= r + seq_kv - seq_q``, so a chunk of
            queries over a longer key range (chunked prefill, prefix caching,
            speculative decode) sees its whole prefix. Equal lengths give the
            usual diagonal; a row with no keys gets O = 0 and LSE = -inf.
        num_waves: waves per workgroup; each wave owns 16 query rows.
        block_n: keys per main-loop tile (multiple of 16).
        vt_rows: keys packed per LDS write in the V -> V^T transpose (2, 4, 8).
        double_buffer: ping-pong the K/V^T tile in LDS (one barrier per tile).
            Dropped automatically when two buffers do not fit in LDS.
        head_group: heads walked fastest in the block order (``None`` = all of
            them, 1 = one head at a time). See the block mapping in the kernel.
        waves_per_eu: ``rocdl.waves_per_eu`` occupancy hint.
        q_token_stride, k_token_stride, v_token_stride: elements between
            consecutive tokens of Q / K / V (``None`` = plain THD: ``num_heads *
            head_dim`` for Q, ``num_kv_heads * head_dim`` for K / V). Larger
            values read views of a packed tensor such as ``[total, 3, H, D]`` in
            place; heads must stay ``head_dim`` apart. O is always plain THD.
        num_kv_heads: K / V heads (``None`` = ``num_heads``). Fewer than
            ``num_heads`` is GQA / MQA: query head ``h`` reads K / V head
            ``h // (num_heads // num_kv_heads)``; ``num_heads`` must be a multiple.
        return_lse: also write the natural-log softmax log-sum-exp of every query
            row, fp32 ``[num_heads, total_q]``, to the ``LSE`` launch argument.
        rope: ``None``, or the dtype of the RoPE tables (``"bf16"`` / ``"f32"``)
            to rotate Q and K in the kernel before the attention: NeoX style
            over the whole head, ``x[d] * cos[d] - x[d + D/2] * sin[d]`` and
            ``x[d + D/2] * cos[d] + x[d] * sin[d]`` for ``d < D/2``, in fp32,
            rounded once. ``cos`` / ``sin`` are ``[total, head_dim / 2]``, one
            row per packed token, shared by Q and K, so rope needs self-attention
            (one ``cu_seqlens`` table for both) and ``head_dim % 16 == 0``. Q is
            rotated in registers; K while it is staged into LDS.

    Returns:
        ``launch(Q, K, V, O, cu_seqlens_q, cu_seqlens_k, batch, max_seqlen_q,
        stream=..., lse=..., cos=..., sin=...)``; the ``cu_seqlens`` are int32
        device tensors of ``batch + 1`` entries, ``lse`` is required with
        ``return_lse``, ``cos`` / ``sin`` (row stride a multiple of 8 elements,
        16-byte aligned) with ``rope``.

    The builder does not check launch arguments. ``flydsl_flash_attn_varlen_fwd``
    does; a direct caller must guarantee ``max_seqlen_q >= 1``, ``sm_scale > 0``,
    every sequence of Q / K / V / O below 2 GiB (32-bit buffer offsets; the
    dropped stores past ``head_dim`` rely on it), 16-byte aligned tensors, and O
    as plain THD. Sequences may be empty.
    """
    if head_dim % 8 != 0 or head_dim <= 0:
        raise ValueError(f"head_dim must be a positive multiple of 8, got {head_dim}")
    if dtype_str not in ("bf16", "f16"):
        raise ValueError(f"dtype_str must be bf16 or f16, got {dtype_str!r}")
    if block_n % WMMA_MNK != 0 or vt_rows not in (2, 4, 8) or block_n % vt_rows != 0:
        raise ValueError(f"unsupported block_n={block_n} / vt_rows={vt_rows}")
    if rope not in (None, "bf16", "f32"):
        raise ValueError(f"rope must be None, 'bf16' or 'f32', got {rope!r}")
    if rope is not None and head_dim % 16 != 0:
        # The rotation pairs 8-element chunks d and d + D/2.
        raise ValueError(f"rope needs head_dim % 16 == 0, got {head_dim}")

    H = num_heads
    D = head_dim
    HKV = num_heads if num_kv_heads is None else int(num_kv_heads)
    if HKV < 1 or H % HKV != 0:
        raise ValueError(f"num_heads={H} must be a multiple of num_kv_heads={HKV}")
    GROUP = H // HKV  # query heads per K/V head
    if head_group is None:
        head_group = num_heads
    HEAD_GROUP = head_group if head_group >= 1 and num_heads % head_group == 0 else 1
    DP = -(-D // WMMA_MNK) * WMMA_MNK
    ND = DP // WMMA_MNK
    BLOCK_M = WMMA_MNK * num_waves
    BLOCK_N = block_n
    NK = BLOCK_N // WMMA_MNK
    THREADS = num_waves * WAVE_SIZE
    TOK = H * D  # elements between consecutive Q / O tokens (THD)
    TOK_KV = HKV * D  # elements of one K / V token row
    TOK_Q = TOK if q_token_stride is None else int(q_token_stride)
    TOK_K = TOK_KV if k_token_stride is None else int(k_token_stride)
    TOK_V = TOK_KV if v_token_stride is None else int(v_token_stride)
    for name, s, row in (("q", TOK_Q, TOK), ("k", TOK_K, TOK_KV), ("v", TOK_V, TOK_KV)):
        # 128-bit loads need every row start 16-byte aligned.
        if s < row or s % 8 != 0:
            raise ValueError(
                f"{name}_token_stride must be a multiple of 8 and >= {row}, got {s}"
            )
    is_bf16 = dtype_str == "bf16"
    elem = fx.BFloat16 if is_bf16 else fx.Float16

    if lds_bytes(D, BLOCK_N, False) > LDS_LIMIT_BYTES:
        # A launch over the LDS limit leaves the HIP context unusable; refuse here.
        raise ValueError(
            f"head_dim={D}, block_n={BLOCK_N} exceeds {LDS_LIMIT_BYTES} B of LDS"
        )
    double_buffer = double_buffer and lds_bytes(D, BLOCK_N, True) <= LDS_LIMIT_BYTES
    NBUF = 2 if double_buffer else 1

    K_ROW = DP + LDS_PAD
    V_ROW = BLOCK_N + LDS_PAD
    K_SIZE = BLOCK_N * K_ROW
    BUF = K_SIZE + DP * V_ROW

    DBLKS = D // 8
    # K staging unit: one 8-element chunk per (key, d0); with rope a pair of
    # them, d0 and d0 + D/2, which the rotation combines.
    K_CHUNK_ELEMS = 2 if rope is not None else 1
    K_DBLKS = DBLKS // K_CHUNK_ELEMS  # units per key row
    K_CHUNKS = BLOCK_N * K_DBLKS
    K_PER_THR = -(-K_CHUNKS // THREADS)
    HALF = D // 2  # rope: partner offset; cos / sin row length
    V_BLOCKS = (BLOCK_N // vt_rows) * DBLKS  # vt_rows keys x 8 d per block
    V_PER_THR = -(-V_BLOCKS // THREADS)

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(D)
    SCALE_LOG2 = float(sm_scale) * _LOG2E
    CONCAT16 = list(range(16))

    @fx.struct
    class SharedStorage:
        kv: fx.Array[elem, BUF * NBUF, 16]

    def _wmma(a, b, acc):
        if const_expr(is_bf16):
            return fx.rocdl.wmma_f32_16x16x16_bf16(
                acc.type, a.bitcast(fx.Int16), b.bitcast(fx.Int16), acc
            ).result
        return fx.rocdl.wmma_f32_16x16x16_f16(acc.type, a, b, acc).result

    def _lds_barrier():
        # Waits on LDS only. gpu.barrier() also drains vmcnt, which would block on
        # the next tile's global prefetch issued at the top of the iteration.
        _llvm.inline_asm(
            res=None,
            operands_=[],
            asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
            constraints="",
            has_side_effects=True,
        )

    def _exp2(x):
        # Arguments are <= 0 (<= -huge for masked keys); a flushed denormal is fine.
        return fx.Float32(fx.rocdl.exp2(T.f32, fx.Float32(x).ir_value()))

    def _log2(x):
        # Only applied to the row sum, which is >= 1 unless the sequence has no keys.
        return fx.Float32(fx.rocdl.log(T.f32, fx.Float32(x).ir_value()))

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def flash_attn_varlen_func_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        LSE: fx.Pointer,
        CU_Q: fx.Pointer,
        CU_K: fx.Pointer,
        COS: fx.Pointer,
        SIN: fx.Pointer,
        batch_size: fx.Int32,
        n_qblk_max: fx.Int32,
        lse_stride: fx.Int32,
        cs_stride: fx.Int32,
    ):
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().kv.ptr

        def lds_view(off, n):
            p = fx.recast_iter(
                elem, fx.add_offset(lds_ptr, fx.make_int_tuple(fx.Int32(off)))
            )
            return fx.make_view(p, fx.make_layout(n, 1))

        def lds_v16(off):
            return Vec(lds_view(off, 8).load()).shuffle(
                Vec(lds_view(off + 8, 8).load()), CONCAT16
            )

        tid = fx.thread_idx.x
        # Sequence slowest (grid z): consecutive blocks then work on the same
        # tokens of one sequence (several heads of the same rows share cache
        # lines). Interleaving the sequences instead measured 2-3x slower on
        # batches of 8-16 images.
        lin = fx.block_idx.x
        bat = fx.block_idx.z

        # This sequence's token ranges: cu_seqlens[bat] to cu_seqlens[bat + 1],
        # as scalar (SMEM) loads: the lengths then stay in SGPRs and the loop
        # bounds and descriptors below are uniform. The table descriptor is
        # bounded, so an index past it reads 0.
        def cu_pair(ptr, idx):
            rsrc = buffer_ops.create_buffer_resource_from_addr(
                buf_base_i64(ptr),
                num_records_bytes=(fx.Int64(batch_size) + 1) * 4,
            )
            return tuple(
                fx.Int32(
                    buffer_ops.buffer_load(
                        rsrc, fx.Int32(idx) + i, vec_width=1, is_scalar=True
                    )
                )
                for i in range_constexpr(2)
            )

        q_start, q_end = cu_pair(CU_Q, bat)
        kv_start, kv_end = cu_pair(CU_K, bat)
        seq_q = q_end - q_start
        seq_kv = kv_end - kv_start
        # Causal diagonal offset (bottom-right alignment): row r sees keys
        # <= r + diag. 0 for equal lengths, negative when the keys are fewer.
        diag = seq_kv - seq_q
        n_qblk = (seq_q + (BLOCK_M - 1)) // BLOCK_M
        # Linear block id -> (head, qblk), HEAD_GROUP heads fastest. In THD one
        # token row holds every head, so workgroups resident together that cover
        # several heads of the same tokens share pages; walking one head at a time
        # instead sweeps the whole tensor touching 2*D bytes per row, which on
        # gfx1151 measured up to 4x slower (address translation bound) once a
        # slice passed ~128 MiB. The grid is sized for n_qblk_max query blocks
        # (the longest sequence).
        per_group = n_qblk_max * HEAD_GROUP
        grp = lin // per_group
        rem = lin - grp * per_group
        head = grp * HEAD_GROUP + rem % HEAD_GROUP
        qblk = rem // HEAD_GROUP
        wave = tid // WAVE_SIZE
        lane = tid % WAVE_SIZE
        l16 = lane % 16
        hh = lane // 16
        is_h0 = hh == 0

        n_kv_tiles = (seq_kv + (BLOCK_N - 1)) // BLOCK_N
        # A block past this sequence's last query block does nothing: it walks no
        # KV tile and its rows are past seq_q, so the Q loads return 0 and the
        # O / LSE stores are dropped by the descriptors.
        active = qblk < n_qblk
        if const_expr(causal):
            # Longest rows first: the diagonal blocks have the most KV tiles.
            qblk = active.select(n_qblk - 1 - qblk, qblk)

        def seq_buf(ptr, seqlen, tok, row, start):
            # Bounded per sequence, up to the end of its last token's row: the
            # kernel relies on OOB loads returning 0 (rows past seq_q / seq_kv)
            # and OOB stores being dropped. With a token stride above H*D (packed
            # views) the bytes between rows belong to other tensors and are never
            # addressed. row: elements of one token row (H*D for Q / O, HKV*D for
            # K / V). start: element offset of the sequence's first token row. An
            # empty sequence gets an empty descriptor.
            slice_bytes = (fx.Int64(seqlen) * tok - (tok - row)) * 2
            slice_bytes = (seqlen > 0).select(slice_bytes, fx.Int64(0))
            base = buf_base_i64(ptr) + fx.Int64(start) * 2
            return ptr_buf_tensor(
                base,
                elem=elem,
                unit_elems=8,
                unit_stride=1,
                num_records_bytes=slice_bytes,
            )

        q_buf = seq_buf(Q, seq_q, TOK_Q, TOK, fx.Int64(q_start) * TOK_Q)
        k_buf = seq_buf(K, seq_kv, TOK_K, TOK_KV, fx.Int64(kv_start) * TOK_K)
        v_buf = seq_buf(V, seq_kv, TOK_V, TOK_KV, fx.Int64(kv_start) * TOK_V)
        o_buf = seq_buf(O, seq_q, TOK, TOK, fx.Int64(q_start) * TOK)

        def g_load8(buf, off):
            return buf_copy_load(buf, off, elem=elem, unit_elems=8)

        def tok_off(tok, stride=TOK):
            return tok * stride + head * D

        kv_head = head // GROUP if GROUP > 1 else head  # GROUP is a Python int

        def kv_off(tok, stride):
            return tok * stride + kv_head * D

        if const_expr(rope is not None):
            # RoPE tables, [total, D/2] rows cs_stride elements apart, bounded
            # per sequence like Q / K (rope implies q_start == kv_start).
            cs_elem = fx.BFloat16 if rope == "bf16" else fx.Float32
            cs_bytes = 2 if rope == "bf16" else 4
            cs_unit = 8 if rope == "bf16" else 4  # 16-byte loads

            def cs_buf(ptr):
                rows_bytes = (
                    fx.Int64(seq_q) * fx.Int64(cs_stride) - (fx.Int64(cs_stride) - HALF)
                ) * cs_bytes
                return ptr_buf_tensor(
                    buf_base_i64(ptr)
                    + fx.Int64(q_start) * fx.Int64(cs_stride) * cs_bytes,
                    elem=cs_elem,
                    unit_elems=cs_unit,
                    unit_stride=1,
                    num_records_bytes=(seq_q > 0).select(rows_bytes, fx.Int64(0)),
                )

            cos_buf = cs_buf(COS)
            sin_buf = cs_buf(SIN)

            def cs_raw8(buf, tok, d0):
                """cos or sin of token `tok`, columns d0 .. d0 + 7, as loaded
                (kept narrow while a prefetch is in flight)."""
                off = tok * cs_stride + d0
                if const_expr(rope == "bf16"):
                    return [buf_copy_load(buf, off, elem=cs_elem, unit_elems=8)]
                return [
                    buf_copy_load(buf, off + 4 * h, elem=cs_elem, unit_elems=4)
                    for h in range_constexpr(2)
                ]

            def cs_f32(raw):
                if const_expr(rope == "bf16"):
                    return Vec(raw[0]).to(fx.Float32)
                return Vec(raw[0]).shuffle(Vec(raw[1]), list(range(8)))

            def cs_load8(buf, tok, d0):
                """cos or sin of token `tok`, columns d0 .. d0 + 7, as 8 x f32."""
                return cs_f32(cs_raw8(buf, tok, d0))

            def rotate(x1, x2, c, s):
                """NeoX rotation of the chunk pair (d0, d0 + D/2): fp32 math,
                one rounding to the input dtype."""
                f1, f2 = Vec(x1).to(fx.Float32), Vec(x2).to(fx.Float32)
                return (f1 * c - f2 * s).to(elem), (f2 * c + f1 * s).to(elem)

        # ---- Q -> VGPRs as WMMA B operands (zero past seq_q / past head_dim) ----
        zero_e8 = Vec.filled(8, 0.0, elem)
        q_row = qblk * BLOCK_M + wave * 16 + l16
        q_base = tok_off(q_row, TOK_Q)
        if const_expr(rope is not None):
            # Rotated in registers: the lane holds its query row's whole head.
            chunks = [g_load8(q_buf, q_base + 8 * g) for g in range_constexpr(DBLKS)]
            for g in range_constexpr(DBLKS // 2):
                c = cs_load8(cos_buf, q_row, 8 * g)
                s = cs_load8(sin_buf, q_row, 8 * g)
                chunks[g], chunks[g + DBLKS // 2] = rotate(
                    chunks[g], chunks[g + DBLKS // 2], c, s
                )
        q_vecs = []
        for c in range_constexpr(ND):
            lo = zero_e8
            hi = zero_e8
            if const_expr(rope is not None):
                lo, hi = chunks[2 * c], chunks[2 * c + 1]  # D % 16 == 0
            else:
                if const_expr(16 * c < D):
                    lo = g_load8(q_buf, q_base + 16 * c)
                if const_expr(16 * c + 8 < D):
                    hi = g_load8(q_buf, q_base + 16 * c + 8)
            q_vecs.append(Vec(lo).shuffle(Vec(hi), CONCAT16))

        if const_expr(causal):
            # Keys this block's last row can see: its row index + diag, clamped
            # at 0 so a block with no visible keys walks no tile.
            last_key = (qblk + 1) * BLOCK_M + diag
            last_key = (last_key > 0).select(last_key, fx.Int32(0))
            n_tiles = (last_key + (BLOCK_N - 1)) // BLOCK_N
            n_tiles = (n_tiles < n_kv_tiles).select(n_tiles, n_kv_tiles)
            q_lim = q_row + diag
        else:
            n_tiles = n_kv_tiles
        n_tiles = active.select(n_tiles, fx.Int32(0))

        # ---- K / V tile staging: global -> VGPRs -> LDS (K row-major, V^T) ----
        def k_chunk(i):
            chunk = tid + i * THREADS
            if const_expr(K_PER_THR * THREADS > K_CHUNKS):
                # Surplus threads redo the last chunk: same data, same address.
                chunk = (chunk < K_CHUNKS).select(chunk, K_CHUNKS - 1)
            return chunk

        def v_block(i):
            blk = tid + i * THREADS
            if const_expr(V_PER_THR * THREADS > V_BLOCKS):
                blk = (blk < V_BLOCKS).select(blk, V_BLOCKS - 1)
            return blk

        def gload_tile(tile):
            key0 = tile * BLOCK_N
            kregs = []
            for i in range_constexpr(K_PER_THR):
                chunk = k_chunk(i)
                row = chunk // K_DBLKS
                d0 = (chunk % K_DBLKS) * 8
                k_off = kv_off(key0 + row, TOK_K) + d0
                if const_expr(rope is not None):
                    # The chunk pair (d0, d0 + D/2) and the key's cos / sin.
                    kregs.append(
                        (
                            g_load8(k_buf, k_off),
                            g_load8(k_buf, k_off + HALF),
                            cs_raw8(cos_buf, key0 + row, d0),
                            cs_raw8(sin_buf, key0 + row, d0),
                        )
                    )
                else:
                    kregs.append(g_load8(k_buf, k_off))
            vregs = []
            for i in range_constexpr(V_PER_THR):
                blk = v_block(i)
                kb0 = key0 + (blk // DBLKS) * vt_rows
                d0 = (blk % DBLKS) * 8
                vregs.append(
                    [
                        g_load8(v_buf, kv_off(kb0 + r, TOK_V) + d0)
                        for r in range_constexpr(vt_rows)
                    ]
                )
            return kregs, vregs

        def store_tile(regs, bb):
            kregs, vregs = regs
            for i in range_constexpr(K_PER_THR):
                chunk = k_chunk(i)
                dst = bb + (chunk // K_DBLKS) * K_ROW + (chunk % K_DBLKS) * 8
                if const_expr(rope is not None):
                    k1, k2, c, s = kregs[i]
                    if const_expr(K_PER_THR * THREADS > K_CHUNKS):
                        # The rotation costs ~200 VALU per unit; the waves past
                        # the last unit (they would redo it) skip it instead of
                        # competing for the SIMDs.
                        if tid + i * THREADS < K_CHUNKS:
                            r1, r2 = rotate(k1, k2, cs_f32(c), cs_f32(s))
                            lds_view(dst, 8).store(Vec(r1))
                            lds_view(dst + HALF, 8).store(Vec(r2))
                    else:
                        r1, r2 = rotate(k1, k2, cs_f32(c), cs_f32(s))
                        lds_view(dst, 8).store(Vec(r1))
                        lds_view(dst + HALF, 8).store(Vec(r2))
                else:
                    lds_view(dst, 8).store(Vec(kregs[i]))
            for i in range_constexpr(V_PER_THR):
                blk = v_block(i)
                krow0 = (blk // DBLKS) * vt_rows
                d0 = (blk % DBLKS) * 8
                # Dword t of a row holds d0+2t (low half) and d0+2t+1 (high half);
                # pair up consecutive keys of one d into dwords along V^T rows.
                words = [Vec(r).bitcast(fx.Int32) for r in vregs[i]]
                for t in range_constexpr(4):
                    lo_w = []
                    hi_w = []
                    for pr in range_constexpr(vt_rows // 2):
                        a = fx.Int32(words[2 * pr][t])
                        b = fx.Int32(words[2 * pr + 1][t])
                        lo_w.append((a & 0xFFFF) | (b << 16))
                        hi_w.append(a.shrui(fx.Int32(16)) | (b & fx.Int32(-65536)))
                    for half, ws in ((0, lo_w), (1, hi_w)):
                        dst = bb + K_SIZE + (d0 + 2 * t + half) * V_ROW + krow0
                        lds_view(dst, vt_rows).store(
                            Vec.from_elements(ws, fx.Int32).bitcast(elem)
                        )

        if const_expr(DP != D):
            # K pad columns [D, DP) feed the S^T WMMA but are never loaded. LDS is
            # uninitialized and NaN * 0 = NaN, so zero them once per buffer.
            for i in range_constexpr(-(-BLOCK_N // THREADS)):
                zrow = tid + i * THREADS
                zrow = (zrow < BLOCK_N).select(zrow, BLOCK_N - 1)
                for b in range_constexpr(NBUF):
                    lds_view(b * BUF + zrow * K_ROW + D, 8).store(zero_e8)

        # ---- P^T operand assembly ----
        def pack2(a, b):
            """Two f32 P values in [0, 1] -> one dword of elem, a in the low half."""
            if const_expr(is_bf16):
                # Round-to-nearest-even without the NaN guard (P is never NaN):
                # gfx11 has no f32 -> bf16 convert instruction.
                ra = a.bitcast(fx.Int32)
                rb = b.bitcast(fx.Int32)
                ra = ra + 0x7FFF + (ra.shrui(fx.Int32(16)) & 1)
                rb = rb + 0x7FFF + (rb.shrui(fx.Int32(16)) & 1)
                return fx.Int32(fx.rocdl.perm_b32(rb, ra, fx.Int32(0x07060302)))
            return Vec.from_elements([a.to(elem), b.to(elem)], elem).bitcast(fx.Int32)[
                0
            ]

        # Lane L holds keys 2*si + L/16 of its query; the B operand wants all 16 keys
        # in order. After swapping packed pairs with lane L^16, one v_perm_b32 per
        # dword interleaves own/partner halves (selector depends on the lane half).
        sel_even = is_h0.select(fx.Int32(0x05040100), fx.Int32(0x01000504))
        sel_odd = is_h0.select(fx.Int32(0x07060302), fx.Int32(0x03020706))

        zero8 = Vec.filled(8, 0.0, fx.Float32)
        m0 = fx.Float32(NEG_BIG)
        if const_expr(causal):
            # A row that can see no key at all (fewer keys than queries): with
            # m = NEG_BIG its masked x - m would be rounding noise and exp2 of it
            # 1 or inf. Starting such a row at +BIG makes every p exactly 0, so
            # l and O stay 0 and the LSE is -inf. Rows with keys see key 0 in
            # the first tile and are unchanged.
            m0 = (q_lim < 0).select(fx.Float32(POS_BIG), m0)
        init = [m0, fx.Float32(0.0)] + [zero8 for _ in range_constexpr(ND)]

        store_tile(gload_tile(fx.Int32(0)), 0)
        _lds_barrier()

        def tile_step(j32, state, mask_tail):
            """One KV tile: S^T, online softmax, O^T update, next tile into LDS.
            mask_tail (compile-time): mask keys at or past seq_kv; only the last
            KV tile can hold such keys, so the main loop runs without it."""
            m_old = fx.Float32(state[0])
            l_old = fx.Float32(state[1])
            o_acc = [Vec(state[2 + i]) for i in range_constexpr(ND)]
            cur = 0
            if const_expr(double_buffer):
                cur = (j32 & 1) * BUF

            nxt = j32 + 1
            nxt = (nxt < n_tiles).select(nxt, j32)
            next_regs = gload_tile(nxt)
            # Keep the prefetch at the top of the body; the machine scheduler would
            # otherwise sink it next to its LDS store and lose the overlap.
            fx.rocdl.sched_barrier(0)

            # ---- S^T = K Q^T; c outer so consecutive WMMAs hit independent accs ----
            s_accs = [zero8 for _ in range_constexpr(NK)]
            for c in range_constexpr(ND):
                for kb in range_constexpr(NK):
                    a = lds_v16(cur + (kb * 16 + l16) * K_ROW + 16 * c)
                    s_accs[kb] = _wmma(a, q_vecs[c], s_accs[kb])

            s_vals = []
            for kb in range_constexpr(NK):
                acc = Vec(s_accs[kb])
                row = []
                for si in range_constexpr(8):
                    x = fx.Float32(acc[si])
                    if const_expr(causal or mask_tail):
                        key = j32 * BLOCK_N + kb * 16 + 2 * si + hh
                    if const_expr(causal):
                        x = (key > q_lim).select(fx.Float32(NEG_BIG), x)
                    if const_expr(mask_tail):
                        x = (key >= seq_kv).select(fx.Float32(NEG_BIG), x)
                    row.append(x)
                s_vals.append(row)

            # ---- online softmax; m in raw S units, the scale folds into one fma ----
            m_loc = s_vals[0][0]
            for kb in range_constexpr(NK):
                for si in range_constexpr(8):
                    if const_expr(kb != 0 or si != 0):
                        m_loc = fx.maxnumf(m_loc, s_vals[kb][si])
            m_loc = fx.maxnumf(m_loc, m_loc.shuffle_xor(16, WAVE_SIZE))
            m_new = fx.maxnumf(m_old, m_loc)
            alpha = _exp2((m_old - m_new) * SCALE_LOG2)
            neg_mc = m_new * (-SCALE_LOG2)
            l_new = l_old * alpha
            o_acc = [
                Vec.from_elements(
                    [fx.Float32(o[si]) * alpha for si in range_constexpr(8)], fx.Float32
                )
                for o in o_acc
            ]

            for kb in range_constexpr(NK):
                own = []
                for si in range_constexpr(8):
                    p = _exp2(fx.fma(s_vals[kb][si], fx.Float32(SCALE_LOG2), neg_mc))
                    l_new = l_new + p
                    own.append(p)
                ownp = [
                    pack2(own[2 * j2], own[2 * j2 + 1]) for j2 in range_constexpr(4)
                ]
                partp = [
                    ownp[j2].shuffle_xor(16, WAVE_SIZE) for j2 in range_constexpr(4)
                ]
                dwords = []
                for j2 in range_constexpr(4):
                    dwords.append(
                        fx.Int32(fx.rocdl.perm_b32(partp[j2], ownp[j2], sel_even))
                    )
                    dwords.append(
                        fx.Int32(fx.rocdl.perm_b32(partp[j2], ownp[j2], sel_odd))
                    )
                p_vec = Vec.from_elements(dwords, fx.Int32).bitcast(elem)
                for dc in range_constexpr(ND):
                    a = lds_v16(cur + K_SIZE + (dc * 16 + l16) * V_ROW + kb * 16)
                    o_acc[dc] = Vec(_wmma(a, p_vec, o_acc[dc]))

            if const_expr(double_buffer):
                # The other buffer was last read in iteration j-1 and every wave has
                # passed that iteration's barrier, so it is free to overwrite now.
                store_tile(next_regs, BUF - cur)
                _lds_barrier()
            else:
                _lds_barrier()
                store_tile(next_regs, 0)
                _lds_barrier()

            return [m_new, l_new] + o_acc

        # Only the last KV tile can hold keys at or past seq_kv, and only when the
        # tiles walked here reach it (causal rows may stop earlier). The tiles
        # before it run without the tail mask; the partial one gets it.
        partial = ((seq_kv % BLOCK_N) > 0).select(fx.Int32(1), fx.Int32(0))
        n_full = n_tiles - (n_tiles == n_kv_tiles).select(partial, fx.Int32(0))

        for j, state in range(0, n_full, 1, init=init):
            results = yield tile_step(fx.Int32(j), state, False)

        for j, state in range(n_full, n_tiles, 1, init=results):
            results = yield tile_step(fx.Int32(j), state, True)

        # ---- epilogue: O = O^T / l, 128-bit stores ----
        l_fin = fx.Float32(results[1])
        l_fin = l_fin + l_fin.shuffle_xor(16, WAVE_SIZE)
        # l is 0 only when no key was seen (a sequence without keys): O = 0 then.
        inv = (l_fin > 0.0).select(fx.Float32(1.0) / l_fin, fx.Float32(0.0))
        # Recomputed rather than kept live across the main loop (VGPR pressure).
        q_base = tok_off(qblk * BLOCK_M + (tid // WAVE_SIZE) * 16 + (tid % 16))
        for dc in range_constexpr(ND):
            if const_expr(dc * 16 < D):
                o = Vec(results[2 + dc])
                # Lane L holds d = 16*dc + 2*si + L/16. After a swap with lane L^16
                # both lanes see all 16 d of the query; lanes 0-15 store d 0..7 and
                # lanes 16-31 store d 8..15.
                own = [fx.Float32(o[si]) * inv for si in range_constexpr(8)]
                part = [own[si].shuffle_xor(16, WAVE_SIZE) for si in range_constexpr(8)]
                ev = [is_h0.select(own[si], part[si]) for si in range_constexpr(8)]
                od = [is_h0.select(part[si], own[si]) for si in range_constexpr(8)]
                out = []
                for si in range_constexpr(4):
                    out.append(is_h0.select(ev[si], ev[si + 4]).to(elem))
                    out.append(is_h0.select(od[si], od[si + 4]).to(elem))
                off = q_base + dc * 16 + hh * 8
                if const_expr(dc * 16 + 8 >= D):
                    # D % 16 == 8: the upper half is padding. Point it past the
                    # descriptor bound so the hardware drops the store.
                    off = is_h0.select(off, 0x3FFFFFFF)
                buf_copy_store(
                    o_buf, off, Vec.from_elements(out, elem), elem=elem, unit_elems=8
                )

        if const_expr(return_lse):
            # lse = scale * m + ln(l), m in raw S units: in log2 units first, then
            # times ln 2. Both lane halves hold the row's m and l; lanes 0-15
            # store. [num_heads, total_q]: this sequence's row of head `head`;
            # rows past seq_q and the upper lane half are pointed past the bound
            # and dropped.
            lse = (
                fx.Float32(results[0]) * fx.Float32(SCALE_LOG2) + _log2(l_fin)
            ) * fx.Float32(_LN2)
            q_row_fin = qblk * BLOCK_M + (tid // WAVE_SIZE) * 16 + (tid % 16)
            lse_base = buf_base_i64(LSE) + (
                (fx.Int64(head) * fx.Int64(lse_stride) + fx.Int64(q_start)) * 4
            )
            lse_buf = ptr_buf_tensor(
                lse_base,
                elem=fx.Float32,
                unit_elems=1,
                num_records_bytes=fx.Int64(seq_q) * 4,
            )
            buf_copy_store(
                lse_buf,
                is_h0.select(q_row_fin, seq_q),
                lse,
                elem=fx.Float32,
                unit_elems=1,
            )

    @flyc.jit
    def launch_flash_attn_varlen_func(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        LSE: fx.Pointer,
        CU_Q: fx.Pointer,
        CU_K: fx.Pointer,
        COS: fx.Pointer,
        SIN: fx.Pointer,
        batch_size: fx.Int32,
        max_seqlen_q: fx.Int32,
        lse_stride: fx.Int32,
        cs_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(  # noqa: B008  framework idiom: default is evaluated once at import on purpose
            None
        ),
    ):
        # The grid covers the longest sequence.
        n_qblk = (max_seqlen_q + (BLOCK_M - 1)) // BLOCK_M
        flash_attn_varlen_func_kernel(
            Q,
            K,
            V,
            O,
            LSE,
            CU_Q,
            CU_K,
            COS,
            SIN,
            batch_size,
            n_qblk,
            lse_stride,
            cs_stride,
            value_attrs={
                "rocdl.waves_per_eu": (
                    int(waves_per_eu)
                    if const_expr(waves_per_eu is not None and int(waves_per_eu) >= 1)
                    else None
                ),
            },
        ).launch(grid=(n_qblk * H, 1, batch_size), block=(THREADS, 1, 1), stream=stream)

    launch_flash_attn_varlen_func.compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
    }

    def _launch(
        Q,
        K,
        V,
        O,
        cu_seqlens_q,
        cu_seqlens_k,
        batch_size,
        max_seqlen_q,
        stream=None,
        lse=None,
        cos=None,
        sin=None,
    ):
        if return_lse and lse is None:
            raise ValueError(
                "built with return_lse=True: pass lse=, fp32 [num_heads, total_q]"
            )
        if rope is not None and (cos is None or sin is None):
            raise ValueError(
                "built with rope: pass cos= and sin=, [total, head_dim / 2]"
            )
        o_ptr = ptr_arg(O)
        _run_compiled(
            launch_flash_attn_varlen_func,
            ptr_arg(Q),
            ptr_arg(K),
            ptr_arg(V),
            o_ptr,
            o_ptr if lse is None else ptr_arg(lse),  # not read unless return_lse
            # int32 tables: the kernel indexes them in elements
            ptr_arg(cu_seqlens_q, fx.Int32),
            ptr_arg(cu_seqlens_k, fx.Int32),
            o_ptr if cos is None else ptr_arg(cos),  # not read unless rope
            o_ptr if sin is None else ptr_arg(sin),
            batch_size,
            max_seqlen_q,
            0 if lse is None else lse.shape[-1],
            0 if cos is None else cos.stride(0),
            fx.Stream(None) if stream is None else stream,
        )

    _launch.block_m = BLOCK_M
    _launch.block_n = BLOCK_N
    _launch.double_buffer = double_buffer
    _launch.rope = rope
    return _launch

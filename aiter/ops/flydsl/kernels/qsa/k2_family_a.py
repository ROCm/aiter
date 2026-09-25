# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Family A FlyDSL QSA K2, shaped after the live AMD Triton kernel.

One workgroup owns ``(row, kv_head, split)``.  Q is staged once, paged K/V
are gathered in ``BLOCK_N`` tiles, QK and PV use BF16 MFMA, and online
softmax is maintained in log2 space.  Host dispatch mirrors live AMD:
small decode uses BLOCK_N=16 / four waves / split-K, while prefill uses
BLOCK_N=64 / two waves / one split and writes output directly.

gfx942 aliases K and V in one LDS tile so BLOCK_N=64 stays under 64 KiB.
gfx950 stores K and V separately and gathers this tile's V after K is
visible so QK can run while V is in flight.  Softmax stays in registers
(full-D QK on every wave, in-wave P transpose); LDS is MMA scratch only.
Expand, partial RoPE, and the sigmoid output gate remain outside K2.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import (
    BFloat16,
    Float32,
    Int32,
    Int64,
    const_expr,
    gpu,
    range_constexpr,
)
from flydsl.expr import math as fxmath
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from aiter.ops.flydsl.kernels.qsa.shapes import FAMILY_A_GQA
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, buf_copy_atom

_HQ = FAMILY_A_GQA.n_heads
_HK = FAMILY_A_GQA.kv_heads
_GROUP = FAMILY_A_GQA.group_size
_D = FAMILY_A_GQA.head_dim
# Pad K (and aliased KV) rows so consecutive columns do not share LDS banks
# on 128-bit stores. D=256 makes n*D a multiple of 32 banks otherwise.
_K_STRIDE = _D + 8
_HEAD_PAD = 16
# 256 CUs on MI355X times the four BN32 prefill workgroups each keeps resident.
_PREFILL_WGS = 256 * 4
_DEFAULT_SCALE = _D**-0.5
_LSE_EMPTY = -1.0e20
_LOG2E = 1.4426950408889634


def _idiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def _neg_inf():
    return Float32(float("-inf"))


def _exp2(x):
    return Float32(fx.rocdl.exp2(Float32.ir_type, Float32(x).ir_value()))


def _ds_write2st64_b64(addr, data0, data1, offset0=0, offset1=16):
    """Store two BF16x4 vectors in gfx950 LDS. Offsets are st64 units (512 B)."""
    off = f"offset0:{offset0} offset1:{offset1}"
    if offset0 == 0:
        off = f"offset1:{offset1}"
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [as_mlir_value(addr), as_mlir_value(data0), as_mlir_value(data1)],
        f"ds_write2st64_b64 $0, $1, $2 {off}\n",
        "v,v,v,~{memory}",
        has_side_effects=True,
    )


def _launch_config(rows: int, n_sel: int) -> tuple[int, int, int]:
    """Return ``(BLOCK_N, threads, splits)`` using the tuned AMD-shaped policy."""
    base_programs = rows * _HK
    # Live AMD: base_programs <= 4 -> 64 splits / 128 WGs; 4 < base < 32
    # -> 32 splits / 512 WGs. Same BN16 / 4-wave tile in both bands.
    if base_programs <= 4:
        block_n, target_splits, threads = 16, 64, 256
    elif base_programs < 32:
        block_n, target_splits, threads = 16, 32, 256
    # Prefill runs BN32 rather than AMD's BN64. Halving the tile halves the
    # live gather and QK state, which drops the kernel from 204 to 122 VGPRs
    # and all but removes its instruction-issue stalls. Occupancy is not the
    # reason: the grid is unchanged, so the extra residency headroom BN32
    # unlocks goes unused at these shapes.
    #
    # Split so the grid lands on the four workgroups per CU that the BN32
    # K-plus-V LDS footprint keeps resident. Splitting past that point buys
    # no extra parallelism and still pays merge and address-chase duplication.
    else:
        block_n, threads = 32, 128
        target_splits = max(1, _PREFILL_WGS // base_programs)

    tiles = max(1, (n_sel + block_n - 1) // block_n)
    max_useful_splits = 1 << (tiles.bit_length() - 1)
    return block_n, threads, min(max_useful_splits, target_splits)


def build_qsa_k2_family_a_module(
    page_size: int,
    use_k32: bool,
    block_n: int,
    block_threads: int,
    n_splits: int,
):
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if block_n not in (16, 32, 64):
        raise ValueError(f"BLOCK_N must be 16, 32 or 64, got {block_n}")
    if block_threads not in (128, 256):
        raise ValueError(f"block_threads must be 128 or 256, got {block_threads}")
    if block_threads % 64 or block_threads % block_n:
        raise ValueError("thread and column mappings must divide evenly")
    if n_splits < 1 or n_splits > 64:
        raise ValueError(f"n_splits must be in 1..64, got {n_splits}")
    if _HQ != _HK * _GROUP:
        raise ValueError("family A GQA head counts do not form groups")

    num_waves = block_threads // 64
    n_subtiles = block_n // 16
    qk_k = 32 if use_k32 else 16
    qk_vec = qk_k // 4
    qk_steps = _D // qk_k
    out_chunks = _D // (num_waves * 16)
    vec = 8
    d_chunks = _D // vec
    col_owners = block_threads // block_n
    gather_rounds = d_chunks // col_owners
    # Every gather round in flight costs 4 VGPRs. BN64 owns only two threads
    # per column, so it runs 16 rounds and holding them all pins 64 registers
    # for K and another 64 for the V prefetch that stays live across QK. Four
    # rounds in flight is enough to keep the memory pipe fed on the long
    # prefill shapes and measured best there; BN16 decode already runs two
    # rounds total, so it stays a single chunk and is structurally unchanged.
    gather_chunk = next(
        c for c in range(min(gather_rounds, 4), 0, -1) if gather_rounds % c == 0
    )
    n_gather_chunks = gather_rounds // gather_chunk
    gather_span = col_owners * vec
    # The V LDS image is a pure (token, dim) swizzle -- see the PV read below,
    # which never mentions the thread count. The only place the launch shape
    # leaks in is the ds_write2st64 immediate, so derive it here rather than
    # pinning it to the 128-thread case. Consecutive 8-element dim chunks sit
    # 32 tokens * 4 elements * 2 B = 256 B apart, a round advances by
    # col_owners chunks, and an st64 unit is 512 B.
    v_round_st64 = col_owners * 256 // 512
    # K LDS image, following the live AMD prefill lowering. A dim chunk is
    # split three ways: ``owner`` picks one of the col_owners threads sharing
    # a token, and the rest splits into a quarter and a group whose radices
    # multiply to gather_rounds. One group slot holds a 16 B vector for every
    # thread, so the whole image is block_n * _D * 2 B for any launch shape.
    k_quarter_radix = min(4, gather_rounds)
    k_group_radix = gather_rounds // k_quarter_radix
    k_group_slot = block_threads * 16
    k_quarter_stride = k_group_radix * k_group_slot
    token_major_v = use_k32 and block_n == 16
    decode_tr_pv = token_major_v
    # Prefill gives V its own LDS region instead of overlaying the K tile.
    # Overlaying costs a barrier per tile to drain every wave's QK reads
    # before the first V store, which stalls the V gather behind QK. BN32
    # leaves enough LDS to just pay for the second region. Decode keeps the
    # overlay: its tile is a quarter the size and it uses a different V map.
    split_kv_lds = use_k32 and not token_major_v
    v_elem_off = block_n * _D if split_kv_lds else 0
    v_pf_rounds = gather_rounds if split_kv_lds else gather_chunk
    v_pf_chunks = v_pf_rounds // gather_chunk
    # PV splits the output dimension across waves, so every wave needs the
    # whole score tile and by default every wave recomputes all of QK. When
    # the token subtiles divide across waves each wave can instead compute
    # its own share and trade scores through LDS, which removes
    # (num_waves - 1) / num_waves of the QK MFMA. Scores are exchanged
    # pre-softmax so the existing per-wave max and sum reductions still see
    # all block_n tokens and need no cross-wave reduction.
    qk_split = not decode_tr_pv and num_waves > 1 and n_subtiles % num_waves == 0
    qk_sub_per_wave = n_subtiles // num_waves if qk_split else n_subtiles
    # One 4-wide f32 C fragment per lane per subtile.
    qk_score_slots = n_subtiles * 64 * 4 if qk_split else 1

    def make_k_lds_view(k_arr, offset, shape):
        # gfx950 XOR on stride D. Prefill V overlay reuses this map;
        # decode V uses the AMD 8 KiB b64 XOR map, not this swizzle.
        if use_k32:
            layout = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(3, 3, 3)),
                offset,
                fx.make_layout(shape, (_D, 1)),
            )
            ptr = k_arr.ptr
        else:
            layout = fx.make_layout(shape, (_K_STRIDE, 1))
            ptr = k_arr.ptr + offset
        return fx.make_view(ptr, layout)

    if use_k32:

        @fx.struct
        class SharedStorage:
            kv: fx.Array[BFloat16, block_n * _D + v_elem_off, 16]
            scores: fx.Array[Float32, qk_score_slots, 16]

        _k_field, _v_field = "kv", "kv"
    else:

        @fx.struct
        class SharedStorage:
            kv: fx.Array[BFloat16, block_n * _K_STRIDE, 16]
            scores: fx.Array[Float32, qk_score_slots, 16]

        _k_field, _v_field = "kv", "kv"

    @fx.struct
    class MergeStorage:
        weights: fx.Array[Float32, 64, 16]
        denominator: fx.Array[Float32, 1, 16]

    @flyc.kernel(
        name="qsa_k2_family_a_port_split_"
        + kernel_signature(
            ps=page_size,
            bn=block_n,
            blk=block_threads,
            ns=n_splits,
            qkk=qk_k,
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def split_kernel(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        v_cache: fx.Tensor,
        indices: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        partial_out: fx.Tensor,
        partial_lse: fx.Tensor,
        out: fx.Tensor,
        n_sel: Int32,
        n_req: Int32,
        table_width: Int32,
        n_cache_blocks: Int32,
        softmax_scale_log2: Float32,
    ):
        row = Int32(gpu.block_id("x"))
        kv_h = Int32(gpu.block_id("y"))
        split = Int32(gpu.block_id("z"))
        tid = Int32(gpu.thread_id("x"))
        zero = Int32(0)
        one = Int32(1)
        page = Int32(page_size)
        thread_coord = fx.idx2crd(tid, fx.make_layout((num_waves, 64), (64, 1)))
        wave = Int32(fx.get(thread_coord, 0))
        lane = Int32(fx.get(thread_coord, 1))
        lane_m = lane % Int32(16)
        lane_kg = _idiv(lane, Int32(16))

        vec_layout = fx.make_layout(vec, 1)
        g_copy = buf_copy_atom(16, BFloat16)
        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), BFloat16)
        lds_copy64 = fx.make_copy_atom(fx.UniversalCopy64b(), BFloat16)
        kv_tile, kv_tv = fx.make_layout_tv(
            fx.make_layout((block_n, col_owners), (1, block_n)),
            fx.make_layout((1, vec), (vec, 1)),
        )
        kv_store = fx.make_tiled_copy(lds_copy, kv_tv, kv_tile).get_slice(tid)
        fx.make_tiled_copy(lds_copy64, kv_tv, kv_tile).get_slice(tid)
        q_buf = fx.rocdl.make_buffer_tensor(q, max_size=False)
        k_buf = fx.rocdl.make_buffer_tensor(k_cache)
        v_buf = fx.rocdl.make_buffer_tensor(v_cache)

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        k_arr = getattr(storage, _k_field)
        score_arr = storage.scores
        score_copy = fx.make_copy_atom(fx.UniversalCopy128b(), Float32)
        score_layout = fx.make_layout(4, 1)
        v_lds = getattr(storage, _v_field).view(
            fx.make_layout(
                (block_n, _K_STRIDE if not use_k32 else _D),
                (_K_STRIDE if not use_k32 else _D, 1),
            )
        )

        qk_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, qk_k, BFloat16))
        pv_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, BFloat16))
        qk_wave_mma = fx.make_tiled_mma(qk_mma, fx.make_layout((1, 1, 1), (0, 0, 0)))
        qk_b_atom = fx.make_copy_atom(
            fx.UniversalCopy128b() if qk_k == 32 else fx.UniversalCopy64b(),
            BFloat16,
        )
        qk_a_copy = fx.make_tiled_copy_A(qk_b_atom, qk_wave_mma).get_slice(lane)
        qk_q_copy = fx.make_tiled_copy_B(g_copy, qk_wave_mma).get_slice(lane)
        pv_wave_mma = fx.make_tiled_mma(pv_mma, fx.make_layout((1, 1, 1), (0, 0, 0)))
        pv_b_atom = fx.make_copy_atom(
            (fx.rocdl.cdna4.LDSReadTrans16_64b() if use_k32 else fx.UniversalCopy64b()),
            BFloat16,
        )
        fx.make_tiled_copy_B(pv_b_atom, pv_wave_mma).get_slice(lane)

        def qk_mfma(a_vec, b_vec, c_vec):
            qk_a = fx.make_rmem_tensor(fx.make_layout(qk_vec, 1), BFloat16)
            qk_b = fx.make_rmem_tensor(fx.make_layout(qk_vec, 1), BFloat16)
            qk_c = fx.make_rmem_tensor(fx.make_layout(4, 1), Float32)
            fx.memref_store_vec(a_vec, qk_a)
            fx.memref_store_vec(b_vec, qk_b)
            fx.memref_store_vec(c_vec, qk_c)
            fx.gemm(qk_mma, qk_c, qk_a, qk_b, qk_c)
            return fx.Vector(fx.memref_load_vec(qk_c))

        def pv_mfma(a_vec, b_vec, c_vec):
            pv_a = fx.make_rmem_tensor(fx.make_layout(4, 1), BFloat16)
            pv_b = fx.make_rmem_tensor(fx.make_layout(4, 1), BFloat16)
            pv_c = fx.make_rmem_tensor(fx.make_layout(4, 1), Float32)
            fx.memref_store_vec(a_vec, pv_a)
            fx.memref_store_vec(b_vec, pv_b)
            fx.memref_store_vec(c_vec, pv_c)
            fx.gemm(pv_mma, pv_c, pv_a, pv_b, pv_c)
            return fx.Vector(fx.memref_load_vec(pv_c))

        def amd_k_elem(n_tok, d0):
            # Match the live AMD prefill K LDS lowering. Each lane owns one
            # token and one 8xbf16 vector; D chunks are permuted in 4x4
            # groups while bits 5:6 of tid XOR the 128-bit bank address.
            d_chunk = _idiv(d0, Int32(8))
            owner = d_chunk % Int32(col_owners)
            store_tid = n_tok + owner * Int32(block_n)
            rest = _idiv(d_chunk, Int32(col_owners))
            group = _idiv(rest, Int32(k_quarter_radix))
            quarter = rest % Int32(k_quarter_radix)
            base_bytes = store_tid * Int32(16)
            base_bytes = base_bytes ^ _idiv(store_tid & Int32(0x60), Int32(2))
            base_bytes = base_bytes ^ (group * Int32(64))
            byte_offset = (
                base_bytes
                + group * Int32(k_group_slot)
                + quarter * Int32(k_quarter_stride)
            )
            return _idiv(byte_offset, Int32(2))

        def amd_decode_k_elem(n_tok, d0):
            # Live AMD decode K (BN16 / 256 threads / 8 KiB): two 128-bit
            # stores per thread. Physical bytes are
            # ``(tid*16) ^ ((tid & 0xe0)>>1)`` and that value ``^ 0x80 + 4096``.
            # Gather maps token ``tid%16`` and D-chunks ``tid//16`` / ``+16``.
            d_chunk = _idiv(d0, Int32(8))
            store_tid = (d_chunk % Int32(16)) * Int32(16) + n_tok
            base_bytes = store_tid * Int32(16)
            base_bytes = base_bytes ^ _idiv(store_tid & Int32(0xE0), Int32(2))
            hi = d_chunk >= Int32(16)
            byte_offset = hi.select(
                (base_bytes ^ Int32(0x80)) + Int32(4096), base_bytes
            )
            return _idiv(byte_offset, Int32(2))

        def amd_decode_v_pack(n_tok, d0):
            # Live AMD decode V (BN16 / 256 threads / 8 KiB): four 64-bit
            # stores per thread, the compiler split of two 8xbf16 gathers.
            # Bytes are ``A``, ``A^8``, ``(A^64)+4096``, ``(A^0x48)+4096``
            # with ``A = (tid*16) ^ ((tid & 0xe0)>>2)``.
            d_chunk = _idiv(d0, Int32(8))
            store_tid = (d_chunk % Int32(16)) * Int32(16) + n_tok
            base_bytes = store_tid * Int32(16)
            base_bytes = base_bytes ^ _idiv(store_tid & Int32(0xE0), Int32(4))
            hi = d_chunk >= Int32(16)
            half = (d0 % Int32(8)) >= Int32(4)
            lo_addr = half.select(base_bytes ^ Int32(8), base_bytes)
            hi_addr = half.select(
                (base_bytes ^ Int32(0x48)) + Int32(4096),
                (base_bytes ^ Int32(64)) + Int32(4096),
            )
            return _idiv(hi.select(hi_addr, lo_addr), Int32(2))

        def load_amd_k8(n_tok, d0):
            off = (
                amd_decode_k_elem(n_tok, d0)
                if const_expr(decode_tr_pv)
                else amd_k_elem(n_tok, d0)
            )
            src = fx.make_view(k_arr.ptr + off, fx.make_layout(8, 1))
            frag = fx.make_rmem_tensor(fx.make_layout(8, 1), BFloat16)
            fx.copy_atom_call(lds_copy, src, frag)
            return fx.Vector(fx.memref_load_vec(frag))

        def store_amd_v4(byte_addr, vec4):
            dst = fx.make_view(
                k_arr.ptr + _idiv(byte_addr, Int32(2)),
                fx.make_layout(4, 1),
            )
            frag = fx.make_rmem_tensor(fx.make_layout(4, 1), BFloat16)
            fx.memref_store_vec(vec4, frag)
            fx.copy_atom_call(lds_copy64, frag, dst)

        req = token_to_req[row]
        valid_req = (req >= zero) & (req < n_req)
        safe_req = valid_req.select(req, zero)
        total_tiles = _idiv(n_sel + Int32(block_n - 1), Int32(block_n))
        tile_start = _idiv(split * total_tiles, Int32(n_splits))
        tile_end = _idiv((split + one) * total_tiles, Int32(n_splits))
        col_start = tile_start * Int32(block_n)
        col_end_unclamped = tile_end * Int32(block_n)
        col_end = (col_end_unclamped < n_sel).select(col_end_unclamped, n_sel)

        # Form Q as the B operand of K @ Q^T. This makes the QK C fragment
        # token-major in registers, which is already the PV A fragment map.
        q_live = lane_m < Int32(_GROUP)
        q_regs = []
        for ks in range_constexpr(qk_steps):
            q_base = (row * Int32(_HQ) + kv_h * Int32(_GROUP)) * Int32(_D) + Int32(
                ks * qk_k
            )
            q_tile = fx.make_view(
                fx.get_iter(q_buf) + q_base,
                fx.make_layout((16, qk_k), (_D, 1)),
            )
            q_src = qk_q_copy.partition_S(q_tile)
            q_frag = fx.make_fragment_like(q_src)
            fx.copy(g_copy, q_src, q_frag)
            q_vec = fx.Vector(fx.memref_load_vec(q_frag))
            if const_expr(decode_tr_pv):
                # Packed cndmask. The per-element f32 round-trip packed with
                # v_perm; AMD zeros OOB heads at the load mask instead.
                q_regs.append(
                    q_live.select(q_vec, fx.Vector.filled(qk_vec, 0.0, BFloat16))
                )
            else:
                q_regs.append(
                    fx.Vector.from_elements(
                        [
                            q_live.select(q_vec[i].to(Float32), Float32(0.0)).to(
                                BFloat16
                            )
                            for i in range_constexpr(qk_vec)
                        ],
                        BFloat16,
                    )
                )

        init_acc = [fx.Vector.filled(4, 0.0, Float32) for _ in range(out_chunks)]
        init_acc.append(Float32(float("-inf")))
        init_acc.append(Float32(0.0))
        col = tid % Int32(block_n)
        chunk_owner = _idiv(tid, Int32(block_n))

        def tile_column(tile_i32):
            """This lane's column in ``tile_i32``, clamped to stay in bounds.

            Tiles past the split's last one clamp to ``col_start`` and report
            ``in_col`` false, so prefetching past the end is safe and masks off.
            """
            col_i = col_start + tile_i32 * Int32(block_n) + col
            in_col = col_i < col_end
            return in_col.select(col_i, col_start), in_col

        def load_index(tile_i32):
            """Issue a tile's index load. Not consumed by the caller."""
            safe_col, _in_col = tile_column(tile_i32)
            return indices[row, safe_col]

        def logical_page_of(tok):
            safe_tok = (tok >= zero).select(tok, zero)
            lp = _idiv(safe_tok, page)
            return lp, safe_tok - lp * page

        def load_page(tok):
            """Issue a tile's page-table load. Not consumed by the caller."""
            lp, _off = logical_page_of(tok)
            return page_table[safe_req, (lp < table_width).select(lp, zero)]

        def resolve(tok, tile_i32, phys):
            """Rebuild a tile's masks once its page load has landed."""
            _sc, in_col = tile_column(tile_i32)
            lp, page_off_i = logical_page_of(tok)
            phys_live = (phys >= zero) & (phys < n_cache_blocks)
            live = valid_req & in_col & (tok >= zero) & (lp < table_width) & phys_live
            return phys_live.select(phys, zero), page_off_i, live

        def tile_body(safe_phys, page_off_i, live, state):
            gpu.barrier()

            # One V address view for every gather path below. It has to live
            # at function scope: the frontend does not leak names assigned
            # inside a const_expr branch out to later code.
            v_row = fx.logical_divide(
                fx.slice(v_buf, (safe_phys, page_off_i, kv_h, None)), vec_layout
            )

            v_frags = []
            if const_expr(decode_tr_pv):
                for gr in range_constexpr(gather_rounds):
                    d_chunk = chunk_owner + Int32(gr * col_owners)
                    v_src = fx.slice(v_row, (None, d_chunk))
                    v_frag = fx.make_fragment_like(v_src)
                    fx.copy(g_copy, v_src, v_frag)
                    v_frags.append(v_frag)

            k_row = fx.logical_divide(
                fx.slice(k_buf, (safe_phys, page_off_i, kv_h, None)), vec_layout
            )
            for gc in range_constexpr(n_gather_chunks):
                k_frags = []
                for j in range_constexpr(gather_chunk):
                    gr = gc * gather_chunk + j
                    d_chunk = chunk_owner + Int32(gr * col_owners)
                    k_src = fx.slice(k_row, (None, d_chunk))
                    k_frag = fx.make_fragment_like(k_src)
                    fx.copy(g_copy, k_src, k_frag)
                    k_frags.append(k_frag)
                for j in range_constexpr(gather_chunk):
                    gr = gc * gather_chunk + j
                    k_vec = live.select(
                        fx.Vector(fx.memref_load_vec(k_frags[j])),
                        fx.Vector.filled(vec, 0.0, BFloat16),
                    )
                    if const_expr(decode_tr_pv):
                        base_bytes = tid * Int32(16)
                        base_bytes = base_bytes ^ _idiv(tid & Int32(0xE0), Int32(2))
                        if const_expr(gr != 0):
                            base_bytes = (base_bytes ^ Int32(0x80)) + Int32(4096)
                        k_dst = fx.make_view(
                            k_arr.ptr + _idiv(base_bytes, Int32(2)),
                            fx.make_layout(8, 1),
                        )
                        k_store_frag = fx.make_rmem_tensor(
                            fx.make_layout(8, 1), BFloat16
                        )
                        fx.memref_store_vec(k_vec, k_store_frag)
                        fx.copy_atom_call(lds_copy, k_store_frag, k_dst)
                    elif const_expr(use_k32):
                        d_chunk = chunk_owner + Int32(gr * col_owners)
                        k_dst = fx.make_view(
                            k_arr.ptr + amd_k_elem(col, d_chunk * Int32(8)),
                            fx.make_layout(8, 1),
                        )
                        k_store_frag = fx.make_rmem_tensor(
                            fx.make_layout(8, 1), BFloat16
                        )
                        fx.memref_store_vec(k_vec, k_store_frag)
                        fx.copy_atom_call(lds_copy, k_store_frag, k_dst)
                    else:
                        k_tile = make_k_lds_view(
                            k_arr,
                            Int32(gr * gather_span),
                            (block_n, gather_span),
                        )
                        k_dst = kv_store.partition_D(k_tile)
                        k_store_frag = fx.make_fragment_like(k_dst)
                        fx.memref_store_vec(k_vec, k_store_frag)
                        fx.copy(lds_copy, k_store_frag, k_dst)
            if const_expr(token_major_v):
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.rocdl.s_barrier()
            else:
                gpu.barrier()

            # V rounds issued ahead of QK, so their latency lands under the
            # MFMA block. Every round in flight costs 4 VGPRs, so when V
            # overlays K the prefetch is one chunk deep; with a separate V
            # region there is no aliasing barrier to sit behind and the whole
            # gather can ride across QK, which is where the memory-level
            # parallelism comes from.
            v_frags_pf = []
            if const_expr(use_k32) and const_expr(not decode_tr_pv):
                for gr in range_constexpr(v_pf_rounds):
                    d_chunk = chunk_owner + Int32(gr * col_owners)
                    v_src = fx.slice(v_row, (None, d_chunk))
                    v_frag = fx.make_fragment_like(v_src)
                    fx.copy(g_copy, v_src, v_frag)
                    v_frags_pf.append(v_frag)

            # Compute K @ Q^T. The transposed QK C map is token-major in each
            # lane and can feed PV A without a P-LDS or bpermute transpose.
            qk_local = []
            for ngl in range_constexpr(qk_sub_per_wave):
                if const_expr(qk_split):
                    n0 = (wave * Int32(qk_sub_per_wave) + Int32(ngl)) * Int32(16)
                else:
                    n0 = Int32(ngl * 16)
                acc4 = fx.Vector.filled(4, 0.0, Float32)
                if const_expr(use_k32):
                    n_tok = n0 + lane_m
                    a_vec = load_amd_k8(n_tok, lane_kg * Int32(8))
                    for ks in range_constexpr(qk_steps):
                        if const_expr(ks > 0):
                            a_vec = load_amd_k8(
                                n_tok, Int32(ks * qk_k) + lane_kg * Int32(8)
                            )
                        acc4 = qk_mfma(a_vec, fx.Vector(q_regs[ks]), acc4)
                else:
                    k_row_bytes = Int32(_D if use_k32 else _K_STRIDE)
                    sA = make_k_lds_view(k_arr, n0 * k_row_bytes, (16, qk_k))
                    a_src = qk_a_copy.partition_S(sA)
                    a_frag = fx.make_fragment_like(a_src)
                    fx.copy(qk_b_atom, a_src, a_frag)
                    for ks in range_constexpr(qk_steps - 1):
                        sA_n = make_k_lds_view(
                            k_arr,
                            n0 * k_row_bytes + Int32((ks + 1) * qk_k),
                            (16, qk_k),
                        )
                        a_src_n = qk_a_copy.partition_S(sA_n)
                        a_frag_n = fx.make_fragment_like(a_src_n)
                        fx.copy(qk_b_atom, a_src_n, a_frag_n)
                        acc4 = qk_mfma(
                            fx.Vector(fx.memref_load_vec(a_frag)),
                            fx.Vector(q_regs[ks]),
                            acc4,
                        )
                        a_frag = a_frag_n
                    acc4 = qk_mfma(
                        fx.Vector(fx.memref_load_vec(a_frag)),
                        fx.Vector(q_regs[qk_steps - 1]),
                        acc4,
                    )
                qk_local.append(acc4)

            if const_expr(qk_split):
                # Publish this wave's subtiles. The two barriers already in
                # the V path separate this write from the read below, so the
                # exchange costs no extra synchronization.
                for ngl in range_constexpr(qk_sub_per_wave):
                    ng_rt = wave * Int32(qk_sub_per_wave) + Int32(ngl)
                    sc_frag = fx.make_rmem_tensor(score_layout, Float32)
                    fx.memref_store_vec(qk_local[ngl], sc_frag)
                    fx.copy_atom_call(
                        score_copy,
                        sc_frag,
                        fx.make_view(
                            score_arr.ptr + ng_rt * Int32(256) + lane * Int32(4),
                            score_layout,
                        ),
                    )

            if const_expr(not split_kv_lds):
                # K and V share one MMA scratch, so every wave must finish
                # its QK reads before any lane overlays that storage with V.
                gpu.barrier()
            if const_expr(decode_tr_pv):
                for gr in range_constexpr(gather_rounds):
                    v_vec = live.select(
                        fx.Vector(fx.memref_load_vec(v_frags[gr])),
                        fx.Vector.filled(vec, 0.0, BFloat16),
                    )
                    base_bytes = tid * Int32(16)
                    base_bytes = base_bytes ^ _idiv(tid & Int32(0xE0), Int32(4))
                    a0 = base_bytes
                    a1 = base_bytes ^ Int32(8)
                    if const_expr(gr != 0):
                        a0 = (base_bytes ^ Int32(64)) + Int32(4096)
                        a1 = (base_bytes ^ Int32(0x48)) + Int32(4096)
                    lo = fx.Vector.from_elements(
                        [v_vec[i] for i in range_constexpr(4)], BFloat16
                    )
                    hi = fx.Vector.from_elements(
                        [v_vec[i + 4] for i in range_constexpr(4)], BFloat16
                    )
                    store_amd_v4(a0, lo)
                    store_amd_v4(a1, hi)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.rocdl.s_barrier()
            elif const_expr(use_k32):
                store_elem = (
                    (col % Int32(32)) * Int32(4)
                    + chunk_owner * Int32(128)
                    + _idiv(col, Int32(32)) * Int32(32 * _D)
                )
                store_addr = Int32((store_elem + Int32(v_elem_off)) * Int32(2))
                for gc in range_constexpr(n_gather_chunks):
                    if const_expr(gc < v_pf_chunks):
                        base = gc * gather_chunk
                        v_chunk = v_frags_pf[base : base + gather_chunk]
                    else:
                        v_chunk = []
                        for j in range_constexpr(gather_chunk):
                            d_chunk = chunk_owner + Int32(
                                (gc * gather_chunk + j) * col_owners
                            )
                            v_src = fx.slice(v_row, (None, d_chunk))
                            v_frag = fx.make_fragment_like(v_src)
                            fx.copy(g_copy, v_src, v_frag)
                            v_chunk.append(v_frag)
                    for j in range_constexpr(gather_chunk):
                        gr = gc * gather_chunk + j
                        v_vec = live.select(
                            fx.Vector(fx.memref_load_vec(v_chunk[j])),
                            fx.Vector.filled(vec, 0.0, BFloat16),
                        )
                        lo = fx.Vector.from_elements(
                            [v_vec[i] for i in range_constexpr(4)], BFloat16
                        ).bitcast(fx.Int64)[0]
                        hi = fx.Vector.from_elements(
                            [v_vec[i + 4] for i in range_constexpr(4)], BFloat16
                        ).bitcast(fx.Int64)[0]
                        # One base VGPR. Round gr advances d by col_owners
                        # chunks, i.e. col_owners * 256 B, which is
                        # col_owners/2 st64 units; hi is the d+4 half at a
                        # fixed +16 st64 (8192 B). Matches AMD's offset pairs.
                        _ds_write2st64_b64(
                            store_addr,
                            lo,
                            hi,
                            offset0=gr * v_round_st64,
                            offset1=gr * v_round_st64 + 16,
                        )
                gpu.barrier()
            elif const_expr(not use_k32):
                for gr in range_constexpr(gather_rounds):
                    d_chunk = chunk_owner + Int32(gr * col_owners)
                    v_src = fx.slice(v_row, (None, d_chunk))
                    v_frag = fx.make_fragment_like(v_src)
                    fx.copy(g_copy, v_src, v_frag)
                    v_vec = live.select(
                        fx.Vector(fx.memref_load_vec(v_frag)),
                        fx.Vector.filled(vec, 0.0, BFloat16),
                    )
                    fx.memref_store_vec(v_vec, v_frag)
                    v_tile = fx.make_view(
                        fx.get_iter(v_lds) + Int32(gr * gather_span),
                        fx.make_layout((block_n, gather_span), (_K_STRIDE, 1)),
                    )
                    v_dst = kv_store.partition_D(v_tile)
                    v_store_frag = fx.make_fragment_like(v_dst)
                    fx.memref_store_vec(v_vec, v_store_frag)
                    fx.copy(lds_copy, v_store_frag, v_dst)
                gpu.barrier()

            if const_expr(qk_split):
                qk_accs = []
                for ng in range_constexpr(n_subtiles):
                    sc_frag = fx.make_rmem_tensor(score_layout, Float32)
                    fx.copy_atom_call(
                        score_copy,
                        fx.make_view(
                            score_arr.ptr + Int32(ng * 256) + lane * Int32(4),
                            score_layout,
                        ),
                        sc_frag,
                    )
                    qk_accs.append(fx.Vector(fx.memref_load_vec(sc_frag)))
            else:
                qk_accs = qk_local

            m_prev = Float32(state[out_chunks])
            l_prev = Float32(state[out_chunks + 1])
            tile_max = _neg_inf()
            scores = []
            score_lives = []
            live_bits = Int64(0)
            if const_expr(decode_tr_pv):
                live_bits = Int64(fx.rocdl.ballot(Int64.ir_type, live))
            for ng in range_constexpr(n_subtiles):
                sc = []
                lives = []
                for i in range_constexpr(4):
                    n = Int32(ng * 16) + lane_kg * Int32(4) + Int32(i)
                    if const_expr(decode_tr_pv):
                        score_live = ((live_bits >> Int64(n)) & Int64(1)) != Int64(0)
                    else:
                        score_live = (
                            gpu.shuffle_idx(live.select(one, zero), n, Int32(64))
                            != zero
                        )
                    score = qk_accs[ng][i] * softmax_scale_log2
                    score = score_live.select(score, _neg_inf())
                    sc.append(score)
                    lives.append(score_live)
                    tile_max = tile_max.maximumf(score)
                scores.append(sc)
                score_lives.append(lives)
            tile_peer = tile_max.shuffle_xor(Int32(32), Int32(64))
            tile_max = tile_max.maximumf(tile_peer)
            tile_peer = tile_max.shuffle_xor(Int32(16), Int32(64))
            tile_max = tile_max.maximumf(tile_peer)
            m_new = m_prev.maximumf(tile_max)
            alpha = _exp2(m_prev - m_new)
            p_sum = Float32(0.0)
            p_vecs = []
            for ng in range_constexpr(n_subtiles):
                probs = []
                for i in range_constexpr(4):
                    p = score_lives[ng][i].select(
                        _exp2(scores[ng][i] - m_new), Float32(0.0)
                    )
                    p_sum = p_sum + p
                    if const_expr(decode_tr_pv):
                        probs.append(p)
                    else:
                        probs.append(p.to(BFloat16))
                if const_expr(decode_tr_pv):
                    # Pack P as the MFMA-B fragment directly. Building four
                    # scalar bf16 elements made LLVM add two v_perm ops.
                    p_lo = fx.rocdl.cvt_pk_bf16_f32(probs[0], probs[1])
                    p_hi = fx.rocdl.cvt_pk_bf16_f32(probs[2], probs[3])
                    p_vecs.append(
                        fx.Vector.from_elements(
                            [Int32(p_lo), Int32(p_hi)], Int32
                        ).bitcast(BFloat16)
                    )
                else:
                    p_vecs.append(fx.Vector.from_elements(probs, BFloat16))
            p_peer = p_sum.shuffle_xor(Int32(32), Int32(64))
            p_sum = p_sum + p_peer
            p_peer = p_sum.shuffle_xor(Int32(16), Int32(64))
            p_sum = p_sum + p_peer
            l_new = l_prev * alpha + p_sum
            h0 = lane_kg * Int32(4)
            if const_expr(decode_tr_pv):
                # Decode matches live AMD's second dot: V is MFMA A and P is
                # B, so output C is four D values on the head lane. Alpha is
                # already scalar on that lane; no head gather is required.
                alpha4 = fx.Vector.from_elements([alpha], Float32).broadcast_to(4)
            else:
                alpha4 = fx.Vector.from_elements(
                    [
                        gpu.shuffle_idx(alpha, h0 + Int32(i), Int32(64))
                        for i in range_constexpr(4)
                    ],
                    Float32,
                )
            next_acc = []
            for c in range_constexpr(out_chunks):
                d = wave * Int32(_D // num_waves) + Int32(c * 16) + lane_m
                acc4 = fx.Vector(state[c]) * alpha4
                v_ops = []
                for ng in range_constexpr(n_subtiles):
                    n0 = Int32(ng * 16) + lane_kg * Int32(4)
                    if const_expr(use_k32):
                        d_base = wave * Int32(_D // num_waves) + Int32(c * 16)
                        if const_expr(decode_tr_pv):
                            src_n = (
                                Int32(ng * 16)
                                + lane_kg * Int32(4)
                                + _idiv(lane_m, Int32(4))
                            )
                            src_d = d_base + (lane_m % Int32(4)) * Int32(4)
                            src = fx.make_view(
                                k_arr.ptr + amd_decode_v_pack(src_n, src_d),
                                fx.make_layout(4, 1),
                            )
                            b_frag = fx.make_rmem_tensor(fx.make_layout(4, 1), BFloat16)
                            fx.copy_atom_call(pv_b_atom, src, b_frag)
                            v_ops.append(b_frag)
                        else:
                            # ds_read_tr16 transposes a 16-lane x 4-bf16
                            # footprint. Lane t must source token t//4 and
                            # dimensions (t%4)*4 so its output is four tokens
                            # at this lane's output dimension.
                            src_n = (
                                Int32(ng * 16)
                                + lane_kg * Int32(4)
                                + _idiv(lane_m, Int32(4))
                            )
                            src_d = d_base + (lane_m % Int32(4)) * Int32(4)
                            src_off = (
                                Int32(v_elem_off)
                                + src_d % Int32(4)
                                + (src_n % Int32(32)) * Int32(4)
                                + _idiv(src_d, Int32(8)) * Int32(128)
                                + (_idiv(src_d, Int32(4)) % Int32(2)) * Int32(16 * _D)
                                + _idiv(src_n, Int32(32)) * Int32(32 * _D)
                            )
                            src = fx.make_view(
                                k_arr.ptr + src_off, fx.make_layout(4, 1)
                            )
                            b_frag = fx.make_rmem_tensor(fx.make_layout(4, 1), BFloat16)
                            fx.copy_atom_call(pv_b_atom, src, b_frag)
                            v_ops.append(b_frag)
                    else:
                        v_ops.append(
                            fx.Vector.from_elements(
                                [
                                    v_lds[n0, d],
                                    v_lds[n0 + one, d],
                                    v_lds[n0 + Int32(2), d],
                                    v_lds[n0 + Int32(3), d],
                                ],
                                BFloat16,
                            )
                        )
                for ng in range_constexpr(n_subtiles):
                    if const_expr(use_k32):
                        v_vec = fx.Vector(fx.memref_load_vec(v_ops[ng]))
                    else:
                        v_vec = v_ops[ng]
                    if const_expr(decode_tr_pv):
                        # V @ P^T: A owns D rows and B owns query heads. The
                        # resulting C fragment is D-in-vector/head-on-lane.
                        acc4 = pv_mfma(v_vec, p_vecs[ng], acc4)
                    else:
                        acc4 = pv_mfma(p_vecs[ng], v_vec, acc4)
                next_acc.append(acc4)
            return next_acc + [m_new, l_new]

        # Software-pipelined address resolution. Each iteration issues the
        # index two tiles ahead and the page one tile ahead; neither is read
        # in the iteration that issues it, so both round trips are covered by
        # a whole tile body instead of stalling in front of the K/V gather.
        # The three carried registers ride behind the accumulator so the
        # epilogue's ``results`` indices are unchanged.
        tok0 = load_index(Int32(0))
        tok1 = load_index(Int32(1))
        phys0 = load_page(tok0)
        n_tiles = tile_end - tile_start
        for tile64, state in range(
            fx.Int64(0),
            fx.Int64(n_tiles),
            fx.Int64(1),
            init=init_acc + [tok0, tok1, phys0],
        ):
            t = Int32(tile64)
            tok_cur = Int32(state[out_chunks + 2])
            tok_n1 = Int32(state[out_chunks + 3])
            phys_cur = Int32(state[out_chunks + 4])
            tok_n2 = load_index(t + Int32(2))
            phys_n1 = load_page(tok_n1)
            safe_phys, page_off_i, live = resolve(tok_cur, t, phys_cur)
            acc = tile_body(safe_phys, page_off_i, live, state)
            results = yield acc + [tok_n1, tok_n2, phys_n1]

        m_final = Float32(results[out_chunks])
        l_final = Float32(results[out_chunks + 1])
        if const_expr(decode_tr_pv):
            # PV C now keeps one query head on lane_m and four contiguous D
            # rows in each VGPR vector. Store that fragment directly.
            if lane_m < Int32(_GROUP):
                head = kv_h * Int32(_GROUP) + lane_m
                has = l_final > Float32(0.0)
                for c in range_constexpr(out_chunks):
                    d_base = wave * Int32(_D // num_waves) + Int32(c * 16)
                    for i in range_constexpr(4):
                        d = d_base + lane_kg * Int32(4) + Int32(i)
                        value = has.select(
                            fx.Vector(results[c])[i] / l_final, Float32(0.0)
                        )
                        if n_splits == 1:
                            out[row, head, d] = value.to(BFloat16)
                        else:
                            partial_out[split, row, head, d] = value
        else:
            h0 = lane_kg * Int32(4)
            l4 = fx.Vector.from_elements(
                [
                    gpu.shuffle_idx(l_final, h0 + Int32(i), Int32(64))
                    for i in range_constexpr(4)
                ],
                Float32,
            )
            for i in range_constexpr(4):
                local_head = lane_kg * Int32(4) + Int32(i)
                if local_head < Int32(_GROUP):
                    head = kv_h * Int32(_GROUP) + local_head
                    den = l4[i]
                    has = den > Float32(0.0)
                    for c in range_constexpr(out_chunks):
                        d = wave * Int32(_D // num_waves) + Int32(c * 16) + lane_m
                        value = has.select(fx.Vector(results[c])[i] / den, Float32(0.0))
                        if n_splits == 1:
                            out[row, head, d] = value.to(BFloat16)
                        else:
                            partial_out[split, row, head, d] = value
        if n_splits > 1 and lane_kg == zero and lane_m < Int32(_GROUP):
            head = kv_h * Int32(_GROUP) + lane_m
            den = l_final
            has = den > Float32(0.0)
            lse = has.select(
                m_final + fxmath.log(den) * Float32(_LOG2E),
                Float32(_LSE_EMPTY),
            )
            partial_lse[split, row, head] = lse

    @flyc.kernel(
        name="qsa_k2_family_a_port_merge_"
        + kernel_signature(ns=n_splits, blk=128, d=_D),
        known_block_size=[128, 1, 1],
    )
    def merge_kernel(
        partial_out: fx.Tensor,
        partial_lse: fx.Tensor,
        out: fx.Tensor,
    ):
        row = Int32(gpu.block_id("x"))
        head = Int32(gpu.block_id("y"))
        tid = Int32(gpu.thread_id("x"))
        lane = tid % Int32(64)
        zero_f = Float32(0.0)
        split_live = lane < Int32(n_splits)
        safe_split = split_live.select(lane, Int32(0))
        lse = partial_lse[safe_split, row, head]
        lse = split_live.select(lse, Float32(_LSE_EMPTY))
        lse_max = lse
        for sh in (32, 16, 8, 4, 2, 1):
            lse_max = lse_max.maximumf(lse_max.shuffle_xor(Int32(sh), Int32(64)))
        live = split_live & (lse > Float32(_LSE_EMPTY))
        weight = live.select(_exp2(lse - lse_max), zero_f)
        den = weight
        for sh in (32, 16, 8, 4, 2, 1):
            den = den + den.shuffle_xor(Int32(sh), Int32(64))

        merge_storage = fx.SharedAllocator().allocate(MergeStorage).peek()
        weights = merge_storage.weights.view(fx.make_layout(64, 1))
        denominator = merge_storage.denominator.view(fx.make_layout(1, 1))
        if tid < Int32(64):
            weights[tid] = weight
        if tid == Int32(0):
            denominator[0] = den
        gpu.barrier()

        for di in range_constexpr(2):
            d = tid + Int32(di * 128)
            merged = zero_f
            for s in range_constexpr(n_splits):
                part = Float32(partial_out[Int32(s), row, head, d])
                merged = merged + weights[Int32(s)] * part
            den_f = denominator[0]
            value = (den_f > zero_f).select(merged / den_f, zero_f)
            out[row, head, d] = value.to(BFloat16)

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        v_cache: fx.Tensor,
        indices: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        partial_out: fx.Tensor,
        partial_lse: fx.Tensor,
        out: fx.Tensor,
        n_sel: Int32,
        n_req: Int32,
        table_width: Int32,
        n_cache_blocks: Int32,
        softmax_scale_log2: Float32,
        rows: Int32,
        kv_heads: Int32,
        n_heads: Int32,
        stream: fx.Stream,
    ):
        split_kernel(
            q,
            k_cache,
            v_cache,
            indices,
            page_table,
            token_to_req,
            partial_out,
            partial_lse,
            out,
            n_sel,
            n_req,
            table_width,
            n_cache_blocks,
            softmax_scale_log2,
        ).launch(
            grid=(rows, kv_heads, n_splits),
            block=(block_threads, 1, 1),
            stream=stream,
        )
        if n_splits > 1:
            merge_kernel(partial_out, partial_lse, out).launch(
                grid=(rows, n_heads, 1),
                block=(128, 1, 1),
                stream=stream,
            )

    launch.compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
    }
    return launch


@lru_cache(maxsize=32)
def _plan(
    page_size: int,
    use_k32: bool,
    block_n: int,
    block_threads: int,
    n_splits: int,
):
    return build_qsa_k2_family_a_module(
        page_size, use_k32, block_n, block_threads, n_splits
    )


def qsa_k2_family_a_serves(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    page_table: torch.Tensor,
) -> str | None:
    """Why this K2 kernel cannot serve these tensors, or None if it can."""
    gqa = FAMILY_A_GQA
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        return f"q and caches must be bfloat16, got {q.dtype} and {k_cache.dtype}"
    if v_cache.dtype != k_cache.dtype or v_cache.shape != k_cache.shape:
        return "v_cache must match k_cache dtype and shape"
    if q.dim() != 3 or q.shape[1] != gqa.n_heads or q.shape[2] != gqa.head_dim:
        return f"q must be [M, {gqa.n_heads}, {gqa.head_dim}], got {tuple(q.shape)}"
    if k_cache.dim() != 4:
        return f"k_cache must be [pages, page_size, H, D], got {tuple(k_cache.shape)}"
    if k_cache.shape[2] != gqa.kv_heads or k_cache.shape[3] != gqa.head_dim:
        return (
            f"k_cache KV/D must be ({gqa.kv_heads}, {gqa.head_dim}), "
            f"got {k_cache.shape[2:]}"
        )
    if indices.dim() != 2 or indices.shape[0] != q.shape[0]:
        return "indices must be [M, W]"
    if indices.dtype != torch.int32:
        return f"indices must be int32, got {indices.dtype}"
    if page_table.dim() != 2 or page_table.dtype != torch.int32:
        return (
            f"page_table must be int32 [n_req, n_pages], got {tuple(page_table.shape)}"
        )
    return None


def qsa_k2_family_a(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Write family A sparse GQA ``o [M, 24, 256]`` from paged K/V."""
    reason = qsa_k2_family_a_serves(q, k_cache, v_cache, indices, page_table)
    if reason is not None:
        raise ValueError(f"[FlyDSL qsa_k2_family_a] {reason}")
    rows = q.shape[0]
    if token_to_req.shape != (rows,) or token_to_req.dtype != torch.int32:
        raise ValueError(f"token_to_req must be int32 [{rows}]")
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype:
        raise ValueError(f"out must match q, got {tuple(out.shape)} {out.dtype}")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    tensors = (q, k_cache, v_cache, indices, page_table, token_to_req, out)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("every tensor must be on the GPU")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("every tensor must be on the same GPU")
    if softmax_scale is None:
        softmax_scale = _DEFAULT_SCALE
    if not rows or not indices.shape[1]:
        return out.zero_()

    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    indices = indices.contiguous()
    page_table = page_table.contiguous()
    token_to_req = token_to_req.contiguous()
    page_size = k_cache.shape[1]
    use_k32 = torch.cuda.get_device_properties(q.device).gcnArchName.startswith(
        "gfx950"
    )
    n_sel = int(indices.shape[1])
    block_n, block_threads, n_splits = _launch_config(rows, n_sel)
    if n_splits == 1:
        partial_out = out
        partial_lse = out
    else:
        partial_out = torch.empty(
            (n_splits, rows, _HQ, _D), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (n_splits, rows, _HQ), dtype=torch.float32, device=q.device
        )

    _run_compiled(
        _plan(page_size, use_k32, block_n, block_threads, n_splits),
        q,
        k_cache,
        v_cache,
        indices,
        page_table,
        token_to_req,
        partial_out,
        partial_lse,
        out,
        n_sel,
        int(page_table.shape[0]),
        int(page_table.shape[1]),
        int(k_cache.shape[0]),
        float(softmax_scale * _LOG2E),
        int(rows),
        int(_HK),
        int(_HQ),
        torch.cuda.current_stream(q.device),
    )
    return out

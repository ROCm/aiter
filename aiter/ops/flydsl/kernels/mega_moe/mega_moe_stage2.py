# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: B023, I001
"""Fused GEMM2 and weighted cross-rank P2P scatter."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.protocol import construct_from_ir_values, extract_to_ir_values
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Int8, T
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.aot_launch import aot_factory

from .. import communication_ops_utils as comm_ops
from ..mxfp4_gemm_common import _fabs_f32 as fabs_f32
from ..mxfp4_gemm_common import (
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)
from ..tensor_shim import (
    _preload_compiled,
    _run_compiled,
    buf_copy_store,
    ptr_buf_tensor,
)

from .gemm2 import (
    _resolve_g2_knobs,
    _spart_output_tile_index,
    gemm2_compute_v2,
    issue_a_load_lds_dt,
    kStages,
)

_BUFFER_OFFSET_ABI_BYTES = 1 << 31
# sc0 sc1 cache policy (system coherent): a store is written through to memory, visible to other CUs and peers.
_SC0_SC1_CACHE = 17


@flyc.jit
def _fp8_scale_for_leader(is_leader, local_max):
    e8m0 = fx.Int32(0)
    if is_leader:
        max_bits = local_max.bitcast(fx.Int32)
        working_bits = (local_max * fx.Int32(0x3B124925).bitcast(fx.Float32)).bitcast(
            fx.Int32
        )
        mantissa = working_bits & fx.Int32(0x7FFFFF)
        biased_exp = (working_bits >> fx.Int32(23)) & fx.Int32(0xFF)
        e8m0 = (mantissa != fx.Int32(0)).select(biased_exp + fx.Int32(1), biased_exp)
        e8m0 = (e8m0 > fx.Int32(0xFF)).select(fx.Int32(0xFF), e8m0)
        e8m0 = (max_bits == fx.Int32(0)).select(fx.Int32(0), e8m0)
    return e8m0


def _inv_e8m0_scale(e8m0):
    """1 / 2^(e8m0 - 127) as an f32: the exact power of two 2^(127 - e8m0), for E8M0 scales in [0, 254]."""
    return ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(fx.Float32)


def _pack_fp8_e4m3(vals, e8m0):
    """Pack ``vals / 2^(e8m0 - 127)`` as E4M3 bytes, four per i32 word (RNE, as v_cvt_scalef32_pk_fp8_f32).

    Multiplying by the exact power-of-two inverse scale first (one v_pk_mul_f32 per pair) matches the scaled
    convert bit for bit for finite inputs. The first convert of a word seeds its tied destination with the
    scaled value it consumes (the word_sel=True convert then overwrites the other half), so no zero seed is
    materialized.
    """
    scaled = fx.Vector.from_elements(vals, fx.Float32) * _inv_e8m0_scale(e8m0)
    words = []
    for word in range_constexpr(len(vals) // 4):
        v = [fx.Float32(scaled[word * 4 + q]) for q in range_constexpr(4)]
        lo = fx.rocdl.cvt_pk_fp8_f32(T.i32, v[0], v[1], v[0].bitcast(fx.Int32), False)
        words.append(fx.Int32(fx.rocdl.cvt_pk_fp8_f32(T.i32, v[2], v[3], lo, True)))
    return fx.Vector.from_elements(words, fx.Int32)


def _pack_offset_u8(vals, e8m0):
    """Pack ``round(vals / 2^(e8m0 - 127)) + 128`` as u8 bytes, four per i32 word, as v_cvt_pk_u8_f32 does.

    x = v * inv_scale + 128 lies in [1, 255] (the scale bounds |v| / scale by 127), so the RNE add of 2^23
    leaves round_to_nearest_even(x) in the low byte, exactly the v_cvt_pk_u8_f32 result (whose [0, 255] clamp
    never triggers here); v_perm_b32 gathers the four low bytes.
    """
    inv_scale = _inv_e8m0_scale(e8m0)
    n = len(vals)
    # One v_pk_fma_f32 per pair: v * inv_scale is exact (a power-of-two scaling of a finite f32 that stays below
    # 128 in magnitude), so the fused multiply-add rounds once exactly where the separate add rounded.
    x = fx.Vector(
        fx.math.fma(
            fx.Vector.from_elements(vals, fx.Float32),
            fx.Vector.from_elements([inv_scale] * n, fx.Float32),
            fx.Vector.filled(n, 128.0, fx.Float32),
        )
    )
    y = (x + fx.Float32(8388608.0)).bitcast(fx.Int32)
    words = []
    for word in range_constexpr(len(vals) // 4):
        b = [fx.Int32(y[word * 4 + q]) for q in range_constexpr(4)]
        # Selector bytes 0-3 take src_lo, 4-7 src_hi, 0x0C a zero byte.
        lo = fx.Int32(fx.rocdl.perm_b32(b[1], b[0], fx.Int32(0x0C0C0400)))
        hi = fx.Int32(fx.rocdl.perm_b32(b[3], b[2], fx.Int32(0x0C0C0400)))
        words.append((hi << fx.Int32(16)) + lo)
    return fx.Vector.from_elements(words, fx.Int32)


def _fp8_quant_1x32(vals, lane, group_lane=None, leader_gate=None):
    """Quantize this lane's ``vals`` to FP8 with one E8M0 scale per 32-wide lane group.

    The ``32 // len(vals)`` adjacent lanes of a group share the |max|; the group's
    leader (``group_lane``, default ``lane``, a multiple of the group size, and
    ``leader_gate`` if given) derives the scale and broadcasts it.  Returns the E8M0
    scale, the packed FP8 words and the leader predicate.
    """
    scatter_vec = len(vals)
    local_max = fabs_f32(vals[0])
    for q in range_constexpr(1, scatter_vec):
        local_max = local_max.maximumf(fabs_f32(vals[q]))
    scale_group_lanes = 32 // scatter_vec
    # Lowers to v_mov_b32_dpp quad_perm (xor 1, 2) + v_maximum3: no ds_bpermute address VGPRs or lgkmcnt waits.
    local_max = fx.coop.warp_max(local_max, width=scale_group_lanes)
    leader_lane = lane & fx.Int32(~(scale_group_lanes - 1))
    if group_lane is None:
        group_lane = lane
    is_leader = (group_lane & fx.Int32(scale_group_lanes - 1)) == fx.Int32(0)
    if leader_gate is not None:
        is_leader = leader_gate & is_leader
    leader_e8m0 = _fp8_scale_for_leader(is_leader, local_max)
    e8m0 = fx.Int32(fx.shuffle_idx(leader_e8m0, leader_lane, 64))
    packed_words = _pack_fp8_e4m3(vals, e8m0)
    return e8m0, packed_words, is_leader


def _fp8_scale_e8m0(local_max, qmax_inv_bits=0x3B124925):
    """Branch-free _fp8_scale_for_leader: the E8M0 scale of a block whose absolute max is local_max
    (the smallest power of two with local_max / scale <= qmax; qmax_inv_bits = f32 bits of 1 / qmax, 448 default).
    """
    max_bits = local_max.bitcast(fx.Int32)
    working_bits = (local_max * fx.Int32(qmax_inv_bits).bitcast(fx.Float32)).bitcast(
        fx.Int32
    )
    mantissa = working_bits & fx.Int32(0x7FFFFF)
    biased_exp = (working_bits >> fx.Int32(23)) & fx.Int32(0xFF)
    e8m0 = (mantissa != fx.Int32(0)).select(biased_exp + fx.Int32(1), biased_exp)
    e8m0 = (e8m0 > fx.Int32(0xFF)).select(fx.Int32(0xFF), e8m0)
    return (max_bits == fx.Int32(0)).select(fx.Int32(0), e8m0)


def _fp8_quant_1x32_lean(vals, route_i8=False):
    """_fp8_quant_1x32 for 8 values a lane without LDS round trips: the 4-lane |max| is a width-4 warp_max and every
    lane derives the E8M0 scale itself. ``route_i8``: the values are packed as offset int8 (u8 = q + 128,
    q = round(v / scale) in [-127, 127], max / scale <= 127) instead of E4M3. Returns the scale and the packed
    words."""
    assert len(vals) == 8
    local_max = fabs_f32(vals[0])
    for q in range_constexpr(1, 8):
        local_max = local_max.maximumf(fabs_f32(vals[q]))
    local_max = fx.coop.warp_max(local_max, width=4)
    e8m0 = _fp8_scale_e8m0(local_max, 0x3C010204 if route_i8 else 0x3B124925)
    if const_expr(route_i8):
        return e8m0, _pack_offset_u8(vals, e8m0)
    return e8m0, _pack_fp8_e4m3(vals, e8m0)


def _pack_fp4_e2m1(vals, e8m0):
    """Pack ``vals / 2^(e8m0 - 127)`` as E2M1, eight per i32 word (two per byte, low nibble first)."""
    block_scale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
    words = []
    for word in range_constexpr(len(vals) // 8):
        packed = fx.Int32(0)
        for pair in range_constexpr(4):
            value = word * 8 + pair * 2
            packed = fx.rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, packed, vals[value], vals[value + 1], block_scale, pair
            )
        words.append(fx.Int32(packed))
    return words


def _fp4_quant_1x32(vals, lane):
    """_fp8_quant_1x32 for MXFP4: the E8M0 scale is the smallest power of two with |max| / scale <= 6
    (AITER's MXFP4 rounding, ceil_pow2(amax / 6)). Every lane of a 32-wide group derives it from the
    group's |max|. Returns the scale, the packed E2M1 words (a list) and the leader predicate.
    """
    scatter_vec = len(vals)
    local_max = fabs_f32(vals[0])
    for q in range_constexpr(1, scatter_vec):
        local_max = local_max.maximumf(fabs_f32(vals[q]))
    scale_group_lanes = 32 // scatter_vec
    local_max = fx.coop.warp_max(local_max, width=scale_group_lanes)
    is_leader = (lane & fx.Int32(scale_group_lanes - 1)) == fx.Int32(0)
    e8m0 = _fp8_scale_e8m0(local_max, 0x3E2AAAAB)
    return e8m0, _pack_fp4_e2m1(vals, e8m0), is_leader


class _LdsTable:
    """One typed table of Stage2Meta: ``t[i]`` loads row ``i``, ``t + i`` is its pointer (for ``fx.ptr_store``).

    Addresses are i32 byte arithmetic from the pool base, as the GEMM2 helpers address LDS, so the backend
    folds the table offset into the ``ds_*`` immediate and pairs neighbouring tables (``ds_read2``); a typed
    ``add_offset`` chain off the pool pointer measurably does not (it splits those reads and adds SALU work).
    """

    def __init__(self, base_i32, byte_off, elem_ty, align):
        self._base, self._off, self._ty, self._align = (
            base_i32,
            byte_off,
            elem_ty,
            align,
        )
        self._bytes = elem_ty.width // 8

    def at_byte(self, byte_off):
        """Pointer at a byte offset into the table."""
        return lds_typed_ptr(
            self._base + fx.Int32(self._off) + byte_off, self._ty, align=self._align
        )

    def __add__(self, i):
        return self.at_byte(fx.Int32(i) * fx.Int32(self._bytes))

    def __getitem__(self, i):
        return fx.ptr_load(self + i)


class Stage2Meta:
    """Typed LDS views of the Stage2 epilogue's route metadata, which follows the GEMM2 compute region of
    one LDS pool: ``packed`` route ids (i32) and ``weight`` (f32) per tile row, the ``peer`` combine-input
    base table (i64 per rank) and, for a publishing kernel, the close ``flag`` word.

    The views hang off the pool's own base address (``ptrtoint`` of its SharedAllocator pointer), so no
    access assumes the pool sits at LDS address 0. That i32 is the only IR value, so a Stage2Meta crosses
    rewritten ``if``/loop regions like a scalar.
    """

    def __init__(self, pool_base_i32, packed_off, weight_off, peer_off, flag_off=None):
        self._base = fx.Int32(pool_base_i32)
        self._offs = (packed_off, weight_off, peer_off, flag_off)

    @classmethod
    def from_pool(cls, pool_ptr, packed_off, weight_off, peer_off, flag_off=None):
        """Views over the SharedAllocator pool at ``pool_ptr``."""
        return cls(fx.ptrtoint(pool_ptr), packed_off, weight_off, peer_off, flag_off)

    def __extract_to_ir_values__(self):
        return extract_to_ir_values(self._base)

    @classmethod
    def __construct_from_ir_values__(cls, values, exemplar=None):
        obj = object.__new__(cls)
        obj._base = construct_from_ir_values(
            type(exemplar._base), exemplar._base, values
        )
        obj._offs = exemplar._offs
        return obj

    @property
    def packed(self):
        return _LdsTable(self._base, self._offs[0], T.i32, 4)

    @property
    def weight(self):
        return _LdsTable(self._base, self._offs[1], T.f32, 4)

    @property
    def peer(self):
        return _LdsTable(self._base, self._offs[2], T.i64, 8)

    @property
    def flag(self):
        return _LdsTable(self._base, self._offs[3], T.i32, 4)

    def rows_from(self, row0):
        """The same views with the per-row tables starting at tile row ``row0``."""
        packed_off, weight_off, peer_off, flag_off = self._offs
        obj = object.__new__(Stage2Meta)
        obj._base = self._base
        obj._offs = (packed_off + row0 * 4, weight_off + row0 * 4, peer_off, flag_off)
        return obj

    def peer_pair(self, pe):
        """Peer ``pe``'s base as its (lo, hi) i32 words, one 8-byte LDS read."""
        v = fx.Vector(
            lds_vec_load(
                self._base,
                fx.Int32(self._offs[2]) + pe * fx.Int32(8),
                fx.Vector.make_type(2, fx.Int32),
                fx.Int32,
                align=8,
            )
        )
        return fx.Int32(v[0]), fx.Int32(v[1])


# fmt: off
def p2p_scatter_epilog(lds_acc_base, accm, n_block_idx, wave, lane, *, N_OUT, BM, BN, npes, topk,
    log2_max_tok, mask_max_tok, recv_cap, comb_inp_nbytes, meta, g2_bf16_lds=False, p2p_quant_type="none",
    scatter_vec=8, NW=4, store_cm=2, fp8_store_cm=None, fp8_rows=None, elem_view_stores=False, lean=0, tok_off=None, bf16_head_tokens=0,
    row_nbytes=None, scale_stage=None, epi_lean=False, route_i8=False):
# fmt: on
    """CShuffle one GEMM2 tile into weighted BF16 rows and scatter them to peers.

    ``meta`` (Stage2Meta) holds the tile's route rows and peer bases.
    ``NW`` is the wave count: each wave owns ``BN // NW`` accumulator columns and
    scatters rows ``wave + row_iter * NW``.  ``store_cm`` is the cache policy of the
    BF16 row stores (2 = NT) and ``fp8_store_cm`` (default ``store_cm``) that of the
    FP8 payload and scale stores.  ``p2p_quant_type="dynamic"``: the wave-uniform
    ``fp8_rows`` picks FP8 (1x32 E8M0) or BF16 rows at run time; both use the BF16
    row stride.  ``elem_view_stores`` addresses the peer stores through
    an element-stride view (overlapping units), as the fused kernels do.

    Fused-kernel knobs (all off by default):
    ``lean``: lane l reads row (wave + NW l)'s packed id and weight once; each row
    picks them with readlane (2: lane l also reads that row's peer base).
    ``tok_off`` (needs lean 2): byte offset from a peer's combine input to this
    launch's per-token counters; each wave adds one to its rows' tokens there once
    its stores landed.
    ``bf16_head_tokens`` (dynamic): rows of destination tokens below it stay BF16
    even when ``fp8_rows`` picks FP8.
    ``row_nbytes`` overrides the receiver row stride.
    ``scale_stage = (AddrBuffer, first 16-byte group)`` writes the FP8 row scales of
    this tile to rank-local 16-byte groups (one per row) instead of to the peers.
    ``epi_lean`` (FP8): lean ids / weights (as ``lean`` 1, with an offset opaque per
    unit so it is not hoisted out of the caller's loop), the 4-lane scale max as a
    width-4 warp_max with every lane deriving the scale itself (_fp8_quant_1x32_lean) and a
    branch-free staged scale store (non-leaders store out of bounds).
    ``route_i8`` (epi_lean): offset int8 row values instead of E4M3.
    """
    kMChunks = BM // 16
    numAccN = (BN // NW) // 16
    wave_n = BN // NW
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)
    lds_base_bf16 = (
        lds_typed_ptr(lds_acc_base, T.bf16, align=2)
        if const_expr(g2_bf16_lds)
        else None
    )

    dynamic = p2p_quant_type == "dynamic"
    quant_fp8 = p2p_quant_type == "fp8_blockwise_1x32"
    quant_fp4 = p2p_quant_type == "fp4_blockwise_1x32"
    if p2p_quant_type not in ("none", "fp8_blockwise_1x32", "fp4_blockwise_1x32", "dynamic"):
        raise ValueError(f"unsupported p2p_quant_type={p2p_quant_type!r}")
    assert not (dynamic and g2_bf16_lds)
    assert not quant_fp4 or not (lean or epi_lean or g2_bf16_lds or dynamic)
    if fp8_store_cm is None:
        fp8_store_cm = store_cm
    out_elem_bytes = 1 if quant_fp8 else 2
    token_nbytes = N_OUT * 2 if dynamic else stage2_p2p_row_nbytes(N_OUT, p2p_quant_type)
    # MXFP4 row payload and BN-wide column block bytes; the E8M0 scales follow the row payload.
    fp4_payload_nbytes = N_OUT // 2
    tile_payload_nbytes = BN // 2 if quant_fp4 else BN * out_elem_bytes
    if row_nbytes is not None:
        token_nbytes = int(row_nbytes)
    assert not (lean and epi_lean) and not ((lean or epi_lean) and g2_bf16_lds)
    # The lane-fed peer bases decode the top-k slot as topk <= 8 does.
    assert tok_off is None or (lean >= 2 and topk <= 8)
    assert not bf16_head_tokens or dynamic
    assert not epi_lean or (quant_fp8 and scatter_vec == 8 and elem_view_stores)
    assert not route_i8 or epi_lean
    assert scale_stage is None or quant_fp8

    def store_peer(peer_base, byte_off, value, elem, unit_elems, cache_modifier):
        # elem_view_stores: units overlap at element stride, so the index counts elements, not units.
        index_bytes = elem.width // 8 if const_expr(elem_view_stores) else unit_elems * (elem.width // 8)
        buf_copy_store(
            ptr_buf_tensor(
                peer_base,
                elem,
                unit_elems=unit_elems,
                num_records_bytes=comb_inp_nbytes,
                unit_stride=1 if const_expr(elem_view_stores) else None,
            ),
            byte_off if const_expr(index_bytes == 1) else byte_off // fx.Int32(index_bytes),
            value,
            elem,
            unit_elems=unit_elems,
            cache_modifier=cache_modifier,
        )

    def store_stage_scale(stage_buf, stage_off, e8m0):
        stage = ptr_buf_tensor(stage_buf.addr, fx.Int8, num_records_bytes=stage_buf.num_records_bytes)
        buf_copy_store(stage, stage_off, e8m0.to(fx.Int8), fx.Int8, cache_modifier=_SC0_SC1_CACHE)

    fx.barrier()

    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * 4
        if const_expr(g2_bf16_lds):
            w_row = [meta.weight[row_base + v] for v in range_constexpr(4)]
        for J in range_constexpr(numAccN):
            col = wave * fx.Int32(wave_n) + J * 16 + lane_mod_16
            vec = fx.Vector(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + v) * BN + col
                if const_expr(g2_bf16_lds):
                    lds_base_bf16[idx] = fx.BFloat16(fx.Float32(vec[v]) * fx.Float32(w_row[v]))
                else:
                    lds_base_fptr[idx] = fx.Float32(vec[v])

    fx.barrier()

    if const_expr(lean or epi_lean):
        # Lane l holds row (wave + l * NW)'s packed id and weight.
        meta_off = (wave + (lane % fx.Int32(BM // NW)) * fx.Int32(NW)) * fx.Int32(4)
        if const_expr(epi_lean):
            # unit_zero == 0 but is opaque per unit, so the address is not hoisted out of the G2 loop and kept live.
            meta_off = meta_off + (n_block_idx < fx.Int32(0)).select(n_block_idx, fx.Int32(0))
        p_lanes = fx.ptr_load(meta.packed.at_byte(meta_off))
        w_lanes = fx.ptr_load(meta.weight.at_byte(meta_off)).bitcast(fx.Int32)
        if const_expr(lean >= 2):
            t_l = p_lanes & fx.Int32(0x00FFFFFF)
            pe_l = t_l >> fx.Int32(log2_max_tok)
            valid_l = ((t_l < fx.Int32(recv_cap)) & ((p_lanes >> fx.Int32(24)) < fx.Int32(topk))
                & (pe_l < fx.Int32(npes)))
            pe_l = valid_l.select(pe_l, fx.Int32(0))
            peer_lo_l, peer_hi_l = meta.peer_pair(pe_l)

    for row_iter in range_constexpr(BM // NW):
        row = wave + fx.Int32(row_iter * NW)
        if const_expr(lean or epi_lean):
            p = fx.Int32(rocdl.readlane(T.i32, p_lanes, row_iter))
            weight = fx.Int32(rocdl.readlane(T.i32, w_lanes, row_iter)).bitcast(fx.Float32)
        else:
            p = meta.packed[row]
            if const_expr(not g2_bf16_lds):
                weight = meta.weight[row]
            p = fx.Int32(rocdl.readfirstlane(T.i32, p))
            if const_expr(not g2_bf16_lds):
                weight = fx.Float32(rocdl.readfirstlane(T.f32, weight))
        t = p & fx.Int32(0x00FFFFFF)
        if const_expr(topk <= 8 and (lean or epi_lean)):
            # Preserve the established DSV4-Pro instruction sequence.
            s = p >> fx.Int32(24)
        else:
            # Slots 8..15 set bit 31, so decode the high byte as unsigned.
            s = p.shrui(fx.Int32(24)) & fx.Int32(0xFF)
        dest_pe = t >> fx.Int32(log2_max_tok)
        dest_lid = t & fx.Int32(mask_max_tok)
        valid = (t < fx.Int32(recv_cap)) & (s < fx.Int32(topk)) & (dest_pe < fx.Int32(npes))
        dest_pe_safe = valid.select(dest_pe, fx.Int32(0))
        if const_expr(lean >= 2):
            lo = fx.Uint32(rocdl.readlane(T.i32, peer_lo_l, row_iter))
            hi = fx.Uint32(rocdl.readlane(T.i32, peer_hi_l, row_iter))
            peer_base = fx.Int64((fx.Uint64(hi) << 32) | fx.Uint64(lo))
        else:
            peer_base = fx.Int64(rocdl.readfirstlane(T.i64, meta.peer[dest_pe_safe]))
        slot = dest_lid * fx.Int32(topk) + s
        row_base = slot * fx.Int32(token_nbytes)
        if const_expr(not dynamic):
            row_off = row_base + n_block_idx * fx.Int32(tile_payload_nbytes)

        # Inactive lanes read safe LDS and issue a bounded OOB store.
        active = lane < fx.Int32(BN // scatter_vec)
        col = active.select(lane * fx.Int32(scatter_vec), fx.Int32(0))
        idx0 = row * fx.Int32(BN) + col
        if const_expr(g2_bf16_lds):
            pk = fx.Vector(
                lds_vec_load(
                    lds_acc_base,
                    idx0 * fx.Int32(2),
                    fx.Vector.make_type(scatter_vec, fx.BFloat16),
                    fx.BFloat16,
                    align=16,
                )
            )
        else:
            v8 = fx.Vector(
                lds_vec_load(
                    lds_acc_base,
                    idx0 * fx.Int32(4),
                    fx.Vector.make_type(scatter_vec, fx.Float32),
                    fx.Float32,
                    align=16,
                )
            )
            weighted_v8 = fx.Vector.from_elements(
                [v8[i] * weight for i in range_constexpr(scatter_vec)],
                fx.Float32,
            )
            if const_expr(not quant_fp8 and not quant_fp4 and not dynamic):
                pk = weighted_v8.to(fx.BFloat16)

        def store_fp8(row_off):
            vals = [
                weighted_v8[i] for i in range_constexpr(scatter_vec)
            ]
            scale_group_lanes = 32 // scatter_vec
            if const_expr(epi_lean):
                e8m0, payload = _fp8_quant_1x32_lean(vals, route_i8)
                scale_leader = active & ((lane & fx.Int32(3)) == fx.Int32(0))
            else:
                e8m0, payload, is_scale_leader = _fp8_quant_1x32(vals, lane)
                scale_leader = active & is_scale_leader
            payload_off = (valid & active).select(
                row_off + col,
                fx.Int32(comb_inp_nbytes),
            )
            # Adjacent active lanes issue contiguous 8-byte stores without ds_bpermute gathers.
            store_peer(peer_base, payload_off, payload, fx.Int32, scatter_vec // 4, fp8_store_cm)

            if const_expr(scale_stage is not None):
                stage_buf, stage_group0 = scale_stage
                if const_expr(epi_lean):
                    # Every lane stores; non-leaders go out of bounds (no branch around the store).
                    stage_off = scale_leader.select(
                        (stage_group0 + row) * fx.Int32(16) + lane // fx.Int32(4),
                        fx.Int32(stage_buf.num_records_bytes),
                    )
                    store_stage_scale(stage_buf, stage_off, e8m0)
                else:

                    @flyc.jit
                    def stage_scale_if_leader():
                        if scale_leader:
                            stage_off = (stage_group0 + row) * fx.Int32(16) + lane // fx.Int32(4)
                            store_stage_scale(stage_buf, stage_off, e8m0)

                    stage_scale_if_leader()
            else:

                @flyc.jit
                def store_scale_if_leader():
                    if scale_leader:
                        scale_off = valid.select(
                        row_base
                        + fx.Int32(N_OUT)
                        + n_block_idx * fx.Int32(BN // 32)
                        + lane // fx.Int32(scale_group_lanes),
                            fx.Int32(comb_inp_nbytes),
                        )
                        store_peer(peer_base, scale_off, e8m0.to(fx.Int8), fx.Int8, 1, fp8_store_cm)

                store_scale_if_leader()

        def store_fp4(row_off):
            vals = [weighted_v8[i] for i in range_constexpr(scatter_vec)]
            e8m0, words, is_scale_leader = _fp4_quant_1x32(vals, lane)
            scale_leader = active & is_scale_leader
            payload_off = (valid & active).select(
                row_off + col // fx.Int32(2),
                fx.Int32(comb_inp_nbytes),
            )
            payload = words[0] if len(words) == 1 else fx.Vector.from_elements(words, fx.Int32)
            store_peer(peer_base, payload_off, payload, fx.Int32, len(words), fp8_store_cm)

            @flyc.jit
            def store_scale_if_leader():
                if scale_leader:
                    scale_off = valid.select(
                        row_base
                        + fx.Int32(fp4_payload_nbytes)
                        + n_block_idx * fx.Int32(BN // 32)
                        + lane // fx.Int32(32 // scatter_vec),
                        fx.Int32(comb_inp_nbytes),
                    )
                    store_peer(peer_base, scale_off, e8m0.to(fx.Int8), fx.Int8, 1, fp8_store_cm)

            store_scale_if_leader()

        def store_bf16(row_off, pk):
            off = (valid & active).select(
                row_off + col * fx.Int32(2),
                fx.Int32(comb_inp_nbytes),
            )
            store_peer(peer_base, off, pk, fx.BFloat16, scatter_vec, store_cm)

        if const_expr(dynamic):

            @flyc.jit
            def store_dynamic():
                if fp8_rows != fx.Int32(0):
                    store_fp8(row_base + n_block_idx * fx.Int32(BN))
                else:
                    store_bf16(row_base + n_block_idx * fx.Int32(BN * 2), weighted_v8.to(fx.BFloat16))

            @flyc.jit
            def store_dynamic_head_bf16():
                if (fp8_rows != fx.Int32(0)) & (dest_lid >= fx.Int32(bf16_head_tokens)):
                    store_fp8(row_base + n_block_idx * fx.Int32(BN))
                else:
                    store_bf16(row_base + n_block_idx * fx.Int32(BN * 2), weighted_v8.to(fx.BFloat16))

            if const_expr(bf16_head_tokens):
                store_dynamic_head_bf16()
            else:
                store_dynamic()
        elif const_expr(quant_fp8):
            store_fp8(row_off)
        elif const_expr(quant_fp4):
            store_fp4(row_off)
        else:
            store_bf16(row_off, pk)

    if const_expr(tok_off is not None):

        @flyc.jit
        def count_rows():
            # A row is stored by one wave only, and the heap is uncached: once this wave's stores completed its
            # rows are visible on their destinations.
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            if valid_l & (lane < fx.Int32(BM // NW)):
                peer_l = fx.Int64(fx.Vector.from_elements([peer_lo_l, peer_hi_l], fx.Int32).bitcast(fx.Int64)[0])
                lid_l = t_l & fx.Int32(mask_max_tok)
                comm_ops.atomic_add_system(peer_l + tok_off + fx.Int64(lid_l) * fx.Int64(4), fx.Int32(1))

        count_rows()


# fmt: off
def p2p_scatter_epilog_wide(lds_acc_base, accm, n_block_idx, wave, lane, *, N_OUT, BM, BN, npes, topk,
    log2_max_tok, mask_max_tok, recv_cap, meta, scatter_vec=8):
# fmt: on
    """p2p_scatter_epilog for the MXFP4 combine with every lane busy: each wave scatters
    ``64 // (BN // scatter_vec)`` rows at once, each lane group with its own row's peer base."""
    kMChunks = BM // 16
    numAccN = (BN // 4) // 16
    wave_n = BN // 4
    lanes_per_row = BN // scatter_vec
    rows_per_iter = 64 // lanes_per_row
    assert scatter_vec == 8 and lanes_per_row * rows_per_iter == 64 and BM % (4 * rows_per_iter) == 0
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)
    token_nbytes = stage2_p2p_row_nbytes(N_OUT, "fp4_blockwise_1x32")
    scale_group_lanes = 32 // scatter_vec

    fx.barrier()
    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * 4
        for J in range_constexpr(numAccN):
            col = wave * fx.Int32(wave_n) + J * 16 + lane_mod_16
            vec = fx.Vector(accm[i][J])
            for v in range_constexpr(4):
                lds_base_fptr[(row_base + v) * BN + col] = fx.Float32(vec[v])
    fx.barrier()

    row_group = lane // fx.Int32(lanes_per_row)
    row_lane = lane - row_group * fx.Int32(lanes_per_row)
    col = row_lane * fx.Int32(scatter_vec)
    for it in range_constexpr(BM // (4 * rows_per_iter)):
        row = wave + fx.Int32(4) * (fx.Int32(it * rows_per_iter) + row_group)
        p = meta.packed[row]
        weight = meta.weight[row]
        t = p & fx.Int32(0x00FFFFFF)
        if const_expr(topk <= 8):
            s = p >> fx.Int32(24)
        else:
            s = p.shrui(fx.Int32(24)) & fx.Int32(0xFF)
        dest_pe = t >> fx.Int32(log2_max_tok)
        dest_lid = t & fx.Int32(mask_max_tok)
        valid = (t < fx.Int32(recv_cap)) & (s < fx.Int32(topk)) & (dest_pe < fx.Int32(npes))
        peer_base = meta.peer[valid.select(dest_pe, fx.Int32(0))]
        row_off = (dest_lid * fx.Int32(topk) + s) * fx.Int32(token_nbytes)
        v8 = fx.Vector(lds_vec_load(lds_acc_base, (row * fx.Int32(BN) + col) * fx.Int32(4),
                                    fx.Vector.make_type(scatter_vec, fx.Float32), fx.Float32, align=16))
        vals = [v8[q] * weight for q in range_constexpr(scatter_vec)]
        e8m0, words, is_scale_leader = _fp4_quant_1x32(vals, lane)
        payload_addr = peer_base + fx.Int64(
            row_off + n_block_idx * fx.Int32(BN // 2) + col // fx.Int32(2))
        scale_addr = peer_base + fx.Int64(
            row_off + fx.Int32(N_OUT // 2) + n_block_idx * fx.Int32(BN // 32)
            + row_lane // fx.Int32(scale_group_lanes))

        @flyc.jit
        def store_row():
            if valid:
                fx.ptr_store(words[0], global_typed_ptr(payload_addr, T.i32))
                if is_scale_leader:
                    fx.ptr_store(e8m0.to(fx.Int8), global_typed_ptr(scale_addr, T.i8, align=1))

        store_row()


_P2P_ROW_NBYTES = {
    "none": lambda n: n * 2,
    "fp8_blockwise_1x32": lambda n: n + n // 32,
    "fp4_blockwise_1x32": lambda n: n // 2 + n // 32,
}


def stage2_p2p_row_nbytes(model_dim, p2p_quant_type):
    """Receiver row stride for one (token, top-k slot) under the selected transport format."""
    if p2p_quant_type not in _P2P_ROW_NBYTES:
        raise ValueError(f"unsupported p2p_quant_type={p2p_quant_type!r}")
    return int(_P2P_ROW_NBYTES[p2p_quant_type](int(model_dim)))


def _stage2_lds_bytes(BM, BN, BK, a_dtype, aStages, g2_bf16_lds=False):
    is_f8 = a_dtype == "fp8"
    KH_TILE_A = BK // (1 if is_f8 else 2)
    slot_bytes = BM * KH_TILE_A
    c_lds_bytes = BM * BN * (2 if g2_bf16_lds else 4)
    return max(c_lds_bytes, aStages * slot_bytes)


class Stage2Unit:
    """One GEMM2 output tile and its weighted P2P scatter, emitted into a 256-thread kernel.

    The standalone Stage2 kernel and fixed-slot Stage1's fused GEMM2 phase run their tiles through it.
    It uses ``lds_bytes`` of LDS from the pool base it is given: the A stages / CShuffle slab, then the
    Stage2Meta tables (each row's packed route and weight, and the peer combine-input bases that
    ``load_peers`` fills once per CTA).
    """

    # fmt: off
    def __init__(self, *, model_dim, inter_dim, experts, topk, rank, npes, max_tok, recv_cap, comb_inp_nbytes,
        BM, BN, BK, use_nt, INTER_MAX, a_dtype, SBM, has_pad, g2_bhoist, g2_ascale_pf, g2_b2stage,
        g2_deep_a_pipeline, g2_bf16_lds, p2p_quant_type, scatter_vec, wide_scatter=False):
    # fmt: on
        self.BM, self.BN, self.BK, self.SBM = BM, BN, BK, SBM
        # Every lane scatters (MXFP4 combine and the f32 CShuffle slab only).
        self.wide_scatter = (
            bool(wide_scatter) and p2p_quant_type == "fp4_blockwise_1x32" and not g2_bf16_lds
            and scatter_vec == 8 and 64 % (BN // scatter_vec) == 0
        )
        self.use_nt, self.INTER_MAX, self.a_dtype, self.has_pad = use_nt, INTER_MAX, a_dtype, has_pad
        self.g2_bhoist, self.g2_ascale_pf = g2_bhoist, g2_ascale_pf
        self.g2_b2stage, self.g2_deep_a_pipeline, self.g2_bf16_lds = g2_b2stage, g2_deep_a_pipeline, g2_bf16_lds
        self.p2p_quant_type, self.scatter_vec = p2p_quant_type, scatter_vec
        self.npes, self.topk = npes, topk
        self.log2_max_tok = max_tok.bit_length() - 1
        self.mask_max_tok = max_tok - 1
        self.N_OUT = model_dim
        self.is_f8 = a_dtype == "fp8"
        self.aStages = 4 if g2_deep_a_pipeline else kStages + 1
        self.KH_TILE_A = BK // (1 if self.is_f8 else 2)
        compute_lds_bytes = _stage2_lds_bytes(BM, BN, BK, a_dtype, self.aStages, g2_bf16_lds)
        self.lds_packed_off = compute_lds_bytes
        self.lds_weight_off = self.lds_packed_off + BM * 4
        self.lds_peer_off = self.lds_weight_off + BM * 4
        self.lds_bytes = self.lds_peer_off + npes * 8
        if self.lds_bytes > 160 * 1024:
            raise ValueError(f"Stage2 LDS use {self.lds_bytes} exceeds 160 KiB")
        self.recv_cap = npes * max_tok if recv_cap is None else int(recv_cap)
        row_nbytes = stage2_p2p_row_nbytes(model_dim, p2p_quant_type)
        self.comb_inp_nbytes = max_tok * topk * row_nbytes if comb_inp_nbytes is None else int(comb_inp_nbytes)
        if not 0 < self.comb_inp_nbytes < _BUFFER_OFFSET_ABI_BYTES:
            raise ValueError("MegaMoE v2 stage2 P2P buffer exceeds the 32-bit buffer-resource ABI")
        self.expert_offset = rank * experts

    def meta(self, lds_base_i32):
        return Stage2Meta(lds_base_i32, self.lds_packed_off, self.lds_weight_off, self.lds_peer_off)

    @flyc.jit
    def load_peers(self, lds_base_i32, arg_p2p_comb_inp, tx_i32):
        """Copy the peers' combine-input bases into LDS (first ``npes`` threads)."""
        p2p_table = ptr_buf_tensor(arg_p2p_comb_inp, fx.Int64)
        if tx_i32 < fx.Int32(self.npes):
            fx.ptr_store(p2p_table[tx_i32], self.meta(lds_base_i32).peer + tx_i32)

    # fmt: off
    @flyc.jit
    def run(self, lds_base_i32, unit_bx, m_block_idx, *, arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids,
        i32_max_m_blocks, i32_inter, i32_hidden, i32_kpad, i32_npad, k_bytes, trb_buf, stids_buf, sweights_buf,
        tx_i32, lane, wave):
    # fmt: on
        """Load the tile's A stages, then run GEMM2 and scatter tile ``unit_bx`` of M block ``m_block_idx``."""
        BM, SBM = self.BM, self.SBM
        meta = self.meta(lds_base_i32)
        for slot in range_constexpr(kStages):
            issue_a_load_lds_dt(arg_aq, lds_base_i32, slot, slot, m_block_idx * fx.Int32(BM), wave, lane,
                self.is_f8, self.KH_TILE_A, k_bytes, BM=BM)
        rocdl.sched_barrier(0)
        # Map each Stage2 BM sub-tile to its Stage1 SBM metadata row.
        m_row = m_block_idx * fx.Int32(BM)
        sort_block_idx = m_row // fx.Int32(SBM)
        row_in_sort_block = m_row - sort_block_idx * fx.Int32(SBM)
        srcmap_row_base = trb_buf[sort_block_idx] + row_in_sort_block
        if tx_i32 < fx.Int32(BM):
            sorted_pos = srcmap_row_base + tx_i32
            fx.ptr_store(stids_buf[sorted_pos], meta.packed + tx_i32)
            fx.ptr_store(sweights_buf[sorted_pos], meta.weight + tx_i32)
        # fmt: off
        accm_vecs, m_row, n_block_idx, _n_out_rt = gemm2_compute_v2(lds_base_i32, arg_ascale, arg_bq,
            arg_bscale, arg_eids, arg_aq, i32_max_m_blocks, unit_bx, lane, wave, i32_inter, i32_hidden,
            i32_kpad, i32_npad, BM=BM, BN=self.BN, BK=self.BK, use_nt=self.use_nt, INTER_MAX=self.INTER_MAX,
            aStages=self.aStages, a_dtype=self.a_dtype, has_pad=self.has_pad, SBM=SBM, g2_bhoist=self.g2_bhoist,
            g2_b2stage=self.g2_b2stage, g2_ascale_pf=self.g2_ascale_pf,
            g2_deep_a_pipeline=self.g2_deep_a_pipeline, expert_offset=self.expert_offset)
        if const_expr(self.wide_scatter):
            p2p_scatter_epilog_wide(lds_base_i32, accm_vecs, n_block_idx, wave, lane, N_OUT=self.N_OUT,
                BM=BM, BN=self.BN, npes=self.npes, topk=self.topk, log2_max_tok=self.log2_max_tok,
                mask_max_tok=self.mask_max_tok, recv_cap=self.recv_cap, meta=meta,
                scatter_vec=self.scatter_vec)
        else:
            p2p_scatter_epilog(lds_base_i32, accm_vecs, n_block_idx, wave, lane, N_OUT=self.N_OUT,
                BM=BM, BN=self.BN, npes=self.npes, topk=self.topk,
                log2_max_tok=self.log2_max_tok, mask_max_tok=self.mask_max_tok, recv_cap=self.recv_cap,
                comb_inp_nbytes=self.comb_inp_nbytes, meta=meta, g2_bf16_lds=self.g2_bf16_lds,
                p2p_quant_type=self.p2p_quant_type, scatter_vec=self.scatter_vec)
        # fmt: on


# fmt: off
@aot_factory("mega_moe_stage2")
@functools.cache
def compile_mega_moe_stage2(*, model_dim: int, inter_dim: int, experts: int, topk: int, rank: int, npes: int,
    max_tok: int, recv_cap: int | None = None, comb_inp_nbytes: int | None = None, BM: int = 32, BN: int = 256,
    BK: int = 256, use_nt: bool = True, HIDDEN_MAX: int = 8192, INTER_MAX: int = 8192, a_dtype: str = "fp8",
    SBM: int | None = None,
    persist: bool = False, cu_num: int = 0, has_pad: bool = False, g2_bhoist=None, g2_b2stage: bool = False,
    g2_ascale_pf=None, g2_spart=None, g2_deep_a_pipeline: bool = False, persist_strided: bool = False,
    g2_bf16_lds: bool = False, p2p_quant_type: str = "none",
    fixed_slot_dispatch: bool = False, skew_cu: int = 0,
    runtime_pair_skip: bool = False, scatter_vec: int = 8):
# fmt: on
    """Compile fused GEMM2 and weighted cross-rank P2P scatter."""
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx95"):
        raise RuntimeError(f"MegaMoE v2 stage2 requires CDNA4 (gfx95x), got {arch or 'unknown'}")
    assert max_tok > 0 and (max_tok & (max_tok - 1)) == 0, "max_tok must be power of two"
    assert model_dim % BN == 0 and HIDDEN_MAX % BN == 0
    assert INTER_MAX % BK == 0, f"INTER_MAX must be a multiple of {BK}"
    if BM not in (16, 32, 64, 128):
        raise AssertionError(f"BM must be in {{16,32,64,128}}, got {BM}")
    SBM = BM if SBM is None else int(SBM)
    if SBM % BM != 0:
        raise AssertionError(f"SBM ({SBM}) must be a multiple of BM ({BM})")
    if p2p_quant_type not in _P2P_ROW_NBYTES:
        raise ValueError(f"unsupported p2p_quant_type={p2p_quant_type!r}")
    if p2p_quant_type != "none" and g2_bf16_lds:
        raise ValueError(f"{p2p_quant_type} requires f32 CShuffle input (g2_bf16_lds=False)")
    if scatter_vec not in (8, 16):
        raise ValueError("Stage2 scatter_vec must be 8 or 16")
    if BN % scatter_vec or 32 % scatter_vec:
        raise ValueError("Stage2 scatter_vec must divide BN and the 1x32 scale group")
    if a_dtype not in ("fp4", "fp8"):
        raise AssertionError(f"a_dtype must be 'fp4' or 'fp8', got {a_dtype!r}")
    if persist and cu_num <= 0:
        raise AssertionError(f"persist=True requires cu_num>0, got {cu_num}")
    if g2_deep_a_pipeline and not g2_b2stage:
        raise AssertionError("g2_deep_a_pipeline requires g2_b2stage=True")
    if skew_cu and (not persist or not 0 < skew_cu <= cu_num):
        raise AssertionError(f"skew_cu={skew_cu} requires persist=True and 0<skew_cu<=cu_num={cu_num}")
    # The scatter path uses the f32 CShuffle slab rather than BF16 LDS.
    g2_bhoist, g2_ascale_pf, g2_spart, g2_group_num, g2_m01, _g2_bf16_lds = _resolve_g2_knobs(
        g2_bhoist, g2_ascale_pf, g2_spart, False, False
    )
    is_f8 = a_dtype == "fp8"
    # fmt: off
    unit = Stage2Unit(model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, rank=rank, npes=npes,
        max_tok=max_tok, recv_cap=recv_cap, comb_inp_nbytes=comb_inp_nbytes, BM=BM, BN=BN, BK=BK, use_nt=use_nt,
        INTER_MAX=INTER_MAX, a_dtype=a_dtype, SBM=SBM, has_pad=has_pad, g2_bhoist=g2_bhoist,
        g2_ascale_pf=g2_ascale_pf, g2_b2stage=g2_b2stage, g2_deep_a_pipeline=g2_deep_a_pipeline,
        g2_bf16_lds=g2_bf16_lds, p2p_quant_type=p2p_quant_type, scatter_vec=scatter_vec)
    # fmt: on
    lds_bytes = unit.lds_bytes
    _expert_offset = unit.expert_offset
    if runtime_pair_skip and fixed_slot_dispatch:
        raise ValueError("runtime pair skipping requires compact dispatch")
    _total_experts = npes * experts
    _total_segments = _total_experts + npes

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    dispatch_path = "fixedslot" if fixed_slot_dispatch else "compact"
    kernel_name = (
        f"megamoe_stage2_{dispatch_path}_t{BM}x{BN}x{BK}"
        f"_sbm{SBM}_{a_dtype}_nt{int(use_nt)}"
        f"_p{int(persist)}cu{cu_num}s{int(persist_strided)}_pad{int(has_pad)}"
        f"_sk{skew_cu}"
        f"_bh{int(g2_bhoist)}b2{int(g2_b2stage)}apf{int(g2_ascale_pf)}"
        f"da{int(g2_deep_a_pipeline)}sp{g2_group_num}x{g2_m01}"
        f"_bf16lds{int(g2_bf16_lds)}_{p2p_quant_type}"
        f"_rps{int(runtime_pair_skip)}_rtv3{int(runtime_pair_skip)}"
        f"_sv{scatter_vec}_tb2_rsm1"
    )

    # fmt: off
    @flyc.kernel(name=kernel_name, known_block_size=[256, 1, 1])
    def kernel_epilog_v2(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_max_expert_tiles: fx.Int64, arg_stids: fx.Int64,
        arg_sweights: fx.Int64, arg_trb: fx.Int64, arg_expert_tile_end: fx.Int64,
        arg_count_matrix: fx.Int64, arg_pair_config: fx.Int64, arg_parity: fx.Int64,
        arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32, i32_hidden: fx.Int32, i32_kpad: fx.Int32, i32_npad: fx.Int32):
    # fmt: on
        tx_i32 = fx.thread_idx.x
        bx_i32 = fx.block_idx.x
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))

        num_n_blocks = i32_hidden // fx.Int32(BN)
        k_bytes = i32_inter // fx.Int32(1 if is_f8 else 2)
        # kernel-invariant scatter resources + peer-base table (loaded into registers once).
        trb_buf = ptr_buf_tensor(arg_trb, fx.Int32)
        eids_buf = ptr_buf_tensor(arg_eids, fx.Int32)
        stids_buf = ptr_buf_tensor(arg_stids, fx.Int32)
        sweights_buf = ptr_buf_tensor(arg_sweights, fx.Float32)
        skip_base_a = fx.Int32(0)
        skip_base_b = fx.Int32(0)
        skip_rows = fx.Int32(0)
        if const_expr(runtime_pair_skip):
            pair_config = ptr_buf_tensor(arg_pair_config, fx.Int32)
            parity_buf = ptr_buf_tensor(arg_parity, fx.Int32)
            active_parity = parity_buf[fx.Int32(0)]
            packed_pair = pair_config[
                active_parity * fx.Int32(npes) + fx.Int32(rank)
            ]
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            pair_enabled = (packed_pair & fx.Int32(1 << 16)) != fx.Int32(0)
            skip_a = packed_pair & fx.Int32(0xFF)
            skip_b = packed_pair.shrui(fx.Int32(8)) & fx.Int32(0xFF)
            expert_tile_end = ptr_buf_tensor(arg_expert_tile_end, fx.Int32)
            count_matrix = ptr_buf_tensor(arg_count_matrix, fx.Int32)
            skip_base_a_lane = fx.Int32(0)
            skip_base_b_lane = fx.Int32(0)
            skip_rows_lane = fx.Int32(0)
            if lane == fx.Int32(0):
                safe_prev_a = (skip_a > fx.Int32(0)).select(
                    skip_a - fx.Int32(1), fx.Int32(0)
                )
                safe_prev_b = (skip_b > fx.Int32(0)).select(
                    skip_b - fx.Int32(1), fx.Int32(0)
                )
                prev_a = expert_tile_end[safe_prev_a] * fx.Int32(SBM)
                prev_b = expert_tile_end[safe_prev_b] * fx.Int32(SBM)
                skip_base_a_lane = (skip_a > fx.Int32(0)).select(
                    prev_a, fx.Int32(0)
                )
                skip_base_b_lane = (skip_b > fx.Int32(0)).select(
                    prev_b, fx.Int32(0)
                )
                group_count = fx.Int32(0)
                group_column = fx.Int32(_total_experts + rank)
                for source in range_constexpr(npes):
                    group_count = group_count + count_matrix[
                        fx.Int32(source * _total_segments) + group_column
                    ]
                skip_rows_lane = (
                    (group_count + fx.Int32(SBM - 1)) // fx.Int32(SBM)
                ) * fx.Int32(SBM)
            skip_base_a = fx.Int32(
                rocdl.readfirstlane(T.i32, skip_base_a_lane)
            )
            skip_base_b = fx.Int32(
                rocdl.readfirstlane(T.i32, skip_base_b_lane)
            )
            skip_rows = fx.Int32(rocdl.readfirstlane(T.i32, skip_rows_lane))
        unit.load_peers(lds_base_i32, arg_p2p_comb_inp, tx_i32)

        def run_unskipped_unit(unit_bx, m_block_idx):
            skip = fx.Int32(0)
            if const_expr(runtime_pair_skip):
                m_row = m_block_idx * fx.Int32(BM)
                expert = eids_buf[m_row // fx.Int32(SBM)]
                in_a = (
                    pair_enabled
                    & (expert == fx.Int32(_expert_offset) + skip_a)
                    & (m_row >= skip_base_a)
                    & (m_row < skip_base_a + skip_rows)
                )
                in_b = (
                    pair_enabled
                    & (expert == fx.Int32(_expert_offset) + skip_b)
                    & (m_row >= skip_base_b)
                    & (m_row < skip_base_b + skip_rows)
                )
                skip = (in_a | in_b).select(fx.Int32(1), fx.Int32(0))
            if skip == fx.Int32(0):
                # fmt: off
                unit.run(lds_base_i32, unit_bx, m_block_idx, arg_aq=arg_aq, arg_ascale=arg_ascale, arg_bq=arg_bq,
                    arg_bscale=arg_bscale, arg_eids=arg_eids, i32_max_m_blocks=i32_max_m_blocks,
                    i32_inter=i32_inter, i32_hidden=i32_hidden, i32_kpad=i32_kpad, i32_npad=i32_npad,
                    k_bytes=k_bytes, trb_buf=trb_buf, stids_buf=stids_buf, sweights_buf=sweights_buf,
                    tx_i32=tx_i32, lane=lane, wave=wave)
                # fmt: on

        cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
        total_m_blocks = (cumsum0 + fx.Int32(BM - 1)) // fx.Int32(BM)

        if const_expr(not persist and g2_spart <= 0):
            bound = total_m_blocks * num_n_blocks
            if fx.Int32(bx_i32) < bound:
                run_unskipped_unit(bx_i32, bx_i32 // num_n_blocks)
        elif const_expr(not persist):
            bound = total_m_blocks * num_n_blocks
            if fx.Int32(bx_i32) < bound:
                m_block_idx, n_block_idx = _spart_output_tile_index(
                    bx_i32, total_m_blocks, num_n_blocks, g2_group_num, g2_m01
                )
                unit_bx = m_block_idx * num_n_blocks + n_block_idx
                run_unskipped_unit(unit_bx, m_block_idx)
        elif const_expr(skew_cu > 0):
            total_stage1_tiles = (cumsum0 + fx.Int32(SBM - 1)) // fx.Int32(SBM)
            max_expert_tiles = global_typed_ptr(arg_max_expert_tiles, T.i32)[0]
            skewed = max_expert_tiles * fx.Int32(4) > total_stage1_tiles
            skew_n_block = bx_i32 // fx.Int32(cu_num)
            skew_m_slot = bx_i32 - skew_n_block * fx.Int32(cu_num)
            normal_m_slot = bx_i32 // num_n_blocks
            normal_n_block = bx_i32 - normal_m_slot * num_n_blocks
            n_block = skewed.select(skew_n_block, normal_n_block)
            m_slot = skewed.select(skew_m_slot, normal_m_slot)
            active_cu = skewed.select(fx.Int32(skew_cu), fx.Int32(cu_num))
            strided_diff = total_m_blocks - m_slot
            strided_rem = (strided_diff > fx.Int32(0)).select(strided_diff, fx.Int32(0))
            strided_iters = (strided_rem + active_cu - fx.Int32(1)) // active_cu
            tiles_per_slot = (
                total_m_blocks + active_cu - fx.Int32(1)
            ) // active_cu
            m_tile0 = m_slot * tiles_per_slot
            contiguous_diff = total_m_blocks - m_tile0
            contiguous_rem = (contiguous_diff > fx.Int32(0)).select(
                contiguous_diff, fx.Int32(0)
            )
            contiguous_iters = (contiguous_rem < tiles_per_slot).select(
                contiguous_rem, tiles_per_slot
            )
            n_iters = skewed.select(strided_iters, contiguous_iters)
            active = m_slot < active_cu
            for _it in range(fx.Int32(0), n_iters, fx.Int32(1)):
                strided_m = m_slot + fx.Int32(_it) * active_cu
                contiguous_m = m_tile0 + fx.Int32(_it)
                m_block = skewed.select(strided_m, contiguous_m)
                if active:
                    unit_bx = m_block * num_n_blocks + n_block
                    fx.barrier()
                    if m_block < total_m_blocks:
                        run_unskipped_unit(unit_bx, m_block)
        else:
            m_slot = bx_i32 // num_n_blocks
            n_block = bx_i32 - m_slot * num_n_blocks
            if const_expr(persist_strided):
                diff = total_m_blocks - m_slot
                rem = (diff > fx.Int32(0)).select(diff, fx.Int32(0))
                n_iters = (rem + fx.Int32(cu_num - 1)) // fx.Int32(cu_num)
            else:
                tiles_per_slot = (
                    total_m_blocks + fx.Int32(cu_num - 1)
                ) // fx.Int32(cu_num)
                m_tile0 = m_slot * tiles_per_slot
                diff = total_m_blocks - m_tile0
                rem = (diff > fx.Int32(0)).select(diff, fx.Int32(0))
                n_iters = (rem < tiles_per_slot).select(rem, tiles_per_slot)
            for _it in range(fx.Int32(0), n_iters, fx.Int32(1)):
                if const_expr(persist_strided):
                    m_block = m_slot + fx.Int32(_it) * fx.Int32(cu_num)
                else:
                    m_block = m_tile0 + fx.Int32(_it)
                unit_bx = m_block * num_n_blocks + n_block
                fx.barrier()  # separate prev-iter epilog LDS reads from this iter's A-load into the LDS union
                if fx.Int32(m_block) < total_m_blocks:
                    run_unskipped_unit(unit_bx, m_block)

    # fmt: off
    @flyc.jit
    def launch(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_max_expert_tiles: fx.Int64, arg_stids: fx.Int64,
        arg_sweights: fx.Int64, arg_trb: fx.Int64, arg_expert_tile_end: fx.Int64,
        arg_count_matrix: fx.Int64, arg_pair_config: fx.Int64, arg_parity: fx.Int64,
        arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_grid_blocks: fx.Int32, i32_inter: fx.Int32, i32_hidden: fx.Int32, i32_kpad: fx.Int32,
        i32_npad: fx.Int32, stream: fx.Stream):
    # fmt: on
        num_n_blocks = i32_hidden // fx.Int32(BN)
        grid_x = i32_grid_blocks * num_n_blocks
        kernel_epilog_v2(
            arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_max_expert_tiles,
            arg_stids, arg_sweights, arg_trb, arg_expert_tile_end, arg_count_matrix,
            arg_pair_config, arg_parity,
            arg_p2p_comb_inp, i32_max_m_blocks, i32_inter,
            i32_hidden, i32_kpad, i32_npad,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def _get_g2_launch(**compile_kw):
    """The launch handle of a fused-stage2 launcher for a full compile-param set
    (compile_mega_moe_stage2.lazy): pass it to _run_compiled or _preload_compiled."""
    return compile_mega_moe_stage2.lazy(**compile_kw)


# fmt: off
def run_mega_moe_stage2(arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum,
    arg_max_expert_tiles, arg_stids, arg_sweights, arg_trb, arg_expert_tile_end,
    arg_count_matrix, arg_pair_config, arg_parity, arg_p2p, row_capacity,
    i32_inter, i32_hidden, stream, *,
    model_dim, inter_dim, experts, topk, rank, npes, max_tok, recv_cap, comb_inp_nbytes, BM, SBM,
    HIDDEN_MAX, INTER_MAX, cu_num, BN=256, BK=256, use_nt=True, g2_bhoist=True, g2_b2stage=False,
    g2_ascale_pf=True, g2_spart=402, g2_deep_a_pipeline=False, persist=False, persist_cu=0, persist_strided=False,
    g2_bf16_lds=False, p2p_quant_type="none", fixed_slot_dispatch=False, skew_cu=0,
    runtime_pair_skip=False, scatter_vec=8, a_dtype="fp8"):
    # fmt: on
    """Compile or reuse one fused Stage2 configuration and launch it."""
    launch_cu_num = min(cu_num, persist_cu) if persist and persist_cu > 0 else cu_num
    launch = _get_g2_launch(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, rank=rank, npes=npes,
        max_tok=max_tok, recv_cap=recv_cap, comb_inp_nbytes=comb_inp_nbytes, BM=BM, BN=BN, BK=BK,
        use_nt=use_nt, HIDDEN_MAX=HIDDEN_MAX, INTER_MAX=INTER_MAX, SBM=SBM, persist=persist,
        cu_num=launch_cu_num, g2_bhoist=g2_bhoist, g2_b2stage=g2_b2stage,
        g2_ascale_pf=g2_ascale_pf, g2_spart=g2_spart, g2_deep_a_pipeline=g2_deep_a_pipeline,
        persist_strided=persist_strided, g2_bf16_lds=g2_bf16_lds,
        p2p_quant_type=p2p_quant_type, fixed_slot_dispatch=fixed_slot_dispatch, skew_cu=skew_cu,
        runtime_pair_skip=runtime_pair_skip,
        scatter_vec=scatter_vec,
        a_dtype=a_dtype,
    )
    max_m_blocks = (row_capacity + BM - 1) // BM
    grid_blocks = launch_cu_num if persist else max_m_blocks
    _run_compiled(
        launch, arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum,
        arg_max_expert_tiles, arg_stids, arg_sweights, arg_trb, arg_expert_tile_end,
        arg_count_matrix, arg_pair_config, arg_parity, arg_p2p, fx.Int32(max_m_blocks),
        fx.Int32(grid_blocks), fx.Int32(i32_inter), fx.Int32(i32_hidden), fx.Int32(0), fx.Int32(0), stream,
    )


# fmt: off
def preload_mega_moe_stage2(arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum,
    arg_max_expert_tiles, arg_stids, arg_sweights, arg_trb, arg_expert_tile_end,
    arg_count_matrix, arg_pair_config, arg_parity, arg_p2p, row_capacity,
    i32_inter, i32_hidden, stream, *,
    model_dim, inter_dim, experts, topk, rank, npes, max_tok, recv_cap, comb_inp_nbytes, BM, SBM,
    HIDDEN_MAX, INTER_MAX, cu_num, BN=256, BK=256, use_nt=True, g2_bhoist=True, g2_b2stage=False,
    g2_ascale_pf=True, g2_spart=402, g2_deep_a_pipeline=False, persist=False, persist_cu=0, persist_strided=False,
    g2_bf16_lds=False, p2p_quant_type="none", fixed_slot_dispatch=False, skew_cu=0,
    runtime_pair_skip=False, scatter_vec=8, a_dtype="fp8"):
# fmt: on
    """Compile and load one fused Stage2 variant without dispatching it."""
    launch_cu_num = min(cu_num, persist_cu) if persist and persist_cu > 0 else cu_num
    launch = _get_g2_launch(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, rank=rank, npes=npes,
        max_tok=max_tok, recv_cap=recv_cap, comb_inp_nbytes=comb_inp_nbytes, BM=BM, BN=BN, BK=BK,
        use_nt=use_nt, HIDDEN_MAX=HIDDEN_MAX, INTER_MAX=INTER_MAX, SBM=SBM, persist=persist,
        cu_num=launch_cu_num, g2_bhoist=g2_bhoist, g2_b2stage=g2_b2stage,
        g2_ascale_pf=g2_ascale_pf, g2_spart=g2_spart, g2_deep_a_pipeline=g2_deep_a_pipeline,
        persist_strided=persist_strided, g2_bf16_lds=g2_bf16_lds,
        p2p_quant_type=p2p_quant_type, fixed_slot_dispatch=fixed_slot_dispatch, skew_cu=skew_cu,
        runtime_pair_skip=runtime_pair_skip,
        scatter_vec=scatter_vec,
        a_dtype=a_dtype,
    )
    max_m_blocks = (row_capacity + BM - 1) // BM
    grid_blocks = launch_cu_num if persist else max_m_blocks
    return _preload_compiled(
        launch,
        arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum,
        arg_max_expert_tiles, arg_stids, arg_sweights, arg_trb, arg_expert_tile_end,
        arg_count_matrix, arg_pair_config, arg_parity, arg_p2p, fx.Int32(max_m_blocks),
        fx.Int32(grid_blocks), fx.Int32(i32_inter), fx.Int32(i32_hidden), fx.Int32(0), fx.Int32(0), stream,
    )

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL AttnRes training kernels (forward and backward) for gfx950.

Same math, arguments and saved statistics as :mod:`.attn_res_train`, the gfx1250
kernels. Each token mixes ``L = K + 1`` candidates, the ``K`` banked block
representations ``bank[t, 0..K)`` and the intra-block ``prefix[t]``::

    r_l = rsqrt(mean(v_l ** 2) + eps)          z_l = r_l * <v_l, q * w>
    p   = softmax(scale * z)                   o   = sum_l p_l * v_l
    y   = o * rsqrt(mean(o ** 2) + out_eps) * w_o      (optional output RMSNorm)

Both kernels are persistent: a workgroup walks tokens in a grid-stride loop. Every
row a token reads, its L candidates and in the backward ``dy`` ahead of them,
streams through a ring of ``R`` LDS slots with BUFFER_LOAD_DWORDX4 LDS=1, the
16-byte load to LDS that gfx950 adds, so rows in flight take no VGPRs. The first
pass over a token copies each row from its slot into VGPRs as soon as it lands and
then refills the slot with the row ``R`` places later in the stream, normally the
same row of the next token; the later passes run from VGPRs. The next token is
therefore in flight during all of this token's work, and every row is read from
HBM once. ``R`` is the rows per token when they fit in LDS; past that it drops
below them, and for short rows the backward streams several tokens ahead, with
each row's slot computed at run time in both cases.

A wave's load writes 1024 contiguous LDS bytes, lane ``l`` at ``M0 + 16 l``, and
each lane reads back only the bytes its own load wrote, so the slots need no
barrier and every LDS access is a contiguous ds_read_b128. The loads are async:
the compiler adds no vmcnt wait for them, and each row is waited for with
``wait_asyncmark(R - 1)``, which leaves the ``R - 1`` younger rows in flight.

Rows are held as bf16 pairs, so VGPRs rather than LDS cap the candidates per
token: at hidden 7168, K up to 16 with the output norm and 17 without. From
K = 11 with the output norm (16 without) the backward moves its weight-gradient
accumulators and ``w_o`` to LDS to make room.

Block reductions are batched across candidates as in the gfx1250 kernels. A wave
reduce-scatters its values with V_PERMLANE32_SWAP / V_PERMLANE16_SWAP and DPP row
permutations, one lane per value then sums the wave partials, and V_READLANE_B32
moves the totals to SGPRs. Partial buffers alternate with the token's parity, so
a token costs one barrier per reduction and none besides.
"""

# Do not add ``from __future__ import annotations``: FlyDSL inspects annotations
# at trace time.

import math
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from .kernels_common import create_llvm_ptr
from .tensor_shim import buf_scalar_load

WAVE = 64
_VEC = 8  # bf16 per 16-byte chunk
_F32_VEC = 4  # fp32 per 16-byte chunk
_CHUNK = 16  # bytes a lane moves per access
_WAVE_CHUNK = WAVE * _CHUNK  # bytes a wave moves per access
_WAVE_ELEMS = WAVE * _VEC  # bf16 a wave moves per access
_LOG2E = math.log2(math.e)
# gfx940+ cache policy: NT is the SLC bit. Rows and gradients are touched once.
_NT = 2
# A wave's tiles share one M0 and differ in the 12-bit instruction offset, which
# the LDS DMA adds to both addresses.
_MAX_TILES = 4
_MAX_WAVES = 16
LDS_BYTES = 160 * 1024
# The backward caps its workgroups to amortize their partial rows; short rows
# then stream this far ahead, so that a few workgroups per CU keep HBM busy.
_INFLIGHT_BYTES = 32 * 1024
# vmcnt counts up to 63; leave room for the stores issued between the loads.
_MAX_LOADS_IN_FLIGHT = 40
# A CU streams best with about two waves per SIMD: more workgroups of a width
# that spans its SIMDs only contend, up to 11% slower at Kimi-K3's 7168.
# Narrower workgroups want more company, and so do rings that hold fewer row
# bytes than keep a CU's loads in flight.
_CU_WAVES = 8
_CU_RING_BYTES = 28 * 1024
# 7 waves of 256 VGPRs fill a CU; a kernel past that would not launch.
_MAX_VGPRS = 256
# VGPRs a lane needs at two tiles besides the rows it holds (weights,
# accumulators, temporaries), keyed by (backward, output norm, accumulators in
# LDS): where the compiled kernels start to spill at hidden 7168. They scale with
# the tiles.
_OTHER_VGPRS = {
    (False, False, False): 80,
    (False, True, False): 108,
    (True, False, False): 114,
    (True, True, False): 156,
    (True, False, True): 98,
    (True, True, True): 108,
}
# DPP controls that exchange the two halves of 2, 4, 8 and 16-lane groups:
# quad_perm [1,0,3,2] and [2,3,0,1], row_half_mirror and row_mirror. A mirror is
# an xor with 7 or 15, which pairs the halves just as well for a sum.
_DPP_XCHG = {1: 0xB1, 2: 0x4E, 4: 0x141, 8: 0x140}
# Without the attributor's verdict LLVM keeps half of gfx950's unified register
# file for AGPRs; these kernels use none, and need the 256 VGPRs.
_KERNEL_ATTRS = {"llvm.passthrough": [["amdgpu-agpr-alloc", "0"]]}


def pick_layout(hidden_size: int) -> tuple:
    """``(waves, tiles)``: a wave covers ``tiles`` 1024-byte chunks of each row.

    Fewest waves from 7 up whose tiles stay within ``_MAX_TILES``, so that the row
    a lane holds in VGPRs stays small; Kimi-K3's 7168 runs 7 waves of two tiles.
    """
    if hidden_size <= 0 or hidden_size % _WAVE_ELEMS:
        raise ValueError(
            f"hidden size {hidden_size} must be a positive multiple of {_WAVE_ELEMS}"
        )
    chunks = hidden_size // _WAVE_ELEMS
    for waves in range(min(chunks, 7), _MAX_WAVES + 1):
        if chunks % waves == 0 and chunks // waves <= _MAX_TILES:
            return waves, chunks // waves
    raise ValueError(f"hidden size {hidden_size} has no supported wave split")


def _stream_rows(num_bank: int, output_norm: bool, backward: bool) -> int:
    """Rows a token streams: its candidates, and ``dy`` first in the backward."""
    if not backward:
        return num_bank + 1
    # With one candidate and no output norm, dv is dy and the row is never read.
    if num_bank == 0 and not output_norm:
        return 1
    return num_bank + 2


def _reduced_values(num_bank: int, output_norm: bool, backward: bool) -> int:
    """Most values one block reduction takes.

    The forward: two row sums per candidate, plus the previous token's output sum
    of squares. The backward: ``dp`` per candidate, or the two output-norm sums.
    """
    n_src = num_bank + 1
    if not backward:
        return 2 * n_src + (1 if output_norm else 0)
    return max(2, n_src)


def _partial_floats(num_bank, output_norm, n_waves, backward) -> int:
    """Up to two partial buffers per token parity; see ``block_sum`` in the kernels."""
    return 4 * _reduced_values(num_bank, output_norm, backward) * n_waves


def _acc_rows(num_bank: int, output_norm: bool) -> int:
    """fp32 rows the backward keeps in LDS when VGPRs run out: dqw, dwo and w_o."""
    return (1 if num_bank else 0) + (2 if output_norm else 0)


class _Plan(NamedTuple):
    slots: int  # LDS row slots
    rows: int  # rows a token streams
    acc_in_lds: bool  # backward accumulators and w_o in LDS rather than VGPRs
    lds_bytes: int


def _plan(hidden_size: int, num_bank: int, output_norm: bool, backward: bool):
    """How a site's kernel holds its data, or a ValueError if it cannot.

    The current token's rows live in VGPRs as bf16 pairs, ``rows * tiles * 4`` of
    them per lane next to the weights, accumulators and temporaries; that, not
    LDS, caps the candidates per token. Past it the backward moves its
    accumulators and ``w_o`` to LDS, at the cost of ring slots.
    """
    n_waves, tiles = pick_layout(hidden_size)
    rows = _stream_rows(num_bank, output_norm, backward)
    row_bytes = 2 * hidden_size
    row_vgprs = rows * tiles * 4

    def fits(acc_in_lds):
        other = _OTHER_VGPRS[(backward, output_norm, acc_in_lds)] * tiles // 2
        return row_vgprs + other <= _MAX_VGPRS

    acc_in_lds = not fits(False)
    if acc_in_lds and not (backward and fits(True)):
        raise ValueError(
            f"{rows} rows of {hidden_size} per token take {row_vgprs} VGPRs a lane, "
            "more than gfx950 holds next to the rest; use fewer candidates"
        )
    acc_bytes = 4 * hidden_size * _acc_rows(num_bank, output_norm) if acc_in_lds else 0
    part_bytes = 4 * _partial_floats(num_bank, output_norm, n_waves, backward)
    fit = (LDS_BYTES - part_bytes - acc_bytes) // row_bytes
    # The forward has no partial rows to amortize: each workgroup streams one
    # token ahead, and ``blocks_per_cu_cap`` says how many share a CU.
    ahead = 2
    if backward:
        ahead = min(_INFLIGHT_BYTES // row_bytes, _MAX_LOADS_IN_FLIGHT // tiles)
    slots = min(fit, max(rows, ahead))
    if slots < 2:
        raise ValueError(
            f"two rows of {hidden_size} do not fit in {LDS_BYTES} bytes of LDS"
        )
    return _Plan(slots, rows, acc_in_lds, slots * row_bytes + part_bytes + acc_bytes)


def check_supported(hidden_size: int, num_bank: int, output_norm: bool) -> None:
    """Raise ValueError unless both directions build for this site."""
    _plan(hidden_size, num_bank, output_norm, False)
    _plan(hidden_size, num_bank, output_norm, True)


def fallback_blocks_per_cu(hidden_size, num_bank, output_norm, backward) -> int:
    """Workgroups the LDS of a CU holds, for when the occupancy query cannot run."""
    plan = _plan(hidden_size, num_bank, output_norm, backward)
    return max(1, LDS_BYTES // plan.lds_bytes)


def blocks_per_cu_cap(hidden_size, num_bank, output_norm, backward) -> int:
    """Most workgroups worth running on a CU, or 0 for as many as fit."""
    n_waves, _ = pick_layout(hidden_size)
    plan = _plan(hidden_size, num_bank, output_norm, backward)
    per_cu = max(1, _CU_WAVES // n_waves)
    ring_bytes = per_cu * plan.slots * 2 * hidden_size
    if n_waves < 4 or ring_bytes < _CU_RING_BYTES:
        return 0
    return per_cu


def _waves_per_eu(plan: _Plan, n_waves: int) -> int:
    """Waves per SIMD to compile for: as many as the CU's LDS holds, but at most two,
    so that one 7-wave workgroup may take 256 VGPRs a lane."""
    waves = (LDS_BYTES // plan.lds_bytes) * n_waves
    return max(1, min(2, -(-waves // 4)))


def _fma(a, b, c):
    """``a * b + c`` on 8-lane fp32 vectors, scalars broadcast.

    FlyDSL emits ``*`` and ``+`` without contraction, so ``c + a * b`` costs a
    V_PK_MUL_F32 and a V_PK_ADD_F32 where this is one V_PK_FMA_F32.
    """

    def lanes(x):
        return x if isinstance(x, fx.Vector) else fx.full(_VEC, x, fx.Float32)

    return fmath.fma(lanes(a), lanes(b), lanes(c))


def _uniform(value):
    """A wave-uniform fp32 value moved to an SGPR, so it costs no VGPR."""
    bits = value.bitcast(fx.Int32)
    return fx.Int32(fx.rocdl.readfirstlane(T.i32, bits)).bitcast(fx.Float32)


def _load_uniform(buf, index):
    """fp32 ``buf[index]`` for a wave-uniform ``index``, loaded straight to an SGPR."""
    return fx.Int32(buf_scalar_load(buf, index)).bitcast(fx.Float32)


def _dpp_xchg(x, d):
    """``x`` from the lane that pairs with this one across the halves of ``2 d`` lanes."""
    bits = x.bitcast(fx.Int32).ir_value()
    moved = fx.rocdl.update_dpp(T.i32, bits, bits, _DPP_XCHG[d], 0xF, 0xF, False)
    return fx.Int32(moved).bitcast(fx.Float32)


def _swap(x, y, d):
    """V_PERMLANE{16,32}_SWAP_B32: the ``d``-lane blocks of ``x`` with bit ``d`` set
    trade places with the blocks of ``y`` with it clear."""
    op = fx.rocdl.permlane16_swap if d == 16 else fx.rocdl.permlane32_swap
    pair = op(
        llvm.StructType.get_literal([T.i32, T.i32]),
        x.bitcast(fx.Int32).ir_value(),
        y.bitcast(fx.Int32).ir_value(),
        False,
        False,
    )
    return [
        fx.Int32(llvm.extractvalue(T.i32, pair, [j])).bitcast(fx.Float32)
        for j in range(2)
    ]


def _xchg_add(x, d):
    if d >= 16:
        lo, hi = _swap(x, x, d)
        return lo + hi
    return x + _dpp_xchg(x, d)


def _wave_sum(value):
    """Total over the wave, in every lane."""
    for d in (1, 2, 4, 8, 16, 32):
        value = _xchg_add(value, d)
    return value


def _wave_scatter_sums(values, lane):
    """Wave totals of ``2^k`` per-lane values, scattered so lane ``l`` holds value ``l >> (6 - k)``.

    Each halving step pairs value ``j`` with ``j + s/2`` between the two halves of
    ``2 d``-lane groups and each lane keeps one of the two plus its partner's copy:
    one permlane swap and an add per pair at ``d = 32, 16``, two selects, a DPP
    move and an add below. ``2^k - 1`` pairs, where reducing each value whole
    costs six steps.
    """
    vals = list(values)
    d = WAVE // 2
    while len(vals) > 1:
        half = len(vals) // 2
        nxt = []
        if d >= 16:
            for j in range(half):
                lo, hi = _swap(vals[j], vals[j + half], d)
                nxt.append(lo + hi)
        else:
            upper = (lane & d) != 0
            for j in range(half):
                send = upper.select(vals[j], vals[j + half])
                keep = upper.select(vals[j + half], vals[j])
                nxt.append(keep + _dpp_xchg(send, d))
        vals = nxt
        d //= 2
    total = vals[0]
    while d >= 1:
        total = _xchg_add(total, d)
        d //= 2
    return total


def _wave_partials(values, lane):
    """``(scattered, rest, k)``: the first ``2^k`` values reduce-scattered, the rest whole."""
    k = min(len(values), WAVE).bit_length() - 1
    scattered = _wave_scatter_sums(values[: 1 << k], lane)
    rest = [_wave_sum(value) for value in values[1 << k :]]
    return scattered, rest, k


def _readlane(value, lane):
    bits = value.bitcast(fx.Int32)
    return fx.Int32(fx.rocdl.readlane(T.i32, bits, fx.Int32(lane))).bitcast(fx.Float32)


def _lane_sums(s_part, base, n_values, n_waves, lane):
    """The block totals of ``n_values`` wave partials laid out ``[value][wave]``.

    Lane ``i`` adds up value ``i`` of each 64-value chunk and V_READLANE_B32 hands
    the totals to SGPRs, so a wave issues ``n_waves`` LDS reads per chunk instead
    of ``n_waves`` per value.
    """
    sums = []
    for start in range_constexpr(0, n_values, WAVE):
        count = min(WAVE, n_values - start)
        mine = start + (lane < count).select(lane, fx.Int32(0))
        total = fx.memref_load(s_part, base + mine * n_waves)
        for w in range_constexpr(1, n_waves):
            total = total + fx.memref_load(s_part, base + mine * n_waves + w)
        sums += [_readlane(total, i) for i in range_constexpr(count)]
    return sums


def _row_io(chunk, tiles):
    """Global row access by 16-byte chunk: tile ``t`` of a lane is chunk ``chunk(t)``."""
    bf16_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    f32_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    f32x1_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

    def load_weight(row, bf16):
        """A weight row in fp32; weights are shared by every workgroup, so cached."""
        if const_expr(bf16):
            div = fx.logical_divide(row, fx.make_layout(_VEC, 1))
            values = []
            for t in range_constexpr(tiles):
                reg = fx.make_rmem_tensor(_VEC, fx.BFloat16)
                fx.copy(bf16_atom, fx.slice(div, (None, chunk(t))), reg)
                values.append(fx.memref_load_vec(reg).to(fx.Float32))
            return values
        div = fx.logical_divide(row, fx.make_layout(_F32_VEC, 1))
        values = []
        for t in range_constexpr(tiles):
            halves = []
            for half in range_constexpr(2):
                reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
                fx.copy(f32_atom, fx.slice(div, (None, 2 * chunk(t) + half)), reg)
                halves.append(fx.memref_load_vec(reg))
            values.append(halves[0].shuffle(halves[1], list(range(_VEC))))
        return values

    def store_f32(row, values):
        div = fx.logical_divide(row, fx.make_layout(_F32_VEC, 1))
        for t in range_constexpr(tiles):
            for half in range_constexpr(2):
                reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
                lanes = list(range(half * _F32_VEC, (half + 1) * _F32_VEC))
                fx.memref_store_vec(values[t].shuffle(values[t], lanes), reg)
                fx.copy(f32_atom, reg, fx.slice(div, (None, 2 * chunk(t) + half)))

    def store_scalar(row, index, value):
        div = fx.logical_divide(row, fx.make_layout(1, 1))
        reg = fx.make_rmem_tensor(1, fx.Float32)
        fx.memref_store_vec(fx.full(1, value, fx.Float32), reg)
        fx.copy(f32x1_atom, reg, fx.slice(div, (None, index)))

    return load_weight, store_f32, store_scalar


def _dot(a, b, tiles, fm):
    # Accumulate lane-wise across tiles, then reduce the 8 lanes once.
    acc = a[0] * b[0]
    for t in range_constexpr(1, tiles):
        acc = _fma(a[t], b[t], acc)
    return acc.reduce(ReductionOp.ADD, fastmath=fm)


def _widen(packed):
    return [p.to(fx.Float32) for p in packed]


def _reload(packed):
    """``packed`` rows as values the compiler cannot match to earlier ones.

    A row stays in VGPRs as bf16 pairs and every pass widens it again. Without
    this, CSE reuses the first pass's fp32 copy and keeps it live across passes,
    twice the registers. FlyDSL has no optimization barrier for values, so an
    empty asm with a tied operand stands in for one. The asm overwrites its input:
    callers use the result in place of ``packed`` from then on, and the original
    dies here instead of costing a copy.
    """
    out = []
    for p in packed:
        bits = p.bitcast(fx.Int32).ir_value()
        bits = llvm.inline_asm(bits.type, [bits], "", "=v,0", has_side_effects=True)
        out.append(fx.Vector(bits).bitcast(fx.BFloat16))
    return out


def _row_ring(lds_rows, slots, hidden_size, tiles, wave, lane, sources, num_tokens):
    """Rows streamed through ``slots`` LDS slots: ``issue`` starts one, ``read`` loads one.

    ``sources[j]`` is ``(rsrc, token_stride, offset, records)`` in bytes for row
    ``j`` of a token. A row of a token past the end points its soffset at the
    buffer's record count: every lane is out of range, so nothing is fetched, but
    the loads still form a group and the sliding wait keeps counting right.
    """
    row_bytes = hidden_size * 2
    lds_base = fx.Int32(fx.ptrtoint(fx.recast_iter(fx.Uint8, lds_rows.ptr)))
    wave_bytes = wave * (tiles * _WAVE_CHUNK)
    lds_wave = lds_base + wave_bytes
    voffset = (wave_bytes + lane * _CHUNK).ir_value()
    view = fx.logical_divide(
        lds_rows.view(fx.make_layout(slots * hidden_size, 1)), fx.make_layout(_VEC, 1)
    )
    lds_atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.BFloat16)
    size = fx.Int32(_CHUNK).ir_value()
    aux = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), _NT)

    def issue(token, j, slot):
        rsrc, token_stride, offset, records = sources[j]
        soffset = (token < num_tokens).select(token * token_stride + offset, records)
        lds_ptr = create_llvm_ptr(
            fx.Int64(lds_wave + slot * row_bytes), address_space=3
        )
        for t in range_constexpr(tiles):
            # The async copy atom has no cache-policy operand; the raw op takes
            # the NT bit, and asyncmark groups track it all the same.
            fx.rocdl.raw_ptr_buffer_load_async_lds(
                rsrc,
                lds_ptr,
                size,
                voffset,
                soffset.ir_value(),
                fx.Int32(t * _WAVE_CHUNK).ir_value(),
                aux=aux,
            )
        fx.rocdl.asyncmark()

    def read(slot):
        values = []
        for t in range_constexpr(tiles):
            reg = fx.make_rmem_tensor(_VEC, fx.BFloat16)
            index = slot * (hidden_size // _VEC) + (wave * tiles + t) * WAVE + lane
            fx.copy(lds_atom, fx.slice(view, (None, index)), reg)
            values.append(fx.memref_load_vec(reg))
        return values

    def store(rsrc, row_offset, values):
        """``values`` to the row at byte ``row_offset`` of ``rsrc``, in bf16.

        Every row shares the lane's one voffset VGPR and takes its own offset in
        soffset. Through ``fx.copy`` LLVM strength-reduces each row's address
        into a VGPR carried from token to token, one per row and tile.
        """
        for t in range_constexpr(tiles):
            bits = values[t].to(fx.BFloat16).bitcast(fx.Int32).ir_value()
            fx.rocdl.raw_ptr_buffer_store(
                bits,
                rsrc,
                (fx.Int32(voffset) + t * _WAVE_CHUNK).ir_value(),
                row_offset.ir_value(),
                aux=aux,
            )

    return issue, read, store


def _rsrc(tensor, records):
    """Buffer resource of ``tensor`` whose record count is its ``records`` bytes."""
    buf = fx.rocdl.make_buffer_tensor(tensor, num_records_bytes=fx.Int64(records))
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(buf))


def _candidate_sources(bank, prefix, num_bank, bank_strides, prefix_stride, records):
    """Stream sources of the candidates: the bank rows, then the prefix."""
    bank_records, prefix_records = records
    sources = []
    if num_bank > 0:
        bank_rsrc = _rsrc(bank, bank_records)
        for src in range(num_bank):
            sources.append(
                (
                    bank_rsrc,
                    bank_strides[0] * 2,
                    src * bank_strides[1] * 2,
                    bank_records,
                )
            )
    sources.append(
        (_rsrc(prefix, prefix_records), prefix_stride * 2, 0, prefix_records)
    )
    return sources


class _Stream:
    """Where row ``j`` of the current token sits and what refills its slot.

    With a slot per row, row ``j`` keeps slot ``j`` and the refill is row ``j`` of
    the next token. Otherwise row ``j`` of the ``m``-th token of the workgroup is
    stream position ``m * rows + j``, its slot is that modulo ``slots``, and the
    refill is the row ``slots`` positions on, in this token or a later one;
    ``base`` carries ``m * rows`` modulo ``slots`` from token to token.
    """

    def __init__(self, slots, rows):
        self.slots = slots
        self.rows = rows
        self.ring = slots != rows

    def slot(self, base, j):
        if not self.ring:
            return j
        return (base + j) % self.slots

    def position(self, i):
        """``(tokens ahead, row)`` of stream position ``i`` of the current token."""
        return i // self.rows, i % self.rows

    def refill(self, j):
        """``(tokens ahead, row)`` of the row that refills row ``j``'s slot."""
        return self.position(j + self.slots)

    def next_base(self, base):
        if not self.ring:
            return base
        nxt = base + self.rows % self.slots
        return (nxt >= self.slots).select(nxt - self.slots, nxt)


def build_attn_res_fwd(
    hidden_size: int,
    num_bank: int,
    output_norm: bool,
    eps: float,
    out_eps: float,
    scale: float,
    bank_strides: tuple,
    prefix_stride: int,
    weight_bf16: tuple,
):
    """``bank_strides`` (token, candidate) and ``prefix_stride`` are in elements;
    ``weight_bf16`` says which of query, rms_weight and the output weight are bf16
    rather than fp32.

    Pass 1 takes both row sums of every candidate as its row lands and reduces
    them across the block at once; pass 2 mixes the rows held in VGPRs.
    """
    n_waves, tiles = pick_layout(hidden_size)
    threads = n_waves * WAVE
    n_src = num_bank + 1
    red = _reduced_values(num_bank, output_norm, False)
    part_floats = _partial_floats(num_bank, output_norm, n_waves, False)
    plan = _plan(hidden_size, num_bank, output_norm, False)
    slots, rows = plan.slots, plan.rows
    stream = _Stream(slots, rows)
    inv_h = 1.0 / hidden_size
    q_bf16, w_bf16, wo_bf16 = weight_bf16
    name = f"attn_res_train_fwd_gfx950_h{hidden_size}_k{num_bank}" + (
        "_onorm" if output_norm else ""
    )

    @fx.struct
    class SharedStorage:
        rows: fx.Array[fx.BFloat16, slots * hidden_size, 1024]
        partials: fx.Array[fx.Float32, part_floats, 16]

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def fwd_kernel(
        bank: fx.Tensor,
        prefix: fx.Tensor,
        query: fx.Tensor,
        rms_weight: fx.Tensor,
        out_weight: fx.Tensor,
        y: fx.Tensor,
        logit: fx.Tensor,
        rstd: fx.Tensor,
        lse: fx.Tensor,
        out_rstd: fx.Tensor,
        num_tokens: fx.Int32,
        num_groups: fx.Int32,
        bank_records: fx.Int32,
        prefix_records: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        group = fx.Int32(fx.block_idx.x)
        lane = tid % WAVE
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tid // WAVE))
        fm = arith.FastMathFlags.fast
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_part = lds.partials.view(fx.make_layout(part_floats, 1))

        # A token reduces into partial buffer 2p for its parity p (the backward's
        # second reduction into 2p + 1). A buffer comes round again two tokens on,
        # past a barrier that every wave reaches only after reading the partials
        # it held: no other barrier is needed between tokens.
        def block_sum(values, buf):
            scattered, rest, k = _wave_partials(values, lane)
            group_lanes = WAVE >> k
            if const_expr(n_waves == 1):
                sums = [_readlane(scattered, i * group_lanes) for i in range(1 << k)]
                return sums + [_uniform(value) for value in rest]
            base = buf * (red * n_waves)
            if lane % group_lanes == 0:
                index = base + (lane // group_lanes) * n_waves + wave
                fx.memref_store(scattered, s_part, index)
            if const_expr(len(rest) > 0):  # noqa: SIM102
                if lane == 0:
                    for i in range_constexpr(len(rest)):
                        index = base + ((1 << k) + i) * n_waves + wave
                        fx.memref_store(rest[i], s_part, index)
            fx.gpu.barrier()
            return _lane_sums(s_part, base, len(values), n_waves, lane)

        def chunk(t):
            return (wave * tiles + t) * WAVE + lane

        def dot(a, b):
            return _dot(a, b, tiles, fm)

        load_weight, _store_f32, store_scalar = _row_io(chunk, tiles)
        sources = _candidate_sources(
            bank,
            prefix,
            num_bank,
            bank_strides,
            prefix_stride,
            (bank_records, prefix_records),
        )
        issue_row, read_row, store_row = _row_ring(
            lds.rows, slots, hidden_size, tiles, wave, lane, sources, num_tokens
        )

        # y is [N, H] and contiguous; a store past its records is dropped.
        row_bytes = hidden_size * 2
        y_records = num_tokens * row_bytes
        y_rsrc = _rsrc(y, y_records)
        logit_buf = fx.rocdl.make_buffer_tensor(logit)
        rstd_buf = fx.rocdl.make_buffer_tensor(rstd)
        lse_buf = fx.rocdl.make_buffer_tensor(lse)
        out_rstd_buf = fx.rocdl.make_buffer_tensor(out_rstd)

        # Weights first: their first use then waits on them alone, not on the
        # row loads issued after them.
        q_vals = load_weight(fx.rocdl.make_buffer_tensor(query), q_bf16)
        w_vals = load_weight(fx.rocdl.make_buffer_tensor(rms_weight), w_bf16)
        qw_vals = [q_vals[t] * w_vals[t] for t in range_constexpr(tiles)]
        if const_expr(output_norm):
            w_out = load_weight(fx.rocdl.make_buffer_tensor(out_weight), wo_bf16)
        fx.rocdl.sched_barrier(0)
        for i in range_constexpr(slots):
            ahead, row = stream.position(i)
            issue_row(group + ahead * num_groups, row, i)

        def finish_output(token, out, sumsq):
            """``y = o * rstd * w_o`` of ``token``; nothing for a negative one."""
            out_r = fmath.rsqrt(sumsq * inv_h + out_eps, fastmath=fm)
            out = [out[t] * out_r * w_out[t] for t in range_constexpr(tiles)]
            valid = token >= 0
            store_row(y_rsrc, valid.select(token * row_bytes, y_records), out)
            if (tid == 0) & valid:
                store_scalar(out_rstd_buf, token, out_r)

        def token_step(token, base, parity, prev):
            """One token; ``prev`` is the previous token's ``[token, o..., <o, o>]``.

            With the output norm, a token's ``<o, o>`` rides in the next token's
            reduction and its ``y`` is written then: one barrier per token instead
            of two. The first token's ``prev`` is token -1, which writes nothing.
            """
            packed_rows = []
            partial = []
            for src in range_constexpr(n_src):
                slot = stream.slot(base, src)
                # The rows issued after this one stay in flight.
                fx.rocdl.wait_asyncmark(slots - 1)
                packed = read_row(slot)
                packed_rows.append(packed)
                vals = _widen(packed)
                partial.append(dot(vals, vals))
                partial.append(dot(vals, qw_vals))
                # The row is in VGPRs once its reads return: refill the slot.
                ahead, row = stream.refill(src)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.rocdl.sched_barrier(0)
                issue_row(token + ahead * num_groups, row, slot)
            if const_expr(output_norm):
                partial.append(prev[-1])
            sums = block_sum(partial, 2 * parity)
            if const_expr(output_norm):
                finish_output(prev[0], prev[1:-1], sums[-1])
            zs = []
            rs = []
            for src in range_constexpr(n_src):
                r = _uniform(fmath.rsqrt(sums[2 * src] * inv_h + eps, fastmath=fm))
                rs.append(r)
                zs.append(sums[2 * src + 1] * r)
            run_max = zs[0] * scale
            for src in range_constexpr(1, n_src):
                run_max = run_max.maximumf(zs[src] * scale)
            weights = [
                fmath.exp2((zs[src] * scale - run_max) * _LOG2E, fastmath=fm)
                for src in range_constexpr(n_src)
            ]
            denom = weights[0]
            for src in range_constexpr(1, n_src):
                denom = denom + weights[src]
            inv_denom = 1.0 / denom
            if tid == 0:
                logit_row = fx.slice(logit_buf, (token, None))
                rstd_row = fx.slice(rstd_buf, (token, None))
                for src in range_constexpr(n_src):
                    store_scalar(logit_row, src, zs[src])
                    store_scalar(rstd_row, src, rs[src])
                store_scalar(lse_buf, token, run_max + fmath.log(denom, fastmath=fm))

            acc = None
            for src in range_constexpr(n_src):
                vals = _widen(_reload(packed_rows[src]))
                p = _uniform(weights[src] * inv_denom)
                if const_expr(src == 0):
                    acc = [vals[t] * p for t in range_constexpr(tiles)]
                else:
                    acc = [_fma(vals[t], p, acc[t]) for t in range_constexpr(tiles)]
                fx.rocdl.sched_barrier(0)

            head = [stream.next_base(base), 1 - parity]
            if const_expr(output_norm):
                return head + [token] + acc + [dot(acc, acc)]
            store_row(y_rsrc, token * row_bytes, acc)
            return head

        init = [fx.Int32(0), fx.Int32(0)]
        if const_expr(output_norm):
            zero = fx.Vector.filled(_VEC, 0.0, fx.Float32)
            init += (
                [fx.Int32(-1)]
                + [zero for _ in range_constexpr(tiles)]
                + [fx.Float32(0.0)]
            )
        for it, state in range(group, num_tokens, num_groups, init=init):
            # Carried values come back untyped.
            state = list(state)
            prev = []
            if const_expr(output_norm):
                prev = (
                    [fx.Int32(state[2])]
                    + [fx.Vector(v) for v in state[3 : 3 + tiles]]
                    + [fx.Float32(state[3 + tiles])]
                )
            results = yield token_step(
                fx.Int32(it), fx.Int32(state[0]), fx.Int32(state[1]), prev
            )
        if const_expr(output_norm):
            # The last token's output norm, reduced on its own.
            results = list(results)
            parity = fx.Int32(results[1])
            (sumsq,) = block_sum([fx.Float32(results[3 + tiles])], 2 * parity)
            last = [fx.Vector(v) for v in results[3 : 3 + tiles]]
            finish_output(fx.Int32(results[2]), last, sumsq)

    @flyc.jit
    def launch_fwd(
        bank: fx.Tensor,
        prefix: fx.Tensor,
        query: fx.Tensor,
        rms_weight: fx.Tensor,
        out_weight: fx.Tensor,
        y: fx.Tensor,
        logit: fx.Tensor,
        rstd: fx.Tensor,
        lse: fx.Tensor,
        out_rstd: fx.Tensor,
        num_tokens: fx.Int32,
        num_groups: fx.Int32,
        bank_records: fx.Int32,
        prefix_records: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        fwd_kernel(
            bank,
            prefix,
            query,
            rms_weight,
            out_weight,
            y,
            logit,
            rstd,
            lse,
            out_rstd,
            num_tokens,
            num_groups,
            bank_records,
            prefix_records,
            value_attrs=_KERNEL_ATTRS,
        ).launch(grid=(num_groups, 1, 1), block=(threads, 1, 1), stream=stream)

    launch_fwd.compile_hints = {"waves_per_eu": _waves_per_eu(plan, n_waves)}
    return launch_fwd


def build_attn_res_bwd(
    hidden_size: int,
    num_bank: int,
    output_norm: bool,
    scale: float,
    bank_strides: tuple,
    prefix_stride: int,
    dy_stride: int,
    weight_bf16: tuple,
):
    """``bank_strides`` (token, candidate), ``prefix_stride`` and ``dy_stride`` are
    in elements; ``weight_bf16`` is as for :func:`build_attn_res_fwd`.

    Pass 1 takes ``dy`` and then each candidate into VGPRs as they land. With the
    output norm it rebuilds ``o`` on the way and one reduction gives ``g``, pass 2
    takes every ``dp_l = <v_l, g>`` in a second reduction, and pass 3 writes
    ``dv``. Without it ``g`` is ``dy``, so pass 1 takes every ``dp_l`` as its row
    lands and pass 2 writes ``dv``.

    Each token's statistics are loaded a token ahead: scalar loads share lgkmcnt
    with LDS and return out of order, so one still in flight turns every LDS wait
    into a full drain.
    """
    n_waves, tiles = pick_layout(hidden_size)
    threads = n_waves * WAVE
    n_src = num_bank + 1
    red = _reduced_values(num_bank, output_norm, True)
    part_floats = _partial_floats(num_bank, output_norm, n_waves, True)
    plan = _plan(hidden_size, num_bank, output_norm, True)
    slots, rows, acc_in_lds = plan.slots, plan.rows, plan.acc_in_lds
    stream = _Stream(slots, rows)
    reads_candidates = rows > 1
    inv_h = 1.0 / hidden_size
    q_bf16, w_bf16, wo_bf16 = weight_bf16
    has_dqw = n_src > 1
    n_stats = 1 + 2 * n_src + (1 if output_norm else 0)
    # The accumulators ride in the loop state unless they live in LDS.
    n_dqw = tiles if has_dqw and not acc_in_lds else 0
    n_dwo = tiles if output_norm and not acc_in_lds else 0
    # LDS rows of the accumulators and w_o: dqw first, then dwo and w_o.
    acc_dqw = 0
    acc_dwo = 1 if has_dqw else 0
    acc_wo = acc_dwo + 1
    acc_elems = hidden_size * _acc_rows(num_bank, output_norm) if acc_in_lds else 0
    name = f"attn_res_train_bwd_gfx950_h{hidden_size}_k{num_bank}" + (
        "_onorm" if output_norm else ""
    )

    @fx.struct
    class SharedStorage:
        rows: fx.Array[fx.BFloat16, slots * hidden_size, 1024]
        partials: fx.Array[fx.Float32, part_floats, 16]
        acc: fx.Array[fx.Float32, max(acc_elems, _F32_VEC), 16]

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def bwd_kernel(
        dy: fx.Tensor,
        bank: fx.Tensor,
        prefix: fx.Tensor,
        query: fx.Tensor,
        rms_weight: fx.Tensor,
        out_weight: fx.Tensor,
        logit: fx.Tensor,
        rstd: fx.Tensor,
        lse: fx.Tensor,
        out_rstd: fx.Tensor,
        dbank: fx.Tensor,
        dprefix: fx.Tensor,
        part_dqw: fx.Tensor,
        part_dwo: fx.Tensor,
        num_tokens: fx.Int32,
        num_groups: fx.Int32,
        dy_records: fx.Int32,
        bank_records: fx.Int32,
        prefix_records: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        group = fx.Int32(fx.block_idx.x)
        lane = tid % WAVE
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tid // WAVE))
        fm = arith.FastMathFlags.fast
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_part = lds.partials.view(fx.make_layout(part_floats, 1))
        s_acc = fx.logical_divide(
            lds.acc.view(fx.make_layout(max(acc_elems, _F32_VEC), 1)),
            fx.make_layout(_F32_VEC, 1),
        )
        lds_f32_atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.Float32)

        # Partial buffers by token parity, as in the forward.
        def block_sum(values, buf):
            scattered, rest, k = _wave_partials(values, lane)
            group_lanes = WAVE >> k
            if const_expr(n_waves == 1):
                sums = [_readlane(scattered, i * group_lanes) for i in range(1 << k)]
                return sums + [_uniform(value) for value in rest]
            base = buf * (red * n_waves)
            if lane % group_lanes == 0:
                index = base + (lane // group_lanes) * n_waves + wave
                fx.memref_store(scattered, s_part, index)
            if const_expr(len(rest) > 0):  # noqa: SIM102
                if lane == 0:
                    for i in range_constexpr(len(rest)):
                        index = base + ((1 << k) + i) * n_waves + wave
                        fx.memref_store(rest[i], s_part, index)
            fx.gpu.barrier()
            return _lane_sums(s_part, base, len(values), n_waves, lane)

        # An LDS accumulator row is thread-private: half ``h`` of tile ``t`` of each
        # lane sits in one 16-byte slot, lanes adjacent, so no access needs a
        # barrier and a wave's access is contiguous.
        def acc_slot(row, t, half):
            return (row * tiles * 2 + 2 * t + half) * threads + tid

        def acc_load(row):
            values = []
            for t in range_constexpr(tiles):
                halves = []
                for half in range_constexpr(2):
                    reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
                    index = acc_slot(row, t, half)
                    fx.copy(lds_f32_atom, fx.slice(s_acc, (None, index)), reg)
                    halves.append(fx.memref_load_vec(reg))
                values.append(halves[0].shuffle(halves[1], list(range(_VEC))))
            return values

        def acc_store(row, values):
            for t in range_constexpr(tiles):
                for half in range_constexpr(2):
                    reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
                    lanes = list(range(half * _F32_VEC, (half + 1) * _F32_VEC))
                    fx.memref_store_vec(values[t].shuffle(values[t], lanes), reg)
                    index = acc_slot(row, t, half)
                    fx.copy(lds_f32_atom, reg, fx.slice(s_acc, (None, index)))

        def chunk(t):
            return (wave * tiles + t) * WAVE + lane

        def dot(a, b):
            return _dot(a, b, tiles, fm)

        load_weight, store_f32, _store_scalar = _row_io(chunk, tiles)
        sources = [(_rsrc(dy, dy_records), dy_stride * 2, 0, dy_records)]
        if const_expr(reads_candidates):
            sources += _candidate_sources(
                bank,
                prefix,
                num_bank,
                bank_strides,
                prefix_stride,
                (bank_records, prefix_records),
            )
        issue_row, read_row, store_row = _row_ring(
            lds.rows, slots, hidden_size, tiles, wave, lane, sources, num_tokens
        )
        row_bytes = hidden_size * 2

        def out_rsrc(tensor):
            return fx.rocdl.get_buffer_rsrc(
                fx.get_iter(fx.rocdl.make_buffer_tensor(tensor))
            )

        # dbank is [N, K, H] and dprefix [N, H], both contiguous.
        dprefix_rsrc = out_rsrc(dprefix)
        if const_expr(num_bank):
            dbank_rsrc = out_rsrc(dbank)
        logit_buf = fx.rocdl.make_buffer_tensor(logit)
        rstd_buf = fx.rocdl.make_buffer_tensor(rstd)
        lse_buf = fx.rocdl.make_buffer_tensor(lse)
        out_rstd_buf = fx.rocdl.make_buffer_tensor(out_rstd)

        def load_stats(token):
            """``[lse, z_0.., r_0.., out_r]`` of ``token``, of the last token past the end."""
            token = (token < num_tokens).select(token, num_tokens - 1)
            stat_base = token * n_src
            stats = [_load_uniform(lse_buf, token)]
            stats += [
                _load_uniform(logit_buf, stat_base + src)
                for src in range_constexpr(n_src)
            ]
            stats += [
                _load_uniform(rstd_buf, stat_base + src)
                for src in range_constexpr(n_src)
            ]
            if const_expr(output_norm):
                stats.append(_load_uniform(out_rstd_buf, token))
            return stats

        # Weights first: their first use then waits on them alone.
        if const_expr(has_dqw):
            q_vals = load_weight(fx.rocdl.make_buffer_tensor(query), q_bf16)
            w_vals = load_weight(fx.rocdl.make_buffer_tensor(rms_weight), w_bf16)
            qw_vals = [q_vals[t] * w_vals[t] for t in range_constexpr(tiles)]
        if const_expr(output_norm):
            w_out = load_weight(fx.rocdl.make_buffer_tensor(out_weight), wo_bf16)
        if const_expr(acc_in_lds):
            zero = fx.Vector.filled(_VEC, 0.0, fx.Float32)
            if const_expr(has_dqw):
                acc_store(acc_dqw, [zero for _ in range_constexpr(tiles)])
            if const_expr(output_norm):
                acc_store(acc_dwo, [zero for _ in range_constexpr(tiles)])
                acc_store(acc_wo, w_out)
        fx.rocdl.sched_barrier(0)
        for i in range_constexpr(slots):
            ahead, row = stream.position(i)
            issue_row(group + ahead * num_groups, row, i)

        def output_weight():
            # From LDS, where it lives when VGPRs are short, each time it is used.
            return acc_load(acc_wo) if acc_in_lds else w_out

        def take(token, base, j):
            """Row ``j`` of ``token`` into VGPRs, its slot then refilled."""
            slot = stream.slot(base, j)
            # The rows issued after this one stay in flight.
            fx.rocdl.wait_asyncmark(slots - 1)
            packed = read_row(slot)
            ahead, row = stream.refill(j)
            # The DMA must not overwrite the slot before its reads return.
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.rocdl.sched_barrier(0)
            issue_row(token + ahead * num_groups, row, slot)
            return packed

        def copy_step(token, base):
            # One candidate, no output norm: dv is dy.
            store_row(dprefix_rsrc, token * row_bytes, _widen(take(token, base, 0)))
            return stream.next_base(base)

        def token_step(token, base, parity, stats, dqw_acc, dwo_acc):
            lse_t = stats[0]
            zs = stats[1 : 1 + n_src]
            rs = stats[1 + n_src : 1 + 2 * n_src]
            ps = [
                _uniform(fmath.exp2((zs[src] * scale - lse_t) * _LOG2E, fastmath=fm))
                for src in range_constexpr(n_src)
            ]
            dy_packed = take(token, base, 0)
            dy_vals = _widen(dy_packed)
            vrows = []
            if const_expr(output_norm):
                out_r = stats[1 + 2 * n_src]
                # Pass 1: rebuild o = sum_l p_l v_l as the rows land.
                out = None
                for src in range_constexpr(n_src):
                    packed = take(token, base, 1 + src)
                    vrows.append(packed)
                    vals = _widen(packed)
                    if const_expr(src == 0):
                        out = [vals[t] * ps[src] for t in range_constexpr(tiles)]
                    else:
                        out = [
                            _fma(vals[t], ps[src], out[t])
                            for t in range_constexpr(tiles)
                        ]
                    fx.rocdl.sched_barrier(0)
                # Output RMSNorm backward and delta = <g, o>, in one reduction:
                # with xhat = o * r_o, <xhat, w_o dy> is r_o * <w_o dy, o>. Folding
                # the dp below into it too costs a second dot per candidate and
                # measures no faster than the barrier it saves.
                dy_packed = _reload(dy_packed)
                dy_vals = _widen(dy_packed)
                dwo = acc_load(acc_dwo) if acc_in_lds else dwo_acc
                dwo = [
                    _fma(out[t], dy_vals[t] * out_r, dwo[t])
                    for t in range_constexpr(tiles)
                ]
                if const_expr(acc_in_lds):
                    acc_store(acc_dwo, dwo)
                else:
                    dwo_acc = dwo
                w_vals = output_weight()
                wdy = [w_vals[t] * dy_vals[t] for t in range_constexpr(tiles)]
                b, c = block_sum([dot(wdy, out), dot(out, out)], 2 * parity)
                c1 = out_r * b * inv_h
                delta = out_r * b - out_r * out_r * c1 * c
                xhat_coef = out_r * c1
                # w_o dy again from the bf16 dy, rather than live across the barrier.
                dy_vals = _widen(_reload(dy_packed))
                w_vals = output_weight()
                g = [
                    _fma(out[t], -xhat_coef, w_vals[t] * dy_vals[t]) * out_r
                    for t in range_constexpr(tiles)
                ]
                if const_expr(n_src > 1):
                    # Pass 2: every candidate's dp = <v, g>, reduced at once.
                    partial = []
                    for src in range_constexpr(n_src):
                        vrows[src] = _reload(vrows[src])
                        partial.append(dot(_widen(vrows[src]), g))
                        # One row at a time: this is memory bound, and overlapping
                        # rows only adds live fp32 copies.
                        fx.rocdl.sched_barrier(0)
                    dps = block_sum(partial, 2 * parity + 1)
            else:
                g = dy_vals
                # Pass 1: every dp = <v, dy> as the rows land, reduced at once;
                # delta = <dy, o> is then sum_l p_l dp_l.
                partial = []
                for src in range_constexpr(n_src):
                    packed = take(token, base, 1 + src)
                    vrows.append(packed)
                    partial.append(dot(_widen(packed), g))
                    fx.rocdl.sched_barrier(0)
                dps = block_sum(partial, 2 * parity)
                delta = None
                for src in range_constexpr(n_src):
                    term = ps[src] * dps[src]
                    delta = term if delta is None else delta + term

            # The next token's statistics land while the dv pass runs.
            next_stats = load_stats(token + num_groups)

            # Last pass: per-candidate softmax and RMSNorm backward from the rows in
            # VGPRs. A single candidate has p = 1 whatever its score, so its score
            # gets no gradient.
            if const_expr(acc_in_lds and has_dqw):
                dqw_acc = acc_load(acc_dqw)
            for src in range_constexpr(n_src):
                if const_expr(n_src == 1):
                    dv = g
                else:
                    vrows[src] = _reload(vrows[src])
                    vals = _widen(vrows[src])
                    dz = (dps[src] - delta) * (ps[src] * scale)
                    key_coef = dz * rs[src]
                    key_shift = rs[src] * zs[src] * inv_h
                    dv = [
                        _fma(
                            _fma(vals[t], -key_shift, qw_vals[t]),
                            key_coef,
                            g[t] * ps[src],
                        )
                        for t in range_constexpr(tiles)
                    ]
                    dqw_acc = [
                        _fma(vals[t], key_coef, dqw_acc[t])
                        for t in range_constexpr(tiles)
                    ]
                if const_expr(src < num_bank):
                    store_row(dbank_rsrc, (token * num_bank + src) * row_bytes, dv)
                else:
                    store_row(dprefix_rsrc, token * row_bytes, dv)
                fx.rocdl.sched_barrier(0)
            if const_expr(acc_in_lds and has_dqw):
                acc_store(acc_dqw, dqw_acc)
                dqw_acc = []
            return stream.next_base(base), 1 - parity, next_stats, dqw_acc, dwo_acc

        if const_expr(not reads_candidates):
            if const_expr(stream.ring):
                for it, state in range(
                    group, num_tokens, num_groups, init=[fx.Int32(0)]
                ):
                    # Carried values come back untyped.
                    yield copy_step(fx.Int32(it), fx.Int32(state[0]))
            else:
                for it in range(group, num_tokens, num_groups):
                    copy_step(fx.Int32(it), 0)
        else:
            zero = fx.Vector.filled(_VEC, 0.0, fx.Float32)
            init = (
                [fx.Int32(0), fx.Int32(0)]
                + load_stats(group)
                + [zero for _ in range_constexpr(n_dqw + n_dwo)]
            )
            n_head = 2 + n_stats
            for it, state in range(group, num_tokens, num_groups, init=init):
                # Carried values come back untyped.
                state = list(state)
                base, parity, stats, dqw_acc, dwo_acc = token_step(
                    fx.Int32(it),
                    fx.Int32(state[0]),
                    fx.Int32(state[1]),
                    [fx.Float32(v) for v in state[2:n_head]],
                    [fx.Vector(v) for v in state[n_head : n_head + n_dqw]],
                    [fx.Vector(v) for v in state[n_head + n_dqw :]],
                )
                results = yield [base, parity] + stats + dqw_acc + dwo_acc
            sums = [fx.Vector(v) for v in list(results)[n_head:]]
            if const_expr(has_dqw):
                store_f32(
                    fx.slice(fx.rocdl.make_buffer_tensor(part_dqw), (group, None)),
                    acc_load(acc_dqw) if acc_in_lds else sums[:n_dqw],
                )
            if const_expr(output_norm):
                store_f32(
                    fx.slice(fx.rocdl.make_buffer_tensor(part_dwo), (group, None)),
                    acc_load(acc_dwo) if acc_in_lds else sums[n_dqw:],
                )

    @flyc.jit
    def launch_bwd(
        dy: fx.Tensor,
        bank: fx.Tensor,
        prefix: fx.Tensor,
        query: fx.Tensor,
        rms_weight: fx.Tensor,
        out_weight: fx.Tensor,
        logit: fx.Tensor,
        rstd: fx.Tensor,
        lse: fx.Tensor,
        out_rstd: fx.Tensor,
        dbank: fx.Tensor,
        dprefix: fx.Tensor,
        part_dqw: fx.Tensor,
        part_dwo: fx.Tensor,
        num_tokens: fx.Int32,
        num_groups: fx.Int32,
        dy_records: fx.Int32,
        bank_records: fx.Int32,
        prefix_records: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        bwd_kernel(
            dy,
            bank,
            prefix,
            query,
            rms_weight,
            out_weight,
            logit,
            rstd,
            lse,
            out_rstd,
            dbank,
            dprefix,
            part_dqw,
            part_dwo,
            num_tokens,
            num_groups,
            dy_records,
            bank_records,
            prefix_records,
            value_attrs=_KERNEL_ATTRS,
        ).launch(grid=(num_groups, 1, 1), block=(threads, 1, 1), stream=stream)

    launch_bwd.compile_hints = {"waves_per_eu": _waves_per_eu(plan, n_waves)}
    return launch_bwd

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Settings and gfx942/gfx950 BF16 D256 helpers shared by the QSA sparse attention kernels.

The first part holds the per-platform calibration and the private resource
policy. The platform entry is picked by gfx model and CU count (nearest
calibrated CU count of the same model) and read once per process;
AITER_QSA_PLATFORM overrides fields of the picked entry, e.g.
``AITER_QSA_PLATFORM="union_step=3.0,max_packed_kv_bytes=0"``.

The second part holds raw-buffer operations, register pinning and pipeline
barriers. The third part holds the BM128/BN64 MFMA, softmax and LDS pieces
of the eight-wave staggered D256 pipeline that the union kernel runs; the fourth
holds their gfx950 (CDNA4) forms: 16x16x32 MFMAs, ds_read_b64_tr_b16 V operands,
v_cvt_pk_bf16_f32 and K/V LDS swizzles for 64 banks. On gfx950 the union kernel and the
packed direct kernel take their gfx950 paths; the raw direct kernel runs as on gfx942.
"""

import dataclasses
import functools
import logging
import math
import os
from collections.abc import Callable
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx

# for inline asm and target features
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm

# Raw rocdl ops, here and in the sparse_prefill kernels, that flydsl.expr.rocdl.__all__
# (FlyDSL 0.4) has no stable form of: s_setprio and s_barrier; sched_barrier, and
# sched_group_barrier with mask_mfma or VALU/TRANS masks (sched_mfma, sched_vmem, sched_dsrd
# and sched_dswr name only their groups); ballot, readlane, readfirstlane, mbcnt_lo/hi,
# update_dpp and ds_bpermute; perm_b32 and cvt_pk_bf16_f32; exp2 as a bare v_exp_f32;
# raw_ptr_buffer_load (the union's membership bits) and gfx942's LDS DMA
# raw_ptr_buffer_load_lds.
from flydsl.expr import rocdl

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

# ---- Platform calibration and private resource policy ----

# Resource policy; private, not configurable from outside.
WORKSPACE_CACHE = 8  # layouts kept per process; captured workspaces are never evicted
ARENA_GROWTH_PERCENT = 125  # a growing arena takes at least this share of its size
SCRATCH_ALIGN = 256  # bytes between the scratch views of an arena
UNION_SCRATCH_LIMIT = 4 << 30  # union scratch bytes; past it a layout runs direct only
MAX_ROW_PACKED_KV_BYTES = 1 << 30  # PK+PV bytes of the pack_tokens_per_row rule


@dataclasses.dataclass(frozen=True)
class Platform:
    """Calibrated per GPU. The route-model terms are microseconds where an entry says
    they are phase timings (gfx942); a decision-tuned entry (gfx950) gives weights
    chosen for the route they produce, not timings."""

    union_fixed: float  # fixed cost of the union phase
    union_step: float  # per N64 step of the slowest union CTA
    row_step: float  # per BN32 step of the longest direct row
    route_margin: float  # keeping every union tile must be this share slower to lose
    direct_fixed: float  # fixed cost of the direct phase
    pack_block: float  # per packed KV block and KV head
    direct_cu_step: float  # per BN32 direct step and KV head, times the CU count
    union_ratio: float  # union while padded union steps <= ratio x direct steps
    union_min_rows: int  # below max(rows, rows_per_head x H/HK) direct-only wins
    union_min_rows_per_head: int
    wide_tiles_per_cu: float  # full H6/H3 prefill: BQ21/BQ42 from this many tiles/CU
    max_packed_kv_bytes: int  # direct packs KV while PK+PV fits, else reads the raw KV
    # Past that budget, direct still packs a context of at most this many tokens per query row
    # (packing copies the whole context each call).
    pack_tokens_per_row: int
    # sparse_prefill_order_masks also weighs keeping every union tile against none when the
    # union phase is set by throughput (sparse_prefill_prepare.py _route); a compile-time
    # switch of that kernel.
    route_all_or_none: bool = False


PLATFORMS = {
    # MI308X phase timings.
    ("gfx942", 80): Platform(
        union_fixed=10.0,
        union_step=3.2,
        row_step=0.7,
        route_margin=0.05,
        direct_fixed=41.0,
        pack_block=0.0041,
        direct_cu_step=0.304,
        union_ratio=1.7,
        union_min_rows=384,
        union_min_rows_per_head=64,
        wide_tiles_per_cu=1.0,
        max_packed_kv_bytes=64 * 1024 * 1024,
        pack_tokens_per_row=32,
    ),
    # MI355X, decision-tuned: the route terms and route_margin of sparse_prefill_order_masks
    # were chosen on 91 calibration layouts so that routed calls stay near the faster of
    # all-union and all-direct under a union plan (90 within 1.05x, worst 1.06x); they are not
    # phase timings.
    # union_ratio 1e30: sparse_prefill_compact proposes union for every tile;
    # route_all_or_none: sparse_prefill_order_masks also weighs keeping every union tile
    # against none when the union phase is throughput-bound.
    ("gfx950", 256): Platform(
        union_fixed=345.2,
        union_step=2.228,
        row_step=8.528,
        route_margin=0.0,
        direct_fixed=0.001456,
        pack_block=0.001067,
        direct_cu_step=0.5711,
        union_ratio=1e30,
        union_min_rows=1536,
        union_min_rows_per_head=0,
        wide_tiles_per_cu=8 / 21,  # BQ21 from 2048 rows, BQ42 from 4096 (256 CUs)
        max_packed_kv_bytes=64 * 1024 * 1024,
        pack_tokens_per_row=32,
        route_all_or_none=True,
    ),
}
OVERRIDE = "AITER_QSA_PLATFORM"


@functools.cache
def platform() -> Platform:
    gfx, cus = get_gfx_runtime(), get_cu_num()
    keys = [key for key in PLATFORMS if key[0] == gfx]
    if not keys:
        raise ValueError(f"QSA sparse attention has no calibration for {gfx}")
    key = min(keys, key=lambda key: abs(key[1] - cus))
    if key[1] != cus:
        logging.getLogger("aiter").warning(
            "QSA sparse attention: %s with %d CUs is not calibrated; using the %d-CU entry",
            gfx,
            cus,
            key[1],
        )
    entry = PLATFORMS[key]
    text = os.environ.get(OVERRIDE, "").strip()
    if not text:
        return entry
    names = {field.name for field in dataclasses.fields(Platform)}
    values = {}
    for item in text.split(","):
        name, _, value = (part.strip() for part in item.partition("="))
        if name not in names or not value:
            raise ValueError(
                f"{OVERRIDE}: expected name=value with name in {sorted(names)}"
            )
        kind = type(getattr(entry, name))
        if kind is bool and value not in ("0", "1"):
            raise ValueError(f"{OVERRIDE}: {name} is a switch, written 0 or 1")
        values[name] = value == "1" if kind is bool else kind(value)
    entry = dataclasses.replace(entry, **values)
    values = dataclasses.astuple(entry)
    if not all(math.isfinite(v) and v >= 0 for v in values) or entry.route_margin >= 1:
        raise ValueError(f"{OVERRIDE}: values must be finite, >= 0, route_margin < 1")
    return entry


# ---- Raw buffers, register pinning, pipeline barriers ----


def _uniform(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(value)))


def _min(a, b):
    return (a < b).select(a, b)


def _pin_i32(value):
    # Tied empty asm materializes the value in a VGPR here; no fx equivalent.
    return fx.Int32(
        llvm.inline_asm(
            fx.Int32.ir_type,
            [fx.Int32(value).ir_value()],
            "",
            "=v,0",
            has_side_effects=True,
        )
    )


def _pin(values, chunk=8):
    values = fx.Vector(values)
    dtype = values.dtype
    words = values.bitcast(fx.Int32) if dtype.width < 32 else values
    result = []
    for start in range(0, words.numel, chunk):
        part = fx.Vector.from_elements(
            [words[i] for i in range(start, min(start + chunk, words.numel))],
            words.dtype,
        )
        tied = fx.Vector(
            llvm.inline_asm(
                part.ir_value().type,
                [part.ir_value()],
                "",
                "=v,0",
                has_side_effects=True,
            )
        )
        result.extend(tied[i] for i in range(tied.numel))
    return fx.Vector.from_elements(result, words.dtype).bitcast(dtype)


def _keep_vgpr(words, chunk=4):
    # Empty asm use keeps LDS loads in VGPRs; a tied pin's def would cost an s_nop.
    words = fx.Vector(words)
    for start in range(0, words.numel, chunk):
        part = fx.Vector.from_elements(
            [words[i] for i in range(start, start + chunk)], words.dtype
        )
        llvm.inline_asm(None, [part.ir_value()], "", "v", has_side_effects=True)


def _target_features():
    # Packed FP32 VALU makes the union phase loop about 8% slower.
    return ir.Attribute.parse('#llvm.target_features<["-packed-fp32-ops"]>')


def _join(a, b):
    a, b = fx.Vector(a), fx.Vector(b)
    return fx.Vector.from_elements(
        [a[i] for i in range(a.numel)] + [b[i] for i in range(b.numel)], fx.Float32
    )


def _exp(value):
    # Bare v_exp_f32: math.exp2 adds denormal range reduction under IEEE f32.
    return fx.Float32(rocdl.exp2(fx.Float32.ir_type, fx.Float32(value).ir_value()))


def _mfma(a, b, c):
    """One 16x16x16 BF16 MFMA on register vectors: a/b hold 4 BF16 bit patterns, c 4 FP32."""
    frag_a = fx.make_rmem_tensor(4, fx.BFloat16)
    frag_b = fx.make_rmem_tensor(4, fx.BFloat16)
    frag_c = fx.make_rmem_tensor(4, fx.Float32)
    frag_a.store(fx.Vector(a).bitcast(fx.BFloat16))
    frag_b.store(fx.Vector(b).bitcast(fx.BFloat16))
    frag_c.store(fx.Vector(c))
    mma = fx.make_mma_atom(rocdl.MFMA(16, 16, 16, fx.BFloat16))
    fx.gemm(mma, frag_c, frag_a, frag_b, frag_c)
    return fx.Vector(frag_c.load())


def _maximum(a, b):
    # nnan drops the canonicalizing maxes that quiet signaling NaNs.
    return fx.Float32(fx.maxnumf(fx.Float32(a), fx.Float32(b), fastmath="nnan"))


# Lowest finite float32: floors an empty row's -inf max without clipping any finite logit.
_MAX_FLOOR = -3.4028234663852886e38


def _pack_bf16(values):
    # gfx942 has no native packed FP32 -> BF16 instruction; software RNE.
    bits = fx.Vector(values).bitcast(fx.Uint32)
    rounded = bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))
    # One v_perm_b32 joins both high halves; element i lands in the low half.
    words = [
        fx.Uint32(rocdl.perm_b32(rounded[i + 1], rounded[i], 0x07060302))
        for i in range(0, values.numel, 2)
    ]
    return fx.Vector.from_elements(words, fx.Uint32)


def _stage_end():
    # Compiler fences and CTA rendezvous, not a memory-counter wait.
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _schedule(pairs, count, group, exp=False):
    # 0x002 = VALU, 0x400 = TRANS; the pinned FlyDSL names only the MFMA/VMEM/DS masks.
    for _ in range(pairs):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group)
        rocdl.sched_group_barrier(0x400 if exp else 0x002, count, group)


def _schedule_exps(count, group, mfmas=2):
    # Per value: subtraction, MFMA, exp, MFMA (the fastest order measured on gfx942); gfx950's
    # K32 QK has one MFMA per value.
    for _ in range(count):
        for mask in (0x002, rocdl.mask_mfma, 0x400, rocdl.mask_mfma)[: 2 + mfmas]:
            rocdl.sched_group_barrier(mask, 1, group)


class _LdsQueue:
    """Issue-order ledger of LDS reads and ds_bpermute, which retire in order."""

    def __init__(self):
        self.issued = 0

    def issue(self, count):
        """Count issued ops; the returned ready(n) waits until the first n of them retired."""
        self.issued += count
        first = self.issued - count
        return lambda done: self.wait(first + done)

    def wait(self, upto=None, vmcnt=None):
        """Wait until the ops issued before index upto (default: all) have retired.

        lgkmcnt has 4 bits: past 15 younger ops it waits for 15, which retires more than needed.
        """
        lgkmcnt = 0 if upto is None else min(self.issued - upto, 15)
        rocdl.s_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt)


def _buffer(tensor, size_bytes):
    """Byte-addressed buffer pointer over tensor; loads at or past size_bytes return zero."""
    return rocdl.make_buffer_ptr(
        fx.recast_iter(fx.Int8, fx.get_iter(tensor)), num_records_bytes=size_bytes
    )


def _buffer_at(address, size_bytes):
    """Byte-addressed buffer pointer at a 16-byte aligned 64-bit global address; loads at or
    past size_bytes return zero."""
    pointer = fx.inttoptr(
        fx.PointerType.get(fx.Int8.ir_type, fx.AddressSpace.Global, 16),
        fx.Int64(address),
    )
    return rocdl.make_buffer_ptr(pointer, num_records_bytes=size_bytes)


def _buffer_words(buffer, offset, words=4):
    data = fx.make_rmem_tensor(4 * words, fx.Int8)
    fx.copy(
        fx.make_copy_atom(rocdl.BufferCopy(32 * words), fx.Int8),
        fx.make_view(buffer + offset, fx.make_layout(4 * words, 1)),
        data,
    )
    return fx.Vector(data.load()).bitcast(fx.Int32)


def _buffer_store_words(words, buffer, offset):
    words = fx.Vector(words)
    data = fx.make_rmem_tensor(4 * words.numel, fx.Int8)
    data.store(words.bitcast(fx.Int8))
    fx.copy(
        fx.make_copy_atom(rocdl.BufferCopy(32 * words.numel), fx.Int8),
        data,
        fx.make_view(buffer + offset, fx.make_layout(4 * words.numel, 1)),
    )


def _load_lds(storage, offset, immediate=0):
    # Volatile keeps the hand-written order; the LDS variable's GEP gives it an alias scope.
    ptr = fx.get_iter(storage) + (fx.Int32(offset) + immediate)
    ptr = fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 16), ptr
    )
    return fx.Vector(fx.generic_load(ptr, count=4, volatile=True))


def _write_address(address, words):
    ptr_type = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 16)
    lds = fx.make_view(fx.inttoptr(ptr_type, fx.Int32(address)), fx.make_layout(4, 1))
    data = fx.make_rmem_tensor(4, fx.Int32)
    data.store(fx.Vector(words).bitcast(fx.Int32))
    fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32), data, fx.Tensor(lds))


def _lds_pointer(address, dtype):
    ptr_type = fx.PointerType.get(
        dtype.ir_type, fx.AddressSpace.Shared, dtype.width // 8
    )
    return fx.inttoptr(ptr_type, fx.Int32(address))


def _lds_load(address, dtype=fx.Int32):
    """One value at an LDS byte address."""
    return fx.generic_load(_lds_pointer(address, dtype))


def _lds_store(address, value, dtype=fx.Int32):
    fx.generic_store(_lds_pointer(address, dtype), dtype(value))


@flyc.jit
def _rescale(o0, o1, row_sum, old_max, new_max, ballot):
    o0, o1, row_sum = fx.Vector(o0), fx.Vector(o1), fx.Float32(row_sum)
    if ballot != fx.Int64(0):
        correction = _exp(fx.Float32(old_max) - fx.Float32(new_max))
        o0, o1, row_sum = o0 * correction, o1 * correction, row_sum * correction
    return o0, o1, row_sum


# ---- D256 BM128/BN64 eight-wave pipeline: MFMA fragments, softmax, LDS ----

BM, BN, D, THREADS = 128, 64, 256, 512


def _mma(depth=16):
    """Eight waves along the query rows, each 16x16xdepth BF16 MFMAs (depth 32: gfx950)."""
    return fx.make_tiled_mma(
        fx.make_mma_atom(rocdl.MFMA(16, 16, depth, fx.BFloat16)),
        fx.make_layout((1, 8, 1), (1, 1, 0)),
    )


def _read_q_words(resource, offset, lane):
    """Q words of the lane: four-word load k holds the eight BF16 dims from
    32k + 8 * (lane >> 4) on; offsets are in bytes."""
    parts = [
        _buffer_words(resource, offset + (lane >> 4) * 16 + k * 64) for k in range(8)
    ]
    return fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)


def _read_q_g2r(resource, offset, lane, depth=16):
    words = _read_q_words(resource, offset, lane)
    frag = _mma(depth).make_fragment_B(
        fx.make_rmem_tensor(fx.make_layout((BM, D), (1, BM)), fx.BFloat16)
    )
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _qk(q, k, depth=16, order="kmn"):
    mma = _mma(depth)
    acc = mma.make_fragment_C(
        fx.make_rmem_tensor(fx.make_layout((32, BM), (1, 32)), fx.Float32)
    )
    acc.fill(0.0)
    fx.gemm(mma, acc, k, q, acc, traversal_order=order)
    return acc.load()


def _sum(values):
    partials = [values[i] + values[i + 1] for i in range(0, 16, 2)]
    for width in (4, 2, 1):
        partials = [partials[2 * i] + partials[2 * i + 1] for i in range(width)]
    return partials[0]


def _max(values):
    # Keep MFMA -> max dependencies visible to LLVM: an opaque max tree failed
    # repeated bit-exact checks, and maxnum avoids maximumf's NaN expansion.
    parts = [fx.maxnumf(values[i], values[i + 1]) for i in range(0, 16, 2)]
    for width in (4, 2, 1):
        parts = [fx.maxnumf(parts[2 * i], parts[2 * i + 1]) for i in range(width)]
    return parts[0]


def _cross(value, addresses, queue):
    bits = fx.Float32(value).bitcast(fx.Int32)
    result = tuple(
        fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, address, bits)).bitcast(fx.Float32)
        for address in addresses
    )
    queue.issue(len(addresses))
    # LLVM would sink the ds_bpermutes below the LDS reads that the queue counts after them.
    rocdl.sched_barrier(0)
    return result


def _exps(values, maximum):
    # values are the rounded S * scale the max came from: an FMA overshoots by ulp / 2.
    return fx.Vector.from_elements(
        [_exp(values[i] - maximum) for i in range(16)], fx.Float32
    )


def _key_row(lane):
    """Key of the lane's first K read: lane 16g + 4b + a reads keys a + 8b (+ 4n, + 32 per
    half), in the mask order of sparse_prefill_order_masks."""
    return (lane & 3) + ((lane & 12) << 1)


def _k_phase(token):
    return (token & 3) | ((token >> 3) & 1) * 4


def _k_lds_address(token, dimension):
    return token * 512 + ((dimension // 2) ^ (_k_phase(token) * 4)) * 4


def _v_lds_address(token, dimension):
    return token * 512 + dimension * 2


def _copy_coordinates(wave, packet, page=0):
    token = (wave & 3) * 16 + packet if page == 4 else (wave & 3) + packet * 4
    return token, (wave >> 2) * 128


def _k_read_order(lower_first):
    """(n, k) of a K half's 16 reads in issue order; lower_first takes k 0-3 of both n first."""
    if lower_first:
        return [
            (n, k + high * 4) for high in range(2) for n in range(2) for k in range(4)
        ]
    return [(n, k) for n in range(2) for k in range(8)]


def _k_reads(addresses, half, lower_first=False):
    """(slot, address, immediate) of one K half's 16 LDS reads, in issue order."""
    return [
        (n * 8 + k, addresses[n * 2 + k % 2], half * 16384 + (k // 2) * 128)
        for n, k in _k_read_order(lower_first)
    ]


def _v_reads(address, half):
    """(slot, address, immediate) of one V half's 16 LDS reads, in issue order."""
    return [
        (block * 8 + row, address, half * 256 + block * 16384 + row * 512)
        for block in range(2)
        for row in range(8)
    ]


def _lds_words(parts):
    return fx.Vector.from_elements(
        [part[i] for part in parts for i in range(part.numel)], fx.Int32
    )


class _ReadPlan(NamedTuple):
    """LDS reads of one K or V operand, each issued with load from storage."""

    storage: object
    reads: list  # (slot, address, immediate) in issue order
    load: Callable  # load(storage, address, immediate) -> words


def _read_s2r(plan, queue):
    """Issue a _ReadPlan; ready(n) waits until its first n reads have landed."""
    parts = [None] * len(plan.reads)
    for slot, address, immediate in plan.reads:
        parts[slot] = plan.load(plan.storage, address, immediate)
    return _lds_words(parts), queue.issue(len(plan.reads))


def _k_operands(words):
    # MFMA-only uses leave the loads AV-class; the scheduler then reverts the QK interleave.
    _keep_vgpr(words)
    rocdl.sched_barrier(0)
    frag = fx.make_rmem_tensor(fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16)
    frag.store(words.bitcast(fx.BFloat16))
    return fx.make_view(fx.get_iter(frag), fx.make_layout((4, 2, 16), (1, 64, 4)))


def _v_operands(values, step, ready=None):
    raw = fx.Vector(values)
    row = step * 4
    if ready is not None:
        ready(row + 4)
        _keep_vgpr(
            fx.Vector.from_elements(
                [raw[i] for i in range(row * 4, row * 4 + 16)], fx.Int32
            )
        )
        rocdl.sched_barrier(0)
    words = []
    for n in range(8):
        selector = 0x03020706 if n % 2 else 0x01000504
        words.extend(
            fx.Int32(
                rocdl.perm_b32(
                    raw[(row + r) * 4 + n // 2],
                    raw[(row + r + 1) * 4 + n // 2],
                    selector,
                )
            )
            for r in (0, 2)
        )
    return fx.Vector.from_elements(words, fx.Int32)


def _new_max(center):
    """m' (the row max + 1 once it exceeds m + 7) and its ballot, from center's lane and cross-lane maxima."""
    _, candidate, maxima, scale, maximum = center
    candidate = (
        _maximum(_maximum(candidate, maxima[0]), _maximum(maxima[1], maxima[2])) * scale
    )
    predicate = candidate > maximum + 7.0
    ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, predicate))
    return predicate.select(candidate + 1.0, maximum), ballot


def _sum_and_max(sum_max):
    """l + the lane row sums and the lane max of S, from sum_max = (S, lane sum, other sums, l)."""
    current, total, sums, row_sum = sum_max
    return row_sum + ((total + sums[0]) + (sums[1] + sums[2])), _max(current)


def _pv(
    probabilities,
    values,
    output,
    ready=None,
    prepared=None,
    center=None,
    sum_max=None,
    steps=4,
):
    """O += P V over one V half; center and sum_max fold softmax VALU work into its MFMAs.

    sum_max = (S, lane sum, other sums, l) also returns l + row sum and the lane max of S;
    center = (S, lane max, other maxima, scale, m) returns S * scale, m' and the ballot.
    """
    p = fx.Vector(probabilities)
    acc = [
        fx.Vector.from_elements([output[n * 4 + i] for i in range(4)], fx.Float32)
        for n in range(8)
    ]
    prepared_steps = 0 if prepared is None else prepared.numel // 16
    operands = (
        _v_operands(values, 0, ready)
        if prepared is None
        else fx.Vector.from_elements([prepared[i] for i in range(16)], fx.Int32)
    )
    scaled = []
    rocdl.sched_barrier(0)
    for step in range(steps):
        if step + 1 < prepared_steps:
            following = fx.Vector.from_elements(
                [prepared[(step + 1) * 16 + i] for i in range(16)], fx.Int32
            )
        else:
            following = (
                _v_operands(values, step + 1, ready) if step + 1 < steps else operands
            )
        b = fx.Vector.from_elements([p[step * 4 + i] for i in range(4)], fx.BFloat16)
        for n in range(8):
            a = fx.Vector.from_elements(
                [operands[n * 2 + i] for i in range(2)], fx.Int32
            ).bitcast(fx.Int16)
            acc[n] = _mfma(a, b.bitcast(fx.Int16), acc[n])
        if center is not None:
            if step == 0:
                new_max, ballot = _new_max(center)
            for i in range(step * 4, step * 4 + 4):
                scaled.append(fx.Float32(center[0][i]) * center[3])
        if sum_max is not None and step == 3:
            row_sum, candidate = _sum_and_max(sum_max)
        if step + 1 < steps and step + 1 >= prepared_steps:
            _schedule(8, 3 if center is not None else 2, 20 + step)
        elif sum_max is not None:
            _schedule(8, 3, 23)
        rocdl.sched_barrier(0)
        operands = following
    result = fx.Vector.from_elements(
        [acc[n][i] for n in range(8) for i in range(4)], fx.Float32
    )
    if center is not None:
        # Pinned: unpinned, LLVM sinks the scaling into the next memory stage.
        scaled = _pin(fx.Vector.from_elements(scaled, fx.Float32))
        return result, scaled, new_max, ballot
    if sum_max is not None:
        return result, row_sum, candidate
    return result


# ---- gfx950 (CDNA4): K32 MFMAs, transposed V reads, native BF16, 64-bank swizzles ----
#
# The union pipeline keeps its stages and data flow (lgkmcnt waits capped at 15, the
# rendezvous its double buffer makes redundant dropped: sparse_prefill_union._body); the
# MFMA shape, the V operand path, the BF16 conversion and the K/V LDS layouts change
# (g = lane >> 4 below).
#   QK: per K half, two M tiles of eight v_mfma_f32_16x16x32_bf16; step s takes K read
#       m * 8 + s and Q part s, both dims 32s + 8g .. + 7. The K reads keep gfx942's keys
#       and the S order (value i + 4m), so the masks of sparse_prefill_order_masks are
#       unchanged.
#   PV: per V half, two K32 steps (keys 32s + 8g .. + 7: P values 8s .. 8s + 7) over eight
#       16-dim groups n; the A operand of (s, n) is two ds_read_b64_tr_b16 of four keys.
#       Output word 4n + i of a half is dim 32g + 4(n ^ (g & 1)) + i (_v_tr_bases950).
#   LDS: K and V are half-major: dims half h of key t at _lds_row950(t, h), so one 16-byte
#       (x4) LDS DMA writes four keys' half rows, 1 KiB. The 16-byte chunks of a half row are
#       swizzled by _k_swizzle950 (K) and _v_swizzle950 (V); the DMA (sparse_prefill_union)
#       and the reads below take their addresses from these. Each swizzle stays inside its
#       256-byte half row and leaves the bits the read immediates move alone.


HALF950 = 64 * 256  # bytes of one dims half of a 64-key K or V tile in LDS


def _k_phase950(token):
    """4-bit K chunk swizzle from token bits 0, 1, 3, 4 (keys of one read: token & 3 and token >> 3 & 3). A bijection,
    so the 16 keys of a 16-lane ds_read_b128 group take 16 chunk positions of a 256-byte bank row; bit 1 is the parity
    of the four bits, so lanes 0-7 with 32-39 (bit 4 equal, dims 32 bytes apart) take 16 distinct positions too.
    """
    a0, a1 = token & 1, (token >> 1) & 1
    b0, b1 = (token >> 3) & 1, (token >> 4) & 1
    return a0 | (a0 ^ a1 ^ b0 ^ b1) * 2 | a1 * 4 | b0 * 8


def _k_swizzle950(token, chunk):
    """K LDS swizzle: position of 16-byte chunk `chunk` (0-15) in key `token`'s half row. An XOR, so its own inverse:
    position p holds chunk _k_swizzle950(token, p)."""
    return chunk ^ _k_phase950(token)


def _v_swizzle950(token, chunk):
    """V LDS swizzle: the chunk XORed with gfx942's K phase _k_phase (K uses _k_phase950). In BF16 elements of (key, (dim, dims half)),
    it is Swizzle<2,3,4> o Swizzle<1,5,5> over the layout (64,(128,2)):(128,(1,8192)) of _lds_row950's rows.
    """
    return chunk ^ _k_phase(token)


def _lds_row950(token, half):
    """Byte address of dims half `half` of key `token` in a half-major K or V tile."""
    return half * HALF950 + token * 256


def _k_lds_address950(token, chunk):
    """Byte address of 16-byte chunk `chunk` (0-31, dims 8 * chunk ..) of key `token` in k_lds."""
    half, chunk = chunk >> 4, _k_swizzle950(token, chunk & 15)
    return _lds_row950(token, half) + chunk * 16


def _k_bases950(lane):
    """Per-lane addresses of _k_reads950: key _key_row(lane), chunks (lane >> 4) + 4j, j = 0..3."""
    return tuple(
        _k_lds_address950(_key_row(lane), (lane >> 4) + 4 * j) for j in range(4)
    )


def _k_reads950(addresses, half, lower_first=False):
    """(slot, address, immediate) of one K half's 16 LDS reads (_k_reads' order and slots; read (n, k) is key
    key_row + 4n + 32 * half, dims 8g + 32k .. + 7); addresses = _k_bases950(lane)."""
    return [
        (n * 8 + k, addresses[k & 3], _lds_row950(half * 32 + n * 4, k >> 2))
        for n, k in _k_read_order(lower_first)
    ]


def _v_lds_address950(token, granule):
    """Byte address of 8-byte granule `granule` (0-63, dims 4 * granule .. + 3) of value `token` in v_lds."""
    chunk = _v_swizzle950(token, (granule & 31) >> 1)
    return _lds_row950(token, granule >> 5) + chunk * 16 + (granule & 1) * 8


def _v_tr_bases950(lane):
    """Per-lane addresses of a V half's eight 16-dim groups n for ds_read_b64_tr_b16: lane 16g + 4r + q reads key
    8g + r, dims 32q + 4(n ^ (q & 1)) .. + 3 (n's XOR is the address bits 3-5). The q & 1 term gives the lanes of a
    32-lane group both 8-byte halves of the 16-byte chunks, so they cover the 64 banks once; output word 4n + i is
    then dim 32g + 4(n ^ (g & 1)) + i (_o_piece950). Keys + 4h + 32s and dims + 128 go by immediate.
    """
    q = lane & 3
    token = (lane >> 4) * 8 + ((lane & 15) >> 2)
    base = _v_lds_address950(token, q * 8 + (q & 1))
    return tuple(base ^ (n * 8) for n in range(8))


def _v_tr_reads950(bases, half):
    """(slot, address, immediate) of one V half's 32 transposed reads in issue order: PV step s, group n, key quad h
    (keys 32s + 8g + 4h .. + 3); slot (s * 8 + n) * 2 + h, so PV step s needs the first 16 (s + 1).
    """
    return [
        ((s * 8 + n) * 2 + h, bases[n], _lds_row950(s * 32 + h * 4, half))
        for s in range(2)
        for n in range(8)
        for h in range(2)
    ]


def _load_lds_tr(storage, address, immediate=0):
    """ds_read_b64_tr_b16 at an LDS byte address; 2 words. Lane 4r + q of each 16-lane group reads four BF16 at its
    address; lane i then holds element i % 4 of lane 4r + i // 4's read as BF16 r = 0..3.
    """
    pointer = fx.recast_iter(
        fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Shared, 8),
        fx.get_iter(storage) + (fx.Int32(address) + immediate),
    )
    data = fx.make_rmem_tensor(4, fx.BFloat16)
    fx.copy(
        fx.make_copy_atom(rocdl.cdna4.LDSReadTrans(16, 64), fx.BFloat16),
        fx.make_view(pointer, fx.make_layout(4, 1)),
        data,
    )
    return fx.Vector(data.load()).bitcast(fx.Int32)


def _mfma32(a, b, c):
    """One 16x16x32 BF16 MFMA (gfx950) on register vectors: a/b hold 8 BF16 values (or their 4 words), c 4 FP32."""
    frags = []
    for value in (a, b):
        value = fx.Vector(value)
        frag = fx.make_rmem_tensor(8, fx.BFloat16)
        frag.store(value.bitcast(fx.BFloat16) if value.dtype.width == 32 else value)
        frags.append(frag)
    frag_c = fx.make_rmem_tensor(4, fx.Float32)
    frag_c.store(fx.Vector(c))
    mma = fx.make_mma_atom(rocdl.MFMA(16, 16, 32, fx.BFloat16))
    fx.gemm(mma, frag_c, frags[0], frags[1], frag_c)
    return fx.Vector(frag_c.load())


def _pack_bf16_950(values):
    """_pack_bf16 with v_cvt_pk_bf16_f32 (round to nearest even): element i in the low half."""
    values = fx.Vector(values)
    return fx.Vector.from_elements(
        [
            fx.Uint32(rocdl.cvt_pk_bf16_f32(values[i], values[i + 1]))
            for i in range(0, values.numel, 2)
        ],
        fx.Uint32,
    )


def _k_operands950(words):
    # As _k_operands, with the K32 operands: read m * 8 + s holds K step s of M tile m.
    _keep_vgpr(words)
    rocdl.sched_barrier(0)
    frag = fx.make_rmem_tensor(fx.make_layout((8, 8, 2), (1, 8, 64)), fx.BFloat16)
    frag.store(words.bitcast(fx.BFloat16))
    return fx.make_view(fx.get_iter(frag), fx.make_layout((8, 2, 8), (1, 64, 8)))


def _qk950(q, k):
    """_qk with K32 MFMAs, issued a K step at a time (both M tiles of a step together)."""
    return _qk(q, k, 32, "mkn")


def _pv950_words(step, n):
    """V words of PV MFMA (step, n): reads (step * 8 + n) * 2 + h, h = 0 (keys 0-3 of the lane's eight), 1 (4-7)."""
    return [(step * 8 + n) * 4 + i for i in range(4)]


def _o_piece950(i, g):
    """Output words of the lane's dims 8i .. 8i + 7 of a half in lane group g (dim 32g + 4n + j is word
    4(n ^ (g & 1)) + j, _v_tr_bases950)."""
    return [i * 8 + (e ^ (g & 1) * 4) for e in range(8)]


def _v_operands950(values, step, ready=None):
    """The 32 words of PV step `step`; with ready, waits for its reads and keeps them in VGPRs (as _v_operands)."""
    raw = fx.Vector(values)
    words = fx.Vector.from_elements([raw[step * 32 + i] for i in range(32)], fx.Int32)
    if ready is not None:
        ready(16 * (step + 1))
        _keep_vgpr(words)
        rocdl.sched_barrier(0)
    return words


def _pv950(
    probabilities,
    values,
    output,
    ready=None,
    prepared=None,
    center=None,
    sum_max=None,
):
    """_pv on gfx950: O += P V over one V half in two K32 steps of eight MFMAs. values holds the words of one
    _v_tr_reads950 plan; prepared (any value) says the memory stage already ran _v_operands950(values, 0, ready).
    center and sum_max as for _pv."""
    p, raw = fx.Vector(probabilities), fx.Vector(values)
    acc = [
        fx.Vector.from_elements([output[n * 4 + i] for i in range(4)], fx.Float32)
        for n in range(8)
    ]
    scaled = []
    rocdl.sched_barrier(0)
    for step in range(2):
        if ready is not None and (step or prepared is None):
            _v_operands950(raw, step, ready)
        b = fx.Vector.from_elements([p[step * 8 + i] for i in range(8)], fx.BFloat16)
        for n in range(8):
            a = fx.Vector.from_elements(
                [raw[i] for i in _pv950_words(step, n)], fx.Int32
            )
            acc[n] = _mfma32(a, b, acc[n])
        if center is not None:
            if step == 0:
                new_max, ballot = _new_max(center)
            for i in range(step * 8, step * 8 + 8):
                scaled.append(fx.Float32(center[0][i]) * center[3])
        if sum_max is not None and step == 1:
            row_sum, candidate = _sum_and_max(sum_max)
        if center is not None or sum_max is not None:
            _schedule(8, 2, 20 + step)
        rocdl.sched_barrier(0)
    result = fx.Vector.from_elements(
        [acc[n][i] for n in range(8) for i in range(4)], fx.Float32
    )
    if center is not None:
        # Pinned, as in _pv.
        scaled = _pin(fx.Vector.from_elements(scaled, fx.Float32))
        return result, scaled, new_max, ballot
    if sum_max is not None:
        return result, row_sum, candidate
    return result

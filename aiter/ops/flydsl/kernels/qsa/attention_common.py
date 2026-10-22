# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Settings and gfx942 BF16 D256 helpers shared by the QSA sparse attention kernels.

The first part holds the per-platform calibration and the private resource
policy. The platform entry is picked by gfx model and CU count (nearest
calibrated CU count of the same model) and read once per process;
AITER_QSA_PLATFORM overrides fields of the picked entry, e.g.
``AITER_QSA_PLATFORM="union_step=3.0,max_packed_kv_bytes=0"``.

The second part holds raw-buffer operations, register pinning and pipeline
barriers. The third part holds the BM128/BN64 MFMA, softmax and LDS pieces
of the eight-wave staggered D256 pipeline that the union kernel runs.
"""

import dataclasses
import functools
import logging
import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

# ---- Platform calibration and private resource policy ----

# Resource policy; private, not configurable from outside.
WORKSPACE_CACHE = 8  # layouts kept per process; captured workspaces are never evicted
ARENA_GROWTH_PERCENT = 125  # a growing arena takes at least this share of its size
SCRATCH_ALIGN = 256  # bytes between the scratch views of an arena
UNION_SCRATCH_LIMIT = 4 << 30  # union scratch bytes; past it a layout runs direct only


@dataclasses.dataclass(frozen=True)
class Platform:
    """Calibrated per GPU; the route-model terms are microseconds."""

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
        values[name] = type(getattr(entry, name))(value)
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
        """Wait until the ops issued before index upto (default: all) have retired."""
        lgkmcnt = 0 if upto is None else self.issued - upto
        rocdl.s_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt)


def _buffer(tensor, size_bytes):
    """Byte-addressed buffer pointer over tensor; loads at or past size_bytes return zero."""
    return rocdl.make_buffer_ptr(
        fx.recast_iter(fx.Int8, fx.get_iter(tensor)), num_records_bytes=size_bytes
    )


def _buffer_words(buffer, offset, words=4):
    data = fx.make_rmem_tensor(4 * words, fx.Int8)
    fx.copy(
        fx.make_copy_atom(rocdl.BufferCopy(32 * words), fx.Int8),
        fx.make_view(buffer + offset, fx.make_layout(4 * words, 1)),
        data,
    )
    return fx.Vector(data.load()).bitcast(fx.Int32)


def _buffer_store_words(words, buffer, offset):
    data = fx.make_rmem_tensor(16, fx.Int8)
    data.store(fx.Vector(words).bitcast(fx.Int8))
    fx.copy(
        fx.make_copy_atom(rocdl.BufferCopy128b(), fx.Int8),
        data,
        fx.make_view(buffer + offset, fx.make_layout(16, 1)),
    )


def _load_lds(storage, offset, immediate=0):
    # Volatile keeps the hand-written order; the LDS variable's GEP gives it an alias scope.
    ptr = fx.to_llvm_ptr(fx.get_iter(storage) + (fx.Int32(offset) + immediate))
    return fx.Vector(
        llvm.load(
            ir.VectorType.get([4], fx.Int32.ir_type), ptr, alignment=16, volatile_=True
        )
    )


def _write_address(address, words):
    ptr_type = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 16)
    lds = fx.make_view(fx.inttoptr(ptr_type, fx.Int32(address)), fx.make_layout(4, 1))
    data = fx.make_rmem_tensor(4, fx.Int32)
    data.store(fx.Vector(words).bitcast(fx.Int32))
    fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32), data, fx.Tensor(lds))


@flyc.jit
def _rescale(o0, o1, row_sum, old_max, new_max, ballot):
    o0, o1, row_sum = fx.Vector(o0), fx.Vector(o1), fx.Float32(row_sum)
    if ballot != fx.Int64(0):
        correction = _exp(fx.Float32(old_max) - fx.Float32(new_max))
        o0, o1, row_sum = o0 * correction, o1 * correction, row_sum * correction
    return o0, o1, row_sum


# ---- D256 BM128/BN64 eight-wave pipeline: MFMA fragments, softmax, LDS ----

BM, BN, D, THREADS = 128, 64, 256, 512


def _mma():
    return fx.make_tiled_mma(
        fx.make_mma_atom(rocdl.MFMA(16, 16, 16, fx.BFloat16)),
        fx.make_layout((1, 8, 1), (1, 1, 0)),
    )


def _read_q_g2r(resource, offset, lane):
    parts = [
        _buffer_words(resource, offset + (lane >> 4) * 16 + k * 64) for k in range(8)
    ]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma().make_fragment_B(
        fx.make_rmem_tensor(fx.make_layout((BM, D), (1, BM)), fx.BFloat16)
    )
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _qk(q, k):
    mma = _mma()
    acc = mma.make_fragment_C(
        fx.make_rmem_tensor(fx.make_layout((32, BM), (1, 32)), fx.Float32)
    )
    acc.fill(0.0)
    fx.gemm(mma, acc, k, q, acc, traversal_order="kmn")
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


def _center_one(value, scale, maximum):
    return fx.Float32(fx.fma(fx.Float32(value), scale, -fx.Float32(maximum)))


def _center(values, scale, maximum):
    return fx.Vector.from_elements(
        [_center_one(values[i], scale, maximum) for i in range(16)], fx.Float32
    )


def _exps(values):
    return fx.Vector.from_elements([_exp(values[i]) for i in range(16)], fx.Float32)


def _k_phase(token):
    return (token & 3) | ((token >> 3) & 1) * 4


def _k_lds_address(token, dimension):
    return token * 512 + ((dimension // 2) ^ (_k_phase(token) * 4)) * 4


def _v_lds_address(token, dimension):
    return token * 512 + dimension * 2


def _copy_coordinates(wave, packet, page=0):
    token = (wave & 3) * 16 + packet if page == 4 else (wave & 3) + packet * 4
    return token, (wave >> 2) * 128


def _k_reads(addresses, half, lower_first=False):
    """(slot, address, immediate) of one K half's 16 LDS reads, in issue order."""
    order = (
        [(n, k + high * 4) for high in range(2) for n in range(2) for k in range(4)]
        if lower_first
        else [(n, k) for n in range(2) for k in range(8)]
    )
    return [
        (n * 8 + k, addresses[n * 2 + k % 2], half * 16384 + (k // 2) * 128)
        for n, k in order
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
        [part[i] for part in parts for i in range(4)], fx.Int32
    )


def _read_s2r(plan, queue):
    """Issue a (storage, reads) plan; ready(n) waits until its first n reads have landed."""
    storage, reads = plan
    parts = [None] * len(reads)
    for slot, address, immediate in reads:
        parts[slot] = _load_lds(storage, address, immediate)
    return _lds_words(parts), queue.issue(len(reads))


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
    center = (S, lane max, other maxima, scale, m) returns S * scale - m', m' and the ballot.
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
    centered = []
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
            current, candidate, maxima, scale, maximum = center
            if step == 0:
                candidate = (
                    _maximum(
                        _maximum(candidate, maxima[0]), _maximum(maxima[1], maxima[2])
                    )
                    * scale
                )
                predicate = candidate > maximum + 7.0
                ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, predicate))
                new_max = predicate.select(candidate + 1.0, maximum)
            for i in range(step * 4, step * 4 + 4):
                centered.append(_center_one(current[i], scale, new_max))
        if sum_max is not None and step == 3:
            current, total, sums, row_sum = sum_max
            row_sum = row_sum + ((total + sums[0]) + (sums[1] + sums[2]))
            candidate = _max(current)
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
        # Pinned: unpinned, LLVM moves these FMAs out of the stage (0.4-1.3% slower).
        centered = _pin(fx.Vector.from_elements(centered, fx.Float32))
        return result, centered, new_max, ballot
    if sum_max is not None:
        return result, row_sum, candidate
    return result

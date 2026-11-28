# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Settings and gfx942/gfx950 D256 helpers shared by the QSA sparse attention kernels.

The first part holds the per-platform calibration and the private resource
policy. The platform entry is picked by gfx model and CU count (nearest
calibrated CU count of the same model) and read once per process; FP8 K/V
replace its route terms (FP8_ROUTES); AITER_QSA_PLATFORM overrides fields of the
picked entry, e.g. ``AITER_QSA_PLATFORM="union_step=3.0,max_packed_kv_bytes=0"``.

The second part holds raw-buffer operations and pipeline barriers. The third
part holds the QK and PV MFMA loops and the lazy softmax rescale, each defined
once for the union and both direct kernels. The fourth part holds the
BM128/BN64 MFMA, softmax and LDS pieces of the eight-wave staggered D256
pipeline that the union kernel runs. The fifth part holds their gfx950 (CDNA4)
forms: 16x16x32 MFMAs, ds_read_b64_tr_b16 V operands, v_cvt_pk_bf16_f32 and K/V LDS
swizzles for 64 banks. On gfx950 the union kernel and the packed direct kernel take
their gfx950 paths; the raw direct kernel runs as on gfx942. The sixth part holds what
both FP8 K/V kernels (gfx942) share: the FP8 MFMA, the lazy max of their softmax, the
BF16 output rounding and the FP8 quantization of Q. Everything below FlyDSL's stable API
(flydsl.expr, flydsl.expr.rocdl __all__, flydsl.compiler) is in the low-level
section, each item with the reason it stays: empty inline asm register pins (no
public asm entry or placement hint), the target-features attribute, ROCDL
operations FlyDSL wraps without exporting, volatile LDS reads and the lgkmcnt
ledger of the union pipeline's cross-wave LDS reuse.
"""

import dataclasses
import functools
import logging
import math
import os
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

# ---- Low-level primitives: everything below FlyDSL's stable API ----

# ---- Empty inline asm ----


def _pin_i32(value):
    """value materialized in a VGPR at this point (tied empty asm)."""
    # No fx equivalent; LLVM would sink or rematerialize the value into its uses.
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
    """Vector values materialized in VGPRs at this point, chunk words per tied empty asm."""
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
    """An empty asm use that keeps words in VGPRs (not AV-class operands of MFMA only)."""
    # A tied pin's def would cost an s_nop; only a use is needed here.
    words = fx.Vector(words)
    for start in range(0, words.numel, chunk):
        part = fx.Vector.from_elements(
            [words[i] for i in range(start, start + chunk)], words.dtype
        )
        llvm.inline_asm(None, [part.ir_value()], "", "v", has_side_effects=True)


def hold(*values):
    """Empty asm uses: the FP32 values are computed at this point."""
    # MachineSink ignores sched_barrier: results used only after a later branch would sink
    # past it and run after the last MFMA.
    for value in values:
        llvm.inline_asm(
            None, [fx.Float32(value).ir_value()], "", "v", has_side_effects=True
        )


def compiler_fence():
    """Empty asm with a memory clobber: a compiler memory barrier, no instruction."""
    # fx.memory_fence emits a real fence; this only keeps MachineSink from moving loads
    # issued before it into a later block.
    llvm.inline_asm(None, [], "", "~{memory}", has_side_effects=True)


# ---- Kernel attributes ----


def kernel_attrs():
    """value_attrs of the sparse prefill kernels.

    Packed FP32 VALU (v_pk_*_f32) makes the union phase loop about 8% slower, and FlyDSL
    has no public target-feature option (no compile hint carries one): llvm.passthrough
    forwards the LLVM function attribute.
    """
    return {"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]}


# ---- ROCDL operations without a stable FlyDSL wrapper ----

# sched_group_barrier masks; fx.rocdl names only the MFMA, VMEM and DS ones.
VALU, MFMA, TRANS = 0x002, 0x008, 0x400


def _uniform(value):
    """Lane 0's value for the whole wave (v_readfirstlane_b32): an SGPR."""
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(value)))


def readlane(value, lane):
    """Lane ``lane``'s value for the whole wave (v_readlane_b32)."""
    return fx.Int32(rocdl.readlane(fx.Int32.ir_type, value, lane))


def ballot(predicate):
    """The wave's predicate bits."""
    return fx.Int64(rocdl.ballot(fx.Int64.ir_type, predicate))


def bpermute(address, value):
    """The value of the lane at byte ``address`` (ds_bpermute_b32)."""
    # fx.gpu.shuffle_idx scales the lane and checks its range on every call.
    return fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, address, value))


def perm_b32(high, low, selector):
    """v_perm_b32: bytes of (high, low) picked by selector."""
    return rocdl.perm_b32(high, low, selector)


def sched_barrier(mask=0):
    """Compiler scheduling fence: only instruction classes in mask may cross it."""
    rocdl.sched_barrier(mask)


def sched_group_barrier(mask, size, group):
    """Schedule size instructions of class mask into pipeline group ``group``."""
    # rocdl.sched_mfma etc. fix group 0 and name no VALU/TRANS class.
    rocdl.sched_group_barrier(mask, size, group)


def s_barrier():
    """Workgroup rendezvous without memory fences."""
    # fx.gpu.barrier adds workgroup fences, i.e. waits for every memory counter.
    rocdl.s_barrier()


def s_setprio(level):
    """Wave issue priority."""
    rocdl.s_setprio(level)


def buffer_load_lds(resource, lds, size, voffset, soffset, offset):
    """LDS DMA of size bytes per lane (buffer_load ... lds) without a wait."""
    # rocdl.BufferCopyLDS* copy atoms add their own vmcnt wait.
    rocdl.raw_ptr_buffer_load_lds(resource, lds, size, voffset, soffset, offset)


def buffer_load_i16(resource, voffset, soffset):
    """One 16-bit buffer load with a wave-uniform SGPR soffset."""
    return rocdl.raw_ptr_buffer_load(fx.Int16.ir_type, resource, voffset, soffset, 0)


def dpp_move(value, control):
    """The 32-bit value moved across lanes by one DPP control (row-local, no LDS crossbar)."""
    raw = value.ir_value()
    return fx.Int32(
        rocdl.update_dpp(fx.Int32.ir_type, raw, raw, control, 0xF, 0xF, True)
    )


def lanes_below(mask):
    """Set bits of the 64-bit mask below this lane (v_mbcnt_lo/hi)."""
    low = rocdl.mbcnt_lo(
        fx.Int32.ir_type, fx.Int32(mask).ir_value(), fx.Int32(0).ir_value()
    )
    return fx.Int32(
        rocdl.mbcnt_hi(fx.Int32.ir_type, fx.Int32(mask >> 32).ir_value(), low)
    )


def row_broadcast(value, lane):
    """Lane ``lane`` of each 16-lane row (DPP row_newbcast); LLVM folds it into the VALU
    instruction that uses it."""
    # fx.gpu shuffles go through ds_bpermute or ds_swizzle and cannot fold into a VALU.
    return fx.Float32(
        rocdl.update_dpp(
            fx.Float32.ir_type,
            llvm.mlir_poison(fx.Float32.ir_type),
            fx.Float32(value).ir_value(),
            0x150 + lane,
            0xF,
            0xF,
            True,
        )
    )


def cvt_pk_fp8(a, b, word=None, high=False):
    """FP32 a, b as two E4M3FNUZ bytes (round to nearest even) in the low (high) half of
    word, its other half kept (v_cvt_pk_fp8_f32); no word starts a new one."""
    # Float32.to(Float8E4M3FNUZ) converts one value per instruction.
    old = llvm.mlir_poison(fx.Int32.ir_type) if word is None else fx.Int32(word)
    return fx.Int32(rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, a, b, old, high))


def cvt_pk_bf16_f32(a, b):
    """FP32 a, b as two BF16 (round to nearest even), a in the low half
    (v_cvt_pk_bf16_f32, gfx950)."""
    return fx.Uint32(rocdl.cvt_pk_bf16_f32(a, b))


def log2(value):
    """log2 by one bare v_log_f32."""
    # fx.math.log2 adds denormal range reduction.
    return fx.Float32(rocdl.log(fx.Float32.ir_type, fx.Float32(value).ir_value()))


def rcp(value):
    """1 / value by one bare v_rcp_f32."""
    # Division lowers to a correctly rounded sequence of several instructions.
    return fx.Float32(rocdl.rcp(fx.Float32.ir_type, fx.Float32(value).ir_value()))


# ---- Volatile LDS reads and the lgkmcnt ledger ----


def _load_lds(storage, offset, immediate=0):
    """16 bytes of LDS at storage + offset + immediate (one ds_read_b128).

    Volatile keeps the hand-written order: LLVM neither merges nor reorders these
    loads, which the lgkmcnt ledger counts. The LDS variable's GEP gives it an alias scope.
    """
    ptr = fx.get_iter(storage) + (fx.Int32(offset) + immediate)
    ptr = fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 16), ptr
    )
    return fx.Vector(fx.generic_load(ptr, count=4, volatile=True))


LGKMCNT_MAX = 15  # gfx942/gfx950 s_waitcnt lgkmcnt field


class _LdsQueue:
    """lgkmcnt ledger of a wave's LDS reads and ds_bpermute, which retire in issue order.

    "The first n of them have landed" is lgkmcnt(issued - n). The union pipeline waits so
    before a barrier lets other waves overwrite LDS it still reads, which LLVM cannot see.
    The counts are derived: counted operations are emitted only through load() and
    bpermute(), one instruction each (volatile LDS loads are neither merged nor
    reordered). A ledger starts after every LDS op retired (lgkmcnt(0)), and no SMEM load
    (also counted, out of order) may be issued while it waits partially.

    capped: a wait past LGKMCNT_MAX younger ops waits for LGKMCNT_MAX, which retires
    more than needed (gfx950's 32 transposed V reads per half, waited 16 at a time);
    otherwise such a wait raises.
    """

    def __init__(self, capped=False):
        self.issued = 0
        self.capped = capped

    def load(self, storage, offset, immediate=0, transposed=False):
        """_load_lds (transposed: _load_lds_tr, gfx950), counted."""
        self.issued += 1
        if transposed:
            return _load_lds_tr(storage, offset, immediate)
        return _load_lds(storage, offset, immediate)

    def bpermute(self, address, value):
        self.issued += 1
        return bpermute(address, value)

    def ready(self, first, count):
        """ready(n) waits until the n first of the count ops issued from first landed."""

        def wait(done):
            if not 0 <= done <= count:
                raise ValueError(f"ready({done}) of {count} operations")
            self.wait(first + done)

        return wait

    def wait(self, upto=None, vmcnt=None):
        """Wait until the ops issued before index upto (default: all) have landed."""
        upto = self.issued if upto is None else upto
        if not 0 <= upto <= self.issued:
            raise ValueError(f"wait for {upto} of {self.issued} issued LDS operations")
        lgkmcnt = self.issued - upto
        if lgkmcnt > LGKMCNT_MAX:
            if not self.capped:
                raise ValueError(f"lgkmcnt({lgkmcnt}) exceeds the counter")
            lgkmcnt = LGKMCNT_MAX
        rocdl.s_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt)


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
# The part each gfx model's entries are tuned on.
TUNED_PARTS = {"gfx942": "MI308X", "gfx950": "MI355X"}
# FP8 K/V (sparse_prefill_*_fp8, gfx942 only, by K/V scale layout): route terms fitted
# on the FP8 kernels' phase timings, synthetic and SGLang-captured selections (median
# error 3%); the per-tile ratio is the sweep's best with the global route on. Past the
# packed-KV budget, FP8 direct packs up to 320 tokens per query row (its raw K/V reads
# cost more than BF16's). Union and minimum-row thresholds otherwise follow the BF16
# entry.
FP8_ROUTES = {
    ("gfx942", 80): {
        "tensor": {
            "union_fixed": 5.7,
            "union_step": 2.072,
            "row_step": 1.275,
            "direct_fixed": 11.9,
            "pack_block": 0.00184,
            "direct_cu_step": 0.24,
            "union_ratio": 2.0,
            "pack_tokens_per_row": 320,
        },
        "token": {
            "union_fixed": 5.14,
            "union_step": 2.483,
            "row_step": 1.4,
            "direct_fixed": 11.87,
            "pack_block": 0.00213,
            "direct_cu_step": 0.3,
            "union_ratio": 2.0,
            "pack_tokens_per_row": 320,
        },
    },
}
OVERRIDE = "AITER_QSA_PLATFORM"


@functools.cache
def _calibrated() -> tuple:
    """The PLATFORMS key of this GPU (nearest calibrated CU count of its gfx model); warns
    once per process for a CU count without its own entry."""
    gfx, cus = get_gfx_runtime(), get_cu_num()
    keys = [key for key in PLATFORMS if key[0] == gfx]
    if not keys:
        raise ValueError(f"QSA sparse attention has no calibration for {gfx}")
    key = min(keys, key=lambda key: abs(key[1] - cus))
    if key[1] != cus:
        part = TUNED_PARTS[gfx]
        logging.getLogger("aiter").warning(
            "QSA sparse attention is tuned only for %s (%s, %d CUs); this %s has %d CUs "
            "and reuses the %d-CU %s settings: plans and routes follow the CU count, "
            "performance may not be optimal",
            part,
            gfx,
            key[1],
            gfx,
            cus,
            key[1],
            part,
        )
    return key


@functools.cache
def platform(fp8=None) -> Platform:
    """The calibration of this GPU: BF16 K/V, or FP8 K/V with "tensor" / "token" scales."""
    key = _calibrated()
    entry = PLATFORMS[key]
    if fp8 is not None:
        entry = dataclasses.replace(entry, **FP8_ROUTES[key][fp8])
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


# ---- Raw buffers, pipeline barriers ----


def _min(a, b):
    return (a < b).select(a, b)


def _join(a, b):
    a, b = fx.Vector(a), fx.Vector(b)
    return fx.Vector.from_elements(
        [a[i] for i in range(a.numel)] + [b[i] for i in range(b.numel)], fx.Float32
    )


def _exp(value):
    """2**value by one bare v_exp_f32."""
    # afn lowers to llvm.amdgcn.exp2; plain math.exp2 adds denormal range reduction.
    return fx.math.exp2(fx.Float32(value), fastmath=fx.FastMathFlags.afn)


def _mfma(a, b, c, depth=16):
    """One 16x16xdepth BF16 MFMA on register vectors (depth 32: gfx950): a/b hold
    depth // 4 BF16 values, their bit patterns or their words, c 4 FP32."""
    frags = []
    for value in (a, b):
        value = fx.Vector(value)
        frag = fx.make_rmem_tensor(depth // 4, fx.BFloat16)
        frag.store(value if value.dtype is fx.BFloat16 else value.bitcast(fx.BFloat16))
        frags.append(frag)
    frag_c = fx.make_rmem_tensor(4, fx.Float32)
    frag_c.store(fx.Vector(c))
    mma = fx.make_mma_atom(rocdl.MFMA(16, 16, depth, fx.BFloat16))
    fx.gemm(mma, frag_c, frags[0], frags[1], frag_c)
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
        fx.Uint32(perm_b32(rounded[i + 1], rounded[i], 0x07060302))
        for i in range(0, values.numel, 2)
    ]
    return fx.Vector.from_elements(words, fx.Uint32)


def _stage_end():
    # Compiler fences and CTA rendezvous, not a memory-counter wait.
    sched_barrier(0)
    s_barrier()
    sched_barrier(0)


def _schedule(pairs, count, group, exp=False):
    for _ in range(pairs):
        sched_group_barrier(MFMA, 1, group)
        sched_group_barrier(TRANS if exp else VALU, count, group)


def _schedule_exps(count, group, mfmas=2):
    # Per value: subtraction, MFMA, exp, MFMA (the fastest order measured on gfx942); gfx950's
    # K32 QK has one MFMA per value.
    for _ in range(count):
        for mask in (VALU, MFMA, TRANS, MFMA)[: 2 + mfmas]:
            sched_group_barrier(mask, 1, group)


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


# K/V rows: hk * 256 values (KV heads of 256 dims), then gap elements of padding.
def _kv_extent(rows, hkv, hk, gap):
    """Bytes from KV head hkv of the first of `rows` K/V rows to the end of the last."""
    extent = (fx.Uint32(rows) * hk - hkv) * 512
    if gap:
        extent = extent + fx.Uint32(rows - 1) * (gap * 2)
    return extent


def _kv_offset(row, head, hk, gap, scale=1, dim=None):
    """Element offset of dim `dim` (None: 0) of KV head `head` of K/V row `row`, times
    scale (2: the byte offset), in row's integer type."""
    offset = (row * hk + head) * (256 * scale)
    if dim is not None:
        offset = offset + dim * scale
    if gap:
        offset = offset + row * (gap * scale)
    return offset


def _buffer_words(buffer, offset, words=4, cache=0):
    data = fx.make_rmem_tensor(4 * words, fx.Int8)
    fx.copy(
        fx.make_copy_atom(rocdl.BufferCopy(32 * words, cache), fx.Int8),
        fx.make_view(buffer + offset, fx.make_layout(4 * words, 1)),
        data,
    )
    return fx.Vector(data.load()).bitcast(fx.Int32)


def _buffer_store_words(words, buffer, offset, cache=0):
    words = fx.Vector(words)
    data = fx.make_rmem_tensor(4 * words.numel, fx.Int8)
    data.store(words.bitcast(fx.Int8))
    fx.copy(
        fx.make_copy_atom(rocdl.BufferCopy(32 * words.numel, cache), fx.Int8),
        data,
        fx.make_view(buffer + offset, fx.make_layout(4 * words.numel, 1)),
    )


# gfx942 buffer cache policy (aux bits: sc0 = 1, nt = 2, sc1 = 16): sc0 | sc1 is system scope,
# so stores write through and loads read past the CU's L1 and this XCD's L2 (coherent between
# CTAs of one launch on any XCD).
SC_SYSTEM = 17


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


# ---- QK, PV and the lazy rescale shared by the union and direct kernels ----


@flyc.jit
def _lazy(ballot, values, update):
    """values replaced by update(*values) when any lane's ballot bit is set: the rare branch
    of a lazy online-softmax rescale, so update is only evaluated there."""
    values = tuple(values)
    if ballot != fx.Int64(0):
        values = tuple(update(*values))
    return values


def _rescale(values, ballot, factor):
    """Lazy online-softmax rescale: multiply values (O accumulators, a row sum) by
    factor() when any lane's ballot bit is set; factor is only evaluated there."""

    def update(*values):
        correction = factor()
        return tuple(value * correction for value in values)

    return _lazy(ballot, values, update)


def _read_q_g2r(resource, offset, lane):
    """The lane's 32 Q words: the B operands of a 16x16x16 MFMA over D = 256, four BF16
    per 16-dim step, step s in words 2s and 2s + 1."""
    parts = [
        _buffer_words(resource, offset + (lane >> 4) * 16 + k * 64) for k in range(8)
    ]
    return fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)


def _qk(q, k0, k1):
    """Scores S = K Q^T of 32 keys over D = 256: one 16x16x16 BF16 MFMA per 16-key tile
    and 16-dim step, steps in order (the same dim order in every kernel).

    q holds the B operands (see _read_q_g2r); k0/k1 the A operands of keys 0-15 and
    16-31, four BF16 per step, step s in words 2s and 2s + 1. Returns the 8 FP32 scores:
    keys 0-15's accumulator, then keys 16-31's. The tiled MMA's eight-wave layout only
    shapes the per-lane fragments, which come from registers; it is the form the union
    pipeline's MFMA/VALU interleave was tuned with (per-MFMA calls schedule differently).
    """
    mma = fx.make_tiled_mma(
        fx.make_mma_atom(rocdl.MFMA(16, 16, 16, fx.BFloat16)),
        fx.make_layout((1, 8, 1), (1, 1, 0)),
    )
    b = mma.make_fragment_B(
        fx.make_rmem_tensor(fx.make_layout((128, 256), (1, 128)), fx.BFloat16)
    )
    b.store(fx.Vector(q).bitcast(fx.BFloat16))
    k0, k1 = fx.Vector(k0), fx.Vector(k1)
    a = fx.make_rmem_tensor(fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16)
    a.store(
        fx.Vector.from_elements(
            [k0[i] for i in range(32)] + [k1[i] for i in range(32)], fx.Int32
        ).bitcast(fx.BFloat16)
    )
    a = fx.make_view(fx.get_iter(a), fx.make_layout((4, 2, 16), (1, 64, 4)))
    acc = mma.make_fragment_C(
        fx.make_rmem_tensor(fx.make_layout((32, 128), (1, 32)), fx.Float32)
    )
    acc.fill(0.0)
    fx.gemm(mma, acc, a, b, acc, traversal_order="kmn")
    return acc.load()


def _pv_step(p, operand, acc):
    """O += P V over one 16-key step: one 16x16x16 BF16 MFMA per 16-dim tile of O.

    p holds the step's 4 BF16 probabilities (as Int16), operand(n) the A operand of
    tile n (4 BF16 of V as Int16), acc the tiles' 4 FP32 accumulators, updated in order.
    """
    return [_mfma(operand(n), p, acc[n]) for n in range(len(acc))]


def _v_pair(values, row, n):
    """The A operand words of O tile n: V dim n of rows row .. row + 3 as loaded (four
    words, 8 BF16 dims, per row and lane), two rows per word by v_perm_b32."""
    selector = 0x03020706 if n % 2 else 0x01000504
    return [
        fx.Int32(
            perm_b32(
                values[(row + r) * 4 + n // 2],
                values[(row + r + 1) * 4 + n // 2],
                selector,
            )
        )
        for r in (0, 2)
    ]


def _new_max(center):
    """m' (the row max + 1 once it exceeds m + 7) and its ballot, from center = (S, lane
    max, other maxima, scale, m)."""
    _, candidate, maxima, scale, maximum = center
    candidate = (
        _maximum(_maximum(candidate, maxima[0]), _maximum(maxima[1], maxima[2])) * scale
    )
    predicate = candidate > maximum + 7.0
    return predicate.select(candidate + 1.0, maximum), ballot(predicate)


def _sum_and_max(sum_max):
    """l + the lane row sums and the lane max of S, from sum_max = (S, lane sum, other
    sums, l)."""
    current, total, sums, row_sum = sum_max
    return row_sum + ((total + sums[0]) + (sums[1] + sums[2])), _max(current)


def _tiles(values):
    """FP32 values as MFMA accumulator tiles of 4."""
    values = fx.Vector(values)
    return [
        fx.Vector.from_elements([values[n * 4 + i] for i in range(4)], fx.Float32)
        for n in range(values.numel // 4)
    ]


def _untile(tiles):
    return fx.Vector.from_elements(
        [tile[i] for tile in tiles for i in range(4)], fx.Float32
    )


# ---- D256 BM128/BN64 eight-wave pipeline: MFMA fragments, softmax, LDS ----


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
        queue.bpermute(address, bits).bitcast(fx.Float32) for address in addresses
    )
    # LLVM would sink the ds_bpermutes below the LDS reads that the queue counts after them.
    sched_barrier(0)
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
    """LDS reads of one K or V operand from storage."""

    storage: object
    reads: list  # (slot, address, immediate) in issue order
    transposed: bool = False  # ds_read_b64_tr_b16 (gfx950 V), else ds_read_b128


def _read_s2r(plan, queue):
    """Issue a _ReadPlan; ready(n) waits until its first n reads have landed."""
    first = queue.issued
    parts = [None] * len(plan.reads)
    for slot, address, immediate in plan.reads:
        parts[slot] = queue.load(plan.storage, address, immediate, plan.transposed)
    return _lds_words(parts), queue.ready(first, queue.issued - first)


def _k_operands(words):
    """The _qk A operands of keys 0-15 and 16-31 from one K half's 64 LDS words."""
    # MFMA-only uses leave the loads AV-class; the scheduler then reverts the QK interleave.
    _keep_vgpr(words)
    sched_barrier(0)
    words = fx.Vector(words)
    return tuple(
        fx.Vector.from_elements([words[n * 32 + i] for i in range(32)], fx.Int32)
        for n in range(2)
    )


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
        sched_barrier(0)
    words = []
    for n in range(8):
        words.extend(_v_pair(raw, row, n))
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
    center = (S, lane max, other maxima, scale, m) returns S * scale, m' and the ballot.
    """
    p = fx.Vector(probabilities)
    acc = _tiles(output)
    prepared_steps = 0 if prepared is None else prepared.numel // 16
    operands = (
        _v_operands(values, 0, ready)
        if prepared is None
        else fx.Vector.from_elements([prepared[i] for i in range(16)], fx.Int32)
    )
    scaled = []
    sched_barrier(0)
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
        acc = _pv_step(
            b.bitcast(fx.Int16),
            lambda n, words=operands: fx.Vector.from_elements(
                [words[n * 2 + i] for i in range(2)], fx.Int32
            ).bitcast(fx.Int16),
            acc,
        )
        if center is not None:
            if step == 0:
                new_max, voted = _new_max(center)
            for i in range(step * 4, step * 4 + 4):
                scaled.append(fx.Float32(center[0][i]) * center[3])
        if sum_max is not None and step == 3:
            row_sum, candidate = _sum_and_max(sum_max)
        if step + 1 < steps and step + 1 >= prepared_steps:
            _schedule(8, 3 if center is not None else 2, 20 + step)
        elif sum_max is not None:
            _schedule(8, 3, 23)
        sched_barrier(0)
        operands = following
    result = _untile(acc)
    if center is not None:
        # Pinned: unpinned, LLVM sinks the scaling into the next memory stage.
        scaled = _pin(fx.Vector.from_elements(scaled, fx.Float32))
        return result, scaled, new_max, voted
    if sum_max is not None:
        return result, row_sum, candidate
    return result


# ---- gfx950 (CDNA4): K32 MFMAs, transposed V reads, native BF16, 64-bank swizzles ----
#
# The union pipeline keeps its stages and data flow (a capped lgkmcnt ledger, the
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


def _pack_bf16_950(values):
    """_pack_bf16 with v_cvt_pk_bf16_f32 (round to nearest even): element i in the low half."""
    values = fx.Vector(values)
    return fx.Vector.from_elements(
        [cvt_pk_bf16_f32(values[i], values[i + 1]) for i in range(0, values.numel, 2)],
        fx.Uint32,
    )


def _k_operands950(words):
    """The _qk950 A operands of one K half's 64 LDS words (as _k_operands): read
    m * 8 + s holds K step s of M tile m."""
    _keep_vgpr(words)
    sched_barrier(0)
    return fx.Vector(words)


def _qk950(q, k):
    """_qk with 16x16x32 BF16 MFMAs (q from _read_q_g2r, k from _k_operands950), issued
    a K step at a time (both M tiles of a step together)."""
    mma = fx.make_tiled_mma(
        fx.make_mma_atom(rocdl.MFMA(16, 16, 32, fx.BFloat16)),
        fx.make_layout((1, 8, 1), (1, 1, 0)),
    )
    b = mma.make_fragment_B(
        fx.make_rmem_tensor(fx.make_layout((128, 256), (1, 128)), fx.BFloat16)
    )
    b.store(fx.Vector(q).bitcast(fx.BFloat16))
    a = fx.make_rmem_tensor(fx.make_layout((8, 8, 2), (1, 8, 64)), fx.BFloat16)
    a.store(fx.Vector(k).bitcast(fx.BFloat16))
    a = fx.make_view(fx.get_iter(a), fx.make_layout((8, 2, 8), (1, 64, 8)))
    acc = mma.make_fragment_C(
        fx.make_rmem_tensor(fx.make_layout((32, 128), (1, 32)), fx.Float32)
    )
    acc.fill(0.0)
    fx.gemm(mma, acc, a, b, acc, traversal_order="mkn")
    return acc.load()


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
        sched_barrier(0)
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
    sched_barrier(0)
    for step in range(2):
        if ready is not None and (step or prepared is None):
            _v_operands950(raw, step, ready)
        b = fx.Vector.from_elements([p[step * 8 + i] for i in range(8)], fx.BFloat16)
        for n in range(8):
            a = fx.Vector.from_elements(
                [raw[i] for i in _pv950_words(step, n)], fx.Int32
            )
            acc[n] = _mfma(a, b, acc[n], 32)
        if center is not None:
            if step == 0:
                new_max, ballot = _new_max(center)
            for i in range(step * 8, step * 8 + 8):
                scaled.append(fx.Float32(center[0][i]) * center[3])
        if sum_max is not None and step == 1:
            row_sum, candidate = _sum_and_max(sum_max)
        if center is not None or sum_max is not None:
            _schedule(8, 2, 20 + step)
        sched_barrier(0)
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


# ---- FP8 (E4M3FNUZ, gfx942) K/V: MFMA, lazy max, output rounding, Q quantization ----

FP8_MAX = 240.0  # largest finite E4M3FNUZ
# Lazy max of the FP8 kernels: m moves only when some S'' exceeds LAZY (P' <= 2**7.5 fits
# E4M3FNUZ), and then puts the row max at S'' = P_BIAS: P' of a key down to 2**-(11 + P_BIAS)
# of the max stays nonzero (with P' <= 1, attention sinks over long flat tails lost the tail,
# a quarter of the output).
LAZY = 7.5
P_BIAS = 4.0
# A V scale enters as log2 and reciprocal; the clamp keeps both finite (NaN maps to the floor),
# and its 2**120 spread keeps a row's P' / v_scale, its row sum terms, within FP32's range.
SCALE_MIN, SCALE_MAX = 2.0**-60, 2.0**60
# Q rows of amax below Q_MIN quantize as if their amax were Q_MIN (240 / amax overflows FP32
# below about 2**-120; Q_MIN / 240 is a normal FP32 scale): fewer FP8 bits, but nonzero.
Q_MIN = 2.0**-118
# Q quantization: two rows per 16-lane group (16 dims per lane and row, every row load in
# flight at once), so QROWS rows per 256-thread CTA.
QR = 2
QROWS = 16 * QR


def _as_f32(bits):
    return fx.Int32(bits).bitcast(fx.Float32)


def _as_i32(value):
    return fx.Float32(value).bitcast(fx.Int32)


def _fma(a, b, c):
    return fx.Float32(fx.fma(fx.Float32(a), fx.Float32(b), fx.Float32(c)))


def _mfma_fp8(a, b, acc):
    """One E4M3FNUZ MFMA on register words: a and b hold 8 FP8 values (two words) per lane,
    acc 16 FP32 (32x32x16) or 4 (16x16x32)."""
    acc = fx.Vector(acc)
    m = 32 if acc.numel == 16 else 16
    frag_a = fx.make_rmem_tensor(8, fx.Float8E4M3FNUZ)
    frag_b = fx.make_rmem_tensor(8, fx.Float8E4M3FNUZ)
    frag_c = fx.make_rmem_tensor(acc.numel, fx.Float32)
    frag_a.store(fx.Vector.from_elements(list(a), fx.Int32).bitcast(fx.Float8E4M3FNUZ))
    frag_b.store(fx.Vector.from_elements(list(b), fx.Int32).bitcast(fx.Float8E4M3FNUZ))
    frag_c.store(acc)
    mma = fx.make_mma_atom(rocdl.MFMA(m, m, 512 // m, fx.Float8E4M3FNUZ))
    fx.gemm(mma, frag_c, frag_a, frag_b, frag_c)
    return fx.Vector(frag_c.load())


def _fp8_word(values):
    """Four FP32 -> one word of E4M3FNUZ (byte k = value k), round to nearest even."""
    return cvt_pk_fp8(values[2], values[3], cvt_pk_fp8(values[0], values[1]), True)


def _v_scale_terms(sv):
    """(log2(v_scale), 1 / v_scale) of a clamped V scale."""
    sv = fx.minnumf(
        fx.maxnumf(fx.Float32(sv), fx.Float32(SCALE_MIN)), fx.Float32(SCALE_MAX)
    )
    return log2(sv), rcp(sv)


def _lazy_max(o, row_sum, scores, m_run, ballot, tile_max):
    """The FP8 kernels' lazy max (S'' = S * c - m in log2 units): when any lane's ballot bit
    is set, m moves by max(tile_max() - P_BIAS, 0), and a fresh row (m = FLOOR, centered at
    0) takes tile_max() - P_BIAS (FLOOR while it has no key: its S'' stay -inf); O (a list
    of accumulator vectors) and the row sum l scale by 2**-delta and S'' shifts by -delta.
    Returns (o, row_sum, scores, m_run)."""

    def update(*values):
        *o, row_sum, scores, m_run = values
        tile = tile_max()
        fresh = m_run == fx.Float32(_MAX_FLOOR)
        delta = _maximum(
            fx.Float32(tile) - P_BIAS,
            fresh.select(fx.Float32(_MAX_FLOOR), fx.Float32(0.0)),
        )
        corr = fresh.select(fx.Float32(0.0), _exp(fx.Float32(0.0) - delta))
        return (
            *(fx.Vector(x) * corr for x in o),
            fx.Float32(row_sum) * corr,
            fx.Vector(scores) - delta,
            fresh.select(fx.Float32(0.0), m_run) + delta,
        )

    *o, row_sum, scores, m_run = _lazy(ballot, (*o, row_sum, scores, m_run), update)
    return o, row_sum, scores, m_run


def _scaled_bf16(values, inv):
    """Two BF16 words of four FP32 values times inv: FP32 multiplies, then round to nearest
    even like _pack_bf16 and the high halves joined by v_perm."""

    def rne(value):
        bits = _as_i32(fx.Float32(value) * inv)
        return bits + fx.Int32(0x7FFF) + ((bits >> 16) & fx.Int32(1))

    words = []
    for i in (0, 2):
        lo, hi = rne(values[i]), rne(values[i + 1])
        words.append(fx.Int32(perm_b32(hi, lo, 0x07060302)))
    return words


def _quantize_part(raw):
    """16 BF16 dims of a row (16 lanes per row): FP8 words and the row scale
    max(amax, Q_MIN) / 240 (0 for a zero row)."""
    values = []
    for vec in raw:
        for j in range(4):
            word = fx.Int32(vec[j])
            values += [_as_f32(word << 16), _as_f32(word & fx.Int32(-65536))]
    level = [fx.absf(v) for v in values]
    while len(level) > 1:
        level = [_maximum(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    amax = level[0]
    for shift in (1, 2, 4, 8):
        amax = _maximum(amax, fx.gpu.shuffle_xor(amax, shift, 64))
    floor = _maximum(amax, fx.Float32(Q_MIN))
    inverse = rcp(floor) * FP8_MAX
    words = [
        _fp8_word([values[4 * k + r] * inverse for r in range(4)]) for k in range(4)
    ]
    scale = (amax > fx.Float32(0.0)).select(floor * (1.0 / FP8_MAX), fx.Float32(0.0))
    return words, scale


@flyc.jit
def quantize_rows(Q, Q8, QS, ROWS, cta):
    """Rows cta * QROWS .. + QROWS - 1 of Q (rows of 256 BF16) for a 256-thread CTA: row
    r * 16 + (thread >> 4) of the CTA's rows per step r; FP8 rows (E4M3FNUZ, RNE) and
    their scales amax / 240 (Q_MIN / 240 for rows of amax below Q_MIN, 0 for a zero row).
    Run by the FP8 pack kernel's CTAs past its blocks.
    """
    tid = fx.Int32(fx.thread_idx.x)
    group, part = tid >> 4, tid & 15
    extent = fx.Uint32(ROWS) * 512
    gq = _buffer(Q, extent)
    rows = [cta * QROWS + r * 16 + group for r in range(QR)]
    raws = []
    for row in rows:
        offset = (row < ROWS).select(row * 512 + part * 32, fx.Int32(extent))
        raws.append([_buffer_words(gq, offset + 16 * i, 4) for i in range(2)])
    g8 = _buffer(Q8, fx.Uint32(ROWS) * 256)
    scales = []
    for row, raw in zip(rows, raws):
        words, scale = _quantize_part(raw)
        scales.append(scale)
        _buffer_store_words(
            fx.Vector.from_elements(words, fx.Int32),
            g8,
            (row < ROWS).select(row * 256 + part * 16, fx.Int32(ROWS * 256)),
        )
    # Lane `part` of a group stores the scale of its row step `part`.
    scale = scales[QR - 1]
    for r in fx.range_constexpr(QR - 1):
        scale = (part == r).select(scales[r], scale)
    row = cta * QROWS + part * 16 + group
    gs = _buffer(QS, fx.Uint32(ROWS) * 4)
    _buffer_store_words(
        fx.Vector.from_elements([_as_i32(scale)], fx.Int32),
        gs,
        ((part < QR) & (row < ROWS)).select(row * 4, fx.Int32(ROWS * 4)),
    )

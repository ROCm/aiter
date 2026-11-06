# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QSA index scorer (paged MQA logits), compressed page = 4, gfx1250 Ã¢â‚¬â€ **v0 baseline**.

Contract (frozen; see ``analysis/QSA_index_scorer_dtype_contract.md`` and
``golden_baseline/qsa_golden.py``)::

    visible[t]   = clamp(min((pos[t] + 1) // 4, ctx[req(t)] // 4), min=0)
    logits[t][c] = -inf                                              if c >= visible[t]
                 = sum_{h=0..3} relu(q[t,h,:] . k[c]) / sqrt(128)    otherwise
    k[c]         = dequant(compressed_k_cache[page_table[req, c // 4], c % 4, 0, :])

with ``-inf`` also for columns whose page-table entry is invalid (-1 / out of
range).  Comparisons are restricted to ``valid_page & c < visible``.

Q is always bf16 ``[rows, 4, 128]``; the index cache is either bf16
``[pages, 4, 1, 128]`` or fp8 E4M3 ``[pages, 4, 1, 128]`` with one **fp32
per-compressed-token scale** ``[pages, 4, 1]`` (ATOM's ``qsa_cache_row_layout
("fp8")``: ``k = E4M3 * scale[page, row]``, *not* an MX block scale).

This file is the **v0 correctness baseline**: one thread per output element,
plain vector loads and FMAs, no WMMA/MFMA, no LDS staging, no split-K.  One
workgroup is one query row x ``THREADS`` consecutive columns; a page (4
consecutive compressed columns, 1 KiB bf16 / 512 B fp8 + 16 B scale) is
contiguous, so a thread's 4 page-mates are four independent 256 B / 128 B
reads inside one contiguous 1 KiB / 512 B block.  Every thread re-reads its Q
row and its key row from L1 Ã¢â‚¬â€ deliberately untuned.  See
``records/g2/v0-manifest.json`` for the first-look latency (exploratory, not
evidence).

Layout facts this kernel relies on (all contiguous, checked by the wrapper):

* ``q``: ``[rows, 4, 128]`` bf16, 256 B per head row;
* bf16 cache: one page = 4 rows * 128 * 2 B = 1024 B, row ``j`` at ``j * 256``;
* fp8 cache: one page = 4 rows * 128 B = 512 B, row ``j`` at ``j * 128``;
* fp8 scale: one page = 4 fp32 = 16 B, row ``j`` at ``j * 4``.

The module imports only ``flydsl`` and ``torch``: the few buffer primitives it
needs (byte-addressed ``buffer_load``/``buffer_store`` and a cached
``flyc.compile`` dispatch) are implemented here on published FlyDSL APIs instead
of importing the ``aiter`` package.

GPU-safety revision (G2 builder #2, 2026-10-31).  v0 was never dispatched
successfully: its first dispatch attempt on d07-2 wedged the device and left
``amdgpu 0001:04:00.0: [0] id=4 (VM_PAGE_FAULT) size=0x1000 start=0x16bf5fff000``
in dmesg (see ``records/g2/builder2-log.md``).  That address is *exactly* what a
non-dropped ``base + 0x7FFFFFFF`` looks like: the page start is
``0x1ff000`` mod 2 MiB and the implied base ``0x16b76000000`` is 2 MiB aligned,
i.e. a device allocation Ã¢â‚¬â€ so v0's masked stores were **not** being dropped by
the V# bounds check on this toolchain.  Three changes remove that dependency:

1. every ``logits``/``visible_groups``/``row_starts`` store is now *predicated*
   (``if ok: store``) instead of being steered to a "dropped" out-of-bounds
   byte offset, so every store address is inside its buffer by construction
   (the offset is additionally clamped, so even a mis-lowered predicate cannot
   leave the buffer);
2. descriptor ``num_records`` extents are computed in **i64** and clamped to at
   least 4 bytes, because ``num_records == 0`` means *unbounded* on this ISA
   (a zero-size plane would turn masked accesses into real ones);
3. the op wrapper rejects anything that could break the kernel's i32 byte
   addressing (2 GiB cap), zero-size planes, misalignment, or out-of-int32
   metadata.

The V# extents stay: they are the second line of defence (loads zero-fill,
stores drop) for anything that still slips through Ã¢â‚¬â€ but see the OOB_SELECT note
in the access-layer block: on RDNA (gfx1250) the V# check is *not* the
drop-everything net it is on CDNA, so it must never be the mechanism that keeps
an address in bounds.

GPU-hang revision (G2 builder #3, 2026-10-31).  The ``6c272c3c`` build above was
dispatched once, on d01-3 (single gfx1250, 256 CU): the launch itself returned
(0.8 s), then ``torch.cuda.synchronize()`` never returned, the device stayed
100 % busy, and ``dmesg`` showed ``MES(0, 0) failed to respond to
msg=REMOVE_QUEUE`` / ``Failed to evict queue 1`` / ``Failed to quiesce KFD`` Ã¢â‚¬â€ a
firmware-level queue hang, no ``VM_PAGE_FAULT`` (so no address left its buffer:
the address arithmetic is not the suspect).  The one layer of this file with no
gfx1250 precedent in the tree is the global access layer (see that block), so
the default is now the in-tree ``make_buffer_ptr`` -> ``get_buffer_rsrc`` ->
``raw_ptr_buffer_load/store`` pattern with **one pinned descriptor per plane and
byte offsets in the instruction**, and the previous copy-atom layer is kept
behind ``QSA_G2_ACCESS=cop`` for the bisect ladder
(``_g2/bisect.py`` / ``_g2/BISECT.md``) and as a one-variable revert.  Nothing
observable at the op boundary changes: same planes, same extents, same byte
addresses, same predicates, same values.

Round C (opt-c, 2026-10-31): the **bf16** cache now runs on the matrix core Ã¢â‚¬â€
``_build_wmma_kernel``, gfx1250 WMMA 16x16x32 bf16 -> f32, one wave = 4 query
rows x 4 heads (M = 16) x 64 columns, Q fragments in VGPRs, K fragments loaded
straight from the paged cache, request-dedup passes and an early -inf exit for
invisible column blocks (design ported from the task package's FlyDSL MFMA
scorer; see the block above ``_build_wmma_kernel`` for why it is WMMA and not
MFMA on gfx1250).  The **fp8** cache keeps the v0 scalar kernel unchanged, and
``QSA_BF16_PATH=scalar`` routes bf16 back to it.  Same public API, same
outputs, same safety contract (predicated + clamped stores, i64 extents).

Round G (opt-g, 2026-10-31): the **fp8** cache runs on the matrix core too Ã¢â‚¬â€
``_build_wmma_fp8_kernel``: the bf16 WMMA kernel with B fragments built from
E4M3 bytes converted *exactly* to bf16 in registers and the per-token fp32 scale
applied after the GEMM (in the scalar fp8 kernel's order).  ``_build_wmma_kernel``
(bf16) is byte-for-byte unchanged; ``QSA_FP8_PATH=scalar`` routes fp8 back to
the round-F scalar kernel.  Same public API, outputs and safety contract.

Round J (opt-j, 2026-10-31): bf16 prefill shapes (>=32 rows/request on
average) select ``_r32_c64``: four waves each process eight query rows as two
M16 WMMA tiles (four rows x four heads per tile), sharing each B fragment.
A complete 32-query-row block must have one valid request ID in *every* row;
otherwise each M16 tile uses the unchanged r16 request-dedup path. Decode,
draft, fp8, descriptor construction, host guards and the public API are unchanged.

Round K (opt-k, 2026-10-31): fp8 uses the same shape-selected ``_r32_c64``
geometry and full-block request guard as bf16. Each WMMA step's two 8-byte
E4M3 loads and exact bf16 conversion feed both M16 tiles; each accumulator
retains the r16 post-GEMM fp32 scale, ReLU, left-to-right head sum and divisor
order. Mixed requests and partial blocks use the exact fp8 r16 helpers.
The bf16 paths, scalar fp8 fallback, descriptors and public API are unchanged.

Round TDM (2026-10-31): ``QSA_BF16_K_LOAD=tdm`` opts full, single-request
bf16 r32 blocks into a shared 64x128 bf16 K tile. Round TDM2 pads each
256-byte source row to a 272-byte LDS pitch (17,408 B total), keeping 16-byte
fragment loads aligned while spreading row starts over the LDS bank groups. Each wave
issues two 8-row descriptor gathers using scalar page metadata, waits for
completion, and joins a workgroup barrier before ordinary LDS fragment reads.
Round TDM2 also stages full bf16 r16 blocks in four wave-private padded tiles
(69,632 B LDS): eight 8-row gathers per active request pass, scalar request
and visibility guards, completion wait plus workgroup barrier before reading,
and a second barrier before tile reuse. Partial blocks, mixed/partial r32,
unaligned cache bases, the copy-atom fallback and fp8 retain raw K loads.
Round TDMAUTO defaults to ``auto``: r32 uses padded TDM and r16 stays raw.
``raw`` forces per-lane loads; ``tdm`` keeps both TDM specializations available.
Round TDM4 stages full single-request fp8 r32 blocks in one shared 64x128-byte
K tile with 144-byte padded LDS rows (9,216 B); QSA_FP8_K_LOAD=raw restores
raw r32 and r16/decode always stays raw. Two aligned b128 LDS reads per step
select each lane's 8-byte halves before the unchanged exact E4M3 conversion.
Per-token scale order, mixed/tail fallback and safety stay intact.
Round TDM4 BM64 provides bf16 r64 for >=64 rows/request on average under TDM:
four waves each consume four M16 tiles (=16 query rows x four heads), in two
sequential pairs. One 17,408-byte shared K tile is staged once per 64 query
rows. Full64 valid-request uniformity is checked; mixtures and tails use the
raw per-M-tile path. Raw/cop preserve champion r16/r32 shape routing.
Round R16: the TDM r16 kernel uses 32-column workgroups by default
(``QSA_BF16_R16_TDM_COLS=32``): four wave-private 32x272 B tiles = 34,816 B LDS
(half of TDM2), four 8-row gathers per active pass, grid.x doubled, per-column K
traffic unchanged. ``QSA_BF16_R16_TDM_COLS=64`` restores the 69,632 B kernel.
Round MERGE defaults bf16 r16 decode/draft to TDM c32 and retains bf16/fp8
r32 prefill TDM. BM64 is disabled unless explicitly QSA_BF16_R64=1: its
235 VGPR pressure regressed prefill. Raw forces all-raw and disables BM64;
unaligned bases and mixed/tail blocks retain their exact raw fallbacks.
ROUND-NATIVE64 reads int64 query_positions in-kernel (``_pos_loaders``, ``_qp64``
kernels; no clamp launch). ROUND-MERGE1 makes the GPU-verified Round DRAFT
kernels the default: ``QSA_DRAFT_PATH=1`` (rows <= requests -> one-row-per-wave
draft kernel, TDM K, 32-column wave slabs; ``0`` restores champion routing) and
``QSA_R16_NOBAR=1`` (barrier-free wave-private bf16 r16 TDM; ``0`` restores the
barriered kernel). Both also have pos64 variants.
ROUND-PAGE16: the index page P is a compile-time builder parameter
(``page_size``, P in {4, 16}; KV block 16 -> 4, KV block 64 -> 16). The wrapper
reads P from the cache shape ``[pages, P, 1, 128]`` (other P raise), column c
reads ``page_table[req, c // P]`` row ``c % P`` and visibility still divides by
the compress ratio 4. P = 4 builds pass no new keyword and keep every TDM site's
code verbatim (``if P == 4`` trace-time branches): byte-identical ISA to bf492262.
P = 16: every 8-row TDM gather group (32-bit indices: 8 rows per instruction,
kernel-wiki mi400-tdm-programming) lies inside one 16-row page, so a page is two
gathers sharing one SGPR page-table lookup each (P = 4: two lookups per gather);
the padded LDS tiles (272 B / 144 B pitch, column c -> tile row c % tile) and
hence the conflict-free WMMA B reads (mi400-pattern-lds-conflicts) are
unchanged, as are TENSORcnt / barrier publication (mi400-tdm-sync). Raw and
scalar paths use ``phys * P * row_bytes + (c % P) * row_bytes``.
ROUND-R2: P = 8 (KV block 32): a page is 8 contiguous rows = exactly ONE 8-row
TDM gather (32-bit indices, mi400-tdm-programming) fed by ONE page-table
s_buffer_load (P = 4: two per gather, P = 16: one per half page); new ``else``
(P = 8) branches only -- the P = 4 / P = 16 branches are verbatim (ISA identity
vs 536667d5). Decode-group knobs ``group_rows`` (3 = MTP verify, 3 rows per
request), ``row_mode`` (dedup / pad: padded rows get no stores) and
``wg_waves`` (draft + decode group, short contexts); defaults unchanged.
"""

from __future__ import annotations

import math
import os
import re
from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr

Vec = fx.Vector

PAGE_SIZE = 4  # compressed rows per index-cache page (= KV block 16 / ratio 4)
# ROUND-PAGE16: the index page P is a compile-time builder parameter. P = 4
# (KV block 16) is the default and every P = 4 specialization is byte-identical
# ISA to bf492262; P = 16 (KV block 64) is the second supported geometry.
# The visibility divisor is the *compress ratio* (always 4), never P.
COMPRESS_RATIO = 4
SUPPORTED_PAGE_SIZES = (4, 8, 16)  # ROUND-R2: + P = 8 (KV block 32)
NUM_HEADS = 4  # index heads
HEAD_DIM = 128  # index head dim
DIM_PER_STEP = 8  # head dims consumed per unrolled step (16 B of Q)
THREADS = 256  # one thread per output column
MAX_GRID_Y = 65535
_NEG_INF = float("-inf")
# A V# with num_records == 0 is *unbounded* (no bounds check at all), so every
# descriptor gets at least one dword: a zero-size plane then drops every access
# instead of allowing it.
_MIN_DESC_BYTES = 4
_ELEM_BYTES = {fx.Int8: 1, fx.Int32: 4, fx.Float32: 4, fx.BFloat16: 2}


def _check_page_size(page_size) -> int:
    """Trace-time page geometry guard shared by every builder (ROUND-PAGE16)."""
    p = int(page_size)
    if p not in SUPPORTED_PAGE_SIZES:
        raise ValueError(
            f"index page size must be one of {SUPPORTED_PAGE_SIZES}, got {page_size}"
        )
    return p


def _imin(a, b):
    return (a <= b).select(a, b)


def _imax(a, b):
    return (a >= b).select(a, b)


# ---------------------------------------------------------------------------
# Global access layer: one pinned descriptor (V#) per plane + byte offsets
# ---------------------------------------------------------------------------
# Every plane is addressed through **one** AMD buffer resource descriptor whose
# ``NUM_RECORDS`` is the plane's true extent in bytes, and an explicit **byte**
# offset per access.  Two implementations of that seam exist:
#
#   raw (default)  ``make_buffer_ptr`` -> ``get_buffer_rsrc`` ->
#                  ``rocdl.raw_ptr_buffer_load/store``.  This is the in-tree
#                  pattern of the kernels that are exercised on gfx1250
#                  hardware: ``aiter/ops/flydsl/kernels/buffer_ops.py:389-568``
#                  (``buffer_load``/``buffer_store``, "Buffer load offset is in
#                  BYTES, not elements") and the pinned-descriptor form of
#                  ``.../moe_gemm_2stage/gfx9_fp8/gemm2_1x4.py:353-383``.
#   cop            ``make_copy_atom(BufferCopy(nbytes*8), Int8)`` +
#                  ``copy_atom_call`` through a per-access ``make_view`` Ã¢â‚¬â€ what
#                  commit ``6c272c3c`` shipped, and what the first real dispatch
#                  of this kernel hung the GPU with (``records/g2/
#                  builder3-log.md``).  It is the only construct in this file
#                  with **no gfx1250 precedent in the tree**: its single in-tree
#                  user, ``mqa_logits/qsa_flydsl_layout.py:70-81``, is a
#                  candidate implementation that has never been dispatched on
#                  gfx1250 either, and on the *store* path it additionally
#                  allocates a register array inside a lane-varying ``if``::
#
#                      if col_ok:                       # only 4 of 256 lanes
#                          reg = make_rmem_tensor(...)  # <- allocation in a branch
#                          memref_store_vec(...); copy_atom_call(...)
#
#                  which nothing else in the tree does around a global store.
#                  It stays selectable (``QSA_G2_ACCESS=cop``) so the bisect
#                  ladder can compare the two layers on hardware and so a
#                  compile problem in the raw layer is one environment variable
#                  away from a revert rather than an edit on a wedged node.
#
# Both layers keep the ``6c272c3c`` address guarantees: the descriptor extent is
# the true plane size (i64, at least one dword), every offset is clamped in
# bounds, and stores stay **predicated** Ã¢â‚¬â€ never steered at a sentinel byte
# offset.  That last point is not stylistic: on RDNA the V# OOB check is not the
# drop-everything net it is on CDNA (``buffer_ops._get_buffer_flags`` sets
# OOB_SELECT=2, "no bounds checking", for ``is_rdna_arch``, and gfx1250 is RDNA),
# which is exactly what the d07-2 ``VM_PAGE_FAULT`` at ``base + 0x7FFFFFFF``
# says.  See ``records/g2/builder2-log.md`` Ã‚Â§2.
_ACCESS = os.environ.get("QSA_G2_ACCESS", "raw").strip().lower()
if _ACCESS not in ("raw", "cop"):
    raise ValueError(f"QSA_G2_ACCESS must be 'raw' or 'cop', got {_ACCESS!r}")


def _plane_bytes(count, item_bytes, count2=1):
    """True i64 extent for positive i32 dimensions, at least one dword.

    ``item_bytes`` is a constexpr stride >= 4 for every plane in this kernel.
    Clamp dimensions *before* widening: an i64 signed max has no SALU opcode
    on gfx1250, so it makes the V# divergent and wraps every MUBUF access in a
    serialized waterfall.  Products still use i64 (never multiply in i32).
    The wrapper rejects empty/negative dimensions; their descriptors remain
    bounded here even for a direct compile-only invocation.
    """
    n = fx.Int64(_imax(fx.Int32(count), fx.Int32(1)))
    n2 = fx.Int64(_imax(fx.Int32(count2), fx.Int32(1)))
    return n * n2 * fx.Int64(item_bytes)


def _plane(tensor, num_bytes):
    """Descriptor for a kernel tensor argument, extent ``num_bytes`` (i64).

    ``num_bytes`` is the descriptor's ``num_records``: loads past it are
    zero-filled and stores past it dropped by hardware that implements the
    check (see the OOB_SELECT note above Ã¢â‚¬â€ treat it as a safety net, not as the
    mechanism that keeps stores in bounds in the first place).

    ``num_bytes`` must therefore be the *true* extent of the plane in bytes and
    is computed in i64 (an i32 product can wrap and then either truncate the
    descriptor or, worse, look large enough to admit a bad offset).  Callers
    supply ``_plane_bytes`` or the 4-byte bf16 dummy, both >= _MIN_DESC_BYTES;
    do not re-clamp the i64 product here (that forces a VGPR descriptor).
    """
    ptr = fx.rocdl.make_buffer_ptr(
        fx.recast_iter(fx.Int8, fx.get_iter(tensor)),
        num_records_bytes=fx.Int64(num_bytes),
    )
    return ptr if _ACCESS == "cop" else fx.rocdl.get_buffer_rsrc(ptr)


_RAW: dict = {}


def _raw():
    """``(ir, aux)`` for the raw ROCDL intrinsics, resolved once per process.

    Kept lazy on purpose: this module must still import on a host with no
    ``flydsl`` (``_g2/_load.py`` installs a stub so the op wrapper stays
    testable), and ``flydsl._mlir`` is not part of that stub.
    """
    if not _RAW:
        import inspect

        from flydsl._mlir import ir

        # flydsl moved the buffer cache policy from an i32 operand to an enum
        # attribute; ``buffer_ops.py:44-48`` detects the same way.
        is_attr = (
            inspect.signature(fx.rocdl.RawPtrBufferLoadOp).parameters["aux"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )
        aux = (
            ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)
            if is_attr
            else fx.Int32(0).ir_value()
        )
        _RAW.update(ir=ir, aux=aux)
    return _RAW["ir"], _RAW["aux"]


def _ld_vec(ptr, byte_off, n, elem):
    """``n`` elements of ``elem`` at ``byte_off`` (one buffer_load_dwordx n*4)."""
    if _ACCESS == "cop":
        nbytes = n * _ELEM_BYTES[elem]
        src = fx.make_view(ptr + byte_off, fx.make_layout(nbytes, 1))
        reg = fx.make_rmem_tensor(fx.make_layout(n, 1), elem)
        fx.copy_atom_call(
            fx.make_copy_atom(fx.rocdl.BufferCopy(nbytes * 8), fx.Int8), src, reg
        )
        return Vec(fx.memref_load_vec(reg))
    ir, aux = _raw()
    return Vec(
        fx.rocdl.raw_ptr_buffer_load(
            ir.VectorType.get([n], elem.ir_type),
            ptr,
            fx.Int32(byte_off).ir_value(),
            fx.Int32(0).ir_value(),
            aux=aux,
        )
    )


def _ld_scalar(ptr, byte_off, elem):
    """One element of ``elem`` at ``byte_off`` (buffer_load_dword)."""
    if _ACCESS == "cop":
        nbytes = _ELEM_BYTES[elem]
        src = fx.make_view(ptr + byte_off, fx.make_layout(nbytes, 1))
        reg = fx.make_rmem_tensor(fx.make_layout(1, 1), elem)
        fx.copy_atom_call(
            fx.make_copy_atom(fx.rocdl.BufferCopy(nbytes * 8), fx.Int8), src, reg
        )
        return elem(Vec(fx.memref_load_vec(reg))[0])
    _, aux = _raw()
    return elem(
        fx.rocdl.raw_ptr_buffer_load(
            elem.ir_type,
            ptr,
            fx.Int32(byte_off).ir_value(),
            fx.Int32(0).ir_value(),
            aux=aux,
        )
    )


def _st_scalar(ptr, byte_off, value, elem):
    """One buffer_store_dword of ``value`` at ``byte_off``."""
    if _ACCESS == "cop":
        nbytes = _ELEM_BYTES[elem]
        dst = fx.make_view(ptr + byte_off, fx.make_layout(nbytes, 1))
        reg = fx.make_rmem_tensor(fx.make_layout(1, 1), elem)
        fx.memref_store_vec(Vec.from_elements([value], dtype=elem), reg)
        fx.copy_atom_call(
            fx.make_copy_atom(fx.rocdl.BufferCopy(nbytes * 8), fx.Int8), reg, dst
        )
        return
    _, aux = _raw()
    fx.rocdl.raw_ptr_buffer_store(
        elem(value).ir_value(),
        ptr,
        fx.Int32(byte_off).ir_value(),
        fx.Int32(0).ir_value(),
        aux=aux,
    )


def _ld_i32(ptr, byte_off):
    return _ld_scalar(ptr, byte_off, fx.Int32)


def _ld_f32(ptr, byte_off):
    return _ld_scalar(ptr, byte_off, fx.Float32)


def _st_i32(ptr, byte_off, value):
    _st_scalar(ptr, byte_off, value, fx.Int32)


def _st_f32(ptr, byte_off, value):
    _st_scalar(ptr, byte_off, value, fx.Float32)


# ---------------------------------------------------------------------------
# ROUND-R2D: opt-in in-kernel phase timestamps (builder keyword ``trace=1|2``; the
# default ``trace=0`` emits nothing, so every existing build is byte-identical).
# ---------------------------------------------------------------------------
# kernel-wiki mi400-scalar-pipeline §5: TIME = S_GET_SHADER_CYCLES_U64 (shader clock, ~8
# cycle SALU read, deltas valid only WITHIN one wave); REALTIME = S_SENDMSG_RTN_B64
# GET_REALTIME + S_WAIT_KMcnt (fixed-frequency, valid ACROSS waves/SIMDs; frequency not
# adopted by the wiki -> the analyser derives the TIME/REALTIME ratio from the data).
# HW_ID1/HW_ID2 via S_GETREG_B32 (mi400-occupancy-resources §16: WAVE_ID 4:0, SIMD_ID 9:8,
# WGP_ID 13:10, SA_ID 16; S_GETREG waits XCNT==0, mi400-instruction-latency).
# The record of one wave (TRACE_SLOTS x int64) is written to the bytes right AFTER the
# logits plane: the caller passes a logits view of a larger zeroed allocation
# (rows*columns fp32 + num_waves * TRACE_WAVE_BYTES). Every lane stores the same 8 bytes
# (unpredicated, in-plane by construction). trace=2 additionally waits loadcnt/kmcnt 0
# before the marks flagged `wait` (serialises the phases -> per-phase latency).
TRACE_SLOTS = 16
TRACE_WAVE_BYTES = TRACE_SLOTS * 8
TRACE_MAGIC = (0x51534154, 0x52324431)  # 'QSAT', 'R2D1' (lo, hi dwords)
TRACE_SLOT_NAMES = (
    "rt_entry",
    "t_entry",
    "hw_id",
    "ident",
    "t_prologue",
    "t_guard",
    "t_tdm_issue",
    "t_q_issue",
    "t_tdm_wait",
    "t_barrier",
    "t_compute_store",
    "t_store_drain",
    "rt_exit",
    "t_exit",
    "rt_barrier",
    "magic",
)


def _tr_asm(ty_bits, asm):
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm

    return llvm.inline_asm(
        ir.IntegerType.get_signless(ty_bits), [], asm, "=s", has_side_effects=True
    )


def _tr_void(asm):
    from flydsl._mlir.dialects import llvm

    llvm.inline_asm(None, [], asm, "", has_side_effects=True)


def _tr_time():
    return _tr_asm(64, "s_get_shader_cycles_u64 $0")


def _tr_realtime():
    return _tr_asm(
        64, "s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\n\ts_wait_kmcnt 0x0"
    )


def _tr_store_vec2(rsrc, byte_off, v2):
    _ir, aux = _raw()
    fx.rocdl.raw_ptr_buffer_store(
        v2, rsrc, fx.Int32(byte_off).ir_value(), fx.Int32(0).ir_value(), aux=aux
    )


def _tr_store64(rsrc, byte_off, v64):
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm

    _tr_store_vec2(
        rsrc, byte_off, llvm.bitcast(ir.VectorType.get([2], fx.Int32.ir_type), v64)
    )


def _tr_store_pair(rsrc, byte_off, lo, hi):
    _tr_store_vec2(
        rsrc,
        byte_off,
        Vec.from_elements([fx.Int32(lo), fx.Int32(hi)], dtype=fx.Int32).ir_value(),
    )


def _tr_begin(trs, level, LOGITS, rows, columns, grid_x, grid_y, waves, bx, by):
    """Entry stamps (slots 0-3). ``trs`` is the builder's trace holder dict."""
    if not level:
        return
    rt = _tr_realtime()
    t = _tr_time()
    hw1 = fx.Int32(_tr_asm(32, "s_getreg_b32 $0, hwreg(HW_REG_WAVE_HW_ID1)"))
    hw2 = fx.Int32(_tr_asm(32, "s_getreg_b32 $0, hwreg(HW_REG_WAVE_HW_ID2)"))
    wave_id = fx.Int32(fx.rocdl.wave_id())
    lbytes = rows * columns * 4
    nwaves = grid_x * grid_y * waves
    rsrc = _plane(
        LOGITS,
        _plane_bytes(rows, 4, columns) + fx.Int64(nwaves) * fx.Int64(TRACE_WAVE_BYTES),
    )
    base = lbytes + ((by * grid_x + bx) * waves + wave_id) * TRACE_WAVE_BYTES
    trs.update(rsrc=rsrc, base=base, level=int(level))
    _tr_store64(rsrc, base, rt)
    _tr_store64(rsrc, base + 8, t)
    _tr_store_pair(rsrc, base + 16, hw1, hw2)
    _tr_store_pair(
        rsrc, base + 24, bx, by | (wave_id << 20) | (fx.Int32(int(level)) << 28)
    )


def _tr_mark(trs, slot, wait=False):
    """TIME stamp into ``slot`` (trace=2 + wait: loadcnt/kmcnt 0 first)."""
    if not trs:
        return
    if wait and trs["level"] >= 2:
        _tr_void("s_wait_loadcnt 0x0\n\ts_wait_kmcnt 0x0")
    _tr_store64(trs["rsrc"], trs["base"] + slot * 8, _tr_time())


def _tr_mark_rt(trs, slot):
    if not trs:
        return
    _tr_store64(trs["rsrc"], trs["base"] + slot * 8, _tr_realtime())


def _tr_end(trs):
    """Store drain (slot 11), exit stamps (12, 13), magic (15)."""
    if not trs:
        return
    _tr_void("s_wait_storecnt 0x0")
    rsrc, base = trs["rsrc"], trs["base"]
    _tr_store64(rsrc, base + 11 * 8, _tr_time())
    _tr_store64(rsrc, base + 12 * 8, _tr_realtime())
    _tr_store64(rsrc, base + 13 * 8, _tr_time())
    _tr_store_pair(
        rsrc, base + 15 * 8, fx.Int32(TRACE_MAGIC[0]), fx.Int32(TRACE_MAGIC[1])
    )


# ROUND-R2D ktile knobs (opt-in, route spellings kt<RB>x<C>[l][p<KB>]):
#   lean=True  -> the full-block path issues ALL its SGPR loads (RB t2r guard rows, the
#                 wave's 4 positions, ctx, every TDM page-table entry) as one batch before
#                 any use (the 4c45af10 ISA serialises each s_buffer_load behind its own
#                 s_wait_kmcnt 0: 16-32 dependent scalar round trips before the first TDM
#                 issue), skips the per-lane metadata VMEM chain (t2r -> ctx) in that path
#                 (vis = max(min(pos + 1, ctx), 0) / 4 with the block's request), publishes
#                 the TDM slab with s_wait_tensorcnt + s_barrier (no workgroup fence: the
#                 fenced gpu.barrier also waited storecnt 0 for the vis/row_starts stores),
#                 and stores the logits of a slab that lies inside the column range under
#                 one SGPR-uniform branch (no per-store EXEC branch + s_wait_xcnt 0).
#                 Mixed / partial blocks keep the exact per-wave fallback.
#   ipf=<KB>   -> S_PREFETCH_INST_PC_REL of the next <KB> KiB of code at kernel entry
#                 (kernel-wiki mi400-scalar-pipeline §4.3: 1-32 x 128 B per instruction;
#                 §4.6: s_wait_xcnt 0 before every s_prefetch on MI450/MI430).
IPF_KB = (4, 8, 16, 32)


def _r2d_check(trace, lean, ipf, ktile, trace_ok):
    trace = int(trace)
    if trace not in (0, 1, 2):
        raise ValueError("trace must be 0, 1 or 2")
    if trace and not (trace_ok or ktile):
        raise ValueError("trace (ROUND-R2D) applies to r32 TDM / ktile builds only")
    if (lean or ipf) and not ktile:
        raise ValueError("lean / ipf (ROUND-R2D) apply to k_load='ktile' only")
    if ipf and int(ipf) not in IPF_KB:
        raise ValueError(f"ipf must be one of {IPF_KB} (KiB)")
    return trace


def _r2d_tag(lean, ipf):
    return ("_lean" if lean else "") + (f"_ipf{int(ipf)}" if ipf else "")


def _pin_sgprs(vals):
    """Empty side-effecting inline asm reading `vals` as SGPR operands (ROUND-R2D lean)."""
    from flydsl._mlir.dialects import llvm

    vals = list(vals)
    if not vals:
        return
    llvm.inline_asm(
        None,
        [fx.Int32(v).ir_value() for v in vals],
        "; r2d pin",
        ",".join(["s"] * len(vals)),
        has_side_effects=True,
    )


def _inst_prefetch(kb):
    """Prefetch ``kb`` KiB of code starting at this point (4 KiB per instruction)."""
    if not kb:
        return
    lines = ["s_wait_xcnt 0x0"]
    for i in range(int(kb) // 4):
        lines.append(f"s_prefetch_inst_pc_rel {i * 4096:#x}, null, 31")
    _tr_void("\n\t".join(lines))


# Exact E4M3 (e4m3fn) -> f32 for one byte code, in integer arithmetic.
#
# gfx1250 has no legal scalar/packed fp8->f32 convert: FlyDSL's
# ``cvt_pk_f32_fp8`` lowers to ``v_cvt_pk_f32_fp8_e32``, which the assembler
# rejects with "operands are not valid for this GPU or mode" (see
# records/g2/builder-log.md). The in-tree gfx1250 fp8 kernels convert inside
# WMMA operands instead. v0 therefore converts in software, which is exact:
#
#   normal (e != 0):  value = (1 + m/8) * 2**(e-7)
#                     f32 bits = (b & 0x7F) << 20 | 120 << 23 | sign
#   e == 0:           value = (m/8) * 2**-6 = m * 2**-9
#                     bits(0x46800000 | m) == 2**14 + m * 2**-9, so subtracting
#                     2**14 gives m * 2**-9 exactly (m <= 7).
#
# NaN codes (0x7F / 0xFF) decode as +-480 instead of NaN; the ATOM fp8 writer
# clamps to +-448 before the cast, so its cache never contains them.
_F32_BIAS = fx.Int32(120 << 23)  # 0x3C000000
_F32_SUB_BIAS = fx.Int32(0x46800000)  # 2**14, mantissa LSB = 2**-9
_F32_SUB_SUB = fx.Float32(16384.0)  # 2**14
_NEG_ONE = fx.Float32(-1.0)


def _e4m3_byte_to_f32(b):
    """One E4M3 byte code (Int32, 0..255) as an exact f32 value."""
    b = fx.Int32(b)
    sign = (b & fx.Int32(0x80)) << 24
    normal = (((b & fx.Int32(0x7F)) << 20) + _F32_BIAS | sign).bitcast(fx.Float32)
    sub_mag = (_F32_SUB_BIAS | (b & fx.Int32(7))).bitcast(fx.Float32) - _F32_SUB_SUB
    sub = (b >= fx.Int32(0x80)).select(sub_mag * _NEG_ONE, sub_mag)
    is_sub = ((b >> 3) & fx.Int32(0xF)) == 0
    return is_sub.select(sub, normal)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@cache
def _build_kernel(
    is_fp8: bool,
    threads: int,
    divisor: float,
    pos64: bool = False,
    page_size: int = PAGE_SIZE,
):
    """One specialization per (dtype, block size, score divisor[, int64 positions][, page])."""
    ld_pos, _, qp_bytes, qp_tag = _pos_loaders(pos64)
    P = _check_page_size(page_size)
    key_row_bytes = HEAD_DIM if is_fp8 else HEAD_DIM * 2  # 128 B fp8 / 256 B bf16
    q_row_bytes = NUM_HEADS * HEAD_DIM * 2  # 1024 B bf16
    page_bytes = P * key_row_bytes  # P=4: 512 B fp8 / 1024 B bf16
    k_dwords = 2 if is_fp8 else 4  # dwords per 8 head dims
    k_bytes = k_dwords * 4  # 8 B fp8 / 16 B bf16
    steps = HEAD_DIM // DIM_PER_STEP  # 16 steps of 8 dims
    inv_divisor = 1.0 / divisor
    per_page = NUM_HEADS * HEAD_DIM  # elements of one q row
    q_dwords = q_row_bytes // 4  # 256 packed bf16 words
    q_stage_steps = (q_dwords + threads - 1) // threads if is_fp8 else 0
    dtype_tag = "fp8" if is_fp8 else "bf16"
    kernel_name = (
        f"qsa_paged_mqa_logits_page4_gfx1250_{dtype_tag}"
        f"_D{HEAD_DIM}_H{NUM_HEADS}_P{P}_t{threads}_flydsl"
    ) + qp_tag

    # The kernel body must not branch on this flag: an `if` over a closure
    # variable is lowered as a traced region, and a name bound inside it does not
    # escape (FlyDSL raises NameError on the use). Specialize with Python closures
    # picked here, at build time, instead.
    def _key_vals_bf16(kraw):
        vals = []
        for j in range_constexpr(k_dwords):
            dw = fx.Int32(kraw[j])
            vals.append((dw << 16).bitcast(fx.Float32))
            vals.append(((dw >> 16) << 16).bitcast(fx.Float32))
        return vals

    def _key_vals_fp8(kraw):
        vals = []
        for j in range_constexpr(k_dwords):
            dw = fx.Int32(kraw[j])
            for sh in range_constexpr(4):
                # byte 0 = dim 4j+0 ... byte 3 = dim 4j+3 (little endian)
                vals.append(_e4m3_byte_to_f32((dw >> (8 * sh)) & fx.Int32(0xFF)))
        return vals

    def _ks_bytes_fp8(num_pages):
        return _plane_bytes(num_pages, P * 4)

    def _ks_bytes_bf16(num_pages):  # no scale plane: a 4-byte dummy descriptor
        return fx.Int64(4)

    def _scale_fp8(ks_rsrc, phys, off):
        return _ld_f32(ks_rsrc, (phys * P + off) * 4)

    def _scale_bf16(ks_rsrc, phys, off):  # no scale: the dequant multiplier is 1
        return fx.Float32(1.0)

    # Round F: change only fp8 Q storage, keeping bf16 scalar (and the separate
    # WMMA builder) untouched.  Pick closures at build time, just like key_vals:
    # no pointer is bound inside a traced dtype branch and then used outside it.
    def _q_lds_fp8(q_rsrc):
        return (
            fx.SharedAllocator(base_alignment=16, static=True)
            .allocate(fx.Array[fx.Int32, q_dwords, 16])
            .peek()
            .ptr
        )

    def _q_direct_bf16(q_rsrc):
        return q_rsrc

    def _q_load_fp8(q_ptr, q_base, h, s):
        # Element offsets are dwords.  Every 4-word consumer is 16-byte aligned
        # and entirely inside [0, 256); unpacking/accumulation below is unchanged.
        return Vec(
            fx.ptr_load(
                q_ptr + h * (HEAD_DIM // 2) + s * 4,
                result_type=Vec.make_type(4, fx.Int32),
            )
        )

    def _q_load_bf16(q_ptr, q_base, h, s):
        return _ld_vec(q_ptr, q_base + h * (HEAD_DIM * 2) + s * 16, 4, fx.Int32)

    def _q_barrier_fp8():
        fx.gpu.barrier()

    def _q_barrier_bf16():
        pass

    if is_fp8:
        key_vals, ks_bytes, scale_of = _key_vals_fp8, _ks_bytes_fp8, _scale_fp8
        q_storage, q_load, q_barrier = _q_lds_fp8, _q_load_fp8, _q_barrier_fp8
    else:
        key_vals, ks_bytes, scale_of = _key_vals_bf16, _ks_bytes_bf16, _scale_bf16
        q_storage, q_load, q_barrier = _q_direct_bf16, _q_load_bf16, _q_barrier_bf16

    @flyc.kernel(name=kernel_name, known_block_size=[threads, 1, 1])
    def kernel(
        Q,  # uint8 view of [rows, 4, 128] bf16
        KV,  # uint8 view of [pages, 4, 1, 128] bf16 or fp8
        KS,  # fp32 view of [pages, 4, 1]  (fp8 only; Q is the placeholder for bf16)
        PT,  # int32 [requests, width]
        T2R,  # int32 [rows]
        QP,  # int32 [rows]
        CL,  # int32 [requests]
        LOGITS,  # fp32 [rows, columns]
        VIS,  # int32 [rows]
        RST,  # int32 [rows]
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        row = fx.Int32(fx.block_idx.y)
        ptile = fx.Int32(fx.block_idx.x)
        col = ptile * threads + tid

        # Descriptor extents: bytes, i64 (see `_plane`).  One pinned descriptor
        # per plane, built here and not inside any branch.  `ks_bytes(num_pages)`
        # is already an i64 expression, and the bf16 dummy plane is a constant.
        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))
        ks_rsrc = _plane(KS, ks_bytes(num_pages))

        # fp8 only: cooperatively stage the row's 1 KiB packed bf16 Q once.
        # The allocator and barrier are unconditional (output tails and invalid
        # metadata participate).  The bf16 closures are a descriptor passthrough
        # and a no-op, with a constexpr zero-trip staging loop: no LDS or barrier.
        q_ptr = q_storage(q_rsrc)
        q_base = row * per_page * 2
        for qi in range_constexpr(q_stage_steps):
            q_dw = tid + qi * threads
            safe_q_dw = _imin(q_dw, fx.Int32(q_dwords - 1))
            q_word = _ld_i32(q_rsrc, q_base + safe_q_dw * 4)
            if q_dw < q_dwords:
                fx.ptr_store(q_word, q_ptr + safe_q_dw)
        q_barrier()

        # --- row metadata: visible count (identical to the golden's visible_groups)
        req = _ld_i32(t2r_rsrc, row * 4)
        req_ok = (req >= 0) & (req < num_requests)
        safe_req = req_ok.select(req, 0)
        ctx = _ld_i32(cl_rsrc, safe_req * 4)
        pos = ld_pos(qp_rsrc, row)
        # min is monotone, so min(a, b) // 4 == min(a // 4, b // 4); clamping to
        # >= 0 first keeps the division on a non-negative operand.
        head = _imax(_imin(pos + 1, ctx), 0)
        vis = head // COMPRESS_RATIO
        vis = req_ok.select(vis, 0)

        # One thread per row writes the row metadata.  Predicated (not sentinel-
        # masked): the store address `row * 4` is in bounds for *every* thread
        # that reaches it, so an un-taken branch can never fault.
        writer = (tid == 0) & (ptile == 0) & (row < rows)
        if writer:
            _st_i32(vis_rsrc, row * 4, vis)
            _st_i32(rst_rsrc, row * 4, fx.Int32(0))

        # --- column -> (logical page, row in page)
        col_ok = col < columns
        logical = col // P
        off = col % P
        logical_ok = logical < width
        safe_logical = logical_ok.select(logical, 0)
        phys = _ld_i32(pt_rsrc, (safe_req * width + safe_logical) * 4)
        page_ok = req_ok & logical_ok & (phys >= 0) & (phys < num_pages)
        phys = page_ok.select(phys, 0)

        # --- score: sum_h relu(q[t,h,:] . k[c]) / sqrt(128)
        k_base = phys * page_bytes + off * key_row_bytes
        zero = fx.Float32(0.0)
        acc = [zero, zero, zero, zero]
        for s in range_constexpr(steps):
            kraw = _ld_vec(kv_rsrc, k_base + s * k_bytes, k_dwords, fx.Int32)
            kv = key_vals(kraw)
            for h in range_constexpr(NUM_HEADS):
                qraw = q_load(q_ptr, q_base, h, s)
                qs = []
                for j in range_constexpr(4):
                    dw = fx.Int32(qraw[j])
                    qs.append((dw << 16).bitcast(fx.Float32))
                    qs.append(((dw >> 16) << 16).bitcast(fx.Float32))
                for d in range_constexpr(DIM_PER_STEP):
                    acc[h] = acc[h] + qs[d] * kv[d]

        scale = scale_of(ks_rsrc, phys, off)
        total = (
            (acc[0] * scale).maximumf(zero)
            + (acc[1] * scale).maximumf(zero)
            + (acc[2] * scale).maximumf(zero)
            + (acc[3] * scale).maximumf(zero)
        )
        total = total * inv_divisor

        keep = page_ok & (col < vis)
        value = keep.select(total, fx.Float32(_NEG_INF))
        # Predicated store: the column tail (`col >= columns`) writes nothing, so
        # the address is always inside [0, rows * columns * 4).  It is clamped as
        # well, so even a mis-lowered predicate degrades to one corrupted element
        # (the last column of the last row) instead of an out-of-buffer access.
        # v0 instead stored at the constant byte offset 0x7FFFFFFF, which relies
        # on the V# bounds check actually being armed; the d07-2 VM_PAGE_FAULT
        # (page start 0x16bf5fff000 == a 2 MiB-aligned base + 0x7FFFFFFF) says it
        # was not.  Do not reintroduce that pattern.
        out_idx = _imin(row * columns + col, rows * columns - 1)
        if col_ok:
            _st_f32(logits_rsrc, out_idx * 4, value)

    @flyc.jit
    def launch(
        Q,
        KV,
        KS,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        grid_x,
        rows,
        columns,
        num_pages,
        num_requests,
        width,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel._func.__name__ = kernel_name
        kernel(
            Q,
            KV,
            KS,
            PT,
            T2R,
            QP,
            CL,
            LOGITS,
            VIS,
            RST,
            rows,
            columns,
            num_pages,
            num_requests,
            width,
        ).launch(
            grid=(fx.Int64(grid_x), fx.Int64(rows), 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch.kernel_name = kernel_name
    launch.dtype_tag = dtype_tag
    return launch


# ---------------------------------------------------------------------------
# bf16 matrix-core path (round C): gfx1250 WMMA 16x16x32 bf16 -> f32
# ---------------------------------------------------------------------------
# Port of the *design* of the task package's FlyDSL MFMA scorer
# (``reference/aiter_kernels/flydsl__mqa_logits__qsa_flydsl_layout.py``: Q kept
# in VGPRs, per-head QK^T on the matrix core, ReLU per head then head sum, a
# request-dedup pass loop, early exit for invisible column blocks) onto what
# gfx1250 actually has.  That file is a wave64 CDNA kernel (``rocdl.MFMA`` ->
# ``v_mfma``, ``lane = tid % 64``, 64-lane shuffles); gfx1250 is wave32 and has
# no MFMA, only WMMA (FlyDSL ``MmaOpGFX1250_WMMAType``: bf16 needs K = 32).  The
# operand / accumulator register layouts used below are the ones FlyDSL
# documents and tests for gfx1250 (``tests/unit/test_gfx1250_mma_atom_precision
# _check.py`` and ``kernels/gemm/gemm_bf16_gfx1250.py`` at v0.3.3.dev889, the
# closest tag to the image's 0.3.3.dev886):
#
#   A [16 x 32] / B [16 x 32] (row = M resp. N), lane l, bf16 element v < 16:
#       row = l % 16,   k = (v // 8) * 16 + (l // 16) * 8 + v % 8
#     i.e. two contiguous 16-byte chunks at k-byte offsets (l//16)*16 and +32;
#   C/D [16 x 16] f32, lane l, element v < 8:
#       m = (l // 16) * 8 + v,   n = l % 16.
#
# Mapping (one wave = one WMMA M tile):
#   * M = 16 = 4 consecutive query rows x 4 index heads (slot m = 4*row + head).
#     The four heads of a row land in four consecutive accumulator elements of
#     one lane, so "ReLU per head, then sum over heads" is lane-local:
#     lane (g, n) owns rows 2g and 2g+1 of its column n.
#   * N = 16 columns per WMMA tile, 4 tiles = 64 columns per wave.
#   * K = 128 head dims = 4 WMMA steps; Q fragments are loaded once per wave
#     (8 x 16 B per lane) and kept in VGPRs; K fragments go straight from global
#     (page-4 geometry: column c -> page_table[req, c // 4], row c % 4) into the
#     B operand, 8 x 16 B per lane per tile Ã¢â‚¬â€ every K byte is loaded exactly once
#     per wave, so no LDS staging and no barriers are needed.
#   * Rows of one wave may belong to up to 4 requests (draft: 1 row/request).
#     Pass p handles the rows whose request equals row p's, and only if row p is
#     the first row with that request ("request dedup", the reference's
#     ``req_low..req_high`` loop without its 16x redundant MMA work on decode).
#   * Early exit: a pass whose rows see none of the wave's 64 columns writes -inf
#     without loading K (the reference skipped the block entirely; here the
#     contract needs -inf for every c >= visible, so it is written, not skipped).
#   * Every logit in [0, rows) x [0, columns) is written exactly once (no memset).
#
# Safety contract (unchanged from 6c272c3c): all stores are predicated
# (``if ok: store``) at a clamped, in-plane byte offset; no sentinel offsets;
# every load index is clamped in bounds (rows -> rows-1, pages -> 0, logical
# page -> 0, request -> 0); descriptor extents are the true plane sizes (i64).
WMMA_WAVE = 32  # gfx1250 is wave32
WMMA_WAVES = 4  # waves per workgroup
WMMA_THREADS = WMMA_WAVE * WMMA_WAVES  # 128
WMMA_ROWS_PER_WAVE = 4  # x 4 heads = WMMA M 16
WMMA_ROWS_PER_BLOCK = WMMA_ROWS_PER_WAVE * WMMA_WAVES  # 16
WMMA_N = 16
WMMA_K = 32
WMMA_N_TILES = 4
WMMA_COLS = WMMA_N * WMMA_N_TILES  # 64 columns per workgroup
WMMA_K_STEPS = HEAD_DIM // WMMA_K  # 4
WMMA_PASSES = WMMA_ROWS_PER_WAVE  # at most 4 distinct requests per wave

# QSA_BF16_PATH=scalar restores the v0 one-thread-per-column bf16 kernel (a
# one-variable revert for the GPU bisect; the fp8 path always uses it).
_BF16_PATH = os.environ.get("QSA_BF16_PATH", "wmma").strip().lower()
if _BF16_PATH not in ("wmma", "scalar"):
    raise ValueError(f"QSA_BF16_PATH must be 'wmma' or 'scalar', got {_BF16_PATH!r}")


# Auto selects the measured r32 prefill and r16 c32 decode/draft TDM wins.
# A source row has 64 DWORDs; skip four DWORDs after each row in LDS. Stock
# FlyDSL's descriptor API takes *elements*, not encoded fields or DWORD counts:
# bf16 128/8 elements -> 64/4 DWORDs -> encoded interval=5, amount=3.
# The 272-byte pitch keeps b128 consumers aligned and rotates their bank groups.
TDM_PAD_INTERVAL_DWORDS = 64
TDM_PAD_AMOUNT_DWORDS = 4
TDM_K_ROW_BYTES = (TDM_PAD_INTERVAL_DWORDS + TDM_PAD_AMOUNT_DWORDS) * 4
TDM_K_TILE_BYTES = WMMA_COLS * TDM_K_ROW_BYTES  # 17,408 B, already 128B aligned

_BF16_K_LOAD = os.environ.get("QSA_BF16_K_LOAD", "auto").strip().lower()
if _BF16_K_LOAD not in ("raw", "tdm", "auto"):
    raise ValueError(
        f"QSA_BF16_K_LOAD must be 'raw', 'tdm' or 'auto', got {_BF16_K_LOAD!r}"
    )

# BM64 is a measured regression; only explicit opt-in may select it.
_BF16_R64 = os.environ.get("QSA_BF16_R64", "0").strip()
if _BF16_R64 not in ("0", "1"):
    raise ValueError(f"QSA_BF16_R64 must be '0' or '1', got {_BF16_R64!r}")
_BF16_R64 = _BF16_R64 == "1"

# Round R16: column width of the *TDM* r16 specialization only (raw r16
# and every r32 kernel keep 64). 32 halves each wave-private padded tile to
# 32 x 272 = 8,704 B, i.e. 34,816 B per workgroup instead of 69,632 B; grid.x
# doubles and per-column K traffic is unchanged. 64 restores the TDM2 layout.
_BF16_R16_TDM_COLS = os.environ.get("QSA_BF16_R16_TDM_COLS", "32").strip()
if _BF16_R16_TDM_COLS not in ("32", "64"):
    raise ValueError(
        f"QSA_BF16_R16_TDM_COLS must be '32' or '64', got {_BF16_R16_TDM_COLS!r}"
    )
R16_TDM_COLS = int(_BF16_R16_TDM_COLS)
# Round DRAFT/NOBAR: barrier-free wave-private r16 TDM passes. ROUND-MERGE1:
# default 1 (GPU-verified, decode bf16 c32768 16.9 -> 16.2 us); QSA_R16_NOBAR=0
# restores the barriered champion r16 TDM kernel byte-for-byte.
if os.environ.get("QSA_R16_NOBAR", "1").strip() not in ("0", "1"):
    raise ValueError("QSA_R16_NOBAR must be '0' or '1'")
TDM_R16_NOBAR = os.environ.get("QSA_R16_NOBAR", "1").strip() == "1"

# Round PERSISTDEF (measured default): bf16 r32 *persistent* prefill. One
# workgroup owns 32 query rows and loops over TDM_PERSIST_CHUNKS consecutive
# 64-column K tiles with two 17,408 B (272 B pitch) buffers = 34,816 B LDS:
# chunk k+1's TDM gathers are issued into the idle buffer before chunk k's
# WMMA/store, then tensor_wait(0) + one WG barrier publish it (and retire the
# buffer just read). QSA_BF16_PERSIST=0 forces the byte-identical single-tile
# r32 kernel for A/B; QSA_BF16_PERSIST=1 (default) selects persistent for guarded
# r32 TDM shapes with columns >= 64 * QSA_PERSIST_COLS (default 8).
if os.environ.get("QSA_BF16_PERSIST", "1").strip() not in ("0", "1"):
    raise ValueError("QSA_BF16_PERSIST must be '0' or '1'")
if os.environ.get("QSA_PERSIST_COLS", "8").strip() not in ("2", "4", "8"):
    raise ValueError("QSA_PERSIST_COLS must be '2', '4' or '8'")
TDM_PERSIST = os.environ.get("QSA_BF16_PERSIST", "1").strip() == "1"
TDM_PERSIST_CHUNKS = int(os.environ.get("QSA_PERSIST_COLS", "8").strip())
TDM_PERSIST_BUFFERS = 2
TDM_PERSIST_LDS_BYTES = TDM_PERSIST_BUFFERS * TDM_K_TILE_BYTES  # 34,816 B

# ROUND-R2B: compile time of the persistent kernel. Its exact raw fallback (mixed /
# partial row blocks only) was unrolled over persist_chunks x 2 M tiles x 4 passes x
# 4 N tiles (~1,024 WMMAs, ~80% of the code, ~12 s cold compile). 'loop' (default)
# runs the fallback's chunk loop as a runtime scf.for (unroll disabled); the hot
# single-request path is unchanged (still unrolled + double-buffered). 'unroll'
# restores the d438d2b3 persistent kernel byte-for-byte (same name / ISA) for GPU A/B.
PERSIST_FALLBACKS = ("loop", "unroll")
TDM_PERSIST_FALLBACK = os.environ.get("QSA_PERSIST_FALLBACK", "loop").strip().lower()
if TDM_PERSIST_FALLBACK not in PERSIST_FALLBACKS:
    raise ValueError("QSA_PERSIST_FALLBACK must be 'loop' or 'unroll'")

# ROUND-R2B: K-stationary / Q-reuse r32 prefill ("kstat", builder opt-in or route table).
# A workgroup owns a column slab of `col_tiles` 64-column K tiles, gathers them ONCE
# into LDS (col_tiles x 17,408 B; one TDM issue per tile, no reload), then streams
# `row_tiles` 32-row blocks through a runtime loop: each block's Q fragments are loaded
# once and reused for every K tile of the slab (Q traffic / col_tiles), K is gathered
# once per row_tiles blocks (K traffic / row_tiles). grid = ceil(columns / (64 C)) x
# ceil(rows / (32 R)). Blocks that are mixed / partial / of another request than the
# slab's K request (= the span's first row) take the exact r32 raw fallback.
KSTAT_ROW_TILES = (1, 2, 4, 8)
KSTAT_COL_TILES = (1, 2, 4)


def kstat_grid(
    rows: int, columns: int, row_tiles: int, col_tiles: int
) -> tuple[int, int]:
    """Grid of the kstat kernel (pure arithmetic; CPU tests + op plans)."""
    if row_tiles not in KSTAT_ROW_TILES or col_tiles not in KSTAT_COL_TILES:
        raise ValueError(
            f"kstat row_tiles {row_tiles} / col_tiles {col_tiles} unsupported"
        )
    cw, rh = WMMA_COLS * col_tiles, 32 * row_tiles
    return (int(columns) + cw - 1) // cw, (int(rows) + rh - 1) // rh


# ROUND-R2C: "ktile" r32-routed prefill (builder k_load='ktile', route variant kt<RB>x<C>).
# One wave per 4-row M tile (RB = 16 -> 4 waves = the native MFMA scorer's BMT=16 x BN=128
# tiling at C = 2; RB = 32 -> 8 waves / 256 threads), every wave scores ALL 4C N tiles of the
# workgroup's 64C-column slab from LDS (gathered once by TDM, 8-row gathers spread over the
# waves). Full single-request blocks issue Q, the lane columns' page validity and the TDM
# gathers before the one wait + barrier (the R2C 'early' order); other blocks take the exact
# per-wave raw fallback (request-dedup passes). grid = ceil(cols / 64C) x ceil(rows / RB).
KT_ROWS = (16, 32)
KT_COL_TILES = (1, 2, 4)


def kt_grid(rows: int, columns: int, kt_rows: int, col_tiles: int) -> tuple[int, int]:
    """Grid of the ktile kernel (pure arithmetic; CPU tests + op plans)."""
    if kt_rows not in KT_ROWS or col_tiles not in KT_COL_TILES:
        raise ValueError(f"ktile rows {kt_rows} / col_tiles {col_tiles} unsupported")
    cw = WMMA_COLS * col_tiles
    return (int(columns) + cw - 1) // cw, (int(rows) + kt_rows - 1) // kt_rows


# ROUND-R2B: data-driven bf16 r32 prefill route (rows >= 32 and rows >= 32 * requests,
# TDM, 128 B aligned cache). QSA_PREFILL_ROUTE is a preset name or a rule list
# "P,cmin,cmax,rmin,rmax,variant;..." (bounds inclusive, '*' = any; P in 4/8/16/'*');
# variant = persist | tdm | kstat<R>x<C>. First match wins; no match -> tdm (r32).
# 'legacy' is the d438d2b3 router: persist iff QSA_BF16_PERSIST=1 and
# columns >= 64 * QSA_PERSIST_COLS. QSA_BF16_PERSIST=0 still vetoes persist everywhere.
PREFILL_ROUTE_PRESETS = {
    "legacy": None,
    # R2B proposal (GPU sweep picks the thresholds): persist only for long columns,
    # Q-reuse kstat below (prefill_bench c1024: persist 17.5-19.7 us vs r32 6.0/13.4).
    "r2b": "*,4096,*,*,*,persist;*,*,*,2048,*,kstat2x2;*,*,*,*,*,kstat1x2",
    # ROUND-R2E default: lean ktile RB32 x C2 for columns <= 2048 (GPU c1024: rows 1024
    # 4.14-4.19 vs native 4.76; rows 4096 7.75-7.80 vs 12.2), persist above (long-column
    # A/B pending: records/r2e/GPU_COMMANDS.md (d)).
    "r2e": "*,*,2048,*,16384,kt32x2l;*,*,*,*,*,persist",
    # ROUND-R2E alternative for the long-column A/B: lean ktile everywhere.
    "kt_all": "*,*,*,*,*,kt32x2l",
}
#: ROUND-R2E: the default preset (was 'legacy' up to R2D 685b0f60).
PREFILL_ROUTE_DEFAULT = "r2e"


def _parse_route_variant(v: str):
    v = v.strip().lower()
    if v in ("persist", "tdm"):
        return v, {}
    m = re.fullmatch(r"kstat(\d+)x(\d+)(e?)", v)
    if m and int(m[1]) in KSTAT_ROW_TILES and int(m[2]) in KSTAT_COL_TILES:
        kw = {"row_tiles": int(m[1]), "col_tiles": int(m[2])}
        if m[3]:  # ROUND-R2C: page-table prefetch before the K wait
            kw["early"] = True
        return "kstat", kw
    m = re.fullmatch(
        r"kt(\d+)x(\d+)(l?)(?:p(\d+))?", v
    )  # ROUND-R2C ktile (+R2D l / p<KB>)
    if m and int(m[1]) in KT_ROWS and int(m[2]) in KT_COL_TILES:
        kw = {"kt_rows": int(m[1]), "col_tiles": int(m[2])}
        if m[3]:
            kw["lean"] = True
        if m[4]:
            if int(m[4]) not in IPF_KB:
                raise ValueError(f"QSA_PREFILL_ROUTE: ipf {m[4]} not in {IPF_KB}")
            kw["ipf"] = int(m[4])
        return "ktile", kw
    raise ValueError(f"QSA_PREFILL_ROUTE: unknown variant {v!r}")


def parse_prefill_route(spec: str):
    """Rule tuples (page, cmin, cmax, rmin, rmax, k_load, kw) or None (= legacy)."""
    spec = PREFILL_ROUTE_PRESETS.get(spec.strip().lower(), spec)
    if spec is None:
        return None
    rules = []
    for part in [p for p in spec.split(";") if p.strip()]:
        f = [x.strip().lower() for x in part.split(",")]
        if len(f) != 6:
            raise ValueError(f"QSA_PREFILL_ROUTE rule {part!r}: need 6 fields")
        page = None if f[0] == "*" else int(f[0])
        if page is not None and page not in SUPPORTED_PAGE_SIZES:
            raise ValueError(f"QSA_PREFILL_ROUTE rule {part!r}: page {page}")
        b = [None if x == "*" else int(x) for x in f[1:5]]
        rules.append((page, b[0], b[1], b[2], b[3]) + _parse_route_variant(f[5]))
    return tuple(rules)


PREFILL_ROUTE_SPEC = (
    os.environ.get("QSA_PREFILL_ROUTE", PREFILL_ROUTE_DEFAULT).strip().lower()
)
PREFILL_ROUTE = parse_prefill_route(PREFILL_ROUTE_SPEC)


def prefill_bf16_route(
    rows: int, num_requests: int, columns: int, page_size: int = PAGE_SIZE, table=None
):
    """(k_load, builder keywords) of a guarded bf16 r32 TDM prefill call."""
    table = PREFILL_ROUTE if table is None else table
    if table is None or table == "legacy":
        if TDM_PERSIST and int(columns) >= WMMA_COLS * TDM_PERSIST_CHUNKS:
            return "persist", {}
        return "tdm", {}
    for page, cmin, cmax, rmin, rmax, k_load, kw in table:
        if page is not None and page != int(page_size):
            continue
        if (
            cmin is not None
            and int(columns) < cmin
            or cmax is not None
            and int(columns) > cmax
        ):
            continue
        if (
            rmin is not None
            and int(rows) < rmin
            or rmax is not None
            and int(rows) > rmax
        ):
            continue
        if k_load == "persist" and not (TDM_PERSIST and int(columns) >= WMMA_COLS * 2):
            continue
        return k_load, dict(kw)
    return "tdm", {}


def persist_schedule(n_chunks: int):
    """Trace-time event plan of the persistent kernel's K pipeline.

    Pure Python (used by the kernel body *and* the CPU tests): per workgroup,
    ``("issue", chunk, buffer)`` = every wave's two 8-row TDM gathers of that
    chunk into that buffer (skipped at runtime by an SGPR-uniform guard when
    the chunk is beyond every row's visibility / the column extent), ``("wait",)``
    = ``tensor_wait(0)`` of the wave's own gathers, ``("barrier",)`` = one
    workgroup barrier, ``("consume", chunk, buffer)`` = WMMA + stores.
    """
    if n_chunks not in (1, 2, 4, 8):
        raise ValueError("persistent chunk count must be 1, 2, 4 or 8")
    events = [("issue", 0, 0), ("wait",), ("barrier",)]
    for k in range(n_chunks):
        if k + 1 < n_chunks:
            events.append(("issue", k + 1, (k + 1) % TDM_PERSIST_BUFFERS))
        events.append(("consume", k, k % TDM_PERSIST_BUFFERS))
        if k + 1 < n_chunks:
            events += [("wait",), ("barrier",)]
    return events


def _resolve_bf16_k_load(rows_per_block: int, k_load: str | None = None):
    """Resolve geometry routing before building a compile-time specialization."""
    mode = _BF16_K_LOAD if k_load is None else k_load
    if mode not in ("raw", "tdm", "auto"):
        raise ValueError("bf16 k_load must be raw, tdm or auto")
    return "tdm" if mode == "auto" else mode


def _ld_uniform_i32(rsrc, byte_off):
    """Wave-uniform metadata load directly into SGPRs (no readfirstlane)."""
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm

    bits = llvm.ptrtoint(ir.IntegerType.get_signless(128), rsrc)
    v4 = llvm.bitcast(ir.VectorType.get([4], fx.Int32.ir_type), bits)
    return fx.Int32(
        llvm.call_intrinsic(
            fx.Int32.ir_type,
            "llvm.amdgcn.s.buffer.load.i32",
            [v4, fx.Int32(byte_off).ir_value(), fx.Int32(0).ir_value()],
            [],
            [],
        )
    )


# ROUND-NATIVE64: query_positions load seam shared by every builder.
#
# pos64=False (int32 callers): ``row * 4`` + one dword load -- exactly the IR the
# builders emitted before this round (byte-identical ISA, same kernel names).
# pos64=True: the caller's int64 tensor is passed as its zero-copy int32 view
# ([2*rows], little endian: dword 2r = low word, 2r+1 = high word) and the
# wrapper's former ``clamp(v, INT32_MIN, INT32_MAX-1)`` is computed in-kernel:
#
#   v fits int32  <=>  hi == sign_extension(lo)  ->  min(lo, INT32_MAX-1)
#   otherwise v >= 2**31 iff hi >= 0              ->  INT32_MAX-1
#             v <  -2**31 iff hi <  0             ->  INT32_MIN
#
# All arithmetic is i32 compare/select (no shifts, no wrap). Builders pick the
# loader at build time (never inside a traced branch).
_POS_SAT_HI = 2**31 - 2
_POS_SAT_LO = -(2**31)


def _sat_pos64(lo, hi):
    """Saturated i32 of the int64 ``hi:lo`` == ``clamp(v, INT32_MIN, INT32_MAX-1)``."""
    ext = (lo < 0).select(fx.Int32(-1), fx.Int32(0))
    fits = hi == ext
    outside = (hi >= 0).select(fx.Int32(_POS_SAT_HI), fx.Int32(_POS_SAT_LO))
    return fits.select(_imin(lo, fx.Int32(_POS_SAT_HI)), outside)


def _ld_pos_i32(rsrc, row):
    return _ld_i32(rsrc, row * 4)


def _ld_pos_i64(rsrc, row):
    words = _ld_vec(rsrc, row * 8, 2, fx.Int32)  # one 8-byte load per row
    return _sat_pos64(fx.Int32(words[0]), fx.Int32(words[1]))


def _ld_uniform_pos_i32(rsrc, row):
    return _ld_uniform_i32(rsrc, row * 4)


def _ld_uniform_pos_i64(rsrc, row):
    return _sat_pos64(
        _ld_uniform_i32(rsrc, row * 8), _ld_uniform_i32(rsrc, row * 8 + 4)
    )


def _pos_loaders(pos64: bool):
    """``(per-lane loader, SGPR loader, plane bytes per row, kernel-name suffix)``."""
    if pos64:
        return _ld_pos_i64, _ld_uniform_pos_i64, 8, "_qp64"
    return _ld_pos_i32, _ld_uniform_pos_i32, 4, ""


def _tdm_k_gather(KV, ptr, row_ids, num_pages, byte_off, page_size=PAGE_SIZE):
    """Stock TDM gather API, adapting the Fly LDS pointer to memref extraction."""
    from flydsl._mlir.dialects import memref
    from flydsl.expr.rocdl import tdm_ops
    from flydsl.expr.typing import as_ir_value

    class _LDS:
        def __getattr__(self, name):
            return getattr(memref, name)

        @staticmethod
        def extract_aligned_pointer_as_index(source):
            return fx.Index(fx.ptrtoint(fx.Pointer(as_ir_value(source)))).ir_value()

    saved = tdm_ops.memref_dialect
    tdm_ops.memref_dialect = _LDS()
    try:
        desc = tdm_ops.make_tensor_gather_descriptor(
            global_ptr=KV,
            lds_memref=ptr,
            row_indices=row_ids,
            row_width=HEAD_DIM,
            tensor_dim0=HEAD_DIM,
            tensor_dim1=(num_pages * page_size).ir_value(),
            stride=HEAD_DIM,
            elem_bytes=2,
            index_size=32,
            pad_interval=TDM_PAD_INTERVAL_DWORDS * 2,
            pad_amount=TDM_PAD_AMOUNT_DWORDS * 2,
            workgroup_mask=0,
            lds_byte_offset=fx.Index(byte_off),
        )
    finally:
        tdm_ops.memref_dialect = saved
    tdm_ops.tensor_load_gather(desc)


def _tdm_fp8_k_gather(KV, ptr, row_ids, num_pages, byte_off, page_size=PAGE_SIZE):
    """Stock TDM gather API, adapting the Fly LDS pointer to memref extraction."""
    from flydsl._mlir.dialects import memref
    from flydsl.expr.rocdl import tdm_ops
    from flydsl.expr.typing import as_ir_value

    class _LDS:
        def __getattr__(self, name):
            return getattr(memref, name)

        @staticmethod
        def extract_aligned_pointer_as_index(source):
            return fx.Index(fx.ptrtoint(fx.Pointer(as_ir_value(source)))).ir_value()

    saved = tdm_ops.memref_dialect
    tdm_ops.memref_dialect = _LDS()
    try:
        desc = tdm_ops.make_tensor_gather_descriptor(
            global_ptr=KV,
            lds_memref=ptr,
            row_indices=row_ids,
            row_width=HEAD_DIM,
            tensor_dim0=HEAD_DIM,
            tensor_dim1=(num_pages * page_size).ir_value(),
            stride=HEAD_DIM,
            elem_bytes=1,
            index_size=32,
            pad_interval=TDM_FP8_PAD_INTERVAL_DWORDS * 4,
            pad_amount=TDM_FP8_PAD_AMOUNT_DWORDS * 4,
            workgroup_mask=0,
            lds_byte_offset=fx.Index(byte_off),
        )
    finally:
        tdm_ops.memref_dialect = saved
    tdm_ops.tensor_load_gather(desc)


# ROUND-PAGE16 TDM row-id plan. A 32-bit-index TDM gather moves at most 8 rows
# (kernel-wiki mi400-tdm-programming: "32-bit (8 indices/instruction)"), and
# every gather group of this file is 8 consecutive output columns starting at
# a multiple of 8 (tile/slab origins are multiples of 32). P = 4: a group spans
# two pages -> two s_buffer_load page lookups (the unchanged bf492262 code at
# every call site). P = 16: a group lies inside ONE page (8 | 16), so a 16-row
# page is moved by two 8-row gathers sharing ONE page-table lookup each:
# rows phys * 16 + row0 + r, row0 = (group * 8) % 16 a trace-time constant.
# The LDS destination layout (272 B / 144 B pitch, column c at row c % tile)
# is unchanged, so WMMA B reads keep the TDM2 bank rotation.
def tdm_group_plan(page_size: int, group_col: int):
    """Trace-free mirror of one 8-row gather group (CPU tests and kernels).

    ``group_col`` is the group's first column *relative to a page-aligned
    origin* (a multiple of 8). Returns ``[(page_delta, [row_in_page...])]``:
    the logical page offset of each page-table lookup and the rows it feeds.
    """
    P = _check_page_size(page_size)
    if group_col % 8:
        raise ValueError("TDM gather groups start at multiples of 8 columns")
    if P == 4:
        return [(group_col // 4 + page, list(range(4))) for page in range(2)]
    return [(group_col // P, [group_col % P + r for r in range(8)])]  # P = 8: row0 = 0


def _p16_row_ids(
    pt_rsrc, req, width, num_pages, logical, row0, req_ok=None, page_size=16
):
    """P >= 8 gather group: one SGPR page lookup feeding 8 row ids (see above).

    ROUND-R2 P = 8 (KV block 32): ``page_size=8``, ``row0 = 0`` -- the 8-row
    gather is exactly one page (rows ``phys * 8 + r``). P = 16 callers pass no
    ``page_size`` (trace-identical to 536667d5).
    """
    logical_ok = logical < width
    safe_logical = logical_ok.select(logical, 0)
    phys = _ld_uniform_i32(pt_rsrc, (req * width + safe_logical) * 4)
    valid = logical_ok & (phys >= 0) & (phys < num_pages)
    if req_ok is not None:
        valid = req_ok & valid
    phys = valid.select(phys, 0)
    return [phys * page_size + (row0 + r) for r in range(8)]


class _RowMeta:
    """Per-row metadata of one of a wave's 4 rows (uniform across the wave)."""

    def __init__(self, row, row_ok, req, req_ok, safe_req, vis):
        self.row, self.row_ok, self.req = row, row_ok, req
        self.req_ok, self.safe_req, self.vis = req_ok, safe_req, vis


class _Pass:
    """One request pass of a wave, already specialised to the calling lane."""

    def __init__(
        self, active, vmax, safe_req, req_ok, own_a, own_b, vis_a, vis_b, kgrp
    ):
        self.active, self.vmax = active, vmax
        self.safe_req, self.req_ok = safe_req, req_ok
        self.own_a, self.own_b = own_a, own_b
        self.vis_a, self.vis_b = vis_a, vis_b
        self.kgrp = kgrp


@cache
def _build_wmma_kernel(
    divisor: float,
    rows_per_block: int = 16,
    k_load: str | None = None,
    persist_chunks: int | None = None,
    pos64: bool = False,
    page_size: int = PAGE_SIZE,
    persist_fallback: str | None = None,
    row_tiles: int = 1,
    col_tiles: int = 1,
    early: bool = False,
    kt_rows: int | None = None,
    trace: int = 0,
    lean: bool = False,
    ipf: int = 0,
):
    """bf16 WMMA specialization; r32 shares B across two M tiles per wave.

    ``k_load="persist"`` (wrapper opt-in only) builds the r32 persistent TDM
    kernel looping over ``persist_chunks`` (default TDM_PERSIST_CHUNKS) tiles;
    ``persist_fallback`` (ROUND-R2B, default QSA_PERSIST_FALLBACK) selects its raw
    fallback form. ``k_load="kstat"`` (ROUND-R2B) builds the K-stationary / Q-reuse
    r32 kernel (``row_tiles`` x ``col_tiles``, see KSTAT_*).
    ``pos64=True`` (ROUND-NATIVE64) reads query_positions as int64 (see
    ``_pos_loaders``); ``False`` is the unchanged int32 kernel.
    """
    ld_pos, ld_uniform_pos, qp_bytes, qp_tag = _pos_loaders(pos64)
    P = _check_page_size(page_size)
    if rows_per_block not in (16, 32, 64):
        raise ValueError("bf16 WMMA rows_per_block must be 16, 32 or 64")
    persist = k_load == "persist"
    kstat = k_load == "kstat"
    ktile = k_load == "ktile"  # ROUND-R2C
    if ktile:
        if rows_per_block != 32 or _ACCESS != "raw" or row_tiles != 1 or early:
            raise ValueError(
                "the ktile bf16 kernel is r32-routed TDM (raw access), "
                "row_tiles=1, early built in"
            )
        kt_grid(32, 64, kt_rows, col_tiles)  # validates the knobs
        k_load = "tdm"
    elif kt_rows is not None:
        raise ValueError("kt_rows applies to k_load='ktile' only")
    kt_waves = (int(kt_rows) // WMMA_ROWS_PER_WAVE) if ktile else WMMA_WAVES
    kt_threads = kt_waves * WMMA_WAVE
    fb_loop = False
    if persist:
        fb = TDM_PERSIST_FALLBACK if persist_fallback is None else str(persist_fallback)
        if fb not in PERSIST_FALLBACKS:
            raise ValueError("persist_fallback must be 'loop' or 'unroll'")
        fb_loop = fb == "loop"
    elif persist_fallback is not None:
        raise ValueError("persist_fallback applies to k_load='persist' only")
    if kstat:
        if rows_per_block != 32 or _ACCESS != "raw":
            raise ValueError("the kstat bf16 kernel is r32 TDM (raw access) only")
        kstat_grid(32, 64, row_tiles, col_tiles)  # validates the knobs
        k_load = "tdm"
    elif (row_tiles, col_tiles) != (1, 1) and not ktile:
        raise ValueError("row_tiles / col_tiles apply to k_load='kstat' only")
    early = bool(early)
    if early and not kstat:
        raise ValueError("early (ROUND-R2C) applies to k_load='kstat' only")
    if persist:
        if rows_per_block != 32 or _ACCESS != "raw":
            raise ValueError("the persistent bf16 kernel is r32 TDM (raw access) only")
        persist_chunks = (
            TDM_PERSIST_CHUNKS if persist_chunks is None else int(persist_chunks)
        )
        if persist_chunks not in (2, 4, 8):
            raise ValueError("persistent chunk count must be 2, 4 or 8")
        k_load = "tdm"
    else:
        persist_chunks = 1
    k_load = _resolve_bf16_k_load(rows_per_block, k_load)
    inv_divisor = 1.0 / divisor
    use_tdm = k_load == "tdm" and _ACCESS == "raw"
    # Trace-time geometry: only the TDM r16 kernel may narrow its column block.
    tdm_cols = R16_TDM_COLS if rows_per_block == 16 else WMMA_COLS
    tdm_tile_bytes = tdm_cols * TDM_K_ROW_BYTES  # 8,704 B (c32) / 17,408 B (c64)
    cols_per_block = tdm_cols if use_tdm else WMMA_COLS
    # ROUND-R2D opt-in knobs (defaults emit nothing new): trace (phase timestamps),
    # lean / ipf (ktile only; see _r2d_tag).
    trace = _r2d_check(
        trace,
        lean,
        ipf,
        ktile,
        use_tdm and rows_per_block == 32 and not persist and not kstat,
    )
    trs = {}  # trace holder (filled by _tr_begin at trace time)

    def _pair_storage_raw(KV, kv_rsrc):
        return kv_rsrc

    def _pair_storage_tdm(KV, kv_rsrc):
        return (
            fx.SharedAllocator(base_alignment=128, static=True)
            .allocate(fx.Array[fx.Int32, TDM_K_TILE_BYTES // 4, 128])
            .peek()
            .ptr
        )

    def _r16_storage_tdm():
        # One allocator, four disjoint padded tiles (one per wave). Different
        # requests cannot share a K tile. c32: 34,816 B; c64: 69,632 B.
        return (
            fx.SharedAllocator(base_alignment=128, static=True)
            .allocate(fx.Array[fx.Int32, WMMA_WAVES * tdm_tile_bytes // 4, 128])
            .peek()
            .ptr
        )

    def _rows_meta_uniform(t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests):
        # The TDM r16 kernel supplies SGPR wave_id, so every address and result
        # here is explicitly scalar. Avoid LLVM VGPR-to-SGPR readfirstlane CSE.
        out = []
        for i in range_constexpr(WMMA_ROWS_PER_WAVE):
            row = t0 + i
            row_ok = row < rows
            safe_row = _imin(row, rows - 1)
            req = _ld_uniform_i32(t2r_rsrc, safe_row * 4)
            req_ok = (req >= 0) & (req < num_requests)
            safe_req = req_ok.select(req, 0)
            ctx = _ld_uniform_i32(cl_rsrc, safe_req * 4)
            pos = ld_uniform_pos(qp_rsrc, safe_row)
            head = _imax(_imin(pos + 1, ctx), 0)
            vis = req_ok.select(head // COMPRESS_RATIO, 0)
            out.append(_RowMeta(row, row_ok, req, req_ok, safe_req, vis))
        return out

    def _r16_stage_active(t2r_rsrc, qp_rsrc, cl_rsrc, rblk, col0, num_requests, p):
        """SGPR-only request dedup/visibility guard; no TDM in EXEC-only branches."""
        wave_id = fx.Int32(fx.rocdl.wave_id())
        t0 = rblk * 16 + wave_id * WMMA_ROWS_PER_WAVE
        req = _ld_uniform_i32(t2r_rsrc, (t0 + p) * 4)
        req_ok = (req >= 0) & (req < num_requests)
        safe_req = req_ok.select(req, 0)
        ctx = _ld_uniform_i32(cl_rsrc, safe_req * 4)
        active = req_ok
        for prior in range_constexpr(p):
            other = _ld_uniform_i32(t2r_rsrc, (t0 + prior) * 4)
            active = active & (other != req)
        vmax = fx.Int32(0)
        for i in range_constexpr(WMMA_ROWS_PER_WAVE):
            owner = _ld_uniform_i32(t2r_rsrc, (t0 + i) * 4)
            pos = ld_uniform_pos(qp_rsrc, t0 + i)
            vis = _imax(_imin(pos + 1, ctx), 0) // COMPRESS_RATIO
            vmax = _imax(vmax, (owner == req).select(vis, 0))
        return active & (col0 < vmax)

    def _r16_stage_tdm(
        KV, ptr, pt_rsrc, t2r_rsrc, rblk, col0, width, num_pages, num_requests, p
    ):
        # Called only for a full workgroup. Build all descriptor state from
        # wave-uniform SGPR metadata, never from lane-dependent consumer plans.
        wave_id = fx.Int32(fx.rocdl.wave_id())
        t0 = rblk * 16 + wave_id * WMMA_ROWS_PER_WAVE
        req = _ld_uniform_i32(t2r_rsrc, (t0 + p) * 4)
        req_ok = (req >= 0) & (req < num_requests)
        safe_req = req_ok.select(req, 0)
        for group in range_constexpr(tdm_cols // 8):
            local = group * 8
            if P == 4:
                row_ids = []
                for page in range_constexpr(2):
                    logical = (col0 + local) // 4 + page
                    logical_ok = logical < width
                    safe_logical = logical_ok.select(logical, 0)
                    phys = _ld_uniform_i32(
                        pt_rsrc, (safe_req * width + safe_logical) * 4
                    )
                    valid = req_ok & logical_ok & (phys >= 0) & (phys < num_pages)
                    phys = valid.select(phys, 0)
                    for row in range_constexpr(4):
                        row_ids.append(phys * 4 + row)
            elif P == 16:  # P = 16: one page lookup per 8-row group (col0 % 32 == 0)
                row_ids = _p16_row_ids(
                    pt_rsrc,
                    safe_req,
                    width,
                    num_pages,
                    col0 // 16 + local // 16,
                    local % 16,
                    req_ok,
                )
            else:  # P = 8: the 8-row group IS page col0/8 + group (one lookup)
                row_ids = _p16_row_ids(
                    pt_rsrc, safe_req, width, num_pages, col0 // 8 + group, 0, req_ok, 8
                )
            byte_off = wave_id * tdm_tile_bytes + local * TDM_K_ROW_BYTES
            _tdm_k_gather(KV, ptr, row_ids, num_pages, byte_off, P)

    def _pair_stage_raw(KV, ptr, pt_rsrc, t2r_rsrc, rblk, col0, width, num_pages):
        pass

    def _pair_stage_tdm(KV, ptr, pt_rsrc, t2r_rsrc, rblk, col0, width, num_pages):
        # Only the workgroup-uniform full single-request branch reaches here.
        # Each wave owns 16 rows (two 8-index TDM ops). Holes/tails clamp to
        # page 0; output masks stay unchanged. Random-index OOB is not used.
        wave_id = fx.Int32(fx.rocdl.wave_id())
        req = _ld_uniform_i32(t2r_rsrc, rblk * 32 * 4)
        for group in range_constexpr(2):
            local = wave_id * 16 + group * 8
            if P == 4:
                row_ids = []
                for page in range_constexpr(2):
                    logical = (col0 + local) // 4 + page
                    logical_ok = logical < width
                    safe_logical = logical_ok.select(logical, 0)
                    phys = _ld_uniform_i32(pt_rsrc, (req * width + safe_logical) * 4)
                    valid = logical_ok & (phys >= 0) & (phys < num_pages)
                    phys = valid.select(phys, 0)
                    for row in range_constexpr(4):
                        row_ids.append(phys * 4 + row)
            elif (
                P == 16
            ):  # P = 16: wave w's 16 rows = page col0/16 + w; group = half page
                row_ids = _p16_row_ids(
                    pt_rsrc, req, width, num_pages, col0 // 16 + wave_id, group * 8
                )
            else:  # P = 8: wave w's 16 rows = pages col0/8 + 2w, +1; group = one page
                row_ids = _p16_row_ids(
                    pt_rsrc,
                    req,
                    width,
                    num_pages,
                    col0 // 8 + wave_id * 2 + group,
                    0,
                    None,
                    8,
                )
            _tdm_k_gather(KV, ptr, row_ids, num_pages, local * TDM_K_ROW_BYTES, P)
        fx.rocdl.tdm_ops.tensor_wait(0)
        # Completion on every producer, then rendezvous before cross-wave LDS
        # consumption. No branch depending on per-wave visibility surrounds it.
        fx.gpu.barrier()

    def _pair_base_raw(phys, off, c, kgrp):
        return phys * page_bytes + off * key_row_bytes + kgrp * 16

    def _pair_base_tdm(phys, off, c, kgrp):
        # c lies in [col0, col0 + tdm_cols) with col0 % tdm_cols == 0 (tile row).
        return (c % tdm_cols) * TDM_K_ROW_BYTES + kgrp * 16

    def _pair_frag_raw(ptr, byte_off):
        return _frag(ptr, byte_off)

    def _pair_frag_tdm(ptr, byte_off):
        lo = Vec(
            fx.ptr_load(ptr + byte_off // 4, result_type=Vec.make_type(4, fx.Int32))
        )
        hi = Vec(
            fx.ptr_load(
                ptr + (byte_off + 32) // 4, result_type=Vec.make_type(4, fx.Int32)
            )
        )
        return lo.shuffle(hi, list(range(8)))

    # ---- Round PERSIST helpers (only the persistent kernel calls these) -----
    def _persist_storage_tdm():
        # One allocator, two 17,408 B padded K tiles (ping-pong) = 34,816 B.
        return (
            fx.SharedAllocator(base_alignment=128, static=True)
            .allocate(fx.Array[fx.Int32, TDM_PERSIST_LDS_BYTES // 4, 128])
            .peek()
            .ptr
        )

    def _persist_block_vis(cl_rsrc, req):
        """SGPR visibility bound of a full single-request block: max(ctx, 0) / 4.

        Every row's `_rows_meta` vis = max(min(pos + 1, ctx), 0) // 4 <= this
        bound, so a chunk with col0 >= bound stores only -inf masks and its K
        tile is never needed. (A 32-row max chain was lowered by LLVM to
        v_max3 + v_readfirstlane; this single scalar load keeps it SALU-only.)
        """
        ctx = _ld_uniform_i32(cl_rsrc, req * 4)
        return _imax(ctx, 0) // COMPRESS_RATIO

    def _persist_issue_tdm(KV, ptr, pt_rsrc, req, col0, width, num_pages, buf):
        """`_pair_stage_tdm`'s two 8-row gathers per wave, into buffer `buf`,
        WITHOUT wait/barrier (the persistent loop owns completion). All
        descriptor inputs are SGPR: wave_id, req, page-table s_buffer_loads."""
        wave_id = fx.Int32(fx.rocdl.wave_id())
        for group in range_constexpr(2):
            local = wave_id * 16 + group * 8
            if P == 4:
                row_ids = []
                for page in range_constexpr(2):
                    logical = (col0 + local) // 4 + page
                    logical_ok = logical < width
                    safe_logical = logical_ok.select(logical, 0)
                    phys = _ld_uniform_i32(pt_rsrc, (req * width + safe_logical) * 4)
                    valid = logical_ok & (phys >= 0) & (phys < num_pages)
                    phys = valid.select(phys, 0)
                    for row in range_constexpr(4):
                        row_ids.append(phys * 4 + row)
            elif (
                P == 16
            ):  # P = 16: wave w's 16 rows = page col0/16 + w; group = half page
                row_ids = _p16_row_ids(
                    pt_rsrc, req, width, num_pages, col0 // 16 + wave_id, group * 8
                )
            else:  # P = 8: wave w's 16 rows = pages col0/8 + 2w, +1; group = one page
                row_ids = _p16_row_ids(
                    pt_rsrc,
                    req,
                    width,
                    num_pages,
                    col0 // 8 + wave_id * 2 + group,
                    0,
                    None,
                    8,
                )
            _tdm_k_gather(
                KV,
                ptr,
                row_ids,
                num_pages,
                buf * TDM_K_TILE_BYTES + local * TDM_K_ROW_BYTES,
                P,
            )

    def _persist_next(k):
        """Trace-time stop of `range_constexpr(k + 1, stop)`: chunk k+1 if any."""
        return min(k + 2, persist_chunks)

    # ---- ROUND-R2B kstat helpers (only the kstat kernel calls these) -----------
    def _kstat_storage_tdm():
        # One allocator, col_tiles disjoint 17,408 B padded K tiles (read-only after
        # the first publish): 17,408 / 34,816 / 69,632 B.
        return (
            fx.SharedAllocator(base_alignment=128, static=True)
            .allocate(fx.Array[fx.Int32, col_tiles * TDM_K_TILE_BYTES // 4, 128])
            .peek()
            .ptr
        )

    def _kstat_slab_req(t2r_rsrc, cl_rsrc, row0, rows, num_requests):
        """SGPR (kreq, kreq_ok, safe kreq, vis bound) of the span's first row."""
        kreq = _ld_uniform_i32(t2r_rsrc, _imin(row0, rows - 1) * 4)
        kreq_ok = (kreq >= 0) & (kreq < num_requests)
        ksafe = kreq_ok.select(kreq, 0)
        return kreq, kreq_ok, ksafe, _persist_block_vis(cl_rsrc, ksafe)

    def _kstat_tile_ok(t2r_rsrc, base, rows, num_requests, kreq):
        """Uniform: full single-request 32-row block whose request is the slab's."""
        first = _ld_uniform_i32(t2r_rsrc, _imin(base, rows - 1) * 4)
        return _r32_guard(t2r_rsrc, base, rows, num_requests) & (first == kreq)

    # ---- ROUND-R2C kstat "early" (k_load='kstat', early=True) ------------------
    # The reuse branch's `_tile_pair` loaded each lane column's page-table entry AFTER
    # the TDM wait + barrier (one more dependent memory latency on the critical path:
    # its only consumer is page_ok). Inside that branch every row of the block is of
    # the slab's request (`_kstat_tile_ok`: r32 guard & first == kreq), so
    # ppa.safe_req == kreq and ppa.req_ok holds: page_ok depends on the slab only and
    # is loaded once per workgroup, before the wait, by the same expression with
    # (ksafe, kreq_ok). Scores / masks / store offsets are unchanged.
    def _kstat_page_pre(pt_rsrc, ksafe, kreq_ok, slab0, n_in, width, num_pages):
        out = []
        for ct in range_constexpr(col_tiles):
            oks = []
            for c in _lane_cols(slab0 + ct * WMMA_COLS, n_in):
                logical = c // P
                logical_ok = logical < width
                safe_logical = logical_ok.select(logical, 0)
                phys = _ld_i32(pt_rsrc, (ksafe * width + safe_logical) * 4)
                oks.append(kreq_ok & logical_ok & (phys >= 0) & (phys < num_pages))
            out.append(oks)
        return out

    def _kstat_page_none(pt_rsrc, ksafe, kreq_ok, slab0, n_in, width, num_pages):
        return []  # emits nothing (the b82ef05b kstat trace)

    def _tile_pair_pre(atom, kbuf, qfa, qfb, c, kgrp):
        """`_tile_pair` (TDM) without its page-table load: page_ok comes from the prefetch."""
        k_base = _pair_base_tdm(None, None, c, kgrp)
        acc_a = fx.make_rmem_tensor(8, fx.Float32)
        acc_b = fx.make_rmem_tensor(8, fx.Float32)
        acc_a.store(Vec.filled(8, 0.0, fx.Float32))
        acc_b.store(Vec.filled(8, 0.0, fx.Float32))
        for s in range_constexpr(WMMA_K_STEPS):
            kf = _rmem8(_pair_frag_tdm(kbuf, k_base + s * kstep_bytes))
            fx.gemm(atom, acc_a, _rmem8(qfa[s]), kf, acc_a)
            fx.gemm(atom, acc_b, _rmem8(qfb[s]), kf, acc_b)
        a = acc_a.load()
        b = acc_b.load()
        zero = fx.Float32(0.0)
        scores = []
        for acc in (a, b):
            sa = (
                fx.Float32(acc[0]).maximumf(zero) + fx.Float32(acc[1]).maximumf(zero)
            ) + (fx.Float32(acc[2]).maximumf(zero) + fx.Float32(acc[3]).maximumf(zero))
            sb = (
                fx.Float32(acc[4]).maximumf(zero) + fx.Float32(acc[5]).maximumf(zero)
            ) + (fx.Float32(acc[6]).maximumf(zero) + fx.Float32(acc[7]).maximumf(zero))
            scores.append(sa * inv_divisor)
            scores.append(sb * inv_divisor)
        return scores

    def _kstat_values_pre(
        atom,
        kbuf,
        pt_rsrc,
        qfa,
        qfb,
        ppa,
        ppb,
        cols,
        offsa,
        offsb,
        columns,
        width,
        num_pages,
        kpre,
        ct,
    ):
        out = []
        neg_inf = fx.Float32(_NEG_INF)
        for nt in range_constexpr(WMMA_N_TILES):
            c = cols[nt]
            page_ok = kpre[ct][nt]
            scores = _tile_pair_pre(atom, kbuf, qfa, qfb, c, ppa.kgrp)
            for mi in range_constexpr(2):
                pp = (ppa, ppb)[mi]
                offs = (offsa, offsb)[mi]
                col_ok = c < columns
                va = (page_ok & (c < pp.vis_a)).select(scores[2 * mi], neg_inf)
                vb = (page_ok & (c < pp.vis_b)).select(scores[2 * mi + 1], neg_inf)
                out.append((pp.own_a & col_ok, offs[2 * nt], va))
                out.append((pp.own_b & col_ok, offs[2 * nt + 1], vb))
        return out

    def _kstat_values_base(
        atom,
        kbuf,
        pt_rsrc,
        qfa,
        qfb,
        ppa,
        ppb,
        cols,
        offsa,
        offsb,
        columns,
        width,
        num_pages,
        kpre,
        ct,
    ):
        return _pair_values(
            atom,
            kbuf,
            pt_rsrc,
            qfa,
            qfb,
            ppa,
            ppb,
            cols,
            offsa,
            offsb,
            columns,
            width,
            num_pages,
        )

    kstat_pre = _kstat_page_pre if early else _kstat_page_none
    kstat_values = _kstat_values_pre if early else _kstat_values_base

    # ---- ROUND-R2C ktile helpers (only kernel_kt calls these; see KT_ROWS) -------
    def _kt_guard(t2r_rsrc, base, rows, num_requests):
        """SGPR (request, ok): full kt_rows-row block, every row of one valid request."""
        req = _ld_uniform_i32(t2r_rsrc, _imin(base, rows - 1) * 4)
        ok = (base + kt_rows <= rows) & (req >= 0) & (req < num_requests)
        for i in range_constexpr(1, kt_rows):
            other = _ld_uniform_i32(t2r_rsrc, _imin(base + i, rows - 1) * 4)
            ok = ok & (other == req)
        return req, ok

    def _kt_issue_w8(KV, ptr, pt_rsrc, req, col0, width, num_pages, buf):
        """8 waves: wave w gathers the 64-column tile's columns col0 + 8w .. 8w + 7 into
        buffer `buf` (one 8-row gather per wave; same row-id plan as every TDM site)."""
        wave_id = fx.Int32(fx.rocdl.wave_id())
        local = wave_id * 8
        if P == 4:
            row_ids = []
            for page in range_constexpr(2):
                logical = (col0 + local) // 4 + page
                logical_ok = logical < width
                safe_logical = logical_ok.select(logical, 0)
                phys = _ld_uniform_i32(pt_rsrc, (req * width + safe_logical) * 4)
                valid = logical_ok & (phys >= 0) & (phys < num_pages)
                phys = valid.select(phys, 0)
                for row in range_constexpr(4):
                    row_ids.append(phys * 4 + row)
        elif P == 16:  # wave w's 8 rows = half (w % 2) of page col0/16 + w/2
            row_ids = _p16_row_ids(
                pt_rsrc,
                req,
                width,
                num_pages,
                col0 // 16 + wave_id // 2,
                (wave_id % 2) * 8,
            )
        else:  # P = 8: wave w's 8 rows = page col0/8 + w
            row_ids = _p16_row_ids(
                pt_rsrc, req, width, num_pages, col0 // 8 + wave_id, 0, None, 8
            )
        _tdm_k_gather(
            KV,
            ptr,
            row_ids,
            num_pages,
            buf * TDM_K_TILE_BYTES + local * TDM_K_ROW_BYTES,
            P,
        )

    kt_issue = _kt_issue_w8 if kt_waves == 8 else _persist_issue_tdm

    def _kt_tile(atom, kbuf, qf, c, kgrp):
        """One M16 tile x column c from the LDS slab: the lane's rows a/b scores."""
        k_base = _pair_base_tdm(None, None, c, kgrp)
        acc_t = fx.make_rmem_tensor(8, fx.Float32)
        acc_t.store(Vec.filled(8, 0.0, fx.Float32))
        for s in range_constexpr(WMMA_K_STEPS):
            kf = _rmem8(_pair_frag_tdm(kbuf, k_base + s * kstep_bytes))
            fx.gemm(atom, acc_t, _rmem8(qf[s]), kf, acc_t)
        acc = acc_t.load()
        zero = fx.Float32(0.0)
        sa = (fx.Float32(acc[0]).maximumf(zero) + fx.Float32(acc[1]).maximumf(zero)) + (
            fx.Float32(acc[2]).maximumf(zero) + fx.Float32(acc[3]).maximumf(zero)
        )
        sb = (fx.Float32(acc[4]).maximumf(zero) + fx.Float32(acc[5]).maximumf(zero)) + (
            fx.Float32(acc[6]).maximumf(zero) + fx.Float32(acc[7]).maximumf(zero)
        )
        return sa * inv_divisor, sb * inv_divisor

    def _kt_values(atom, kbuf, qf, pp, cols, offs, columns, oks):
        """`_pass_values` (staged) with prefetched page validity `oks` per N tile."""
        neg_inf = fx.Float32(_NEG_INF)
        out = []
        for nt in range_constexpr(len(cols)):
            c = cols[nt]
            page_ok = oks[nt]
            sa, sb = _kt_tile(atom, kbuf, qf, c, pp.kgrp)
            col_ok = c < columns
            va = (page_ok & (c < pp.vis_a)).select(sa, neg_inf)
            vb = (page_ok & (c < pp.vis_b)).select(sb, neg_inf)
            out.append((pp.own_a & col_ok, offs[2 * nt], va))
            out.append((pp.own_b & col_ok, offs[2 * nt + 1], vb))
        return out

    # Round DRAFT/NOBAR (QSA_R16_NOBAR=1, default since ROUND-MERGE1; TDM r16 only): each wave
    # gathers into and reads only its own tile (byte offset wave_id * tile), so
    # the per-wave TENSORcnt wait already publishes the data to its consumer,
    # and an s_wait_dscnt 0 (this wave's LDS reads retired) is the WAR guard
    # before the next pass's gather overwrites the tile. QSA_R16_NOBAR=0 (and the
    # persistent / r32 / raw kernels): two WG barriers.
    def _r16_barrier():
        fx.gpu.barrier()

    def _r16_nop():
        pass

    def _r16_dscnt0():
        fx.rocdl.s_wait_dscnt(0)

    # globals().get: the stock CPU gates exec this builder inside reduced AST
    # namespaces that predate the knob; absent there means the barriered kernel
    # (still exact; the module default TDM_R16_NOBAR=True applies to real builds).
    r16_nobar = (
        globals().get("TDM_R16_NOBAR", False)
        and use_tdm
        and rows_per_block == 16
        and not persist
    )
    r16_publish = _r16_nop if r16_nobar else _r16_barrier
    r16_reuse = _r16_dscnt0 if r16_nobar else _r16_barrier
    pair_storage = _pair_storage_tdm if use_tdm else _pair_storage_raw
    pair_stage = _pair_stage_tdm if use_tdm else _pair_stage_raw
    pair_base = _pair_base_tdm if use_tdm else _pair_base_raw
    pair_frag = _pair_frag_tdm if use_tdm else _pair_frag_raw
    guard_load = _ld_uniform_i32 if use_tdm else _ld_i32
    q_row_bytes = NUM_HEADS * HEAD_DIM * 2  # 1024 B
    head_bytes = HEAD_DIM * 2  # 256 B
    key_row_bytes = HEAD_DIM * 2  # 256 B
    page_bytes = P * key_row_bytes  # 1024 B
    kstep_bytes = WMMA_K * 2  # 64 B of one row per WMMA step
    kernel_name = (
        f"qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma"
        f"_D{HEAD_DIM}_H{NUM_HEADS}_P{P}"
        f"_r{rows_per_block}_c{cols_per_block}_flydsl" + ("_tdm" if use_tdm else "")
    )
    if persist:
        kernel_name = kernel_name + f"_persist{persist_chunks}"
        if fb_loop:
            kernel_name = kernel_name + "_fbl"
    if kstat:
        kernel_name = (
            kernel_name + f"_kstat{row_tiles}x{col_tiles}" + ("e" if early else "")
        )
    if ktile:
        kernel_name = kernel_name + f"_kt{kt_rows}x{col_tiles}"
        kernel_name = kernel_name + _r2d_tag(lean, ipf)
    if trace:
        kernel_name = kernel_name + f"_trace{int(trace)}"
    if r16_nobar:
        kernel_name = kernel_name + "_nobar"
    kernel_name = kernel_name + qp_tag

    # ---- helpers: plain Python at trace time (no `if`/runtime `for` inside; the
    # AST rewriter only rewrites the kernel body, and the kernel body only calls
    # these through bare names so no outer value becomes loop/if-carried) -----
    def _i01(cond):
        return cond.select(fx.Int32(1), fx.Int32(0))

    def _frag(rsrc, byte_off):
        """One WMMA operand fragment: 2 x 16 B, 32 B apart -> 8 x i32 (16 bf16)."""
        lo = _ld_vec(rsrc, byte_off, 4, fx.Int32)
        hi = _ld_vec(rsrc, byte_off + 32, 4, fx.Int32)
        return lo.shuffle(hi, list(range(8)))

    def _rmem8(vec):
        t = fx.make_rmem_tensor(8, fx.Int32)
        t.store(vec)
        return t

    def _rows_meta(t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests):
        out = []
        for i in range_constexpr(WMMA_ROWS_PER_WAVE):
            row = t0 + i
            row_ok = row < rows
            safe_row = _imin(row, rows - 1)  # loads stay in bounds
            req = _ld_i32(t2r_rsrc, safe_row * 4)
            req_ok = (req >= 0) & (req < num_requests)
            safe_req = req_ok.select(req, 0)
            ctx = _ld_i32(cl_rsrc, safe_req * 4)
            pos = ld_pos(qp_rsrc, safe_row)
            head = _imax(_imin(pos + 1, ctx), 0)
            vis = req_ok.select(head // COMPRESS_RATIO, 0)
            out.append(_RowMeta(row, row_ok, req, req_ok, safe_req, vis))
        return out

    def _pass_plan(meta, rows, kgrp):
        """Request-dedup passes; lane rows are a = 2*kgrp, b = 2*kgrp + 1."""
        g0 = kgrp == 0
        plans = []
        for p in range_constexpr(WMMA_PASSES):
            mp = meta[p]
            active = mp.row_ok
            for q in range_constexpr(p):
                mq = meta[q]
                active = active & ((mq.row >= rows) | (mq.req != mp.req))
            own = [
                _i01(meta[i].row_ok & (meta[i].req == mp.req))
                for i in range_constexpr(WMMA_ROWS_PER_WAVE)
            ]
            vmax = fx.Int32(0)
            for i in range_constexpr(WMMA_ROWS_PER_WAVE):
                vmax = _imax(vmax, (own[i] != 0).select(meta[i].vis, 0))
            own_a = g0.select(own[0], own[2]) != 0
            own_b = g0.select(own[1], own[3]) != 0
            vis_a = g0.select(meta[0].vis, meta[2].vis)
            vis_b = g0.select(meta[1].vis, meta[3].vis)
            plans.append(
                _Pass(
                    active,
                    vmax,
                    mp.safe_req,
                    mp.req_ok,
                    own_a,
                    own_b,
                    vis_a,
                    vis_b,
                    kgrp,
                )
            )
        return plans

    def _pick_vis(lane, meta):
        """visible[t0 + lane] for lanes 0..3 (lanes >= 3 get row 3's; unused)."""
        v = meta[WMMA_ROWS_PER_WAVE - 1].vis
        for i in range_constexpr(WMMA_ROWS_PER_WAVE - 2, -1, -1):
            v = (lane == i).select(meta[i].vis, v)
        return v

    def _lane_cols(col0, n_in, n_tiles=WMMA_N_TILES):
        return [col0 + nt * WMMA_N + n_in for nt in range_constexpr(n_tiles)]

    def _store_offsets(t0, kgrp, cols, rows, columns):
        """Clamped byte offsets of the lane's two outputs per tile (always in plane)."""
        last = rows * columns - 1
        row_a = _imin(t0 + 2 * kgrp, rows - 1)
        row_b = _imin(t0 + 2 * kgrp + 1, rows - 1)
        out = []
        for c in cols:
            out.append(_imin(row_a * columns + c, last) * 4)
            out.append(_imin(row_b * columns + c, last) * 4)
        return out

    def _q_frags(q_rsrc, t0, rows, n_in, kgrp):
        q_row = _imin(t0 + n_in // NUM_HEADS, rows - 1)
        head = n_in % NUM_HEADS
        base = q_row * q_row_bytes + head * head_bytes + kgrp * 16
        return [
            _frag(q_rsrc, base + s * kstep_bytes) for s in range_constexpr(WMMA_K_STEPS)
        ]

    def _tile(atom, kv_rsrc, pt_rsrc, qf, pp, c, kgrp, width, num_pages, staged=False):
        """Scores of the lane's rows a/b; staged is a trace-time Python bool."""
        logical = c // P
        off = c % P
        logical_ok = logical < width
        safe_logical = logical_ok.select(logical, 0)
        phys = _ld_i32(pt_rsrc, (pp.safe_req * width + safe_logical) * 4)
        page_ok = pp.req_ok & logical_ok & (phys >= 0) & (phys < num_pages)
        phys = page_ok.select(phys, 0)
        base_fn = _pair_base_tdm if staged else _pair_base_raw
        frag_fn = _pair_frag_tdm if staged else _frag
        k_base = base_fn(phys, off, c, kgrp)
        kf = [
            frag_fn(kv_rsrc, k_base + s * kstep_bytes)
            for s in range_constexpr(WMMA_K_STEPS)
        ]
        acc_t = fx.make_rmem_tensor(8, fx.Float32)
        acc_t.store(Vec.filled(8, 0.0, fx.Float32))
        for s in range_constexpr(WMMA_K_STEPS):
            # D[m = 4*row + head, n = column] += Q[m, k] * K[n, k]
            fx.gemm(atom, acc_t, _rmem8(qf[s]), _rmem8(kf[s]), acc_t)
        acc = acc_t.load()
        zero = fx.Float32(0.0)
        sa = (fx.Float32(acc[0]).maximumf(zero) + fx.Float32(acc[1]).maximumf(zero)) + (
            fx.Float32(acc[2]).maximumf(zero) + fx.Float32(acc[3]).maximumf(zero)
        )
        sb = (fx.Float32(acc[4]).maximumf(zero) + fx.Float32(acc[5]).maximumf(zero)) + (
            fx.Float32(acc[6]).maximumf(zero) + fx.Float32(acc[7]).maximumf(zero)
        )
        return page_ok, sa * inv_divisor, sb * inv_divisor

    def _pass_values(
        atom,
        kv_rsrc,
        pt_rsrc,
        qf,
        pp,
        cols,
        offs,
        columns,
        width,
        num_pages,
        staged=False,
    ):
        """[(store_ok, byte_off, value)]; staged selects only the K-load layout."""
        neg_inf = fx.Float32(_NEG_INF)
        out = []
        for nt in range_constexpr(len(cols)):
            c = cols[nt]
            page_ok, sa, sb = _tile(
                atom, kv_rsrc, pt_rsrc, qf, pp, c, pp.kgrp, width, num_pages, staged
            )
            col_ok = c < columns
            va = (page_ok & (c < pp.vis_a)).select(sa, neg_inf)
            vb = (page_ok & (c < pp.vis_b)).select(sb, neg_inf)
            out.append((pp.own_a & col_ok, offs[2 * nt], va))
            out.append((pp.own_b & col_ok, offs[2 * nt + 1], vb))
        return out

    def _pass_masks(pp, cols, offs, columns):
        """[(store_ok, byte_off)] for a pass whose rows see none of these columns."""
        out = []
        for nt in range_constexpr(len(cols)):
            col_ok = cols[nt] < columns
            out.append((pp.own_a & col_ok, offs[2 * nt]))
            out.append((pp.own_b & col_ok, offs[2 * nt + 1]))
        return out

    def _r32_guard(t2r_rsrc, base, rows, num_requests):
        """Uniform guard: full 32-row block, all rows of one valid request."""
        req = guard_load(t2r_rsrc, _imin(base, rows - 1) * 4)
        reuse = (base + 32 <= rows) & (req >= 0) & (req < num_requests)
        for i in range_constexpr(1, 32):
            other = guard_load(t2r_rsrc, _imin(base + i, rows - 1) * 4)
            reuse = reuse & (other == req)
        return reuse

    def _r64_guard(t2r_rsrc, base, rows, num_requests):
        """Uniform guard: full 64-row block, all rows of one valid request."""
        req = guard_load(t2r_rsrc, _imin(base, rows - 1) * 4)
        reuse = (base + 64 <= rows) & (req >= 0) & (req < num_requests)
        for i in range_constexpr(1, 64):
            other = guard_load(t2r_rsrc, _imin(base + i, rows - 1) * 4)
            reuse = reuse & (other == req)
        return reuse

    def _tile_pair(atom, kv_rsrc, pt_rsrc, qfa, qfb, pp, c, kgrp, width, num_pages):
        """Two independent M16 accumulators, one B gather per K step."""
        logical = c // P
        off = c % P
        logical_ok = logical < width
        safe_logical = logical_ok.select(logical, 0)
        phys = _ld_i32(pt_rsrc, (pp.safe_req * width + safe_logical) * 4)
        page_ok = pp.req_ok & logical_ok & (phys >= 0) & (phys < num_pages)
        phys = page_ok.select(phys, 0)
        k_base = pair_base(phys, off, c, kgrp)
        acc_a = fx.make_rmem_tensor(8, fx.Float32)
        acc_b = fx.make_rmem_tensor(8, fx.Float32)
        acc_a.store(Vec.filled(8, 0.0, fx.Float32))
        acc_b.store(Vec.filled(8, 0.0, fx.Float32))
        for s in range_constexpr(WMMA_K_STEPS):
            kf = _rmem8(pair_frag(kv_rsrc, k_base + s * kstep_bytes))
            fx.gemm(atom, acc_a, _rmem8(qfa[s]), kf, acc_a)
            fx.gemm(atom, acc_b, _rmem8(qfb[s]), kf, acc_b)
        a = acc_a.load()
        b = acc_b.load()
        zero = fx.Float32(0.0)
        scores = []
        for acc in (a, b):
            sa = (
                fx.Float32(acc[0]).maximumf(zero) + fx.Float32(acc[1]).maximumf(zero)
            ) + (fx.Float32(acc[2]).maximumf(zero) + fx.Float32(acc[3]).maximumf(zero))
            sb = (
                fx.Float32(acc[4]).maximumf(zero) + fx.Float32(acc[5]).maximumf(zero)
            ) + (fx.Float32(acc[6]).maximumf(zero) + fx.Float32(acc[7]).maximumf(zero))
            scores.append(sa * inv_divisor)
            scores.append(sb * inv_divisor)
        return page_ok, scores

    def _pair_values(
        atom,
        kv_rsrc,
        pt_rsrc,
        qfa,
        qfb,
        ppa,
        ppb,
        cols,
        offsa,
        offsb,
        columns,
        width,
        num_pages,
    ):
        out = []
        neg_inf = fx.Float32(_NEG_INF)
        for nt in range_constexpr(WMMA_N_TILES):
            c = cols[nt]
            page_ok, scores = _tile_pair(
                atom, kv_rsrc, pt_rsrc, qfa, qfb, ppa, c, ppa.kgrp, width, num_pages
            )
            for mi in range_constexpr(2):
                pp = (ppa, ppb)[mi]
                offs = (offsa, offsb)[mi]
                col_ok = c < columns
                va = (page_ok & (c < pp.vis_a)).select(scores[2 * mi], neg_inf)
                vb = (page_ok & (c < pp.vis_b)).select(scores[2 * mi + 1], neg_inf)
                out.append((pp.own_a & col_ok, offs[2 * nt], va))
                out.append((pp.own_b & col_ok, offs[2 * nt + 1], vb))
        return out

    @flyc.kernel(name=kernel_name, known_block_size=[WMMA_THREADS, 1, 1])
    def kernel(
        Q,  # uint8 view of [rows, 4, 128] bf16
        KV,  # uint8 view of [pages, 4, 1, 128] bf16
        PT,  # int32 [requests, width]
        T2R,  # int32 [rows]
        QP,  # int32 [rows]
        CL,  # int32 [requests]
        LOGITS,  # fp32 [rows, columns]
        VIS,  # int32 [rows]
        RST,  # int32 [rows]
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        chunk = fx.Int32(fx.block_idx.x)
        rblk = fx.Int32(fx.block_idx.y)
        wave = tid // WMMA_WAVE
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N  # A: slot m / B, C: column n
        kgrp = lane // WMMA_N  # k-half of the operands, row pair of C
        t0 = (rblk * WMMA_WAVES + wave) * WMMA_ROWS_PER_WAVE
        col0 = chunk * WMMA_COLS

        # One pinned descriptor per plane, true extents in i64 (see `_plane`).
        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))

        meta = _rows_meta(t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests)
        plans = _pass_plan(meta, rows, kgrp)

        # Row metadata: lanes 0..3 of each wave of column-block 0, one row each.
        # Predicated store at a clamped in-plane offset (6c272c3c contract).
        my_vis = _pick_vis(lane, meta)
        my_row = t0 + _imin(lane, WMMA_ROWS_PER_WAVE - 1)
        meta_off = _imin(my_row, rows - 1) * 4
        writer = (chunk == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)
        if writer:
            _st_i32(vis_rsrc, meta_off, my_vis)
            _st_i32(rst_rsrc, meta_off, fx.Int32(0))

        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        cols = _lane_cols(col0, n_in)
        offs = _store_offsets(t0, kgrp, cols, rows, columns)
        neg_inf = fx.Float32(_NEG_INF)
        live = t0 < rows
        if live:
            qf = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
            for p in range_constexpr(WMMA_PASSES):
                pp = plans[p]
                if pp.active:
                    if col0 < pp.vmax:
                        res = _pass_values(
                            atom,
                            kv_rsrc,
                            pt_rsrc,
                            qf,
                            pp,
                            cols,
                            offs,
                            columns,
                            width,
                            num_pages,
                        )
                        for j1 in range_constexpr(2 * WMMA_N_TILES):
                            if res[j1][0]:
                                _st_f32(logits_rsrc, res[j1][1], res[j1][2])
                    if col0 >= pp.vmax:
                        msk = _pass_masks(pp, cols, offs, columns)
                        for j2 in range_constexpr(2 * WMMA_N_TILES):
                            if msk[j2][0]:
                                _st_f32(logits_rsrc, msk[j2][1], neg_inf)

    @flyc.kernel(name=kernel_name, known_block_size=[WMMA_THREADS, 1, 1])
    def kernel_r32(
        Q,  # uint8 view of [rows, 4, 128] bf16
        KV,  # uint8 view of [pages, 4, 1, 128] bf16
        PT,  # int32 [requests, width]
        T2R,  # int32 [rows]
        QP,  # int32 [rows]
        CL,  # int32 [requests]
        LOGITS,  # fp32 [rows, columns]
        VIS,  # int32 [rows]
        RST,  # int32 [rows]
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        chunk = fx.Int32(fx.block_idx.x)
        rblk = fx.Int32(fx.block_idx.y)
        wave = tid // WMMA_WAVE
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N  # A: slot m / B, C: column n
        kgrp = lane // WMMA_N  # k-half of the operands, row pair of C
        t0 = (rblk * WMMA_WAVES + wave) * (2 * WMMA_ROWS_PER_WAVE)
        col0 = chunk * WMMA_COLS
        _tr_begin(
            trs,
            trace,
            LOGITS,
            rows,
            columns,
            (columns + WMMA_COLS - 1) // WMMA_COLS,
            (rows + 31) // 32,
            WMMA_WAVES,
            chunk,
            rblk,
        )

        # One pinned descriptor per plane, true extents in i64 (see `_plane`).
        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))

        metas = [
            _rows_meta(
                t2r_rsrc,
                qp_rsrc,
                cl_rsrc,
                t0 + mi * WMMA_ROWS_PER_WAVE,
                rows,
                num_requests,
            )
            for mi in range_constexpr(2)
        ]
        plans = [_pass_plan(metas[mi], rows, kgrp) for mi in range_constexpr(2)]
        for mi in range_constexpr(2):
            my_vis = _pick_vis(lane, metas[mi])
            my_row = t0 + mi * WMMA_ROWS_PER_WAVE + _imin(lane, WMMA_ROWS_PER_WAVE - 1)
            meta_off = _imin(my_row, rows - 1) * 4
            writer = (chunk == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)
            if writer:
                _st_i32(vis_rsrc, meta_off, my_vis)
                _st_i32(rst_rsrc, meta_off, fx.Int32(0))
        _tr_mark(trs, 4, True)

        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        cols = _lane_cols(col0, n_in)
        offs = [
            _store_offsets(t0 + mi * WMMA_ROWS_PER_WAVE, kgrp, cols, rows, columns)
            for mi in range_constexpr(2)
        ]
        neg_inf = fx.Float32(_NEG_INF)
        kptr = pair_storage(KV, kv_rsrc)
        reuse = _r32_guard(t2r_rsrc, rblk * 32, rows, num_requests)
        _tr_mark(trs, 5, True)
        if reuse:
            pair_stage(KV, kptr, pt_rsrc, t2r_rsrc, rblk, col0, width, num_pages)
            _tr_mark(trs, 9)  # ROUND-R2D: stage = TDM issue + wait + barrier
            _tr_mark_rt(trs, 14)
            ppa = plans[0][0]
            ppb = plans[1][0]
            vmax = _imax(ppa.vmax, ppb.vmax)
            if col0 < vmax:
                qfa = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
                qfb = _q_frags(q_rsrc, t0 + WMMA_ROWS_PER_WAVE, rows, n_in, kgrp)
                _tr_mark(trs, 7, True)
                res = _pair_values(
                    atom,
                    kptr,
                    pt_rsrc,
                    qfa,
                    qfb,
                    ppa,
                    ppb,
                    cols,
                    offs[0],
                    offs[1],
                    columns,
                    width,
                    num_pages,
                )
                for j1 in range_constexpr(4 * WMMA_N_TILES):
                    if res[j1][0]:
                        _st_f32(logits_rsrc, res[j1][1], res[j1][2])
            if col0 >= vmax:
                for mi in range_constexpr(2):
                    msk = _pass_masks(plans[mi][0], cols, offs[mi], columns)
                    for j2 in range_constexpr(2 * WMMA_N_TILES):
                        if msk[j2][0]:
                            _st_f32(logits_rsrc, msk[j2][1], neg_inf)
            _tr_mark(trs, 10)
        if reuse == 0:
            # Exact r16 per-M-tile path for mixed requests and incomplete blocks.
            for mi in range_constexpr(2):
                subt0 = t0 + mi * WMMA_ROWS_PER_WAVE
                live = subt0 < rows
                if live:
                    qf = _q_frags(q_rsrc, subt0, rows, n_in, kgrp)
                    for p in range_constexpr(WMMA_PASSES):
                        pp = plans[mi][p]
                        if pp.active:
                            if col0 < pp.vmax:
                                fallback = _pass_values(
                                    atom,
                                    kv_rsrc,
                                    pt_rsrc,
                                    qf,
                                    pp,
                                    cols,
                                    offs[mi],
                                    columns,
                                    width,
                                    num_pages,
                                )
                                for j3 in range_constexpr(2 * WMMA_N_TILES):
                                    if fallback[j3][0]:
                                        _st_f32(
                                            logits_rsrc,
                                            fallback[j3][1],
                                            fallback[j3][2],
                                        )
                            if col0 >= pp.vmax:
                                masks = _pass_masks(pp, cols, offs[mi], columns)
                                for j4 in range_constexpr(2 * WMMA_N_TILES):
                                    if masks[j4][0]:
                                        _st_f32(logits_rsrc, masks[j4][1], neg_inf)
        _tr_mark(trs, 10)
        _tr_end(trs)

    @flyc.kernel(name=kernel_name, known_block_size=[WMMA_THREADS, 1, 1])
    def kernel_r64(
        Q,  # uint8 view of [rows, 4, 128] bf16
        KV,  # uint8 view of [pages, 4, 1, 128] bf16
        PT,  # int32 [requests, width]
        T2R,  # int32 [rows]
        QP,  # int32 [rows]
        CL,  # int32 [requests]
        LOGITS,  # fp32 [rows, columns]
        VIS,  # int32 [rows]
        RST,  # int32 [rows]
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        chunk = fx.Int32(fx.block_idx.x)
        rblk = fx.Int32(fx.block_idx.y)
        wave = tid // WMMA_WAVE
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N  # A: slot m / B, C: column n
        kgrp = lane // WMMA_N  # k-half of the operands, row pair of C
        t0 = (rblk * WMMA_WAVES + wave) * (4 * WMMA_ROWS_PER_WAVE)
        col0 = chunk * WMMA_COLS

        # One pinned descriptor per plane, true extents in i64 (see `_plane`).
        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))

        metas = [
            _rows_meta(
                t2r_rsrc,
                qp_rsrc,
                cl_rsrc,
                t0 + mi * WMMA_ROWS_PER_WAVE,
                rows,
                num_requests,
            )
            for mi in range_constexpr(4)
        ]
        plans = [_pass_plan(metas[mi], rows, kgrp) for mi in range_constexpr(4)]
        for mi in range_constexpr(4):
            my_vis = _pick_vis(lane, metas[mi])
            my_row = t0 + mi * WMMA_ROWS_PER_WAVE + _imin(lane, WMMA_ROWS_PER_WAVE - 1)
            meta_off = _imin(my_row, rows - 1) * 4
            writer = (chunk == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)
            if writer:
                _st_i32(vis_rsrc, meta_off, my_vis)
                _st_i32(rst_rsrc, meta_off, fx.Int32(0))

        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        cols = _lane_cols(col0, n_in)
        offs = [
            _store_offsets(t0 + mi * WMMA_ROWS_PER_WAVE, kgrp, cols, rows, columns)
            for mi in range_constexpr(4)
        ]
        neg_inf = fx.Float32(_NEG_INF)
        kptr = pair_storage(KV, kv_rsrc)
        reuse = _r64_guard(t2r_rsrc, rblk * 64, rows, num_requests)
        if reuse:
            pair_stage(KV, kptr, pt_rsrc, t2r_rsrc, rblk * 2, col0, width, num_pages)
            # Four M16 tiles per wave = 16 query rows x four heads. Keep only
            # two Q/accumulator tiles live at once; both pairs share the same K.
            for pair_group in range_constexpr(2):
                ma = pair_group * 2
                mb = ma + 1
                ppa = plans[ma][0]
                ppb = plans[mb][0]
                vmax = _imax(ppa.vmax, ppb.vmax)
                if col0 < vmax:
                    qfa = _q_frags(
                        q_rsrc, t0 + ma * WMMA_ROWS_PER_WAVE, rows, n_in, kgrp
                    )
                    qfb = _q_frags(
                        q_rsrc, t0 + mb * WMMA_ROWS_PER_WAVE, rows, n_in, kgrp
                    )
                    res = _pair_values(
                        atom,
                        kptr,
                        pt_rsrc,
                        qfa,
                        qfb,
                        ppa,
                        ppb,
                        cols,
                        offs[ma],
                        offs[mb],
                        columns,
                        width,
                        num_pages,
                    )
                    for j1 in range_constexpr(4 * WMMA_N_TILES):
                        if res[j1][0]:
                            _st_f32(logits_rsrc, res[j1][1], res[j1][2])
                if col0 >= vmax:
                    for pi in range_constexpr(2):
                        mi = ma + pi
                        msk = _pass_masks(plans[mi][0], cols, offs[mi], columns)
                        for j2 in range_constexpr(2 * WMMA_N_TILES):
                            if msk[j2][0]:
                                _st_f32(logits_rsrc, msk[j2][1], neg_inf)
        if reuse == 0:
            # Exact r16 per-M-tile path for mixed requests and incomplete blocks.
            for mi in range_constexpr(4):
                subt0 = t0 + mi * WMMA_ROWS_PER_WAVE
                live = subt0 < rows
                if live:
                    qf = _q_frags(q_rsrc, subt0, rows, n_in, kgrp)
                    for p in range_constexpr(WMMA_PASSES):
                        pp = plans[mi][p]
                        if pp.active:
                            if col0 < pp.vmax:
                                fallback = _pass_values(
                                    atom,
                                    kv_rsrc,
                                    pt_rsrc,
                                    qf,
                                    pp,
                                    cols,
                                    offs[mi],
                                    columns,
                                    width,
                                    num_pages,
                                )
                                for j3 in range_constexpr(2 * WMMA_N_TILES):
                                    if fallback[j3][0]:
                                        _st_f32(
                                            logits_rsrc,
                                            fallback[j3][1],
                                            fallback[j3][2],
                                        )
                            if col0 >= pp.vmax:
                                masks = _pass_masks(pp, cols, offs[mi], columns)
                                for j4 in range_constexpr(2 * WMMA_N_TILES):
                                    if masks[j4][0]:
                                        _st_f32(logits_rsrc, masks[j4][1], neg_inf)

    @flyc.kernel(name=kernel_name, known_block_size=[WMMA_THREADS, 1, 1])
    def kernel_tdm_r16(
        Q,
        KV,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        chunk = fx.Int32(fx.block_idx.x)
        rblk = fx.Int32(fx.block_idx.y)
        wave = fx.Int32(fx.rocdl.wave_id())
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N
        kgrp = lane // WMMA_N
        t0 = (rblk * WMMA_WAVES + wave) * WMMA_ROWS_PER_WAVE
        col0 = chunk * tdm_cols
        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))
        meta = _rows_meta_uniform(t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests)
        plans = _pass_plan(meta, rows, kgrp)
        my_vis = _pick_vis(lane, meta)
        my_row = t0 + _imin(lane, WMMA_ROWS_PER_WAVE - 1)
        meta_off = _imin(my_row, rows - 1) * 4
        writer = (chunk == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)
        if writer:
            _st_i32(vis_rsrc, meta_off, my_vis)
            _st_i32(rst_rsrc, meta_off, fx.Int32(0))
        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        cols = _lane_cols(col0, n_in, tdm_cols // WMMA_N)
        offs = _store_offsets(t0, kgrp, cols, rows, columns)
        neg_inf = fx.Float32(_NEG_INF)
        storage = _r16_storage_tdm()
        kptr = storage + wave * (tdm_tile_bytes // 4)
        full = rblk * 16 + 16 <= rows
        if full:
            qf = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
            for p in range_constexpr(WMMA_PASSES):
                # Only a scalar-active wave issues. All waves still wait and
                # rendezvous, including duplicate/invisible/invalid passes.
                # Hardware throttles 3/wave XACK; completion capacity is 63.
                stage_active = _r16_stage_active(
                    t2r_rsrc, qp_rsrc, cl_rsrc, rblk, col0, num_requests, p
                )
                if stage_active:
                    _r16_stage_tdm(
                        KV,
                        storage,
                        pt_rsrc,
                        t2r_rsrc,
                        rblk,
                        col0,
                        width,
                        num_pages,
                        num_requests,
                        p,
                    )
                fx.rocdl.tdm_ops.tensor_wait(0)
                fx.gpu.barrier()
                pp = plans[p]
                if pp.active:
                    if col0 < pp.vmax:
                        res = _pass_values(
                            atom,
                            kptr,
                            pt_rsrc,
                            qf,
                            pp,
                            cols,
                            offs,
                            columns,
                            width,
                            num_pages,
                            True,
                        )
                        for j1 in range_constexpr(len(res)):
                            if res[j1][0]:
                                _st_f32(logits_rsrc, res[j1][1], res[j1][2])
                    if col0 >= pp.vmax:
                        msk = _pass_masks(pp, cols, offs, columns)
                        for j2 in range_constexpr(len(msk)):
                            if msk[j2][0]:
                                _st_f32(logits_rsrc, msk[j2][1], neg_inf)
                # All prior LDS reads finish before the next pass overwrites it.
                # This barrier is outside every per-wave request/visibility guard.
                fx.gpu.barrier()
        if full == 0:
            live = t0 < rows
            if live:
                qf = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
                for p in range_constexpr(WMMA_PASSES):
                    pp = plans[p]
                    if pp.active:
                        if col0 < pp.vmax:
                            res = _pass_values(
                                atom,
                                kv_rsrc,
                                pt_rsrc,
                                qf,
                                pp,
                                cols,
                                offs,
                                columns,
                                width,
                                num_pages,
                            )
                            for j3 in range_constexpr(len(res)):
                                if res[j3][0]:
                                    _st_f32(logits_rsrc, res[j3][1], res[j3][2])
                        if col0 >= pp.vmax:
                            msk = _pass_masks(pp, cols, offs, columns)
                            for j4 in range_constexpr(len(msk)):
                                if msk[j4][0]:
                                    _st_f32(logits_rsrc, msk[j4][1], neg_inf)

    @flyc.kernel(name=kernel_name, known_block_size=[WMMA_THREADS, 1, 1])
    def kernel_r32_persist(
        Q,
        KV,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        # Round PERSIST: block (x, y) owns rows [32y, 32y + 32) and the
        # persist_chunks 64-column tiles starting at span0 = x * 64 * chunks.
        tid = fx.Int32(fx.thread_idx.x)
        chunk = fx.Int32(fx.block_idx.x)
        rblk = fx.Int32(fx.block_idx.y)
        wave = tid // WMMA_WAVE
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N  # A: slot m / B, C: column n
        kgrp = lane // WMMA_N  # k-half of the operands, row pair of C
        t0 = (rblk * WMMA_WAVES + wave) * (2 * WMMA_ROWS_PER_WAVE)
        span0 = chunk * (WMMA_COLS * persist_chunks)

        # One pinned descriptor per plane, true extents in i64 (see `_plane`).
        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))

        metas = [
            _rows_meta(
                t2r_rsrc,
                qp_rsrc,
                cl_rsrc,
                t0 + mi * WMMA_ROWS_PER_WAVE,
                rows,
                num_requests,
            )
            for mi in range_constexpr(2)
        ]
        plans = [_pass_plan(metas[mi], rows, kgrp) for mi in range_constexpr(2)]
        for mi in range_constexpr(2):
            my_vis = _pick_vis(lane, metas[mi])
            my_row = t0 + mi * WMMA_ROWS_PER_WAVE + _imin(lane, WMMA_ROWS_PER_WAVE - 1)
            meta_off = _imin(my_row, rows - 1) * 4
            writer = (chunk == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)
            if writer:
                _st_i32(vis_rsrc, meta_off, my_vis)
                _st_i32(rst_rsrc, meta_off, fx.Int32(0))

        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        neg_inf = fx.Float32(_NEG_INF)
        kbase = _persist_storage_tdm()
        reuse = _r32_guard(t2r_rsrc, rblk * 32, rows, num_requests)
        if reuse:
            # Workgroup-uniform (SGPR) request, block visibility and per-chunk
            # issue guards; the guards never surround a wait or a barrier.
            preq = _ld_uniform_i32(t2r_rsrc, rblk * 32 * 4)
            bvis = _persist_block_vis(cl_rsrc, preq)
            issue = [
                (span0 + k0 * WMMA_COLS < bvis) & (span0 + k0 * WMMA_COLS < columns)
                for k0 in range_constexpr(persist_chunks)
            ]
            ppa = plans[0][0]
            ppb = plans[1][0]
            vmax = _imax(ppa.vmax, ppb.vmax)
            qfa = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
            qfb = _q_frags(q_rsrc, t0 + WMMA_ROWS_PER_WAVE, rows, n_in, kgrp)
            # Prologue: chunk 0 -> buffer 0, completed and published.
            if issue[0]:
                _persist_issue_tdm(KV, kbase, pt_rsrc, preq, span0, width, num_pages, 0)
            fx.rocdl.tdm_ops.tensor_wait(0)
            fx.gpu.barrier()
            for k in range_constexpr(persist_chunks):
                # (1) Issue chunk k+1 into buffer (k+1)%2. Its last readers
                # (chunk k-1) all passed the end-of-iteration barrier below.
                for kn in range_constexpr(k + 1, _persist_next(k)):
                    if issue[kn]:
                        _persist_issue_tdm(
                            KV,
                            kbase,
                            pt_rsrc,
                            preq,
                            span0 + kn * WMMA_COLS,
                            width,
                            num_pages,
                            kn % TDM_PERSIST_BUFFERS,
                        )
                # (2) Consume chunk k from buffer k%2 while chunk k+1 is in flight.
                col0 = span0 + k * WMMA_COLS
                cols = _lane_cols(col0, n_in)
                offa = _store_offsets(t0, kgrp, cols, rows, columns)
                offb = _store_offsets(
                    t0 + WMMA_ROWS_PER_WAVE, kgrp, cols, rows, columns
                )
                kbuf = kbase + (k % TDM_PERSIST_BUFFERS) * (TDM_K_TILE_BYTES // 4)
                if col0 < vmax:
                    res = _pair_values(
                        atom,
                        kbuf,
                        pt_rsrc,
                        qfa,
                        qfb,
                        ppa,
                        ppb,
                        cols,
                        offa,
                        offb,
                        columns,
                        width,
                        num_pages,
                    )
                    for j1 in range_constexpr(4 * WMMA_N_TILES):
                        if res[j1][0]:
                            _st_f32(logits_rsrc, res[j1][1], res[j1][2])
                if col0 >= vmax:
                    for mi2 in range_constexpr(2):
                        msk = _pass_masks(
                            plans[mi2][0], cols, (offa, offb)[mi2], columns
                        )
                        for j2 in range_constexpr(2 * WMMA_N_TILES):
                            if msk[j2][0]:
                                _st_f32(logits_rsrc, msk[j2][1], neg_inf)
                # (3) Complete chunk k+1 (own gathers), then one rendezvous:
                # publishes buffer (k+1)%2 and retires buffer k%2 for chunk k+2.
                for kw in range_constexpr(k + 1, _persist_next(k)):
                    fx.rocdl.tdm_ops.tensor_wait(0)
                    fx.gpu.barrier()
        if reuse == 0:
            # Exact r16 per-M-tile raw path for mixed requests / incomplete blocks.
            for kf in range_constexpr(persist_chunks):
                fcol0 = span0 + kf * WMMA_COLS
                fcols = _lane_cols(fcol0, n_in)
                for fm in range_constexpr(2):
                    subt0 = t0 + fm * WMMA_ROWS_PER_WAVE
                    foffs = _store_offsets(subt0, kgrp, fcols, rows, columns)
                    live = subt0 < rows
                    if live:
                        qf = _q_frags(q_rsrc, subt0, rows, n_in, kgrp)
                        for p in range_constexpr(WMMA_PASSES):
                            pp = plans[fm][p]
                            if pp.active:
                                if fcol0 < pp.vmax:
                                    fallback = _pass_values(
                                        atom,
                                        kv_rsrc,
                                        pt_rsrc,
                                        qf,
                                        pp,
                                        fcols,
                                        foffs,
                                        columns,
                                        width,
                                        num_pages,
                                    )
                                    for j3 in range_constexpr(2 * WMMA_N_TILES):
                                        if fallback[j3][0]:
                                            _st_f32(
                                                logits_rsrc,
                                                fallback[j3][1],
                                                fallback[j3][2],
                                            )
                                if fcol0 >= pp.vmax:
                                    masks = _pass_masks(pp, fcols, foffs, columns)
                                    for j4 in range_constexpr(2 * WMMA_N_TILES):
                                        if masks[j4][0]:
                                            _st_f32(logits_rsrc, masks[j4][1], neg_inf)

    @flyc.kernel(name=kernel_name, known_block_size=[WMMA_THREADS, 1, 1])
    def kernel_r32_persist_fbl(
        Q,
        KV,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        # ROUND-R2B: kernel_r32_persist with its raw fallback's chunk loop as a
        # runtime scf.for (unroll disabled). The reuse path is verbatim.
        tid = fx.Int32(fx.thread_idx.x)
        chunk = fx.Int32(fx.block_idx.x)
        rblk = fx.Int32(fx.block_idx.y)
        wave = tid // WMMA_WAVE
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N  # A: slot m / B, C: column n
        kgrp = lane // WMMA_N  # k-half of the operands, row pair of C
        t0 = (rblk * WMMA_WAVES + wave) * (2 * WMMA_ROWS_PER_WAVE)
        span0 = chunk * (WMMA_COLS * persist_chunks)

        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))

        metas = [
            _rows_meta(
                t2r_rsrc,
                qp_rsrc,
                cl_rsrc,
                t0 + mi * WMMA_ROWS_PER_WAVE,
                rows,
                num_requests,
            )
            for mi in range_constexpr(2)
        ]
        plans = [_pass_plan(metas[mi], rows, kgrp) for mi in range_constexpr(2)]
        for mi in range_constexpr(2):
            my_vis = _pick_vis(lane, metas[mi])
            my_row = t0 + mi * WMMA_ROWS_PER_WAVE + _imin(lane, WMMA_ROWS_PER_WAVE - 1)
            meta_off = _imin(my_row, rows - 1) * 4
            writer = (chunk == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)
            if writer:
                _st_i32(vis_rsrc, meta_off, my_vis)
                _st_i32(rst_rsrc, meta_off, fx.Int32(0))

        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        neg_inf = fx.Float32(_NEG_INF)
        kbase = _persist_storage_tdm()
        reuse = _r32_guard(t2r_rsrc, rblk * 32, rows, num_requests)
        if reuse:
            preq = _ld_uniform_i32(t2r_rsrc, rblk * 32 * 4)
            bvis = _persist_block_vis(cl_rsrc, preq)
            issue = [
                (span0 + k0 * WMMA_COLS < bvis) & (span0 + k0 * WMMA_COLS < columns)
                for k0 in range_constexpr(persist_chunks)
            ]
            ppa = plans[0][0]
            ppb = plans[1][0]
            vmax = _imax(ppa.vmax, ppb.vmax)
            qfa = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
            qfb = _q_frags(q_rsrc, t0 + WMMA_ROWS_PER_WAVE, rows, n_in, kgrp)
            if issue[0]:
                _persist_issue_tdm(KV, kbase, pt_rsrc, preq, span0, width, num_pages, 0)
            fx.rocdl.tdm_ops.tensor_wait(0)
            fx.gpu.barrier()
            for k in range_constexpr(persist_chunks):
                for kn in range_constexpr(k + 1, _persist_next(k)):
                    if issue[kn]:
                        _persist_issue_tdm(
                            KV,
                            kbase,
                            pt_rsrc,
                            preq,
                            span0 + kn * WMMA_COLS,
                            width,
                            num_pages,
                            kn % TDM_PERSIST_BUFFERS,
                        )
                col0 = span0 + k * WMMA_COLS
                cols = _lane_cols(col0, n_in)
                offa = _store_offsets(t0, kgrp, cols, rows, columns)
                offb = _store_offsets(
                    t0 + WMMA_ROWS_PER_WAVE, kgrp, cols, rows, columns
                )
                kbuf = kbase + (k % TDM_PERSIST_BUFFERS) * (TDM_K_TILE_BYTES // 4)
                if col0 < vmax:
                    res = _pair_values(
                        atom,
                        kbuf,
                        pt_rsrc,
                        qfa,
                        qfb,
                        ppa,
                        ppb,
                        cols,
                        offa,
                        offb,
                        columns,
                        width,
                        num_pages,
                    )
                    for j1 in range_constexpr(4 * WMMA_N_TILES):
                        if res[j1][0]:
                            _st_f32(logits_rsrc, res[j1][1], res[j1][2])
                if col0 >= vmax:
                    for mi2 in range_constexpr(2):
                        msk = _pass_masks(
                            plans[mi2][0], cols, (offa, offb)[mi2], columns
                        )
                        for j2 in range_constexpr(2 * WMMA_N_TILES):
                            if msk[j2][0]:
                                _st_f32(logits_rsrc, msk[j2][1], neg_inf)
                for kw in range_constexpr(k + 1, _persist_next(k)):
                    fx.rocdl.tdm_ops.tensor_wait(0)
                    fx.gpu.barrier()
        if reuse == 0:
            # Exact r16 per-M-tile raw path; the chunk loop is a runtime loop (R2B).
            for kf in range(
                fx.Int32(0), fx.Int32(persist_chunks), fx.Int32(1), unroll=1
            ):
                fcol0 = span0 + fx.Int32(kf) * WMMA_COLS
                fcols = _lane_cols(fcol0, n_in)
                for fm in range_constexpr(2):
                    subt0 = t0 + fm * WMMA_ROWS_PER_WAVE
                    foffs = _store_offsets(subt0, kgrp, fcols, rows, columns)
                    live = subt0 < rows
                    if live:
                        qf = _q_frags(q_rsrc, subt0, rows, n_in, kgrp)
                        for p in range_constexpr(WMMA_PASSES):
                            pp = plans[fm][p]
                            if pp.active:
                                if fcol0 < pp.vmax:
                                    fallback = _pass_values(
                                        atom,
                                        kv_rsrc,
                                        pt_rsrc,
                                        qf,
                                        pp,
                                        fcols,
                                        foffs,
                                        columns,
                                        width,
                                        num_pages,
                                    )
                                    for j3 in range_constexpr(2 * WMMA_N_TILES):
                                        if fallback[j3][0]:
                                            _st_f32(
                                                logits_rsrc,
                                                fallback[j3][1],
                                                fallback[j3][2],
                                            )
                                if fcol0 >= pp.vmax:
                                    masks = _pass_masks(pp, fcols, foffs, columns)
                                    for j4 in range_constexpr(2 * WMMA_N_TILES):
                                        if masks[j4][0]:
                                            _st_f32(logits_rsrc, masks[j4][1], neg_inf)

    @flyc.kernel(name=kernel_name, known_block_size=[WMMA_THREADS, 1, 1])
    def kernel_r32_kstat(
        Q,
        KV,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        # ROUND-R2B: block (x, y) owns columns [x*64C, x*64C + 64C) and the R
        # 32-row blocks y*R .. y*R + R - 1 (runtime loop, unroll disabled).
        tid = fx.Int32(fx.thread_idx.x)
        slab = fx.Int32(fx.block_idx.x)
        rspan = fx.Int32(fx.block_idx.y)
        wave = tid // WMMA_WAVE
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N  # A: slot m / B, C: column n
        kgrp = lane // WMMA_N  # k-half of the operands, row pair of C
        slab0 = slab * (WMMA_COLS * col_tiles)

        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))

        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        neg_inf = fx.Float32(_NEG_INF)
        kbase = _kstat_storage_tdm()
        # The slab's K request (SGPR): one TDM issue per K tile for the whole span.
        # Issue guards are SGPR-uniform and never surround a wait or a barrier.
        kreq, kreq_ok, ksafe, kvis = _kstat_slab_req(
            t2r_rsrc, cl_rsrc, rspan * (32 * row_tiles), rows, num_requests
        )
        kissue = [
            kreq_ok
            & (slab0 + ki * WMMA_COLS < kvis)
            & (slab0 + ki * WMMA_COLS < columns)
            for ki in range_constexpr(col_tiles)
        ]
        for ki2 in range_constexpr(col_tiles):
            if kissue[ki2]:
                _persist_issue_tdm(
                    KV,
                    kbase,
                    pt_rsrc,
                    ksafe,
                    slab0 + ki2 * WMMA_COLS,
                    width,
                    num_pages,
                    ki2,
                )
        # ROUND-R2C early: the slab's page validity per lane column (no-op otherwise).
        kpre = kstat_pre(pt_rsrc, ksafe, kreq_ok, slab0, n_in, width, num_pages)
        for it in range(fx.Int32(0), fx.Int32(row_tiles), fx.Int32(1), unroll=1):
            rb = rspan * row_tiles + fx.Int32(it)
            t0 = (rb * WMMA_WAVES + wave) * (2 * WMMA_ROWS_PER_WAVE)
            metas = [
                _rows_meta(
                    t2r_rsrc,
                    qp_rsrc,
                    cl_rsrc,
                    t0 + mi * WMMA_ROWS_PER_WAVE,
                    rows,
                    num_requests,
                )
                for mi in range_constexpr(2)
            ]
            plans = [_pass_plan(metas[mi], rows, kgrp) for mi in range_constexpr(2)]
            for mi in range_constexpr(2):
                my_vis = _pick_vis(lane, metas[mi])
                my_row = (
                    t0 + mi * WMMA_ROWS_PER_WAVE + _imin(lane, WMMA_ROWS_PER_WAVE - 1)
                )
                meta_off = _imin(my_row, rows - 1) * 4
                writer = (slab == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)
                if writer:
                    _st_i32(vis_rsrc, meta_off, my_vis)
                    _st_i32(rst_rsrc, meta_off, fx.Int32(0))
            reuse = _kstat_tile_ok(t2r_rsrc, rb * 32, rows, num_requests, kreq)
            # Q of both M tiles is issued before the K publish so its latency overlaps
            # the first (only) TDM wait; clamped rows keep every load in bounds.
            qfa = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
            qfb = _q_frags(q_rsrc, t0 + WMMA_ROWS_PER_WAVE, rows, n_in, kgrp)
            fx.rocdl.tdm_ops.tensor_wait(0)
            fx.gpu.barrier()
            if reuse:
                ppa = plans[0][0]
                ppb = plans[1][0]
                vmax = _imax(ppa.vmax, ppb.vmax)
                for ct in range_constexpr(col_tiles):
                    col0 = slab0 + ct * WMMA_COLS
                    cols = _lane_cols(col0, n_in)
                    offa = _store_offsets(t0, kgrp, cols, rows, columns)
                    offb = _store_offsets(
                        t0 + WMMA_ROWS_PER_WAVE, kgrp, cols, rows, columns
                    )
                    kbuf = kbase + ct * (TDM_K_TILE_BYTES // 4)
                    if col0 < vmax:
                        res = kstat_values(
                            atom,
                            kbuf,
                            pt_rsrc,
                            qfa,
                            qfb,
                            ppa,
                            ppb,
                            cols,
                            offa,
                            offb,
                            columns,
                            width,
                            num_pages,
                            kpre,
                            ct,
                        )
                        for j1 in range_constexpr(4 * WMMA_N_TILES):
                            if res[j1][0]:
                                _st_f32(logits_rsrc, res[j1][1], res[j1][2])
                    if col0 >= vmax:
                        for mi2 in range_constexpr(2):
                            msk = _pass_masks(
                                plans[mi2][0], cols, (offa, offb)[mi2], columns
                            )
                            for j2 in range_constexpr(2 * WMMA_N_TILES):
                                if msk[j2][0]:
                                    _st_f32(logits_rsrc, msk[j2][1], neg_inf)
            if reuse == 0:
                # Exact r16 per-M-tile raw path (mixed / partial / other-request blocks).
                for kf in range(
                    fx.Int32(0), fx.Int32(col_tiles), fx.Int32(1), unroll=1
                ):
                    fcol0 = slab0 + fx.Int32(kf) * WMMA_COLS
                    fcols = _lane_cols(fcol0, n_in)
                    for fm in range_constexpr(2):
                        subt0 = t0 + fm * WMMA_ROWS_PER_WAVE
                        foffs = _store_offsets(subt0, kgrp, fcols, rows, columns)
                        live = subt0 < rows
                        if live:
                            qf = _q_frags(q_rsrc, subt0, rows, n_in, kgrp)
                            for p in range_constexpr(WMMA_PASSES):
                                pp = plans[fm][p]
                                if pp.active:
                                    if fcol0 < pp.vmax:
                                        fallback = _pass_values(
                                            atom,
                                            kv_rsrc,
                                            pt_rsrc,
                                            qf,
                                            pp,
                                            fcols,
                                            foffs,
                                            columns,
                                            width,
                                            num_pages,
                                        )
                                        for j3 in range_constexpr(2 * WMMA_N_TILES):
                                            if fallback[j3][0]:
                                                _st_f32(
                                                    logits_rsrc,
                                                    fallback[j3][1],
                                                    fallback[j3][2],
                                                )
                                    if fcol0 >= pp.vmax:
                                        masks = _pass_masks(pp, fcols, foffs, columns)
                                        for j4 in range_constexpr(2 * WMMA_N_TILES):
                                            if masks[j4][0]:
                                                _st_f32(
                                                    logits_rsrc, masks[j4][1], neg_inf
                                                )

    @flyc.kernel(name=kernel_name, known_block_size=[kt_threads, 1, 1])
    def kernel_kt(
        Q,
        KV,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        # ROUND-R2C: block (x, y) owns columns [x*64C, x*64C + 64C) and rows
        # [y*RB, y*RB + RB); wave w owns the M tile of rows y*RB + 4w .. +3.
        _inst_prefetch(ipf)  # ROUND-R2D (ipf=0: nothing)
        tid = fx.Int32(fx.thread_idx.x)
        slab = fx.Int32(fx.block_idx.x)
        rblk = fx.Int32(fx.block_idx.y)
        wave = tid // WMMA_WAVE
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N  # A: slot m / B, C: column n
        kgrp = lane // WMMA_N  # k-half of the operands, row pair of C
        slab0 = slab * (WMMA_COLS * col_tiles)
        base = rblk * kt_rows
        t0 = base + wave * WMMA_ROWS_PER_WAVE
        _tr_begin(
            trs,
            trace,
            LOGITS,
            rows,
            columns,
            (columns + WMMA_COLS * col_tiles - 1) // (WMMA_COLS * col_tiles),
            (rows + kt_rows - 1) // kt_rows,
            kt_waves,
            slab,
            rblk,
        )

        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))

        meta = _rows_meta(t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests)
        plans = _pass_plan(meta, rows, kgrp)
        my_vis = _pick_vis(lane, meta)
        my_row = t0 + _imin(lane, WMMA_ROWS_PER_WAVE - 1)
        meta_off = _imin(my_row, rows - 1) * 4
        writer = (slab == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)
        if writer:
            _st_i32(vis_rsrc, meta_off, my_vis)
            _st_i32(rst_rsrc, meta_off, fx.Int32(0))
        _tr_mark(trs, 4, True)

        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        neg_inf = fx.Float32(_NEG_INF)
        kbase = _kstat_storage_tdm()
        kreq, full = _kt_guard(t2r_rsrc, base, rows, num_requests)
        _tr_mark(trs, 5, True)
        if full:
            # Uniform branch: TDM issue (SGPR guards, no wait inside), then Q + page
            # validity, then the single wait + barrier, then LDS-only scoring.
            kvis = _persist_block_vis(cl_rsrc, kreq)
            kissue = [
                (slab0 + ki * WMMA_COLS < kvis) & (slab0 + ki * WMMA_COLS < columns)
                for ki in range_constexpr(col_tiles)
            ]
            for ki2 in range_constexpr(col_tiles):
                if kissue[ki2]:
                    kt_issue(
                        KV,
                        kbase,
                        pt_rsrc,
                        kreq,
                        slab0 + ki2 * WMMA_COLS,
                        width,
                        num_pages,
                        ki2,
                    )
            _tr_mark(trs, 6)
            qf = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
            kpre = _kstat_page_pre(pt_rsrc, kreq, full, slab0, n_in, width, num_pages)
            _tr_mark(trs, 7, True)
            fx.rocdl.tdm_ops.tensor_wait(0)
            _tr_mark(trs, 8)
            fx.gpu.barrier()
            _tr_mark(trs, 9)
            _tr_mark_rt(trs, 14)
            pf = plans[0]
            for ct in range_constexpr(col_tiles):
                col0 = slab0 + ct * WMMA_COLS
                cols = _lane_cols(col0, n_in)
                offs = _store_offsets(t0, kgrp, cols, rows, columns)
                kbuf = kbase + ct * (TDM_K_TILE_BYTES // 4)
                if col0 < pf.vmax:
                    res = _kt_values(atom, kbuf, qf, pf, cols, offs, columns, kpre[ct])
                    for j1 in range_constexpr(2 * WMMA_N_TILES):
                        if res[j1][0]:
                            _st_f32(logits_rsrc, res[j1][1], res[j1][2])
                if col0 >= pf.vmax:
                    msk = _pass_masks(pf, cols, offs, columns)
                    for j2 in range_constexpr(2 * WMMA_N_TILES):
                        if msk[j2][0]:
                            _st_f32(logits_rsrc, msk[j2][1], neg_inf)
            _tr_mark(trs, 10)
        if full == 0:
            # Exact per-wave raw path (mixed / partial / invalid-request blocks).
            for kf in range(fx.Int32(0), fx.Int32(col_tiles), fx.Int32(1), unroll=1):
                fcol0 = slab0 + fx.Int32(kf) * WMMA_COLS
                fcols = _lane_cols(fcol0, n_in)
                foffs = _store_offsets(t0, kgrp, fcols, rows, columns)
                live = t0 < rows
                if live:
                    fqf = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
                    for p in range_constexpr(WMMA_PASSES):
                        pp = plans[p]
                        if pp.active:
                            if fcol0 < pp.vmax:
                                fallback = _pass_values(
                                    atom,
                                    kv_rsrc,
                                    pt_rsrc,
                                    fqf,
                                    pp,
                                    fcols,
                                    foffs,
                                    columns,
                                    width,
                                    num_pages,
                                )
                                for j3 in range_constexpr(2 * WMMA_N_TILES):
                                    if fallback[j3][0]:
                                        _st_f32(
                                            logits_rsrc,
                                            fallback[j3][1],
                                            fallback[j3][2],
                                        )
                            if fcol0 >= pp.vmax:
                                masks = _pass_masks(pp, fcols, foffs, columns)
                                for j4 in range_constexpr(2 * WMMA_N_TILES):
                                    if masks[j4][0]:
                                        _st_f32(logits_rsrc, masks[j4][1], neg_inf)
            _tr_mark(trs, 10)
        _tr_end(trs)

    # ---- ROUND-R2D lean ktile (k_load='ktile', lean=True; see _r2d_check) -------------
    def _ktl_pre(
        t2r_rsrc,
        qp_rsrc,
        cl_rsrc,
        pt_rsrc,
        base,
        t0s,
        rows,
        columns,
        num_requests,
        slab0,
        width,
        num_pages,
        wave_id,
    ):
        """ONE batch of SGPR loads for the full-block path, then the derived values.

        Returns (full, safe request, vis[4] of the wave's rows, kissue[ct], gathers).
        Offsets are clamped exactly like `_kt_guard` / `_rows_meta_uniform` /
        `_kt_issue_w8` / `_persist_issue_tdm` (in-plane by construction); the TDM row ids
        are the same expressions (valid -> phys, else page 0), so the gathers are identical.
        """
        # -- issue (no use in between) --
        g = [
            _ld_uniform_i32(t2r_rsrc, _imin(base + i, rows - 1) * 4)
            for i in range_constexpr(kt_rows)
        ]
        req0 = g[0]
        req_ok = (req0 >= 0) & (req0 < num_requests)
        safe = req_ok.select(req0, 0)
        pos = [
            ld_uniform_pos(qp_rsrc, _imin(t0s + i, rows - 1))
            for i in range_constexpr(WMMA_ROWS_PER_WAVE)
        ]
        ctx = _ld_uniform_i32(cl_rsrc, safe * 4)
        gpw = 8 // kt_waves  # 8-row gathers per wave per 64-col tile
        span = WMMA_COLS // kt_waves  # columns per wave per 64-col tile
        plan = []
        for ct in range_constexpr(col_tiles):
            for gi in range_constexpr(gpw):
                local = wave_id * span + gi * 8
                col = slab0 + ct * WMMA_COLS + local
                ents = []
                for page in range_constexpr(2 if P == 4 else 1):
                    logical = col // P + page
                    logical_ok = logical < width
                    safe_logical = logical_ok.select(logical, 0)
                    ents.append(
                        (
                            logical_ok,
                            _ld_uniform_i32(pt_rsrc, (safe * width + safe_logical) * 4),
                        )
                    )
                plan.append((ct, local, ents))
        # -- use --
        full = (base + kt_rows <= rows) & req_ok
        for i in range_constexpr(1, kt_rows):
            full = full & (g[i] == req0)
        # Pin the batch before the (uniform) full branch: LLVM otherwise sinks the ctx /
        # page-table loads into it and re-serialises them behind the ctx wait. An empty
        # side-effecting asm reading the SGPRs needs them all landed (ONE s_wait_kmcnt);
        # it emits no instruction and changes no value (ull = the kernel_kt guard).
        _pin_sgprs(
            g + [ctx] + [phys for _ct, _local, _ents in plan for _lok, phys in _ents]
        )
        vis = [
            _imax(_imin(pos[i] + 1, ctx), 0) // COMPRESS_RATIO
            for i in range_constexpr(WMMA_ROWS_PER_WAVE)
        ]
        kvis = _imax(ctx, 0) // COMPRESS_RATIO
        kissue = [
            (slab0 + ct * WMMA_COLS < kvis) & (slab0 + ct * WMMA_COLS < columns)
            for ct in range_constexpr(col_tiles)
        ]
        gathers = []
        for ct, local, ents in plan:
            row_ids = []
            for logical_ok, phys in ents:
                valid = logical_ok & (phys >= 0) & (phys < num_pages)
                phys = valid.select(phys, 0)
                if P == 4:
                    for row in range_constexpr(4):
                        row_ids.append(phys * 4 + row)
                else:
                    row0 = local % 16 if P == 16 else 0
                    for r in range_constexpr(8):
                        row_ids.append(phys * P + (row0 + r))
            gathers.append((ct, local, row_ids))
        return full, safe, vis, kissue, gathers

    def _ktl_issue(KV, kbase, gathers, ct, num_pages):
        for gct, local, row_ids in gathers:
            if gct == ct:
                _tdm_k_gather(
                    KV,
                    kbase,
                    row_ids,
                    num_pages,
                    ct * TDM_K_TILE_BYTES + local * TDM_K_ROW_BYTES,
                    P,
                )

    def _ktl_rowvals(lane, kgrp, vis):
        """(this lane's vis/rst value for lanes 0..3, vis_a, vis_b, vmax) from SGPR vis."""
        v = vis[WMMA_ROWS_PER_WAVE - 1]
        for i in range_constexpr(WMMA_ROWS_PER_WAVE - 2, -1, -1):
            v = (lane == i).select(vis[i], v)
        g0 = kgrp == 0
        vmax = _imax(_imax(vis[0], vis[1]), _imax(vis[2], vis[3]))
        return v, g0.select(vis[0], vis[2]), g0.select(vis[1], vis[3]), vmax

    def _ktl_values(atom, kbuf, qf, kgrp, cols, offs, columns, oks, vis_a, vis_b):
        """`_kt_values` of a full block (every row owned, ok = column in range)."""
        neg_inf = fx.Float32(_NEG_INF)
        out = []
        for nt in range_constexpr(len(cols)):
            c = cols[nt]
            sa, sb = _kt_tile(atom, kbuf, qf, c, kgrp)
            va = (oks[nt] & (c < vis_a)).select(sa, neg_inf)
            vb = (oks[nt] & (c < vis_b)).select(sb, neg_inf)
            out.append((c < columns, offs[2 * nt], va))
            out.append((c < columns, offs[2 * nt + 1], vb))
        return out

    @flyc.kernel(name=kernel_name, known_block_size=[kt_threads, 1, 1])
    def kernel_kt_lean(
        Q,
        KV,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        # ROUND-R2D: kernel_kt with the lean full-block path (see _r2d_check / _ktl_pre).
        _inst_prefetch(ipf)
        tid = fx.Int32(fx.thread_idx.x)
        slab = fx.Int32(fx.block_idx.x)
        rblk = fx.Int32(fx.block_idx.y)
        wave_id = fx.Int32(fx.rocdl.wave_id())
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N
        kgrp = lane // WMMA_N
        slab0 = slab * (WMMA_COLS * col_tiles)
        base = rblk * kt_rows
        t0s = base + wave_id * WMMA_ROWS_PER_WAVE
        _tr_begin(
            trs,
            trace,
            LOGITS,
            rows,
            columns,
            (columns + WMMA_COLS * col_tiles - 1) // (WMMA_COLS * col_tiles),
            (rows + kt_rows - 1) // kt_rows,
            kt_waves,
            slab,
            rblk,
        )

        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))
        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        neg_inf = fx.Float32(_NEG_INF)
        kbase = _kstat_storage_tdm()
        full, kreq, lvis, kissue, gathers = _ktl_pre(
            t2r_rsrc,
            qp_rsrc,
            cl_rsrc,
            pt_rsrc,
            base,
            t0s,
            rows,
            columns,
            num_requests,
            slab0,
            width,
            num_pages,
            wave_id,
        )
        _tr_mark(trs, 5, True)
        if full:
            for ki in range_constexpr(col_tiles):
                if kissue[ki]:
                    _ktl_issue(KV, kbase, gathers, ki, num_pages)
            _tr_mark(trs, 6)
            qf = _q_frags(q_rsrc, t0s, rows, n_in, kgrp)
            kpre = _kstat_page_pre(pt_rsrc, kreq, full, slab0, n_in, width, num_pages)
            lv, vis_a, vis_b, vmax = _ktl_rowvals(lane, kgrp, lvis)
            lrow_off = _imin(t0s + _imin(lane, WMMA_ROWS_PER_WAVE - 1), rows - 1) * 4
            lwriter = (slab == 0) & (lane < WMMA_ROWS_PER_WAVE)
            if lwriter:
                _st_i32(vis_rsrc, lrow_off, lv)
                _st_i32(rst_rsrc, lrow_off, fx.Int32(0))
            _tr_mark(trs, 7, True)
            fx.rocdl.tdm_ops.tensor_wait(0)
            _tr_mark(trs, 8)
            fx.rocdl.s_barrier()  # TENSORcnt==0 => slab in LDS (mi400-tdm-sync)
            _tr_mark(trs, 9)
            _tr_mark_rt(trs, 14)
            inside = slab0 + WMMA_COLS * col_tiles <= columns
            for ct in range_constexpr(col_tiles):
                col0 = slab0 + ct * WMMA_COLS
                cols = _lane_cols(col0, n_in)
                offs = _store_offsets(t0s, kgrp, cols, rows, columns)
                kbuf = kbase + ct * (TDM_K_TILE_BYTES // 4)
                if col0 < vmax:
                    res = _ktl_values(
                        atom,
                        kbuf,
                        qf,
                        kgrp,
                        cols,
                        offs,
                        columns,
                        kpre[ct],
                        vis_a,
                        vis_b,
                    )
                    if inside:
                        for j1 in range_constexpr(2 * WMMA_N_TILES):
                            _st_f32(logits_rsrc, res[j1][1], res[j1][2])
                    if inside == 0:
                        for j2 in range_constexpr(2 * WMMA_N_TILES):
                            if res[j2][0]:
                                _st_f32(logits_rsrc, res[j2][1], res[j2][2])
                if col0 >= vmax:
                    if inside:
                        for j3 in range_constexpr(2 * WMMA_N_TILES):
                            _st_f32(logits_rsrc, offs[j3], neg_inf)
                    if inside == 0:
                        for j4 in range_constexpr(2 * WMMA_N_TILES):
                            if cols[j4 // 2] < columns:
                                _st_f32(logits_rsrc, offs[j4], neg_inf)
            _tr_mark(trs, 10)
        if full == 0:
            # Exact kernel_kt per-wave raw path (mixed / partial / invalid-request blocks).
            meta = _rows_meta(t2r_rsrc, qp_rsrc, cl_rsrc, t0s, rows, num_requests)
            plans = _pass_plan(meta, rows, kgrp)
            my_vis = _pick_vis(lane, meta)
            my_row = t0s + _imin(lane, WMMA_ROWS_PER_WAVE - 1)
            meta_off = _imin(my_row, rows - 1) * 4
            writer = (slab == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)
            if writer:
                _st_i32(vis_rsrc, meta_off, my_vis)
                _st_i32(rst_rsrc, meta_off, fx.Int32(0))
            for kf in range(fx.Int32(0), fx.Int32(col_tiles), fx.Int32(1), unroll=1):
                fcol0 = slab0 + fx.Int32(kf) * WMMA_COLS
                fcols = _lane_cols(fcol0, n_in)
                foffs = _store_offsets(t0s, kgrp, fcols, rows, columns)
                live = t0s < rows
                if live:
                    fqf = _q_frags(q_rsrc, t0s, rows, n_in, kgrp)
                    for p in range_constexpr(WMMA_PASSES):
                        pp = plans[p]
                        if pp.active:
                            if fcol0 < pp.vmax:
                                fallback = _pass_values(
                                    atom,
                                    kv_rsrc,
                                    pt_rsrc,
                                    fqf,
                                    pp,
                                    fcols,
                                    foffs,
                                    columns,
                                    width,
                                    num_pages,
                                )
                                for j5 in range_constexpr(2 * WMMA_N_TILES):
                                    if fallback[j5][0]:
                                        _st_f32(
                                            logits_rsrc,
                                            fallback[j5][1],
                                            fallback[j5][2],
                                        )
                            if fcol0 >= pp.vmax:
                                masks = _pass_masks(pp, fcols, foffs, columns)
                                for j6 in range_constexpr(2 * WMMA_N_TILES):
                                    if masks[j6][0]:
                                        _st_f32(logits_rsrc, masks[j6][1], neg_inf)
            _tr_mark(trs, 10)
        _tr_end(trs)

    @flyc.kernel(name=kernel_name, known_block_size=[WMMA_THREADS, 1, 1])
    def kernel_tdm_r16_nobar(
        Q,
        KV,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        # kernel_tdm_r16 with its two per-pass WG barriers replaced by the
        # r16_publish / r16_reuse hooks (QSA_R16_NOBAR=1 only; see above).
        tid = fx.Int32(fx.thread_idx.x)
        chunk = fx.Int32(fx.block_idx.x)
        rblk = fx.Int32(fx.block_idx.y)
        wave = fx.Int32(fx.rocdl.wave_id())
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N
        kgrp = lane // WMMA_N
        t0 = (rblk * WMMA_WAVES + wave) * WMMA_ROWS_PER_WAVE
        col0 = chunk * tdm_cols
        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))
        meta = _rows_meta_uniform(t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests)
        plans = _pass_plan(meta, rows, kgrp)
        my_vis = _pick_vis(lane, meta)
        my_row = t0 + _imin(lane, WMMA_ROWS_PER_WAVE - 1)
        meta_off = _imin(my_row, rows - 1) * 4
        writer = (chunk == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)
        if writer:
            _st_i32(vis_rsrc, meta_off, my_vis)
            _st_i32(rst_rsrc, meta_off, fx.Int32(0))
        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        cols = _lane_cols(col0, n_in, tdm_cols // WMMA_N)
        offs = _store_offsets(t0, kgrp, cols, rows, columns)
        neg_inf = fx.Float32(_NEG_INF)
        storage = _r16_storage_tdm()
        kptr = storage + wave * (tdm_tile_bytes // 4)
        full = rblk * 16 + 16 <= rows
        if full:
            qf = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
            for p in range_constexpr(WMMA_PASSES):
                stage_active = _r16_stage_active(
                    t2r_rsrc, qp_rsrc, cl_rsrc, rblk, col0, num_requests, p
                )
                if stage_active:
                    _r16_stage_tdm(
                        KV,
                        storage,
                        pt_rsrc,
                        t2r_rsrc,
                        rblk,
                        col0,
                        width,
                        num_pages,
                        num_requests,
                        p,
                    )
                fx.rocdl.tdm_ops.tensor_wait(0)
                r16_publish()
                pp = plans[p]
                if pp.active:
                    if col0 < pp.vmax:
                        res = _pass_values(
                            atom,
                            kptr,
                            pt_rsrc,
                            qf,
                            pp,
                            cols,
                            offs,
                            columns,
                            width,
                            num_pages,
                            True,
                        )
                        for j1 in range_constexpr(len(res)):
                            if res[j1][0]:
                                _st_f32(logits_rsrc, res[j1][1], res[j1][2])
                    if col0 >= pp.vmax:
                        msk = _pass_masks(pp, cols, offs, columns)
                        for j2 in range_constexpr(len(msk)):
                            if msk[j2][0]:
                                _st_f32(logits_rsrc, msk[j2][1], neg_inf)
                r16_reuse()
        if full == 0:
            live = t0 < rows
            if live:
                qf = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
                for p in range_constexpr(WMMA_PASSES):
                    pp = plans[p]
                    if pp.active:
                        if col0 < pp.vmax:
                            res = _pass_values(
                                atom,
                                kv_rsrc,
                                pt_rsrc,
                                qf,
                                pp,
                                cols,
                                offs,
                                columns,
                                width,
                                num_pages,
                            )
                            for j3 in range_constexpr(len(res)):
                                if res[j3][0]:
                                    _st_f32(logits_rsrc, res[j3][1], res[j3][2])
                        if col0 >= pp.vmax:
                            msk = _pass_masks(pp, cols, offs, columns)
                            for j4 in range_constexpr(len(msk)):
                                if msk[j4][0]:
                                    _st_f32(logits_rsrc, msk[j4][1], neg_inf)

    selected_kernel = (
        (kernel_tdm_r16 if use_tdm else kernel)
        if rows_per_block == 16
        else (kernel_r64 if rows_per_block == 64 else kernel_r32)
    )
    if persist:
        selected_kernel = kernel_r32_persist_fbl if fb_loop else kernel_r32_persist
    if kstat:
        selected_kernel = kernel_r32_kstat
    if ktile:
        selected_kernel = kernel_kt_lean if lean else kernel_kt
    if r16_nobar:
        selected_kernel = kernel_tdm_r16_nobar

    @flyc.jit
    def launch(
        Q,
        KV,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        grid_x,
        grid_y,
        rows,
        columns,
        num_pages,
        num_requests,
        width,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        selected_kernel._func.__name__ = kernel_name
        selected_kernel(
            Q,
            KV,
            PT,
            T2R,
            QP,
            CL,
            LOGITS,
            VIS,
            RST,
            rows,
            columns,
            num_pages,
            num_requests,
            width,
        ).launch(
            grid=(fx.Int64(grid_x), fx.Int64(grid_y), 1),
            block=(kt_threads, 1, 1),
            stream=stream,
        )

    launch.kernel_name = kernel_name
    launch.dtype_tag = "bf16_wmma"
    launch.cols_per_block = cols_per_block  # host grid.x must use this width
    launch.persist_chunks = persist_chunks
    if persist:
        # A persistent block spans persist_chunks 64-column tiles.
        launch.cols_per_block = cols_per_block * persist_chunks
    # ROUND-R2B: rows one workgroup covers (grid.y = ceil(rows / rows_per_launch)).
    launch.rows_per_launch = rows_per_block
    if kstat:
        launch.cols_per_block = cols_per_block * col_tiles
        launch.rows_per_launch = 32 * row_tiles
        launch.kstat = (row_tiles, col_tiles)
        launch.early = early
    if ktile:  # ROUND-R2C
        launch.cols_per_block = WMMA_COLS * col_tiles
        launch.rows_per_launch = kt_rows
        launch.kt = (kt_rows, col_tiles)
        launch.threads = kt_threads
        launch.lean, launch.ipf = bool(lean), int(ipf)
    launch.trace = trace
    return launch


# ---------------------------------------------------------------------------
# fp8 matrix-core path (round G): E4M3 -> bf16 in registers, same bf16 WMMA
# ---------------------------------------------------------------------------
# The fp8 cache is E4M3 with one fp32 scale per compressed token (= per column),
# and ``dot(q, s * k) = s * dot(q, k)``.  So the fp8 specialisation below is the
# bf16 WMMA kernel above, helper for helper (``_g2/test_fp8_wmma_cpu.py`` checks
# the shared helpers are textually identical), with exactly three differences:
#
#   1. a B fragment is built from 2 x 8 E4M3 bytes (``buffer_load_dwordx2`` at
#      the *same* k positions as bf16: dims ``(l//16)*8 + 0..7`` and ``+16``
#      of each 32-dim WMMA step) and converted in registers to 16 bf16 values
#      **exactly** (``_e4m3x4_to_bf16x2``); the A (Q) fragments are unchanged;
#   2. after the 4 WMMA steps each head's f32 dot is multiplied by the
#      column's fp32 scale ``scale[phys, c % 4]`` *before* ReLU, then the heads
#      are summed and multiplied by ``1/divisor`` Ã¢â‚¬â€ the scalar fp8 kernel's
#      order of operations (``(acc_h * scale).maximumf(0)`` summed h = 0..3,
#      then ``* inv_divisor``);
#   3. the scale load uses the page-table path of the K load: ``phys`` is the
#      already-masked physical page (``page_ok.select(phys, 0)``) and ``c % 4``
#      is in [0, 4), so the byte offset ``(phys * 4 + c % 4) * 4`` is always
#      inside the ``num_pages * 16`` B scale plane.
#
# Everything else (tiling, Q fragments, request-dedup passes, early -inf exit,
# predicated stores at clamped in-plane offsets, i64 descriptor extents) is the
# bf16 kernel's.  ``QSA_FP8_PATH=scalar`` routes fp8 back to the round-F scalar
# kernel (LDS-staged Q), which stays built and selectable.
# Only the full r32 padded-TDM pair conversion is eligible; native is default.
# Explicit current, r16, raw/mixed/tail and bf16 retain the champion converter.
_FP8_CVT = os.environ.get("QSA_FP8_CVT", "fast").strip().lower()
if _FP8_CVT not in ("current", "fast"):
    raise ValueError(f"QSA_FP8_CVT must be 'current' or 'fast', got {_FP8_CVT!r}")

_FP8_PATH = os.environ.get("QSA_FP8_PATH", "wmma").strip().lower()
if _FP8_PATH not in ("wmma", "scalar"):
    raise ValueError(f"QSA_FP8_PATH must be 'wmma' or 'scalar', got {_FP8_PATH!r}")


# Stock dev886: element counts 128/16 at 8 bits -> 32/4 DWORDs -> fields 4/3.
# Source rows stay 128 B; padding is LDS-only. Every b128 consumer is aligned.
TDM_FP8_PAD_INTERVAL_DWORDS = 32
TDM_FP8_PAD_AMOUNT_DWORDS = 4
TDM_FP8_K_ROW_BYTES = (TDM_FP8_PAD_INTERVAL_DWORDS + TDM_FP8_PAD_AMOUNT_DWORDS) * 4
TDM_FP8_K_TILE_BYTES = WMMA_COLS * TDM_FP8_K_ROW_BYTES  # 9,216 B
# The shared K-load switch forces both dtypes unless explicitly overridden.
_FP8_K_LOAD = os.environ.get("QSA_FP8_K_LOAD", _BF16_K_LOAD).strip().lower()
if _FP8_K_LOAD not in ("raw", "tdm", "auto"):
    raise ValueError(
        f"QSA_FP8_K_LOAD must be 'raw', 'tdm' or 'auto', got {_FP8_K_LOAD!r}"
    )
# ROUND-R2C: QSA_FP8_EARLY=1 routes fp8 r32 TDM prefill to kernel_r32_early (default 0:
# the unchanged kernel_r32, byte-identical ISA).
FP8_R32_EARLY = os.environ.get("QSA_FP8_EARLY", "0").strip() == "1"
# ROUND-R2C: QSA_FP8_KT=<RB>x<C> (e.g. 16x2) routes fp8 r32 TDM prefill to the fp8 ktile
# kernel (default "": off, the unchanged kernel_r32 / kernel_r32_early).
_FP8_KT = os.environ.get("QSA_FP8_KT", "").strip().lower()
FP8_KT = None
if _FP8_KT:
    _m = re.fullmatch(r"(\d+)x(\d+)", _FP8_KT)
    if not _m:
        raise ValueError("QSA_FP8_KT must be <rows>x<col_tiles>, e.g. 16x2")
    FP8_KT = (int(_m[1]), int(_m[2]))

# ROUND-R2E: data-driven fp8 r32 TDM prefill route (rows >= 32 and rows >= 32 * requests,
# TDM, 128 B aligned cache, raw access). QSA_FP8_PREFILL_ROUTE is a preset name or a rule
# list "P,cmin,cmax,rmin,rmax,variant;..." (bounds inclusive, '*' = any; P in 4/8/16/'*');
# variant = tdm (kernel_r32) | early (kernel_r32_early) | kt<RB>x<C>[l][k][p<KB>] (fp8
# ktile; l = lean, k = kconv, needs l). First match wins; no match -> tdm. The R2C env
# switches QSA_FP8_KT / QSA_FP8_EARLY=1, when set, take precedence over the table.
FP8_PREFILL_ROUTE_PRESETS = {
    "legacy": None,  # kernel_r32 everywhere (<= R2D 685b0f60)
    # ROUND-R2E default (GPU c1024: rows 1024 kt32x2lk 4.23-4.25 vs native 5.22-5.24;
    # rows 4096 kt32x4lk 8.14-8.36 vs 13.3-13.5); r32 fp8 above 2048 columns
    # (long-column A/B pending: records/r2e/GPU_COMMANDS.md (d)).
    "r2e": "*,*,2048,*,2048,kt32x2lk;*,*,2048,2049,16384,kt32x4lk",
    # ROUND-R2E alternative for the long-column A/B: lean+kconv ktile everywhere.
    "kt_all": "*,*,*,*,2048,kt32x2lk;*,*,*,2049,*,kt32x4lk",
}
FP8_PREFILL_ROUTE_DEFAULT = "r2e"


def _parse_fp8_route_variant(v: str):
    v = v.strip().lower()
    if v in ("tdm", "early"):
        return v, {}
    m = re.fullmatch(r"kt(\d+)x(\d+)(l?)(k?)(?:p(\d+))?", v)
    if m and int(m[1]) in KT_ROWS and int(m[2]) in KT_COL_TILES and (m[3] or not m[4]):
        kw = {"kt_rows": int(m[1]), "col_tiles": int(m[2])}
        if m[3]:
            kw["lean"] = True
        if m[4]:
            kw["kconv"] = True
        if m[5]:
            if int(m[5]) not in IPF_KB:
                raise ValueError(f"QSA_FP8_PREFILL_ROUTE: ipf {m[5]} not in {IPF_KB}")
            kw["ipf"] = int(m[5])
        return "ktile", kw
    raise ValueError(f"QSA_FP8_PREFILL_ROUTE: unknown variant {v!r}")


def parse_fp8_prefill_route(spec: str):
    """Rule tuples (page, cmin, cmax, rmin, rmax, kind, kw) or None (= legacy)."""
    spec = FP8_PREFILL_ROUTE_PRESETS.get(spec.strip().lower(), spec)
    if spec is None:
        return None
    rules = []
    for part in [p for p in spec.split(";") if p.strip()]:
        f = [x.strip().lower() for x in part.split(",")]
        if len(f) != 6:
            raise ValueError(f"QSA_FP8_PREFILL_ROUTE rule {part!r}: need 6 fields")
        page = None if f[0] == "*" else int(f[0])
        if page is not None and page not in SUPPORTED_PAGE_SIZES:
            raise ValueError(f"QSA_FP8_PREFILL_ROUTE rule {part!r}: page {page}")
        b = [None if x == "*" else int(x) for x in f[1:5]]
        rules.append((page, b[0], b[1], b[2], b[3]) + _parse_fp8_route_variant(f[5]))
    return tuple(rules)


FP8_PREFILL_ROUTE_SPEC = (
    os.environ.get("QSA_FP8_PREFILL_ROUTE", FP8_PREFILL_ROUTE_DEFAULT).strip().lower()
)
FP8_PREFILL_ROUTE = parse_fp8_prefill_route(FP8_PREFILL_ROUTE_SPEC)


def prefill_fp8_route(
    rows: int, num_requests: int, columns: int, page_size: int = PAGE_SIZE, table=None
):
    """(kind, builder keywords) of a guarded fp8 r32 TDM prefill call: kind 'tdm' (no
    keyword), 'early' ({'early': True}) or 'ktile' (kt_rows / col_tiles [/ lean / kconv /
    ipf]). QSA_FP8_KT / QSA_FP8_EARLY=1 (R2C env) win over the table."""
    if FP8_KT is not None:
        return "ktile", {"kt_rows": FP8_KT[0], "col_tiles": FP8_KT[1]}
    if FP8_R32_EARLY:
        return "early", {"early": True}
    table = FP8_PREFILL_ROUTE if table is None else table
    if table is None or table == "legacy":
        return "tdm", {}
    for page, cmin, cmax, rmin, rmax, kind, kw in table:
        if page is not None and page != int(page_size):
            continue
        if (
            cmin is not None
            and int(columns) < cmin
            or cmax is not None
            and int(columns) > cmax
        ):
            continue
        if (
            rmin is not None
            and int(rows) < rmin
            or rmax is not None
            and int(rows) > rmax
        ):
            continue
        return kind, ({"early": True} if kind == "early" else dict(kw))
    return "tdm", {}


def _resolve_fp8_k_load(rows_per_block: int, k_load: str | None = None):
    """Only full r32 prefill supports TDM; decode/r16 always remains raw."""
    mode = _FP8_K_LOAD if k_load is None else k_load
    if mode not in ("raw", "tdm", "auto"):
        raise ValueError("fp8 k_load must be raw, tdm or auto")
    return "tdm" if rows_per_block == 32 and mode != "raw" else "raw"


def _s32(v):
    """A 32-bit pattern as the signed Python int an i32 constant must carry."""
    return v - (1 << 32) if v >= (1 << 31) else v


# E4M3 -> bf16, exact, two codes per 32-bit op (round G).
#
# Placing an E4M3 code ``b`` into a 16-bit slot as
#
#     f16 bits = (b & 0x80) << 8 | (b & 0x7F) << 7
#
# gives the IEEE half whose sign/exponent/mantissa fields are E4M3's (4-bit
# exponent in the low bits of f16's 5-bit field, 3 mantissa bits on top of
# f16's 10), i.e. exactly ``E4M3(b) * 2**-8`` Ã¢â‚¬â€ for E4M3 normals (f16
# exponent 1..15) and E4M3 subnormals (f16 subnormals ``m * 2**-17``) alike,
# and ``+-0`` for 0x00/0x80.  0x7F/0xFF become ``+-1.875`` (f16 exponent 15
# is an ordinary exponent), i.e. ``+-480`` after the rescale, exactly what the
# scalar decoder ``_e4m3_byte_to_f32`` returns for them (no NaN).  Then
#
#     * ``* 256`` in f16 is exact (every product is in [2**-9, 480], f16
#       normal range, 4 significant bits);
#     * ``f16 -> f32`` is exact;
#     * the f32 has <= 4 significant bits, so its low 16 bits are zero and its
#       high half *is* the bf16 encoding (no rounding anywhere).
#
# Exactness for all 256 codes, in every byte position of a dword, against the
# scalar decoder is checked by ``_g2/test_fp8_wmma_cpu.py`` (pure-Python
# mirror) and by ``_g2/trace_check_cpu.py`` (this source, executed).
# Caveat: E4M3 subnormal codes are f16 *subnormals* before the ``* 256`` Ã¢â‚¬â€ this
# relies on the default FP16 denormal mode (``FLOAT_DENORM_MODE_16_64``
# = preserve, LLVM's AMDGPU default).  A flush-to-zero mode would turn those 14
# codes (|v| <= 7 * 2**-9) into +-0; the root's GPU plan includes a bitwise
# code-sweep against the scalar path to rule that out.
_E4M3_F16_MAG = fx.Int32(0x3F803F80)  # (b & 0x7F) << 7 in both slots
_E4M3_F16_SIGN = fx.Int32(_s32(0x80008000))  # sign bit of both slots
_BF16_LO_MASK = fx.Int32(0xFFFF)
_BF16_HI_MASK = fx.Int32(_s32(0xFFFF0000))


def _e4m3x4_to_bf16x2(x):
    """Four E4M3 codes (one little-endian dword) -> ``(lo, hi)`` packed bf16.

    ``lo = bf16(b0) | bf16(b1) << 16`` and ``hi = bf16(b2) | bf16(b3) << 16``:
    the same element order as two dwords of a bf16 row, so the result drops
    into a WMMA B fragment exactly where the bf16 path's loaded words go.
    """
    x = fx.Int32(x)
    # slot 0 <- byte 0, slot 1 <- byte 2   /   slot 0 <- byte 1, slot 1 <- byte 3
    h_even = ((x << 7) & _E4M3_F16_MAG) | ((x << 8) & _E4M3_F16_SIGN)
    h_odd = ((x >> 1) & _E4M3_F16_MAG) | (x & _E4M3_F16_SIGN)
    h = Vec.from_elements([h_even, h_odd], dtype=fx.Int32).bitcast(fx.Float16)
    f = (h * Vec.filled(4, 256.0, fx.Float16)).to(fx.Float32).bitcast(fx.Int32)
    # f = f32 bits of [b0, b2, b1, b3]; bf16 = the high half (low half is zero)
    lo = ((fx.Int32(f[0]) >> 16) & _BF16_LO_MASK) | (fx.Int32(f[2]) & _BF16_HI_MASK)
    hi = ((fx.Int32(f[1]) >> 16) & _BF16_LO_MASK) | (fx.Int32(f[3]) & _BF16_HI_MASK)
    return lo, hi


def _e4m3x8_to_bf16x4_asm(src0, src1):
    """Eight exact legacy E4M3 codes -> four bf16 pairs on stock gfx1250 (inline asm).

    ROUND-W64FIX2: opt-out only (``QSA_FP8_CVT_ASM=1``); see ``_e4m3x8_to_bf16x4_fast``.
    Native E4M3FN treats 0x7f/0xff as NaN, while this kernel deliberately
    treats them as +/-480. Convert *positive* magnitudes, then packed unsigned
    min with bf16(480): all finite magnitudes are <=448, and every bf16 NaN
    payload is >480. Finally restore the original signs, including -0.
    Direct bf16 conversion has no half-subnormal intermediate; the production
    kernel's FLOAT_DENORM_MODE_16_64=preserve is required for legacy identity.
    No scale or head-sum arithmetic is changed here.
    """
    from flydsl._mlir.dialects import llvm

    src = (fx.Int32(src0), fx.Int32(src1))
    magnitudes = Vec.from_elements(
        [v & fx.Int32(0x7F7F7F7F) for v in src], dtype=fx.Int32
    )
    converted = Vec(
        llvm.inline_asm(
            Vec.make_type(4, fx.Int32),
            [magnitudes.ir_value()],
            # LLVM marks the pk8 destination early-clobber: four output registers
            # must not overlap the two input registers during its blocking cycles.
            "v_cvt_scale_pk8_bf16_fp8 $0, $1, 0x7f",
            "=&v,v",
            has_side_effects=False,
        )
    )
    words = []
    for j in range_constexpr(4):
        magnitude = llvm.inline_asm(
            fx.Int32.ir_type,
            [fx.Int32(converted[j]).ir_value(), fx.Int32(0x43F043F0).ir_value()],
            "v_pk_min_u16 $0, $1, $2",
            "=v,v,s",
            has_side_effects=False,
        )
        selector = "0x01010000" if j % 2 == 0 else "0x03030202"
        sign = llvm.inline_asm(
            fx.Int32.ir_type,
            [src[j // 2].ir_value()],
            "v_perm_b32 $0, $1, $1, " + selector,
            "=v,v",
            has_side_effects=False,
        )
        word = llvm.inline_asm(
            fx.Int32.ir_type,
            [sign, fx.Int32(_s32(0x80008000)).ir_value(), magnitude],
            "v_and_or_b32 $0, $1, $2, $3",
            "=v,v,s,v",
            has_side_effects=False,
        )
        words.append(fx.Int32(word))
    return Vec.from_elements(words, dtype=fx.Int32)


# ROUND-W64FIX2: the pk8 converter is compiler-visible by default.
#
# Root cause (GPU-confirmed, root a08-1 GPU3 2026-11-03: the inline-asm fp8 w64
# decode-group build failed 5/22 cases, the 4-v_nop asm build and the intrinsic
# build passed 22/22): every instruction of ``_e4m3x8_to_bf16x4_asm`` is an
# inline asm statement, which LLVM's hazard recognizer does not treat as a VALU.
# gfx1250 dense bf16 WMMA has *software-enforced* hazards (kernel-wiki
# mi400-wmma-hazards Â§1.2 row B1): a VALU whose vdst overlaps the previous
# WMMA's Matrix A/B (WAR) or Matrix D (WAW), or that reads D (RAW), needs 4
# V_NOPs or independent VALUs in between. Register allocation decided whether
# the unpadded asm landed inside that window (w64: yes -> wrong results; w32 /
# draft / r32: latent violations, correct by luck).
#
# ``_e4m3x8_to_bf16x4_intrin`` emits the same instructions as compiler-visible
# IR (llvm.amdgcn.cvt.scale.pk8.bf16.fp8 with the same scale 0x7f / sel 0,
# llvm.amdgcn.perm, llvm.umin.v2i16, and/or), so GCNHazardRecognizer pads the
# WMMA hazards itself. Bit-identical results: same pk8 conversion,
# perm(x, x, sel) == v_perm_b32 x, x, sel, umin on u16 lanes == v_pk_min_u16,
# (sign & 0x80008000) | mag == v_and_or_b32. Every fp8 WMMA kernel that used the
# asm converter (r32 TDM ``_cvtfast``, draft tdm/raw, decode-group w32/w64)
# calls ``_e4m3x8_to_bf16x4_fast`` -> intrinsic. ``QSA_FP8_CVT_ASM=1`` restores
# the asm converter (debug / A-B only; kernel names gain ``_cvtasm``; fp8
# decode-group w64 then stays disabled because it is GPU-incorrect).
# The ROUND-W64FIX bisect knob QSA_DECG_FP8_W64_DEBUG (asis/nop4/intrin/tdmwait)
# is removed: its question is answered.
_FP8_CVT_ASM = os.environ.get("QSA_FP8_CVT_ASM", "0").strip()
if _FP8_CVT_ASM not in ("0", "1"):
    raise ValueError(f"QSA_FP8_CVT_ASM must be '0' or '1', got {_FP8_CVT_ASM!r}")
_FP8_CVT_ASM = _FP8_CVT_ASM == "1"
_CVT_ASM_TAG = "_cvtasm" if _FP8_CVT_ASM else ""


def _e4m3x8_to_bf16x4_intrin(src0, src1):
    """``_e4m3x8_to_bf16x4_asm`` as compiler-visible intrinsics (no inline asm)."""
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm

    i16x2 = ir.VectorType.get([2], ir.IntegerType.get_signless(16))
    src = (fx.Int32(src0), fx.Int32(src1))
    magnitudes = Vec.from_elements(
        [v & fx.Int32(0x7F7F7F7F) for v in src], dtype=fx.Int32
    )
    bf = llvm.call_intrinsic(
        ir.VectorType.get([8], ir.BF16Type.get()),
        "llvm.amdgcn.cvt.scale.pk8.bf16.fp8",
        [magnitudes.ir_value(), fx.Int32(0x7F).ir_value(), fx.Int32(0).ir_value()],
        [],
        [],
    )
    converted = Vec(llvm.bitcast(Vec.make_type(4, fx.Int32), bf))
    cap = llvm.bitcast(i16x2, fx.Int32(0x43F043F0).ir_value())
    words = []
    for j in range_constexpr(4):
        pair = llvm.bitcast(i16x2, fx.Int32(converted[j]).ir_value())
        magnitude = fx.Int32(
            llvm.bitcast(
                fx.Int32.ir_type,
                llvm.call_intrinsic(i16x2, "llvm.umin.v2i16", [pair, cap], [], []),
            )
        )
        selector = 0x01010000 if j % 2 == 0 else 0x03030202
        x = src[j // 2].ir_value()
        sign = fx.Int32(
            llvm.call_intrinsic(
                fx.Int32.ir_type,
                "llvm.amdgcn.perm",
                [x, x, fx.Int32(selector).ir_value()],
                [],
                [],
            )
        )
        words.append((sign & fx.Int32(_s32(0x80008000))) | magnitude)
    return Vec.from_elements(words, dtype=fx.Int32)


def _e4m3x8_to_bf16x4_fast(src0, src1):
    """The exact pk8 E4M3 -> bf16 converter used by every fp8 WMMA kernel.

    ROUND-W64FIX2: compiler-visible intrinsics by default (WMMA hazards padded by
    LLVM); ``QSA_FP8_CVT_ASM=1`` selects the original inline-asm sequence.
    """
    if _FP8_CVT_ASM:
        return _e4m3x8_to_bf16x4_asm(src0, src1)
    return _e4m3x8_to_bf16x4_intrin(src0, src1)


@cache
def _build_wmma_fp8_kernel(
    divisor: float,
    rows_per_block: int = 16,
    k_load: str | None = None,
    pos64: bool = False,
    page_size: int = PAGE_SIZE,
    early: bool = False,
    kt_rows: int | None = None,
    col_tiles: int = 1,
    trace: int = 0,
    lean: bool = False,
    ipf: int = 0,
    kconv: bool = False,
):
    """fp8 WMMA specialization; r32 shares exact E4M3 B fragments across M tiles.

    ``k_load="ktile"`` (ROUND-R2C, r32-routed TDM) builds the fp8 ``kernel_kt``: the bf16
    ktile geometry (``kt_rows`` 16 | 32 -> 4 | 8 waves, one 4-row M tile per wave, all
    4 x ``col_tiles`` N tiles from one E4M3 LDS slab) with Q, page validity and the
    per-column scales issued before the single TDM wait + barrier.

    ``early=True`` (ROUND-R2C, opt-in; r32 TDM only) builds ``kernel_r32_early``: the
    full-block branch issues Q fragments, the lane columns' page-table entries and the
    per-column fp32 scales BEFORE the TDM stage's wait + barrier (default r32 issues
    all three after it: two more dependent latencies). Same values, same order of
    floating-point operations, same stores.

    The r16 path is unchanged. Full, single-request r32 blocks reuse one K
    gather/conversion per step; mixes and tails use the exact r16 helpers.
    Both paths apply each column's fp32 scale after GEMM, before ReLU and
    the left-to-right head sum, in the scalar fp8 order.
    ``pos64=True`` (ROUND-NATIVE64) reads query_positions as int64.
    """
    ld_pos, _, qp_bytes, qp_tag = _pos_loaders(pos64)
    P = _check_page_size(page_size)
    if rows_per_block not in (16, 32):
        raise ValueError("WMMA rows_per_block must be 16 or 32")
    ktile = k_load == "ktile"  # ROUND-R2C
    if ktile:
        if rows_per_block != 32 or _ACCESS != "raw" or early:
            raise ValueError("the fp8 ktile kernel is r32-routed TDM (raw access) only")
        kt_grid(32, 64, kt_rows, col_tiles)  # validates the knobs
        k_load = "tdm"
    elif kt_rows is not None or col_tiles != 1:
        raise ValueError("kt_rows / col_tiles apply to k_load='ktile' only")
    kt_waves = (int(kt_rows) // WMMA_ROWS_PER_WAVE) if ktile else WMMA_WAVES
    kt_threads = kt_waves * WMMA_WAVE
    k_load = _resolve_fp8_k_load(rows_per_block, k_load)
    use_tdm = k_load == "tdm" and _ACCESS == "raw"
    guard_load = _ld_uniform_i32 if use_tdm else _ld_i32
    early = bool(early)
    if early and not (use_tdm and rows_per_block == 32):
        raise ValueError("fp8 early (ROUND-R2C) is r32 TDM only")
    # ROUND-R2D opt-in knobs (defaults emit nothing new): trace, lean / ipf (ktile only).
    trace = _r2d_check(
        trace, lean, ipf, ktile, use_tdm and rows_per_block == 32 and not early
    )
    if kconv and not (lean and use_tdm and _FP8_CVT == "fast"):
        raise ValueError(
            "kconv (ROUND-R2D) applies to the lean fp8 ktile (fast cvt) only"
        )
    kconv = bool(kconv)
    trs = {}
    ld_uniform_pos = _pos_loaders(pos64)[1]

    def _pair_storage_raw(KV, kv_rsrc):
        return kv_rsrc

    def _pair_storage_tdm(KV, kv_rsrc):
        return (
            fx.SharedAllocator(base_alignment=128, static=True)
            .allocate(fx.Array[fx.Int32, TDM_FP8_K_TILE_BYTES // 4, 128])
            .peek()
            .ptr
        )

    def _pair_stage_raw(KV, ptr, pt_rsrc, t2r_rsrc, rblk, col0, width, num_pages):
        pass

    def _pair_stage_tdm(KV, ptr, pt_rsrc, t2r_rsrc, rblk, col0, width, num_pages):
        # Workgroup-uniform full single-request branch only. Tensor ignores EXEC.
        wave_id = fx.Int32(fx.rocdl.wave_id())
        req = _ld_uniform_i32(t2r_rsrc, rblk * 32 * 4)
        for group in range_constexpr(2):
            local = wave_id * 16 + group * 8
            if P == 4:
                row_ids = []
                for page in range_constexpr(2):
                    logical = (col0 + local) // 4 + page
                    logical_ok = logical < width
                    safe_logical = logical_ok.select(logical, 0)
                    phys = _ld_uniform_i32(pt_rsrc, (req * width + safe_logical) * 4)
                    valid = logical_ok & (phys >= 0) & (phys < num_pages)
                    phys = valid.select(phys, 0)
                    for row in range_constexpr(4):
                        row_ids.append(phys * 4 + row)
            elif (
                P == 16
            ):  # P = 16: wave w's 16 rows = page col0/16 + w; group = half page
                row_ids = _p16_row_ids(
                    pt_rsrc, req, width, num_pages, col0 // 16 + wave_id, group * 8
                )
            else:  # P = 8: wave w's 16 rows = pages col0/8 + 2w, +1; group = one page
                row_ids = _p16_row_ids(
                    pt_rsrc,
                    req,
                    width,
                    num_pages,
                    col0 // 8 + wave_id * 2 + group,
                    0,
                    None,
                    8,
                )
            _tdm_fp8_k_gather(
                KV, ptr, row_ids, num_pages, local * TDM_FP8_K_ROW_BYTES, P
            )
        fx.rocdl.tdm_ops.tensor_wait(0)
        fx.gpu.barrier()  # producer completion + cross-wave consumer rendezvous
        # This one-shot tile is never overwritten; no LDS reuse loop exists.

    def _pair_base_raw(phys, off, c, kgrp):
        return phys * page_bytes + off * key_row_bytes + kgrp * 8

    def _pair_base_tdm(phys, off, c, kgrp):
        return (c % WMMA_COLS) * TDM_FP8_K_ROW_BYTES

    def _pair_frag_raw(ptr, byte_off, kgrp):
        return _kfrag_fp8(ptr, byte_off)

    def _pair_frag_tdm(ptr, byte_off, kgrp):
        # Each 16B-aligned read holds both k-halves. Pick this lane's 8B half
        # before exactly the same E4M3 conversion as the raw B fragment helper.
        lo = Vec(
            fx.ptr_load(ptr + byte_off // 4, result_type=Vec.make_type(4, fx.Int32))
        )
        hi = Vec(
            fx.ptr_load(
                ptr + (byte_off + 16) // 4, result_type=Vec.make_type(4, fx.Int32)
            )
        )
        words = []
        for j in range_constexpr(4):
            vec = lo if j < 2 else hi
            idx = j % 2
            src = (kgrp == 0).select(fx.Int32(vec[idx]), fx.Int32(vec[idx + 2]))
            w0, w1 = _e4m3x4_to_bf16x2(src)
            words.append(w0)
            words.append(w1)
        return Vec.from_elements(words, dtype=fx.Int32)

    def _pair_frag_tdm_fast(ptr, byte_off, kgrp):
        # Exactly the champion addresses and lane-half selection; only conversion
        # widens from four to eight codes. Result is the identical eight-i32 B.
        lo = Vec(
            fx.ptr_load(ptr + byte_off // 4, result_type=Vec.make_type(4, fx.Int32))
        )
        hi = Vec(
            fx.ptr_load(
                ptr + (byte_off + 16) // 4, result_type=Vec.make_type(4, fx.Int32)
            )
        )
        words = []
        for vec in (lo, hi):
            src0 = (kgrp == 0).select(fx.Int32(vec[0]), fx.Int32(vec[2]))
            src1 = (kgrp == 0).select(fx.Int32(vec[1]), fx.Int32(vec[3]))
            converted = _e4m3x8_to_bf16x4_fast(src0, src1)
            for j in range_constexpr(4):
                words.append(fx.Int32(converted[j]))
        return Vec.from_elements(words, dtype=fx.Int32)

    fast_cvt = use_tdm and _FP8_CVT == "fast"
    pair_storage = _pair_storage_tdm if use_tdm else _pair_storage_raw
    pair_stage = _pair_stage_tdm if use_tdm else _pair_stage_raw
    pair_base = _pair_base_tdm if use_tdm else _pair_base_raw
    pair_frag = _pair_frag_tdm if use_tdm else _pair_frag_raw
    if fast_cvt:
        pair_frag = _pair_frag_tdm_fast
    inv_divisor = 1.0 / divisor
    q_row_bytes = NUM_HEADS * HEAD_DIM * 2  # 1024 B
    head_bytes = HEAD_DIM * 2  # 256 B
    key_row_bytes = HEAD_DIM  # 128 B (E4M3)
    page_bytes = P * key_row_bytes  # 512 B
    kstep_bytes = WMMA_K * 2  # 64 B of one Q row per WMMA step
    kstep_k_bytes = WMMA_K  # 32 B of one fp8 key row per step
    kernel_name = (
        f"qsa_paged_mqa_logits_page4_gfx1250_fp8_wmma"
        f"_D{HEAD_DIM}_H{NUM_HEADS}_P{P}"
        f"_r{rows_per_block}_c{WMMA_COLS}_flydsl"
        + ("_tdm" if use_tdm else "")
        + ("_cvtfast" + _CVT_ASM_TAG if fast_cvt else "")
        + ("_early" if early else "")
        + (f"_kt{kt_rows}x{col_tiles}" if ktile else "")
        + _r2d_tag(lean, ipf)
        + ("_kc" if kconv else "")
        + (f"_trace{trace}" if trace else "")
        + qp_tag
    )

    # ---- helpers: plain Python at trace time (no `if`/runtime `for` inside; the
    # AST rewriter only rewrites the kernel body, and the kernel body only calls
    # these through bare names so no outer value becomes loop/if-carried) -----
    def _i01(cond):
        return cond.select(fx.Int32(1), fx.Int32(0))

    def _frag(rsrc, byte_off):
        """One WMMA operand fragment: 2 x 16 B, 32 B apart -> 8 x i32 (16 bf16)."""
        lo = _ld_vec(rsrc, byte_off, 4, fx.Int32)
        hi = _ld_vec(rsrc, byte_off + 32, 4, fx.Int32)
        return lo.shuffle(hi, list(range(8)))

    def _kfrag_fp8(rsrc, byte_off):
        """B fragment from E4M3: 2 x 8 B, 16 B apart -> 8 x i32 (16 bf16, exact).

        Same k positions as ``_frag`` on a bf16 row (16 dims = 32 B there,
        16 B here): v = 0..7 from ``byte_off``, v = 8..15 from ``byte_off + 16``.
        """
        lo = _ld_vec(rsrc, byte_off, 2, fx.Int32)
        hi = _ld_vec(rsrc, byte_off + 16, 2, fx.Int32)
        words = []
        for j in range_constexpr(4):
            src = lo[j] if j < 2 else hi[j - 2]
            w0, w1 = _e4m3x4_to_bf16x2(src)
            words.append(w0)
            words.append(w1)
        return Vec.from_elements(words, dtype=fx.Int32)

    def _rmem8(vec):
        t = fx.make_rmem_tensor(8, fx.Int32)
        t.store(vec)
        return t

    def _rows_meta(t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests):
        out = []
        for i in range_constexpr(WMMA_ROWS_PER_WAVE):
            row = t0 + i
            row_ok = row < rows
            safe_row = _imin(row, rows - 1)  # loads stay in bounds
            req = _ld_i32(t2r_rsrc, safe_row * 4)
            req_ok = (req >= 0) & (req < num_requests)
            safe_req = req_ok.select(req, 0)
            ctx = _ld_i32(cl_rsrc, safe_req * 4)
            pos = ld_pos(qp_rsrc, safe_row)
            head = _imax(_imin(pos + 1, ctx), 0)
            vis = req_ok.select(head // COMPRESS_RATIO, 0)
            out.append(_RowMeta(row, row_ok, req, req_ok, safe_req, vis))
        return out

    def _pass_plan(meta, rows, kgrp):
        """Request-dedup passes; lane rows are a = 2*kgrp, b = 2*kgrp + 1."""
        g0 = kgrp == 0
        plans = []
        for p in range_constexpr(WMMA_PASSES):
            mp = meta[p]
            active = mp.row_ok
            for q in range_constexpr(p):
                mq = meta[q]
                active = active & ((mq.row >= rows) | (mq.req != mp.req))
            own = [
                _i01(meta[i].row_ok & (meta[i].req == mp.req))
                for i in range_constexpr(WMMA_ROWS_PER_WAVE)
            ]
            vmax = fx.Int32(0)
            for i in range_constexpr(WMMA_ROWS_PER_WAVE):
                vmax = _imax(vmax, (own[i] != 0).select(meta[i].vis, 0))
            own_a = g0.select(own[0], own[2]) != 0
            own_b = g0.select(own[1], own[3]) != 0
            vis_a = g0.select(meta[0].vis, meta[2].vis)
            vis_b = g0.select(meta[1].vis, meta[3].vis)
            plans.append(
                _Pass(
                    active,
                    vmax,
                    mp.safe_req,
                    mp.req_ok,
                    own_a,
                    own_b,
                    vis_a,
                    vis_b,
                    kgrp,
                )
            )
        return plans

    def _pick_vis(lane, meta):
        """visible[t0 + lane] for lanes 0..3 (lanes >= 3 get row 3's; unused)."""
        v = meta[WMMA_ROWS_PER_WAVE - 1].vis
        for i in range_constexpr(WMMA_ROWS_PER_WAVE - 2, -1, -1):
            v = (lane == i).select(meta[i].vis, v)
        return v

    def _lane_cols(col0, n_in):
        return [col0 + nt * WMMA_N + n_in for nt in range_constexpr(WMMA_N_TILES)]

    def _store_offsets(t0, kgrp, cols, rows, columns):
        """Clamped byte offsets of the lane's two outputs per tile (always in plane)."""
        last = rows * columns - 1
        row_a = _imin(t0 + 2 * kgrp, rows - 1)
        row_b = _imin(t0 + 2 * kgrp + 1, rows - 1)
        out = []
        for c in cols:
            out.append(_imin(row_a * columns + c, last) * 4)
            out.append(_imin(row_b * columns + c, last) * 4)
        return out

    def _q_frags(q_rsrc, t0, rows, n_in, kgrp):
        q_row = _imin(t0 + n_in // NUM_HEADS, rows - 1)
        head = n_in % NUM_HEADS
        base = q_row * q_row_bytes + head * head_bytes + kgrp * 16
        return [
            _frag(q_rsrc, base + s * kstep_bytes) for s in range_constexpr(WMMA_K_STEPS)
        ]

    def _tile(atom, kv_rsrc, ks_rsrc, pt_rsrc, qf, pp, c, kgrp, width, num_pages):
        """Scores of the lane's rows a/b at column c (one 16x16x128 WMMA tile)."""
        logical = c // P
        off = c % P
        logical_ok = logical < width
        safe_logical = logical_ok.select(logical, 0)
        phys = _ld_i32(pt_rsrc, (pp.safe_req * width + safe_logical) * 4)
        page_ok = pp.req_ok & logical_ok & (phys >= 0) & (phys < num_pages)
        phys = page_ok.select(phys, 0)
        k_base = phys * page_bytes + off * key_row_bytes + kgrp * 8
        kf = [
            _kfrag_fp8(kv_rsrc, k_base + s * kstep_k_bytes)
            for s in range_constexpr(WMMA_K_STEPS)
        ]
        # per-token fp32 scale of column c: same masked phys as the K rows
        scale = _ld_f32(ks_rsrc, (phys * P + off) * 4)
        acc_t = fx.make_rmem_tensor(8, fx.Float32)
        acc_t.store(Vec.filled(8, 0.0, fx.Float32))
        for s in range_constexpr(WMMA_K_STEPS):
            # D[m = 4*row + head, n = column] += Q[m, k] * K[n, k]
            fx.gemm(atom, acc_t, _rmem8(qf[s]), _rmem8(kf[s]), acc_t)
        acc = acc_t.load()
        zero = fx.Float32(0.0)
        # scalar fp8 order: relu(acc_h * scale) summed over h = 0..3, then / divisor
        sa = (
            (fx.Float32(acc[0]) * scale).maximumf(zero)
            + (fx.Float32(acc[1]) * scale).maximumf(zero)
            + (fx.Float32(acc[2]) * scale).maximumf(zero)
            + (fx.Float32(acc[3]) * scale).maximumf(zero)
        )
        sb = (
            (fx.Float32(acc[4]) * scale).maximumf(zero)
            + (fx.Float32(acc[5]) * scale).maximumf(zero)
            + (fx.Float32(acc[6]) * scale).maximumf(zero)
            + (fx.Float32(acc[7]) * scale).maximumf(zero)
        )
        return page_ok, sa * inv_divisor, sb * inv_divisor

    def _pass_values(
        atom, kv_rsrc, ks_rsrc, pt_rsrc, qf, pp, cols, offs, columns, width, num_pages
    ):
        """[(store_ok, byte_off, value)] for the lane's 2 x N_TILES outputs."""
        neg_inf = fx.Float32(_NEG_INF)
        out = []
        for nt in range_constexpr(WMMA_N_TILES):
            c = cols[nt]
            page_ok, sa, sb = _tile(
                atom, kv_rsrc, ks_rsrc, pt_rsrc, qf, pp, c, pp.kgrp, width, num_pages
            )
            col_ok = c < columns
            va = (page_ok & (c < pp.vis_a)).select(sa, neg_inf)
            vb = (page_ok & (c < pp.vis_b)).select(sb, neg_inf)
            out.append((pp.own_a & col_ok, offs[2 * nt], va))
            out.append((pp.own_b & col_ok, offs[2 * nt + 1], vb))
        return out

    def _pass_masks(pp, cols, offs, columns):
        """[(store_ok, byte_off)] for a pass whose rows see none of these columns."""
        out = []
        for nt in range_constexpr(WMMA_N_TILES):
            col_ok = cols[nt] < columns
            out.append((pp.own_a & col_ok, offs[2 * nt]))
            out.append((pp.own_b & col_ok, offs[2 * nt + 1]))
        return out

    def _r32_guard(t2r_rsrc, base, rows, num_requests):
        """Uniform guard: full 32-row block, all rows of one valid request."""
        req = guard_load(t2r_rsrc, _imin(base, rows - 1) * 4)
        reuse = (base + 32 <= rows) & (req >= 0) & (req < num_requests)
        for i in range_constexpr(1, 32):
            other = guard_load(t2r_rsrc, _imin(base + i, rows - 1) * 4)
            reuse = reuse & (other == req)
        return reuse

    def _tile_pair(
        atom, kv_rsrc, ks_rsrc, pt_rsrc, qfa, qfb, pp, c, kgrp, width, num_pages
    ):
        """Two M16 accumulators, one exact E4M3 gather/conversion per K step."""
        logical = c // P
        off = c % P
        logical_ok = logical < width
        safe_logical = logical_ok.select(logical, 0)
        phys = _ld_i32(pt_rsrc, (pp.safe_req * width + safe_logical) * 4)
        page_ok = pp.req_ok & logical_ok & (phys >= 0) & (phys < num_pages)
        phys = page_ok.select(phys, 0)
        k_base = pair_base(phys, off, c, kgrp)
        scale = _ld_f32(ks_rsrc, (phys * P + off) * 4)
        acc_a = fx.make_rmem_tensor(8, fx.Float32)
        acc_b = fx.make_rmem_tensor(8, fx.Float32)
        acc_a.store(Vec.filled(8, 0.0, fx.Float32))
        acc_b.store(Vec.filled(8, 0.0, fx.Float32))
        for s in range_constexpr(WMMA_K_STEPS):
            kf = _rmem8(pair_frag(kv_rsrc, k_base + s * kstep_k_bytes, kgrp))
            fx.gemm(atom, acc_a, _rmem8(qfa[s]), kf, acc_a)
            fx.gemm(atom, acc_b, _rmem8(qfb[s]), kf, acc_b)
        a = acc_a.load()
        b = acc_b.load()
        zero = fx.Float32(0.0)
        scores = []
        for acc in (a, b):
            # Exact r16/scalar order: scale each dot, ReLU, left-to-right sum.
            sa = (
                (fx.Float32(acc[0]) * scale).maximumf(zero)
                + (fx.Float32(acc[1]) * scale).maximumf(zero)
                + (fx.Float32(acc[2]) * scale).maximumf(zero)
                + (fx.Float32(acc[3]) * scale).maximumf(zero)
            )
            sb = (
                (fx.Float32(acc[4]) * scale).maximumf(zero)
                + (fx.Float32(acc[5]) * scale).maximumf(zero)
                + (fx.Float32(acc[6]) * scale).maximumf(zero)
                + (fx.Float32(acc[7]) * scale).maximumf(zero)
            )
            scores.append(sa * inv_divisor)
            scores.append(sb * inv_divisor)
        return page_ok, scores

    def _pair_values(
        atom,
        kv_rsrc,
        ks_rsrc,
        pt_rsrc,
        qfa,
        qfb,
        ppa,
        ppb,
        cols,
        offsa,
        offsb,
        columns,
        width,
        num_pages,
    ):
        out = []
        neg_inf = fx.Float32(_NEG_INF)
        for nt in range_constexpr(WMMA_N_TILES):
            c = cols[nt]
            page_ok, scores = _tile_pair(
                atom,
                kv_rsrc,
                ks_rsrc,
                pt_rsrc,
                qfa,
                qfb,
                ppa,
                c,
                ppa.kgrp,
                width,
                num_pages,
            )
            for mi in range_constexpr(2):
                pp = (ppa, ppb)[mi]
                offs = (offsa, offsb)[mi]
                col_ok = c < columns
                va = (page_ok & (c < pp.vis_a)).select(scores[2 * mi], neg_inf)
                vb = (page_ok & (c < pp.vis_b)).select(scores[2 * mi + 1], neg_inf)
                out.append((pp.own_a & col_ok, offs[2 * nt], va))
                out.append((pp.own_b & col_ok, offs[2 * nt + 1], vb))
        return out

    # ---- ROUND-R2C early helpers (kernel_r32_early only) ----------------------
    def _page_scale_pre(pt_rsrc, ks_rsrc, pp, cols, width, num_pages):
        """[(page_ok, scale)] per N tile: `_tile_pair`'s page-table + scale loads."""
        out = []
        for nt in range_constexpr(WMMA_N_TILES):
            c = cols[nt]
            logical = c // P
            off = c % P
            logical_ok = logical < width
            safe_logical = logical_ok.select(logical, 0)
            phys = _ld_i32(pt_rsrc, (pp.safe_req * width + safe_logical) * 4)
            page_ok = pp.req_ok & logical_ok & (phys >= 0) & (phys < num_pages)
            phys = page_ok.select(phys, 0)
            out.append((page_ok, _ld_f32(ks_rsrc, (phys * P + off) * 4)))
        return out

    def _tile_pair_pre(atom, kptr, qfa, qfb, c, kgrp, scale):
        """`_tile_pair` (TDM) with page_ok / scale prefetched: same WMMA + epilogue."""
        k_base = pair_base(None, None, c, kgrp)
        acc_a = fx.make_rmem_tensor(8, fx.Float32)
        acc_b = fx.make_rmem_tensor(8, fx.Float32)
        acc_a.store(Vec.filled(8, 0.0, fx.Float32))
        acc_b.store(Vec.filled(8, 0.0, fx.Float32))
        for s in range_constexpr(WMMA_K_STEPS):
            kf = _rmem8(pair_frag(kptr, k_base + s * kstep_k_bytes, kgrp))
            fx.gemm(atom, acc_a, _rmem8(qfa[s]), kf, acc_a)
            fx.gemm(atom, acc_b, _rmem8(qfb[s]), kf, acc_b)
        a = acc_a.load()
        b = acc_b.load()
        zero = fx.Float32(0.0)
        scores = []
        for acc in (a, b):
            sa = (
                (fx.Float32(acc[0]) * scale).maximumf(zero)
                + (fx.Float32(acc[1]) * scale).maximumf(zero)
                + (fx.Float32(acc[2]) * scale).maximumf(zero)
                + (fx.Float32(acc[3]) * scale).maximumf(zero)
            )
            sb = (
                (fx.Float32(acc[4]) * scale).maximumf(zero)
                + (fx.Float32(acc[5]) * scale).maximumf(zero)
                + (fx.Float32(acc[6]) * scale).maximumf(zero)
                + (fx.Float32(acc[7]) * scale).maximumf(zero)
            )
            scores.append(sa * inv_divisor)
            scores.append(sb * inv_divisor)
        return scores

    # ---- ROUND-R2C fp8 ktile helpers (kernel_kt only) ---------------------------
    def _kt8_storage():
        return (
            fx.SharedAllocator(base_alignment=128, static=True)
            .allocate(fx.Array[fx.Int32, col_tiles * TDM_FP8_K_TILE_BYTES // 4, 128])
            .peek()
            .ptr
        )

    def _kt8_guard(t2r_rsrc, base, rows, num_requests):
        """SGPR (request, ok): full kt_rows-row block, every row of one valid request."""
        req = _ld_uniform_i32(t2r_rsrc, _imin(base, rows - 1) * 4)
        ok = (base + kt_rows <= rows) & (req >= 0) & (req < num_requests)
        for i in range_constexpr(1, kt_rows):
            other = _ld_uniform_i32(t2r_rsrc, _imin(base + i, rows - 1) * 4)
            ok = ok & (other == req)
        return req, ok

    def _kt8_issue(KV, ptr, pt_rsrc, req, col0, width, num_pages, buf):
        """kt_waves waves gather the 64-column E4M3 tile at col0 into buffer `buf`:
        wave w owns columns col0 + w * (64 / waves) .. (8-row gathers, SGPR row ids)."""
        wave_id = fx.Int32(fx.rocdl.wave_id())
        per = WMMA_COLS // kt_waves  # 16 (4 waves) or 8 (8 waves)
        for group in range_constexpr(per // 8):
            local = wave_id * per + group * 8
            if P == 4:
                row_ids = []
                for page in range_constexpr(2):
                    logical = (col0 + local) // 4 + page
                    logical_ok = logical < width
                    safe_logical = logical_ok.select(logical, 0)
                    phys = _ld_uniform_i32(pt_rsrc, (req * width + safe_logical) * 4)
                    valid = logical_ok & (phys >= 0) & (phys < num_pages)
                    phys = valid.select(phys, 0)
                    for row in range_constexpr(4):
                        row_ids.append(phys * 4 + row)
            elif P == 16:  # 8 | 16: the group is one half of page (col0 + local) / 16
                row_ids = _p16_row_ids(
                    pt_rsrc, req, width, num_pages, (col0 + local) // 16, local % 16
                )
            else:  # P = 8: the group IS page (col0 + local) / 8
                row_ids = _p16_row_ids(
                    pt_rsrc, req, width, num_pages, (col0 + local) // 8, 0, None, 8
                )
            _tdm_fp8_k_gather(
                KV,
                ptr,
                row_ids,
                num_pages,
                buf * TDM_FP8_K_TILE_BYTES + local * TDM_FP8_K_ROW_BYTES,
                P,
            )

    def _kt8_pre(pt_rsrc, ks_rsrc, req, req_ok, slab0, n_in, width, num_pages):
        """[[(page_ok, scale)] per N tile] per K tile: `_tile_pair`'s page + scale loads."""
        out = []
        for ct in range_constexpr(col_tiles):
            ps = []
            for c in _lane_cols(slab0 + ct * WMMA_COLS, n_in):
                logical = c // P
                off = c % P
                logical_ok = logical < width
                safe_logical = logical_ok.select(logical, 0)
                phys = _ld_i32(pt_rsrc, (req * width + safe_logical) * 4)
                page_ok = req_ok & logical_ok & (phys >= 0) & (phys < num_pages)
                phys = page_ok.select(phys, 0)
                ps.append((page_ok, _ld_f32(ks_rsrc, (phys * P + off) * 4)))
            out.append(ps)
        return out

    def _kt8_tile(atom, kbuf, qf, c, kgrp, scale):
        k_base = pair_base(None, None, c, kgrp)
        acc_t = fx.make_rmem_tensor(8, fx.Float32)
        acc_t.store(Vec.filled(8, 0.0, fx.Float32))
        for s in range_constexpr(WMMA_K_STEPS):
            kf = _rmem8(pair_frag(kbuf, k_base + s * kstep_k_bytes, kgrp))
            fx.gemm(atom, acc_t, _rmem8(qf[s]), kf, acc_t)
        acc = acc_t.load()
        zero = fx.Float32(0.0)
        sa = (
            (fx.Float32(acc[0]) * scale).maximumf(zero)
            + (fx.Float32(acc[1]) * scale).maximumf(zero)
            + (fx.Float32(acc[2]) * scale).maximumf(zero)
            + (fx.Float32(acc[3]) * scale).maximumf(zero)
        )
        sb = (
            (fx.Float32(acc[4]) * scale).maximumf(zero)
            + (fx.Float32(acc[5]) * scale).maximumf(zero)
            + (fx.Float32(acc[6]) * scale).maximumf(zero)
            + (fx.Float32(acc[7]) * scale).maximumf(zero)
        )
        return sa * inv_divisor, sb * inv_divisor

    def _kt8_values(atom, kbuf, qf, pp, cols, offs, columns, pre):
        neg_inf = fx.Float32(_NEG_INF)
        out = []
        for nt in range_constexpr(WMMA_N_TILES):
            c = cols[nt]
            page_ok, scale = pre[nt]
            sa, sb = _kt8_tile(atom, kbuf, qf, c, pp.kgrp, scale)
            col_ok = c < columns
            va = (page_ok & (c < pp.vis_a)).select(sa, neg_inf)
            vb = (page_ok & (c < pp.vis_b)).select(sb, neg_inf)
            out.append((pp.own_a & col_ok, offs[2 * nt], va))
            out.append((pp.own_b & col_ok, offs[2 * nt + 1], vb))
        return out

    def _pair_values_pre(
        atom, kptr, qfa, qfb, ppa, ppb, cols, offsa, offsb, columns, pre
    ):
        out = []
        neg_inf = fx.Float32(_NEG_INF)
        for nt in range_constexpr(WMMA_N_TILES):
            c = cols[nt]
            page_ok, scale = pre[nt]
            scores = _tile_pair_pre(atom, kptr, qfa, qfb, c, ppa.kgrp, scale)
            for mi in range_constexpr(2):
                pp = (ppa, ppb)[mi]
                offs = (offsa, offsb)[mi]
                col_ok = c < columns
                va = (page_ok & (c < pp.vis_a)).select(scores[2 * mi], neg_inf)
                vb = (page_ok & (c < pp.vis_b)).select(scores[2 * mi + 1], neg_inf)
                out.append((pp.own_a & col_ok, offs[2 * nt], va))
                out.append((pp.own_b & col_ok, offs[2 * nt + 1], vb))
        return out

    @flyc.kernel(name=kernel_name, known_block_size=[WMMA_THREADS, 1, 1])
    def kernel(
        Q,  # uint8 view of [rows, 4, 128] bf16
        KV,  # uint8 view of [pages, 4, 1, 128] fp8 E4M3
        KS,  # fp32 [pages, 4, 1] per-token scale
        PT,  # int32 [requests, width]
        T2R,  # int32 [rows]
        QP,  # int32 [rows]
        CL,  # int32 [requests]
        LOGITS,  # fp32 [rows, columns]
        VIS,  # int32 [rows]
        RST,  # int32 [rows]
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        chunk = fx.Int32(fx.block_idx.x)
        rblk = fx.Int32(fx.block_idx.y)
        wave = tid // WMMA_WAVE
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N  # A: slot m / B, C: column n
        kgrp = lane // WMMA_N  # k-half of the operands, row pair of C
        t0 = (rblk * WMMA_WAVES + wave) * WMMA_ROWS_PER_WAVE
        col0 = chunk * WMMA_COLS

        # One pinned descriptor per plane, true extents in i64 (see `_plane`).
        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        ks_rsrc = _plane(KS, _plane_bytes(num_pages, P * 4))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))

        meta = _rows_meta(t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests)
        plans = _pass_plan(meta, rows, kgrp)

        # Row metadata: lanes 0..3 of each wave of column-block 0, one row each.
        # Predicated store at a clamped in-plane offset (6c272c3c contract).
        my_vis = _pick_vis(lane, meta)
        my_row = t0 + _imin(lane, WMMA_ROWS_PER_WAVE - 1)
        meta_off = _imin(my_row, rows - 1) * 4
        writer = (chunk == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)
        if writer:
            _st_i32(vis_rsrc, meta_off, my_vis)
            _st_i32(rst_rsrc, meta_off, fx.Int32(0))

        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        cols = _lane_cols(col0, n_in)
        offs = _store_offsets(t0, kgrp, cols, rows, columns)
        neg_inf = fx.Float32(_NEG_INF)
        live = t0 < rows
        if live:
            qf = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
            for p in range_constexpr(WMMA_PASSES):
                pp = plans[p]
                if pp.active:
                    if col0 < pp.vmax:
                        res = _pass_values(
                            atom,
                            kv_rsrc,
                            ks_rsrc,
                            pt_rsrc,
                            qf,
                            pp,
                            cols,
                            offs,
                            columns,
                            width,
                            num_pages,
                        )
                        for j1 in range_constexpr(2 * WMMA_N_TILES):
                            if res[j1][0]:
                                _st_f32(logits_rsrc, res[j1][1], res[j1][2])
                    if col0 >= pp.vmax:
                        msk = _pass_masks(pp, cols, offs, columns)
                        for j2 in range_constexpr(2 * WMMA_N_TILES):
                            if msk[j2][0]:
                                _st_f32(logits_rsrc, msk[j2][1], neg_inf)

    @flyc.kernel(name=kernel_name, known_block_size=[WMMA_THREADS, 1, 1])
    def kernel_r32(
        Q,  # uint8 view of [rows, 4, 128] bf16
        KV,  # uint8 view of [pages, 4, 1, 128] fp8 E4M3
        KS,  # fp32 [pages, 4, 1] per-token scale
        PT,  # int32 [requests, width]
        T2R,  # int32 [rows]
        QP,  # int32 [rows]
        CL,  # int32 [requests]
        LOGITS,  # fp32 [rows, columns]
        VIS,  # int32 [rows]
        RST,  # int32 [rows]
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        chunk = fx.Int32(fx.block_idx.x)
        rblk = fx.Int32(fx.block_idx.y)
        wave = tid // WMMA_WAVE
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N  # A: slot m / B, C: column n
        kgrp = lane // WMMA_N  # k-half of the operands, row pair of C
        t0 = (rblk * WMMA_WAVES + wave) * (2 * WMMA_ROWS_PER_WAVE)
        col0 = chunk * WMMA_COLS
        _tr_begin(
            trs,
            trace,
            LOGITS,
            rows,
            columns,
            (columns + WMMA_COLS - 1) // WMMA_COLS,
            (rows + 31) // 32,
            WMMA_WAVES,
            chunk,
            rblk,
        )

        # One pinned descriptor per plane, true extents in i64 (see `_plane`).
        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        ks_rsrc = _plane(KS, _plane_bytes(num_pages, P * 4))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))

        metas = [
            _rows_meta(
                t2r_rsrc,
                qp_rsrc,
                cl_rsrc,
                t0 + mi * WMMA_ROWS_PER_WAVE,
                rows,
                num_requests,
            )
            for mi in range_constexpr(2)
        ]
        plans = [_pass_plan(metas[mi], rows, kgrp) for mi in range_constexpr(2)]
        for mi in range_constexpr(2):
            my_vis = _pick_vis(lane, metas[mi])
            my_row = t0 + mi * WMMA_ROWS_PER_WAVE + _imin(lane, WMMA_ROWS_PER_WAVE - 1)
            meta_off = _imin(my_row, rows - 1) * 4
            writer = (chunk == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)
            if writer:
                _st_i32(vis_rsrc, meta_off, my_vis)
                _st_i32(rst_rsrc, meta_off, fx.Int32(0))
        _tr_mark(trs, 4, True)

        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        cols = _lane_cols(col0, n_in)
        offs = [
            _store_offsets(t0 + mi * WMMA_ROWS_PER_WAVE, kgrp, cols, rows, columns)
            for mi in range_constexpr(2)
        ]
        neg_inf = fx.Float32(_NEG_INF)
        kptr = pair_storage(KV, kv_rsrc)
        reuse = _r32_guard(t2r_rsrc, rblk * 32, rows, num_requests)
        _tr_mark(trs, 5, True)
        if reuse:
            pair_stage(KV, kptr, pt_rsrc, t2r_rsrc, rblk, col0, width, num_pages)
            _tr_mark(trs, 9)  # ROUND-R2D: stage = TDM issue + wait + barrier
            _tr_mark_rt(trs, 14)
            ppa = plans[0][0]
            ppb = plans[1][0]
            vmax = _imax(ppa.vmax, ppb.vmax)
            if col0 < vmax:
                qfa = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
                qfb = _q_frags(q_rsrc, t0 + WMMA_ROWS_PER_WAVE, rows, n_in, kgrp)
                _tr_mark(trs, 7, True)
                res = _pair_values(
                    atom,
                    kptr,
                    ks_rsrc,
                    pt_rsrc,
                    qfa,
                    qfb,
                    ppa,
                    ppb,
                    cols,
                    offs[0],
                    offs[1],
                    columns,
                    width,
                    num_pages,
                )
                for j1 in range_constexpr(4 * WMMA_N_TILES):
                    if res[j1][0]:
                        _st_f32(logits_rsrc, res[j1][1], res[j1][2])
            if col0 >= vmax:
                for mi in range_constexpr(2):
                    msk = _pass_masks(plans[mi][0], cols, offs[mi], columns)
                    for j2 in range_constexpr(2 * WMMA_N_TILES):
                        if msk[j2][0]:
                            _st_f32(logits_rsrc, msk[j2][1], neg_inf)
            _tr_mark(trs, 10)
        if reuse == 0:
            # Exact r16 per-M-tile path for mixed requests and incomplete blocks.
            for mi in range_constexpr(2):
                subt0 = t0 + mi * WMMA_ROWS_PER_WAVE
                live = subt0 < rows
                if live:
                    qf = _q_frags(q_rsrc, subt0, rows, n_in, kgrp)
                    for p in range_constexpr(WMMA_PASSES):
                        pp = plans[mi][p]
                        if pp.active:
                            if col0 < pp.vmax:
                                fallback = _pass_values(
                                    atom,
                                    kv_rsrc,
                                    ks_rsrc,
                                    pt_rsrc,
                                    qf,
                                    pp,
                                    cols,
                                    offs[mi],
                                    columns,
                                    width,
                                    num_pages,
                                )
                                for j3 in range_constexpr(2 * WMMA_N_TILES):
                                    if fallback[j3][0]:
                                        _st_f32(
                                            logits_rsrc,
                                            fallback[j3][1],
                                            fallback[j3][2],
                                        )
                            if col0 >= pp.vmax:
                                masks = _pass_masks(pp, cols, offs[mi], columns)
                                for j4 in range_constexpr(2 * WMMA_N_TILES):
                                    if masks[j4][0]:
                                        _st_f32(logits_rsrc, masks[j4][1], neg_inf)
        _tr_mark(trs, 10)
        _tr_end(trs)

    @flyc.kernel(name=kernel_name, known_block_size=[WMMA_THREADS, 1, 1])
    def kernel_r32_early(
        Q,
        KV,
        KS,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        # ROUND-R2C: kernel_r32 (TDM) with the full-block branch's Q fragments, page-table
        # entries and per-column scales issued before the stage's tensor wait + barrier.
        tid = fx.Int32(fx.thread_idx.x)
        chunk = fx.Int32(fx.block_idx.x)
        rblk = fx.Int32(fx.block_idx.y)
        wave = tid // WMMA_WAVE
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N  # A: slot m / B, C: column n
        kgrp = lane // WMMA_N  # k-half of the operands, row pair of C
        t0 = (rblk * WMMA_WAVES + wave) * (2 * WMMA_ROWS_PER_WAVE)
        col0 = chunk * WMMA_COLS

        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        ks_rsrc = _plane(KS, _plane_bytes(num_pages, P * 4))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))

        metas = [
            _rows_meta(
                t2r_rsrc,
                qp_rsrc,
                cl_rsrc,
                t0 + mi * WMMA_ROWS_PER_WAVE,
                rows,
                num_requests,
            )
            for mi in range_constexpr(2)
        ]
        plans = [_pass_plan(metas[mi], rows, kgrp) for mi in range_constexpr(2)]
        for mi in range_constexpr(2):
            my_vis = _pick_vis(lane, metas[mi])
            my_row = t0 + mi * WMMA_ROWS_PER_WAVE + _imin(lane, WMMA_ROWS_PER_WAVE - 1)
            meta_off = _imin(my_row, rows - 1) * 4
            writer = (chunk == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)
            if writer:
                _st_i32(vis_rsrc, meta_off, my_vis)
                _st_i32(rst_rsrc, meta_off, fx.Int32(0))

        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        cols = _lane_cols(col0, n_in)
        offs = [
            _store_offsets(t0 + mi * WMMA_ROWS_PER_WAVE, kgrp, cols, rows, columns)
            for mi in range_constexpr(2)
        ]
        neg_inf = fx.Float32(_NEG_INF)
        kptr = pair_storage(KV, kv_rsrc)
        reuse = _r32_guard(t2r_rsrc, rblk * 32, rows, num_requests)
        if reuse:
            ppa = plans[0][0]
            ppb = plans[1][0]
            vmax = _imax(ppa.vmax, ppb.vmax)
            # Issued before the TDM stage (clamped rows / masked pages: always in bounds).
            qfa = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
            qfb = _q_frags(q_rsrc, t0 + WMMA_ROWS_PER_WAVE, rows, n_in, kgrp)
            pre = _page_scale_pre(pt_rsrc, ks_rsrc, ppa, cols, width, num_pages)
            pair_stage(KV, kptr, pt_rsrc, t2r_rsrc, rblk, col0, width, num_pages)
            if col0 < vmax:
                res = _pair_values_pre(
                    atom, kptr, qfa, qfb, ppa, ppb, cols, offs[0], offs[1], columns, pre
                )
                for j1 in range_constexpr(4 * WMMA_N_TILES):
                    if res[j1][0]:
                        _st_f32(logits_rsrc, res[j1][1], res[j1][2])
            if col0 >= vmax:
                for mi in range_constexpr(2):
                    msk = _pass_masks(plans[mi][0], cols, offs[mi], columns)
                    for j2 in range_constexpr(2 * WMMA_N_TILES):
                        if msk[j2][0]:
                            _st_f32(logits_rsrc, msk[j2][1], neg_inf)
        if reuse == 0:
            for mi in range_constexpr(2):
                subt0 = t0 + mi * WMMA_ROWS_PER_WAVE
                live = subt0 < rows
                if live:
                    qf = _q_frags(q_rsrc, subt0, rows, n_in, kgrp)
                    for p in range_constexpr(WMMA_PASSES):
                        pp = plans[mi][p]
                        if pp.active:
                            if col0 < pp.vmax:
                                fallback = _pass_values(
                                    atom,
                                    kv_rsrc,
                                    ks_rsrc,
                                    pt_rsrc,
                                    qf,
                                    pp,
                                    cols,
                                    offs[mi],
                                    columns,
                                    width,
                                    num_pages,
                                )
                                for j3 in range_constexpr(2 * WMMA_N_TILES):
                                    if fallback[j3][0]:
                                        _st_f32(
                                            logits_rsrc,
                                            fallback[j3][1],
                                            fallback[j3][2],
                                        )
                            if col0 >= pp.vmax:
                                masks = _pass_masks(pp, cols, offs[mi], columns)
                                for j4 in range_constexpr(2 * WMMA_N_TILES):
                                    if masks[j4][0]:
                                        _st_f32(logits_rsrc, masks[j4][1], neg_inf)

    @flyc.kernel(name=kernel_name, known_block_size=[kt_threads, 1, 1])
    def kernel_kt(
        Q,
        KV,
        KS,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        # ROUND-R2C fp8 ktile: block (x, y) owns columns [x*64C, x*64C + 64C) and rows
        # [y*RB, y*RB + RB); wave w owns the M tile of rows y*RB + 4w .. +3.
        _inst_prefetch(ipf)  # ROUND-R2D (ipf=0: nothing)
        tid = fx.Int32(fx.thread_idx.x)
        slab = fx.Int32(fx.block_idx.x)
        rblk = fx.Int32(fx.block_idx.y)
        wave = tid // WMMA_WAVE
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N
        kgrp = lane // WMMA_N
        slab0 = slab * (WMMA_COLS * col_tiles)
        base = rblk * kt_rows
        t0 = base + wave * WMMA_ROWS_PER_WAVE
        _tr_begin(
            trs,
            trace,
            LOGITS,
            rows,
            columns,
            (columns + WMMA_COLS * col_tiles - 1) // (WMMA_COLS * col_tiles),
            (rows + kt_rows - 1) // kt_rows,
            kt_waves,
            slab,
            rblk,
        )

        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        ks_rsrc = _plane(KS, _plane_bytes(num_pages, P * 4))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))

        meta = _rows_meta(t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests)
        plans = _pass_plan(meta, rows, kgrp)
        my_vis = _pick_vis(lane, meta)
        my_row = t0 + _imin(lane, WMMA_ROWS_PER_WAVE - 1)
        meta_off = _imin(my_row, rows - 1) * 4
        writer = (slab == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)
        if writer:
            _st_i32(vis_rsrc, meta_off, my_vis)
            _st_i32(rst_rsrc, meta_off, fx.Int32(0))
        _tr_mark(trs, 4, True)

        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        neg_inf = fx.Float32(_NEG_INF)
        kbase = _kt8_storage()
        kreq, full = _kt8_guard(t2r_rsrc, base, rows, num_requests)
        _tr_mark(trs, 5, True)
        if full:
            kvis = _imax(_ld_uniform_i32(cl_rsrc, kreq * 4), 0) // COMPRESS_RATIO
            kissue = [
                (slab0 + ki * WMMA_COLS < kvis) & (slab0 + ki * WMMA_COLS < columns)
                for ki in range_constexpr(col_tiles)
            ]
            for ki2 in range_constexpr(col_tiles):
                if kissue[ki2]:
                    _kt8_issue(
                        KV,
                        kbase,
                        pt_rsrc,
                        kreq,
                        slab0 + ki2 * WMMA_COLS,
                        width,
                        num_pages,
                        ki2,
                    )
            _tr_mark(trs, 6)
            qf = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
            kpre = _kt8_pre(pt_rsrc, ks_rsrc, kreq, full, slab0, n_in, width, num_pages)
            _tr_mark(trs, 7, True)
            fx.rocdl.tdm_ops.tensor_wait(0)
            _tr_mark(trs, 8)
            fx.gpu.barrier()
            _tr_mark(trs, 9)
            _tr_mark_rt(trs, 14)
            pf = plans[0]
            for ct in range_constexpr(col_tiles):
                col0 = slab0 + ct * WMMA_COLS
                cols = _lane_cols(col0, n_in)
                offs = _store_offsets(t0, kgrp, cols, rows, columns)
                kbuf = kbase + ct * (TDM_FP8_K_TILE_BYTES // 4)
                if col0 < pf.vmax:
                    res = _kt8_values(atom, kbuf, qf, pf, cols, offs, columns, kpre[ct])
                    for j1 in range_constexpr(2 * WMMA_N_TILES):
                        if res[j1][0]:
                            _st_f32(logits_rsrc, res[j1][1], res[j1][2])
                if col0 >= pf.vmax:
                    msk = _pass_masks(pf, cols, offs, columns)
                    for j2 in range_constexpr(2 * WMMA_N_TILES):
                        if msk[j2][0]:
                            _st_f32(logits_rsrc, msk[j2][1], neg_inf)
            _tr_mark(trs, 10)
        if full == 0:
            for kf in range(fx.Int32(0), fx.Int32(col_tiles), fx.Int32(1), unroll=1):
                fcol0 = slab0 + fx.Int32(kf) * WMMA_COLS
                fcols = _lane_cols(fcol0, n_in)
                foffs = _store_offsets(t0, kgrp, fcols, rows, columns)
                live = t0 < rows
                if live:
                    fqf = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
                    for p in range_constexpr(WMMA_PASSES):
                        pp = plans[p]
                        if pp.active:
                            if fcol0 < pp.vmax:
                                fallback = _pass_values(
                                    atom,
                                    kv_rsrc,
                                    ks_rsrc,
                                    pt_rsrc,
                                    fqf,
                                    pp,
                                    fcols,
                                    foffs,
                                    columns,
                                    width,
                                    num_pages,
                                )
                                for j3 in range_constexpr(2 * WMMA_N_TILES):
                                    if fallback[j3][0]:
                                        _st_f32(
                                            logits_rsrc,
                                            fallback[j3][1],
                                            fallback[j3][2],
                                        )
                            if fcol0 >= pp.vmax:
                                masks = _pass_masks(pp, fcols, foffs, columns)
                                for j4 in range_constexpr(2 * WMMA_N_TILES):
                                    if masks[j4][0]:
                                        _st_f32(logits_rsrc, masks[j4][1], neg_inf)
            _tr_mark(trs, 10)
        _tr_end(trs)

    def _ktl_pre(
        t2r_rsrc,
        qp_rsrc,
        cl_rsrc,
        pt_rsrc,
        base,
        t0s,
        rows,
        columns,
        num_requests,
        slab0,
        width,
        num_pages,
        wave_id,
    ):
        """ONE batch of SGPR loads for the full-block path, then the derived values.

        Returns (full, safe request, vis[4] of the wave's rows, kissue[ct], gathers).
        Offsets are clamped exactly like `_kt_guard` / `_rows_meta_uniform` /
        `_kt_issue_w8` / `_persist_issue_tdm` (in-plane by construction); the TDM row ids
        are the same expressions (valid -> phys, else page 0), so the gathers are identical.
        """
        # -- issue (no use in between) --
        g = [
            _ld_uniform_i32(t2r_rsrc, _imin(base + i, rows - 1) * 4)
            for i in range_constexpr(kt_rows)
        ]
        req0 = g[0]
        req_ok = (req0 >= 0) & (req0 < num_requests)
        safe = req_ok.select(req0, 0)
        pos = [
            ld_uniform_pos(qp_rsrc, _imin(t0s + i, rows - 1))
            for i in range_constexpr(WMMA_ROWS_PER_WAVE)
        ]
        ctx = _ld_uniform_i32(cl_rsrc, safe * 4)
        gpw = 8 // kt_waves  # 8-row gathers per wave per 64-col tile
        span = WMMA_COLS // kt_waves  # columns per wave per 64-col tile
        plan = []
        for ct in range_constexpr(col_tiles):
            for gi in range_constexpr(gpw):
                local = wave_id * span + gi * 8
                col = slab0 + ct * WMMA_COLS + local
                ents = []
                for page in range_constexpr(2 if P == 4 else 1):
                    logical = col // P + page
                    logical_ok = logical < width
                    safe_logical = logical_ok.select(logical, 0)
                    ents.append(
                        (
                            logical_ok,
                            _ld_uniform_i32(pt_rsrc, (safe * width + safe_logical) * 4),
                        )
                    )
                plan.append((ct, local, ents))
        # -- use --
        full = (base + kt_rows <= rows) & req_ok
        for i in range_constexpr(1, kt_rows):
            full = full & (g[i] == req0)
        # Pin the batch before the (uniform) full branch: LLVM otherwise sinks the ctx /
        # page-table loads into it and re-serialises them behind the ctx wait. An empty
        # side-effecting asm reading the SGPRs needs them all landed (ONE s_wait_kmcnt);
        # it emits no instruction and changes no value (ull = the kernel_kt guard).
        _pin_sgprs(
            g + [ctx] + [phys for _ct, _local, _ents in plan for _lok, phys in _ents]
        )
        vis = [
            _imax(_imin(pos[i] + 1, ctx), 0) // COMPRESS_RATIO
            for i in range_constexpr(WMMA_ROWS_PER_WAVE)
        ]
        kvis = _imax(ctx, 0) // COMPRESS_RATIO
        kissue = [
            (slab0 + ct * WMMA_COLS < kvis) & (slab0 + ct * WMMA_COLS < columns)
            for ct in range_constexpr(col_tiles)
        ]
        gathers = []
        for ct, local, ents in plan:
            row_ids = []
            for logical_ok, phys in ents:
                valid = logical_ok & (phys >= 0) & (phys < num_pages)
                phys = valid.select(phys, 0)
                if P == 4:
                    for row in range_constexpr(4):
                        row_ids.append(phys * 4 + row)
                else:
                    row0 = local % 16 if P == 16 else 0
                    for r in range_constexpr(8):
                        row_ids.append(phys * P + (row0 + r))
            gathers.append((ct, local, row_ids))
        return full, safe, vis, kissue, gathers

    def _ktl_rowvals(lane, kgrp, vis):
        """(this lane's vis/rst value for lanes 0..3, vis_a, vis_b, vmax) from SGPR vis."""
        v = vis[WMMA_ROWS_PER_WAVE - 1]
        for i in range_constexpr(WMMA_ROWS_PER_WAVE - 2, -1, -1):
            v = (lane == i).select(vis[i], v)
        g0 = kgrp == 0
        vmax = _imax(_imax(vis[0], vis[1]), _imax(vis[2], vis[3]))
        return v, g0.select(vis[0], vis[2]), g0.select(vis[1], vis[3]), vmax

    # ---- ROUND-R2D kconv (lean fp8 ktile, kconv=True): after the TDM publish every thread
    # converts its 16-byte chunks of the E4M3 slab ONCE into a bf16 slab with the bf16 kt
    # layout (272 B pitch), one more s_barrier, then the WMMA B fragments are plain bf16 LDS
    # reads (the per-wave per-fragment pk8 conversion of kernel_kt disappears). Same exact
    # converter (_e4m3x8_to_bf16x4_fast), same k positions, same epilogue (per-column scale
    # per head before ReLU) -> identical scores.
    def _ktl8_storage():
        if not kconv:
            return _kt8_storage()
        return (
            fx.SharedAllocator(base_alignment=128, static=True)
            .allocate(
                fx.Array[
                    fx.Int32,
                    col_tiles * (TDM_FP8_K_TILE_BYTES + TDM_K_TILE_BYTES) // 4,
                    128,
                ]
            )
            .peek()
            .ptr
        )

    def _ktl8_kbuf(kbase, ct):
        if not kconv:
            return kbase + ct * (TDM_FP8_K_TILE_BYTES // 4)
        return kbase + (col_tiles * TDM_FP8_K_TILE_BYTES + ct * TDM_K_TILE_BYTES) // 4

    def _kc_publish(kbase, tid):
        if not kconv:
            return
        kb16 = kbase + (col_tiles * TDM_FP8_K_TILE_BYTES) // 4
        for ct in range_constexpr(col_tiles):
            for i in range_constexpr((WMMA_COLS * 8) // kt_threads):
                j = tid + i * kt_threads
                row = j // 8
                q = j % 8
                src = Vec(
                    fx.ptr_load(
                        kbase
                        + (
                            ct * TDM_FP8_K_TILE_BYTES
                            + row * TDM_FP8_K_ROW_BYTES
                            + q * 16
                        )
                        // 4,
                        result_type=Vec.make_type(4, fx.Int32),
                    )
                )
                c0 = _e4m3x8_to_bf16x4_fast(fx.Int32(src[0]), fx.Int32(src[1]))
                c1 = _e4m3x8_to_bf16x4_fast(fx.Int32(src[2]), fx.Int32(src[3]))
                dst = (ct * TDM_K_TILE_BYTES + row * TDM_K_ROW_BYTES + q * 32) // 4
                fx.ptr_store(c0, kb16 + dst)
                fx.ptr_store(c1, kb16 + dst + 4)
        fx.rocdl.s_barrier()

    def _kc_tile(atom, kbuf, qf, c, kgrp, scale):
        """`_kt8_tile` reading the converted bf16 slab (bf16 kt fragment addresses)."""
        k_base = (c % WMMA_COLS) * TDM_K_ROW_BYTES + kgrp * 16
        acc_t = fx.make_rmem_tensor(8, fx.Float32)
        acc_t.store(Vec.filled(8, 0.0, fx.Float32))
        for s in range_constexpr(WMMA_K_STEPS):
            off = k_base + s * (WMMA_K * 2)
            lo = Vec(
                fx.ptr_load(kbuf + off // 4, result_type=Vec.make_type(4, fx.Int32))
            )
            hi = Vec(
                fx.ptr_load(
                    kbuf + (off + 32) // 4, result_type=Vec.make_type(4, fx.Int32)
                )
            )
            kf = _rmem8(lo.shuffle(hi, list(range(8))))
            fx.gemm(atom, acc_t, _rmem8(qf[s]), kf, acc_t)
        acc = acc_t.load()
        zero = fx.Float32(0.0)
        sa = (
            (fx.Float32(acc[0]) * scale).maximumf(zero)
            + (fx.Float32(acc[1]) * scale).maximumf(zero)
            + (fx.Float32(acc[2]) * scale).maximumf(zero)
            + (fx.Float32(acc[3]) * scale).maximumf(zero)
        )
        sb = (
            (fx.Float32(acc[4]) * scale).maximumf(zero)
            + (fx.Float32(acc[5]) * scale).maximumf(zero)
            + (fx.Float32(acc[6]) * scale).maximumf(zero)
            + (fx.Float32(acc[7]) * scale).maximumf(zero)
        )
        return sa * inv_divisor, sb * inv_divisor

    ktl8_tile = _kc_tile if kconv else _kt8_tile

    def _ktl8_values(atom, kbuf, qf, kgrp, cols, offs, columns, pre, vis_a, vis_b):
        """`_kt8_values` of a full block (every row owned, ok = column in range)."""
        neg_inf = fx.Float32(_NEG_INF)
        out = []
        for nt in range_constexpr(WMMA_N_TILES):
            c = cols[nt]
            page_ok, scale = pre[nt]
            sa, sb = ktl8_tile(atom, kbuf, qf, c, kgrp, scale)
            va = (page_ok & (c < vis_a)).select(sa, neg_inf)
            vb = (page_ok & (c < vis_b)).select(sb, neg_inf)
            out.append((c < columns, offs[2 * nt], va))
            out.append((c < columns, offs[2 * nt + 1], vb))
        return out

    def _ktl8_issue(KV, kbase, gathers, ct, num_pages):
        for gct, local, row_ids in gathers:
            if gct == ct:
                _tdm_fp8_k_gather(
                    KV,
                    kbase,
                    row_ids,
                    num_pages,
                    ct * TDM_FP8_K_TILE_BYTES + local * TDM_FP8_K_ROW_BYTES,
                    P,
                )

    @flyc.kernel(name=kernel_name, known_block_size=[kt_threads, 1, 1])
    def kernel_kt_lean(
        Q,
        KV,
        KS,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        # ROUND-R2D: fp8 kernel_kt with the lean full-block path (bf16 kernel_kt_lean).
        _inst_prefetch(ipf)
        tid = fx.Int32(fx.thread_idx.x)
        slab = fx.Int32(fx.block_idx.x)
        rblk = fx.Int32(fx.block_idx.y)
        wave_id = fx.Int32(fx.rocdl.wave_id())
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N
        kgrp = lane // WMMA_N
        slab0 = slab * (WMMA_COLS * col_tiles)
        base = rblk * kt_rows
        t0s = base + wave_id * WMMA_ROWS_PER_WAVE
        _tr_begin(
            trs,
            trace,
            LOGITS,
            rows,
            columns,
            (columns + WMMA_COLS * col_tiles - 1) // (WMMA_COLS * col_tiles),
            (rows + kt_rows - 1) // kt_rows,
            kt_waves,
            slab,
            rblk,
        )

        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        ks_rsrc = _plane(KS, _plane_bytes(num_pages, P * 4))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))
        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        neg_inf = fx.Float32(_NEG_INF)
        kbase = _ktl8_storage()
        full, kreq, lvis, kissue, gathers = _ktl_pre(
            t2r_rsrc,
            qp_rsrc,
            cl_rsrc,
            pt_rsrc,
            base,
            t0s,
            rows,
            columns,
            num_requests,
            slab0,
            width,
            num_pages,
            wave_id,
        )
        _tr_mark(trs, 5, True)
        if full:
            for ki in range_constexpr(col_tiles):
                if kissue[ki]:
                    _ktl8_issue(KV, kbase, gathers, ki, num_pages)
            _tr_mark(trs, 6)
            qf = _q_frags(q_rsrc, t0s, rows, n_in, kgrp)
            kpre = _kt8_pre(pt_rsrc, ks_rsrc, kreq, full, slab0, n_in, width, num_pages)
            lv, vis_a, vis_b, vmax = _ktl_rowvals(lane, kgrp, lvis)
            lrow_off = _imin(t0s + _imin(lane, WMMA_ROWS_PER_WAVE - 1), rows - 1) * 4
            lwriter = (slab == 0) & (lane < WMMA_ROWS_PER_WAVE)
            if lwriter:
                _st_i32(vis_rsrc, lrow_off, lv)
                _st_i32(rst_rsrc, lrow_off, fx.Int32(0))
            _tr_mark(trs, 7, True)
            fx.rocdl.tdm_ops.tensor_wait(0)
            _tr_mark(trs, 8)
            fx.rocdl.s_barrier()  # TENSORcnt==0 => slab in LDS (mi400-tdm-sync)
            _tr_mark(trs, 9)
            _tr_mark_rt(trs, 14)
            _kc_publish(
                kbase, tid
            )  # ROUND-R2D kconv: E4M3 slab -> bf16 slab (else nothing)
            inside = slab0 + WMMA_COLS * col_tiles <= columns
            for ct in range_constexpr(col_tiles):
                col0 = slab0 + ct * WMMA_COLS
                cols = _lane_cols(col0, n_in)
                offs = _store_offsets(t0s, kgrp, cols, rows, columns)
                kbuf = _ktl8_kbuf(kbase, ct)
                if col0 < vmax:
                    res = _ktl8_values(
                        atom,
                        kbuf,
                        qf,
                        kgrp,
                        cols,
                        offs,
                        columns,
                        kpre[ct],
                        vis_a,
                        vis_b,
                    )
                    if inside:
                        for j1 in range_constexpr(2 * WMMA_N_TILES):
                            _st_f32(logits_rsrc, res[j1][1], res[j1][2])
                    if inside == 0:
                        for j2 in range_constexpr(2 * WMMA_N_TILES):
                            if res[j2][0]:
                                _st_f32(logits_rsrc, res[j2][1], res[j2][2])
                if col0 >= vmax:
                    if inside:
                        for j3 in range_constexpr(2 * WMMA_N_TILES):
                            _st_f32(logits_rsrc, offs[j3], neg_inf)
                    if inside == 0:
                        for j4 in range_constexpr(2 * WMMA_N_TILES):
                            if cols[j4 // 2] < columns:
                                _st_f32(logits_rsrc, offs[j4], neg_inf)
            _tr_mark(trs, 10)
        if full == 0:
            # Exact fp8 kernel_kt per-wave raw path (mixed / partial / invalid requests).
            meta = _rows_meta(t2r_rsrc, qp_rsrc, cl_rsrc, t0s, rows, num_requests)
            plans = _pass_plan(meta, rows, kgrp)
            my_vis = _pick_vis(lane, meta)
            my_row = t0s + _imin(lane, WMMA_ROWS_PER_WAVE - 1)
            meta_off = _imin(my_row, rows - 1) * 4
            writer = (slab == 0) & (lane < WMMA_ROWS_PER_WAVE) & (my_row < rows)
            if writer:
                _st_i32(vis_rsrc, meta_off, my_vis)
                _st_i32(rst_rsrc, meta_off, fx.Int32(0))
            for kf in range(fx.Int32(0), fx.Int32(col_tiles), fx.Int32(1), unroll=1):
                fcol0 = slab0 + fx.Int32(kf) * WMMA_COLS
                fcols = _lane_cols(fcol0, n_in)
                foffs = _store_offsets(t0s, kgrp, fcols, rows, columns)
                live = t0s < rows
                if live:
                    fqf = _q_frags(q_rsrc, t0s, rows, n_in, kgrp)
                    for p in range_constexpr(WMMA_PASSES):
                        pp = plans[p]
                        if pp.active:
                            if fcol0 < pp.vmax:
                                fallback = _pass_values(
                                    atom,
                                    kv_rsrc,
                                    ks_rsrc,
                                    pt_rsrc,
                                    fqf,
                                    pp,
                                    fcols,
                                    foffs,
                                    columns,
                                    width,
                                    num_pages,
                                )
                                for j5 in range_constexpr(2 * WMMA_N_TILES):
                                    if fallback[j5][0]:
                                        _st_f32(
                                            logits_rsrc,
                                            fallback[j5][1],
                                            fallback[j5][2],
                                        )
                            if fcol0 >= pp.vmax:
                                masks = _pass_masks(pp, fcols, foffs, columns)
                                for j6 in range_constexpr(2 * WMMA_N_TILES):
                                    if masks[j6][0]:
                                        _st_f32(logits_rsrc, masks[j6][1], neg_inf)
            _tr_mark(trs, 10)
        _tr_end(trs)

    selected_kernel = kernel if rows_per_block == 16 else kernel_r32
    if early:
        selected_kernel = kernel_r32_early
    if ktile:
        selected_kernel = kernel_kt_lean if lean else kernel_kt

    @flyc.jit
    def launch(
        Q,
        KV,
        KS,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        grid_x,
        grid_y,
        rows,
        columns,
        num_pages,
        num_requests,
        width,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        selected_kernel._func.__name__ = kernel_name
        selected_kernel(
            Q,
            KV,
            KS,
            PT,
            T2R,
            QP,
            CL,
            LOGITS,
            VIS,
            RST,
            rows,
            columns,
            num_pages,
            num_requests,
            width,
        ).launch(
            grid=(fx.Int64(grid_x), fx.Int64(grid_y), 1),
            block=(kt_threads, 1, 1),
            stream=stream,
        )

    launch.kernel_name = kernel_name
    launch.dtype_tag = "fp8_wmma"
    launch.early = early
    if ktile:  # ROUND-R2C
        launch.cols_per_block = WMMA_COLS * col_tiles
        launch.rows_per_launch = kt_rows
        launch.kt = (kt_rows, col_tiles)
        launch.threads = kt_threads
        launch.lean, launch.ipf = bool(lean), int(ipf)
    launch.trace = trace
    return launch


def _build_wmma_kernel_for(dtype: str, divisor: float):
    """The WMMA builder's dtype specialisations: ``bf16`` (round C) / ``fp8``."""
    if dtype == "bf16":
        return _build_wmma_kernel(divisor)
    if dtype == "fp8":
        return _build_wmma_fp8_kernel(divisor)
    raise ValueError(f"unknown WMMA dtype {dtype!r}")


# ---------------------------------------------------------------------------
# Round DRAFT (QSA_DRAFT_PATH=1, default since ROUND-MERGE1): one query row per wave
# ---------------------------------------------------------------------------
# Draft (rows <= requests, typically one query row per request) sends every
# row's K pages through the r16 kernels' request-dedup loop: a wave owns 4
# rows of 4 *different* requests, so it runs 4 serial passes (TDM r16: gather,
# wait, 2 WG barriers each; fp8 r16: raw loads + the 18-VALU/dword "current"
# E4M3 converter), each storing only 1/4 of its WMMA rows.
#
# The draft kernel instead gives each wave exactly one row and a private
# ``wave_cols`` (32/64) column slab; a workgroup = one row x 4 slabs:
#   * all row metadata (request, ctx, pos, visible) is SGPR (s_buffer_load);
#     a slab with col0 >= visible stores -inf only (no K/Q traffic);
#   * K for the slab is loaded once: TDM (``tdm``: wave_cols/8 8-row gathers
#     into a wave-private padded LDS tile, ``s_wait_tensorcnt 0``, no WG
#     barrier because no other wave reads that tile) or raw per-lane buffer
#     loads (``raw``); fp8 converts with the GPU-verified exact pk8 converter
#     (``_e4m3x8_to_bf16x4_fast``, +-480 for 0x7F/0xFF like every other path);
#   * WMMA 16x16x32 bf16, A = the row's Q with slot m -> head m % 4 (identical
#     per-slot Q/K/step inputs to the champion r16 tiles, so identical sums),
#     accumulator elements 0..3 of lane (kgrp, n) = heads 0..3 of column n of
#     tile ``2j + kgrp``; post-GEMM order is the champion's per dtype (bf16:
#     (r0+r1)+(r2+r3); fp8: relu(a_h*scale) left-to-right), then * 1/divisor;
#   * lane l stores column col0 + 32j + l exactly once (coalesced 128 B per
#     wave store), predicated on c < columns at a clamped in-plane offset;
#     -inf for c >= visible or an invalid page/request (champion masks).
# Row metadata (visible_groups / row_starts) is written by (bx 0, tid 0).
DRAFT_WAVES = 4
# ROUND-MERGE1: default 1 (GPU-verified TDM w32: draft bf16 c65536 30.6 -> 23.3 us,
# fp8 25.9 -> 19.4 us); QSA_DRAFT_PATH=0 opts out to the champion r16 routing.
_DRAFT_PATH = os.environ.get("QSA_DRAFT_PATH", "1").strip()
if _DRAFT_PATH not in ("0", "1"):
    raise ValueError(f"QSA_DRAFT_PATH must be '0' or '1', got {_DRAFT_PATH!r}")
_DRAFT_PATH = _DRAFT_PATH == "1"
_DRAFT_K_LOAD = os.environ.get("QSA_DRAFT_K_LOAD", "tdm").strip().lower()
if _DRAFT_K_LOAD not in ("tdm", "raw"):
    raise ValueError(f"QSA_DRAFT_K_LOAD must be 'tdm' or 'raw', got {_DRAFT_K_LOAD!r}")
_DRAFT_COLS = os.environ.get("QSA_DRAFT_COLS", "auto").strip().lower()
if _DRAFT_COLS not in ("auto", "32", "64"):
    raise ValueError(
        f"QSA_DRAFT_COLS must be 'auto', '32' or '64', got {_DRAFT_COLS!r}"
    )


# ROUND-FP8PERF: draft fp8 schedule (intrinsic converter only; QSA_FP8_CVT_ASM=1
# keeps the champion order 'base'). Root GPU (a08-1 GPU3, 5fb8740e): the hazard-
# free intrinsic converter regressed draft fp8 c65536 19.3 -> 20.6 us. ISA cause:
# with the compiler-visible converter the machine scheduler sank the 4 Q-fragment
# buffer loads next to their WMMAs (VGPR 106 -> 91) -> 4 serial
# ``s_wait_loadcnt 0x0`` round trips per wave, after ``s_wait_tensorcnt 0``
# (the asm build issued all 8 Q loads at once). Variants:
#   qpre  (default) Q fragments are issued between the TDM gathers and
#         ``s_wait_tensorcnt 0`` (the decode-group order): the wait is a
#         side-effecting fence the scheduler cannot sink loads across, so the Q
#         latency hides under the gather latency; LLVM interleaves freely
#         (and pads the WMMA hazards of its register reuse with v_nop).
#   kall  qpre + every K fragment of the slab converted before a
#         ``sched_barrier`` and an uninterrupted WMMA burst (no VALU can land in
#         a WMMA hazard window -> no conversion padding; no co-execution).
#   kpipe qpre + software pipeline: tile t's WMMAs share a scheduling
#         region with tile t+1's conversion (WMMA / VALU co-execution, kernel-wiki
#         mi400-valu-coexec Â§2.2), separated by ``sched_barrier``s; conversion
#         temps of tile t+1 are distinct from tile t's live A/B/D registers.
#   base  5fb8740e order (Q loaded after ``s_wait_tensorcnt 0``).
# Non-default get a ``_s<variant>`` tag. FP8PERF2 GPU c65536/c32768: qpre 18.4/11.6 kpipe 20.6/11.9
_DRAFT_FP8_SCHED_DEFAULT = "qpre"
_DRAFT_FP8_SCHED = (
    os.environ.get("QSA_DRAFT_FP8_SCHED", _DRAFT_FP8_SCHED_DEFAULT).strip().lower()
)
if _DRAFT_FP8_SCHED not in ("qpre", "kall", "kpipe", "base"):
    raise ValueError(
        f"QSA_DRAFT_FP8_SCHED must be 'qpre', 'kall', 'kpipe' or 'base', got {_DRAFT_FP8_SCHED!r}"
    )


# ROUND-R2 short contexts: waves (= wave_cols column slabs) per draft workgroup.
# Default 4 (unchanged). At columns 1024 / w32 a row is 8 WGs x 4 waves; small C
# then fills few CUs (grid = 8 x C) -- 2 / 1 give 2x / 4x more, smaller WGs.
# ROUND-R2E: default 'auto' = per dtype (bf16 4, fp8 2: GPU fp8 single decode b16 wg2
# 2.66-3.10 us vs native 3.0-3.17, wg4 1.02-1.04x); '4' / '2' / '1' force both dtypes.
DRAFT_AUTO_WG_WAVES = {False: 4, True: 2}  # is_fp8 -> wg_waves
_DRAFT_WG_WAVES_SPEC = os.environ.get("QSA_DRAFT_WG_WAVES", "auto").strip().lower()
if _DRAFT_WG_WAVES_SPEC not in ("auto", "4", "2", "1"):
    raise ValueError(
        f"QSA_DRAFT_WG_WAVES must be auto, 4, 2 or 1, got {_DRAFT_WG_WAVES_SPEC!r}"
    )
_DRAFT_WG_WAVES = None if _DRAFT_WG_WAVES_SPEC == "auto" else int(_DRAFT_WG_WAVES_SPEC)


def draft_wg_waves(is_fp8: bool) -> int:
    """Effective draft workgroup waves of the dtype (QSA_DRAFT_WG_WAVES, 'auto' default)."""
    if _DRAFT_WG_WAVES is None:
        return DRAFT_AUTO_WG_WAVES[bool(is_fp8)]
    return _DRAFT_WG_WAVES


def draft_fp8_sched() -> str:
    """Effective draft fp8 schedule: the knob, or 'base' with the asm converter."""
    return "base" if _FP8_CVT_ASM else _DRAFT_FP8_SCHED


def draft_wave_cols(k_load: str) -> int:
    """Per-wave column slab: auto = 32 for TDM (8.7/4.6 KB LDS per wave), 64 raw."""
    if _DRAFT_COLS != "auto":
        return int(_DRAFT_COLS)
    return 32 if k_load == "tdm" else 64


def draft_route(rows: int, num_requests: int, is_fp8: bool, kv_ptr: int):
    """Pure-Python routing of the draft kernel (default on): ``None`` or (k_load, cols).

    Only when QSA_DRAFT_PATH=1, raw access, the dtype's WMMA path, and the
    draft regime ``rows <= requests`` (one row per request on average; decode
    keeps the r16 dedup kernels that read each K tile once for 4 rows).
    TDM needs a 128-byte aligned cache base, else the raw draft variant.
    """
    if not _DRAFT_PATH or _ACCESS != "raw":
        return None
    if (_FP8_PATH if is_fp8 else _BF16_PATH) != "wmma":
        return None
    if int(rows) > int(num_requests):
        return None
    k_load = _DRAFT_K_LOAD
    if k_load == "tdm" and int(kv_ptr) % 128:
        k_load = "raw"
    return k_load, draft_wave_cols(k_load)


def draft_grid(
    rows: int, columns: int, wave_cols: int, wg_waves: int = DRAFT_WAVES
) -> tuple[int, int]:
    """grid.x = column spans of ``wg_waves`` slabs, grid.y = one row per block."""
    if wave_cols not in (32, 64):
        raise ValueError("draft wave_cols must be 32 or 64")
    span = int(wg_waves) * wave_cols
    return (int(columns) + span - 1) // span, int(rows)


def draft_store_plan(rows: int, columns: int, wave_cols: int):
    """Trace-free mirror of the draft kernel's store map (CPU tests).

    Yields (row, column) per (block x, block y, wave, lane, pair j) store whose
    predicate ``c < columns`` holds, in kernel order.
    """
    gx, gy = draft_grid(rows, columns, wave_cols)
    for by in range(gy):
        for bx in range(gx):
            for wave in range(DRAFT_WAVES):
                col0 = bx * DRAFT_WAVES * wave_cols + wave * wave_cols
                for j in range(wave_cols // WMMA_WAVE):
                    for lane in range(WMMA_WAVE):
                        c = col0 + j * WMMA_WAVE + lane
                        if c < columns:
                            yield by, c


@cache
def _build_draft_kernel(
    is_fp8: bool,
    divisor: float,
    k_load: str = "tdm",
    wave_cols: int = 32,
    pos64: bool = False,
    page_size: int = PAGE_SIZE,
    wg_waves: int = DRAFT_WAVES,
):
    """One-row-per-wave draft specialization (see the Round DRAFT block).

    ``pos64=True`` (ROUND-MERGE1) reads query_positions as int64 through the
    shared ``_pos_loaders`` seam (SGPR variant); ``False`` is the DRAFT kernel.
    ROUND-R2 ``wg_waves`` 4 (default, trace-identical) | 2 | 1: wave column slabs
    per workgroup (short contexts / small batches: more, smaller workgroups).
    """
    _, ld_uniform_pos, qp_bytes, qp_tag = _pos_loaders(pos64)
    P = _check_page_size(page_size)
    waves = int(wg_waves)
    if waves not in _WG_WAVES:
        raise ValueError(f"draft wg_waves must be one of {_WG_WAVES}")
    threads = WMMA_WAVE * waves
    if k_load not in ("tdm", "raw"):
        raise ValueError("draft k_load must be 'tdm' or 'raw'")
    if wave_cols not in (32, 64):
        raise ValueError("draft wave_cols must be 32 or 64")
    if _ACCESS != "raw":
        raise ValueError("the draft kernel requires QSA_G2_ACCESS=raw")
    use_tdm = k_load == "tdm"
    n_tiles = wave_cols // WMMA_N  # 2 / 4 WMMA N tiles per wave
    n_pairs = wave_cols // WMMA_WAVE  # 1 / 2 stores per lane
    span = waves * wave_cols  # columns per workgroup
    inv_divisor = 1.0 / divisor
    q_row_bytes = NUM_HEADS * HEAD_DIM * 2  # 1024 B
    head_bytes = HEAD_DIM * 2  # 256 B
    kstep_q = WMMA_K * 2  # 64 B of a Q row per step
    key_row_bytes = HEAD_DIM if is_fp8 else HEAD_DIM * 2
    page_bytes = P * key_row_bytes
    kstep_k = WMMA_K * (1 if is_fp8 else 2)  # 32 B fp8 / 64 B bf16
    pitch = TDM_FP8_K_ROW_BYTES if is_fp8 else TDM_K_ROW_BYTES  # 144 / 272 B
    tile_bytes = wave_cols * pitch  # per wave
    gather = _tdm_fp8_k_gather if is_fp8 else _tdm_k_gather
    dtag = "fp8" if is_fp8 else "bf16"
    # ROUND-FP8PERF: bf16 always keeps the champion order ('base').
    sched = draft_fp8_sched() if is_fp8 else "base"
    sched_tag = (
        f"_s{sched}"
        if (is_fp8 and not _FP8_CVT_ASM and sched != _DRAFT_FP8_SCHED_DEFAULT)
        else ""
    )
    kernel_name = (
        (
            f"qsa_paged_mqa_logits_page4_gfx1250_{dtag}_draft"
            f"_D{HEAD_DIM}_H{NUM_HEADS}_P{P}_w{wave_cols}_flydsl"
            + ("_tdm" if use_tdm else "_raw")
        )
        + qp_tag
        + (_CVT_ASM_TAG if is_fp8 else "")
        + sched_tag
        + (f"_wg{waves}" if waves != DRAFT_WAVES else "")
    )

    def _frag(rsrc, byte_off):
        lo = _ld_vec(rsrc, byte_off, 4, fx.Int32)
        hi = _ld_vec(rsrc, byte_off + 32, 4, fx.Int32)
        return lo.shuffle(hi, list(range(8)))

    def _rmem8(vec):
        t = fx.make_rmem_tensor(8, fx.Int32)
        t.store(vec)
        return t

    def _storage_tdm(kv_rsrc):
        return (
            fx.SharedAllocator(base_alignment=128, static=True)
            .allocate(fx.Array[fx.Int32, waves * tile_bytes // 4, 128])
            .peek()
            .ptr
        )

    def _storage_raw(kv_rsrc):
        return kv_rsrc

    def _stage_tdm(
        KV, storage, pt_rsrc, safe_req, req_ok, col0, width, num_pages, wave
    ):
        # SGPR-only descriptors (wave_id, request, page table); wave-private tile.
        for group in range_constexpr(wave_cols // 8):
            if P == 4:
                row_ids = []
                for page in range_constexpr(2):
                    logical = col0 // 4 + group * 2 + page
                    logical_ok = logical < width
                    safe_logical = logical_ok.select(logical, 0)
                    phys = _ld_uniform_i32(
                        pt_rsrc, (safe_req * width + safe_logical) * 4
                    )
                    valid = req_ok & logical_ok & (phys >= 0) & (phys < num_pages)
                    phys = valid.select(phys, 0)
                    for r in range_constexpr(4):
                        row_ids.append(phys * 4 + r)
            elif P == 16:  # P = 16: one page lookup per 8-row group (col0 % 32 == 0)
                row_ids = _p16_row_ids(
                    pt_rsrc,
                    safe_req,
                    width,
                    num_pages,
                    col0 // 16 + group // 2,
                    (group % 2) * 8,
                    req_ok,
                )
            else:  # P = 8: the 8-row group IS page col0/8 + group (one lookup)
                row_ids = _p16_row_ids(
                    pt_rsrc, safe_req, width, num_pages, col0 // 8 + group, 0, req_ok, 8
                )
            gather(
                KV,
                storage,
                row_ids,
                num_pages,
                wave * tile_bytes + group * 8 * pitch,
                P,
            )

    def _wait_tdm():
        # Same-wave consumer: data is in LDS once TENSORcnt reaches 0 (tensor
        # done); the tile is wave-private and never reused, so no WG barrier.
        fx.rocdl.tdm_ops.tensor_wait(0)

    def _stage_raw(
        KV, storage, pt_rsrc, safe_req, req_ok, col0, width, num_pages, wave
    ):
        pass

    def _wait_raw():
        pass

    def _kbase_raw(phys, off, local, kgrp):
        return phys * page_bytes + off * key_row_bytes + kgrp * (8 if is_fp8 else 16)

    def _kbase_tdm(phys, off, local, kgrp):
        return local * pitch + (0 if is_fp8 else kgrp * 16)

    def _kfrag_bf16_raw(src, byte_off, kgrp):
        return _frag(src, byte_off)

    def _kfrag_bf16_tdm(src, byte_off, kgrp):
        lo = Vec(
            fx.ptr_load(src + byte_off // 4, result_type=Vec.make_type(4, fx.Int32))
        )
        hi = Vec(
            fx.ptr_load(
                src + (byte_off + 32) // 4, result_type=Vec.make_type(4, fx.Int32)
            )
        )
        return lo.shuffle(hi, list(range(8)))

    def _kfrag_fp8_raw(src, byte_off, kgrp):
        lo = _ld_vec(src, byte_off, 2, fx.Int32)
        hi = _ld_vec(src, byte_off + 16, 2, fx.Int32)
        words = []
        for vec in (lo, hi):
            cvt = _e4m3x8_to_bf16x4_fast(fx.Int32(vec[0]), fx.Int32(vec[1]))
            for j in range_constexpr(4):
                words.append(fx.Int32(cvt[j]))
        return Vec.from_elements(words, dtype=fx.Int32)

    def _kfrag_fp8_tdm(src, byte_off, kgrp):
        lo = Vec(
            fx.ptr_load(src + byte_off // 4, result_type=Vec.make_type(4, fx.Int32))
        )
        hi = Vec(
            fx.ptr_load(
                src + (byte_off + 16) // 4, result_type=Vec.make_type(4, fx.Int32)
            )
        )
        words = []
        for vec in (lo, hi):
            src0 = (kgrp == 0).select(fx.Int32(vec[0]), fx.Int32(vec[2]))
            src1 = (kgrp == 0).select(fx.Int32(vec[1]), fx.Int32(vec[3]))
            cvt = _e4m3x8_to_bf16x4_fast(src0, src1)
            for j in range_constexpr(4):
                words.append(fx.Int32(cvt[j]))
        return Vec.from_elements(words, dtype=fx.Int32)

    def _ks_bytes_fp8(num_pages):
        return _plane_bytes(num_pages, P * 4)

    def _ks_bytes_bf16(num_pages):
        return fx.Int64(4)

    def _finish_bf16(acc, ks_rsrc, phys, off):
        zero = fx.Float32(0.0)
        s = (acc[0].maximumf(zero) + acc[1].maximumf(zero)) + (
            acc[2].maximumf(zero) + acc[3].maximumf(zero)
        )
        return s * inv_divisor

    def _finish_fp8(acc, ks_rsrc, phys, off):
        zero = fx.Float32(0.0)
        scale = _ld_f32(ks_rsrc, (phys * P + off) * 4)
        s = (
            (acc[0] * scale).maximumf(zero)
            + (acc[1] * scale).maximumf(zero)
            + (acc[2] * scale).maximumf(zero)
            + (acc[3] * scale).maximumf(zero)
        )
        return s * inv_divisor

    def _wave_src_tdm(storage, wave):
        return storage + wave * (tile_bytes // 4)  # Int32 elements

    def _wave_src_raw(storage, wave):
        return storage

    def _out_plan(col0, lane, safe_row, rows, columns):
        """Lane columns col0 + 32 j + lane and their clamped in-plane byte offsets."""
        last = rows * columns - 1
        cols = [col0 + j * WMMA_WAVE + lane for j in range_constexpr(n_pairs)]
        return cols, [_imin(safe_row * columns + c, last) * 4 for c in cols]

    storage_fn = _storage_tdm if use_tdm else _storage_raw
    wave_src = _wave_src_tdm if use_tdm else _wave_src_raw
    stage = _stage_tdm if use_tdm else _stage_raw
    stage_wait = _wait_tdm if use_tdm else _wait_raw
    kbase = _kbase_tdm if use_tdm else _kbase_raw
    if is_fp8:
        kfrag = _kfrag_fp8_tdm if use_tdm else _kfrag_fp8_raw
        ks_bytes, finish = _ks_bytes_fp8, _finish_fp8
    else:
        kfrag = _kfrag_bf16_tdm if use_tdm else _kfrag_bf16_raw
        ks_bytes, finish = _ks_bytes_bf16, _finish_bf16

    def _q_frags(q_rsrc, safe_row, n_in, kgrp):
        base_q = safe_row * q_row_bytes + (n_in % NUM_HEADS) * head_bytes + kgrp * 16
        return [
            _frag(q_rsrc, base_q + s * kstep_q) for s in range_constexpr(WMMA_K_STEPS)
        ]

    def _wait_and_q(q_rsrc, safe_row, n_in, kgrp):
        """K staging wait + Q fragments, in the schedule's order (plain Python)."""
        if sched == "base":
            stage_wait()
            return _q_frags(q_rsrc, safe_row, n_in, kgrp)
        # ROUND-FP8PERF qpre: the Q loads are in flight under the TDM gathers.
        qf = _q_frags(q_rsrc, safe_row, n_in, kgrp)
        stage_wait()
        return qf

    def _tile_k(
        ksrc, pt_rsrc, safe_req, req_ok, col0, n_in, kgrp, width, num_pages, nt
    ):
        """Page lookup + the WMMA_K_STEPS (converted) K fragments of N tile ``nt``."""
        local = nt * WMMA_N + n_in
        c = col0 + local
        logical = c // P
        off = c % P
        logical_ok = logical < width
        safe_logical = logical_ok.select(logical, 0)
        phys = _ld_i32(pt_rsrc, (safe_req * width + safe_logical) * 4)
        page_ok = req_ok & logical_ok & (phys >= 0) & (phys < num_pages)
        phys = page_ok.select(phys, 0)
        k_base = kbase(phys, off, local, kgrp)
        kf = [
            kfrag(ksrc, k_base + s * kstep_k, kgrp)
            for s in range_constexpr(WMMA_K_STEPS)
        ]
        return kf, page_ok, phys

    def _tile_mma(atom, qf, kf):
        acc_t = fx.make_rmem_tensor(8, fx.Float32)
        acc_t.store(Vec.filled(8, 0.0, fx.Float32))
        for s in range_constexpr(WMMA_K_STEPS):
            fx.gemm(atom, acc_t, _rmem8(qf[s]), _rmem8(kf[s]), acc_t)
        return acc_t.load()

    def _values(
        atom,
        ksrc,
        ks_rsrc,
        pt_rsrc,
        qf,
        safe_row,
        safe_req,
        req_ok,
        col0,
        lane,
        n_in,
        kgrp,
        width,
        num_pages,
        vis,
    ):
        """[value] per store pair j: column col0 + 32 j + lane (tile 2 j + kgrp)."""
        g0 = kgrp == 0
        accs, oks, physs = [], [], []
        if sched in ("base", "qpre"):
            # Champion structure (LLVM schedules conversion / WMMA freely).
            for nt in range_constexpr(n_tiles):
                kf, page_ok, phys = _tile_k(
                    ksrc,
                    pt_rsrc,
                    safe_req,
                    req_ok,
                    col0,
                    n_in,
                    kgrp,
                    width,
                    num_pages,
                    nt,
                )
                accs.append(_tile_mma(atom, qf, kf))
                oks.append(page_ok.select(fx.Int32(1), fx.Int32(0)))
                physs.append(phys)
        elif sched == "kall":
            # Every K fragment of the slab converted, then one WMMA burst.
            tiles = [
                _tile_k(
                    ksrc,
                    pt_rsrc,
                    safe_req,
                    req_ok,
                    col0,
                    n_in,
                    kgrp,
                    width,
                    num_pages,
                    nt,
                )
                for nt in range_constexpr(n_tiles)
            ]
            fx.rocdl.sched_barrier(0)
            for nt in range_constexpr(n_tiles):
                accs.append(_tile_mma(atom, qf, tiles[nt][0]))
            fx.rocdl.sched_barrier(0)
            for nt in range_constexpr(n_tiles):
                oks.append(tiles[nt][1].select(fx.Int32(1), fx.Int32(0)))
                physs.append(tiles[nt][2])
        else:
            # kpipe: region t = WMMAs of tile t-1 + conversion of tile t.
            tiles = []
            for nt in range_constexpr(n_tiles + 1):
                if nt < n_tiles:
                    tiles.append(
                        _tile_k(
                            ksrc,
                            pt_rsrc,
                            safe_req,
                            req_ok,
                            col0,
                            n_in,
                            kgrp,
                            width,
                            num_pages,
                            nt,
                        )
                    )
                if nt > 0:
                    accs.append(_tile_mma(atom, qf, tiles[nt - 1][0]))
                fx.rocdl.sched_barrier(0)
            for nt in range_constexpr(n_tiles):
                oks.append(tiles[nt][1].select(fx.Int32(1), fx.Int32(0)))
                physs.append(tiles[nt][2])
        neg_inf = fx.Float32(_NEG_INF)
        out = []
        for j in range_constexpr(n_pairs):
            a, b = accs[2 * j], accs[2 * j + 1]
            acc = [
                g0.select(fx.Float32(a[h]), fx.Float32(b[h]))
                for h in range_constexpr(4)
            ]
            ok = g0.select(oks[2 * j], oks[2 * j + 1]) != 0
            phys = g0.select(physs[2 * j], physs[2 * j + 1])
            c = col0 + j * WMMA_WAVE + lane
            score = finish(acc, ks_rsrc, phys, c % P)
            out.append((ok & (c < vis)).select(score, neg_inf))
        return out

    @flyc.kernel(name=kernel_name, known_block_size=[threads, 1, 1])
    def kernel(
        Q,
        KV,
        KS,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bx = fx.Int32(fx.block_idx.x)
        row = fx.Int32(fx.block_idx.y)
        wave = fx.Int32(fx.rocdl.wave_id())
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N
        kgrp = lane // WMMA_N
        col0 = bx * span + wave * wave_cols
        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        ks_rsrc = _plane(KS, ks_bytes(num_pages))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))
        # SGPR row metadata (grid.y == rows, still clamped for loads/stores).
        safe_row = _imin(row, rows - 1)
        req = _ld_uniform_i32(t2r_rsrc, safe_row * 4)
        req_ok = (req >= 0) & (req < num_requests)
        safe_req = req_ok.select(req, 0)
        ctx = _ld_uniform_i32(cl_rsrc, safe_req * 4)
        pos = ld_uniform_pos(qp_rsrc, safe_row)
        vis = req_ok.select(_imax(_imin(pos + 1, ctx), 0) // COMPRESS_RATIO, 0)
        writer = (bx == 0) & (tid == 0) & (row < rows)
        if writer:
            _st_i32(vis_rsrc, safe_row * 4, vis)
            _st_i32(rst_rsrc, safe_row * 4, fx.Int32(0))
        out_cols, out_offs = _out_plan(col0, lane, safe_row, rows, columns)
        storage = storage_fn(kv_rsrc)
        ksrc = wave_src(storage, wave)
        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        neg_inf = fx.Float32(_NEG_INF)
        live = (col0 < vis) & (row < rows)
        if live:
            stage(KV, storage, pt_rsrc, safe_req, req_ok, col0, width, num_pages, wave)
            qf = _wait_and_q(q_rsrc, safe_row, n_in, kgrp)
            vals = _values(
                atom,
                ksrc,
                ks_rsrc,
                pt_rsrc,
                qf,
                safe_row,
                safe_req,
                req_ok,
                col0,
                lane,
                n_in,
                kgrp,
                width,
                num_pages,
                vis,
            )
            for j1 in range_constexpr(n_pairs):
                if out_cols[j1] < columns:
                    _st_f32(logits_rsrc, out_offs[j1], vals[j1])
        dead = (col0 >= vis) & (row < rows)
        if dead:
            for j2 in range_constexpr(n_pairs):
                if out_cols[j2] < columns:
                    _st_f32(logits_rsrc, out_offs[j2], neg_inf)

    @flyc.jit
    def launch(
        Q,
        KV,
        KS,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        grid_x,
        grid_y,
        rows,
        columns,
        num_pages,
        num_requests,
        width,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel._func.__name__ = kernel_name
        kernel(
            Q,
            KV,
            KS,
            PT,
            T2R,
            QP,
            CL,
            LOGITS,
            VIS,
            RST,
            rows,
            columns,
            num_pages,
            num_requests,
            width,
        ).launch(
            grid=(fx.Int64(grid_x), fx.Int64(grid_y), 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch.kernel_name = kernel_name
    launch.dtype_tag = f"{dtag}_draft"
    launch.wg_waves = waves
    launch.cols_per_block = span
    launch.wave_cols = wave_cols
    launch.k_load = k_load
    return launch


# ---------------------------------------------------------------------------
# Round DECG (QSA_DECODE_GROUP, default 'all' since ROUND-DECG2): decode row-group kernel
# ---------------------------------------------------------------------------
# Decode (rows > requests, typically 4 consecutive rows per request) on the
# default bf16 r16 TDM nobar kernel: a wave owns 4 rows x 32 columns and runs
# WMMA_PASSES = 4 request-dedup passes, 3 of them empty for decode (each still
# re-loads the 4 rows' t2r/pos/ctx through s_buffer_load, waits TENSORcnt and
# DScnt, and evaluates the VGPR pass plan), and the 4 waves of a workgroup own
# 4 different requests.
#
# The row-group kernel: one workgroup = one group of DECG_ROWS = 4 consecutive
# rows (WMMA M = 4 rows x 4 heads = 16, exactly the r16 A layout) x 4 waves,
# each wave a private ``wave_cols`` (32/64) column slab:
#   * all 4 rows' metadata is SGPR (s_buffer_load; positions via the shared
#     ``_pos_loaders`` seam); ``same`` = the 4 (clamped) rows share one request
#     id -- a uniform SCC test;
#   * same (decode): ONE pass. vmax = max visible of the group; col0 >= vmax
#     stores -inf only; else the slab's K is TDM-gathered once into the
#     wave-private padded tile (wave_cols/8 8-row gathers, SGPR descriptors),
#     ``s_wait_tensorcnt 0`` (no WG barrier: no other wave reads the tile;
#     kernel-wiki mi400-tdm-sync: TENSORcnt decrements on tensor-done of this
#     wave), WMMA, stores of all 4 rows;
#   * mixed (any token_to_request): 4 passes p = 0..3, pass p handles row p
#     alone (own mask row == p) with request req_p and vmax = visible[p]; an
#     ``s_wait_dscnt 0`` after each live pass is the WAR guard before the next
#     pass's gather overwrites the tile (the nobar r16 rule). Every row is
#     stored exactly once in both regimes;
#   * per-slot math == the champion r16 tile: lane (n_in, kgrp) A = Q[t0 +
#     n_in // 4, head n_in % 4], B = K[column], C elements 0..3 / 4..7 = heads
#     of rows 2 kgrp / 2 kgrp + 1; post-processing in the champion's order per
#     dtype (bf16: (r0+r1)+(r2+r3); fp8: relu(a_h*scale) left-to-right), then
#     * 1/divisor; masks (page_ok & c < visible[row]) / -inf; stores at the
#     r16 clamped in-plane offsets. fp8 K uses the GPU-verified exact pk8
#     converter of the draft kernel (bit-equal to the champion converter).
# Row metadata (visible_groups / row_starts) is written by (bx 0, wave 0,
# lanes 0..3). Grid = (ceil(columns / (4 wave_cols)), ceil(rows / 4)).
#
# ROUND-DECG2: default ON (QSA_DECODE_GROUP=all; '0' opts out to the previous
# c45b24e0 routing). GPU (a08-1 GPU3, root run 2026-11-03): 22/22 correct with
# all/w32, decode 0.64-0.74x ATOM. QSA_DECODE_GROUP_COLS (ROUND-FP8PERF default
# 'auto' = bf16 w32 / fp8 w64; '32' / '64' force both dtypes) selects the slab. ROUND-W64FIX2: fp8 w64 is re-enabled with the
# compiler-visible pk8 converter (GPU root a08-1 GPU3 2026-11-03: 22/22, decode
# fp8 c65536 26.8 vs w32 26.6 us); its earlier failure was the unpadded WMMA
# hazard of the inline-asm converter, so with QSA_FP8_CVT_ASM=1 fp8 stays w32
# and the builder rejects fp8 w64.
DECG_ROWS = WMMA_ROWS_PER_WAVE  # 4 rows x 4 heads = WMMA M 16
DECG_WAVES = 4
_DECODE_GROUP = os.environ.get("QSA_DECODE_GROUP", "all").strip().lower()
if _DECODE_GROUP not in ("0", "1", "bf16", "fp8", "all"):
    raise ValueError(
        f"QSA_DECODE_GROUP must be '0', '1' (= bf16), 'bf16', 'fp8' or 'all', got {_DECODE_GROUP!r}"
    )
_DECODE_GROUP_BF16 = _DECODE_GROUP in ("1", "bf16", "all")
_DECODE_GROUP_FP8 = _DECODE_GROUP in ("fp8", "all")
# ROUND-FP8PERF: per-dtype default ('auto'): bf16 w32, fp8 w64 (root GPU a08-1
# GPU3, 5fb8740e: decode fp8 c65536 w64 26.6 vs w32 27.6 us, c32768 14.7 vs 15.5;
# bf16 w64 is slower: 29.0 / 16.1 vs 27.0 / 15.4). '32' / '64' force both dtypes.
_DECODE_GROUP_COLS = os.environ.get("QSA_DECODE_GROUP_COLS", "auto").strip().lower()
if _DECODE_GROUP_COLS not in ("auto", "32", "64"):
    raise ValueError(
        f"QSA_DECODE_GROUP_COLS must be 'auto', '32' or '64', got {_DECODE_GROUP_COLS!r}"
    )
_DECODE_GROUP_COLS = None if _DECODE_GROUP_COLS == "auto" else int(_DECODE_GROUP_COLS)
_DECODE_GROUP_AUTO_COLS = {False: 32, True: 64}  # is_fp8 -> wave_cols
# ROUND-R2E: 'auto' slab of kernel-native 3-row groups (group_rows=3, MTP verify3):
# fp8 w32 (GPU verify3 C8-C64 0.90-0.94x native; w64 1.05-1.15x; C>=192 w64 ~ w32).
_DECODE_GROUP_AUTO_COLS_G3 = {False: 32, True: 32}  # is_fp8 -> wave_cols
# ROUND-FP8PERF: decode-group fp8 tile schedule (intrinsic converter only).
#   base  (default) 5fb8740e structure (per tile: convert, 4 WMMAs, epilogue;
#         LLVM interleaves and pads the WMMA hazards with v_nop);
#   kpipe software pipeline as the draft kernel's: tile t's WMMAs share a
#         sched_barrier-fenced region with tile t+1's K conversion, epilogues
#         after the last WMMA. Non-default get ``_s<variant>``. FP8PERF2 GPU: base 27.0/14.7 us
_DECG_FP8_SCHED_DEFAULT = "base"
_DECG_FP8_SCHED = (
    os.environ.get("QSA_DECG_FP8_SCHED", _DECG_FP8_SCHED_DEFAULT).strip().lower()
)
if _DECG_FP8_SCHED not in ("base", "kpipe"):
    raise ValueError(
        f"QSA_DECG_FP8_SCHED must be 'base' or 'kpipe', got {_DECG_FP8_SCHED!r}"
    )


def decg_fp8_sched() -> str:
    """Effective decode-group fp8 schedule: the knob, or 'base' with the asm converter."""
    return "base" if _FP8_CVT_ASM else _DECG_FP8_SCHED


# ROUND-R2 decode-group knobs (builder keywords; defaults = 536667d5 kernel):
#   group_rows 4 | 3   rows per group. 3 = MTP verify with 3 rows per request
#                      (rows = 3C): each group is ONE request -> one pass, one K
#                      slab gather per 3 rows (K traffic of the 4-row decode
#                      group per request); WMMA M = 3 rows x 4 heads = 12 of 16
#                      slots, slot row 3 (Q of the next group, clamped) is
#                      computed but never stored. grid.y = ceil(rows / 3).
#   row_mode  base     536667d5 plan (one pass if all 4 rows share a request,
#                      else 4 single-row passes);
#             dedup    request-dedup passes (pass p iff row p's request is new in
#                      the group; it owns every row of that request): a 4-row
#                      group straddling two 3-row requests costs 2 passes, not 4;
#             pad      dedup + rows with an invalid token_to_request are PADDING:
#                      no logits / visible_groups / row_starts stores at all
#                      (op-level 3 -> 4 token padding with t2r = -1 on pad rows;
#                      the real rows are bit-identical to the unpadded call).
#   wg_waves  4 | 2 | 1  wave column slabs per workgroup (short contexts / small
#                      C: more, smaller workgroups -> more CUs; span = waves*wc).
# Env (wrapper only, default off): QSA_DECG_ROWS3=1 routes rows == 3*requests to
# group_rows=3; QSA_DECG_MODE=dedup|pad; QSA_DECG_WG_WAVES=4|2|1.
DECG_ROW_MODES = ("base", "dedup", "pad")
_WG_WAVES = (4, 2, 1)


def _env_choice(name, value, choices):
    v = value.strip().lower()
    if v not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {v!r}")
    return v


# Direct os.environ.get reads: the op profile pins every import-time switch.
_DECG_ROWS3 = (
    _env_choice("QSA_DECG_ROWS3", os.environ.get("QSA_DECG_ROWS3", "0"), ("0", "1"))
    == "1"
)
_DECG_MODE = _env_choice(
    "QSA_DECG_MODE", os.environ.get("QSA_DECG_MODE", "base"), DECG_ROW_MODES
)
_DECG_WG_WAVES = int(
    _env_choice(
        "QSA_DECG_WG_WAVES", os.environ.get("QSA_DECG_WG_WAVES", "4"), ("4", "2", "1")
    )
)


def _check_decg_knobs(group_rows, row_mode, wg_waves):
    g, mode, w = int(group_rows), str(row_mode), int(wg_waves)
    if g not in (3, 4):
        raise ValueError("decode-group group_rows must be 3 or 4")
    if mode not in DECG_ROW_MODES:
        raise ValueError(f"decode-group row_mode must be one of {DECG_ROW_MODES}")
    if w not in _WG_WAVES:
        raise ValueError(f"decode-group wg_waves must be one of {_WG_WAVES}")
    if g != DECG_ROWS and mode == "base":
        mode = "dedup"  # 3-row groups always use the dedup plan
    return g, mode, w


def _decg_knob_tag(group_rows, row_mode, waves):
    """Kernel-name suffix of the ROUND-R2 knobs ('' for the defaults)."""
    return (
        (f"_g{group_rows}" if group_rows != DECG_ROWS else "")
        + (f"_{row_mode}" if row_mode != "base" else "")
        + (f"_wg{waves}" if waves != DECG_WAVES else "")
    )


def decode_group_passes_x(reqs, real):
    """Trace-free mirror of ``_group_plans_x`` (dedup / pad / 3-row groups).

    ``reqs`` = raw request ids of the group's rows, ``real`` = stored rows.
    Returns [(active, owned rows)] per pass p < len(reqs).
    """
    out = []
    for p in range(len(reqs)):
        active = real[p] and not any(real[q] and reqs[q] == reqs[p] for q in range(p))
        own = [r for r in range(len(reqs)) if real[r] and reqs[r] == reqs[p]]
        out.append((bool(active), own))
    return out


def decode_group_knobs(rows: int, num_requests: int):
    """Wrapper defaults of the ROUND-R2 knobs: (group_rows, row_mode, wg_waves)."""
    g = 3 if (_DECG_ROWS3 and int(rows) == 3 * int(num_requests)) else DECG_ROWS
    return _check_decg_knobs(g, _DECG_MODE, _DECG_WG_WAVES)


def decode_group_route(
    rows: int, num_requests: int, is_fp8: bool, kv_ptr: int, group_rows: int = DECG_ROWS
):
    """Pure-Python routing of the decode row-group kernel: ``None`` or wave_cols.

    Only when QSA_DECODE_GROUP (default 'all') enables the dtype, raw access,
    the dtype's WMMA path, rows > requests (decode; draft has its own kernel),
    rows % 4 == 0, the champion would pick an r16 kernel (rows < 32 * requests:
    prefill keeps r32/persist), and a 128 B aligned cache base (TDM). The
    kernel itself is exact for any token_to_request (mixed groups fall back
    in-kernel). wave_cols = QSA_DECODE_GROUP_COLS ('auto' default: bf16 32,
    fp8 64); fp8 is 32 whenever the inline-asm converter is forced
    (QSA_FP8_CVT_ASM=1: its w64 build is GPU-wrong).
    """
    if not (_DECODE_GROUP_FP8 if is_fp8 else _DECODE_GROUP_BF16):
        return None
    if _ACCESS != "raw" or (_FP8_PATH if is_fp8 else _BF16_PATH) != "wmma":
        return None
    rows, num_requests = int(rows), int(num_requests)
    if rows <= num_requests or rows % int(group_rows):
        return None
    rpb = (
        wmma_rows_per_block(rows, num_requests)
        if is_fp8
        else wmma_bf16_rows_per_block(rows, num_requests)
    )
    if rpb != 16 or int(kv_ptr) % 128:
        return None
    if is_fp8 and _FP8_CVT_ASM:
        return 32  # asm-converter fp8 w64 is GPU-incorrect
    if _DECODE_GROUP_COLS is None:
        if int(group_rows) == 3:  # ROUND-R2E
            return _DECODE_GROUP_AUTO_COLS_G3[bool(is_fp8)]
        return _DECODE_GROUP_AUTO_COLS[bool(is_fp8)]
    return _DECODE_GROUP_COLS


def decode_group_grid(
    rows: int,
    columns: int,
    wave_cols: int,
    group_rows: int = DECG_ROWS,
    wg_waves: int = DECG_WAVES,
) -> tuple[int, int]:
    """grid.x = column spans of ``wg_waves`` slabs, grid.y = ``group_rows``-row groups."""
    if wave_cols not in (32, 64):
        raise ValueError("decode-group wave_cols must be 32 or 64")
    span = int(wg_waves) * wave_cols
    g = int(group_rows)
    return (int(columns) + span - 1) // span, (int(rows) + g - 1) // g


def decode_group_passes(reqs4, rows_ok4):
    """Trace-free mirror of the kernel's pass plan for one group (CPU tests).

    ``reqs4`` = request ids of the group's 4 clamped rows, ``rows_ok4`` = row < rows.
    Returns [(active, owned rows)] for p = 0..3: one pass owning every valid row
    when all 4 ids are equal, else pass p owns row p alone.
    """
    same = all(r == reqs4[0] for r in reqs4)
    out = []
    for p in range(DECG_ROWS):
        active = rows_ok4[0] if p == 0 else (not same) and rows_ok4[p]
        own = [rows_ok4[r] and (r == p or same) for r in range(DECG_ROWS)]
        out.append((bool(active), [r for r in range(DECG_ROWS) if own[r]]))
    return out


def decode_group_store_plan(rows: int, columns: int, wave_cols: int, t2r):
    """Trace-free mirror of the decode-group kernel's store map (CPU tests).

    Yields (row, column) per predicated store (pass, block, wave, lane, tile,
    lane row a/b), in kernel order, using the in-kernel pass plan.
    """
    gx, gy = decode_group_grid(rows, columns, wave_cols)
    for by in range(gy):
        t0 = by * DECG_ROWS
        safe = [min(t0 + r, rows - 1) for r in range(DECG_ROWS)]
        plan = decode_group_passes(
            [int(t2r[s]) for s in safe], [t0 + r < rows for r in range(DECG_ROWS)]
        )
        for bx in range(gx):
            for wave in range(DECG_WAVES):
                col0 = bx * DECG_WAVES * wave_cols + wave * wave_cols
                for active, owned in plan:
                    if not active:
                        continue
                    for lane in range(WMMA_WAVE):
                        n_in, kgrp = lane % WMMA_N, lane // WMMA_N
                        for nt in range(wave_cols // WMMA_N):
                            c = col0 + nt * WMMA_N + n_in
                            for r in (2 * kgrp, 2 * kgrp + 1):
                                if r in owned and c < columns:
                                    yield t0 + r, c


@cache
def _build_decode_group_kernel(
    is_fp8: bool,
    divisor: float,
    wave_cols: int = 32,
    pos64: bool = False,
    page_size: int = PAGE_SIZE,
    group_rows: int = DECG_ROWS,
    row_mode: str = "base",
    wg_waves: int = DECG_WAVES,
):
    """Decode row-group specialization (see the Round DECG / ROUND-R2 blocks).

    ROUND-R2 knobs (defaults = the 536667d5 kernel, trace-identical):
    ``group_rows`` 4 | 3 (rows per group; 3 = MTP verify with 3 rows per request,
    the WMMA M slots of lane row 3 are never stored), ``row_mode`` 'base' |
    'dedup' | 'pad' (see ``decode_group_passes_x``), ``wg_waves`` 4 | 2 | 1
    (wave column slabs per workgroup).
    """
    _, ld_uniform_pos, qp_bytes, qp_tag = _pos_loaders(pos64)
    P = _check_page_size(page_size)
    group_rows, row_mode, waves = _check_decg_knobs(group_rows, row_mode, wg_waves)
    threads = WMMA_WAVE * waves
    n_pass = group_rows
    if wave_cols not in (32, 64):
        raise ValueError("decode-group wave_cols must be 32 or 64")
    if is_fp8 and wave_cols == 64 and _FP8_CVT_ASM:
        raise ValueError(
            "fp8 decode-group w64 with the inline-asm converter (QSA_FP8_CVT_ASM=1) "
            "is GPU-incorrect (unpadded WMMA hazard), disabled"
        )
    if _ACCESS != "raw":
        raise ValueError("the decode-group kernel requires QSA_G2_ACCESS=raw")
    n_tiles = wave_cols // WMMA_N  # 2 / 4 WMMA N tiles per wave
    span = waves * wave_cols  # columns per workgroup
    inv_divisor = 1.0 / divisor
    q_row_bytes = NUM_HEADS * HEAD_DIM * 2  # 1024 B
    head_bytes = HEAD_DIM * 2  # 256 B
    kstep_q = WMMA_K * 2  # 64 B of a Q row per step
    key_row_bytes = HEAD_DIM if is_fp8 else HEAD_DIM * 2
    page_bytes = P * key_row_bytes
    kstep_k = WMMA_K * (1 if is_fp8 else 2)  # 32 B fp8 / 64 B bf16
    pitch = TDM_FP8_K_ROW_BYTES if is_fp8 else TDM_K_ROW_BYTES  # 144 / 272 B
    tile_bytes = wave_cols * pitch  # per wave
    gather = _tdm_fp8_k_gather if is_fp8 else _tdm_k_gather
    dtag = "fp8" if is_fp8 else "bf16"
    sched = decg_fp8_sched() if is_fp8 else "base"  # ROUND-FP8PERF; bf16 unchanged
    sched_tag = (
        f"_s{sched}"
        if (is_fp8 and not _FP8_CVT_ASM and sched != _DECG_FP8_SCHED_DEFAULT)
        else ""
    )
    kernel_name = (
        (
            f"qsa_paged_mqa_logits_page4_gfx1250_{dtag}_decg"
            f"_D{HEAD_DIM}_H{NUM_HEADS}_P{P}_w{wave_cols}_flydsl_tdm"
        )
        + qp_tag
        + (_CVT_ASM_TAG if is_fp8 else "")
        + sched_tag
        + _decg_knob_tag(group_rows, row_mode, waves)
    )

    def _i01(cond):
        return cond.select(fx.Int32(1), fx.Int32(0))

    def _frag(rsrc, byte_off):
        lo = _ld_vec(rsrc, byte_off, 4, fx.Int32)
        hi = _ld_vec(rsrc, byte_off + 32, 4, fx.Int32)
        return lo.shuffle(hi, list(range(8)))

    def _rmem8(vec):
        t = fx.make_rmem_tensor(8, fx.Int32)
        t.store(vec)
        return t

    def _storage():
        return (
            fx.SharedAllocator(base_alignment=128, static=True)
            .allocate(fx.Array[fx.Int32, waves * tile_bytes // 4, 128])
            .peek()
            .ptr
        )

    def _group_meta(t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests):
        """SGPR metadata of the group's 4 rows (clamped loads)."""
        out = []
        for i in range_constexpr(DECG_ROWS):
            row = t0 + i
            row_ok = row < rows
            safe_row = _imin(row, rows - 1)
            req = _ld_uniform_i32(t2r_rsrc, safe_row * 4)
            req_ok = (req >= 0) & (req < num_requests)
            safe_req = req_ok.select(req, 0)
            ctx = _ld_uniform_i32(cl_rsrc, safe_req * 4)
            pos = ld_uniform_pos(qp_rsrc, safe_row)
            vis = req_ok.select(_imax(_imin(pos + 1, ctx), 0) // COMPRESS_RATIO, 0)
            out.append(_RowMeta(row, row_ok, req, req_ok, safe_req, vis))
        return out

    def _group_plans(meta, kgrp):
        """Pass p: (p == 0) always; p > 0 only for mixed groups (row p alone)."""
        same = meta[0].row_ok
        for i in range_constexpr(1, DECG_ROWS):
            same = same & (meta[i].req == meta[0].req)
        vall = meta[0].vis
        for i in range_constexpr(1, DECG_ROWS):
            vall = _imax(vall, meta[i].vis)
        g0 = kgrp == 0
        plans = []
        for p in range_constexpr(DECG_ROWS):
            mp = meta[p]
            if p == 0:
                active = mp.row_ok
                vmax = same.select(vall, mp.vis)
                own = [
                    _i01(meta[r].row_ok) if r == 0 else _i01(meta[r].row_ok & same)
                    for r in range_constexpr(DECG_ROWS)
                ]
            else:
                active = mp.row_ok & (same == 0)
                vmax = mp.vis
                own = [
                    _i01(meta[r].row_ok) if r == p else fx.Int32(0)
                    for r in range_constexpr(DECG_ROWS)
                ]
            own_a = g0.select(own[0], own[2]) != 0
            own_b = g0.select(own[1], own[3]) != 0
            vis_a = g0.select(meta[0].vis, meta[2].vis)
            vis_b = g0.select(meta[1].vis, meta[3].vis)
            plans.append(
                _Pass(
                    active,
                    vmax,
                    mp.safe_req,
                    mp.req_ok,
                    own_a,
                    own_b,
                    vis_a,
                    vis_b,
                    kgrp,
                )
            )
        return plans

    def _pick_vis(lane, meta):
        v = meta[DECG_ROWS - 1].vis
        for i in range_constexpr(DECG_ROWS - 2, -1, -1):
            v = (lane == i).select(meta[i].vis, v)
        return v

    # ---- ROUND-R2 row modes (group_rows 3 or row_mode != 'base' only) --------
    def _group_meta_x(t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests):
        """SGPR metadata of the group's ``group_rows`` rows + ``real`` (stored rows).

        real = row < min(rows, t0 + group_rows) [& valid request in 'pad' mode:
        padded rows (token_to_request = -1 / out of range) get NO stores at all].
        """
        lim = _imin(rows, t0 + group_rows)
        out = []
        for i in range_constexpr(group_rows):
            row = t0 + i
            row_ok = row < lim
            safe_row = _imin(row, rows - 1)
            req = _ld_uniform_i32(t2r_rsrc, safe_row * 4)
            req_ok = (req >= 0) & (req < num_requests)
            safe_req = req_ok.select(req, 0)
            ctx = _ld_uniform_i32(cl_rsrc, safe_req * 4)
            pos = ld_uniform_pos(qp_rsrc, safe_row)
            vis = req_ok.select(_imax(_imin(pos + 1, ctx), 0) // COMPRESS_RATIO, 0)
            m = _RowMeta(row, row_ok, req, req_ok, safe_req, vis)
            m.real = (row_ok & req_ok) if row_mode == "pad" else row_ok
            out.append(m)
        return out

    def _group_plans_x(meta, kgrp):
        """Request-dedup passes: pass p is active iff row p is real and no earlier
        real row has its request; it owns every real row of that request
        (each real row exactly once; one pass when the group is one request)."""
        g0 = kgrp == 0
        plans = []
        for p in range_constexpr(group_rows):
            mp = meta[p]
            active = mp.real
            for q in range_constexpr(p):
                dup = meta[q].real & (meta[q].req == mp.req)
                active = active & (dup == 0)
            own, vmax = [], fx.Int32(0)
            for r in range_constexpr(group_rows):
                mine = meta[r].real & (meta[r].req == mp.req)
                own.append(_i01(mine))
                vmax = _imax(vmax, mine.select(meta[r].vis, 0))
            while len(own) < DECG_ROWS:  # lane row 3 of a 3-row group: never stored
                own.append(fx.Int32(0))
            vis4 = [
                meta[r].vis if r < group_rows else fx.Int32(0) for r in range(DECG_ROWS)
            ]
            own_a = g0.select(own[0], own[2]) != 0
            own_b = g0.select(own[1], own[3]) != 0
            vis_a = g0.select(vis4[0], vis4[2])
            vis_b = g0.select(vis4[1], vis4[3])
            plans.append(
                _Pass(
                    active,
                    vmax,
                    mp.safe_req,
                    mp.req_ok,
                    own_a,
                    own_b,
                    vis_a,
                    vis_b,
                    kgrp,
                )
            )
        return plans

    def _writer_x(lane, meta, bx, wave, my_row, rows):
        """Row-metadata writer lanes: lane < group_rows of (bx 0, wave 0), real rows only."""
        real = _i01(meta[group_rows - 1].real)
        for i in range_constexpr(group_rows - 2, -1, -1):
            real = (lane == i).select(_i01(meta[i].real), real)
        return (
            (bx == 0)
            & (wave == 0)
            & (lane < group_rows)
            & (my_row < rows)
            & (real != 0)
        )

    def _pick_vis_x(lane, meta):
        v = meta[group_rows - 1].vis
        for i in range_constexpr(group_rows - 2, -1, -1):
            v = (lane == i).select(meta[i].vis, v)
        return v

    def _head_base(
        t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests, kgrp, lane, bx, wave
    ):
        """536667d5 kernel prologue, same op order: (plans, my_vis, meta_off, writer)."""
        meta = _group_meta(t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests)
        plans = _group_plans(meta, kgrp)
        my_vis = _pick_vis(lane, meta)
        my_row = t0 + _imin(lane, DECG_ROWS - 1)
        meta_off = _imin(my_row, rows - 1) * 4
        writer = (bx == 0) & (wave == 0) & (lane < DECG_ROWS) & (my_row < rows)
        return plans, my_vis, meta_off, writer

    def _head_x(
        t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests, kgrp, lane, bx, wave
    ):
        """ROUND-R2 dedup / pad / 3-row prologue."""
        meta = _group_meta_x(t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests)
        plans = _group_plans_x(meta, kgrp)
        my_vis = _pick_vis_x(lane, meta)
        my_row = t0 + _imin(lane, group_rows - 1)
        meta_off = _imin(my_row, rows - 1) * 4
        writer = _writer_x(lane, meta, bx, wave, my_row, rows)
        return plans, my_vis, meta_off, writer

    group_head = _head_base if row_mode == "base" else _head_x

    def _store_offsets(t0, kgrp, cols, rows, columns):
        """The r16 clamped byte offsets of the lane's rows a/b per tile column."""
        last = rows * columns - 1
        row_a = _imin(t0 + 2 * kgrp, rows - 1)
        row_b = _imin(t0 + 2 * kgrp + 1, rows - 1)
        out = []
        for c in cols:
            out.append(_imin(row_a * columns + c, last) * 4)
            out.append(_imin(row_b * columns + c, last) * 4)
        return out

    def _q_frags(q_rsrc, t0, rows, n_in, kgrp):
        q_row = _imin(t0 + n_in // NUM_HEADS, rows - 1)
        base = q_row * q_row_bytes + (n_in % NUM_HEADS) * head_bytes + kgrp * 16
        return [
            _frag(q_rsrc, base + s * kstep_q) for s in range_constexpr(WMMA_K_STEPS)
        ]

    def _issue(KV, storage, pt_rsrc, pp, col0, width, num_pages, wave):
        # SGPR-only descriptors (wave_id, pass request, page table); wave-private tile.
        for group in range_constexpr(wave_cols // 8):
            if P == 4:
                row_ids = []
                for page in range_constexpr(2):
                    logical = col0 // 4 + group * 2 + page
                    logical_ok = logical < width
                    safe_logical = logical_ok.select(logical, 0)
                    phys = _ld_uniform_i32(
                        pt_rsrc, (pp.safe_req * width + safe_logical) * 4
                    )
                    valid = pp.req_ok & logical_ok & (phys >= 0) & (phys < num_pages)
                    phys = valid.select(phys, 0)
                    for r in range_constexpr(4):
                        row_ids.append(phys * 4 + r)
            elif P == 16:  # P = 16: one page lookup per 8-row group (col0 % 32 == 0)
                row_ids = _p16_row_ids(
                    pt_rsrc,
                    pp.safe_req,
                    width,
                    num_pages,
                    col0 // 16 + group // 2,
                    (group % 2) * 8,
                    pp.req_ok,
                )
            else:  # P = 8: the 8-row group IS page col0/8 + group (one lookup)
                row_ids = _p16_row_ids(
                    pt_rsrc,
                    pp.safe_req,
                    width,
                    num_pages,
                    col0 // 8 + group,
                    0,
                    pp.req_ok,
                    8,
                )
            gather(
                KV,
                storage,
                row_ids,
                num_pages,
                wave * tile_bytes + group * 8 * pitch,
                P,
            )

    def _tensor_done():
        # Same-wave consumer: data is in LDS once TENSORcnt reaches 0.
        fx.rocdl.tdm_ops.tensor_wait(0)

    def _war_guard():
        # This wave's LDS reads retired before a later pass's gather overwrites the tile.
        fx.rocdl.s_wait_dscnt(0)

    def _kfrag_bf16(src, byte_off, kgrp):
        lo = Vec(
            fx.ptr_load(src + byte_off // 4, result_type=Vec.make_type(4, fx.Int32))
        )
        hi = Vec(
            fx.ptr_load(
                src + (byte_off + 32) // 4, result_type=Vec.make_type(4, fx.Int32)
            )
        )
        return lo.shuffle(hi, list(range(8)))

    def _kfrag_fp8(src, byte_off, kgrp):
        lo = Vec(
            fx.ptr_load(src + byte_off // 4, result_type=Vec.make_type(4, fx.Int32))
        )
        hi = Vec(
            fx.ptr_load(
                src + (byte_off + 16) // 4, result_type=Vec.make_type(4, fx.Int32)
            )
        )
        words = []
        for vec in (lo, hi):
            src0 = (kgrp == 0).select(fx.Int32(vec[0]), fx.Int32(vec[2]))
            src1 = (kgrp == 0).select(fx.Int32(vec[1]), fx.Int32(vec[3]))
            cvt = _e4m3x8_to_bf16x4_fast(src0, src1)
            for j in range_constexpr(4):
                words.append(fx.Int32(cvt[j]))
        return Vec.from_elements(words, dtype=fx.Int32)

    def _scale_fp8(ks_rsrc, phys, off):
        return _ld_f32(ks_rsrc, (phys * P + off) * 4)

    def _scale_bf16(ks_rsrc, phys, off):
        return None

    def _finish_bf16(a, scale):
        zero = fx.Float32(0.0)
        s = (a[0].maximumf(zero) + a[1].maximumf(zero)) + (
            a[2].maximumf(zero) + a[3].maximumf(zero)
        )
        return s * inv_divisor

    def _finish_fp8(a, scale):
        zero = fx.Float32(0.0)
        s = (
            (a[0] * scale).maximumf(zero)
            + (a[1] * scale).maximumf(zero)
            + (a[2] * scale).maximumf(zero)
            + (a[3] * scale).maximumf(zero)
        )
        return s * inv_divisor

    def _ks_bytes_fp8(num_pages):
        return _plane_bytes(num_pages, P * 4)

    def _ks_bytes_bf16(num_pages):
        return fx.Int64(4)

    if is_fp8:
        kfrag, scale_fn, finish, ks_bytes = (
            _kfrag_fp8,
            _scale_fp8,
            _finish_fp8,
            _ks_bytes_fp8,
        )
        kgrp_bytes = 0
    else:
        kfrag, scale_fn, finish, ks_bytes = (
            _kfrag_bf16,
            _scale_bf16,
            _finish_bf16,
            _ks_bytes_bf16,
        )
        kgrp_bytes = 16

    def _pass_values(
        atom,
        ksrc,
        ks_rsrc,
        pt_rsrc,
        qf,
        pp,
        col0,
        cols,
        offs,
        n_in,
        columns,
        width,
        num_pages,
    ):
        """[(store_ok, byte_off, value)]: rows a/b of every tile column (r16 order)."""
        neg_inf = fx.Float32(_NEG_INF)

        def _tile_k(nt):
            c = cols[nt]
            local = nt * WMMA_N + n_in
            logical = c // P
            off = c % P
            logical_ok = logical < width
            safe_logical = logical_ok.select(logical, 0)
            phys = _ld_i32(pt_rsrc, (pp.safe_req * width + safe_logical) * 4)
            page_ok = pp.req_ok & logical_ok & (phys >= 0) & (phys < num_pages)
            phys = page_ok.select(phys, 0)
            k_base = local * pitch + pp.kgrp * kgrp_bytes
            kf = [
                kfrag(ksrc, k_base + s * kstep_k, pp.kgrp)
                for s in range_constexpr(WMMA_K_STEPS)
            ]
            scale = scale_fn(ks_rsrc, phys, off)
            return kf, page_ok, scale

        def _tile_mma(kf):
            acc_t = fx.make_rmem_tensor(8, fx.Float32)
            acc_t.store(Vec.filled(8, 0.0, fx.Float32))
            for s in range_constexpr(WMMA_K_STEPS):
                # D[m = 4*row + head, n = column] += Q[m, k] * K[n, k]
                fx.gemm(atom, acc_t, _rmem8(qf[s]), _rmem8(kf[s]), acc_t)
            return acc_t.load()

        if sched == "kpipe":
            # ROUND-FP8PERF: region t = WMMAs of tile t-1 + K conversion of tile t
            # (co-execution, kernel-wiki mi400-valu-coexec Â§2.2), sched_barrier-fenced.
            tiles, accs = [], []
            for nt in range_constexpr(n_tiles + 1):
                if nt < n_tiles:
                    tiles.append(_tile_k(nt))
                if nt > 0:
                    accs.append(_tile_mma(tiles[nt - 1][0]))
                fx.rocdl.sched_barrier(0)
        out = []
        for nt in range_constexpr(n_tiles):
            c = cols[nt]
            if sched == "kpipe":
                page_ok, scale, acc = tiles[nt][1], tiles[nt][2], accs[nt]
            else:
                kf, page_ok, scale = _tile_k(nt)
                acc = _tile_mma(kf)
            sa = finish([fx.Float32(acc[h]) for h in range_constexpr(4)], scale)
            sb = finish([fx.Float32(acc[4 + h]) for h in range_constexpr(4)], scale)
            col_ok = c < columns
            va = (page_ok & (c < pp.vis_a)).select(sa, neg_inf)
            vb = (page_ok & (c < pp.vis_b)).select(sb, neg_inf)
            out.append((pp.own_a & col_ok, offs[2 * nt], va))
            out.append((pp.own_b & col_ok, offs[2 * nt + 1], vb))
        return out

    def _pass_masks(pp, cols, offs, columns):
        out = []
        for nt in range_constexpr(n_tiles):
            col_ok = cols[nt] < columns
            out.append((pp.own_a & col_ok, offs[2 * nt]))
            out.append((pp.own_b & col_ok, offs[2 * nt + 1]))
        return out

    @flyc.kernel(name=kernel_name, known_block_size=[threads, 1, 1])
    def kernel(
        Q,
        KV,
        KS,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        rows: fx.Int32,
        columns: fx.Int32,
        num_pages: fx.Int32,
        num_requests: fx.Int32,
        width: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bx = fx.Int32(fx.block_idx.x)
        grp = fx.Int32(fx.block_idx.y)
        wave = fx.Int32(fx.rocdl.wave_id())
        lane = tid % WMMA_WAVE
        n_in = lane % WMMA_N
        kgrp = lane // WMMA_N
        t0 = grp * group_rows
        col0 = bx * span + wave * wave_cols
        q_rsrc = _plane(Q, _plane_bytes(rows, q_row_bytes))
        _kv_rsrc = _plane(KV, _plane_bytes(num_pages, page_bytes))
        ks_rsrc = _plane(KS, ks_bytes(num_pages))
        pt_rsrc = _plane(PT, _plane_bytes(num_requests, 4, width))
        t2r_rsrc = _plane(T2R, _plane_bytes(rows, 4))
        qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))
        cl_rsrc = _plane(CL, _plane_bytes(num_requests, 4))
        logits_rsrc = _plane(LOGITS, _plane_bytes(rows, 4, columns))
        vis_rsrc = _plane(VIS, _plane_bytes(rows, 4))
        rst_rsrc = _plane(RST, _plane_bytes(rows, 4))
        # Trace-time row-mode selection lives outside the kernel AST (an `if` here
        # would be lowered as a runtime branch); 'base' = the 536667d5 sequence.
        plans, my_vis, meta_off, writer = group_head(
            t2r_rsrc, qp_rsrc, cl_rsrc, t0, rows, num_requests, kgrp, lane, bx, wave
        )
        if writer:
            _st_i32(vis_rsrc, meta_off, my_vis)
            _st_i32(rst_rsrc, meta_off, fx.Int32(0))
        atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_N, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32)
        )
        cols = [col0 + nt * WMMA_N + n_in for nt in range_constexpr(n_tiles)]
        offs = _store_offsets(t0, kgrp, cols, rows, columns)
        neg_inf = fx.Float32(_NEG_INF)
        storage = _storage()
        ksrc = storage + wave * (tile_bytes // 4)
        for p in range_constexpr(n_pass):
            pp = plans[p]
            if pp.active:
                if col0 < pp.vmax:
                    _issue(KV, storage, pt_rsrc, pp, col0, width, num_pages, wave)
                    qf = _q_frags(q_rsrc, t0, rows, n_in, kgrp)
                    _tensor_done()
                    res = _pass_values(
                        atom,
                        ksrc,
                        ks_rsrc,
                        pt_rsrc,
                        qf,
                        pp,
                        col0,
                        cols,
                        offs,
                        n_in,
                        columns,
                        width,
                        num_pages,
                    )
                    for j1 in range_constexpr(len(res)):
                        if res[j1][0]:
                            _st_f32(logits_rsrc, res[j1][1], res[j1][2])
                    _war_guard()
                if col0 >= pp.vmax:
                    msk = _pass_masks(pp, cols, offs, columns)
                    for j2 in range_constexpr(len(msk)):
                        if msk[j2][0]:
                            _st_f32(logits_rsrc, msk[j2][1], neg_inf)

    @flyc.jit
    def launch(
        Q,
        KV,
        KS,
        PT,
        T2R,
        QP,
        CL,
        LOGITS,
        VIS,
        RST,
        grid_x,
        grid_y,
        rows,
        columns,
        num_pages,
        num_requests,
        width,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel._func.__name__ = kernel_name
        kernel(
            Q,
            KV,
            KS,
            PT,
            T2R,
            QP,
            CL,
            LOGITS,
            VIS,
            RST,
            rows,
            columns,
            num_pages,
            num_requests,
            width,
        ).launch(
            grid=(fx.Int64(grid_x), fx.Int64(grid_y), 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch.kernel_name = kernel_name
    launch.dtype_tag = f"{dtag}_decg"
    launch.cols_per_block = span
    launch.wave_cols = wave_cols
    launch.group_rows = group_rows
    launch.row_mode = row_mode
    launch.wg_waves = waves
    return launch


# ---------------------------------------------------------------------------
# Cached compile-and-dispatch (flyc.compile compiles *and* runs on first call)
# ---------------------------------------------------------------------------
def _run(exe, *args):
    cf = getattr(exe, "_cf", None)
    if cf is not None:
        cf(*args)
        return cf
    cf = flyc.compile(exe, *args)
    exe._cf = cf
    return cf


def wmma_rows_per_block(rows: int, num_requests: int) -> int:
    """Shape-only prefill selection; decode/draft (<=4 rows/request) stay r16.

    This is not a metadata uniformity assertion: the r32 kernel checks every
    request ID in each full block and falls back per M tile for mixes/tails.
    """
    return 32 if int(rows) >= 32 and int(rows) >= 32 * int(num_requests) else 16


def wmma_bf16_rows_per_block(rows: int, num_requests: int) -> int:
    """BM64 requires explicit opt-in; default/raw/cop keep r16/r32 routing."""
    if (
        _BF16_R64
        and _BF16_K_LOAD != "raw"
        and _ACCESS == "raw"
        and int(rows) >= 64
        and int(rows) >= 64 * int(num_requests)
    ):
        return 64
    return wmma_rows_per_block(rows, num_requests)


def wmma_grid(
    rows: int, columns: int, rows_per_block: int = 16, cols_per_block: int = WMMA_COLS
) -> tuple[int, int]:
    """64-column (32 for TDM r16) x compile-time 16/32/64-row workgroups."""
    if rows_per_block not in (16, 32, 64):
        raise ValueError("WMMA rows_per_block must be 16, 32 or 64")
    # Round PERSIST: r32 persistent blocks span 2/4/8 x 64 columns.
    persist_span = rows_per_block == 32 and cols_per_block in (128, 256, 512)
    if (cols_per_block not in (32, 64) and not persist_span) or (
        cols_per_block == 32 and rows_per_block != 16
    ):
        raise ValueError("WMMA cols_per_block must be 64, or 32 for r16")
    return (
        (int(columns) + cols_per_block - 1) // cols_per_block,
        (int(rows) + rows_per_block - 1) // rows_per_block,
    )


def wmma_launch_grid(rows: int, columns: int, launcher) -> tuple[int, int]:
    """ROUND-R2B: grid of any bf16 WMMA launcher (kstat spans R x 32 rows, C x 64 cols)."""
    kt = getattr(launcher, "kt", None)
    if kt is not None:  # ROUND-R2C ktile
        return kt_grid(rows, columns, kt[0], kt[1])
    ks = getattr(launcher, "kstat", None)
    if ks is not None:
        return kstat_grid(rows, columns, ks[0], ks[1])
    return wmma_grid(rows, columns, launcher.rows_per_launch, launcher.cols_per_block)


_DUMMY_SCALE: dict = {}


def _dummy_scale(dev):
    """4-byte stand-in for the scale plane on the bf16 path (never read)."""
    t = _DUMMY_SCALE.get(dev)
    if t is None:
        t = torch.zeros(4, dtype=torch.float32, device=dev)
        _DUMMY_SCALE[dev] = t
    return t


# ---------------------------------------------------------------------------
# Op wrapper with explicit contract checks
# ---------------------------------------------------------------------------
# The kernel addresses every buffer with an i32 byte offset (`buffer_*`
# voffsets, and FlyDSL passes tensor extents as i32).  These are the guard rails
# that keep that assumption true, so a legal-looking call can never wrap an
# offset and land where it does not belong.
_MAX_I32 = 2**31 - 1
# Read once, like the specialization switches: no second getenv on exact calls.
# Set before module import. This diagnostic intentionally allocates/synchronizes.
_QSA_PAD_INT32_GUARD = os.environ.get("QSA_PAD_INT32_GUARD", "0").strip() == "1"
# Native bucket units are import-time specialization facts, not tensor facts.
# Resolve once so PAD=1 adds only integer geometry/extent tests per launch.
_QSA_PAD_BF16_R64 = _BF16_R64 and _BF16_K_LOAD != "raw" and _ACCESS == "raw"
_QSA_PAD_BF16_R16_COLS = (
    R16_TDM_COLS
    if _resolve_bf16_k_load(16) == "tdm" and _ACCESS == "raw"
    else WMMA_COLS
)
_QSA_PAD_BF16_R32_COLS = (
    WMMA_COLS * TDM_PERSIST_CHUNKS
    if _resolve_bf16_k_load(32) == "tdm" and TDM_PERSIST and _ACCESS == "raw"
    else WMMA_COLS
)


def _check_plane(name, nbytes):
    """Non-empty and byte-addressable with i32 offsets."""
    if nbytes <= 0:
        raise ValueError(
            f"{name} is empty ({nbytes} B): a zero-length V# has no bounds check"
        )
    if nbytes > _MAX_I32:
        raise ValueError(
            f"{name} is {nbytes} B > {_MAX_I32} B: the kernel addresses its buffers "
            "with int32 byte offsets (split the call / chunk the rows)"
        )


def _check_align(tensor, name, align):
    ptr = int(tensor.data_ptr())
    if ptr % align:
        raise ValueError(
            f"{name} base address {ptr:#x} is not {align}-byte aligned; the kernel "
            f"issues {align}-byte vector buffer accesses from it"
        )


def _to_i32_checked(tensor, name):
    """Narrow to int32 only after proving the values fit (no silent wrap)."""
    if tensor.dtype == torch.int32:
        return tensor
    lo, hi = int(tensor.min()), int(tensor.max())
    if lo < -(2**31) or hi > _MAX_I32:
        raise ValueError(f"{name} values [{lo}, {hi}] do not fit in int32")
    return tensor.to(torch.int32)


# ROUND-NOSYNC: opt-in strict narrowing. Read once at import, like the other
# switches. QSA_STRICT_I32=1 restores the historical _to_i32_checked contract
# exactly (int64 values outside int32 raise; two host synchronizations plus a
# cast launch per int64 tensor). Default: _to_i32_saturated (no host sync).
_QSA_STRICT_I32 = os.environ.get("QSA_STRICT_I32", "0").strip() == "1"
_MIN_I32 = -(2**31)
_I32_BOUNDS: dict = {}


# ROUND-LAUNCH: the launch stream as FlyDSL's raw-pointer form.  FlyDSL's
# ``Stream`` C-ABI fill accepts an ``int`` (raw hipStream_t) directly, and
# ``torch._C._cuda_getCurrentRawStream(index)`` returns exactly
# ``torch.cuda.current_stream(index).cuda_stream`` without building a
# ``torch.cuda.Stream`` object (lazy-init check + device-index parsing + Python
# object construction).  Same stream, same compiled specialization (FlyDSL's
# Stream cache signature is the type only).  If ``torch.cuda.current_stream``
# has been replaced (CPU gates, tracing tools) the replacement is honoured.
_TORCH_CURRENT_STREAM = torch.cuda.current_stream
_RAW_CURRENT_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _launch_stream(dev):
    """Current stream of ``dev``: raw pointer (fast) or the patched lookup."""
    current = torch.cuda.current_stream
    if (
        current is _TORCH_CURRENT_STREAM
        and _RAW_CURRENT_STREAM is not None
        and dev.type == "cuda"
        and dev.index is not None
    ):
        return _RAW_CURRENT_STREAM(dev.index)
    return current(dev)


def _launch_stream_ptr(dev):
    """Raw pointer of the current stream of a CUDA ``dev`` (cache keys)."""
    current = torch.cuda.current_stream
    if (
        current is _TORCH_CURRENT_STREAM
        and _RAW_CURRENT_STREAM is not None
        and dev.index is not None
    ):
        return _RAW_CURRENT_STREAM(dev.index)
    return current(dev).cuda_stream


def _i32_bounds(device, hi):
    """0-d int64 clamp bounds, created once per (device, stream, hi).

    Created with ``torch.full`` (a fill launch, graph-capture safe, no H2D copy)
    on the stream that first consumes them, so every reader is stream-ordered
    after its own creation fill; steady-state calls allocate/launch nothing here.
    """
    stream = _launch_stream_ptr(device) if device.type == "cuda" else None
    key = (device, stream, hi)
    bounds = _I32_BOUNDS.get(key)
    if bounds is None:
        bounds = (
            torch.full((), _MIN_I32, dtype=torch.int64, device=device),
            torch.full((), hi, dtype=torch.int64, device=device),
        )
        _I32_BOUNDS[key] = bounds
    return bounds


def _to_i32_saturated(tensor, hi=_MAX_I32):
    """int32: zero-copy passthrough. int64: ONE fused clamp+narrow, no host sync.

    ``torch.clamp(int64, lo_t, hi_t, out=int32)`` is a single TensorIterator
    launch on the device (the int64 -> int32 store cast is done in-kernel); on
    CPU it is the same op (with an internal temporary). No ``.min()``/``.max()``
    /``.item()`` host round-trip. Values are saturated to ``[INT32_MIN, hi]``.

    Why saturation is exact (proof, see also ``test_nosync_i32_cpu.py``):
    every metadata semantic the kernel computes is monotone in the value and
    constant outside the int32 range, given the plane caps enforced by
    ``_check_plane`` (rows/columns/requests/pages are all < 2**29):

    * ``token_to_request``: valid iff ``0 <= r < requests``. Saturation fixes
      every in-range value; ``r > INT32_MAX`` -> ``INT32_MAX >= requests`` and
      ``r < INT32_MIN`` -> ``INT32_MIN < 0``: invalid stays invalid.
    * ``page_table``: valid iff ``0 <= p < cache_pages``; identical argument
      (invalid pages stay invalid and are clamped to page 0 by the kernel).
    * ``context_lens`` ``c`` and ``query_positions`` ``p`` (``hi = INT32_MAX-1``
      for positions so the kernel's i32 ``pos + 1`` can never wrap):
      ``V = max(0, min((p+1)//4, c//4))``. Let ``p', c'`` be the saturated
      values and ``V'`` the same formula. If ``p <= INT32_MAX-2`` and
      ``INT32_MIN <= c <= INT32_MAX`` (and ``p >= INT32_MIN``) nothing changes.
      ``p >= INT32_MAX-1``: ``(p'+1)//4 = 2**29 > c'//4``, so
      ``V' = max(0, c'//4)``; ``V = max(0, c//4)`` too because
      ``(p+1)//4 >= 2**29``, unless ``c > INT32_MAX``. ``c > INT32_MAX``:
      ``c'//4 = 2**29-1 >= (p'+1)//4`` for ``p' <= INT32_MAX-2``, so
      ``V' = max(0, (p+1)//4) = V``; only when both ``p+1 >= 2**31`` and
      ``c > INT32_MAX`` is ``V' = 2**29-1 < V``. Negative ``p`` or ``c``
      (saturated or not) give 0 both ways. Hence ``V' = min(V, 2**29-1)``.
      Since ``columns*4 <= INT32_MAX`` (logits plane cap) every column is
      ``< 2**29-1``, so the masks ``c < V`` and ``c < V'`` are identical:
      every logit is golden-exact. ``visible_groups`` is golden-exact
      whenever ``V < 2**29``, i.e. for every input the strict contract
      accepted (and it fixes the historical int64 ``pos = INT32_MAX`` wrap);
      for ``V >= 2**29`` (only reachable with out-of-int32 metadata) the
      returned count saturates at ``2**29-1``, still >= every column.
    """
    if tensor.dtype == torch.int32:
        return tensor
    lo_t, hi_t = _i32_bounds(tensor.device, hi)
    out = torch.empty(tensor.shape, dtype=torch.int32, device=tensor.device)
    return torch.clamp(tensor, lo_t, hi_t, out=out)


def qsa_pad_plan(shape, dtype):
    """CPU-only bucket map; no tensor allocation, compiler or device access.

    ``shape`` is a mapping, or ``(rows, heads, columns, requests,
    page_table_width, cache_pages[, regime])``. Dtype is ``bf16``/``fp8``
    (torch dtype objects also work). The specialization is selected from the
    TRUE shape before rounding: request padding must not downgrade prefill.
    Environment overrides are the same ones used by the unpadded wrapper.
    Zero-size planes are rejected, just like the default wrapper; no zero V#
    is ever launched. This plan does not assert metadata uniformity or cache
    alignment: the dispatcher retains the kernel's mixed/tail/raw fallback.
    """
    keys = ("rows", "heads", "columns", "requests", "page_table_width", "cache_pages")
    if isinstance(shape, dict):
        dims = {key: int(shape[key]) for key in keys}
        regime = shape.get("regime")
    else:
        if len(shape) not in (6, 7):
            raise ValueError("shape must contain six dimensions and optional regime")
        dims = dict(zip(keys, (int(value) for value in shape[:6])))
        regime = shape[6] if len(shape) == 7 else None
    tag = str(dtype).replace("torch.", "")
    tag = {"bfloat16": "bf16", "float8_e4m3fn": "fp8"}.get(tag, tag)
    if tag not in ("bf16", "fp8"):
        raise ValueError(f"padding supports bf16/fp8, got {dtype!r}")
    if dims["heads"] != NUM_HEADS:
        raise ValueError(f"padding requires {NUM_HEADS} heads")
    if any(value < 1 for value in dims.values()):
        raise ValueError(
            "padding rejects zero/negative dimensions (no empty V# launch)"
        )
    rows, columns, requests = dims["rows"], dims["columns"], dims["requests"]
    if rows > MAX_GRID_Y:
        raise ValueError(f"rows must be <= {MAX_GRID_Y}")
    if columns > PAGE_SIZE * dims["page_table_width"]:
        raise ValueError("columns exceed the true page-table capacity; never truncate")
    inferred = (
        "prefill"
        if rows >= 32 * requests
        else ("decode" if rows >= 4 * requests else "draft")
    )
    regime = inferred if regime is None else str(regime)
    if regime not in ("decode", "draft", "prefill"):
        raise ValueError(f"unknown regime {regime!r}")
    # Regime is descriptive only: geometry follows the champion shape router.
    geometry = (
        wmma_bf16_rows_per_block(rows, requests)
        if tag == "bf16"
        else wmma_rows_per_block(rows, requests)
    )
    mode = (
        _resolve_bf16_k_load(geometry)
        if tag == "bf16"
        else _resolve_fp8_k_load(geometry)
    )
    if _ACCESS != "raw":
        mode = "raw"  # copy-atom access never instantiates a TDM specialization
    if (
        tag == "bf16"
        and geometry == 32
        and mode == "tdm"
        and TDM_PERSIST
        and _ACCESS == "raw"
        and columns >= WMMA_COLS * TDM_PERSIST_CHUNKS
    ):
        mode = "persist"
    tile_cols = (
        WMMA_COLS * TDM_PERSIST_CHUNKS
        if mode == "persist"
        else (
            R16_TDM_COLS
            if tag == "bf16" and geometry == 16 and mode == "tdm"
            else WMMA_COLS
        )
    )
    padded = dict(dims)
    padded["rows"] = ((rows + geometry - 1) // geometry) * geometry
    padded["columns"] = ((columns + tile_cols - 1) // tile_cols) * tile_cols
    padded["requests"] = ((requests + 3) // 4) * 4
    width_unit = tile_cols // PAGE_SIZE
    padded["page_table_width"] = (
        (dims["page_table_width"] + width_unit - 1) // width_unit
    ) * width_unit
    # Python integers do not overflow; enforce exactly the same i32-address
    # budget before allocating any padded plane. _plane_bytes remains i64.
    for name, nbytes in (
        ("padded_q", padded["rows"] * NUM_HEADS * HEAD_DIM * 2),
        ("padded_logits", padded["rows"] * padded["columns"] * 4),
        ("padded_page_table", padded["requests"] * padded["page_table_width"] * 4),
        ("padded_row_metadata", padded["rows"] * 4),
        ("padded_context_lens", padded["requests"] * 4),
        (
            "compressed_k_cache",
            dims["cache_pages"] * PAGE_SIZE * HEAD_DIM * (2 if tag == "bf16" else 1),
        ),
        ("compressed_k_scale", dims["cache_pages"] * PAGE_SIZE * 4),
    ):
        _check_plane(name, nbytes)
    return {
        "dtype": tag,
        "regime": regime,
        "true_shape": dims,
        "padded_shape": padded,
        "rows_per_block": geometry,
        "tile_cols": tile_cols,
        "k_load": mode,
        "needs_padding": padded != dims,
        "copy_back": (rows, columns),
        "env": "QSA_PAD_BUCKET=1",
    }


def _qsa_pad_validate_buffers(inputs, outputs):
    """Opt-in guard: output writes must not race input or other output reads.

    Bounds cover strided metadata as well as contiguous planes. Disjoint views
    of the same storage are permitted; overlapping views are rejected before
    launch. The historical default-off acceptance contract is unchanged.
    """

    def interval(tensor):
        lo = int(tensor.data_ptr())
        extent = 1 + sum(
            (int(n) - 1) * int(s) for n, s in zip(tensor.shape, tensor.stride())
        )
        return lo, lo + extent * tensor.element_size()

    named_inputs = [(name, tensor, interval(tensor)) for name, tensor in inputs]
    named_outputs = [(name, tensor, interval(tensor)) for name, tensor in outputs]
    for index, (name, tensor, (lo, hi)) in enumerate(named_outputs):
        if not tensor.is_contiguous():
            raise ValueError(f"padding requires contiguous {name}")
        for other, value, (olo, ohi) in named_inputs + named_outputs[:index]:
            if tensor.device == value.device and lo < ohi and olo < hi:
                raise ValueError(f"padding output {name} overlaps {other}")


def _qsa_pad_dispatch(
    plan, q, cache, scale, pt, t2r, qp, cl, logits, visible, starts, divisor, stream
):
    """Launch an existing bucket specialization with allocation-safe padding.

    Only called after the unchanged wrapper has validated TRUE inputs. Cache
    pages/scales are never padded or changed. TDM alignment fallback is identical
    to the exact path (fp8 8B base allowed, 16/128B TDM constraint => raw).
    Every padded allocation is checked again before constructing a descriptor.
    """
    true, pad = plan["true_shape"], plan["padded_shape"]
    rows, columns, requests = true["rows"], true["columns"], true["requests"]
    pr, pc, pn, pw = (
        pad[key] for key in ("rows", "columns", "requests", "page_table_width")
    )
    dev = q.device
    # pos+1 is i32 in the frozen kernel. Since ctx <= INT32_MAX, saturating
    # this one endpoint preserves min((pos+1)//4, ctx//4) exactly and prevents
    # overflow. All other positions (including negative values) are unchanged.
    qp = qp.clamp(max=_MAX_I32 - 1)
    if pr != rows:
        pq = torch.zeros((pr, NUM_HEADS, HEAD_DIM), dtype=q.dtype, device=dev)
        pq[:rows].copy_(q)
        pqp = torch.zeros(pr, dtype=torch.int32, device=dev)
        pqp[:rows].copy_(qp)
    else:
        pq, pqp = q, qp
    # A previously invalid request must remain invalid after pn >= requests.
    safe_t2r = torch.where((t2r >= 0) & (t2r < requests), t2r, -1)
    pt2r = torch.full((pr,), -1, dtype=torch.int32, device=dev)
    pt2r[:rows].copy_(safe_t2r)
    if pn != requests or pw != true["page_table_width"]:
        ppt = torch.full((pn, pw), -1, dtype=torch.int32, device=dev)
        ppt[:requests, : true["page_table_width"]].copy_(pt)
        pcl = torch.zeros(pn, dtype=torch.int32, device=dev)
        pcl[:requests].copy_(cl)
    else:
        ppt, pcl = pt, cl
    output_padding = pr != rows or pc != columns
    plogits = (
        torch.full((pr, pc), _NEG_INF, dtype=torch.float32, device=dev)
        if output_padding
        else logits
    )
    pvis = (
        torch.empty(pr, dtype=torch.int32, device=dev)
        if pr != rows
        else visible.view(-1)
    )
    prst = (
        torch.empty(pr, dtype=torch.int32, device=dev)
        if pr != rows
        else starts.view(-1)
    )
    for name, tensor, alignment in (
        ("padded_q", pq, 16),
        ("padded_page_table", ppt, 4),
        ("padded_token_to_request", pt2r, 4),
        ("padded_query_positions", pqp, 4),
        ("padded_context_lens", pcl, 4),
        ("padded_logits", plogits, 4),
        ("padded_visible_groups", pvis, 4),
        ("padded_row_starts", prst, 4),
    ):
        _check_plane(name, tensor.numel() * tensor.element_size())
        _check_align(tensor, name, alignment)
    geometry, mode = plan["rows_per_block"], plan["k_load"]
    kv_u8 = cache.view(torch.uint8).view(true["cache_pages"], -1)
    if mode in ("tdm", "persist") and kv_u8.data_ptr() % 128:
        geometry, mode = 16, "raw"
    if plan["dtype"] == "bf16":
        launch = _build_wmma_kernel(float(divisor), geometry, mode)
        gx, gy = wmma_grid(pr, pc, geometry, launch.cols_per_block)
        _run(
            launch,
            pq.view(torch.uint8).view(pr, -1),
            kv_u8,
            ppt,
            pt2r,
            pqp,
            pcl,
            plogits,
            pvis,
            prst,
            gx,
            gy,
            pr,
            pc,
            true["cache_pages"],
            pn,
            pw,
            stream,
        )
    else:
        launch = _build_wmma_fp8_kernel(float(divisor), geometry, mode)
        gx, gy = wmma_grid(pr, pc, geometry)
        _run(
            launch,
            pq.view(torch.uint8).view(pr, -1),
            kv_u8,
            scale.view(true["cache_pages"], PAGE_SIZE, 1),
            ppt,
            pt2r,
            pqp,
            pcl,
            plogits,
            pvis,
            prst,
            gx,
            gy,
            pr,
            pc,
            true["cache_pages"],
            pn,
            pw,
            stream,
        )
    # Padded columns can share the last valid page with true columns. Mask them
    # after launch without changing true visible_groups or arithmetic order.
    if pc != columns:
        plogits[:, columns:].fill_(_NEG_INF)
    if output_padding:
        logits.copy_(plogits[:rows, :columns])
    if pr != rows:
        visible.view(-1).copy_(pvis[:rows])
        starts.view(-1).copy_(prst[:rows])
    return logits, visible.view(-1)


def _qsa_pad_checked_dispatch(
    rows,
    columns,
    requests,
    width,
    pages,
    is_fp8,
    q,
    cache,
    scale,
    pt,
    t2r,
    qp,
    cl,
    logits,
    visible,
    starts,
    divisor,
    dev,
    raw_pt,
    raw_t2r,
    raw_qp,
    raw_cl,
):
    """Slow opt-in path only; preserve the ROUND-PAD validation and plan."""
    for name, tensor in (
        ("compressed_k_cache", cache),
        ("page_table", raw_pt),
        ("token_to_request", raw_t2r),
        ("query_positions", raw_qp),
        ("context_lens", raw_cl),
    ):
        if tensor.device != dev:
            raise ValueError(f"{name} must be on {dev}, got {tensor.device}")
    if is_fp8 and scale.device != dev:
        raise ValueError(f"compressed_k_scale must be on {dev}")
    checked_inputs = [
        ("q", q),
        ("compressed_k_cache", cache),
        ("page_table", raw_pt),
        ("token_to_request", raw_t2r),
        ("query_positions", raw_qp),
        ("context_lens", raw_cl),
    ]
    if is_fp8:
        checked_inputs.append(("compressed_k_scale", scale))
    _qsa_pad_validate_buffers(
        checked_inputs,
        [("logits", logits), ("visible_groups", visible), ("row_starts", starts)],
    )
    if (not is_fp8 and _BF16_PATH == "wmma") or (is_fp8 and _FP8_PATH == "wmma"):
        plan = qsa_pad_plan(
            (rows, NUM_HEADS, columns, requests, width, pages),
            "fp8" if is_fp8 else "bf16",
        )
        if plan["needs_padding"]:
            return _qsa_pad_dispatch(
                plan,
                q,
                cache,
                scale,
                pt,
                t2r,
                qp,
                cl,
                logits,
                visible,
                starts,
                divisor,
                torch.cuda.current_stream(dev),
            )
    return None  # Scalar, or a candidate-exact shape after alignment fallback.


_POS64_KW = {"pos64": True}
_NO_KW: dict = {}
_PAGE_KW = {8: {"page_size": 8}, 16: {"page_size": 16}}  # builder keyword (P != 4 only)


def _qp_pad_i32(query_positions, qp_arg, pos64):
    """QSA_PAD_BUCKET (opt-in) planner input: int32 positions [rows].

    The padding kernels copy positions into int32 planes, so a pos64 call
    narrows here exactly as ROUND-NOSYNC did (one clamp launch, no sync); only
    reached on the opt-in padding route. int32 callers: the argument as-is.
    """
    return _to_i32_saturated(query_positions, _MAX_I32 - 1) if pos64 else qp_arg


def qsa_paged_mqa_logits_page4(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    compress_ratio: int = COMPRESS_RATIO,
    divisor: float | None = None,
    max_columns: int | None = None,
    compressed_k_scale: torch.Tensor | None = None,
    logits: torch.Tensor | None = None,
    visible_groups: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paged MQA logits for the QSA index scorer, index page P in {4, 8, 16} (gfx1250).

    Returns ``(logits [rows, columns] fp32, visible_groups [rows] int32)``.
    ``-inf`` is written for every column with ``c >= visible[t]`` and for every
    column whose page-table entry is invalid. ROUND-PAGE16: P is read from the
    cache shape ``[pages, P, 1, 128]`` (P = KV block / compress ratio 4: block
    16 -> 4, block 32 -> 8, block 64 -> 16); any other P raises. Column c reads
    ``page_table[req, c // P]`` row ``c % P``; visibility always divides by 4.
    """
    if compress_ratio != COMPRESS_RATIO:
        raise ValueError(
            f"compress_ratio must be {COMPRESS_RATIO}, got {compress_ratio}"
        )
    if q.dtype != torch.bfloat16:
        raise ValueError(f"q must be bfloat16, got {q.dtype}")
    if q.ndim != 3 or tuple(q.shape[1:]) != (NUM_HEADS, HEAD_DIM):
        raise ValueError(
            f"q must be [rows, {NUM_HEADS}, {HEAD_DIM}], got {tuple(q.shape)}"
        )
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    rows = int(q.shape[0])
    if rows == 0 or rows > MAX_GRID_Y:
        raise ValueError(f"rows must be in [1, {MAX_GRID_Y}], got {rows}")

    if (
        compressed_k_cache.ndim != 4
        or int(compressed_k_cache.shape[1]) not in SUPPORTED_PAGE_SIZES
        or tuple(compressed_k_cache.shape[2:]) != (1, HEAD_DIM)
    ):
        raise ValueError(
            f"compressed_k_cache must be [pages, P, 1, {HEAD_DIM}] with P in "
            f"{SUPPORTED_PAGE_SIZES}, got {tuple(compressed_k_cache.shape)}"
        )
    page = int(compressed_k_cache.shape[1])
    # P = 4 passes no extra builder keyword: the very same lru_cache entries
    # and byte-identical specializations as bf492262.
    pg_kw = _NO_KW if page == PAGE_SIZE else _PAGE_KW[page]
    if not compressed_k_cache.is_contiguous():
        raise ValueError("compressed_k_cache must be contiguous")
    num_pages = int(compressed_k_cache.shape[0])
    if num_pages < 1:
        raise ValueError("compressed_k_cache must have at least one page")
    is_fp8 = compressed_k_cache.dtype == torch.float8_e4m3fn
    if not is_fp8 and compressed_k_cache.dtype != torch.bfloat16:
        raise ValueError(
            "compressed_k_cache must be bfloat16 or float8_e4m3fn, got "
            f"{compressed_k_cache.dtype}"
        )
    if is_fp8:
        if compressed_k_scale is None:
            raise ValueError(
                f"fp8 index cache requires its fp32 [pages, {page}, 1] scale"
            )
        if compressed_k_scale.dtype != torch.float32:
            raise ValueError(
                f"compressed_k_scale must be float32, got {compressed_k_scale.dtype}"
            )
        if tuple(compressed_k_scale.shape) != (num_pages, page, 1):
            raise ValueError(
                f"compressed_k_scale must be {(num_pages, page, 1)}, got "
                f"{tuple(compressed_k_scale.shape)}"
            )
        if not compressed_k_scale.is_contiguous():
            raise ValueError("compressed_k_scale must be contiguous")
    elif compressed_k_scale is not None:
        raise ValueError("bf16 index cache takes no scale (got compressed_k_scale)")

    if page_table.ndim != 2 or page_table.shape[0] == 0:
        raise ValueError(
            f"page_table must be [requests, width], got {tuple(page_table.shape)}"
        )
    if page_table.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"page_table must be int32/int64, got {page_table.dtype}")
    if not page_table.is_contiguous():
        raise ValueError("page_table must be contiguous")
    num_requests = int(page_table.shape[0])
    width = int(page_table.shape[1])
    if width < 1:
        # width == 0 makes the page-table V# zero-length: on this ISA a
        # num_records of 0 disables the bounds check entirely.
        raise ValueError("page_table must have at least one column (width >= 1)")

    columns_full = width * page
    if max_columns is None:
        columns = columns_full
    else:
        if int(max_columns) < 1:
            raise ValueError(f"max_columns must be >= 1, got {max_columns}")
        columns = min(columns_full, int(max_columns))
    # `columns % PAGE_SIZE` needs no special case: the kernel derives
    # `logical = col // 4` / `off = col % 4` per column and predicates every
    # store, so a partial trailing page is written column by column and nothing
    # outside [0, rows * columns) is touched.  (Covered by the CPU contract
    # tests: columns = 1, 3, 5, 255, 257, 1000.)

    for name, t, n in (
        ("token_to_request", token_to_request, rows),
        ("query_positions", query_positions, rows),
        ("context_lens", context_lens, num_requests),
    ):
        if t.ndim != 1 or int(t.shape[0]) != n:
            raise ValueError(f"{name} must be [{n}], got {tuple(t.shape)}")
        if t.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be int32/int64, got {t.dtype}")

    dev = q.device
    if logits is None:
        logits = torch.empty((rows, columns), dtype=torch.float32, device=dev)
    elif (
        logits.shape != (rows, columns)
        or logits.dtype != torch.float32
        or not logits.is_contiguous()
    ):
        raise ValueError(f"logits must be contiguous fp32 {(rows, columns)}")
    # ROUND-LAUNCH: wrapper-allocated outputs are already 1-D contiguous, so
    # their ``.view(-1)`` is the identity; caller tensors keep the view.
    if visible_groups is None:
        visible_groups = vg_flat = torch.empty(rows, dtype=torch.int32, device=dev)
    elif visible_groups.numel() != rows or visible_groups.dtype != torch.int32:
        raise ValueError(f"visible_groups must be int32 [{rows}]")
    else:
        vg_flat = None
    if row_starts is None:
        row_starts = rs_flat = torch.empty(rows, dtype=torch.int32, device=dev)
    elif row_starts.numel() != rows or row_starts.dtype != torch.int32:
        raise ValueError(f"row_starts must be int32 [{rows}]")
    else:
        rs_flat = None
    for name, t in (
        ("logits", logits),
        ("visible_groups", visible_groups),
        ("row_starts", row_starts),
    ):
        if t.device != dev:
            raise ValueError(f"{name} must be on {dev}, got {t.device}")

    # --- GPU memory-safety guard rails (see the module docstring) ------------
    # 1. every plane is non-empty (a zero-length V# is unbounded) and fits the
    #    kernel's i32 byte addressing;
    # 2. every base address is aligned for the widest access the kernel issues
    #    from it (16 B dwordx4 K/Q loads, 4 B scalar metadata/scale/logits).
    kv_elem_bytes = 1 if is_fp8 else 2
    page_bytes = page * HEAD_DIM * kv_elem_bytes
    _check_plane("q", rows * NUM_HEADS * HEAD_DIM * 2)
    _check_plane("compressed_k_cache", num_pages * page_bytes)
    if is_fp8:
        _check_plane("compressed_k_scale", num_pages * page * 4)
    _check_plane("page_table", num_requests * width * 4)
    _check_plane("token_to_request", rows * 4)
    _check_plane("query_positions", rows * 4)
    _check_plane("context_lens", num_requests * 4)
    _check_plane("logits", rows * columns * 4)
    _check_plane("visible_groups", rows * 4)
    _check_plane("row_starts", rows * 4)
    _check_align(q, "q", 16)
    _check_align(compressed_k_cache, "compressed_k_cache", 16 if not is_fp8 else 8)
    _check_align(page_table, "page_table", 4)
    _check_align(logits, "logits", 4)
    _check_align(visible_groups, "visible_groups", 4)
    _check_align(row_starts, "row_starts", 4)
    _check_align(token_to_request, "token_to_request", 4)
    _check_align(query_positions, "query_positions", 4)
    _check_align(context_lens, "context_lens", 4)
    if is_fp8:
        _check_align(compressed_k_scale, "compressed_k_scale", 4)

    # Explicit narrowings (never a wrapping cast): the kernel reads int32
    # indices and FlyDSL passes tensor extents as int32.  ROUND-NOSYNC: int32
    # inputs are zero-copy; each int64 input costs ONE device clamp+narrow
    # launch and NO host synchronization (saturation is semantics-preserving,
    # see _to_i32_saturated; positions saturate at INT32_MAX-1 so `pos + 1`
    # cannot wrap).  QSA_STRICT_I32=1 restores the historical raising probe.
    # ROUND-NATIVE64: int64 query_positions (golden fixtures, ATOM) cost NO
    # narrowing launch: the kernel is built with pos64=True and reads the
    # zero-copy int32 view [2*rows] (lo, hi dwords), computing the very same
    # clamp(v, INT32_MIN, INT32_MAX-1) in registers (see _pos_loaders).
    pos64 = False
    if _QSA_STRICT_I32:
        page_table_i32 = _to_i32_checked(page_table, "page_table")
        t2r_i32 = _to_i32_checked(token_to_request, "token_to_request")
        qp_i32 = _to_i32_checked(query_positions, "query_positions")
        cl_i32 = _to_i32_checked(context_lens, "context_lens")
    else:
        page_table_i32 = _to_i32_saturated(page_table)
        t2r_i32 = _to_i32_saturated(token_to_request)
        pos64 = query_positions.dtype == torch.int64
        qp_i32 = query_positions  # int32: zero copy; int64: viewed below
        cl_i32 = _to_i32_saturated(context_lens)
    # ROUND-LAUNCH: all three are validated 1-D above, so ``.view(-1)`` is an
    # identity view when the stride is already 1 (same ptr/shape/stride reach
    # FlyDSL); any other stride takes the historical path verbatim.
    if t2r_i32.stride(0) != 1:
        t2r_i32 = t2r_i32.contiguous().view(-1)
    if qp_i32.stride(0) != 1:
        qp_i32 = qp_i32.contiguous().view(-1)
    # int32 callers pass no keyword: the very same lru_cache entry/specialization
    # as before; int64 positions select the pos64 specialization.
    qp_kw = _POS64_KW if pos64 else _NO_KW
    if pos64:
        qp_i32 = qp_i32.view(torch.int32)  # [2*rows] zero-copy (lo, hi) dwords
    if cl_i32.stride(0) != 1:
        cl_i32 = cl_i32.contiguous().view(-1)

    if divisor is None:
        divisor = math.sqrt(HEAD_DIM)
    divisor = float(divisor)
    if not math.isfinite(divisor) or divisor <= 0.0:
        raise ValueError(f"divisor must be finite and > 0, got {divisor}")

    # Native bucket routing is deliberately outside the original launch
    # branches, which remain verbatim (including the stock CPU AST gates).
    # ROUND-PAGE16: the opt-in bucket planner is P = 4 only (P = 16 runs exact).
    if page == PAGE_SIZE and os.environ.get("QSA_PAD_BUCKET", "0").strip() == "1":
        if (not is_fp8 and _BF16_PATH == "wmma") or (is_fp8 and _FP8_PATH == "wmma"):
            # Import-time units + true-shape integers only: no planner, tensor
            # work, pointer/device/alias inspection, or extra helper calls.
            pad_rows = (
                64
                if not is_fp8
                and _QSA_PAD_BF16_R64
                and rows >= 64
                and rows >= 64 * num_requests
                else 32 if rows >= 32 and rows >= 32 * num_requests else 16
            )
            pad_cols = (
                WMMA_COLS
                if is_fp8 or pad_rows == 64
                else (
                    _QSA_PAD_BF16_R16_COLS
                    if pad_rows == 16
                    else (
                        _QSA_PAD_BF16_R32_COLS
                        if columns >= _QSA_PAD_BF16_R32_COLS
                        else WMMA_COLS
                    )
                )
            )
            # True request count need not be a multiple of4: native req1/2
            # buckets use the original kernel's true request/metadata guards.
            if rows % pad_rows or columns % pad_cols or width % (pad_cols // PAGE_SIZE):
                padded = _qsa_pad_checked_dispatch(
                    rows,
                    columns,
                    num_requests,
                    width,
                    num_pages,
                    is_fp8,
                    q,
                    compressed_k_cache,
                    compressed_k_scale,
                    page_table_i32,
                    t2r_i32,
                    _qp_pad_i32(query_positions, qp_i32, pos64),
                    cl_i32,
                    logits,
                    visible_groups,
                    row_starts,
                    divisor,
                    dev,
                    page_table,
                    token_to_request,
                    query_positions,
                    context_lens,
                )
                if padded is not None:
                    return padded
        else:  # Scalar opt-in keeps the previous validation/true-shape bypass.
            _qsa_pad_checked_dispatch(
                rows,
                columns,
                num_requests,
                width,
                num_pages,
                is_fp8,
                q,
                compressed_k_cache,
                compressed_k_scale,
                page_table_i32,
                t2r_i32,
                _qp_pad_i32(query_positions, qp_i32, pos64),
                cl_i32,
                logits,
                visible_groups,
                row_starts,
                divisor,
                dev,
                page_table,
                token_to_request,
                query_positions,
                context_lens,
            )
        # pos64 positions are saturated at INT32_MAX-1: never INT32_MAX.
        if _QSA_PAD_INT32_GUARD and not pos64 and bool((qp_i32 == _MAX_I32).any()):
            raise ValueError(
                "QSA_PAD_BUCKET identity/scalar path cannot use query_positions=INT32_MAX"
            )

    kv_u8 = compressed_k_cache.view(torch.uint8).view(num_pages, -1)
    q_u8 = q.view(torch.uint8).view(rows, -1)
    stream = _launch_stream(dev)
    if vg_flat is None:
        vg_flat = visible_groups.view(-1)
    if rs_flat is None:
        rs_flat = row_starts.view(-1)
    # Round DRAFT: one-row-per-wave kernel for rows <= requests (default on since
    # ROUND-MERGE1; int64 positions use the pos64 draft kernel, no clamp launch).
    # Kept outside the original branches (verbatim below, incl. their AST gates);
    # QSA_DRAFT_PATH=0 (opt-out) never evaluates more than this flag.
    if _DRAFT_PATH:
        draft = draft_route(rows, num_requests, is_fp8, kv_u8.data_ptr())
        if draft is not None:
            # ROUND-R2 QSA_DRAFT_WG_WAVES (4: no keyword, same lru entry; R2E 'auto' = per dtype).
            dr_w = draft_wg_waves(is_fp8)
            dr_kw = _NO_KW if dr_w == DRAFT_WAVES else {"wg_waves": dr_w}
            draft_launch = _build_draft_kernel(
                is_fp8, float(divisor), draft[0], draft[1], **qp_kw, **pg_kw, **dr_kw
            )
            grid_x, grid_y = draft_grid(rows, columns, draft[1], dr_w)
            _run(
                draft_launch,
                q_u8,
                kv_u8,
                (
                    compressed_k_scale.view(num_pages, page, 1)
                    if is_fp8
                    else _dummy_scale(dev)
                ),
                page_table_i32,
                t2r_i32,
                qp_i32,
                cl_i32,
                logits,
                vg_flat,
                rs_flat,
                grid_x,
                grid_y,
                rows,
                columns,
                num_pages,
                num_requests,
                width,
                stream,
            )
            return logits, vg_flat
    # Round DECG (QSA_DECODE_GROUP, default 'all' since ROUND-DECG2; '0' opts
    # out): decode row-group kernel for rows > requests, rows % 4 == 0 (r16
    # regime). With '0' only this flag is read (previous c45b24e0 routing).
    if _DECODE_GROUP != "0":
        # ROUND-R2: knobs default to (4, 'base', 4) -> no keyword, same lru entry.
        dg_rows, dg_mode, dg_waves = decode_group_knobs(rows, num_requests)
        dg_kw = (
            _NO_KW
            if (dg_rows, dg_mode, dg_waves) == (DECG_ROWS, "base", DECG_WAVES)
            else {"group_rows": dg_rows, "row_mode": dg_mode, "wg_waves": dg_waves}
        )
        decg = decode_group_route(rows, num_requests, is_fp8, kv_u8.data_ptr(), dg_rows)
        if decg is not None:
            decg_launch = _build_decode_group_kernel(
                is_fp8, float(divisor), decg, **qp_kw, **pg_kw, **dg_kw
            )
            grid_x, grid_y = decode_group_grid(rows, columns, decg, dg_rows, dg_waves)
            _run(
                decg_launch,
                q_u8,
                kv_u8,
                (
                    compressed_k_scale.view(num_pages, page, 1)
                    if is_fp8
                    else _dummy_scale(dev)
                ),
                page_table_i32,
                t2r_i32,
                qp_i32,
                cl_i32,
                logits,
                vg_flat,
                rs_flat,
                grid_x,
                grid_y,
                rows,
                columns,
                num_pages,
                num_requests,
                width,
                stream,
            )
            return logits, vg_flat
    if not is_fp8 and _BF16_PATH == "wmma":
        # bf16: WMMA kernel, one workgroup = 16 rows x 64 columns (see the
        # matrix-core block above).  It writes every logit itself (-inf outside
        # `valid_page & c < visible`), so no pre-fill / memset is needed.
        rows_per_block = wmma_bf16_rows_per_block(rows, num_requests)
        # The API permits 16-byte-aligned cache subviews; TDM's full-row
        # direct-copy path needs 128B alignment. Select the actual raw builder
        # specialization, not merely r16 geometry (r16 now also has TDM).
        k_load = _resolve_bf16_k_load(rows_per_block)
        if k_load == "tdm" and kv_u8.data_ptr() % 128:
            rows_per_block = 16
            k_load = "raw"
        # Round PERSIST (opt-in QSA_BF16_PERSIST=1): r32 TDM prefill with at
        # least one full persistent span loops TDM_PERSIST_CHUNKS tiles/block.
        # Mixed/partial row blocks keep the kernel's exact raw fallback.
        # ROUND-R2B: the choice is the data-driven prefill route table (QSA_PREFILL_ROUTE;
        # default 'legacy' == the condition above: persist iff columns >= 64 * chunks).
        pf_kw = _NO_KW
        if rows_per_block == 32 and k_load == "tdm" and _ACCESS == "raw":
            k_load, pf_kw = prefill_bf16_route(rows, num_requests, columns, page)
            if k_load == "ktile":
                pf_kw = dict(
                    {"kt_rows": pf_kw["kt_rows"], "col_tiles": pf_kw["col_tiles"]},
                    **({"lean": True} if pf_kw.get("lean") else {}),  # ROUND-R2D
                    **({"ipf": pf_kw["ipf"]} if pf_kw.get("ipf") else {}),
                )
            elif k_load == "kstat":
                pf_kw = dict(
                    {"row_tiles": pf_kw["row_tiles"], "col_tiles": pf_kw["col_tiles"]},
                    **({"early": True} if pf_kw.get("early") else {}),
                )
            else:
                pf_kw = _NO_KW
        # grid.x is bound to the actual specialization's column width (only a
        # forced-TDM r16 kernel may use 32 columns; every other one uses 64).
        launch = _build_wmma_kernel(
            float(divisor), rows_per_block, k_load, **qp_kw, **pg_kw, **pf_kw
        )
        grid_x, grid_y = wmma_launch_grid(rows, columns, launch)
        _run(
            launch,
            q_u8,
            kv_u8,
            page_table_i32,
            t2r_i32,
            qp_i32,
            cl_i32,
            logits,
            vg_flat,
            rs_flat,
            grid_x,
            grid_y,
            rows,
            columns,
            num_pages,
            num_requests,
            width,
            stream,
        )
        return logits, vg_flat
    if is_fp8 and _FP8_PATH == "wmma":
        # fp8 (round G): the same WMMA geometry with E4M3 -> bf16 B fragments
        # and the per-token fp32 scale applied after the GEMM.  Same planes,
        # same full coverage (every logit written once), no memset.
        rows_per_block = wmma_rows_per_block(rows, num_requests)
        k_load = _resolve_fp8_k_load(rows_per_block)
        if k_load == "tdm" and kv_u8.data_ptr() % 128:
            rows_per_block = 16
            k_load = "raw"
        grid_x, grid_y = wmma_grid(rows, columns, rows_per_block)
        e_kw = _NO_KW
        if rows_per_block == 32 and k_load == "tdm" and _ACCESS == "raw":
            # ROUND-R2E: data-driven fp8 prefill route (QSA_FP8_PREFILL_ROUTE; the R2C
            # env switches QSA_FP8_KT / QSA_FP8_EARLY=1 take precedence, unchanged).
            f_kind, f_kw = prefill_fp8_route(rows, num_requests, columns, page)
            if f_kind == "early":
                e_kw = {"early": True}
            elif f_kind == "ktile":
                k_load = "ktile"  # ROUND-R2C fp8 ktile
                e_kw = dict(
                    {"kt_rows": f_kw["kt_rows"], "col_tiles": f_kw["col_tiles"]},
                    **({"lean": True} if f_kw.get("lean") else {}),
                    **({"ipf": f_kw["ipf"]} if f_kw.get("ipf") else {}),
                    **({"kconv": True} if f_kw.get("kconv") else {}),
                )
                grid_x, grid_y = kt_grid(
                    rows, columns, f_kw["kt_rows"], f_kw["col_tiles"]
                )
        _run(
            _build_wmma_fp8_kernel(
                float(divisor), rows_per_block, k_load, **qp_kw, **pg_kw, **e_kw
            ),
            q_u8,
            kv_u8,
            compressed_k_scale.view(num_pages, page, 1),
            page_table_i32,
            t2r_i32,
            qp_i32,
            cl_i32,
            logits,
            vg_flat,
            rs_flat,
            grid_x,
            grid_y,
            rows,
            columns,
            num_pages,
            num_requests,
            width,
            stream,
        )
        return logits, vg_flat

    threads = THREADS
    grid_x = (columns + threads - 1) // threads

    launcher = _build_kernel(is_fp8, threads, float(divisor), **qp_kw, **pg_kw)
    ks = compressed_k_scale.view(num_pages, page, 1) if is_fp8 else _dummy_scale(dev)
    _run(
        launcher,
        q_u8,
        kv_u8,
        ks,
        page_table_i32,
        t2r_i32,
        qp_i32,
        cl_i32,
        logits,
        vg_flat,
        rs_flat,
        grid_x,
        rows,
        columns,
        num_pages,
        num_requests,
        width,
        stream,
    )
    return logits, vg_flat

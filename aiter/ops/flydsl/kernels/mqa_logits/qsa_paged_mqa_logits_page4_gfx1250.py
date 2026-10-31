# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QSA index scorer (paged MQA logits), compressed page = 4, gfx1250 — **v0 baseline**.

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
row and its key row from L1 — deliberately untuned.  See
``records/g2/v0-manifest.json`` for the first-look latency (exploratory, not
evidence).

Layout facts this kernel relies on (all contiguous, checked by the wrapper):

* ``q``: ``[rows, 4, 128]`` bf16, 256 B per head row;
* bf16 cache: one page = 4 rows * 128 * 2 B = 1024 B, row ``j`` at ``j * 256``;
* fp8 cache: one page = 4 rows * 128 B = 512 B, row ``j`` at ``j * 128``;
* fp8 scale: one page = 4 fp32 = 16 B, row ``j`` at ``j * 4``.

The module imports only ``flydsl`` and ``torch``: the few buffer primitives it
needs (byte-addressed ``buffer_load``/``buffer_store`` with hardware OOB
drop, and a cached ``flyc.compile`` dispatch) are implemented here on
published FlyDSL APIs instead of importing the ``aiter`` package.
"""

from __future__ import annotations

import math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr

Vec = fx.Vector

PAGE_SIZE = 4          # compressed rows per index-cache page (= KV block 16 / ratio 4)
NUM_HEADS = 4          # index heads
HEAD_DIM = 128         # index head dim
DIM_PER_STEP = 8       # head dims consumed per unrolled step (16 B of Q)
THREADS = 256          # one thread per output column
MAX_GRID_Y = 65535
_NEG_INF = float("-inf")
_DROP_BYTE = 0x7FFFFFFF  # out-of-bounds byte offset: the V# drops the access
_ELEM_BYTES = {fx.Int8: 1, fx.Int32: 4, fx.Float32: 4, fx.BFloat16: 2}


# ---------------------------------------------------------------------------
# Minimal buffer primitives (published FlyDSL APIs only)
# ---------------------------------------------------------------------------
def _byte_ptr(tensor, num_bytes):
    """Byte-addressed buffer pointer (V#) for a kernel tensor argument.

    ``num_bytes`` is the descriptor's ``num_records``: loads and stores past it
    are dropped by the hardware, which is how the masked tail stores work.
    """
    return fx.rocdl.make_buffer_ptr(
        fx.recast_iter(fx.Int8, fx.get_iter(tensor)),
        num_records_bytes=num_bytes,
    )


def _gload(ptr, byte_off, n, elem):
    """``n`` elements of ``elem`` at ``byte_off`` (one buffer_load_dwordx n*4)."""
    nbytes = n * _ELEM_BYTES[elem]
    src = fx.make_view(ptr + byte_off, fx.make_layout(nbytes, 1))
    reg = fx.make_rmem_tensor(fx.make_layout(n, 1), elem)
    fx.copy_atom_call(
        fx.make_copy_atom(fx.rocdl.BufferCopy(nbytes * 8), fx.Int8), src, reg
    )
    return fx.memref_load_vec(reg)


def _gstore(ptr, byte_off, vec, n, elem):
    nbytes = n * _ELEM_BYTES[elem]
    dst = fx.make_view(ptr + byte_off, fx.make_layout(nbytes, 1))
    reg = fx.make_rmem_tensor(fx.make_layout(n, 1), elem)
    fx.memref_store_vec(vec, reg)
    fx.copy_atom_call(
        fx.make_copy_atom(fx.rocdl.BufferCopy(nbytes * 8), fx.Int8), reg, dst
    )


def _ld_i32(ptr, byte_off):
    return fx.Int32(Vec(_gload(ptr, byte_off, 1, fx.Int32))[0])


def _ld_f32(ptr, byte_off):
    return fx.Float32(Vec(_gload(ptr, byte_off, 1, fx.Float32))[0])


def _st_i32(ptr, byte_off, value):
    _gstore(ptr, byte_off, Vec.from_elements([value], dtype=fx.Int32), 1, fx.Int32)


def _st_f32(ptr, byte_off, value):
    _gstore(ptr, byte_off, Vec.from_elements([value], dtype=fx.Float32), 1, fx.Float32)


def _imin(a, b):
    return (a <= b).select(a, b)


def _imax(a, b):
    return (a >= b).select(a, b)


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
_F32_BIAS = fx.Int32(120 << 23)        # 0x3C000000
_F32_SUB_BIAS = fx.Int32(0x46800000)   # 2**14, mantissa LSB = 2**-9
_F32_SUB_SUB = fx.Float32(16384.0)     # 2**14
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
@lru_cache(maxsize=None)
def _build_kernel(is_fp8: bool, threads: int, divisor: float):
    """One specialization per (dtype, block size, score divisor)."""
    key_row_bytes = HEAD_DIM if is_fp8 else HEAD_DIM * 2   # 128 B fp8 / 256 B bf16
    q_row_bytes = NUM_HEADS * HEAD_DIM * 2                 # 1024 B bf16
    page_bytes = PAGE_SIZE * key_row_bytes                 # 512 B fp8 / 1024 B bf16
    k_dwords = 2 if is_fp8 else 4                          # dwords per 8 head dims
    k_bytes = k_dwords * 4                                 # 8 B fp8 / 16 B bf16
    steps = HEAD_DIM // DIM_PER_STEP                       # 16 steps of 8 dims
    inv_divisor = 1.0 / divisor
    per_page = NUM_HEADS * HEAD_DIM                        # elements of one q row
    dtype_tag = "fp8" if is_fp8 else "bf16"
    kernel_name = (
        f"qsa_paged_mqa_logits_page4_gfx1250_{dtype_tag}"
        f"_D{HEAD_DIM}_H{NUM_HEADS}_P{PAGE_SIZE}_t{threads}_flydsl"
    )

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
        return num_pages * PAGE_SIZE * 4

    def _ks_bytes_bf16(num_pages):   # no scale plane: a 4-byte dummy descriptor
        return 4

    def _scale_fp8(ks_ptr, phys, off):
        return _ld_f32(ks_ptr, (phys * PAGE_SIZE + off) * 4)

    def _scale_bf16(ks_ptr, phys, off):   # no scale: the dequant multiplier is 1
        return fx.Float32(1.0)

    if is_fp8:
        key_vals, ks_bytes, scale_of = _key_vals_fp8, _ks_bytes_fp8, _scale_fp8
    else:
        key_vals, ks_bytes, scale_of = _key_vals_bf16, _ks_bytes_bf16, _scale_bf16

    @flyc.kernel(name=kernel_name, known_block_size=[threads, 1, 1])
    def kernel(
        Q,           # uint8 view of [rows, 4, 128] bf16
        KV,          # uint8 view of [pages, 4, 1, 128] bf16 or fp8
        KS,          # fp32 view of [pages, 4, 1]  (fp8 only; Q is the placeholder for bf16)
        PT,          # int32 [requests, width]
        T2R,         # int32 [rows]
        QP,          # int32 [rows]
        CL,          # int32 [requests]
        LOGITS,      # fp32 [rows, columns]
        VIS,         # int32 [rows]
        RST,         # int32 [rows]
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

        q_ptr = _byte_ptr(Q, rows * q_row_bytes)
        kv_ptr = _byte_ptr(KV, num_pages * page_bytes)
        pt_ptr = _byte_ptr(PT, num_requests * width * 4)
        t2r_ptr = _byte_ptr(T2R, rows * 4)
        qp_ptr = _byte_ptr(QP, rows * 4)
        cl_ptr = _byte_ptr(CL, num_requests * 4)
        logits_ptr = _byte_ptr(LOGITS, rows * columns * 4)
        vis_ptr = _byte_ptr(VIS, rows * 4)
        rst_ptr = _byte_ptr(RST, rows * 4)
        ks_ptr = _byte_ptr(KS, ks_bytes(num_pages))

        # --- row metadata: visible count (identical to the golden's visible_groups)
        req = _ld_i32(t2r_ptr, row * 4)
        req_ok = (req >= 0) & (req < num_requests)
        safe_req = req_ok.select(req, 0)
        ctx = _ld_i32(cl_ptr, safe_req * 4)
        pos = _ld_i32(qp_ptr, row * 4)
        # min is monotone, so min(a, b) // 4 == min(a // 4, b // 4); clamping to
        # >= 0 first keeps the division on a non-negative operand.
        head = _imax(_imin(pos + 1, ctx), 0)
        vis = head // PAGE_SIZE
        vis = req_ok.select(vis, 0)

        writer = (tid == 0) & (ptile == 0) & (row < rows)
        _st_i32(vis_ptr, writer.select(row * 4, _DROP_BYTE), vis)
        _st_i32(rst_ptr, writer.select(row * 4, _DROP_BYTE), fx.Int32(0))

        # --- column -> (logical page, row in page)
        col_ok = col < columns
        logical = col // PAGE_SIZE
        off = col % PAGE_SIZE
        logical_ok = logical < width
        safe_logical = logical_ok.select(logical, 0)
        phys = _ld_i32(pt_ptr, (safe_req * width + safe_logical) * 4)
        page_ok = req_ok & logical_ok & (phys >= 0) & (phys < num_pages)
        phys = page_ok.select(phys, 0)

        # --- score: sum_h relu(q[t,h,:] . k[c]) / sqrt(128)
        q_base = row * per_page * 2
        k_base = phys * page_bytes + off * key_row_bytes
        zero = fx.Float32(0.0)
        acc = [zero, zero, zero, zero]
        for s in range_constexpr(steps):
            kraw = Vec(_gload(kv_ptr, k_base + s * k_bytes, k_dwords, fx.Int32))
            kv = key_vals(kraw)
            for h in range_constexpr(NUM_HEADS):
                qraw = Vec(_gload(q_ptr, q_base + h * (HEAD_DIM * 2) + s * 16, 4, fx.Int32))
                qs = []
                for j in range_constexpr(4):
                    dw = fx.Int32(qraw[j])
                    qs.append((dw << 16).bitcast(fx.Float32))
                    qs.append(((dw >> 16) << 16).bitcast(fx.Float32))
                for d in range_constexpr(DIM_PER_STEP):
                    acc[h] = acc[h] + qs[d] * kv[d]

        scale = scale_of(ks_ptr, phys, off)
        total = (
            (acc[0] * scale).maximumf(zero)
            + (acc[1] * scale).maximumf(zero)
            + (acc[2] * scale).maximumf(zero)
            + (acc[3] * scale).maximumf(zero)
        )
        total = total * inv_divisor

        keep = page_ok & (col < vis)
        value = keep.select(total, fx.Float32(_NEG_INF))
        # masked store: `col_ok` sends the byte offset past the descriptor and
        # the hardware drops the access (same device as the visible mask).
        out_off = col_ok.select((row * columns + col) * 4, _DROP_BYTE)
        _st_f32(logits_ptr, out_off, value)

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
            Q, KV, KS, PT, T2R, QP, CL, LOGITS, VIS, RST,
            rows, columns, num_pages, num_requests, width,
        ).launch(
            grid=(fx.Int64(grid_x), fx.Int64(rows), 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch.kernel_name = kernel_name
    launch.dtype_tag = dtype_tag
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
def qsa_paged_mqa_logits_page4(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    compress_ratio: int = PAGE_SIZE,
    divisor: float | None = None,
    max_columns: int | None = None,
    compressed_k_scale: torch.Tensor | None = None,
    logits: torch.Tensor | None = None,
    visible_groups: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paged MQA logits for the QSA index scorer, ``page = 4`` (gfx1250).

    Returns ``(logits [rows, columns] fp32, visible_groups [rows] int32)``.
    ``-inf`` is written for every column with ``c >= visible[t]`` and for every
    column whose page-table entry is invalid.
    """
    if compress_ratio != PAGE_SIZE:
        raise ValueError(f"compress_ratio must be {PAGE_SIZE}, got {compress_ratio}")
    if q.dtype != torch.bfloat16:
        raise ValueError(f"q must be bfloat16, got {q.dtype}")
    if q.ndim != 3 or tuple(q.shape[1:]) != (NUM_HEADS, HEAD_DIM):
        raise ValueError(f"q must be [rows, {NUM_HEADS}, {HEAD_DIM}], got {tuple(q.shape)}")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    rows = int(q.shape[0])
    if rows == 0 or rows > MAX_GRID_Y:
        raise ValueError(f"rows must be in [1, {MAX_GRID_Y}], got {rows}")

    if compressed_k_cache.ndim != 4 or tuple(compressed_k_cache.shape[1:]) != (
        PAGE_SIZE, 1, HEAD_DIM,
    ):
        raise ValueError(
            "compressed_k_cache must be [pages, "
            f"{PAGE_SIZE}, 1, {HEAD_DIM}], got {tuple(compressed_k_cache.shape)}"
        )
    if not compressed_k_cache.is_contiguous():
        raise ValueError("compressed_k_cache must be contiguous")
    num_pages = int(compressed_k_cache.shape[0])
    is_fp8 = compressed_k_cache.dtype == torch.float8_e4m3fn
    if not is_fp8 and compressed_k_cache.dtype != torch.bfloat16:
        raise ValueError(
            "compressed_k_cache must be bfloat16 or float8_e4m3fn, got "
            f"{compressed_k_cache.dtype}"
        )
    if is_fp8:
        if compressed_k_scale is None:
            raise ValueError("fp8 index cache requires its fp32 [pages, 4, 1] scale")
        if compressed_k_scale.dtype != torch.float32:
            raise ValueError(
                f"compressed_k_scale must be float32, got {compressed_k_scale.dtype}"
            )
        if tuple(compressed_k_scale.shape) != (num_pages, PAGE_SIZE, 1):
            raise ValueError(
                f"compressed_k_scale must be {(num_pages, PAGE_SIZE, 1)}, got "
                f"{tuple(compressed_k_scale.shape)}"
            )
        if not compressed_k_scale.is_contiguous():
            raise ValueError("compressed_k_scale must be contiguous")
    elif compressed_k_scale is not None:
        raise ValueError("bf16 index cache takes no scale (got compressed_k_scale)")

    if page_table.ndim != 2 or page_table.shape[0] == 0:
        raise ValueError(f"page_table must be [requests, width], got {tuple(page_table.shape)}")
    if page_table.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"page_table must be int32/int64, got {page_table.dtype}")
    if not page_table.is_contiguous():
        raise ValueError("page_table must be contiguous")
    num_requests = int(page_table.shape[0])
    width = int(page_table.shape[1])

    columns_full = width * PAGE_SIZE
    columns = columns_full if max_columns is None else max(1, min(columns_full, int(max_columns)))

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
    elif logits.shape != (rows, columns) or logits.dtype != torch.float32 or not logits.is_contiguous():
        raise ValueError(f"logits must be contiguous fp32 {(rows, columns)}")
    if visible_groups is None:
        visible_groups = torch.empty(rows, dtype=torch.int32, device=dev)
    elif visible_groups.numel() != rows or visible_groups.dtype != torch.int32:
        raise ValueError(f"visible_groups must be int32 [{rows}]")
    if row_starts is None:
        row_starts = torch.empty(rows, dtype=torch.int32, device=dev)
    elif row_starts.numel() != rows or row_starts.dtype != torch.int32:
        raise ValueError(f"row_starts must be int32 [{rows}]")

    # Explicit, checked narrowings (never a silent cast): the kernel reads int32
    # indices and FlyDSL passes tensor extents as int32.
    page_table_i32 = page_table.to(torch.int32) if page_table.dtype != torch.int32 else page_table
    t2r_i32 = token_to_request.to(torch.int32).contiguous().view(-1)
    if query_positions.dtype != torch.int32:
        if bool((query_positions.abs() > 2**31 - 1).any()):
            raise ValueError("query_positions values do not fit int32")
        qp_i32 = query_positions.to(torch.int32).contiguous().view(-1)
    else:
        qp_i32 = query_positions.contiguous().view(-1)
    cl_i32 = context_lens.to(torch.int32).contiguous().view(-1)

    if divisor is None:
        divisor = math.sqrt(HEAD_DIM)
    threads = THREADS
    grid_x = (columns + threads - 1) // threads

    launcher = _build_kernel(is_fp8, threads, float(divisor))
    kv_u8 = compressed_k_cache.view(torch.uint8).view(num_pages, -1)
    q_u8 = q.view(torch.uint8).view(rows, -1)
    ks = compressed_k_scale.view(num_pages, PAGE_SIZE, 1) if is_fp8 else _dummy_scale(dev)
    stream = torch.cuda.current_stream(dev)
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
        visible_groups.view(-1),
        row_starts.view(-1),
        grid_x,
        rows,
        columns,
        num_pages,
        num_requests,
        width,
        stream,
    )
    return logits, visible_groups.view(-1)

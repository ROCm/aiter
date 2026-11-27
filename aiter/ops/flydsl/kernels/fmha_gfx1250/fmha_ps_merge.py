# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Split-KV merge for the persistent a8w8 m32x8 launch (``thd_ps``).

The FMHA kernel writes, per split unit u (one m-tile x head whose KV range was cut into
``parts`` pieces), a table entry ``meta[HDR + u*STRIDE + 0..4] = (slot0, parts, out_row0,
nrows, head)`` and each part's normalized O (bf16, ``[256, D]``) and natural-log LSE
(``[256]``) at workspace slot ``slot0 + part``. Word 0 of the table is the split-unit
count. Persistent programs stride over the units, combine the parts by LSE and write the
unit's rows of ``out`` [T, H, D] and ``lse`` [H, T] (strided).
"""

import triton
import triton.language as tl


@triton.jit
def _ps_merge_kernel(
    ws_o,
    ws_lse,
    meta,
    out,
    lse,
    s_out_t,
    s_out_h,
    s_lse_h,
    s_lse_t,
    HDR: tl.constexpr,
    STRIDE: tl.constexpr,
    BM: tl.constexpr,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    MAXP: tl.constexpr,
    CARRY: tl.constexpr,
):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    n_units = tl.load(meta)
    n_work = n_units * (BM // ROWS)
    d = tl.arange(0, D)
    for w in range(pid, n_work, npg):
        u = w // (BM // ROWS)
        rb = w % (BM // ROWS)
        e = meta + HDR + u * STRIDE
        slot0 = tl.load(e)
        parts = tl.load(e + 1)
        row0 = tl.load(e + 2)
        nrows = tl.load(e + 3)
        head = tl.load(e + 4)
        r = rb * ROWS + tl.arange(0, ROWS)
        rmask = r < nrows
        # all parts' LSE first (independent loads), then the weighted O sum
        m = tl.full([ROWS], float("-inf"), tl.float32)
        for p in tl.static_range(MAXP):
            base = (slot0 + p).to(tl.int64) * BM
            m = tl.maximum(
                m,
                tl.load(
                    ws_lse + base + r, mask=rmask & (p < parts), other=float("-inf")
                ),
            )
        t = (row0 + r).to(tl.int64)
        if CARRY:  # the running state of these rows joins the merge
            l_old = tl.load(
                lse + head * s_lse_h + t * s_lse_t,
                mask=rmask,
                other=float("-inf"),
                cache_modifier=".cv",
            )
            m = tl.maximum(m, l_old)
        ms = tl.where(m == float("-inf"), 0.0, m)
        acc = tl.zeros([ROWS, D], tl.float32)
        den = tl.zeros([ROWS], tl.float32)
        if CARRY:
            has_old = l_old > float("-inf")
            w_old = tl.where(has_old, tl.exp(l_old - ms), 0.0)
            o_old = tl.load(
                out + t[:, None] * s_out_t + head * s_out_h + d[None, :],
                mask=(rmask & has_old)[:, None],
                other=0.0,
                cache_modifier=".cv",
            )
            acc += w_old[:, None] * o_old.to(tl.float32)
            den += w_old
        for p in tl.static_range(MAXP):
            if p < parts:
                base = (slot0 + p).to(tl.int64) * BM
                wgt = tl.exp(
                    tl.load(ws_lse + base + r, mask=rmask, other=float("-inf")) - ms
                )
                op = tl.load(
                    ws_o + (base + r)[:, None] * D + d[None, :],
                    mask=rmask[:, None],
                    other=0.0,
                )
                acc += wgt[:, None] * op.to(tl.float32)
                den += wgt
        o = tl.where(den[:, None] > 0, acc / den[:, None], 0.0)
        tl.store(
            out + t[:, None] * s_out_t + head * s_out_h + d[None, :],
            o.to(out.dtype.element_ty),
            mask=rmask[:, None],
        )
        tl.store(
            lse + head * s_lse_h + t * s_lse_t,
            tl.where(den > 0, ms + tl.log(den), float("-inf")),
            mask=rmask,
        )


try:  # bind launch arguments once per shape (eager prefill calls are host-sensitive)
    from aiter.ops.triton._triton_kernels.kimi_delta_attn.fast_launch import fast_launch

    _ps_merge_kernel = fast_launch(_ps_merge_kernel)
except ImportError:  # pragma: no cover
    pass

_ROWS = 32


def ps_merge(
    ws_o,
    ws_lse,
    meta,
    out,
    lse,
    *,
    num_programs,
    block_m,
    meta_hdr,
    meta_stride,
    max_parts,
    carry_state=False,
):
    D = out.shape[-1]
    _ps_merge_kernel[(num_programs,)](
        ws_o=ws_o,
        ws_lse=ws_lse,
        meta=meta,
        out=out,
        lse=lse,
        s_out_t=out.stride(0),
        s_out_h=out.stride(1),
        s_lse_h=lse.stride(0),
        s_lse_t=lse.stride(1),
        HDR=meta_hdr,
        STRIDE=meta_stride,
        BM=block_m,
        ROWS=_ROWS,
        D=D,
        MAXP=max_parts,
        CARRY=bool(carry_state),
        num_warps=4,
    )

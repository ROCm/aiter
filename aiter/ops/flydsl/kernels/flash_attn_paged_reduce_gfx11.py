# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Segment merge for the gfx11 paged decode kernel: one wave per (sequence,
query token, head) reads the segments' running max, exp-sum and unnormalized O
that ``flash_attn_paged_decode_gfx11.py`` wrote, and writes the normalized
output.

The grid covers ``max_seqlen_q`` token slots per sequence; a wave whose slot
is past its sequence's tokens, or whose sequence has more than ``q_small``
queries (the prefill kernel's), exits at once, so a mixed batch can split the
decode rows' keys into segments while the prefill rows are produced by the
other kernel. A segment without keys for a row carries exp-sum 0 (and max
-inf) and contributes nothing; a row whose segments are all empty gets 0.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from . import buffer_ops
from .flash_attn_func_gfx11 import WAVE_SIZE, WMMA_MNK
from .tensor_shim import (
    _run_compiled,
    buf_base_i64,
    buf_copy_load,
    buf_copy_store,
    ptr_arg,
    ptr_buf_tensor,
)

KERNEL_NAME = "flash_attn_paged_reduce_gfx11_kernel"
NEG_INF = float("-inf")


def build_flash_attn_paged_reduce_module_gfx11(
    num_heads, head_dim, dtype_str="bf16", fast_fp_math=True, unsafe_fp_math=True
):
    """Build the merge and return ``launch(segm_output, segm_max, segm_expsum,
    out, cu_seqlens_q, batch_size, num_segments, q_small, stream=None)``.

    ``segm_output`` fp32 ``[batch * q_small, num_heads, num_segments,
    next_pow2(head_dim)]``, ``segm_max`` / ``segm_expsum`` fp32 ``[batch *
    q_small, num_heads, num_segments]`` (``segm_max`` in log2 units), row ``b *
    q_small + t`` for token ``t`` of sequence ``b``, as the decode kernel writes
    them with ``max_seqlen_q = q_small``; ``out`` packed THD ``[total_q,
    num_heads, head_dim]`` in ``dtype_str``. Tokens of sequences with more than
    ``q_small`` queries are not written.
    """
    if head_dim % 8 != 0 or head_dim <= 0:
        raise ValueError(f"head_dim must be a positive multiple of 8, got {head_dim}")
    if dtype_str not in ("bf16", "f16"):
        raise ValueError(f"dtype_str must be bf16 or f16, got {dtype_str!r}")
    H = num_heads
    D = head_dim
    DP = -(-D // WMMA_MNK) * WMMA_MNK
    DPAD2 = 1 << (D - 1).bit_length()  # the partials' row length
    CHUNKS = -(-DP // (WAVE_SIZE * 8))  # 8 fp32 per lane per chunk
    TOK = H * D
    elem = fx.BFloat16 if dtype_str == "bf16" else fx.Float16

    def _exp2(x):
        return fx.Float32(fx.rocdl.exp2(T.f32, fx.Float32(x).ir_value()))

    @flyc.kernel(known_block_size=[WAVE_SIZE, 1, 1])
    def flash_attn_paged_reduce_kernel(
        SEGM_O: fx.Pointer,
        SEGM_M: fx.Pointer,
        SEGM_L: fx.Pointer,
        O: fx.Pointer,
        CU_Q: fx.Pointer,
        batch_size: fx.Int32,
        num_segments: fx.Int32,
        q_small: fx.Int32,
        total_q: fx.Int32,
    ):
        lane = fx.thread_idx.x
        head = fx.block_idx.x
        t = fx.block_idx.y
        bat = fx.block_idx.z

        cu_rsrc = buffer_ops.create_buffer_resource_from_addr(
            buf_base_i64(CU_Q), num_records_bytes=(fx.Int64(batch_size) + 1) * 4
        )

        def cu_at(idx):
            return fx.Int32(
                buffer_ops.buffer_load(
                    cu_rsrc, fx.Int32(idx), vec_width=1, is_scalar=True
                )
            )

        q_start = cu_at(bat)
        seq_q = cu_at(bat + 1) - q_start
        # A decode-kernel sequence's token (q_small is at most the grid's slots).
        mine = (t < seq_q) & (seq_q <= q_small)

        def _body():
            # Segment statistics: scalar loads (the same for every lane).
            rows64 = fx.Int64(batch_size) * fx.Int64(q_small) * H
            nseg64 = fx.Int64(num_segments)
            m_rsrc = buffer_ops.create_buffer_resource_from_addr(
                buf_base_i64(SEGM_M), num_records_bytes=rows64 * nseg64 * 4
            )
            l_rsrc = buffer_ops.create_buffer_resource_from_addr(
                buf_base_i64(SEGM_L), num_records_bytes=rows64 * nseg64 * 4
            )
            o_tab = ptr_buf_tensor(
                buf_base_i64(SEGM_O),
                elem=fx.Float32,
                unit_elems=4,  # a buffer access is at most 16 bytes: 4 fp32
                unit_stride=1,
                num_records_bytes=rows64 * nseg64 * (DPAD2 * 4),
            )
            base = ((bat * q_small + t) * H + head) * num_segments

            def stat(rsrc, s):
                v = buffer_ops.buffer_load(rsrc, base + s, vec_width=1, is_scalar=True)
                return fx.Int32(v).bitcast(fx.Float32)

            # Pass 1: the row's overall max over the segments with keys. (Two
            # loop-carried values: a single one comes back as a bare scalar.)
            for _s, st in range(
                0, num_segments, 1, init=[fx.Float32(NEG_INF), fx.Float32(0.0)]
            ):
                s = fx.Int32(_s)
                m_s = stat(m_rsrc, s)
                l_s = stat(l_rsrc, s)
                m_s = (l_s > 0.0).select(m_s, fx.Float32(NEG_INF))
                mx = yield [fx.maxnumf(fx.Float32(st[0]), m_s), fx.Float32(st[1])]
            M = fx.Float32(mx[0])

            # Pass 2: weighted exp-sum and O.
            zero8 = Vec.filled(8, 0.0, fx.Float32)
            d0 = [lane * 8 + c * (WAVE_SIZE * 8) for c in range_constexpr(CHUNKS)]
            OOB = fx.Int32(0x3FFFFFF0)
            init = [fx.Float32(0.0)] + [zero8 for _ in range_constexpr(CHUNKS)]
            for _s, st in range(0, num_segments, 1, init=init):
                s = fx.Int32(_s)
                m_s = stat(m_rsrc, s)
                l_s = stat(l_rsrc, s)
                has = l_s > 0.0
                w = has.select(_exp2(m_s - M), fx.Float32(0.0))
                L = fx.Float32(st[0]) + w * l_s
                accs = []
                row = (base + s) * DPAD2
                for c in range_constexpr(CHUNKS):
                    acc = Vec(st[1 + c])
                    # columns past the head dim (lanes idle for this chunk) read 0
                    live = d0[c] < D
                    off_lo = live.select(row + d0[c], OOB)
                    off_hi = live.select(row + d0[c] + 4, OOB)
                    o_lo = Vec(
                        buf_copy_load(o_tab, off_lo, elem=fx.Float32, unit_elems=4)
                    )
                    o_hi = Vec(
                        buf_copy_load(o_tab, off_hi, elem=fx.Float32, unit_elems=4)
                    )
                    o_s = [fx.Float32(o_lo[i]) for i in range_constexpr(4)] + [
                        fx.Float32(o_hi[i]) for i in range_constexpr(4)
                    ]
                    upd = Vec.from_elements(
                        [fx.Float32(acc[i]) + o_s[i] * w for i in range_constexpr(8)],
                        fx.Float32,
                    )
                    # an empty segment's O slot is unwritten: do not touch acc with it
                    accs.append(has.select(upd, acc))
                res = yield [L] + accs

            L = fx.Float32(res[0])
            inv = (L > 0.0).select(fx.Float32(1.0) / L, fx.Float32(0.0))
            o_buf = ptr_buf_tensor(
                buf_base_i64(O),
                elem=elem,
                unit_elems=8,
                unit_stride=1,
                num_records_bytes=fx.Int64(total_q) * (TOK * 2),
            )
            token = q_start + t
            for c in range_constexpr(CHUNKS):
                o = Vec(res[1 + c])
                vals = [(fx.Float32(o[i]) * inv).to(elem) for i in range_constexpr(8)]
                off = token * TOK + head * D + d0[c]
                off = (d0[c] < D).select(off, OOB)
                buf_copy_store(
                    o_buf, off, Vec.from_elements(vals, elem), elem=elem, unit_elems=8
                )

        if mine:
            _body()

    @flyc.jit
    def launch_flash_attn_paged_reduce(
        SEGM_O: fx.Pointer,
        SEGM_M: fx.Pointer,
        SEGM_L: fx.Pointer,
        O: fx.Pointer,
        CU_Q: fx.Pointer,
        batch_size: fx.Int32,
        num_segments: fx.Int32,
        q_small: fx.Int32,
        total_q: fx.Int32,
        stream: fx.Stream = fx.Stream(  # noqa: B008  framework idiom: default is evaluated once at import on purpose
            None
        ),
    ):
        flash_attn_paged_reduce_kernel(
            SEGM_O,
            SEGM_M,
            SEGM_L,
            O,
            CU_Q,
            batch_size,
            num_segments,
            q_small,
            total_q,
        ).launch(grid=(H, q_small, batch_size), block=(WAVE_SIZE, 1, 1), stream=stream)

    launch_flash_attn_paged_reduce.compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
    }

    def _launch(
        segm_output,
        segm_max,
        segm_expsum,
        out,
        cu_seqlens_q,
        batch_size,
        num_segments,
        q_small,
        stream=None,
    ):
        _run_compiled(
            launch_flash_attn_paged_reduce,
            ptr_arg(segm_output),
            ptr_arg(segm_max),
            ptr_arg(segm_expsum),
            ptr_arg(out),
            ptr_arg(cu_seqlens_q, fx.Int32),
            batch_size,
            int(num_segments),
            int(q_small),
            out.shape[0],
            fx.Stream(None) if stream is None else stream,
        )

    _launch.dpad2 = DPAD2
    return _launch

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 Gluon kernel of ``attn_res_gate``.

The math is the packed + prefix ``SEPARATE`` path of ``attnres_fwd_kernel``.
The block_residual rows go from HBM to LDS with TENSOR_LOAD_TO_LDS, ``PREFETCH``
rows before the row that the loop reduces. This hides the row load latency
without more VGPRs.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_attn_res_gate_tdm_kernel_gfx1250_repr = make_kernel_repr(
    "attn_res_gate_tdm_kernel_gfx1250",
    [
        "D",
        "BD",
        "NUM_WARPS",
        "PREFETCH",
        "HAS_ONORM",
        "DO_ADD",
        "DO_ADD2",
        "WRITE_PREF",
        "QUANT_FP8",
    ],
)


@gluon.jit(repr=_attn_res_gate_tdm_kernel_gfx1250_repr)
def attn_res_gate_tdm_kernel_gfx1250(
    res_packed,
    prefix,
    w,
    ow,
    o,
    o_scale,
    add_hidden,
    add_hidden2,
    prefix_out,
    B,
    stride_res_n,
    stride_res_l,
    eps,
    out_eps,
    scale,
    FP8_MAX,
    D: gl.constexpr,
    BD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PREFETCH: gl.constexpr,
    HAS_ONORM: gl.constexpr,
    DO_ADD: gl.constexpr,
    DO_ADD2: gl.constexpr,
    WRITE_PREF: gl.constexpr,
    QUANT_FP8: gl.constexpr,
):
    """One program per token. ``prefix``, ``add_hidden``, ``add_hidden2``,
    ``prefix_out`` and ``o`` are contiguous ``[N, D]``. Requires B >= 1."""
    gl.static_assert(PREFETCH >= 1 and PREFETCH <= 3)
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [NUM_WARPS], [0])
    t = gl.program_id(0).to(gl.int64)
    o_d = gl.arange(0, BD, layout=layout)
    m_d = o_d < D
    b_qw = gl.amd.cdna5.buffer_load(w, o_d, mask=m_d, other=0.0).to(gl.float32)
    # Issue the prefix loads before the TDM transfers, so that all loads of the
    # token are in flight together.
    ps_raw = gl.amd.cdna5.buffer_load(prefix + t * D, o_d, mask=m_d, other=0.0)
    if DO_ADD:
        hs_raw = gl.amd.cdna5.buffer_load(add_hidden + t * D, o_d, mask=m_d, other=0.0)
    if DO_ADD2:
        hs2_raw = gl.amd.cdna5.buffer_load(
            add_hidden2 + t * D, o_d, mask=m_d, other=0.0
        )

    # One [1, BD] LDS tile for each row. TDM writes zeros in columns D..BD.
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=res_packed + t * stride_res_n,
        shape=(B, D),
        strides=(stride_res_l, 1),
        block_shape=(1, BD),
        layout=shared,
    )
    rows = gl.allocate_shared_memory(
        res_packed.dtype.element_ty, [PREFETCH + 1, 1, BD], shared
    )
    # The first rows load while the prefix work runs.
    for j in gl.static_range(PREFETCH):
        gl.amd.cdna5.tdm.async_load(desc, [j, 0], rows.index(j), pred=j < B)

    ps = ps_raw.to(gl.float32)
    if DO_ADD:
        ps += hs_raw.to(gl.float32)
    if DO_ADD2:
        ps += hs2_raw.to(gl.float32)
    if WRITE_PREF:
        gl.amd.cdna5.buffer_store(
            ps.to(prefix_out.dtype.element_ty), prefix_out + t * D, o_d, mask=m_d
        )
    # The prefix is the first softmax term.
    rstd = gl.rsqrt(gl.sum(ps * ps, axis=0) * (1.0 / D) + eps)
    b_m = gl.sum(ps * b_qw, axis=0) * rstd * scale
    b_acc = gl.full([], 1.0, dtype=gl.float32)
    b_o = ps
    for i_l in range(B):
        # Wait until row i_l is in LDS. Each wave waits for its own slice.
        if PREFETCH == 1:
            gl.amd.cdna5.tdm.async_wait(0)
        elif PREFETCH == 2:
            if i_l + 1 < B:
                gl.amd.cdna5.tdm.async_wait(1)
            else:
                gl.amd.cdna5.tdm.async_wait(0)
        else:
            if i_l + 2 < B:
                gl.amd.cdna5.tdm.async_wait(2)
            elif i_l + 1 < B:
                gl.amd.cdna5.tdm.async_wait(1)
            else:
                gl.amd.cdna5.tdm.async_wait(0)
        # The compiler puts an s_barrier after this wait. After it, all slices of
        # row i_l are in LDS and no wave reads row i_l - 1, which the refill
        # overwrites. Do not add gl.barrier(): it is a second barrier and a fence.
        gl.amd.cdna5.tdm.async_load(
            desc,
            [i_l + PREFETCH, 0],
            rows.index((i_l + PREFETCH) % (PREFETCH + 1)),
            pred=i_l + PREFETCH < B,
        )
        v = rows.index(i_l % (PREFETCH + 1)).reshape([BD]).load(layout).to(gl.float32)
        rstd = gl.rsqrt(gl.sum(v * v, axis=0) * (1.0 / D) + eps)
        s = gl.sum(v * b_qw, axis=0) * rstd * scale
        b_mp = b_m
        b_m = gl.maximum(b_m, s)
        r = gl.exp(b_mp - b_m)
        p = gl.exp(s - b_m)
        b_acc = b_acc * r + p
        b_o = b_o * r + p * v

    b_o = b_o * (1.0 / b_acc)
    if HAS_ONORM:
        b_o_rstd = gl.rsqrt(
            gl.sum(gl.where(m_d, b_o * b_o, 0.0), axis=0) * (1.0 / D) + out_eps
        )
        b_ow = gl.amd.cdna5.buffer_load(ow, o_d, mask=m_d, other=0.0).to(gl.float32)
        b_o = b_o * b_o_rstd * b_ow
    if QUANT_FP8:
        # Same scale as the Triton kernel, which is bit-exact with the HIP
        # per-token quant. An all-zero row gets scale 0.
        b_amax = gl.max(gl.abs(gl.where(m_d, b_o, 0.0)), axis=0)
        b_oscale = b_amax * (1.0 / FP8_MAX)
        gl.store(o_scale + t, b_oscale.to(o_scale.dtype.element_ty))
        b_o = b_o * gl.where(b_oscale > 0.0, 1.0 / b_oscale, 0.0)
    gl.amd.cdna5.buffer_store(b_o.to(o.dtype.element_ty), o + t * D, o_d, mask=m_d)

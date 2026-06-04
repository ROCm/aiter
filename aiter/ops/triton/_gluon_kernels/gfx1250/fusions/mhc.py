# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx1250) port of ``_mhc_post_pre_split_kernel``.

Mirrors ``aiter/ops/triton/_triton_kernels/fusions/mhc.py::_mhc_post_pre_split_kernel``
but replaces every ``tl.load`` with ``gfx1250.tdm.async_load``. Generic over
``n`` (still a ``gl.constexpr``): per-stream scalars are extracted by slicing
the shared-memory tiles that hold the 2D post / comb loads, so the code does
not need any ``n``-specific ``gl.split`` chain.

Per CTA (one (M-tile, C-tile)):
  TDM loads in flight before the single ``async_wait(0)``:
    * post_mix (BLOCK_M, n)         : 1 desc, 1 load
    * comb_mix (BLOCK_M, n*n) flat  : 1 desc, 1 load
    * layer_input (BLOCK_M, BLOCK_C): 1 desc, 1 load
    * residual_in per stream        : 1 desc, n loads
    * phi per stream                : 1 desc, n loads
  → 3 + 2n in-flight TDM groups.

Per-stream extraction:
  After the wait, post_mix and comb_mix stay in shared. For each h_dst (and
  for each h_src in the inner loop), we ``slice`` the 2D smem to a (BLOCK_M, 1)
  column and ``gl.sum(..., axis=1)`` to a (BLOCK_M,) 1D tensor. The slice +
  sum pair avoids the ``gl.split`` chain entirely and works for any ``n``.
"""

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_mhc_post_pre_split_kernel_repr = make_kernel_repr(
    "_mhc_post_pre_split_kernel",
    [
        "n",
        "C",
        "stride_phi_k",
        "stride_phi_n",
        "BLOCK_M",
        "BLOCK_C",
        "N_TOTAL_POW2",
    ],
)


@gluon.jit(repr=_mhc_post_pre_split_kernel_repr)
def _mhc_post_pre_split_kernel(
    layer_input_ptr,
    residual_in_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    residual_out_ptr,
    phi_ptr,
    acc_ptr,
    acc_sq_ptr,
    M,
    N: gl.constexpr,
    n: gl.constexpr,
    C: gl.constexpr,
    stride_x_m,
    stride_x_c,
    stride_resin_m,
    stride_resin_n,
    stride_resin_c,
    stride_post_m,
    stride_post_n,
    stride_comb_m,
    stride_comb_src,
    stride_comb_dst,
    stride_resout_m,
    stride_resout_n,
    stride_resout_c,
    stride_phi_k: gl.constexpr,
    stride_phi_n: gl.constexpr,
    stride_acc_k,
    stride_acc_m,
    stride_acc_n,
    stride_acc_sq_k,
    stride_acc_sq_m,
    BLOCK_M: gl.constexpr,
    BLOCK_C: gl.constexpr,
    N_TOTAL_POW2: gl.constexpr,
):
    pid_m = gl.program_id(0)
    pid_c = gl.program_id(1)

    NUM_WARPS: gl.constexpr = 4
    WARP_SIZE: gl.constexpr = 32

    out_dtype: gl.constexpr = residual_out_ptr.dtype.element_ty
    phi_dtype: gl.constexpr = phi_ptr.dtype.element_ty
    in_dtype: gl.constexpr = residual_in_ptr.dtype.element_ty
    x_dtype: gl.constexpr = layer_input_ptr.dtype.element_ty

    # ---- WMMA / dot layouts for the next-pre GEMM (BLOCK_M x N_TOTAL_POW2). ----
    mfma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        instr_shape=[16, 16, 32],
        warp_bases=[[1, 0], [0, 1]],
    )
    K_WIDTH: gl.constexpr = 8
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=K_WIDTH
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=K_WIDTH
    )

    # 2D blocked layout for (BLOCK_M, BLOCK_C) tiles. Contiguous along C.
    layout_mc: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 4],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    # 1D-along-M layout (sliced from layout_mc on dim 1).
    layout_m: gl.constexpr = gl.SliceLayout(1, layout_mc)
    # 2D layout for the (BLOCK_M, 1) column extracted from shared.
    layout_m1: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[32, 1],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    # Layout for the phi (BLOCK_C, N_TOTAL_POW2) tile.
    layout_phi: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 4],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    m_base = pid_m * BLOCK_M
    c_base = pid_c * BLOCK_C

    # ---- Shared staging buffers ----
    pm_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, n], SH)
    cm_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, n * n], SH)
    x_smem = gl.allocate_shared_memory(x_dtype, [BLOCK_M, BLOCK_C], SH)
    res_smem = gl.allocate_shared_memory(in_dtype, [n, BLOCK_M, BLOCK_C], SH)
    # phi is K-major in HBM (stride_phi_k=1). TDM requires inner-dim stride 1,
    # so we describe phi with the physical layout (N, n*C) and load (N, BLOCK_C)
    # tiles. Reading from shared for the dot uses ``permute([1, 0])`` to deliver
    # the (BLOCK_C, N) operand-B view.
    phi_smem = gl.allocate_shared_memory(phi_dtype, [n, N_TOTAL_POW2, BLOCK_C], SH)

    # ---- TDM tensor descriptors ----
    pm_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=post_mix_ptr,
        shape=[M, n],
        strides=[stride_post_m, stride_post_n],
        block_shape=[BLOCK_M, n],
        layout=SH,
    )
    cm_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=comb_mix_ptr,
        shape=[M, n * n],
        strides=[stride_comb_m, stride_comb_dst],
        block_shape=[BLOCK_M, n * n],
        layout=SH,
    )
    x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=layer_input_ptr,
        shape=[M, C],
        strides=[stride_x_m, stride_x_c],
        block_shape=[BLOCK_M, BLOCK_C],
        layout=SH,
    )
    resin_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=residual_in_ptr,
        shape=[M, n * C],
        strides=[stride_resin_m, stride_resin_c],
        block_shape=[BLOCK_M, BLOCK_C],
        layout=SH,
    )
    phi_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=phi_ptr,
        shape=[N_TOTAL_POW2, n * C],
        strides=[stride_phi_n, stride_phi_k],
        block_shape=[N_TOTAL_POW2, BLOCK_C],
        layout=SH,
    )

    # ---- Issue all async loads (prologue) ----
    gl.amd.gfx1250.tdm.async_load(pm_desc, [m_base, 0], pm_smem)
    gl.amd.gfx1250.tdm.async_load(cm_desc, [m_base, 0], cm_smem)
    gl.amd.gfx1250.tdm.async_load(x_desc, [m_base, c_base], x_smem)
    for h in tl.static_range(n):
        gl.amd.gfx1250.tdm.async_load(
            resin_desc, [m_base, h * C + c_base], res_smem.index(h)
        )
        gl.amd.gfx1250.tdm.async_load(phi_desc, [0, h * C + c_base], phi_smem.index(h))

    gl.amd.gfx1250.tdm.async_wait(0)

    # ---- Local loads for the big tiles. ----
    x_tile = x_smem.load(layout_mc)
    x_fp32 = x_tile.to(gl.float32)

    # Masks for residual_out / acc stores.
    rm_m = gl.arange(0, BLOCK_M, layout=layout_m)
    rm_m_eff = m_base + rm_m
    m_mask_m = rm_m_eff < M
    rc_mc = gl.arange(0, BLOCK_C, layout=gl.SliceLayout(0, layout_mc))
    rc_mc_eff = c_base + rc_mc
    c_mask_mc = rc_mc_eff < C

    # Accumulators for the next-pre GEMM and sqrsum.
    acc_gemm = gl.zeros([BLOCK_M, N_TOTAL_POW2], dtype=gl.float32, layout=mfma_layout)
    acc_sq = gl.zeros([BLOCK_M], dtype=gl.float32, layout=layout_m)

    # ---- Per-stream pass: post-step mix + per-stream WMMA contribution. ----
    # Each per-h scalar is extracted by slicing the 2D smem tile to a
    # (BLOCK_M, 1) sub-descriptor, loading it with ``layout_m1``, and squeezing
    # the singleton col via ``gl.sum(..., axis=1)``. Generic over any ``n``.
    for h_dst in tl.static_range(n):
        # post[:, h_dst] : 1D (BLOCK_M,)
        pm_h_2d = pm_smem.slice(h_dst, 1, 1).load(layout_m1)
        pm_h = gl.sum(pm_h_2d, axis=1)
        pm_h_m = gl.convert_layout(pm_h, layout_m)

        out_h = pm_h_m[:, None] * x_fp32

        for h_src in tl.static_range(n):
            # comb[:, h_src, h_dst] : col index = h_src * n + h_dst (h_dst fast)
            cm_h_2d = cm_smem.slice(h_src * n + h_dst, 1, 1).load(layout_m1)
            cm_h = gl.sum(cm_h_2d, axis=1)
            cm_h_m = gl.convert_layout(cm_h, layout_m)

            res_h_src = res_smem.index(h_src).load(layout_mc).to(gl.float32)
            out_h += cm_h_m[:, None] * res_h_src

        out_h_dtype = out_h.to(out_dtype)

        # Store residual_out[:, h_dst, :].
        out_offsets = (
            rm_m_eff[:, None] * stride_resout_m + rc_mc_eff[None, :] * stride_resout_c
        )
        gl.store(
            residual_out_ptr + h_dst * stride_resout_n + out_offsets,
            out_h_dtype,
            mask=m_mask_m[:, None] & c_mask_mc[None, :],
            cache_modifier=".cs",
        )

        # WMMA contribution. phi shared is (N, BLOCK_C); permute to deliver
        # the (BLOCK_C, N) operand-B view in dot_b_layout.
        out_h_dot = gl.convert_layout(out_h_dtype, dot_a_layout)
        phi_h_dot = phi_smem.index(h_dst).permute([1, 0]).load(dot_b_layout)
        acc_gemm = gl.amd.gfx1250.wmma(out_h_dot, phi_h_dot, acc_gemm)

        # acc_sq accumulation.
        acc_sq += gl.sum(out_h * out_h, axis=1)

    # ---- Store partials ----
    rn = gl.arange(0, N_TOTAL_POW2, layout=gl.SliceLayout(0, mfma_layout))
    rm_mma = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout))
    rm_mma_eff = m_base + rm_mma
    m_mask_mma = rm_mma_eff < M
    n_mask = rn < N

    acc_offsets = (
        pid_c * stride_acc_k
        + rm_mma_eff[:, None] * stride_acc_m
        + rn[None, :] * stride_acc_n
    )
    gl.store(
        acc_ptr + acc_offsets,
        acc_gemm,
        mask=m_mask_mma[:, None] & n_mask[None, :],
    )

    gl.store(
        acc_sq_ptr + pid_c * stride_acc_sq_k + rm_m_eff * stride_acc_sq_m,
        acc_sq,
        mask=m_mask_m,
    )

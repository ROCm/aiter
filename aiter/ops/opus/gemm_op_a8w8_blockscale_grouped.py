# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FP8 blockscale grouped GEMMs with the DeepGEMM interfaces (gfx950 OPUS MXFP8 1x128 4-wave kids).

    m_grouped_fp8_blockscale_gemm_nt_contiguous(a, a_scale, b, b_scale, d, grouped_layout)
        a [M, K], a_scale [M, K/128], b [G, N, K], b_scale [G, N/128, K/128], d [M, N]
    m_grouped_fp8_blockscale_gemm_nn_contiguous(a, a_scale, b, b_scale, d, grouped_layout)
        a [M, K], a_scale [M, K/128], b [G, K, N], b_scale [G, K/128, N/128], d [M, N]
    m_grouped_fp8_blockscale_gemm_nt_masked(a, a_scale, b, b_scale, d, masked_m, expected_m)
        a [G, max_m, K], a_scale [G, max_m, K/128], b [G, N, K], b_scale [G, N/128, K/128]
    k_grouped_fp8_blockscale_gemm_tn_contiguous(a, a_scale, b, b_scale, d, grouped_layout, c=None)
        a [R, M], a_scale [R/128, M], b [R, N], b_scale [R/128, N], d [G, M, N]

FP8 e4m3 a / b, E8M0 scales (uint8 or float8_e8m0fnu), nothing shuffled; d written in place.
M-grouped: a_scale 1x128 (storage [K/128, M]), b_scale 128x128; K-grouped: scales 128x1 along the rows.
M-grouped grouped_layout: int32 [M], group of each row (-1 padding, groups 128-aligned); masked_m: int32 [G].
K-grouped grouped_layout: int32 [G], rows of each group.
"""

import functools

import torch

from csrc.opus_gemm.opus_gemm_a8w8_blockscale_grouped_common import (
    a8w8_blockscale_k_grouped_kids,
    a8w8_blockscale_m_grouped_kids,
)

from ...jit.core import compile_ops
from ._arch import _device_arch_and_cu

_GROUP_ALIGN = 128  # OPUS_A8W8_BS_GROUP_ALIGN
# Split a group into segments of <= 1 / _SEG_SHARE of a CU's share of the work, at least _SEG_MIN rows.
_SEG_SHARE = 2
_SEG_MIN = 8192


@functools.cache
def _k_grouped_seg_plan(
    b_m: int,
    b_n: int,
    cu_num: int,
    rows: int,
    groups: int,
    m: int,
    n: int,
    ld_bytes: int,
    f32: bool,
    tail_rows: int = 64,
) -> tuple[int, int]:
    """(seg_len, workspace int32 entries) of one launch; seg_len 0: no split. ld_bytes: row pitch."""
    if not f32:  # bf16 d: no read-modify-write, so no split
        return 0, 4 * groups
    tiles = -(-m // b_m) * -(-n // b_n)
    align = _GROUP_ALIGN
    # 32-bit byte offsets within one segment of a / b (the launcher checks the same bound)
    cap = ((2**31 - 1) // ld_bytes - tail_rows) // align * align
    seg = min(
        max(rows * tiles // (cu_num * _SEG_SHARE) // align * align, _SEG_MIN), cap
    )
    if rows <= seg:
        return 0, 4 * groups
    return seg, 4 * (rows // seg + groups) + groups * tiles


def _gen_opus_gemm_a8w8_blockscale_grouped_launch_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    group: torch.Tensor,
    workspace: torch.Tensor,
    kid: int,
    masked: bool,
) -> torch.Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a8w8_blockscale_grouped_launch",
    gen_fake=_gen_opus_gemm_a8w8_blockscale_grouped_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a8w8_blockscale_grouped_launch_raw(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    group: torch.Tensor,
    workspace: torch.Tensor,
    kid: int,
    masked: bool,
) -> torch.Tensor: ...


def _kid_for(n: int) -> int:
    """M-grouped kid for output width n: B_N 256 when it divides n, else 128."""
    b_n = 256 if n % 256 == 0 else 128
    kid = next(
        (
            kid
            for kid, k in sorted(a8w8_blockscale_m_grouped_kids.items())
            if k.B_N == b_n
        ),
        None,
    )
    if kid is None:
        raise RuntimeError(
            f"no M-grouped a8w8 blockscale kid with B_N {b_n}: build its images with "
            "csrc/opus_gemm/gen_co/build_co.py --arch gfx950 --family a8w8_blockscale_grouped"
        )
    return kid


def _launch(a, a_scale, b, b_scale, d, group, kid, masked):
    # contiguous: the plan kernel's rows / first row per group (int32 [2 G])
    workspace = torch.empty(
        0 if masked else 2 * b.shape[0], dtype=torch.int32, device=a.device
    )
    with torch.cuda.device(a.device):
        _opus_gemm_a8w8_blockscale_grouped_launch_raw(
            a, b, d, a_scale, b_scale, group, workspace, kid, masked
        )


def m_grouped_fp8_blockscale_gemm_nt_contiguous(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    kid: int | None = None,
) -> None:
    """d[rows of group g] = (a * a_scale)[rows of g] @ (b[g] * b_scale[g]).T; b [G, N, K]."""
    if kid is None:
        kid = _kid_for(b.shape[1])
    _launch(a, a_scale, b, b_scale, d, grouped_layout, kid, False)


def m_grouped_fp8_blockscale_gemm_nn_contiguous(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    kid: int | None = None,
) -> None:
    """d[rows of group g] = (a * a_scale)[rows of g] @ (b[g] * b_scale[g]); b [G, K, N],
    b_scale [G, K/128, N/128] (the weight of the NT GEMM and its scale, as stored)."""
    if kid is None:
        kid = _kid_for(b.shape[2])
    _launch(
        a,
        a_scale,
        b.transpose(1, 2),
        b_scale.transpose(1, 2),
        d,
        grouped_layout,
        kid,
        False,
    )


def m_grouped_fp8_blockscale_gemm_nt_masked(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    kid: int | None = None,
) -> None:
    """d[g, :masked_m[g]] = (a * a_scale)[g, :masked_m[g]] @ (b[g] * b_scale[g]).T.
    expected_m: kept for the DeepGEMM signature (the tiles follow masked_m)."""
    if kid is None:
        kid = _kid_for(b.shape[1])
    _launch(a, a_scale, b, b_scale, d, masked_m, kid, True)


# ---- K-grouped (weight gradient) -------------------------------------------


def _gen_opus_gemm_a8w8_blockscale_k_grouped_launch_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    grouped_layout: torch.Tensor,
    C: torch.Tensor | None,
    workspace: torch.Tensor,
    kid: int,
    seg_len: int,
) -> torch.Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a8w8_blockscale_k_grouped_launch",
    gen_fake=_gen_opus_gemm_a8w8_blockscale_k_grouped_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a8w8_blockscale_k_grouped_launch_raw(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    grouped_layout: torch.Tensor,
    C: torch.Tensor | None,
    workspace: torch.Tensor,
    kid: int,
    seg_len: int,
) -> torch.Tensor: ...


def _k_kid() -> int:
    """The K-grouped kid (256 x 256 tiles)."""
    if not a8w8_blockscale_k_grouped_kids:
        raise RuntimeError(
            "no K-grouped a8w8 blockscale kid: build its images with "
            "csrc/opus_gemm/gen_co/build_co.py --arch gfx950 --family a8w8_blockscale_grouped"
        )
    return min(a8w8_blockscale_k_grouped_kids)


def k_grouped_fp8_blockscale_gemm_tn_contiguous(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    c: torch.Tensor | None = None,
    kid: int | None = None,
    seg_len: int | None = None,
) -> None:
    """d[g] = (c[g] +) (a * a_scale)[rows of g].T @ (b * b_scale)[rows of g]; a [R, M], b [R, N],
    d [G, M, N] (N % 256 == 0), the scales 128x1 along the rows: a_scale [R/128, M],
    b_scale [R/128, N].

    grouped_layout: int32 [G], rows of each group. c (fp32 d): source, or d itself (in-place +=).
    seg_len: split long groups (default from the shape); segments add in order (deterministic).
    """
    f32 = d.dtype == torch.float32
    if kid is None:
        kid = _k_kid()
    k = a8w8_blockscale_k_grouped_kids[kid]
    _, cu_num = _device_arch_and_cu(a.device)
    rows, groups = a.shape[0], d.shape[0]
    ld = max(a.stride(0), b.stride(0))
    plan_seg, ws_numel = _k_grouped_seg_plan(
        k.B_M, k.B_N, cu_num, rows, groups, d.shape[1], d.shape[2], ld, f32, 128
    )
    if seg_len is None:
        seg_len = plan_seg
    else:
        tiles = -(-d.shape[1] // k.B_M) * (d.shape[2] // k.B_N)
        full = rows // seg_len if seg_len > 0 else 0
        ws_numel = 4 * (full + groups) + (groups * tiles if full else 0)
    workspace = torch.empty(ws_numel, dtype=torch.int32, device=a.device)
    with torch.cuda.device(a.device):
        _opus_gemm_a8w8_blockscale_k_grouped_launch_raw(
            a, b, d, a_scale, b_scale, grouped_layout, c, workspace, kid, seg_len
        )

# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 grouped GEMMs with the DeepGEMM interfaces (gfx950 OPUS 4-wave grouped kids).

    m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout)  a [M, K], b [G, N, K], d [M, N]
    m_grouped_bf16_gemm_nn_contiguous(a, b, d, grouped_layout)  a [M, K], b [G, K, N], d [M, N]
    m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m)
        a [G, max_m, K], b [G, N, K], d [G, max_m, N]
    k_grouped_bf16_gemm_tn_contiguous(a, b, d, grouped_layout, c=None)
        a [R, M], b [R, N], d [G, M, N]

M-grouped grouped_layout: int32 [M], group of each row (-1 padding, groups 128-aligned).
K-grouped grouped_layout: int32 [G], rows of each group; group g starts at the row where the
groups before it end, each rounded up to 128 rows (the last one needs no padding). d is written
in place.
"""

import functools

import torch

from csrc.opus_gemm.opus_gemm_grouped_common import (
    a16w16_k_grouped_kids,
    a16w16_m_grouped_kids,
)

from ...jit.core import compile_ops
from ._arch import _device_arch_and_cu

_GROUP_ALIGN = 128  # OPUS_A16W16_GROUP_ALIGN


def _gen_opus_gemm_a16w16_grouped_launch_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    group: torch.Tensor,
    workspace: torch.Tensor,
    kid: int,
    masked: bool,
) -> torch.Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a16w16_grouped_launch",
    gen_fake=_gen_opus_gemm_a16w16_grouped_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a16w16_grouped_launch_raw(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    group: torch.Tensor,
    workspace: torch.Tensor,
    kid: int,
    masked: bool,
) -> torch.Tensor: ...


def get_mk_alignment_for_contiguous_layout() -> int:
    """Rows every group of a contiguous grouped_layout is padded to."""
    return _GROUP_ALIGN


def _kid_for(n: int) -> int:
    """M-grouped kid for output width n: B_N 256 (128 when n < 256); the kernel picks
    256 / 192 / 128-row tiles per group from its rows (no host-side size hint needed).
    """
    b_n = 256 if n >= 256 else 128
    kid = next(
        (
            kid
            for kid, k in sorted(a16w16_m_grouped_kids.items())
            if k.kind == "m_grouped" and k.B_N == b_n
        ),
        None,
    )
    if kid is None:
        raise RuntimeError(
            f"no M-grouped a16w16 kid with B_N {b_n}: build its images with "
            "csrc/opus_gemm/gen_co/build_co.py --arch gfx950 --family a16w16_grouped"
        )
    return kid


def _launch(a, b, d, group, kid, masked):
    # contiguous: the plan kernel's rows / first row per group (int32 [2 G])
    workspace = torch.empty(
        0 if masked else 2 * b.shape[0], dtype=torch.int32, device=a.device
    )
    with torch.cuda.device(a.device):
        _opus_gemm_a16w16_grouped_launch_raw(a, b, d, group, workspace, kid, masked)


def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    kid: int | None = None,
) -> None:
    """d[rows of group g] = a[rows of group g] @ b[g].T; b [G, N, K]."""
    if kid is None:
        kid = _kid_for(b.shape[1])
    _launch(a, b, d, grouped_layout, kid, False)


def m_grouped_bf16_gemm_nn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    kid: int | None = None,
) -> None:
    """d[rows of group g] = a[rows of group g] @ b[g]; b [G, K, N]."""
    if kid is None:
        kid = _kid_for(b.shape[2])
    _launch(a, b.transpose(1, 2), d, grouped_layout, kid, False)


def m_grouped_bf16_gemm_nt_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    kid: int | None = None,
) -> None:
    """d[g, :masked_m[g]] = a[g, :masked_m[g]] @ b[g].T. expected_m: kept for the DeepGEMM
    signature, the kernel sizes its tiles per group from masked_m."""
    if kid is None:
        kid = _kid_for(b.shape[1])
    _launch(a, b, d, masked_m, kid, True)


# ---- K-grouped (weight gradient) -------------------------------------------


def _gen_opus_gemm_a16w16_k_grouped_launch_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    grouped_layout: torch.Tensor,
    C: torch.Tensor | None,
    workspace: torch.Tensor,
    kid: int,
    seg_len: int,
) -> torch.Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a16w16_k_grouped_launch",
    gen_fake=_gen_opus_gemm_a16w16_k_grouped_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a16w16_k_grouped_launch_raw(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    grouped_layout: torch.Tensor,
    C: torch.Tensor | None,
    workspace: torch.Tensor,
    kid: int,
    seg_len: int,
) -> torch.Tensor: ...


# Split a group into segments of <= 1 / _SEG_SHARE of a CU's share of the work, at least _SEG_MIN rows.
_SEG_SHARE = 2
_SEG_MIN = 8192


def _k_kid_for(m: int, n: int) -> int:
    """K-grouped kid: a 256 tile edge where the output is at least that wide."""
    tile = (256 if m >= 256 else 128, 256 if n >= 256 else 128)
    kid = next(
        (
            kid
            for kid, k in sorted(a16w16_k_grouped_kids.items())
            if (k.B_M, k.B_N) == tile
        ),
        None,
    )
    if kid is None:
        raise RuntimeError(
            f"no K-grouped a16w16 kid with tile {tile[0]}x{tile[1]}: build its images with "
            "csrc/opus_gemm/gen_co/build_co.py --arch gfx950 --family a16w16_grouped"
        )
    return kid


@functools.cache
def _k_grouped_plan(
    kid: int, cu_num: int, rows: int, groups: int, m: int, n: int, ld: int, f32: bool
) -> tuple[int, int]:
    """(seg_len, workspace int32 entries) of one launch; seg_len 0: no group is split."""
    if not f32:  # bf16 d: no read-modify-write, so no split
        return 0, 4 * groups
    k = a16w16_k_grouped_kids[kid]
    tiles = -(-m // k.B_M) * -(-n // k.B_N)
    align = _GROUP_ALIGN
    # 32-bit byte offsets within one segment of a / b (the launcher checks the same bound)
    cap = ((2**31 - 1) // (2 * ld) - 64) // align * align
    seg = min(
        max(rows * tiles // (cu_num * _SEG_SHARE) // align * align, _SEG_MIN), cap
    )
    if rows <= seg:
        return 0, 4 * groups
    return seg, 4 * (rows // seg + groups) + groups * tiles


def _check_k_grouped_layout(grouped_layout: torch.Tensor, rows: int) -> None:
    """Reject a K-grouped layout whose groups, placed as the plan kernel places them (group g at
    the sum of the groups before it, each rounded up to _GROUP_ALIGN rows), run past the rows of
    a / b: the kernel would cut such a group short. Reads grouped_layout on the host."""
    if grouped_layout.dim() != 1 or grouped_layout.is_floating_point():
        return  # the launcher rejects it
    # the kernel reads < 0 as 0
    sizes = grouped_layout.to("cpu", torch.int64).clamp(min=0)
    padded = (sizes + _GROUP_ALIGN - 1) // _GROUP_ALIGN * _GROUP_ALIGN
    ends = padded.cumsum(0) - padded + sizes
    bad = ((sizes > 0) & (ends > rows)).nonzero()
    if bad.numel():
        g = int(bad[0])
        raise ValueError(
            f"k_grouped grouped_layout: group {g} ({int(sizes[g])} rows) ends at row "
            f"{int(ends[g])} of {rows}; every group must start at a multiple of "
            f"{_GROUP_ALIGN} rows (pad the groups before it to {_GROUP_ALIGN} rows)"
        )


def k_grouped_bf16_gemm_tn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    c: torch.Tensor | None = None,
    kid: int | None = None,
    seg_len: int | None = None,
    check_layout: bool = False,
) -> None:
    """d[g] = (c[g] +) a[rows of g].T @ b[rows of g]; a [R, M], b [R, N], d [G, M, N].

    grouped_layout: int32 [G], rows of each group. The groups are not packed: group g starts at
    the row where the groups before it end, each rounded up to
    get_mk_alignment_for_contiguous_layout() (128) rows; the last group needs no padding.
    c (fp32 d): source, or d itself (in-place +=).
    seg_len: split long groups (default from the shape); segments add in order (deterministic).
    check_layout: True raises ValueError for a layout that runs past the rows of a / b (e.g.
    packed groups). It reads grouped_layout on the host, which waits for the device, so it is off
    by default.
    """
    if check_layout:
        _check_k_grouped_layout(grouped_layout, a.shape[0])
    f32 = d.dtype == torch.float32
    if kid is None:
        kid = _k_kid_for(d.shape[1], d.shape[2])
    _, cu_num = _device_arch_and_cu(a.device)
    rows, groups = a.shape[0], d.shape[0]
    ld = max(a.stride(0), b.stride(0)) if a.dim() == b.dim() == 2 else 1
    plan_seg, ws_numel = _k_grouped_plan(
        kid, cu_num, rows, groups, d.shape[1], d.shape[2], ld, f32
    )
    if seg_len is None:
        seg_len = plan_seg
    else:
        k = a16w16_k_grouped_kids[kid]
        tiles = -(-d.shape[1] // k.B_M) * -(-d.shape[2] // k.B_N)
        full = rows // seg_len if seg_len > 0 else 0
        ws_numel = 4 * (full + groups) + (groups * tiles if full else 0)
    workspace = torch.empty(ws_numel, dtype=torch.int32, device=a.device)
    with torch.cuda.device(a.device):
        _opus_gemm_a16w16_k_grouped_launch_raw(
            a, b, d, grouped_layout, c, workspace, kid, seg_len
        )

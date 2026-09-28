# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused EP expert LUT, optional counter reset and num_valid_routes = nvt * topk."""

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.g2l_lut import _g2l_lut_kernel

MAX_G2L_EXPERTS = 16384


def build_g2l_lut(
    expert_mask: torch.Tensor,
    E: int,
    nvt: torch.Tensor | None = None,
    topk: int | None = None,
    *,
    counter: torch.Tensor | None = None,
    clear_counter: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return ``(lut, counter, nvr)`` using exactly one GPU kernel.

    ``expert_mask`` is a 1-D real/bool GPU tensor (strided is supported).
    ``lut[i]`` is the count of preceding enabled experts when mask[i] != 0,
    otherwise sentinel E. The caller owns the mask/E consistency invariant.

    When both nvt and topk are supplied, nvt is a nonempty int32/int64 tensor
    on the same device; its FIRST element times topk is returned as int32 nvr.
    nvt is read dynamically and the valid route count must fit signed int32.
    Missing either argument leaves nvr=None, matching the grouped helper API.

    A supplied counter must be contiguous int32 (E,) on the same device. With
    clear_counter=False it is required and left UNCHANGED: an earlier dispatch
    must reset it before routing. No implicit zero/fill kernel is launched.
    Callers must not share a counter across overlapping calls/streams.

    Allocations use empty; no cast, copy, fill or CPU readback is required.
    Compile/warm up before graph capture, as with other Triton kernels.
    """
    if not expert_mask.is_cuda or expert_mask.ndim != 1 or expert_mask.is_complex():
        raise ValueError("expert_mask must be a 1-D real/bool GPU tensor")
    n = expert_mask.numel()
    if not isinstance(E, int) or E < 0 or max(n, E) > MAX_G2L_EXPERTS:
        raise ValueError(f"require 0 <= E and max(N, E) <= {MAX_G2L_EXPERTS}")
    device = expert_mask.device
    write_nvr = nvt is not None and topk is not None
    if write_nvr:
        if (
            nvt.device != device
            or nvt.numel() < 1
            or nvt.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("nvt must be a nonempty int32/int64 tensor on mask.device")
        if not isinstance(topk, int) or not 0 <= topk <= 2**31 - 1:
            raise ValueError("topk must be a nonnegative int32 Python int")
    if counter is None:
        if not clear_counter:
            raise ValueError("clear_counter=False requires a preallocated counter")
        counter = torch.empty(E, dtype=torch.int32, device=device)
    elif (
        counter.device != device
        or counter.dtype != torch.int32
        or counter.shape != (E,)
        or not counter.is_contiguous()
    ):
        raise ValueError("counter must be contiguous int32 (E,) on mask.device")

    lut = torch.empty(n, dtype=torch.int32, device=device)
    nvr = torch.empty(1, dtype=torch.int32, device=device) if write_nvr else None
    block = triton.next_power_of_2(max(n, E if clear_counter else 0, 1))
    with torch.cuda.device(device):
        _g2l_lut_kernel[(1,)](
            expert_mask,
            nvt if write_nvr else None,
            lut,
            counter,
            nvr,
            n,
            E,
            int(topk) if write_nvr else 0,
            STRIDE=expert_mask.stride(0),
            CLEAR_COUNTER=clear_counter,
            WRITE_NVR=write_nvr,
            BLOCK=block,
            num_warps=4,
        )
    return lut, counter, nvr

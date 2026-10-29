# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass, fields

import pytest
import torch

# the bf16 reference is shared with the a16w4 family
from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_torch
from aiter.ops.triton.moe.moe_op_gemm_a16w16 import moe_gemm_a16w16
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.utils._triton import arch_info
from op_tests.triton_tests.moe.moe_test_utils import assert_close

_SUPPORTED_ARCHS = ("gfx1250",)


def init_routing_data(m, n_expts_tot, n_expts_act, do_gather, do_scatter, device):
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device)
    routing_data, gather_idx, scatter_idx = routing(logits, n_expts_act)
    routing_data.gate_scal = None
    return (
        routing_data,
        gather_idx if do_gather else None,
        scatter_idx if do_scatter else None,
    )


def init_compute_data(m, n, k, n_expts_tot, n_expts_act, gindx, has_y_gammas, device):
    in_m = m * (n_expts_act if gindx is None else 1)
    x = torch.randn((in_m, k), dtype=torch.bfloat16, device=device)
    # (E, K, N) row-major: the N-contiguous layout the kernel requires
    w = torch.randn((n_expts_tot, k, n), dtype=torch.bfloat16, device=device)
    bias = torch.randn((n_expts_tot, n), dtype=torch.float32, device=device)
    if has_y_gammas:
        gammas = 2 ** torch.randint(
            -5, 0, (m * n_expts_act,), device=device, dtype=torch.float32
        )
    else:
        gammas = None
    return x, w, bias, gammas


@dataclass
class Case:
    m: int
    n: int
    k: int
    n_expts_tot: int = 1
    n_expts_act: int = 1


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case))
        for case in [
            Case(4, 128, 128, 2, 1),
            Case(16, 128, 128, 2, 1),
            Case(16, 256, 256, 128, 4),
            Case(64, 256, 256, 8, 2),
            Case(300, 400, 768, 8, 4),
            Case(1000, 704, 768, 8, 2),
            Case(1024, 512, 512, 8, 2),
            Case(1024, 3072, 512, 128, 4),
            Case(4096, 256, 256, 128, 4),
            # K not a multiple of block_k: exercises the clamped-bounds tail
            Case(64, 256, 320, 8, 2),
            Case(300, 256, 400, 8, 4),
        ]
    ],
)
@pytest.mark.parametrize("do_gather", [False, True])
@pytest.mark.parametrize("do_scatter", [False, True])
@pytest.mark.parametrize("has_y_gammas", [False, True])
@pytest.mark.parametrize("apply_swiglu", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
def test_op(
    m,
    n,
    k,
    n_expts_tot,
    n_expts_act,
    do_gather,
    do_scatter,
    has_y_gammas,
    apply_swiglu,
    has_bias,
    device="cuda",
):
    if arch_info.get_arch() not in _SUPPORTED_ARCHS:
        pytest.skip(f"moe_gemm_a16w16 is only implemented for {_SUPPORTED_ARCHS}")

    torch.manual_seed(0)

    rdata, gindx, sindx = init_routing_data(
        m, n_expts_tot, n_expts_act, do_gather, do_scatter, device
    )
    x, w, bias, gammas = init_compute_data(
        m, n, k, n_expts_tot, n_expts_act, gindx, has_y_gammas, device
    )
    if not has_bias:
        bias = None

    ref_y = moe_gemm_torch(
        x, w, bias, rdata, gindx, sindx, gammas, apply_swiglu, alpha=1.0, limit=1.0
    )
    tri_y = moe_gemm_a16w16(
        x,
        w,
        bias=bias,
        routing_data=rdata,
        gather_indx=gindx,
        scatter_indx=sindx,
        gammas=gammas,
        apply_swiglu=apply_swiglu,
        alpha=1.0,
        limit=1.0,
    )

    assert_close(ref_y, tri_y, description=f"a16w16 m={m} n={n} k={k}")


def test_rejects_non_contiguous_weights(device="cuda"):
    """`w` must be N-contiguous; a K-contiguous view must be rejected, not silently wrong."""
    if arch_info.get_arch() not in _SUPPORTED_ARCHS:
        pytest.skip(f"moe_gemm_a16w16 is only implemented for {_SUPPORTED_ARCHS}")

    m, n, k, n_expts_tot, n_expts_act = 16, 128, 128, 2, 1
    rdata, gindx, sindx = init_routing_data(
        m, n_expts_tot, n_expts_act, False, False, device
    )
    x, w, _, _ = init_compute_data(
        m, n, k, n_expts_tot, n_expts_act, gindx, False, device
    )
    w_bad = w.transpose(1, 2).contiguous().transpose(1, 2)  # K-contiguous

    with pytest.raises(AssertionError, match="N-contiguous"):
        moe_gemm_a16w16(
            x, w_bad, routing_data=rdata, gather_indx=gindx, scatter_indx=sindx
        )

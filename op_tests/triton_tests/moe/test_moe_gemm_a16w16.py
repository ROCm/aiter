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


def init_compute_data(
    m, n, k, n_expts_tot, n_expts_act, gindx, has_y_gammas, device, has_bias=True
):
    in_m = m * (n_expts_act if gindx is None else 1)
    x = torch.randn((in_m, k), dtype=torch.bfloat16, device=device)
    # (E, K, N) row-major: the N-contiguous layout the kernel requires
    w = torch.randn((n_expts_tot, k, n), dtype=torch.bfloat16, device=device)
    # bias=None selects TDM_BIAS_WAIT=0, which changes the drain-loop wait
    bias = (
        torch.randn((n_expts_tot, n), dtype=torch.float32, device=device)
        if has_bias
        else None
    )
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
    has_bias: bool = True
    # The store uses Y's element type. A hard-coded bf16 store fails to compile.
    out_dtype: torch.dtype = torch.bfloat16


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case))
        for case in [
            # routing picks block_m from m * n_expts_act // n_expts_tot, so the
            # cases below cover every block_m it can produce: 16, 32, 64 and 128.
            Case(4, 128, 128, 2, 1),  # block_m 16
            Case(16, 128, 128, 2, 1),  # block_m 16
            Case(16, 256, 256, 128, 4),  # block_m 16
            Case(64, 256, 256, 8, 2),  # block_m 16
            Case(1024, 3072, 512, 128, 4),  # block_m 32
            Case(256, 512, 512, 8, 2),  # block_m 64
            Case(512, 256, 256, 8, 1),  # block_m 64
            Case(300, 400, 768, 8, 4),  # block_m 128
            Case(1000, 704, 768, 8, 2),  # block_m 128
            Case(1024, 512, 512, 8, 2),  # block_m 128
            Case(4096, 256, 256, 128, 4),  # block_m 128
            # K not a multiple of block_k: exercises the clamped-bounds tail
            Case(64, 256, 320, 8, 2),
            Case(300, 256, 400, 8, 4),
            Case(256, 256, 320, 8, 2),  # block_m 64, ragged K
            # bias=None, over block_m 16 and block_m 64
            Case(16, 128, 128, 2, 1, has_bias=False),
            Case(256, 512, 512, 8, 2, has_bias=False),
            # fp16 output, with and without bias
            Case(256, 512, 512, 8, 2, out_dtype=torch.float16),
            Case(64, 256, 256, 8, 2, has_bias=False, out_dtype=torch.float16),
        ]
    ],
)
@pytest.mark.parametrize(
    "do_gather, do_scatter",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
@pytest.mark.parametrize("has_y_gammas", [False, True])
@pytest.mark.parametrize("apply_swiglu", [False, True])
def test_op(
    m,
    n,
    k,
    n_expts_tot,
    n_expts_act,
    has_bias,
    out_dtype,
    do_gather,
    do_scatter,
    has_y_gammas,
    apply_swiglu,
    device="cuda",
):
    if arch_info.get_arch() not in _SUPPORTED_ARCHS:
        pytest.skip(f"moe_gemm_a16w16 is only implemented for {_SUPPORTED_ARCHS}")

    torch.manual_seed(0)

    rdata, gindx, sindx = init_routing_data(
        m, n_expts_tot, n_expts_act, do_gather, do_scatter, device
    )
    x, w, bias, gammas = init_compute_data(
        m, n, k, n_expts_tot, n_expts_act, gindx, has_y_gammas, device, has_bias
    )

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
        out_dtype=out_dtype,
    )

    assert tri_y.dtype == out_dtype
    # The reference is bf16. An fp16 result keeps more mantissa bits, so the
    # two do not round together and the largest element error is wider.
    maxtol = 3e-2 if out_dtype == torch.float16 else None
    assert_close(
        ref_y.to(out_dtype),
        tri_y,
        maxtol=maxtol,
        description=f"a16w16 m={m} n={n} k={k} bias={has_bias} out={out_dtype}",
    )

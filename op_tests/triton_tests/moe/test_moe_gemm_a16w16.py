import os
from dataclasses import dataclass, fields

import pytest
import torch

from aiter.ops.triton.moe.moe_op_gemm_a16w16 import (
    moe_gemm_a16w16,
    moe_gemm_torch,
)
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.utils._triton import arch_info
from op_tests.triton_tests.moe.moe_test_utils import assert_close


def init_routing_data(m, n_expts_tot, n_expts_act, do_gather, do_scatter, device="cuda"):
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device)
    routing_data, gather_idx, scatter_idx = routing(logits, n_expts_act)
    routing_data.gate_scal = None
    gather_idx = gather_idx if do_gather else None
    scatter_idx = scatter_idx if do_scatter else None
    return m, routing_data, gather_idx, scatter_idx


def init_compute_data(m, n, k, gindx, sindx, n_expts_tot, n_expts_act, dtype, has_y_gammas, device="cuda"):
    torch.manual_seed(0)
    in_m = m * (n_expts_act if gindx is None else 1)
    x = torch.randn((in_m, k), device=device, dtype=dtype)
    w = torch.randn((n_expts_tot, k, n), device=device, dtype=dtype)
    bias = torch.randn((n_expts_tot, n), device=device, dtype=dtype)
    if has_y_gammas:
        gamma = 2 ** torch.randint(
            -5, 0, (m * n_expts_act,), device=device, dtype=torch.float32
        )
    else:
        gamma = None
    return x, w, bias, gamma


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
            Case(4, 4, 8, 2, 1),
            Case(4, 4, 8, 8, 2),
            Case(4, 4, 8, 128, 4),
            Case(4, 32, 64, 128, 4),
            Case(4, 128, 128, 128, 4),
            Case(32, 256, 128, 128, 4),
            Case(16, 256, 256, 128, 4),
            Case(16, 128, 128, 2, 1),
            Case(32, 512, 512, 8, 2),
            Case(64, 512, 256, 128, 4),
            Case(128, 1024, 512, 128, 4),
            Case(300, 400, 800, 8, 4),
            Case(1024, 256, 256, 128, 4),
            Case(4096, 256, 256, 128, 4),
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
@pytest.mark.parametrize("backend", [None, "triton"])
def test_op(
    m,
    n,
    k,
    do_gather,
    do_scatter,
    has_y_gammas,
    apply_swiglu,
    n_expts_tot,
    n_expts_act,
    backend,
    device="cuda",
):
    if int(os.environ.get("AITER_IN_FFM_AM", "0")) == 1 and (
        m > 1024 or n > 1024 or k > 1024 or n_expts_tot > 128
    ):
        pytest.skip("Test will take too long on FFM")

    torch.manual_seed(0)
    dtype = torch.bfloat16

    m, rdata, gindx, sindx = init_routing_data(
        m, n_expts_tot, n_expts_act, do_gather, do_scatter, device=device
    )
    x, w, bias, gammas = init_compute_data(
        m, n, k, gindx, sindx, n_expts_tot, n_expts_act, dtype, has_y_gammas, device=device
    )

    ref_y = moe_gemm_torch(
        x.clone(), w.clone(), bias.clone(), rdata, gindx, sindx, gammas, apply_swiglu
    )

    tri_y = moe_gemm_a16w16(
        x,
        w,
        bias,
        rdata,
        gindx,
        sindx,
        gammas,
        out_dtype=dtype,
        apply_swiglu=apply_swiglu,
        backend=backend,
    )

    assert_close(ref_y, tri_y, maxtol=1e-1, rmstol=1e-2)


@pytest.mark.parametrize(
    "backend", ["triton"]
)
def test_no_bias(backend, device="cuda"):
    """Verify the kernel works when bias=None."""
    dtype = torch.bfloat16
    torch.manual_seed(42)
    m, rdata, gindx, sindx = init_routing_data(16, 4, 2, True, True, device=device)
    x, w, _, gammas = init_compute_data(m, 128, 128, gindx, sindx, 4, 2, dtype, False, device=device)

    ref_y = moe_gemm_torch(x.clone(), w.clone(), None, rdata, gindx, sindx, None, False)
    tri_y = moe_gemm_a16w16(x, w, None, rdata, gindx, sindx, None, out_dtype=dtype, backend=backend)
    assert_close(ref_y, tri_y, maxtol=1e-1, rmstol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

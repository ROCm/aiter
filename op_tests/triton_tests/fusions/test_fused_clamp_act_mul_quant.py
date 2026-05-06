import torch
import torch.nn.functional as F
import pytest

import aiter
from aiter.ops.triton.fusions.fused_clamp_act_mul_quant import (
    fused_clamp_act_mul_fp8_group_quant,
)
from op_tests.triton_tests.quant.test_fused_fp8_quant import (
    per_token_fp8_group_quant,
    upcast,
)


def _torch_reference(inp, swiglu_limit, weights, dtype_quant):
    gate, up = inp.chunk(2, dim=-1)
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    y = F.silu(gate) * up
    if weights is not None:
        y = weights * y
    return per_token_fp8_group_quant(y.float(), dtype_quant, 128)


@pytest.mark.parametrize("M", [1, 2, 4, 8, 32])
@pytest.mark.parametrize("D", [512])
@pytest.mark.parametrize("swiglu_limit", [0.0, 10.0])
@pytest.mark.parametrize("transpose_scale", [True, False])
@pytest.mark.parametrize(
    "with_weights,weight_broadcast",
    [(False, False), (True, True), (True, False)],
)
def test_fused_clamp_act_mul_quant(
    M, D, swiglu_limit, transpose_scale, with_weights, weight_broadcast
):
    torch.manual_seed(42)
    dtype_quant = aiter.dtypes.fp8
    N = D // 2
    with_weights = False
    if with_weights:
        if weight_broadcast:
            w = torch.randn(M, 1, device="cuda", dtype=torch.float32) * 0.5
        else:
            w = torch.randn(M, N, device="cuda", dtype=torch.float32) * 0.1
    else:
        w = None

    inp = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    out_fp8 = torch.empty((M, N), dtype=dtype_quant, device="cuda")
    if transpose_scale:
        scale = torch.empty(((N + 127) // 128), M, dtype=torch.float32, device="cuda")
    else:
        scale = torch.empty((M, (N + 127) // 128), dtype=torch.float32, device="cuda")

    out_fp8, scale = fused_clamp_act_mul_fp8_group_quant(
        inp,
        out_fp8,
        scale,
        swiglu_limit,
        weights=w,
        activation="silu",
        transpose_scale=transpose_scale,
    )

    ref_q, ref_s = _torch_reference(inp, swiglu_limit, w, dtype_quant)

    if transpose_scale:
        scale = scale.view(((N + 127) // 128), M).T.contiguous()
    out_triton = upcast(out_fp8, scale, torch.bfloat16)
    ref_triton = upcast(ref_q, ref_s, torch.bfloat16)

    torch.testing.assert_close(
        out_triton,
        ref_triton,
        atol=0.1,
        rtol=0.1,
    )

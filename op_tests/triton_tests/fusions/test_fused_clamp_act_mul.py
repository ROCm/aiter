import torch
import torch.nn.functional as F
import pytest

import aiter
from aiter.ops.triton.fusions.fused_clamp_act_mul import (
    fused_clamp_act_mul,
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
    if dtype_quant is None:
        return y.to(inp.dtype)
    return per_token_fp8_group_quant(y.float(), dtype_quant, 128)


@pytest.mark.parametrize("M", [1, 2, 4, 8, 32])
@pytest.mark.parametrize("D", [2048, 3072])
@pytest.mark.parametrize("swiglu_limit", [0.0, 7.0])
@pytest.mark.parametrize("transpose_scale", [True, False])
@pytest.mark.parametrize(
    "with_weights,weight_broadcast",
    [(False, False), (True, True), (True, False)],
)
@pytest.mark.parametrize("dtype_quant", [aiter.dtypes.fp8, None])
def test_fused_clamp_act_mul(
    M, D, swiglu_limit, transpose_scale, with_weights, weight_broadcast, dtype_quant
):
    torch.manual_seed(42)
    N = D // 2
    if with_weights:
        if weight_broadcast:
            w = torch.randn(M, 1, device="cuda", dtype=torch.float32) * 0.5
        else:
            w = torch.randn(M, N, device="cuda", dtype=torch.float32) * 0.1
    else:
        w = None

    inp = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)

    if dtype_quant is not None:
        out_buf = torch.empty((M, N), dtype=dtype_quant, device="cuda")
        if transpose_scale:
            scale = torch.empty(
                ((N + 127) // 128), M, dtype=torch.float32, device="cuda"
            )
        else:
            scale = torch.empty(
                (M, (N + 127) // 128), dtype=torch.float32, device="cuda"
            )

        out_q, scale = fused_clamp_act_mul(
            inp,
            out_buf,
            scale,
            swiglu_limit,
            weights=w,
            activation="silu",
            dtype_quant=dtype_quant,
            transpose_scale=transpose_scale,
        )

        ref_q, ref_s = _torch_reference(inp, swiglu_limit, w, dtype_quant)

        if transpose_scale:
            scale = scale.view(((N + 127) // 128), M).T.contiguous()
        out_triton = upcast(out_q, scale, torch.bfloat16)
        ref_triton = upcast(ref_q, ref_s, torch.bfloat16)

        torch.testing.assert_close(
            out_triton,
            ref_triton,
            atol=0.1,
            rtol=0.1,
        )
    else:
        # transpose_scale is irrelevant when not quantizing; skip the redundant
        # duplicate parametrization to keep the matrix small.
        if transpose_scale:
            pytest.skip("transpose_scale is only meaningful when dtype_quant is set")

        out = fused_clamp_act_mul(
            inp,
            swiglu_limit=swiglu_limit,
            weights=w,
            activation="silu",
            dtype_quant=None,
        )
        ref = _torch_reference(inp, swiglu_limit, w, None)

        assert out.dtype == inp.dtype
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

import torch
import torch.nn.functional as F
import pytest

import aiter
from aiter.ops.triton.fusions.fused_clamp_act_mul import (
    fused_clamp_act_mul,
)
from aiter.utility import fp4_utils
from op_tests.triton_tests.quant.test_fused_fp8_quant import (
    per_token_fp8_group_quant,
    upcast,
)
from aiter.ops.triton.utils.shuffle import unshuffle_scale_gemm


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


def _torch_reference_ue8m0(inp, swiglu_limit, weights, dtype_quant, quant_block_size):
    """Bit-exact torch model of the kernel's ue8m0 path: the exp2-based SiLU
    (matching ``_silu_exp2``) in fp32 followed by per-group MXFP8 quant with
    round-up e8m0 scales. Returns ``(out_q, unshuffled_scale)``."""
    gate, up = inp.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    y = (gate / (1.0 + torch.exp2(-(gate * 1.44269504089)))) * up
    if weights is not None:
        y = weights * y

    M, N = y.shape
    QB = quant_block_size
    dtype_max = torch.finfo(dtype_quant).max
    num_blocks = (N + QB - 1) // QB
    y = y.view(M, num_blocks, QB)
    max_val = y.abs().amax(dim=2, keepdim=True)
    dequant_scale = max_val / dtype_max
    # Round dequant_scale up to a power of two via the fp32 exponent field.
    exp = (dequant_scale.view(torch.int32) + 0x007FFFFF) & 0x7F800000
    rounded = exp.view(torch.float32)
    quant_scale = torch.where(rounded == 0, torch.zeros_like(rounded), 1.0 / rounded)
    out_q = (y * quant_scale).view(M, N).to(dtype_quant)
    scale = (exp >> 23).to(torch.uint8).view(M, num_blocks)
    return out_q, scale


@pytest.mark.parametrize("M", [1, 2, 7, 32, 100, 257])
@pytest.mark.parametrize("D", [2048, 3072])
@pytest.mark.parametrize("swiglu_limit", [0.0, 7.0])
@pytest.mark.parametrize("with_weights", [False, True])
@pytest.mark.parametrize("shuffle_scale", [False, True])
def test_fused_clamp_act_mul_ue8m0(M, D, swiglu_limit, with_weights, shuffle_scale):
    """ue8m0 group quant. The fp8 output and e8m0 scales must match the torch
    reference; when ``shuffle_scale`` is set the kernel must lay the scales out
    exactly like ``fp4_utils.e8m0_shuffle`` applied to the unshuffled scales."""
    torch.manual_seed(42)
    N = D // 2
    quant_block_size = 32
    dtype_quant = torch.float8_e4m3fn
    w = (
        torch.randn(M, 1, device="cuda", dtype=torch.float32) * 0.5
        if with_weights
        else None
    )
    inp = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)

    out_q, scale = fused_clamp_act_mul(
        inp,
        swiglu_limit=swiglu_limit,
        weights=w,
        activation="silu",
        dtype_quant=dtype_quant,
        quant_block_size=quant_block_size,
        scale_dtype_fmt="ue8m0",
        shuffle_scale=shuffle_scale,
    )

    ref_out, ref_scale = _torch_reference_ue8m0(
        inp, swiglu_limit, w, dtype_quant, quant_block_size
    )
    assert torch.equal(out_q.view(torch.uint8), ref_out.view(torch.uint8))

    num_blocks = (N + quant_block_size - 1) // quant_block_size
    if shuffle_scale:
        # Kernel preshuffles in place; the reference shuffles with e8m0_shuffle.
        # Both leave padding undefined, so undo the shuffle and compare the valid
        # region (which also confirms the kernel layout matches e8m0_shuffle).
        expected = fp4_utils.e8m0_shuffle(ref_scale)
        assert scale.shape == expected.shape
        sm = scale.shape[0]
        got = unshuffle_scale_gemm(scale.view(sm // 32, -1), arch="gfx950")[
            :M, :num_blocks
        ]
        exp = unshuffle_scale_gemm(expected.view(sm // 32, -1), arch="gfx950")[
            :M, :num_blocks
        ]
        assert torch.equal(got, exp)
        assert torch.equal(got, ref_scale)
    else:
        assert torch.equal(scale[:M, :num_blocks], ref_scale)

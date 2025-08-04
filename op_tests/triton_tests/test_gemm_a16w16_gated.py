import torch
import torch.nn.functional as F
import triton
import pytest
from aiter.ops.triton.gemm_a16w16_gated import gemm_a16w16_gated
from op_tests.triton_tests.test_gemm_a16w16 import minimal_x_vals
from op_tests.triton_tests.utils.types import str_to_torch_dtype


def generate_gemm_a16w16_gated_inputs(M, N, K, dtype, layout="TN", output=True):
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    # TN is default layout
    if layout[0] == "T":
        x = torch.randn((M, K), dtype=dtype).cuda()
    else:
        x = torch.randn((K, M), dtype=dtype).cuda().T

    if layout[1] == "T":
        weight = torch.randn((K, N), dtype=dtype).cuda().T
    else:
        weight = torch.randn((N, K), dtype=dtype).cuda()

    weight = weight / K**0.5  # scale down output variance to 1

    y = None
    if output:
        assert N % 2 == 0
        y = torch.empty((M, N // 2), dtype=dtype).cuda()
        out_dtype = (None,)
    else:
        out_dtype = dtype

    return x, weight, out_dtype, y


@pytest.mark.parametrize(
    "activation", ["gelu", "gelu_tanh", "silu", "silu_exp2", "relu"]
)
@pytest.mark.parametrize("M, N, K", minimal_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_gemm_a16_w16_gated(M: int, N: int, K: int, dtype, output, activation):
    if N % 2 != 0:
        pytest.skip("Skipping shape incompatible w/gating")
    x, w, out_dtype, y = generate_gemm_a16w16_gated_inputs(
        M, N, K, dtype, output=output
    )

    torch_out = F.linear(x, w, bias=None)
    if activation == "gelu":
        gating = F.gelu(torch_out[:, : N // 2])
    elif activation == "gelu_tanh":
        gating = F.gelu(torch_out[:, : N // 2], approximate="tanh")
    elif activation == "silu":
        gating = F.silu(torch_out[:, : N // 2])
    elif activation == "silu_exp2":
        gating = F.silu(torch_out[:, : N // 2])
    elif activation == "relu":
        gating = F.relu(torch_out[:, : N // 2])
    else:
        raise Exception(f"Unsupported activation: {activation}")
    torch_y = torch_out[:, N // 2 :]
    torch_out = gating * torch_y

    if output:
        triton_out = gemm_a16w16_gated(
            x,
            w,
            out_dtype,
            y,
            activation=activation,
        )
    else:
        triton_out = gemm_a16w16_gated(
            x,
            w,
            out_dtype,
            activation=activation,
        )

    """
    Note: There's a small distinction between Triton and Torch's implementations of silu
    (due to tl.sigmoid() vs torch.sigmoid()). The gated outputs can differ by as much as 3%.
    """
    triton.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)

import torch
import torch.nn.functional as F
import pytest
from aiter.ops.triton.gemm_a16w16_gated import gemm_a16w16_gated
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import get_x_vals
from op_tests.triton_tests.utils.types import str_to_torch_dtype


def generate_gemm_a16w16_gated_inputs(M, N, K, dtype, layout="TN", output=True):
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    # TN is default layout
    if layout[0] == "T":
        x = torch.randn((M, K), dtype=dtype, device="cuda")
    else:
        x = torch.randn((K, M), dtype=dtype, device="cuda").T

    if layout[1] == "T":
        weight = torch.randn((K, N), dtype=dtype, device="cuda").T
    else:
        weight = torch.randn((N, K), dtype=dtype, device="cuda")

    weight = weight / K**0.5  # scale down output variance to 1

    y = None
    if output:
        assert N % 2 == 0
        y = torch.empty((M, N // 2), dtype=dtype, device="cuda")
        out_dtype = (None,)
    else:
        out_dtype = dtype

    return x, weight, out_dtype, y


@pytest.mark.parametrize(
    "activation", ["gelu", "gelu_tanh", "silu", "silu_exp2", "relu", None]
)
@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["TN", "TT", "NN", "NT"])
@pytest.mark.parametrize("output", [True, False])
def test_gemm_a16_w16_gated(M: int, N: int, K: int, dtype, output, layout, activation):
    if N % 2 != 0:
        pytest.skip("Skipping shape incompatible w/gating")
    # This is done to reduce CI execution time
    if layout != "TN" and activation != "relu":
        pytest.skip("Skipping non-TN layouts when activation isn't ReLU")

    x, w, out_dtype, y = generate_gemm_a16w16_gated_inputs(
        M, N, K, dtype, layout=layout, output=output
    )

    torch_out = F.linear(x, w, bias=None)
    if activation == "gelu":
        gating = F.gelu(torch_out[:, : N // 2])
    elif activation == "gelu_tanh":
        gating = F.gelu(torch_out[:, : N // 2], approximate="tanh")
    elif activation == "silu" or activation == "silu_exp2":
        gating = F.silu(torch_out[:, : N // 2])
    elif activation == "relu":
        gating = F.relu(torch_out[:, : N // 2])
    elif activation is None:
        gating = torch_out[:, : N // 2]
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
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)

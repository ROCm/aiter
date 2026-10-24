import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.gemm.basic.gemm_a16w16_gated import gemm_a16w16_gated
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import get_x_vals
from op_tests.triton_tests.utils.types import str_to_torch_dtype

# TODO: Merge this with the regular GEMM A16W16


def generate_gemm_a16w16_gated_inputs(M, N, K, dtype, layout="TN", output=True):
    torch.manual_seed(0)
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
        raise RuntimeError(f"Unsupported activation: {activation}")
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


def get_bias_gate_x_vals():
    x_vals = [(1, 2, 1)]  # minimal case
    x_vals += [(3, 10, 2)]  # irregular shape
    x_vals += [(128, 1024, 512)]
    x_vals += [(333, 2560, 640)]  # SD1.5 GEGLU, M not a multiple of the tile
    x_vals += [(1024, 10240, 1280)]  # SDXL GEGLU
    return x_vals


@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "silu", None])
@pytest.mark.parametrize("M, N, K", get_bias_gate_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("gate_first", [True, False])
def test_gemm_a16_w16_gated_bias_gate_side(
    M: int, N: int, K: int, dtype, bias, gate_first, activation
):
    x, w, out_dtype, _ = generate_gemm_a16w16_gated_inputs(
        M, N, K, dtype, layout="TN", output=False
    )
    bias_tensor = torch.randn((N,), dtype=dtype, device="cuda") if bias else None

    torch_out = F.linear(x, w, bias=bias_tensor)
    first, second = torch_out[:, : N // 2], torch_out[:, N // 2 :]
    gate, value = (first, second) if gate_first else (second, first)
    if activation == "gelu":
        gate = F.gelu(gate)
    elif activation == "gelu_tanh":
        gate = F.gelu(gate, approximate="tanh")
    elif activation == "silu":
        gate = F.silu(gate)
    torch_out = gate * value

    triton_out = gemm_a16w16_gated(
        x,
        w,
        out_dtype,
        activation=activation,
        bias=bias_tensor,
        gate_first=gate_first,
    )

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)

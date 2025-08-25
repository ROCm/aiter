import torch
import torch.nn.functional as F
import pytest
from aiter.ops.triton.gemm_a16w16_gated import gemm_a16w16_gated
from aiter.ops.triton.utils.types import str_to_torch_dtype


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


def basic_shape_set():
    shapes = [(1, 1, 1)]  # minimal case
    shapes += [(3, 5, 2)]  # irregular shape
    shapes += [
        (32, 32, 32),
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (4864, 4096, 8192),
    ]
    shapes = [(2**i, 256, 7168) for i in range(1, 4)]
    return shapes


def extended_shape_set():
    shapes = [(2**i, 256, 7168) for i in range(5, 9)]
    shapes += [(1024 * v, 1024 * v, 1024 * v) for v in range(2, 9)]
    shapes = [(9728, 8192, 65536), (4864, 8192, 4160)]
    shapes += [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
        (1, 8192, 1024),
        (32, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (192, 8192, 1024),
        (256, 8192, 1024),
        (320, 8192, 1024),
        (512, 8192, 1024),
        (1024, 8192, 1024),
        (2048, 8192, 1024),
        (4096, 8192, 1024),
        (8192, 8192, 1024),
        (16384, 8192, 1024),
    ]
    return shapes


@pytest.mark.parametrize(
    "M, N, K, dtype, output, layout, activation",
    [
        (*shape, dtype_str, output, layout, activation)
        for shape in basic_shape_set()
        for dtype_str in ["bf16"]
        for output in [True, False]
        for layout in ["TN", "TT", "NN", "NT"]
        for activation in ["gelu", "gelu_tanh", "silu", "silu_exp2", "relu"]
    ]
    + [
        pytest.param(
            *shape, dtype_str, output, layout, activation, marks=pytest.mark.extended
        )
        for shape in extended_shape_set()
        for dtype_str in ["bf16"]
        for output in [True, False]
        for layout in ["TN", "TT", "NN", "NT"]
        for activation in ["gelu", "gelu_tanh", "silu", "silu_exp2", "relu"]
    ],
)
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

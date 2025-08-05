import torch
import torch.nn.functional as F
import triton
from op_tests.triton_tests.utils.types import str_to_torch_dtype

def generate_ff_inputs(
    batch, hidden_dim, intermediate_dim, dtype, layout="TN", gating=False, output=True
):
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    # TN is default layout
    if layout[0] == "T":
        x = torch.randn((batch, hidden_dim), dtype=dtype).cuda()
    else:
        x = torch.randn((hidden_dim, batch), dtype=dtype).cuda().T

    if layout[1] == "T":
        if gating:
            w1 = torch.randn((hidden_dim, intermediate_dim * 2), dtype=dtype).cuda().T
        else:
            w1 = torch.randn((hidden_dim, intermediate_dim), dtype=dtype).cuda().T
        w2 = torch.randn((intermediate_dim, hidden_dim), dtype=dtype).cuda()
    else:
        if gating:
            w1 = torch.randn((intermediate_dim * 2, hidden_dim), dtype=dtype).cuda()
        else:
            w1 = torch.randn((intermediate_dim, hidden_dim), dtype=dtype).cuda()
        w2 = torch.randn((hidden_dim, intermediate_dim), dtype=dtype).cuda().T

    w1 = w1 / (intermediate_dim**0.5)  # scale down output variance
    w2 = w2 / (hidden_dim**0.5)

    intermediate = None
    y = None
    if output:
        intermediate = torch.empty((batch, intermediate_dim), dtype=dtype).cuda()
        y = torch.empty((batch, hidden_dim), dtype=dtype).cuda()

    out_dtype = dtype

    return x, w1, w2, out_dtype, intermediate, y

def ff_ungated_test(
    fn: callable,
    batch: int, hidden_dim: int, intermediate_dim: int, dtype, output, activation
):
    x, w1, w2, out_dtype, _, y = generate_ff_inputs(
        batch, hidden_dim, intermediate_dim, dtype, gating=False, output=output
    )
    torch_out = F.linear(x, w1, bias=None)
    if activation == "gelu" or activation == "gelu_tanh":
        torch_out = F.gelu(torch_out, approximate="tanh")
    elif activation == "silu" or activation == "silu_exp2":
        torch_out = F.silu(torch_out)
    elif activation == "relu":
        torch_out = F.relu(torch_out)
    elif activation is None:
        pass
    else:
        raise Exception(f"Unsupported activation: {activation}")
    torch_out = torch_out @ w2

    if output:
        triton_out = fn(
            x,
            w1,
            w2,
            out_dtype,
            y=y,
            activation=activation,
        )
    else:
        triton_out = fn(
            x,
            w1,
            w2,
            out_dtype,
            activation=activation,
        )

    triton.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-2)


def ff_gated_test(
    fn: callable,
    batch: int, hidden_dim: int, intermediate_dim: int, dtype, output, activation
):
    x, w1, w2, out_dtype, _, y = generate_ff_inputs(
        batch, hidden_dim, intermediate_dim, dtype, gating=True, output=output
    )
    torch_out = F.linear(x, w1, bias=None)
    if activation == "gelu" or activation == "gelu_tanh":
        gating = F.gelu(torch_out[:, :intermediate_dim], approximate="tanh")
    elif activation == "silu" or activation == "silu_exp2":
        gating = F.silu(torch_out[:, :intermediate_dim])
    elif activation == "relu":
        gating = F.relu(torch_out[:, :intermediate_dim])
    elif activation is None:
        gating = torch_out[:, :intermediate_dim]
    else:
        raise Exception(f"Unsupported activation: {activation}")
    torch_y = torch_out[:, intermediate_dim:]
    torch_intermediate = gating * torch_y
    torch_out = torch_intermediate @ w2

    if output:
        triton_out = fn(
            x,
            w1,
            w2,
            out_dtype,
            y=y,
            activation=activation,
        )
    else:
        triton_out = fn(
            x,
            w1,
            w2,
            out_dtype,
            activation=activation,
        )

    """
    Note: There's a small distinction between Triton and Torch's implementations of silu
    (due to tl.sigmoid() vs torch.sigmoid()). The gated outputs can differ by as much as 3%.
    """
    triton.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)

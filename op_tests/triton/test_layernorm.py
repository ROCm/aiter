import triton
import torch
import triton.language as tl
import pytest
from aiter.ops.triton.norm import layer_norm


# pytest
@pytest.mark.parametrize(
    "M, N",
    [
        (1823, 781),
        (2, 128),
        (1, 4),
        (128, 2),
        (1, 128),
        (8192, 8192),
        (4096, 8192),
        (359, 1),
        (1, 359),
        (1, 131072),
        (1, 89999),
    ],
)
def test_layernorm(M, N, eps=1e-5):

    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda")
    w_shape = (N,)
    w = torch.rand(w_shape, device="cuda")
    b = torch.rand(w_shape, device="cuda")

    # forward pass
    y_triton = layer_norm(x, w, b, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, w, b, eps)

    triton.testing.assert_close(y_triton, y_ref, atol=1e-06, rtol=1e-05)

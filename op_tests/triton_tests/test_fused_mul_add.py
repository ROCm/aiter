import torch
import pytest
from aiter.ops.triton.fused_mul_add import fused_mul_add


def generate_qk_inputs(shape, a_type_is_scalar, b_type_is_scalar, dtype):
    x = torch.randn(*shape, dtype=dtype, device="cuda")

    if a_type_is_scalar[1]:
        a = torch.randn(1, dtype=dtype)
    else:
        a = torch.randn(*shape, dtype=dtype, device="cuda")

    if b_type_is_scalar[1]:
        b = torch.randn(1, dtype=dtype)
    else:
        b = torch.randn(*shape, dtype=dtype, device="cuda")

    if a_type_is_scalar[0] in [float, int]:
        a = a_type_is_scalar[0](a.item() * 100)
    else:
        a = a.to("cuda")

    if b_type_is_scalar[0] in [float, int]:
        b = b_type_is_scalar[0](b.item() * 100)
    else:
        b = b.to("cuda")

    return x, a, b


def ref_mul_add(x, a, b):
    return (a * x.to(torch.float32) + b).to(x.dtype)


@pytest.mark.parametrize(
    "shape", [(1,), (8,), (500,), (10000,), (32, 7168), (16, 50, 4186)]
)
@pytest.mark.parametrize(
    "a_type_is_scalar",
    [(float, True), (int, True), (torch.Tensor, True), (torch.Tensor, False)],
)
@pytest.mark.parametrize(
    "b_type_is_scalar",
    [(float, True), (int, True), (torch.Tensor, True), (torch.Tensor, False)],
)
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mul_add(shape, a_type_is_scalar, b_type_is_scalar, output: bool, dtype):

    x, a, b = generate_qk_inputs(shape, a_type_is_scalar, b_type_is_scalar, dtype)

    x_torch = ref_mul_add(x, a, b).clone()
    if output:
        x_triton = torch.empty_like(x)
        fused_mul_add(x, a, b, x_triton)
    else:
        x_triton = x
        fused_mul_add(x_triton, a, b)

    torch.testing.assert_close(x_torch, x_triton)

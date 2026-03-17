import sys
import types

import torch
from aiter.ops.triton.gluon.gfx1250_gemm_a8w8 import gemm_a8w8

m, n, k = 32, 7168, 4096
in_dtype = torch.float8_e4m3fn
out_dtype = torch.bfloat16

def generate_gemm_a8w8_inputs(
    M: int,
    N: int,
    K: int,
    in_dtype,
    out_dtype,
    layout: str = "TN",
    output: bool = False,
    shuffle: bool = False,
):
    """
    The GEMM kernel expects:
    - x: (M, K) -> row-major format
    - w: (N, K) -> column-major format
    """
    if layout[0] == "T":
        x = torch.randn((M, K), dtype=torch.float32, device="cuda")
    else:
        x = torch.randn((K, M), dtype=torch.float32, device="cuda").T

    if layout[1] == "N":
        weight = torch.randn((N, K), dtype=torch.float32, device="cuda")
    else:
        weight = torch.randn((K, N), dtype=torch.float32, device="cuda").T

    max_x = x.abs().float().amax(dim=1, keepdim=True)
    x_scale = max_x / torch.finfo(in_dtype).max
    x = x / x_scale
    x = x.to(in_dtype)

    max_weight = weight.abs().float().amax(dim=1, keepdim=True).T.contiguous()
    w_scale = max_weight / torch.finfo(in_dtype).max
    weight = weight / w_scale.T
    weight = weight.to(in_dtype)

    bias = torch.rand([1, N], dtype=torch.float32, device="cuda") * 10

    weight_shuffled = weight

    y = None
    if output:
        y = torch.empty((M, N), dtype=out_dtype, device="cuda")

    return x, weight, weight_shuffled, x_scale, w_scale, bias, y


x, weight, weight_triton, x_scale, w_scale, bias, y = generate_gemm_a8w8_inputs(
        M=m,
        N=n,
        K=k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
    )

y = gemm_a8w8(x, weight, x_scale, w_scale, bias, out_dtype, y)
torch.cuda.synchronize()
print("Kernel dispatched successfully")

    # x = x.cuda()
    # w = w.cuda()
    # x_scale = x_scale.cuda()
    # w_scale = w_scale.cuda()
    # bias = bias.cuda()
    # y = y.cuda()
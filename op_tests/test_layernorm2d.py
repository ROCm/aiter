import torch
import torch.nn.functional as F
import ater
from ater.test_common import checkAllclose, perftest


@perftest()
def run_torch(input, weight, bias, eps, residual=None):
    if residual is None:
        residual_out = None
        output = F.layer_norm(
            input=input,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps
        )
    else:
        residual_out = input + residual
        output = F.layer_norm(
            input=residual_out,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps
        )
    return output, residual_out


@perftest()
def run_ck(input, weight, bias, eps, residual=None):
    if residual is None:
        residual_out = None
        output = ater.layer_norm(input, weight, bias, eps)
        # output = torch.empty_like(input)
        # ater.layernorm2d_fwd(
        #     output,
        #     input,
        #     weight,
        #     bias,
        #     eps
        # )
    else:
        residual_out = torch.empty_like(input)
        output = torch.empty_like(input)
        ater.layernorm2d_fwd_with_add(
            output,
            input,
            residual,
            residual_out,
            weight,
            bias,
            eps
        )
    return output, residual_out


def test_layernorm2d(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    hidden_stats = torch.randn(m, n*8, dtype=dtype, device="cuda")
    q, k, v = torch.split(hidden_stats, [6*n, n, n], dim=1)
    input = k
    (a, *_), avg_a = run_torch(input, weight, bias, 1e-5)
    (b, *_), avg_b = run_ck(input, weight, bias, 1e-5)
    msg = f"[perf] dim: {str(dim):<20}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg=msg)


def test_layernorm2d_fuseAdd(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn(dim, dtype=dtype, device="cuda")
    hidden_stats = torch.randn(m, n*8, dtype=dtype, device="cuda")
    q, k, v = torch.split(hidden_stats, [6*n, n, n], dim=1)
    input = k
    (a, res_a, *_), avg_a = run_torch(input, weight, bias, 1e-5, residual=res)
    (b, res_b, *_), avg_b = run_ck(input, weight, bias, 1e-5, residual=res)

    msg = f"[perf] dim: {str(dim):<20}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, atol=0.03, msg=msg)
    checkAllclose(res_a, res_b, msg='res check')


# for dtype in [torch.float16, torch.bfloat16]:
#     for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
#         for n in [4096, 8192, 16384, 32768, 65536]:
#             test_layernorm2d(dtype, m, n)
test_layernorm2d(torch.float16, 128, 8192)


# print('\nstart fuse add test')
# for dtype in [torch.float16, torch.bfloat16]:
#     for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
#         for n in [4096, 8192, 16384, 32768, 65536]:
#             test_layernorm2d_fuseAdd(dtype, m, n)

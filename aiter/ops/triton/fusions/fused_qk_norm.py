import torch
import triton
from aiter.ops.triton._triton_kernels.fusions.fused_qk_norm import (
    _fused_qk_rmsnorm_kernel,
)
from aiter import rmsnorm2d_fwd
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_qk_rmsnorm(
    inp1,
    inp1_weight,
    inp1_epsilon,
    inp2,
    inp2_weight,
    inp2_epsilon,
):

    M, N1 = inp1.shape
    M2, N2 = inp2.shape
    assert (
        M == M2
    ), "The leading dimension should be identical between inp1 and inp2"

    # fallback to split calls for large M
    if M >= 8192:
        inp1_normed = rmsnorm2d_fwd(inp1, inp1_weight, inp1_epsilon)
        inp2_normed = rmsnorm2d_fwd(inp2, inp2_weight, inp2_epsilon)
        return inp1_normed, inp2_normed

    out1 = torch.empty((M, N1), dtype=inp1.dtype, device=inp1.device)
    out2 = torch.empty((M, N2), dtype=inp1.dtype, device=inp1.device)

    # tl.arange requires power-of-2 ranges; keep per-tensor block sizes
    # independent to avoid over-padding the smaller side.
    BLOCK_SIZE_N1 = triton.next_power_of_2(N1)
    BLOCK_SIZE_N2 = triton.next_power_of_2(N2)

    max_block = max(BLOCK_SIZE_N1, BLOCK_SIZE_N2)
    if max_block <= 512:
        num_warps = 2
    elif max_block <= 2048:
        num_warps = 4
    elif max_block <= 4096:
        num_warps = 8
    else:
        num_warps = 16

    _fused_qk_rmsnorm_kernel[(M,)](
        inp1,
        out1,
        out1.stride(0),
        out1.stride(1),
        inp1_weight,
        inp2,
        out2,
        out2.stride(0),
        out2.stride(1),
        inp2_weight,
        inp1_epsilon,
        inp2_epsilon,
        N1,
        N2,
        inp1.stride(0),
        inp2.stride(0),
        inp1.stride(1),
        inp2.stride(1),
        BLOCK_SIZE_N1=BLOCK_SIZE_N1,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        num_warps=num_warps,
    )

    return out1, out2
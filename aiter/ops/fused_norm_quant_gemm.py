"""Fused RMSNorm + FP8 Quant + GEMM host-level fusion.

Calls rmsnorm_quant (CK kernel) then hipb_mm (hipBLASLt GEMM) back-to-back
with minimal Python overhead between them.
"""

import torch


def fused_rmsnorm_quant_gemm(
    input_2d: torch.Tensor,
    weight_fp8: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    scale_a: torch.Tensor,
    scale_w: torch.Tensor,
    fp8_workspace: torch.Tensor,
    solution_index: int = -1,
) -> torch.Tensor:
    """Fused RMSNorm + FP8 Quant + hipBLASLt GEMM.

    Args:
        input_2d: [M, K] BF16 input
        weight_fp8: [N, K] FP8 weight
        norm_w: [K] BF16 norm weight
        eps: RMSNorm epsilon
        scale_a: [1] FP32 activation scale (amax / fp8_max)
        scale_w: [1] FP32 weight scale (amax / fp8_max)
        fp8_workspace: [M, K] pre-allocated FP8 buffer
        solution_index: hipBLASLt solution index (-1 for auto)

    Returns:
        [M, N] BF16 output
    """
    from aiter.ops.gradlib import hipb_mm
    from aiter.ops.rmsnorm_quant import rmsnorm_quant as _rmsnorm_quant

    _rmsnorm_quant(fp8_workspace, input_2d, scale_a, norm_w, eps)

    weight_t = weight_fp8.t()
    sa = scale_a.to(torch.float32).reshape(1, 1)
    sw = scale_w.to(torch.float32).reshape(1, 1)

    return hipb_mm(
        fp8_workspace,
        weight_t,
        solution_index,
        out_dtype=torch.bfloat16,
        scaleA=sa,
        scaleB=sw,
    )

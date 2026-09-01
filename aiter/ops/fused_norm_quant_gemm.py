"""Fused RMSNorm + FP8 Quant + GEMM host-level fusion.

Calls rmsnorm_quant (CK kernel) then hipb_mm (hipBLASLt GEMM) back-to-back
with minimal Python overhead between them.

Falls back to separate Python calls if the fused C++ JIT module is not available.
"""

import logging

import torch

_log = logging.getLogger(__name__)

_fused_cpp = None
_fused_cpp_probed = False


def _get_fused_cpp():
    """Try to load the C++ JIT fused module."""
    global _fused_cpp, _fused_cpp_probed
    if _fused_cpp_probed:
        return _fused_cpp
    _fused_cpp_probed = True
    try:
        from aiter import module_fused_norm_quant_gemm

        _fused_cpp = module_fused_norm_quant_gemm
        _log.info("fused_norm_quant_gemm: C++ JIT module loaded")
    except (ImportError, OSError, RuntimeError) as e:
        _log.debug(
            "fused_norm_quant_gemm: C++ JIT module not available (%s), using Python fallback",
            e,
        )
        _fused_cpp = None
    return _fused_cpp


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
    cpp = _get_fused_cpp()
    if cpp is not None:
        try:
            return cpp.fused_rmsnorm_quant_gemm(
                input_2d,
                weight_fp8,
                norm_w,
                eps,
                scale_a,
                scale_w,
                fp8_workspace,
                solution_index,
            )
        except (ImportError, OSError, RuntimeError) as e:
            _log.debug(
                "fused_norm_quant_gemm: C++ path failed (%s), using Python fallback",
                e,
            )

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

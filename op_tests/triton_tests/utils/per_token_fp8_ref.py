"""Torch reference of dynamic per-token FP8 quantization."""

import torch


def per_token_fp8_ref(y: torch.Tensor, dtype_quant: torch.dtype):
    """Quantize y as stored: returns (fp8 [M, N], fp32 scale [M, 1])."""
    fp8_max = torch.finfo(dtype_quant).max
    yf = y.float()
    s = (yf.abs().amax(-1, keepdim=True) / fp8_max).clamp_min(1.0 / (fp8_max * 512))
    return (yf / s).clamp(-fp8_max, fp8_max).to(dtype_quant), s

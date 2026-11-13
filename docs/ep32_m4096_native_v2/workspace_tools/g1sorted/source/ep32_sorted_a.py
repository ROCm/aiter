"""Experimental sorted FP8 payload for the single EP32/M4096 GEMM1 shape."""
import torch
from .kernels.ep32_sorted_a import compile_sorted_a
from .kernels.tensor_shim import _run_compiled, ptr_arg


def prepare(a, mind, valid, n_tokens):
    from aiter.fused_moe import _stage2_scratch
    assert a.shape == (4096, 3584) and n_tokens == 4096
    rows = ((mind.numel() + 79) // 80) * 80
    target = _stage2_scratch("g1_sorted_a", (rows, 3584), torch.uint8, a.device)
    _run_compiled(compile_sorted_a(a.device), ptr_arg(a), ptr_arg(mind),
        ptr_arg(valid), ptr_arg(target), n_tokens, torch.cuda.current_stream(a.device))
    return target

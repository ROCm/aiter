import torch
from .moe_kernels import _run_compiled
from .kernels.mxmoe_reduce_ep32 import compile_native_reduce
from .kernels.tensor_shim import ptr_arg


def ep32_native_reduce(x, reverse, weights, mask, ids, valid, out):
    if out.shape[1] != 3584 or ids.shape[1] not in (16, 17) or x.dtype != torch.float32:
        raise ValueError("native BM80 reduce requires H3584 top-k16/17 FP32 staging")
    _run_compiled(compile_native_reduce(ids.shape[1], out.device), (
        ptr_arg(x), ptr_arg(reverse), ptr_arg(weights), ptr_arg(mask),
        ptr_arg(ids), ptr_arg(valid), ptr_arg(out), out.shape[0], mask.numel(),
        torch.cuda.current_stream(out.device),
    ))

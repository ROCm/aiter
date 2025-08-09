import torch
import triton
from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.gemm_a16w16_atomic import gemm_a16w16_atomic
from aiter.ops.triton.gemm_a16w16_atomic_fused_db import gemm_a16w16_atomic_fused_db
from aiter.ops.triton.reduce_db import reduce_db

class TritonGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        dtype = x.dtype
        y = gemm_a16w16(x, w.t(), b, dtype=dtype)
        
        ctx.save_for_backward(x, w, b)

        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b = ctx.saved_tensors
        dtype = x.dtype
        dx = dw = db = None

        if ctx.needs_input_grad[0]:
            dx = gemm_a16w16(dy, w, dtype=dtype)
        if ctx.needs_input_grad[1]:
            dw = gemm_a16w16_atomic(x.t(), dy.t(), dtype=torch.float32).to(dtype)
        if ctx.needs_input_grad[2]:
            if b.dim() == 1:
                db = reduce_db(dy).to(dtype)
            elif b.dim() == 2:
                db = dy

        return dx, dw, db


def gemm_a16w16_autograd(x, w, b):
    return TritonGemm.apply(x, w, b)

"""Prepare only filled EP32 sorted rows, preserving the original FP8 codes."""
from functools import cache
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from .mxfp4_gemm_common import global_typed_ptr


@flyc.kernel(name="ep32_prepare_sorted_a_fp8", known_block_size=[256, 1, 1])
def sorted_a_kernel(a: fx.Pointer, mind: fx.Pointer, valid: fx.Pointer,
                    out: fx.Pointer, n_tokens: fx.Int32):
    tx = fx.thread_idx.x
    ids = global_typed_ptr(fx.ptrtoint(mind), T.i32)
    count = global_typed_ptr(fx.ptrtoint(valid), T.i32)
    src = global_typed_ptr(fx.ptrtoint(a), T.i32, align=16)
    dst = global_typed_ptr(fx.ptrtoint(out), T.i32, align=16)
    src = fx.rocdl.make_buffer_tensor(fx.Tensor(fx.make_view(src, fx.make_layout((1, 1), (1, 1)))), max_size=True)
    dst = fx.rocdl.make_buffer_tensor(fx.Tensor(fx.make_view(dst, fx.make_layout((1, 1), (1, 1)))), max_size=True)
    copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
    # Each wave copies four complete rows; its 16-lane subgroups copy 256B.
    for block in range(fx.block_idx.x, (count[0] + 15) // 16, fx.Int32(512)):
        row = block * 16 + tx // 16
        if row < count[0]:
            token = ids[row]
            legal = (token >= 0) & (token < n_tokens)
            for part in range_constexpr(14):
                col_dw = (part * 16 + tx % 16) * 4
                frag = fx.make_rmem_tensor(4, fx.Int32)
                frag.store(fx.full(4, 0, fx.Int32))
                if legal:
                    fx.copy(copy, src[None, token * 896 + col_dw], frag)
                fx.copy(copy, frag, dst[None, row * 896 + col_dw])


@cache
def compile_sorted_a(device):
    @flyc.jit
    def launch(a: fx.Pointer, mind: fx.Pointer, valid: fx.Pointer,
               out: fx.Pointer, n_tokens: fx.Int32, stream: fx.Stream):
        sorted_a_kernel(a, mind, valid, out, n_tokens).launch(
            grid=(512, 1, 1), block=(256, 1, 1), stream=stream)
    return launch

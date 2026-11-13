"""Reduce native 16x16 MFMA fragments for EP32 BM80 H3584 top-k16."""
import functools
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from .mxfp4_gemm_common import global_typed_ptr


@flyc.kernel(name="ep32_bm80_moe_reduce_native_f32", known_block_size=[256, 1, 1])
def native_reduce_kernel(
    arg_x: fx.Pointer, arg_rev: fx.Pointer, arg_weight: fx.Pointer,
    arg_mask: fx.Pointer, arg_ids: fx.Pointer, arg_valid: fx.Pointer,
    arg_out: fx.Pointer, n_experts: fx.Int32, TOPK: fx.Constexpr[int],
):
    token = fx.block_idx.x
    col = fx.block_idx.y * fx.Int32(2048) + fx.thread_idx.x * fx.Int32(8)
    xptr = global_typed_ptr(fx.ptrtoint(arg_x), T.f32, align=16)
    yptr = global_typed_ptr(fx.ptrtoint(arg_out), T.bf16, align=16)
    xbuf = fx.rocdl.make_buffer_tensor(fx.Tensor(fx.make_view(xptr, fx.make_layout((1, 1), (1, 1)))), max_size=True)
    ybuf = fx.rocdl.make_buffer_tensor(fx.Tensor(fx.make_view(yptr, fx.make_layout((1, 1), (1, 1)))), max_size=True)
    reverse = global_typed_ptr(fx.ptrtoint(arg_rev), T.i32)
    weight = global_typed_ptr(fx.ptrtoint(arg_weight), T.f32)
    mask = global_typed_ptr(fx.ptrtoint(arg_mask), T.i32)
    ids = global_typed_ptr(fx.ptrtoint(arg_ids), T.i32)
    valid = global_typed_ptr(fx.ptrtoint(arg_valid), T.i32)
    load = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    store = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    if col < fx.Int32(3584):
        acc = fx.full(8, 0.0, fx.Float32)
        for k in range_constexpr(TOPK):
            route = token * fx.Int32(TOPK) + fx.Int32(k)
            expert = ids[route]
            legal = (token < valid[1]) & (expert >= fx.Int32(0)) & (expert < n_experts)
            if legal:
                if mask[expert] != fx.Int32(0):
                    pos = reverse[route]
                    if (pos >= fx.Int32(0)) & (pos < valid[0]):
                        # Native order: [row16, col16, lane64, fragment4].
                        base = ((pos // 16) * 224 + col // 16) * 256
                        base = base + ((pos % 16) // 4) * 64 + (col % 16) * 4 + pos % 4
                        w = fx.Float32(weight[pos])
                        values = []
                        for j in range_constexpr(8):
                            frag = fx.make_rmem_tensor(1, fx.Float32)
                            fx.copy(load, xbuf[None, base + fx.Int32(j * 4)], frag)
                            value = frag.load()[0] * w
                            # Match GEMM2's per-route BF16 rounding before sum.
                            values.append(value.to(fx.BFloat16).to(fx.Float32))
                        acc = acc + fx.Vector.from_elements(values, fx.Float32)
        out = fx.make_rmem_tensor(8, fx.BFloat16)
        out.store(acc.to(fx.BFloat16))
        fx.copy(store, out, ybuf[None, token * fx.Int32(3584) + col])


@functools.cache
def compile_native_reduce(topk):
    @flyc.jit
    def launch_native_reduce(
        arg_x: fx.Pointer, arg_rev: fx.Pointer, arg_weight: fx.Pointer,
        arg_mask: fx.Pointer, arg_ids: fx.Pointer, arg_valid: fx.Pointer,
        arg_out: fx.Pointer, n_tokens: fx.Int32, n_experts: fx.Int32,
        stream: fx.Stream,
    ):
        native_reduce_kernel(arg_x, arg_rev, arg_weight, arg_mask, arg_ids, arg_valid,
            arg_out, n_experts, TOPK=topk).launch(grid=(n_tokens, 2, 1), block=(256, 1, 1), stream=stream)
    return launch_native_reduce

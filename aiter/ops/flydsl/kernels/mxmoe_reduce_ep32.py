"""Reduce native fragments with shared routes and paired-column BF16 loads."""
import functools
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from .mxfp4_gemm_common import global_typed_ptr


@flyc.kernel(name="ep32_bm80_moe_reduce_native_nativewide", known_block_size=[256, 1, 1])
def native_reduce_kernel(
    arg_x: fx.Pointer, arg_rev: fx.Pointer, arg_weight: fx.Pointer,
    arg_mask: fx.Pointer, arg_ids: fx.Pointer, arg_valid: fx.Pointer,
    arg_out: fx.Pointer, n_experts: fx.Int32, TOPK: fx.Constexpr[int],
    PREWEIGHTED_BF16: fx.Constexpr[bool],
):
    token = fx.block_idx.x
    tx = fx.thread_idx.x
    col = fx.block_idx.y * fx.Int32(2048) + tx * fx.Int32(8)
    x_elem = fx.BFloat16 if const_expr(PREWEIGHTED_BF16) else fx.Float32
    xptr = global_typed_ptr(fx.ptrtoint(arg_x), T.bf16 if const_expr(PREWEIGHTED_BF16) else T.f32, align=16)
    yptr = global_typed_ptr(fx.ptrtoint(arg_out), T.bf16, align=16)
    xbuf = fx.rocdl.make_buffer_tensor(fx.Tensor(fx.make_view(xptr, fx.make_layout((1, 1), (1, 1)))), max_size=True)
    ybuf = fx.rocdl.make_buffer_tensor(fx.Tensor(fx.make_view(yptr, fx.make_layout((1, 1), (1, 1)))), max_size=True)
    reverse = global_typed_ptr(fx.ptrtoint(arg_rev), T.i32)
    weight = global_typed_ptr(fx.ptrtoint(arg_weight), T.f32)
    mask = global_typed_ptr(fx.ptrtoint(arg_mask), T.i32)
    ids = global_typed_ptr(fx.ptrtoint(arg_ids), T.i32)
    valid = global_typed_ptr(fx.ptrtoint(arg_valid), T.i32)
    route_storage = fx.SharedAllocator().allocate(fx.Array[fx.Int32, TOPK + 1]).peek()
    route_pos = route_storage.view(fx.make_layout(TOPK + 1, 1))
    valid_tokens = valid[1]
    if token < valid_tokens:
        if tx < fx.Int32(TOPK):
            route = token * fx.Int32(TOPK) + tx
            expert = ids[route]
            pos = fx.Int32(-1)
            if (expert >= fx.Int32(0)) & (expert < n_experts):
                if mask[expert] != fx.Int32(0):
                    mapped = reverse[route]
                    if (mapped >= fx.Int32(0)) & (mapped < valid[0]):
                        pos = mapped
            route_pos[tx] = pos
            route_bits = fx.Int32(fx.rocdl.ballot(T.i64, pos >= fx.Int32(0)))
            if tx == fx.Int32(0):
                route_pos[fx.Int32(TOPK)] = route_bits
        # The last N tile has an inactive output wave; it must join this barrier.
        gpu.barrier()
    load = fx.make_copy_atom(fx.rocdl.BufferCopy128b() if const_expr(PREWEIGHTED_BF16) else fx.rocdl.BufferCopy32b(), x_elem)
    store = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    if col < fx.Int32(3584):
        acc = fx.full(8, 0.0, fx.Float32)
        if token < valid_tokens:
            route_bits = route_pos[fx.Int32(TOPK)]
            for k in range_constexpr(TOPK):
                if (route_bits & fx.Int32(1 << k)) != fx.Int32(0):
                    pos = route_pos[fx.Int32(k)]
                    # Native order: [row16, col16, lane64, fragment4].
                    # The route bitmap guarantees pos >= 0; columns are nonnegative.
                    base = ((pos >> 4) * 224 + (col >> 4)) * 256
                    base = base + ((pos >> 2) & 3) * 64 + (col & 15) * 4 + (pos & 3)
                    if const_expr(not PREWEIGHTED_BF16):
                        w = fx.Float32(weight[pos])
                    values = []
                    if const_expr(PREWEIGHTED_BF16):
                        row = pos & fx.Int32(3)
                        for j in range_constexpr(4):
                            # A native pair of columns contains all four M rows.
                            frag = fx.make_rmem_tensor(8, x_elem)
                            fx.copy(load, xbuf[None, base - row + fx.Int32(j * 8)], frag)
                            pair = frag.load()
                            values.append(pair[row].to(fx.Float32))
                            values.append(pair[row + fx.Int32(4)].to(fx.Float32))
                    else:
                        for j in range_constexpr(8):
                            frag = fx.make_rmem_tensor(1, x_elem)
                            fx.copy(load, xbuf[None, base + fx.Int32(j * 4)], frag)
                            value = frag.load()[0] * w
                            values.append(value.to(fx.BFloat16).to(fx.Float32))
                    acc = acc + fx.Vector.from_elements(values, fx.Float32)
        out = fx.make_rmem_tensor(8, fx.BFloat16)
        out.store(acc.to(fx.BFloat16))
        fx.copy(store, out, ybuf[None, token * fx.Int32(3584) + col])


@functools.cache
def compile_native_reduce(topk, preweighted_bf16=False):
    @flyc.jit
    def launch_native_reduce(
        arg_x: fx.Pointer, arg_rev: fx.Pointer, arg_weight: fx.Pointer,
        arg_mask: fx.Pointer, arg_ids: fx.Pointer, arg_valid: fx.Pointer,
        arg_out: fx.Pointer, n_tokens: fx.Int32, n_experts: fx.Int32,
        stream: fx.Stream,
    ):
        native_reduce_kernel(arg_x, arg_rev, arg_weight, arg_mask, arg_ids, arg_valid,
            arg_out, n_experts, TOPK=topk, PREWEIGHTED_BF16=preweighted_bf16).launch(grid=(n_tokens, 2, 1), block=(256, 1, 1), stream=stream)
    return launch_native_reduce

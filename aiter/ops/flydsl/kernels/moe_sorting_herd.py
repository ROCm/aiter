from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from .herd_topk import _kept_routes
from .moe_sorting_kernel import _buf_iter, _gld, _gst, _zero_moe_buf_grid_stride


@cache
def build_moe_sorting_herd_module(
    experts: int,
    topk: int,
    max_tokens: int,
    unit: int,
):
    threads = max(128, 1 << (experts - 1).bit_length())
    if not 1 <= experts <= 1024 or not 1 <= max_tokens <= 128:
        raise ValueError("HERD sorting supports E<=1024 and M<=128")
    kp1 = topk + 1

    @fx.struct
    class Storage:
        pop: fx.Array[fx.Int32, threads, 16]
        count: fx.Array[fx.Int32, threads, 16]
        scan0: fx.Array[fx.Int32, threads, 16]
        scan1: fx.Array[fx.Int32, threads, 16]
        slots: fx.Array[fx.Int32, max_tokens * topk, 16]
        route_ids: fx.Array[fx.Int32, max_tokens * topk, 16]
        route_weights: fx.Array[fx.Float32, max_tokens * topk, 16]

    @flyc.kernel(
        name=f"moe_sorting_herd_e{experts}_k{topk}_m{max_tokens}_b{unit}_r1_s0",
        known_block_size=[threads, 1, 1],
    )
    def kernel(
        cand_ids: fx.Tensor,
        cand_values: fx.Tensor,
        kept_ids: fx.Tensor,
        kept_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_experts: fx.Tensor,
        valid_ids: fx.Tensor,
        moe_buf: fx.Tensor,
        tokens: fx.Int32,
        buf_words: fx.Int32,
        scale: fx.Float32,
    ):
        tid = gpu.thread_idx.x
        bid = gpu.block_idx.x
        ids_p = _buf_iter(cand_ids)
        values_p = _buf_iter(cand_values)
        kept_p = _buf_iter(kept_ids)
        weights_p = _buf_iter(kept_weights)
        sorted_p = _buf_iter(sorted_ids)
        sorted_w = _buf_iter(sorted_weights)
        experts_p = _buf_iter(sorted_experts)
        valid_p = _buf_iter(valid_ids)
        buf_p = _buf_iter(moe_buf)
        lds = fx.SharedAllocator().allocate(Storage).peek()
        pop = lds.pop.ptr
        counts = lds.count.ptr
        slots = lds.slots.ptr
        route_ids = lds.route_ids.ptr
        route_weights = lds.route_weights.ptr
        scan0 = lds.scan0.ptr
        scan1 = lds.scan1.ptr

        if bid != 0:
            _zero_moe_buf_grid_stride(
                buf_p,
                (bid - 1) * threads + tid,
                (gpu.grid_dim.x - 1) * threads,
                buf_words // 4,
                fx.Int32(0x7FFFFFFF),
            )
        if bid == 0:
            pop[tid] = fx.Int32(0)
            counts[tid] = fx.Int32(0)
            gpu.barrier()
            for r in range(tid, tokens * kp1, threads):
                expert = _gld(ids_p, r)
                fx.atomic_add(
                    pop + fx.Int64(expert), fx.Int32(1), syncscope="workgroup"
                )
            gpu.barrier()
            if tid < tokens:
                ids = [
                    fx.Int32(_gld(ids_p, tid * kp1 + j)) for j in range_constexpr(kp1)
                ]
                values = [_gld(values_p, tid * kp1 + j) for j in range_constexpr(kp1)]
                pops = [fx.Int32(pop[ids[j]]) for j in range_constexpr(kp1)]
                keeps, ranks, weights = _kept_routes(ids, values, pops, scale)
                for j in range_constexpr(kp1):
                    if keeps[j]:
                        route = tid * topk + ranks[j]
                        slot = fx.atomic_add(
                            counts + fx.Int64(ids[j]),
                            fx.Int32(1),
                            syncscope="workgroup",
                        )
                        slots[route] = fx.Int32(slot)
                        route_ids[route] = ids[j]
                        route_weights[route] = weights[j]
                        _gst(kept_p, ids[j], route)
                        _gst(weights_p, weights[j], route)
            gpu.barrier()
            count = fx.Int32(counts[tid])
            scan0[tid] = (count + unit - 1) // unit * unit
            gpu.barrier()
            src, dst = scan0, scan1
            for shift in range_constexpr(threads.bit_length() - 1):
                offset = 1 << shift
                value = fx.Int32(src[tid])
                if tid >= offset:
                    value = value + fx.Int32(src[tid - offset])
                dst[tid] = value
                gpu.barrier()
                src, dst = dst, src
            start = fx.Int32(0)
            if tid != 0:
                start = fx.Int32(src[tid - 1])
            dst[tid] = start
            if tid == 0:
                _gst(valid_p, fx.Int32(src[threads - 1]), fx.Int32(0))
                _gst(valid_p, tokens, fx.Int32(1))
            if tid < experts:
                padded = (count + unit - 1) // unit * unit
                sentinel = fx.Int32(topk << 24) | tokens
                for row in range(count, padded, 1):
                    _gst(sorted_p, sentinel, start + row)
                    _gst(sorted_w, fx.Float32(0.0), start + row)
                for block in range(fx.Int32(0), padded // unit, 1):
                    _gst(experts_p, tid, start // unit + block)
            gpu.barrier()
            for route in range(tid, tokens * topk, threads):
                expert = fx.Int32(route_ids[route])
                row = fx.Int32(dst[expert]) + fx.Int32(slots[route])
                token = route // topk
                rank = route % topk
                _gst(sorted_p, (rank << 24) | token, row)
                _gst(sorted_w, fx.Float32(route_weights[route]), row)

    @flyc.jit
    def launch(
        cand_ids: fx.Tensor,
        cand_values: fx.Tensor,
        kept_ids: fx.Tensor,
        kept_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_experts: fx.Tensor,
        valid_ids: fx.Tensor,
        moe_buf: fx.Tensor,
        tokens: fx.Int32,
        buf_words: fx.Int32,
        scale: fx.Float32,
        blocks: fx.Int32,
        stream: fx.Stream,
    ):
        if const_expr(max_tokens < 1):
            raise ValueError("max_tokens must be positive")
        kernel(
            cand_ids,
            cand_values,
            kept_ids,
            kept_weights,
            sorted_ids,
            sorted_weights,
            sorted_experts,
            valid_ids,
            moe_buf,
            tokens,
            buf_words,
            scale,
        ).launch(grid=(blocks, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch

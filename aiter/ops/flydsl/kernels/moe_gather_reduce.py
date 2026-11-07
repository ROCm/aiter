# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-pass deterministic MoE weighted gather-reduce epilogue.

Inputs are grouped BF16/FP16 rows, an int32 token/top-k-to-row map, and
FP32/BF16/FP16 route weights. Each CTA owns a token and an output-column tile.
It sums split-K slices first, then weighted routes in their original order,
rounding only the final result to the output dtype. There are no atomics.

The opt-in gfx1250 specialization caches route metadata in wave registers: coalesced lane loads followed
by readlane broadcasts before any output-tail divergence. Other targets keep
the existing CTA-wide LDS cache. DROPPED_ROUTE_ROW uses a zero-length source
descriptor, so even NaNs in an unmapped row are never read. Source descriptors
include the row in a 64-bit base address; their offsets stay within one row.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import Int32, T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.kernels_common import format_kernel_name
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_copy_atom,
    ptr_buf_tensor,
)

BLOCK_THREADS = 256
MAX_GATHER_TOPK = 32


@fx.struct
class _GatherRouteLds:
    rows: fx.Array[fx.Int32, MAX_GATHER_TOPK, 16]
    w_bits: fx.Array[fx.Int32, MAX_GATHER_TOPK, 16]


def _lds_li32(ptr, idx):
    return fx.ptr_load(ptr + fx.Int64(idx))


def _lds_si32(ptr, val, idx):
    fx.ptr_store(val, ptr + fx.Int64(idx))


def _unpack_pair_to_f32(raw_dw, out_dtype):
    lo16 = raw_dw & 0xFFFF
    hi16 = (raw_dw >> 16) & 0xFFFF
    if out_dtype == "bf16":
        return (lo16 << 16).bitcast(fx.Float32), (hi16 << 16).bitcast(fx.Float32)
    return (
        fx.Uint16(lo16).bitcast(fx.Float16).to(fx.Float32),
        fx.Uint16(hi16).bitcast(fx.Float16).to(fx.Float32),
    )


def _pack_pair_from_f32(acc_lo, acc_hi, out_dtype):
    odt = fx.BFloat16 if out_dtype == "bf16" else fx.Float16
    lo_i32 = fx.Uint32(acc_lo.to(odt).bitcast(fx.Uint16))
    hi_i32 = fx.Uint32(acc_hi.to(odt).bitcast(fx.Uint16))
    return lo_i32 | (hi_i32 << 16)


def build_moe_gather_reduce_module(
    model_dim: int,
    topk: int,
    out_dtype: str = "bf16",
    split_k: int = 1,
    vec_dwords: int = 2,
    w_dtype: str = "f32",
    wave_routes: bool = False,
):
    assert model_dim % 2 == 0
    assert out_dtype in ("bf16", "f16")
    assert w_dtype in ("f32", "bf16", "f16")
    assert topk <= MAX_GATHER_TOPK
    if vec_dwords not in (2, 4, 8):
        raise ValueError(f"vec_dwords must be 2, 4, or 8, got {vec_dwords}")

    if wave_routes and not 0 < topk <= 32:
        raise ValueError("wave route cache requires 1..32 routes")
    VEC = int(vec_dwords)
    out_dwords = model_dim // 2  # dwords per output row (also the source row width)
    DWORDS_PER_ITER = BLOCK_THREADS * VEC  # dwords advanced per loop iter
    n_iters = (out_dwords + DWORDS_PER_ITER - 1) // DWORDS_PER_ITER

    module_name = format_kernel_name(
        f"moe_gather_reduce_{out_dtype}_d{model_dim}_tk{topk}_sk{split_k}_v{VEC}"
        f"_w{w_dtype}_{'frwave_coalesced' if wave_routes else 'frlds'}"
    )

    @flyc.kernel(name=module_name)
    def moe_gather_reduce_kernel(
        grouped_out_flat: fx.Pointer,
        topids_to_rows: fx.Pointer,
        gather_w: fx.Pointer,
        out: fx.Pointer,
        num_tokens: Int32,
        slice_stride_dw: Int32,
        num_valid_tokens: fx.Pointer,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        i32 = T.i32
        # Route-weight native dtype. "f32" lets the host pass raw fp32 route
        # weights straight through (no pre-cast); bf16/f16 get extended below.
        # (Ternary, not multi-line if: the flydsl tracer does not capture vars
        # bound in an if/elif block for the nested route-cache closure.)
        w_dt_fx = (
            fx.Float32
            if w_dtype == "f32"
            else (fx.BFloat16 if w_dtype == "bf16" else fx.Float16)
        )

        # Uint32 (not Int32): every index here is a non-negative count, so `<`
        # and `<=` lower to ult/ule.
        out_dwords_i32 = fx.Uint32(out_dwords)
        topk_i32 = fx.Uint32(topk)
        vec_i32 = fx.Uint32(VEC)
        num_tokens_i32 = fx.Uint32(num_tokens)
        bid_i32 = fx.Uint32(bid)

        num_valid_tokens_is_set = fx.Int64(ptrtoint(num_valid_tokens)) != 0
        valid_token_count = num_tokens_i32
        if num_valid_tokens_is_set:
            valid_token_count = fx.Uint32(ptr_buf_tensor(num_valid_tokens)[0])
        tok_valid = bid_i32 < valid_token_count
        if tok_valid:
            rows_t = ptr_buf_tensor(topids_to_rows)
            w_t = ptr_buf_tensor(gather_w, w_dt_fx)
            in_base_i64 = fx.Uint64(ptrtoint(grouped_out_flat))

            map_base = bid_i32 * topk_i32
            out_row_dw_base = bid_i32 * out_dwords_i32

            if fx.const_expr(not wave_routes):
                route_lds = fx.SharedAllocator().allocate(_GatherRouteLds).peek()
                rows_lds = route_lds.rows.ptr
                wbits_lds = route_lds.w_bits.ptr
                tid_u32 = fx.Uint32(tid)
                if tid_u32 < topk_i32:
                    map_off = map_base + tid_u32
                    # DROPPED_ROUTE_ROW (-1) is kept as-is: it owns no grouped row,
                    # and the zero-length descriptor below is what switches its
                    # loads off. Clamping it to a real row instead would read bytes
                    # stage2 need never have written, and a stale NaN there survives
                    # the multiply by the route's (zero) weight.
                    row_i32 = fx.Int32(rows_t[map_off])
                    _lds_si32(rows_lds, row_i32, tid)
                    # .to(Float32) is a no-op when the route weights are already f32.
                    w_f32 = w_dt_fx(w_t[map_off]).to(fx.Float32)
                    _lds_si32(wbits_lds, w_f32.bitcast(fx.Int32), tid)
                gpu.barrier()

            thread_id = fx.Uint32(tid)
            iter_idx_i32 = fx.Uint32(fx.block_idx.y)

            row_bytes_i32 = fx.Int32(out_dwords * 4)

            def _wave_route_cache():
                # One coalesced row/weight load per wave. Clamp the unused
                # lanes to an in-range route; no out-of-bounds metadata read.
                lane_route = fx.Uint32(tid) % fx.Uint32(32)
                lane_route = (lane_route < topk_i32).select(
                    lane_route, topk_i32 - fx.Uint32(1)
                )
                map_off = map_base + lane_route
                lane_row = fx.Int32(rows_t[map_off])
                lane_weight = w_dt_fx(w_t[map_off]).to(fx.Float32).bitcast(fx.Int32)
                # Broadcast while every lane is active, before the output-tail
                # branch, including routes held by lanes past a partial tail.
                rows = [
                    fx.Int32(fx.rocdl.readlane(T.i32, lane_row, k))
                    for k in range_constexpr(topk)
                ]
                weights = [
                    fx.Int32(fx.rocdl.readlane(T.i32, lane_weight, k)).bitcast(
                        fx.Float32
                    )
                    for k in range_constexpr(topk)
                ]
                return rows, weights

            wave_cache = _wave_route_cache() if wave_routes else None

            def _row_weight(k):
                return (
                    (wave_cache[0][k], wave_cache[1][k])
                    if wave_routes
                    else (
                        _lds_li32(rows_lds, k),
                        _lds_li32(wbits_lds, k).bitcast(fx.Float32),
                    )
                )

            # A descriptor per route rather than one for the whole tensor: the
            # buffer offset is 32-bit, so a flat descriptor stops addressing
            # once grouped_out passes 4 GiB, which at topk=6 is around 48k
            # tokens. Folding the row into the 64-bit base keeps the offset
            # inside one row. The base is loop-invariant, so this costs the
            # address math once per route, not once per access.
            def _row_rsrc(row_i32, sk):
                mapped = row_i32 >= fx.Int32(0)
                row_u64 = fx.Uint64(mapped.select(row_i32, fx.Int32(0)))
                off_dw = row_u64 * fx.Uint64(out_dwords)
                if sk != 0:
                    off_dw = off_dw + fx.Uint64(sk) * fx.Uint64(slice_stride_dw)
                return buffer_ops.create_buffer_resource_from_addr(
                    in_base_i64 + off_dw * fx.Uint64(4),
                    num_records_bytes=mapped.select(row_bytes_i32, fx.Int32(0)),
                )

            def load_flat(row_i32, sk, dw_off, vec_width):
                return buffer_ops.buffer_load(
                    _row_rsrc(row_i32, sk), dw_off, vec_width=vec_width, dtype=i32
                )

            dw_base = thread_id * vec_i32 + iter_idx_i32 * DWORDS_PER_ITER
            dw_valid = dw_base < out_dwords_i32
            if dw_valid:
                full_valid = dw_base + vec_i32 <= out_dwords_i32
                if full_valid:
                    acc = [fx.Float32(0.0) for _ in range(2 * VEC)]
                    vec_atom = buf_copy_atom(VEC * 4)
                    out_vec_t = ptr_buf_tensor(out, unit_elems=VEC, unit_stride=1)
                    frag = fx.make_fragment_like(fx.slice(out_vec_t, (0, None)))

                    for k in range_constexpr(topk):
                        row_i32, w_f32 = _row_weight(k)
                        red = [fx.Float32(0.0) for _ in range(2 * VEC)]
                        for sk in range_constexpr(split_k):
                            raw_vec = fx.Vector(load_flat(row_i32, sk, dw_base, VEC))
                            for lane in range_constexpr(VEC):
                                raw_dw = fx.Uint32(raw_vec[lane])
                                lo_f32, hi_f32 = _unpack_pair_to_f32(raw_dw, out_dtype)
                                red[2 * lane] = red[2 * lane] + lo_f32
                                red[2 * lane + 1] = red[2 * lane + 1] + hi_f32
                        for lane in range_constexpr(VEC):
                            acc[2 * lane] = acc[2 * lane] + w_f32 * red[2 * lane]
                            acc[2 * lane + 1] = (
                                acc[2 * lane + 1] + w_f32 * red[2 * lane + 1]
                            )
                    packed = [
                        _pack_pair_from_f32(acc[2 * lane], acc[2 * lane + 1], out_dtype)
                        for lane in range(VEC)
                    ]
                    fx.memref_store_vec(
                        fx.Vector.from_elements(packed, fx.Uint32), frag
                    )
                    fx.copy(
                        vec_atom,
                        frag,
                        fx.slice(out_vec_t, (out_row_dw_base + dw_base, None)),
                    )
                else:
                    out_t = ptr_buf_tensor(out)
                    for lane in range_constexpr(VEC):
                        dw_idx = dw_base + lane
                        lane_valid = dw_idx < out_dwords_i32
                        if lane_valid:
                            acc_lo = fx.Float32(0.0)
                            acc_hi = fx.Float32(0.0)
                            for k in range_constexpr(topk):
                                row_i32, w_f32 = _row_weight(k)
                                red_lo = fx.Float32(0.0)
                                red_hi = fx.Float32(0.0)
                                for sk in range_constexpr(split_k):
                                    raw_dw = fx.Uint32(
                                        load_flat(row_i32, sk, dw_idx, 1)
                                    )
                                    lo_f32, hi_f32 = _unpack_pair_to_f32(
                                        raw_dw, out_dtype
                                    )
                                    red_lo = red_lo + lo_f32
                                    red_hi = red_hi + hi_f32
                                acc_lo = acc_lo + w_f32 * red_lo
                                acc_hi = acc_hi + w_f32 * red_hi
                            out_t[out_row_dw_base + dw_idx] = _pack_pair_from_f32(
                                acc_lo, acc_hi, out_dtype
                            )

    @flyc.jit
    def launch_moe_gather_reduce(
        grouped_out_flat: fx.Pointer,
        topids_to_rows: fx.Pointer,
        gather_w: fx.Pointer,
        out: fx.Pointer,
        num_tokens: fx.Int32,
        slice_stride_dw: fx.Int32,
        num_valid_tokens: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        launcher = moe_gather_reduce_kernel(
            grouped_out_flat,
            topids_to_rows,
            gather_w,
            out,
            num_tokens,
            slice_stride_dw,
            num_valid_tokens,
        )
        launcher.launch(
            grid=(fx.Int64(num_tokens), n_iters, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_moe_gather_reduce.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_moe_gather_reduce

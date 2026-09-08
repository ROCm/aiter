# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE gather-reduce: LDS route cache + single flat in_rsrc dword-offset loads."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import Int32, T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.kernels_common import format_kernel_name
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
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
):
    assert model_dim % 2 == 0
    assert out_dtype in ("bf16", "f16")
    assert w_dtype in ("f32", "bf16", "f16")
    assert topk <= MAX_GATHER_TOPK
    if vec_dwords not in (2, 4, 8):
        raise ValueError(f"vec_dwords must be 2, 4, or 8, got {vec_dwords}")

    VEC = int(vec_dwords)
    out_dwords = model_dim // 2  # dwords per output row (also the source row width)
    DWORDS_PER_ITER = BLOCK_THREADS * VEC  # dwords advanced per loop iter
    n_iters = (out_dwords + DWORDS_PER_ITER - 1) // DWORDS_PER_ITER
    sk_i32 = fx.Int32(split_k)

    module_name = format_kernel_name(
        f"moe_gather_reduce_{out_dtype}_d{model_dim}_tk{topk}_sk{split_k}_v{VEC}"
        f"_w{w_dtype}_frlds"
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
        # bound in an if/elif block for the nested _load_row_weight closure.)
        w_dt = T.f32 if w_dtype == "f32" else (T.bf16 if w_dtype == "bf16" else T.f16)
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
        slice_stride_dw_i32 = fx.Uint32(slice_stride_dw)

        num_valid_tokens_is_set = fx.Int64(ptrtoint(num_valid_tokens)) != 0
        valid_token_count = num_tokens_i32
        if num_valid_tokens_is_set:
            valid_token_count = fx.Uint32(
                buffer_ops.buffer_load(
                    ptr_rsrc(num_valid_tokens), fx.Uint32(0), vec_width=1, dtype=i32
                )
            )
        tok_valid = bid_i32 < valid_token_count
        if tok_valid:
            rows_rsrc = ptr_rsrc(topids_to_rows)
            w_rsrc = ptr_rsrc(gather_w)
            out_rsrc = ptr_rsrc(out)
            in_base_i64 = fx.Uint64(ptrtoint(grouped_out_flat))

            map_base = bid_i32 * topk_i32
            out_row_dw_base = bid_i32 * out_dwords_i32

            route_lds = fx.SharedAllocator().allocate(_GatherRouteLds).peek()
            rows_lds = route_lds.rows.ptr
            wbits_lds = route_lds.w_bits.ptr
            tid_u32 = fx.Uint32(tid)
            if tid_u32 < topk_i32:
                map_off = map_base + tid_u32
                raw_row = fx.Int32(
                    buffer_ops.buffer_load(rows_rsrc, map_off, vec_width=1, dtype=i32)
                )
                is_mapped = raw_row >= fx.Int32(0)
                row_i32 = is_mapped.select(raw_row, fx.Int32(0))
                _lds_si32(rows_lds, row_i32, tid)
                w_loaded = buffer_ops.buffer_load(w_rsrc, map_off, vec_width=1, dtype=w_dt)
                w_f32 = w_dt_fx(w_loaded).to(fx.Float32)
                _lds_si32(wbits_lds, w_f32.bitcast(fx.Int32), tid)
            gpu.barrier()

            flat_bytes = fx.Int32(slice_stride_dw) * sk_i32 * fx.Int32(4)
            in_rsrc = buffer_ops.create_buffer_resource_from_addr(
                in_base_i64, num_records_bytes=flat_bytes
            )

            thread_id = fx.Uint32(tid)
            iter_idx_i32 = fx.Uint32(fx.block_idx.y)

            def _row_weight(k):
                row_u32 = fx.Uint32(_lds_li32(rows_lds, k))
                w_f32 = _lds_li32(wbits_lds, k).bitcast(fx.Float32)
                return row_u32, w_f32

            def load_flat(row_u32, sk, dw_off):
                off_dw = row_u32 * out_dwords_i32 + dw_off
                if sk != 0:
                    off_dw = off_dw + sk * slice_stride_dw_i32
                return buffer_ops.buffer_load(
                    in_rsrc, off_dw, vec_width=VEC, dtype=i32
                )

            def load_flat_dw(row_u32, sk, dw_off):
                off_dw = row_u32 * out_dwords_i32 + dw_off
                if sk != 0:
                    off_dw = off_dw + sk * slice_stride_dw_i32
                return fx.Uint32(
                    buffer_ops.buffer_load(
                        in_rsrc, off_dw, vec_width=1, dtype=i32
                    )
                )

            dw_base = thread_id * vec_i32 + iter_idx_i32 * DWORDS_PER_ITER
            dw_valid = dw_base < out_dwords_i32
            if dw_valid:
                full_valid = dw_base + vec_i32 <= out_dwords_i32
                if full_valid:
                    acc = [fx.Float32(0.0) for _ in range(2 * VEC)]
                    for k in range_constexpr(topk):
                        row_u32, w_f32 = _row_weight(k)
                        red = [fx.Float32(0.0) for _ in range(2 * VEC)]
                        for sk in range_constexpr(split_k):
                            raw_vec = load_flat(row_u32, sk, dw_base)
                            for lane in range_constexpr(VEC):
                                raw_dw = fx.Uint32(fx.Vector(raw_vec)[lane])
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
                    out_vec = fx.Vector.from_elements(packed, fx.Uint32)
                    buffer_ops.buffer_store(
                        out_vec, out_rsrc, out_row_dw_base + dw_base
                    )
                else:
                    for lane in range_constexpr(VEC):
                        dw_idx = dw_base + lane
                        lane_valid = dw_idx < out_dwords_i32
                        if lane_valid:
                            acc_lo = fx.Float32(0.0)
                            acc_hi = fx.Float32(0.0)
                            for k in range_constexpr(topk):
                                row_u32, w_f32 = _row_weight(k)
                                red_lo = fx.Float32(0.0)
                                red_hi = fx.Float32(0.0)
                                for sk in range_constexpr(split_k):
                                    raw_dw = load_flat_dw(row_u32, sk, dw_idx)
                                    lo_f32, hi_f32 = _unpack_pair_to_f32(
                                        raw_dw, out_dtype
                                    )
                                    red_lo = red_lo + lo_f32
                                    red_hi = red_hi + hi_f32
                                acc_lo = acc_lo + w_f32 * red_lo
                                acc_hi = acc_hi + w_f32 * red_hi
                            packed = _pack_pair_from_f32(acc_lo, acc_hi, out_dtype)
                            buffer_ops.buffer_store(
                                packed, out_rsrc, out_row_dw_base + dw_idx
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

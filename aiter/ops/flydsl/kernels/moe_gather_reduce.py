# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-pass MoE gather-reduce (weighted) epilogue kernel (FlyDSL).

Background
----------
After the per-expert stage2 GEMM, MoE output lives in a grouped layout
``grouped_out (E, max_m, model_dim)``: expert ``e`` holds its routed tokens in
rows ``[0, counts[e])``.  The final epilogue scatters those rows back to the
flat per-token output, multiplying by the route weight and summing the ``topk``
contributions of each token::

    moe_out[t] = sum_k  w(t,k) * grouped_out[expert(t,k), pos(t,k)]

The Python reference does this as a per-expert loop of ``index_add_`` (scatter).
This kernel reformulates it as a **gather-reduce**: one thread-block per output
token gathers that token's ``topk`` source rows (via a precomputed inverse index
map), weights them, and sums them in registers in a single pass.  No atomics, so
the route and split-K accumulation order is fixed for each output element.

Layout / grid
-------------
Inputs (all on device):
  grouped_out_flat : (E*max_m, model_dim)  bf16/f16   -- grouped_out viewed flat
  topids_to_rows         : (token_num, topk)     i32        -- flat source row per (t,k)
  gather_w         : (token_num, topk)     f32        -- weight per (t,k)
  out              : (token_num, model_dim) bf16/f16

Grid  : (token_num, ceil((model_dim/2) / (BLOCK_THREADS*vec_dwords)), 1)
Block : (BLOCK_THREADS, 1, 1)

Each thread owns ``vec_dwords`` consecutive dwords (2, 4, or 8). Buffer
transactions are at most 128 bits: vec2 uses one 64-bit access, vec4 one
128-bit access, and vec8 two 128-bit accesses. A partially valid thread span
uses scalar dword accesses, so any even ``model_dim`` is supported.

The first ``topk`` threads cache route rows and weights in LDS for the block.
Each route's source descriptor rebases the row address in 64 bits; its byte
offset only spans one row, including when grouped storage exceeds 4 GiB.
Dropped routes use zero-length descriptors, returning zero without reading a
potentially uninitialized row zero. The optional valid-token bound leaves
output tokens beyond that bound unwritten.

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
):
    assert model_dim % 2 == 0
    assert out_dtype in ("bf16", "f16")
    assert w_dtype in ("f32", "bf16", "f16")
    assert topk <= MAX_GATHER_TOPK
    if vec_dwords not in (2, 4, 8):
        raise ValueError(f"vec_dwords must be 2, 4, or 8, got {vec_dwords}")

    VEC = int(vec_dwords)
    ACCESS = min(VEC, 4)  # Hardware buffer copies are at most 16 bytes.
    out_dwords = model_dim // 2  # dwords per output row (also the source row width)
    DWORDS_PER_ITER = BLOCK_THREADS * VEC  # dwords advanced per loop iter
    n_iters = (out_dwords + DWORDS_PER_ITER - 1) // DWORDS_PER_ITER

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

            def _row_weight(k):
                row_i32 = _lds_li32(rows_lds, k)
                w_f32 = _lds_li32(wbits_lds, k).bitcast(fx.Float32)
                return row_i32, w_f32

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
                    # vec8 retains eight dwords per thread, but issues two
                    # legal 128-bit accesses. The per-element reduction order
                    # is unchanged; vec2/vec4 use one compile-time chunk.
                    for chunk in range_constexpr(0, VEC, ACCESS):
                        acc = [fx.Float32(0.0) for _ in range(2 * ACCESS)]
                        vec_atom = buf_copy_atom(ACCESS * 4)
                        out_vec_t = ptr_buf_tensor(
                            out, unit_elems=ACCESS, unit_stride=1
                        )
                        frag = fx.make_fragment_like(fx.slice(out_vec_t, (0, None)))

                        for k in range_constexpr(topk):
                            row_i32, w_f32 = _row_weight(k)
                            red = [fx.Float32(0.0) for _ in range(2 * ACCESS)]
                            for sk in range_constexpr(split_k):
                                raw_vec = fx.Vector(
                                    load_flat(row_i32, sk, dw_base + chunk, ACCESS)
                                )
                                for lane in range_constexpr(ACCESS):
                                    raw_dw = fx.Uint32(raw_vec[lane])
                                    lo_f32, hi_f32 = _unpack_pair_to_f32(
                                        raw_dw, out_dtype
                                    )
                                    red[2 * lane] = red[2 * lane] + lo_f32
                                    red[2 * lane + 1] = red[2 * lane + 1] + hi_f32
                            for lane in range_constexpr(ACCESS):
                                acc[2 * lane] = acc[2 * lane] + w_f32 * red[2 * lane]
                                acc[2 * lane + 1] = (
                                    acc[2 * lane + 1] + w_f32 * red[2 * lane + 1]
                                )
                        packed = [
                            _pack_pair_from_f32(
                                acc[2 * lane], acc[2 * lane + 1], out_dtype
                            )
                            for lane in range(ACCESS)
                        ]
                        fx.memref_store_vec(
                            fx.Vector.from_elements(packed, fx.Uint32), frag
                        )
                        fx.copy(
                            vec_atom,
                            frag,
                            fx.slice(
                                out_vec_t, (out_row_dw_base + dw_base + chunk, None)
                            ),
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

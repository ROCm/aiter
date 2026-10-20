# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors

"""Wave32 FP8/BF16 WMMA backend for planned paged decode on gfx1250.

Four waves split 64- or 128-token compute tiles within each 256-token plan tile.
FP8 D128/D256/page128+ full attention uses 128-token tiles and K128 QK/PV WMMA.
FP8 D128/D256 transposed caches with one query tile, and
D256/G8/MTP4/page128 full attention with two query tiles, keep their packed
layout without TDM transposition or row padding. D256/page128 full attention
uses two buffers to overlap memory transfers, with asynchronous scale copies
for per-token quantization.
WMMA accumulates into a rescaled running output; empty tiles retain its scale.
Other TDM/vector DMA paths alternate LDS buffers where capacity permits.
FP8 Q and P use native eight-element conversion; BF16 uses K32 WMMA.
BF16 TDM is enabled for power-of-two D64–512, with vector DMA otherwise.
Softmax statistics and accumulation stay FP32.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from . import implementation_cache_tag
from .pipeline_gfx1250 import Gfx1250Pipeline
from .traits_gfx1250 import Gfx1250Traits

_CACHE = {}


def compile_gfx1250_pa_decode(
    *,
    head_dim,
    query_group_size,
    block_size,
    softmax_scale,
    query_dtype,
    per_token_kv,
    query_length,
    trans_v,
    sliding_window,
    compute_dtype="fp8",
    direct_output=False,
    query_splits=1,
):
    if query_splits not in (1, 2, 4) or query_length % query_splits:
        raise ValueError("query_splits must be 1, 2, or 4 and divide query_length")
    t = Gfx1250Traits(
        head_dim,
        query_group_size,
        block_size,
        head_dim**-0.5 if softmax_scale is None else softmax_scale,
        query_dtype,
        per_token_kv,
        query_length,
        trans_v,
        sliding_window,
        compute_dtype,
        direct_output,
        query_splits,
    )
    if compute_dtype not in ("fp8", "bf16"):
        raise ValueError(f"Unsupported gfx1250 PA compute dtype: {compute_dtype}")
    if t.bf16 and per_token_kv:
        raise ValueError("BF16 PA decode does not use quantization scales")
    if t.total_bytes > 327680:
        raise NotImplementedError(
            "gfx1250 PA decode query/KV tiles exceed LDS capacity"
        )
    if t in _CACHE:
        return _CACHE[t]
    tag = ("gfx1250", tuple(t.__dict__.values()), implementation_cache_tag())

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, t.total_bytes // 4, 16]

    # Reuse the argument context and packed task contract of the CDNA backend.
    from .context import PaDecodeContext

    @flyc.kernel(known_block_size=(128, 1, 1))
    def kernel(
        pmax: fx.Pointer,
        psum: fx.Pointer,
        pout: fx.Pointer,
        query: fx.Pointer,
        key: fx.Pointer,
        value: fx.Pointer,
        table: fx.Pointer,
        ks: fx.Pointer,
        vs: fx.Pointer,
        max_blocks: fx.Int32,
        sk_block: fx.Int32,
        sk_head: fx.Int32,
        sq_row: fx.Int32,
        sq_head: fx.Int32,
        work: fx.Pointer,
        sequences: fx.Int32,
    ):
        _ = tag
        task_slot = fx.Int32(gpu.block_id("x"))
        if const_expr(t.direct_output):
            # One task per nonempty sequence; reduce_info supplies its packed slot.
            info = fx.recast_iter(fx.Int32, pmax)
            count = fx.Int32(info[task_slot * 2 + 1])
            # A trailing empty sequence's start can equal capacity. Its dummy
            # task read must stay in bounds even though no compute consumes it.
            task_slot = (count > 0).select(fx.Int32(info[task_slot * 2]), fx.Int32(0))
        task = fx.ptr_load(
            fx.add_offset(fx.recast_iter(fx.Int32, work), task_slot * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
        active = fx.Int32(task[1]) < fx.Int32(task[2])
        if const_expr(t.direct_output):
            active = count > 0
        if active:
            ctx = PaDecodeContext(
                t,
                SharedStorage,
                pmax,
                psum,
                pout,
                query,
                key,
                value,
                table,
                ks,
                vs,
                max_blocks,
                sk_block,
                sk_head,
                sq_row,
                sq_head,
                sequences,
                fx.Int32(task[0]),
                fx.Int32(task[1]),
                fx.Int32(task[2]),
                fx.Int32(task[3]),
            )
            Gfx1250Pipeline(ctx).run()
        elif const_expr(t.direct_output):
            output = fx.recast_iter(t.q_dtype, pout)
            tid = fx.Int32(gpu.thread_id("x"))
            for i in range_constexpr((t.rows * t.head_dim + 127) // 128):
                index = i * 128 + tid
                if index < t.rows * t.head_dim:
                    row, d = index // t.head_dim, index % t.head_dim
                    kv_query = fx.Int32(gpu.block_id("y"))
                    row = row + (kv_query % t.query_splits) * t.rows
                    offset = (
                        (
                            fx.Int64(gpu.block_id("x")) * t.query_length
                            + row // t.query_group_size
                        )
                        * sq_row
                        + (
                            (kv_query // t.query_splits) * t.query_group_size
                            + row % t.query_group_size
                        )
                        * sq_head
                        + d
                    )
                    output[offset] = t.q_dtype(0.0)

    @flyc.jit
    def launch(
        pmax: fx.Pointer,
        psum: fx.Pointer,
        pout: fx.Pointer,
        query: fx.Pointer,
        key: fx.Pointer,
        value: fx.Pointer,
        table: fx.Pointer,
        ks: fx.Pointer,
        vs: fx.Pointer,
        max_blocks: fx.Int32,
        sequences: fx.Int32,
        kv_heads: fx.Int32,
        sk_block: fx.Int32,
        sk_head: fx.Int32,
        sq_row: fx.Int32,
        sq_head: fx.Int32,
        work: fx.Pointer,
        capacity: fx.Int32,
        stream: fx.Stream,
    ):
        _ = tag
        kernel(
            pmax,
            psum,
            pout,
            query,
            key,
            value,
            table,
            ks,
            vs,
            max_blocks,
            sk_block,
            sk_head,
            sq_row,
            sq_head,
            work,
            sequences,
        ).launch(
            grid=(
                sequences if t.direct_output else capacity,
                kv_heads * t.query_splits,
                1,
            ),
            block=(128, 1, 1),
            stream=stream,
        )

    compiled = {"launch": launch, "kernel": kernel}
    return _CACHE.setdefault(t, compiled)

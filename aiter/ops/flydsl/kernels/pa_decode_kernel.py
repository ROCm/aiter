# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""FP8 and BF16 paged-attention tile kernel.

K/V use e4m3 (FNUZ on gfx942, OCP on gfx950/gfx1250); BF16/FP16 Q and probabilities P
are quantized to FP8. Q/key scales fold into QK, value scale and 1/FP8_MAX into
the epilogue; softmax max/sum stay f32. Tuned gfx950 BF16 per-token MTP3/MTP4
and the head-256 page-16/64 pipe use K128 MFMA instead of K32, preserving normalized
Q/P and operand layouts.
Every FP8-path Q row uses absmax normalization; per-tensor and per-token K/V scales
retain range normalization for P.
BF16 K/V use vector-8 caches and native BF16 Q/K/P/V MFMA operands, with
f32 softmax state and accumulation. BF16 KV does not use quantization scales.

gfx1250 uses wave32 WMMA with native FP8 Q/P conversion or BF16 operands.
Its four waves process 64- or 128-token compute tiles within each 256-token
plan tile. Packed FP8 D128/D256 full-attention paths use K128 WMMA; other
FP8 paths use K64 and BF16 uses K32. Supported layouts use TDM K/V transfers,
with two LDS buffers where capacity permits. One-partition plans can write
final output directly; other plans use the shared partition reducer.
BF16 transposed page128 caches at D64/D128/D256 use a packed-page pipeline
with resident Q and fused PV accumulation; D256 is limited to 32 query rows.

Logical layouts (not preshuffled):

* ``query``        [num_seqs * query_length, num_q_heads, head_dim]  f16/bf16
* ``key_cache``    [num_blocks, num_kv_heads, head_dim//x, block_size, x]
* ``value_cache``  [num_blocks, num_kv_heads, block_size//x, head_dim, x] (trans_v)
                   or [num_blocks, num_kv_heads, head_dim, block_size] (FP8 plain)
                   where x=16 for FP8 and x=8 for BF16
* ``block_tables`` [num_seqs, max_blocks_per_seq]  int32
* ``work_info``    [capacity, 4]  int32 sequence/tile bounds/context length
* ``pmax``, ``psum`` [num_kv_heads, capacity, query_length * query_group_size] f32
* ``pout``         [num_kv_heads, capacity, query_length * query_group_size, head_dim]
                   same dtype as query; normalized partition output
* K/V scales      [1] per-tensor or [num_blocks, num_kv_heads, block_size] per-token

Four-wave CTAs process 256-token blocks: QK splits tokens, PV splits head dim,
and P passes through LDS to transpose ownership between the MMAs.

The ``pa_decode`` package separates schedule/layout traits, CTA context, Q/K/V
and LDS movement, MFMA/WMMA, softmax and output operations.
``PaDecodePipeline`` and ``Gfx1250Pipeline`` own prefetch ordering and barriers;
this module keeps the compilation API and launch wrappers.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr

from aiter.jit.utils.chip_info import get_gfx_runtime

from .pa_decode import implementation_cache_tag
from .pa_decode.context import PaDecodeContext
from .pa_decode.pipeline import Gfx1250Bf16Pipeline, Gfx1250Pipeline, PaDecodePipeline
from .pa_decode.traits import (
    FP8_PACK_K,
    KV_COMPUTE_BLOCK,
    LOG2E,
    MFMA_ACC_ELEMS,
    MFMA_MNK,
    WAVE,
    Gfx1250Traits,
    PaDecodeSchedule,
    PaDecodeTraits,
)

__all__ = [
    "FP8_PACK_K",
    "KV_COMPUTE_BLOCK",
    "LOG2E",
    "MFMA_ACC_ELEMS",
    "MFMA_MNK",
    "WAVE",
    "compile_pa_decode_tile",
]

# Key by selected specialization, not batch/head/CU scheduling inputs.
_PA_DECODE_TILE_CACHE = {}
_GFX1250_PA_DECODE_CACHE = {}


def compile_pa_decode_tile(
    *,
    architecture: str | None = None,
    head_dim: int,
    query_group_size: int,
    block_size: int,
    num_seqs: int,
    num_kv_heads: int,
    num_compute_units: int,
    work_capacity: int,
    num_partitions: int = 1,
    softmax_scale: float | None = None,
    query_dtype: str = "f16",
    kv_dtype: str = "fp8",
    per_token_kv: bool = False,
    query_length: int = 1,
    trans_v: bool = True,
    wide_kv_addressing: bool = False,
    kv_buffer_u32: bool = False,
    query_splits: int | None = None,
    max_context_length: int | None = None,
    sliding_window: int = 0,
    direct_output: bool = False,
    trans_k: bool = True,
):
    """Select and cache a PA-decode kernel and launch wrapper.

    ``query_splits=None`` selects from host-known grid bounds; an explicit
    count overrides splitting and selects the matching prefetch policy.
    BF16 also selects the smallest query-length divisor that fits 64 KiB LDS;
    for example D512/MTP4/GQA16 splits twice and D1024/MTP3/GQA16 splits three
    ways. Explicit BF16 splits must fit that limit as well.
    ``max_context_length`` bounds dense planned work for scheduling only; it does not
    change the launch capacity, scratch layout, or single-tile guarantee. Every
    context length, including later graph replays, should be at most this value.
    A smaller value still covers the real context: page ids outside the staged
    window are read from the block table.
    CTAs receive equal groups of MTP positions, flattened with GQA into 16-row
    M-tiles. Partial output stays unsplit for the shared reducer.

    Single-tile specialization requires a standard plan whose capacity proves
    one KV tile per active task; padding skips the task body. Selected plans
    map packed slots over (B, C/B), taking sequence IDs from plan records.
    Launch B must be positive, divide capacity C, and match schedule selection;
    the JIT launch does not recheck these conditions. B and C/B are not cache keys.
    Planned gfx950 BF16 D128 output uses slot-rebased buffer stores when the
    slot's byte span fits signed i32, retaining the dtype's 2-byte alignment.

    ``work_capacity`` is a positive host integer matching the work plan.
    Positive ``sliding_window`` includes the query token.
    Plans cover the MTP window union; scores are masked per query row.
    ``trans_k=False`` is not implemented.
    gfx1250 uses wave32 WMMA, native BF16 or OCP FP8, and its larger LDS budget.
    A one-partition gfx1250 plan can write final output directly when sinks
    are absent and query/output strides match. Other tasks write packed partials
    for the shared reducer, which applies sinks once.

    Masked V bytes must remain finite because ``0 * NaN == NaN`` in PV MFMA.
    Pages past the sequence are pinned to block 0; callers must leave the
    unwritten tail of the last owned page finite. ``wide_kv_addressing`` protects
    offsets at 2 GiB; ``kv_buffer_u32`` proves both FP8 caches are below 4 GiB
    and permits unsigned buffer offsets instead of i64.
    """
    if not trans_k:
        raise NotImplementedError("Non-transposed key cache is not implemented")
    architecture = (architecture or get_gfx_runtime()).split(":", 1)[0]
    if architecture == "gfx1250":
        if direct_output and num_partitions != 1:
            raise ValueError("direct_output requires a one-partition plan")
        if query_splits is None:
            # Split the measured spilling BF16 specialization into one Q tile
            # per CTA. Other shapes retain fused queries and their KV reuse.
            query_splits = (
                4
                if kv_dtype == "bf16"
                and head_dim == 1024
                and query_group_size == 16
                and query_length == 4
                and block_size == 128
                and trans_v
                and sliding_window == 0
                else 1
            )
        return compile_gfx1250_pa_decode(
            head_dim=head_dim,
            query_group_size=query_group_size,
            block_size=block_size,
            softmax_scale=softmax_scale,
            query_dtype=query_dtype,
            per_token_kv=per_token_kv,
            query_length=query_length,
            trans_v=trans_v,
            sliding_window=sliding_window,
            compute_dtype=kv_dtype,
            direct_output=direct_output,
            query_splits=query_splits,
        )
    if architecture not in ("gfx942", "gfx950"):
        raise NotImplementedError(f"PA decode does not support {architecture}")
    if direct_output:
        raise NotImplementedError("direct_output requires gfx1250")
    schedule = PaDecodeSchedule.select(
        head_dim=head_dim,
        query_group_size=query_group_size,
        block_size=block_size,
        num_seqs=num_seqs,
        num_kv_heads=num_kv_heads,
        num_compute_units=num_compute_units,
        num_partitions=num_partitions,
        softmax_scale=softmax_scale,
        query_dtype=query_dtype,
        kv_dtype=kv_dtype,
        per_token_kv=per_token_kv,
        query_length=query_length,
        trans_v=trans_v,
        wide_kv_addressing=wide_kv_addressing,
        kv_buffer_u32=kv_buffer_u32,
        query_splits=query_splits,
        work_capacity=work_capacity,
        max_context_length=max_context_length,
        sliding_window=sliding_window,
        trans_k=trans_k,
    )
    cached = _PA_DECODE_TILE_CACHE.get(schedule.cache_key)
    if cached is not None:
        return cached

    traits = PaDecodeTraits.create(schedule)
    # As in fmha_gfx950, expose scalar specialization data to FlyDSL's closure
    # cache walker: it cannot inspect the fields of a captured traits object.
    _pa_decode_cache_tag = (traits.cache_key, implementation_cache_tag())

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, traits.total_bytes // 4, 16]

    @flyc.jit
    def _pa_decode_tile_task(
        # Packed partials: [KV heads, capacity, query rows].
        pmax_ptr: fx.Pointer,  # Natural-log row max.
        psum_ptr: fx.Pointer,  # Row sum.
        pout_ptr: fx.Pointer,  # Adds head_dim; Q_DTYPE normalized O_p/l_p.
        query_ptr: fx.Pointer,
        key_cache_ptr: fx.Pointer,
        value_cache_ptr: fx.Pointer,
        block_tables_ptr: fx.Pointer,
        key_scale_ptr: fx.Pointer,
        value_scale_ptr: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        num_sequences: fx.Int32,
        planned_seq: fx.Int32,
        planned_start: fx.Int32,
        planned_end: fx.Int32,
        planned_context: fx.Int32,
    ):
        _ = _pa_decode_cache_tag
        ctx = PaDecodeContext(
            traits,
            SharedStorage,
            pmax_ptr,
            psum_ptr,
            pout_ptr,
            query_ptr,
            key_cache_ptr,
            value_cache_ptr,
            block_tables_ptr,
            key_scale_ptr,
            value_scale_ptr,
            max_blocks_per_seq,
            stride_ks_block,
            stride_ks_head,
            stride_q_row,
            stride_q_head,
            num_sequences,
            planned_seq,
            planned_start,
            planned_end,
            planned_context,
        )
        PaDecodePipeline(ctx).run()

    @flyc.kernel(known_block_size=(traits.BLOCK_THREADS, 1, 1))
    def pa_decode_tile_kernel(
        pmax_ptr: fx.Pointer,
        psum_ptr: fx.Pointer,
        pout_ptr: fx.Pointer,
        query_ptr: fx.Pointer,
        key_cache_ptr: fx.Pointer,
        value_cache_ptr: fx.Pointer,
        block_tables_ptr: fx.Pointer,
        key_scale_ptr: fx.Pointer,
        value_scale_ptr: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        work_info_ptr: fx.Pointer,
        num_sequences: fx.Int32,
    ):
        def _run_task(seq, start, end, context):
            _pa_decode_tile_task(
                pmax_ptr,
                psum_ptr,
                pout_ptr,
                query_ptr,
                key_cache_ptr,
                value_cache_ptr,
                block_tables_ptr,
                key_scale_ptr,
                value_scale_ptr,
                max_blocks_per_seq,
                stride_ks_block,
                stride_ks_head,
                stride_q_row,
                stride_q_head,
                num_sequences,
                seq,
                start,
                end,
                context,
            )

        # Skip cleared padding records: no scratch writes or Q loads.
        # This CTA-uniform guard keeps every task barrier convergent.
        if const_expr(traits.batch_first_plan_grid):
            slot = fx.Int32(
                fx.Uint32(gpu.block_id("x")) * fx.Uint32(gpu.grid_dim.z)
                + fx.Uint32(gpu.block_id("z"))
            )
        else:
            slot = fx.Int32(gpu.block_id("x"))
        work = fx.recast_iter(fx.Int32, work_info_ptr)
        task = fx.ptr_load(
            fx.add_offset(work, slot * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
        start = fx.Int32(task[1])
        end = fx.Int32(task[2])
        if start < end:
            _run_task(fx.Int32(task[0]), start, end, fx.Int32(task[3]))

    @flyc.jit
    def pa_decode_tile_launch(
        pmax: fx.Pointer,
        psum: fx.Pointer,
        pout: fx.Pointer,
        query: fx.Pointer,
        key_cache: fx.Pointer,
        value_cache: fx.Pointer,
        block_tables: fx.Pointer,
        key_scale: fx.Pointer,
        value_scale: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        num_seqs: fx.Int32,
        num_kv_heads: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        work_info: fx.Pointer,
        work_capacity: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = _pa_decode_cache_tag
        # Ambient contract permits FMAs; explicit per-op fastmath still wins.
        with CompilationContext.compile_hints({"fastmath": "contract"}):
            pa_decode_tile_kernel(
                pmax,
                psum,
                pout,
                query,
                key_cache,
                value_cache,
                block_tables,
                key_scale,
                value_scale,
                max_blocks_per_seq,
                stride_ks_block,
                stride_ks_head,
                stride_q_row,
                stride_q_head,
                work_info,
                num_seqs,
            ).launch(
                grid=(
                    (
                        num_seqs,
                        num_kv_heads * traits.query_splits,
                        work_capacity // num_seqs,
                    )
                    if traits.batch_first_plan_grid
                    else (
                        work_capacity,
                        num_kv_heads * traits.query_splits,
                        1,
                    )
                ),
                block=(traits.BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    compiled = {
        "launch": pa_decode_tile_launch,
        "kernel": pa_decode_tile_kernel,
    }
    return _PA_DECODE_TILE_CACHE.setdefault(traits.cache_key, compiled)


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
    if t in _GFX1250_PA_DECODE_CACHE:
        return _GFX1250_PA_DECODE_CACHE[t]
    tag = ("gfx1250", tuple(t.__dict__.values()), implementation_cache_tag())

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, t.total_bytes // 4, 16]

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
            if const_expr(t.bf16_packed):
                Gfx1250Bf16Pipeline(ctx).run()
            else:
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
    return _GFX1250_PA_DECODE_CACHE.setdefault(t, compiled)

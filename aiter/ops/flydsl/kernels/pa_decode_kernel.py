# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""FP8 paged-attention tile kernel.

K/V use e4m3 (FNUZ on gfx942, OCP on gfx950); BF16/FP16 Q and probabilities P
are quantized to FP8. Q/key scales fold into QK, value scale and 1/FP8_MAX into
the epilogue; softmax max/sum stay f32. Tuned gfx950 BF16 per-token MTP3/MTP4
uses K128 MFMA instead of K32, preserving normalized Q/P and operand layouts.
Tuned gfx950 BF16 scalar decode casts Q/P directly, without normalization or
1/FP8_MAX compensation: Q must fit FP8, and small Q/probabilities may underflow.
Per-token scales retain range normalization.

Logical layouts (not preshuffled):

* ``query``        [num_seqs, num_q_heads, head_dim]  f16/bf16 (head_dim contiguous)
* ``key_cache``    [num_blocks, num_kv_heads, head_dim//16, block_size, 16]  fp8
* ``value_cache``  [num_blocks, num_kv_heads, block_size//16, head_dim, 16] (trans_v)
                   or [num_blocks, num_kv_heads, head_dim, block_size] (plain), by rank
* ``block_tables`` [num_seqs, max_blocks_per_seq]  int32
* ``context_lengths`` [num_seqs]  int32
* ``output``       [num_seqs, num_q_heads, head_dim]  same dtype as query
* K/V scales      [1] per-tensor or [num_blocks, num_kv_heads, block_size] per-token

Four-wave CTAs process 256-token blocks: QK splits tokens, PV splits head dim,
and P passes through LDS to transpose ownership between the MMAs.

The ``pa_decode`` package separates schedule/layout traits, CTA context, Q/K/V
and LDS movement, MFMA, softmax and output operations. ``PaDecodePipeline``
owns prefetch ordering and barriers; this module keeps the compilation API
and launch wrappers.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu

from .pa_decode import implementation_cache_tag
from .pa_decode.context import PaDecodeContext
from .pa_decode.pipeline import PaDecodePipeline
from .pa_decode.traits import (
    FP8_PACK_K,
    KV_COMPUTE_BLOCK,
    LOG2E,
    MFMA_ACC_ELEMS,
    MFMA_MNK,
    WAVE,
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


def compile_pa_decode_tile(
    *,
    head_dim: int,
    query_group_size: int,
    block_size: int,
    num_seqs: int,
    num_kv_heads: int,
    num_compute_units: int,
    num_partitions: int = 1,
    softmax_scale: float | None = None,
    query_dtype: str = "f16",
    per_token_kv: bool = False,
    query_length: int = 1,
    trans_v: bool = True,
    wide_kv_addressing: bool = False,
    kv_buffer_u32: bool = False,
    query_splits: int | None = None,
    use_work_plan: bool = False,
    work_capacity: int | None = None,
    max_context_length: int | None = None,
    sliding_window: int = 0,
    use_sinks: bool = False,
    sink_dtype_str: str = "f32",
):
    """Select and cache a PA-decode kernel and launch wrapper.

    ``query_splits=None`` selects from host-known grid bounds; an explicit
    count overrides splitting and selects the matching prefetch policy.
    ``max_context_length`` bounds dense planned work for scheduling only; it does not
    change the launch capacity, scratch layout, or single-tile guarantee.
    CTAs receive equal groups of MTP positions, flattened with GQA into 16-row
    M-tiles. Partial output stays unsplit for the shared reducer.

    Single-tile specialization requires a standard plan whose capacity proves
    one KV tile per active task; padding skips the task body. Selected plans
    map packed slots over (B, C/B), taking sequence IDs from plan records.
    Launch B must be positive, divide capacity C, and match schedule selection;
    the JIT launch does not recheck these conditions. B and C/B are not cache keys.
    Planned gfx950 BF16 D128 output uses slot-rebased buffer stores when the
    slot's byte span fits signed i32, retaining the dtype's 2-byte alignment.

    Positive ``sliding_window`` requires a plan and includes the query token.
    Plans cover the MTP window union; scores are masked per query row.
    ``use_sinks`` adds a zero-value per-head logit only to direct NP=1 output;
    partitioned/planned output adds it once in the reducer, not in partials.

    Masked V bytes must remain finite because ``0 * NaN == NaN`` in PV MFMA.
    Pages past the sequence are pinned to block 0; callers must leave the
    unwritten tail of the last owned page finite. ``wide_kv_addressing`` protects
    offsets at 2 GiB; ``kv_buffer_u32`` proves both FP8 caches are below 4 GiB
    and permits unsigned buffer offsets instead of i64.
    """
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
        per_token_kv=per_token_kv,
        query_length=query_length,
        trans_v=trans_v,
        wide_kv_addressing=wide_kv_addressing,
        kv_buffer_u32=kv_buffer_u32,
        query_splits=query_splits,
        use_work_plan=use_work_plan,
        work_capacity=work_capacity,
        max_context_length=max_context_length,
        sliding_window=sliding_window,
        use_sinks=use_sinks,
        sink_dtype_str=sink_dtype_str,
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
        output_ptr: fx.Pointer,  # Direct static NP=1 output.
        # Static partials: [B,H,NP,rows]; planned: [H,capacity,rows].
        pmax_ptr: fx.Pointer,  # Natural-log row max.
        psum_ptr: fx.Pointer,  # Row sum.
        pout_ptr: fx.Pointer,  # Adds head_dim; Q_DTYPE normalized O_p/l_p.
        query_ptr: fx.Pointer,
        key_cache_ptr: fx.Pointer,
        value_cache_ptr: fx.Pointer,
        block_tables_ptr: fx.Pointer,
        context_lengths_ptr: fx.Pointer,
        key_scale_ptr: fx.Pointer,
        value_scale_ptr: fx.Pointer,
        sinks_ptr: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_o_row: fx.Int32,
        stride_o_head: fx.Int32,
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
            output_ptr,
            pmax_ptr,
            psum_ptr,
            pout_ptr,
            query_ptr,
            key_cache_ptr,
            value_cache_ptr,
            block_tables_ptr,
            context_lengths_ptr,
            key_scale_ptr,
            value_scale_ptr,
            sinks_ptr,
            max_blocks_per_seq,
            stride_ks_block,
            stride_ks_head,
            stride_o_row,
            stride_o_head,
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
        output_ptr: fx.Pointer,
        pmax_ptr: fx.Pointer,
        psum_ptr: fx.Pointer,
        pout_ptr: fx.Pointer,
        query_ptr: fx.Pointer,
        key_cache_ptr: fx.Pointer,
        value_cache_ptr: fx.Pointer,
        block_tables_ptr: fx.Pointer,
        context_lengths_ptr: fx.Pointer,
        key_scale_ptr: fx.Pointer,
        value_scale_ptr: fx.Pointer,
        sinks_ptr: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_o_row: fx.Int32,
        stride_o_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        work_info_ptr: fx.Pointer,
        num_sequences: fx.Int32,
    ):
        def _run_task(seq, start, end, context):
            _pa_decode_tile_task(
                output_ptr,
                pmax_ptr,
                psum_ptr,
                pout_ptr,
                query_ptr,
                key_cache_ptr,
                value_cache_ptr,
                block_tables_ptr,
                context_lengths_ptr,
                key_scale_ptr,
                value_scale_ptr,
                sinks_ptr,
                max_blocks_per_seq,
                stride_ks_block,
                stride_ks_head,
                stride_o_row,
                stride_o_head,
                stride_q_row,
                stride_q_head,
                num_sequences,
                seq,
                start,
                end,
                context,
            )

        if const_expr(traits.use_work_plan):
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
        else:
            zero = fx.Int32(0)
            _run_task(zero, zero, zero, zero)

    @flyc.jit
    def pa_decode_tile_launch(
        output: fx.Pointer,
        pmax: fx.Pointer,
        psum: fx.Pointer,
        pout: fx.Pointer,
        query: fx.Pointer,
        key_cache: fx.Pointer,
        value_cache: fx.Pointer,
        block_tables: fx.Pointer,
        context_lengths: fx.Pointer,
        key_scale: fx.Pointer,
        value_scale: fx.Pointer,
        sinks: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        num_seqs: fx.Int32,
        num_kv_heads: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_o_row: fx.Int32,
        stride_o_head: fx.Int32,
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
                output,
                pmax,
                psum,
                pout,
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lengths,
                key_scale,
                value_scale,
                sinks,
                max_blocks_per_seq,
                stride_ks_block,
                stride_ks_head,
                stride_o_row,
                stride_o_head,
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
                        work_capacity if traits.use_work_plan else num_seqs,
                        num_kv_heads * traits.query_splits,
                        1 if traits.use_work_plan else traits.NP,
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

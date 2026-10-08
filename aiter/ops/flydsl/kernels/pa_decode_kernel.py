# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Paged-attention tile kernel with FP8 and gfx1250 native BF16 compute.

For FP8 compute, K/V use e4m3 (FNUZ on gfx942, OCP on gfx950/gfx1250);
BF16/FP16 Q and probabilities P are quantized to FP8.
Q/key scales fold into QK, value scale and 1/FP8_MAX into
the epilogue; softmax max/sum stay f32. Tuned gfx950 BF16 per-token MTP3/MTP4
uses K128 MFMA instead of K32, preserving normalized Q/P and operand layouts.
Every Q row uses absmax normalization; per-tensor and per-token K/V scales
retain range normalization for P.

Logical layouts (not preshuffled):

* ``query``        [num_seqs * query_length, num_q_heads, head_dim]  f16/bf16
* ``key_cache``    [num_blocks, num_kv_heads, head_dim//16, block_size, 16]  fp8
* ``value_cache``  [num_blocks, num_kv_heads, block_size//16, head_dim, 16] (trans_v)
                   or [num_blocks, num_kv_heads, head_dim, block_size] (plain), by rank
* ``block_tables`` [num_seqs, max_blocks_per_seq]  int32
* ``work_info``    [capacity, 4]  int32 sequence/tile bounds/context length
* ``pmax``, ``psum`` [num_kv_heads, capacity, query_length * query_group_size] f32
* ``pout``         [num_kv_heads, capacity, query_length * query_group_size, head_dim]
                   same dtype as query; normalized partition output
* K/V scales      [1] per-tensor or [num_blocks, num_kv_heads, block_size] per-token

Four-wave CTAs process 256-token plan tiles: QK splits tokens, PV splits head
dim, and P passes through LDS to transpose ownership between the MMAs.
gfx1250 uses wave32 WMMA and 64-token subtiles with async K/V LDS staging.
FP8 per-token scales with power-of-two D128+ use TDM; other cases use vector DMA.
Native BF16 uses eight-element cache packing, unquantized Q/P, no K/V scales,
and K32 WMMA with FP32 accumulation. BF16 TDM supports power-of-two D64–512.

The ``pa_decode`` package separates schedule/layout traits, CTA context, Q/K/V
and LDS movement, MFMA, softmax and output operations. ``PaDecodePipeline``
owns prefetch ordering and barriers; this module keeps the compilation API
and launch wrappers.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu
from flydsl.runtime.device import get_rocm_arch

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
    per_token_kv: bool = False,
    query_length: int = 1,
    trans_v: bool = True,
    wide_kv_addressing: bool = False,
    kv_buffer_u32: bool = False,
    query_splits: int | None = None,
    max_context_length: int | None = None,
    sliding_window: int = 0,
    compute_dtype: str = "fp8",
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

    ``work_capacity`` is a positive host integer matching the work plan.
    ``compute_dtype`` is ``fp8`` by default; ``bf16`` requires native BF16
    K/V on gfx1250 and disables quantization scales.
    Positive ``sliding_window`` includes the query token.
    Plans cover the MTP window union; scores are masked per query row.
    All tasks write packed partials; the reducer applies sinks once and writes
    final output, including when the plan allows only one partition.

    Masked V bytes must remain finite because ``0 * NaN == NaN`` in PV MFMA.
    Pages past the sequence are pinned to block 0; callers must leave the
    unwritten tail of the last owned page finite. ``wide_kv_addressing`` protects
    offsets at 2 GiB; ``kv_buffer_u32`` proves both FP8 caches are below 4 GiB
    and permits unsigned buffer offsets instead of i64.
    """
    architecture = (architecture or get_rocm_arch()).split(":")[0]
    if architecture == "gfx1250":
        from .pa_decode.gfx1250 import compile_gfx1250_pa_decode

        if query_splits not in (None, 1):
            raise NotImplementedError(
                "gfx1250 PA decode currently uses fused query rows"
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
            compute_dtype=compute_dtype,
        )
    if compute_dtype != "fp8":
        raise NotImplementedError(
            f"{compute_dtype} PA decode is only supported on gfx1250"
        )
    schedule = PaDecodeSchedule.select(
        architecture=architecture,
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
        work_capacity=work_capacity,
        max_context_length=max_context_length,
        sliding_window=sliding_window,
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

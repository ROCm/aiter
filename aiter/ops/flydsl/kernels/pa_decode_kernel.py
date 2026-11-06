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
FP8 paths use K64 and BF16 uses K32. Four FP8 D128 query tiles over page128
caches give each wave its own tile and keep P in registers. Supported layouts
use TDM K/V transfers, with two LDS buffers where capacity permits.
One-partition plans can write final output directly. Plans with one task slot
per sequence merge a sequence's few partitions in its last-arriving CTA, and
plans whose sequences each fit one workgroup cluster merge within that
cluster; other plans use the shared partition reducer.
BF16 transposed page128 caches at D64/D128/D256 use a packed-page pipeline
with resident Q and fused PV accumulation; D256 is limited to 32 query rows.
Page16 BF16 caches at these dimensions retain vector-8 packing in LDS while
using the 64-token pipeline.

MXFP4 K/V (``kv_dtype="mxfp4"``, gfx950, head_dim 128, page 128, one query token a
row) share this scaffolding and have their own tile: QK is one scaled MFMA (fp4 K x
e4m3 Q), V is widened to bf16 through wave-private LDS. A wave owns 64 tokens of the
tile and all dims, so the tile loop has no barrier; the waves merge once after it.

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
from .pa_decode.op_merge import (
    CLUSTER_PARTITIONS,
    MERGE_HEAD_DIM,
    MERGE_PARTITIONS,
    Gfx1250ClusterMerge,
)
from .pa_decode.pipeline import (
    Gfx1250Bf16DecodePipeline,
    Gfx1250Bf16Pipeline,
    Gfx1250Bf16WavePipeline,
    Gfx1250Pipeline,
    Gfx1250WavePipeline,
    PaDecodePipeline,
)
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
from .tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

__all__ = [
    "CLUSTER_PARTITIONS",
    "FP8_PACK_K",
    "KV_COMPUTE_BLOCK",
    "LOG2E",
    "MERGE_HEAD_DIM",
    "MERGE_PARTITIONS",
    "MFMA_ACC_ELEMS",
    "MFMA_MNK",
    "WAVE",
    "compile_pa_decode_tile",
]

# Key by selected specialization, not batch/head/CU scheduling inputs.
_PA_DECODE_TILE_CACHE = {}
_GFX1250_PA_DECODE_CACHE = {}


def _pa_decode_tile_kernel_name(
    *,
    kv_dtype,
    query_dtype,
    head_dim,
    block_size,
    query_group_size,
    query_length,
    per_token_kv,
    trans_v,
    sliding_window,
    query_splits,
    direct_output=False,
    stage_pages=0,
    merge_partitions=False,
    cluster_merge=0,
):
    """Trace-visible symbol carrying the main compile-time constants."""
    name = (
        f"pa_decode_tile_kernel_kv{kv_dtype}_q{query_dtype}_d{head_dim}"
        f"_bs{block_size}_gqa{query_group_size}_mtp{query_length}"
    )
    if per_token_kv:
        name += "_pertoken"
    if not trans_v:
        name += "_vplain"
    if sliding_window:
        name += f"_sw{sliding_window}"
    if query_splits > 1:
        name += f"_qs{query_splits}"
    if direct_output:
        name += "_direct"
    if stage_pages:
        name += f"_pages{stage_pages}"
    if merge_partitions:
        name += "_merge"
    if cluster_merge:
        name += f"_cluster{cluster_merge}"
    return name


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
    mask_unowned_kv: bool = True,
    cluster_merge: int = 0,
    packed_scale_heads: bool = False,
    cluster_task_stride: int = 0,
    plan_tile_tokens: int = 256,
):
    """Select and cache a PA-decode kernel and launch wrapper.

    ``query_splits=None`` selects from host-known grid bounds; an explicit
    count overrides splitting and selects the matching prefetch policy.
    BF16 also selects the smallest query-length divisor that fits 64 KiB LDS;
    for example D512/MTP4/GQA16 splits twice and D1024/MTP3/GQA16 splits three
    ways. Explicit BF16 splits must fit that limit as well.
    ``max_context_length`` bounds dense planned work for scheduling only; it does not
    change the launch capacity, scratch layout, or single-tile guarantee. Every
    context length, including later graph replays, must be at most this value:
    ``pa_decode`` derives its partition bound from it, so a smaller value can
    truncate attention.
    CTAs receive equal groups of MTP positions, flattened with GQA into 16-row
    M-tiles. Partial output stays unsplit for the shared reducer.

    Single-tile specialization requires a standard plan whose capacity proves
    one KV tile per active task; padding skips the task body. Selected plans
    map packed slots over (B, C/B), taking sequence IDs from plan records.
    Launch B must be positive, divide capacity C, and match schedule selection;
    the JIT launch does not recheck these conditions. B and C/B are not cache keys.
    Planned gfx950 BF16 D128 output uses slot-rebased buffer stores when the
    slot's byte span fits signed i32, retaining the dtype's 2-byte alignment.

    ``direct_output`` (gfx950, MXFP4): no plan -- ``work_info`` is the int32 context
    lengths -- one task per sequence, the output written directly with the query's
    row/head strides (an empty row is zero-filled).
    ``work_capacity`` is a positive host integer matching the work plan.
    Positive ``sliding_window`` includes the query token.
    Plans cover the MTP window union; scores are masked per query row.
    ``trans_k=False`` is not implemented.
    gfx1250 uses wave32 WMMA, native BF16 or OCP FP8, and its larger LDS budget.
    With ``direct_output`` a gfx1250 plan writes final output when sinks are
    absent and query/output strides match: one-partition plans directly, and
    other plans of at most ``MERGE_PARTITIONS`` partitions per sequence by
    merging in each sequence's last-arriving CTA. Other tasks write packed
    partials for the shared reducer, which applies sinks once. A positive
    ``cluster_merge`` P instead merges each sequence's tasks within a cluster
    of P CTAs (see ``compile_gfx1250_pa_decode``); plans must give every
    sequence at most P tasks.

    Pages past the sequence are pinned to block 0 and the unwritten tail of the
    last owned page is read as is. On gfx942/gfx950 with ``mask_unowned_kv``
    (the default), a sequence's last tile zeroes V bytes and K/V scales at or
    past its context before PV, so those rows may hold anything, including NaN
    (``0 * NaN == NaN`` in PV MFMA). ``mask_unowned_kv=False`` skips that
    masking and restores the caller contract that those V bytes and scales
    remain finite. The gfx1250 pipelines ignore the flag and always require it.
    ``wide_kv_addressing`` protects
    offsets at 2 GiB; ``kv_buffer_u32`` proves both FP8 caches are below 4 GiB
    and permits unsigned buffer offsets instead of i64.
    """
    if not trans_k:
        raise NotImplementedError("Non-transposed key cache is not implemented")
    architecture = (architecture or get_gfx_runtime()).split(":", 1)[0]
    if architecture in ("gfx1250", "gfx1250-strict"):
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
        # Capacity covers every row's upper tile bound, so each visible
        # tile gets its own task even with empty/mixed rows.
        tile_bound = (
            max(1, -(-max_context_length // plan_tile_tokens))
            if max_context_length is not None and max_context_length > 0
            else 0
        )
        single_task_tile = bool(
            tile_bound
            and kv_dtype == "fp8"
            and plan_tile_tokens == 128
            and block_size == 128
            and query_group_size == 16
            and query_length == 1
            and sliding_window == 0
            and num_partitions >= tile_bound
            and work_capacity >= num_seqs * tile_bound
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
            direct_output=direct_output and num_partitions == 1,
            query_splits=query_splits,
            merge_partitions=direct_output and num_partitions > 1,
            cluster_merge=cluster_merge,
            packed_scale_heads=packed_scale_heads,
            cluster_task_stride=cluster_task_stride,
            plan_tile_tokens=plan_tile_tokens,
            single_task_tile=single_task_tile,
            direct_context=direct_output
            and num_partitions == 1
            and sliding_window == 0
            and num_seqs <= 4 * num_compute_units,
        )
    if plan_tile_tokens != KV_COMPUTE_BLOCK:
        raise ValueError("128-token work plans require gfx1250 tile kernels")
    if architecture not in ("gfx942", "gfx950"):
        raise NotImplementedError(f"PA decode does not support {architecture}")
    if cluster_merge:
        raise NotImplementedError("cluster_merge requires gfx1250")
    if direct_output and kv_dtype != "mxfp4":
        raise NotImplementedError("direct_output requires gfx1250 or MXFP4")
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
        mask_unowned_kv=mask_unowned_kv,
        direct_output=direct_output,
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

    kernel_name = _pa_decode_tile_kernel_name(
        kv_dtype=kv_dtype,
        query_dtype=query_dtype,
        head_dim=traits.head_dim,
        block_size=traits.block_size,
        query_group_size=traits.query_group_size,
        query_length=traits.query_length,
        per_token_kv=traits.per_token_kv,
        trans_v=traits.trans_v,
        sliding_window=traits.sliding_window,
        query_splits=traits.query_splits,
        direct_output=traits.DIRECT,
        stage_pages=traits.stage_page_capacity if traits.MXFP4 else 0,
    )

    @flyc.kernel(name=kernel_name, known_block_size=(traits.BLOCK_THREADS, 1, 1))
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
        if const_expr(traits.DIRECT):
            # One task per sequence, no plan: ``work_info`` is the context lengths.
            context = fx.Int32(fx.ptr_load(fx.add_offset(work, slot)))
            # Page ids past the staged window come from the block table.
            end = (context + (KV_COMPUTE_BLOCK - 1)) // KV_COMPUTE_BLOCK
            if end > 0:
                _run_task(slot, fx.Int32(0), end, context)
            else:
                # An empty row has no reduce kernel to zero-fill it.
                tid = fx.Int32(gpu.thread_id("x"))
                head = tid // (traits.head_dim // 8)
                if head < traits.query_group_size:
                    out = fx.recast_iter(traits.Q_DTYPE, pout_ptr)
                    offset = (
                        slot * stride_q_row
                        + (fx.Int32(gpu.block_id("y")) * traits.query_group_size + head)
                        * stride_q_head
                        + tid % (traits.head_dim // 8) * 8
                    )
                    fx.ptr_store(
                        fx.full(8, 0.0, traits.Q_DTYPE), fx.add_offset(out, offset)
                    )
        else:
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
    merge_partitions=False,
    cluster_merge=0,
    packed_scale_heads=False,
    cluster_task_stride=0,
    plan_tile_tokens=256,
    single_task_tile=False,
    direct_context=False,
):
    """Compile a gfx1250 tile kernel; merging partitions writes final output.

    ``merge_partitions`` plans keep packed partials and count arrivals per
    (sequence, KV head) in the plan's zeroed ``arrival_counters``. A positive
    ``cluster_merge`` P instead launches P CTAs per sequence as one workgroup
    cluster: CTA i runs the sequence's task i when it has one, and the cluster
    merges P-way after a cluster barrier. Every sequence must have at most P
    tasks. In both modes a sequence's single task stores final rows directly,
    and empty sequences clear their rows.
    ``merge_partitions`` partial storage must be eight-byte aligned for its
    device-scope atomic accesses; the host wrapper falls back for offset views.
    ``single_task_tile`` is internal: the standard planner must have enough
    capacity and partition cap to give every tile its own task within the
    validated context bound. Active tasks then end at start+1, while padding
    still uses its original record bounds. Single-page packed FP8 can retain
    one KV buffer and skip its final reuse barrier; BF16 wave merge keeps two.
    ``plan_tile_tokens`` matches the plan's 128/256-token units; the value
    participates in the kernel cache key and planned range conversion.
    ``direct_context`` replaces the work pointer with context lengths for
    one-partition full-attention output; active CTAs derive their complete
    tile range and empty CTAs clear output without reading packed metadata.
    ``cluster_task_stride`` selects optional [batch, stride, 8] records; it
    must cover the cluster, and ``work`` then points at those records. The
    default zero retains packed work_info plus reduce_info lookups.
    ``packed_scale_heads`` promises that per-token scale head stride equals
    ``block_size``; otherwise the FP8 cluster ABI retains the dynamic stride.
    """
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
        merge_partitions,
        cluster_merge,
        packed_scale_heads,
        cluster_task_stride,
        plan_tile_tokens,
        single_task_tile,
        direct_context,
    )
    if type(plan_tile_tokens) is not int or plan_tile_tokens not in (128, 256):
        raise ValueError("plan_tile_tokens must be 128 or 256")
    if direct_context and (not direct_output or sliding_window != 0):
        raise ValueError("direct_context requires direct full-attention output")
    if compute_dtype not in ("fp8", "bf16"):
        raise ValueError(f"Unsupported gfx1250 PA compute dtype: {compute_dtype}")
    if t.bf16 and per_token_kv:
        raise ValueError("BF16 PA decode does not use quantization scales")
    if merge_partitions and (direct_output or head_dim > MERGE_HEAD_DIM):
        raise ValueError(
            f"merge_partitions excludes direct_output and head_dim > {MERGE_HEAD_DIM}"
        )
    if cluster_merge and (
        direct_output
        or merge_partitions
        or head_dim > MERGE_HEAD_DIM
        or not 2 <= cluster_merge <= CLUSTER_PARTITIONS
    ):
        raise ValueError(
            f"cluster_merge must be in [2, {CLUSTER_PARTITIONS}] and excludes other "
            f"output modes and head_dim > {MERGE_HEAD_DIM}"
        )
    if cluster_task_stride and (
        not cluster_merge or not cluster_merge <= cluster_task_stride <= 8
    ):
        raise ValueError(
            "cluster_task_stride must cover an enabled cluster of at most 8"
        )
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

    kernel_name = _pa_decode_tile_kernel_name(
        kv_dtype=compute_dtype,
        query_dtype=query_dtype,
        head_dim=head_dim,
        block_size=block_size,
        query_group_size=query_group_size,
        query_length=query_length,
        per_token_kv=per_token_kv,
        trans_v=trans_v,
        sliding_window=sliding_window,
        query_splits=query_splits,
        direct_output=direct_output,
        merge_partitions=merge_partitions,
        cluster_merge=cluster_merge,
    )

    @flyc.jit
    def run_tile(
        work: fx.Pointer,
        table: fx.Pointer,
        query: fx.Pointer,
        key: fx.Pointer,
        value: fx.Pointer,
        pmax: fx.Pointer,
        psum: fx.Pointer,
        pout: fx.Pointer,
        ks: fx.Pointer,
        vs: fx.Pointer,
        max_blocks: fx.Int32,
        sk_block: fx.Int32,
        sk_head: fx.Int32,
        sq_row: fx.Int32,
        sq_head: fx.Int32,
        sequences: fx.Int32,
        capacity: fx.Int32,
        kv_heads: fx.Int32,
        reduce_info: fx.Pointer,
        output: fx.Pointer,
        counters: fx.Pointer,
    ):
        _ = tag

        def zero_output(seq, dst):
            # Empty sequences have no attention mass; clear this CTA's rows.
            rows = fx.recast_iter(t.q_dtype, dst)
            tid = fx.Int32(gpu.thread_id("x"))
            for i in range_constexpr(-(-t.rows * t.head_dim // t.BLOCK_THREADS)):
                index = i * t.BLOCK_THREADS + tid
                if index < t.rows * t.head_dim:
                    row, d = index // t.head_dim, index % t.head_dim
                    kv_query = fx.Int32(gpu.block_id("y"))
                    row = row + (kv_query % t.query_splits) * t.rows
                    offset = (
                        (fx.Int64(seq) * t.query_length + row // t.query_group_size)
                        * sq_row
                        + (
                            (kv_query // t.query_splits) * t.query_group_size
                            + row % t.query_group_size
                        )
                        * sq_head
                        + d
                    )
                    rows[offset] = t.q_dtype(0.0)

        task_slot = fx.Int32(gpu.block_id("x"))
        if const_expr(t.cluster_merge and not t.cluster_task_stride):
            # Cluster s runs sequence s's tasks on its leading CTAs.
            # Hardware block IDs are unsigned; avoid signed floor-division
            # fixups before the dependent reduce_info read.
            cluster_slot = fx.Uint32(gpu.block_id("x"))
            seq = fx.Int32(cluster_slot // t.cluster_merge)
            part = fx.Int32(cluster_slot % t.cluster_merge)
            seq_tasks = fx.ptr_load(
                fx.add_offset(fx.recast_iter(fx.Int32, reduce_info), seq * 2),
                result_type=fx.Vector.make_type(2, fx.Int32),
            )
            first, count = fx.Int32(seq_tasks[0]), fx.Int32(seq_tasks[1])
            task_slot = (part < count).select(first + part, fx.Int32(0))
        if const_expr(t.direct_output and not t.direct_context):
            # One task per nonempty sequence; reduce_info supplies its packed slot.
            info = fx.recast_iter(fx.Int32, pmax)
            count = fx.Int32(info[task_slot * 2 + 1])
            # A trailing empty sequence's start can equal capacity. Its dummy
            # task read must stay in bounds even though no compute consumes it.
            task_slot = (count > 0).select(fx.Int32(info[task_slot * 2]), fx.Int32(0))
        if const_expr(t.direct_context):
            # The one-partition full-attention path needs only this sequence's
            # current length; no packed-task lookup or scratch offset is used.
            length = fx.Int32(fx.recast_iter(fx.Int32, work)[task_slot])
            count = fx.Int32(length > 0)
            tiles = fx.Int32(
                (fx.max(fx.Int64(length), fx.Int64(0)) + t.plan_tile_tokens - 1)
                // t.plan_tile_tokens
            )
            task = fx.Vector.from_elements(
                [task_slot, fx.Int32(0), tiles, length], dtype=fx.Int32
            )
        elif const_expr(t.cluster_task_stride):
            cluster_slot = fx.Uint32(gpu.block_id("x"))
            seq = fx.Int32(cluster_slot // t.cluster_merge)
            part = fx.Int32(cluster_slot % t.cluster_merge)
            task = fx.ptr_load(
                fx.add_offset(
                    fx.recast_iter(fx.Int32, work),
                    (seq * t.cluster_task_stride + part) * 8,
                ),
                result_type=fx.Vector.make_type(8, fx.Int32),
            )
            first, count = fx.Int32(task[4]), fx.Int32(task[5])
            task_slot = (part < count).select(first + part, fx.Int32(0))
        else:
            task = fx.ptr_load(
                fx.add_offset(fx.recast_iter(fx.Int32, work), task_slot * 4),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        active = fx.Int32(task[1]) < fx.Int32(task[2])
        if const_expr(t.direct_output):
            active = count > 0
        if const_expr(t.cluster_merge):
            active = part < count
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
                (
                    fx.Int32(task[1]) + 1
                    if const_expr(t.single_task_tile)
                    else fx.Int32(task[2])
                ),
                fx.Int32(task[3]),
            )
            # Packed partials are [KV heads, capacity, rows] by task slot.
            ctx.task_slot, ctx.capacity = task_slot, capacity
            ctx.n_kv = kv_heads
            if const_expr(t.merge_partitions):
                # Issued early: only the epilogue reads the sequence's tasks.
                tasks = fx.ptr_load(
                    fx.add_offset(
                        fx.recast_iter(fx.Int32, reduce_info), fx.Int32(task[0]) * 2
                    ),
                    result_type=fx.Vector.make_type(2, fx.Int32),
                )
                ctx.first_task = fx.Int32(tasks[0])
                ctx.task_count = fx.Int32(tasks[1])
                ctx.output_ptr = output
                ctx.counters_ptr = counters
            if const_expr(t.cluster_merge):
                ctx.task_count = count
                ctx.output_ptr = output
            if const_expr(t.bf16_decode_wave):
                Gfx1250Bf16DecodePipeline(ctx).run()
            elif const_expr(t.bf16_wave_query_tiles):
                Gfx1250Bf16WavePipeline(ctx).run()
            elif const_expr(t.bf16_page128_packed):
                Gfx1250Bf16Pipeline(ctx).run()
            elif const_expr(t.wave_query_tiles):
                Gfx1250WavePipeline(ctx).run()
            else:
                Gfx1250Pipeline(ctx).run()
        elif const_expr(t.direct_output):
            zero_output(gpu.block_id("x"), pout)
        if const_expr(t.merge_partitions):
            # Empty sequences own no task. Capacity >= batch gives each one the
            # CTA at its index, which clears its rows.
            info = fx.recast_iter(fx.Int32, reduce_info)
            if task_slot < sequences:  # noqa: SIM102
                if fx.Int32(info[task_slot * 2 + 1]) == 0:
                    zero_output(task_slot, output)
        if const_expr(t.cluster_merge):
            # Every CTA of a cluster reaches its barrier exactly once.
            if count > 1:
                merge = Gfx1250ClusterMerge(
                    t, pmax, psum, pout, output, sq_row, sq_head
                )
                merge.run(seq, part, first, count, capacity)
            elif (count == 0) & (part == 0):
                zero_output(seq, output)

    if t.bf16:
        # BF16 needs no FP8 scale pointers or strides. Omitting those 24
        # bytes keeps every used argument in one compact preload window.
        @flyc.kernel(name=kernel_name, known_block_size=(t.BLOCK_THREADS, 1, 1))
        def pa_decode_tile_kernel(
            work: fx.Pointer,
            table: fx.Pointer,
            query: fx.Pointer,
            key: fx.Pointer,
            value: fx.Pointer,
            pmax: fx.Pointer,
            psum: fx.Pointer,
            pout: fx.Pointer,
            max_blocks: fx.Int32,
            sq_row: fx.Int32,
            sq_head: fx.Int32,
            sequences: fx.Int32,
            capacity: fx.Int32,
            kv_heads: fx.Int32,
            reduce_info: fx.Pointer,
            output: fx.Pointer,
            counters: fx.Pointer,
        ):
            _ = tag
            null_scale = fx.inttoptr(
                fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 4),
                fx.Int64(0),
            )
            run_tile(
                work,
                table,
                query,
                key,
                value,
                pmax,
                psum,
                pout,
                null_scale,
                null_scale,
                max_blocks,
                fx.Int32(0),
                fx.Int32(0),
                sq_row,
                sq_head,
                sequences,
                capacity,
                kv_heads,
                reduce_info,
                output,
                counters,
            )

    elif t.compact_fp8_cluster:
        # Cluster-only FP8 drops sequence count, arrival counters and the
        # known scale-head stride. All remaining arguments fit in 120 bytes.
        @flyc.kernel(name=kernel_name, known_block_size=(t.BLOCK_THREADS, 1, 1))
        def pa_decode_tile_kernel(
            work: fx.Pointer,
            table: fx.Pointer,
            query: fx.Pointer,
            key: fx.Pointer,
            value: fx.Pointer,
            pmax: fx.Pointer,
            psum: fx.Pointer,
            pout: fx.Pointer,
            ks: fx.Pointer,
            vs: fx.Pointer,
            max_blocks: fx.Int32,
            sk_block: fx.Int32,
            sq_row: fx.Int32,
            sq_head: fx.Int32,
            capacity: fx.Int32,
            kv_heads: fx.Int32,
            reduce_info: fx.Pointer,
            output: fx.Pointer,
        ):
            _ = tag
            null_counter = fx.inttoptr(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                fx.Int64(0),
            )
            run_tile(
                work,
                table,
                query,
                key,
                value,
                pmax,
                psum,
                pout,
                ks,
                vs,
                max_blocks,
                sk_block,
                fx.Int32(t.block_size if t.per_token_kv else 0),
                sq_row,
                sq_head,
                fx.Int32(0),
                capacity,
                kv_heads,
                reduce_info,
                output,
                null_counter,
            )

    else:
        # Reducer-fed and direct-output arguments already fit in 120 bytes.
        @flyc.kernel(name=kernel_name, known_block_size=(t.BLOCK_THREADS, 1, 1))
        def pa_decode_tile_kernel(
            work: fx.Pointer,
            table: fx.Pointer,
            query: fx.Pointer,
            key: fx.Pointer,
            value: fx.Pointer,
            pmax: fx.Pointer,
            psum: fx.Pointer,
            pout: fx.Pointer,
            ks: fx.Pointer,
            vs: fx.Pointer,
            max_blocks: fx.Int32,
            sk_block: fx.Int32,
            sk_head: fx.Int32,
            sq_row: fx.Int32,
            sq_head: fx.Int32,
            sequences: fx.Int32,
            capacity: fx.Int32,
            kv_heads: fx.Int32,
            reduce_info: fx.Pointer,
            output: fx.Pointer,
            counters: fx.Pointer,
        ):
            _ = tag
            run_tile(
                work,
                table,
                query,
                key,
                value,
                pmax,
                psum,
                pout,
                ks,
                vs,
                max_blocks,
                sk_block,
                sk_head,
                sq_row,
                sq_head,
                sequences,
                capacity,
                kv_heads,
                reduce_info,
                output,
                counters,
            )

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
        output: fx.Pointer,
        reduce_info: fx.Pointer,
        counters: fx.Pointer,
        stream: fx.Stream,
    ):
        _ = tag
        grid_x = capacity
        if const_expr(t.direct_output):
            grid_x = sequences
        if const_expr(t.cluster_merge):
            grid_x = sequences * t.cluster_merge
        tile_args = (work, table, query, key, value, pmax, psum, pout)
        if const_expr(t.bf16):
            tile_args += (
                max_blocks,
                sq_row,
                sq_head,
                sequences,
                capacity,
                kv_heads,
                reduce_info,
                output,
                counters,
            )
        elif const_expr(t.compact_fp8_cluster):
            tile_args += (
                ks,
                vs,
                max_blocks,
                sk_block,
                sq_row,
                sq_head,
                capacity,
                kv_heads,
                reduce_info,
                output,
            )
        else:
            tile_args += (
                ks,
                vs,
                max_blocks,
                sk_block,
                sk_head,
                sq_row,
                sq_head,
                sequences,
                capacity,
                kv_heads,
                reduce_info,
                output,
                counters,
            )
        pa_decode_tile_kernel(
            *tile_args,
            value_attrs=(
                {"rocdl.cluster_dims": f"{t.cluster_merge},1,1"}
                if t.cluster_merge
                else None
            ),
        ).launch(
            grid=(grid_x, kv_heads * t.query_splits, 1),
            block=(t.BLOCK_THREADS, 1, 1),
            stream=stream,
            cluster=(t.cluster_merge, 1, 1) if t.cluster_merge else None,
        )

    llvm_options = {}
    if t.bf16_wave_query_tiles or t.bf16_decode_wave:
        # Iterative ILP scheduling leaves fewer VGPR bank switches per page.
        llvm_options["amdgpu-sched-strategy"] = "iterative-ilp"
    if not t.merged_output or t.bf16 or t.compact_fp8_cluster:
        # The compact merge ABIs fit completely. Other FP8 merges retain
        # explicit loads rather than a slower partial preload.
        llvm_options["amdgpu-kernarg-preload"] = AITER_FLYDSL_KERNARG_PRELOAD
        llvm_options["amdgpu-kernarg-preload-count"] = (
            AITER_FLYDSL_KERNARG_PRELOAD_COUNT
        )
    if llvm_options:
        launch.compile_hints["llvm_options"] = llvm_options
    compiled = {"launch": launch, "kernel": pa_decode_tile_kernel}
    return _GFX1250_PA_DECODE_CACHE.setdefault(t, compiled)

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

MXFP4 K/V (``kv_dtype=dtypes.fp4x2``, gfx950/gfx1250, head_dim 128, page 128, one
query token a row) share this scaffolding and have their own tile: QK is one scaled
MFMA/WMMA (fp4 K x e4m3 Q). gfx950 widens V to bf16 through wave-private LDS; gfx1250
dequantizes V with the scaled WMMA against fp4 identity blocks, which leaves it
transposed in registers for the bf16 PV WMMA. A wave owns 64 tokens of the tile and
all dims, so the tile loop has no barrier; the waves merge once after it.

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
import torch
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr

from aiter.jit.utils.chip_info import get_gfx_runtime

from .pa_decode.context import PaDecodeContext
from .pa_decode.op_merge import (
    CLUSTER_PARTITIONS,
    MERGE_HEAD_DIM,
    MERGE_PARTITIONS,
    Gfx1250ClusterMerge,
)
from .pa_decode.pipeline import (
    Gfx1250Bf16Pipeline,
    Gfx1250Bf16WavePipeline,
    Gfx1250Mxfp4Pipeline,
    Gfx1250Pipeline,
    Gfx1250WavePipeline,
    PaDecodePipeline,
)
from .pa_decode.traits import (
    FP8_KV_DTYPES,
    FP8_PACK_K,
    KV_COMPUTE_BLOCK,
    LOG2E,
    MFMA_ACC_ELEMS,
    MFMA_MNK,
    MXFP4_KV_DTYPE,
    WAVE,
    Gfx1250Traits,
    PaDecodeSchedule,
    PaDecodeTraits,
    pa_fx_dtypes,
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


def _dtype_tag(dtype):
    """Short dtype tag of trace-visible kernel names and the gfx1250 traits."""
    if dtype in FP8_KV_DTYPES:
        return "fp8"
    return {fx.Float16: "f16", fx.BFloat16: "bf16", MXFP4_KV_DTYPE: "mxfp4"}[dtype]


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
    query_dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype,
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
):
    """Select and cache a PA-decode kernel and launch wrapper.

    ``kv_dtype`` is the KV cache's torch dtype: ``dtypes.fp8`` (the arch's e4m3
    flavor: FNUZ on gfx942, OCP elsewhere), ``torch.bfloat16``, or
    ``dtypes.fp4x2`` for MXFP4. ``dtypes.fp4x2`` is ``torch.uint8`` on torch
    builds without ``float4_e2m1fn_x2``, so a uint8 ``kv_dtype`` also selects
    MXFP4. ``query_dtype`` is the query's: f16 or bf16, or e4m3fn with MXFP4 KV.
    Non-torch dtypes, such as the former ``"fp8"`` strings, raise TypeError;
    unsupported dtypes and combinations raise ValueError.

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

    ``direct_output`` (MXFP4, gfx950 and gfx1250): no plan -- ``work_info`` is the
    int32 context lengths -- one task per sequence, the output written directly with
    the query's row/head strides (an empty row is zero-filled). FP8/BF16 on gfx942/gfx950 accept it
    with a one-partition plan (capacity equal to the batch): ``pmax`` carries the
    plan's ``reduce_info``, slot s runs packed task s and zero-fills sequence s
    when it owns no task, and ``pout`` is the output.
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
        kv_fx, query_fx = pa_fx_dtypes(kv_dtype, query_dtype)
        if kv_fx == MXFP4_KV_DTYPE:
            if query_fx not in (fx.BFloat16, fx.Float8E4M3FN):
                raise ValueError("MXFP4 KV requires a BF16 or FP8 query")
            # direct_output is the plan-free form: one task per sequence.
            return compile_gfx1250_pa_decode(
                head_dim=head_dim,
                query_group_size=query_group_size,
                block_size=block_size,
                softmax_scale=softmax_scale,
                query_dtype=_dtype_tag(query_fx),
                per_token_kv=per_token_kv,
                query_length=query_length,
                trans_v=trans_v,
                sliding_window=sliding_window,
                compute_dtype="mxfp4",
                plan_free=direct_output,
                cluster_merge=cluster_merge,
            )
        if query_fx not in (fx.Float16, fx.BFloat16) or kv_fx not in (
            fx.Float8E4M3FN,
            fx.BFloat16,
        ):
            raise ValueError(
                "gfx1250 PA decode takes an f16/bf16 query and e4m3fn/bf16 KV, "
                f"got {query_dtype} and {kv_dtype}"
            )
        if query_splits is None:
            # Split the measured spilling BF16 specialization into one Q tile
            # per CTA. Other shapes retain fused queries and their KV reuse.
            query_splits = (
                4
                if kv_fx == fx.BFloat16
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
            query_dtype=_dtype_tag(query_fx),
            per_token_kv=per_token_kv,
            query_length=query_length,
            trans_v=trans_v,
            sliding_window=sliding_window,
            compute_dtype=_dtype_tag(kv_fx),
            direct_output=direct_output and num_partitions == 1,
            query_splits=query_splits,
            merge_partitions=direct_output and num_partitions > 1,
            cluster_merge=cluster_merge,
        )
    if architecture not in ("gfx942", "gfx950"):
        raise NotImplementedError(f"PA decode does not support {architecture}")
    if cluster_merge:
        raise NotImplementedError("cluster_merge requires gfx1250")
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

    @fx.struct
    class SharedStorage:
        buf: fx.Array[
            fx.Int32,
            (
                max(
                    2 * traits.GROUP_LDS_BYTES,
                    # Dual-group merge: (O, m, l) per group-1 lane.
                    traits.BLOCK_THREADS
                    * (traits.VHE_CHUNKS * traits.OP_ELEMS + 2)
                    * 4,
                )
                if traits.DUAL_GROUP
                else traits.total_bytes
            )
            // 4,
            16,
        ]

    @flyc.jit
    def _pa_decode_tile_task(
        # Packed partials: [KV heads, capacity, query rows].
        pmax_ptr: fx.Pointer,  # Natural-log row max.
        psum_ptr: fx.Pointer,  # Row sum.
        pout_ptr: fx.Pointer,  # Adds head_dim; OUT_DTYPE normalized O_p/l_p.
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
        kv_dtype=_dtype_tag(traits.kv_dtype),
        query_dtype=_dtype_tag(traits.query_dtype),
        head_dim=traits.head_dim,
        block_size=traits.block_size,
        query_group_size=traits.query_group_size,
        query_length=traits.query_length,
        per_token_kv=traits.per_token_kv,
        trans_v=traits.trans_v,
        sliding_window=traits.sliding_window,
        query_splits=traits.query_splits,
        direct_output=traits.DIRECT,
        stage_pages=(
            traits.stage_page_capacity if traits.kv_dtype == MXFP4_KV_DTYPE else 0
        ),
    ) + ("_dual" if traits.DUAL_GROUP else "")

    @flyc.jit
    def _pa_decode_tile_body(
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

        def _zero_rows(seq):
            """Zero this CTA's query rows of ``seq`` in the final output."""
            tid = fx.Int32(gpu.thread_id("x"))
            row_units = traits.head_dim // 8
            kv_query = fx.Int32(gpu.block_id("y"))
            kv_h = kv_query // traits.query_splits
            first_row = (
                (kv_query % traits.query_splits)
                * traits.QUERIES_PER_CTA
                * traits.query_group_size
            )
            out = fx.recast_iter(traits.OUT_DTYPE, pout_ptr)
            for it in range_constexpr(
                -(-traits.CTA_ROWS * row_units // traits.CTA_THREADS)
            ):
                unit = tid + it * traits.CTA_THREADS
                if unit < traits.CTA_ROWS * row_units:
                    row = first_row + unit // row_units
                    query_pos = row // traits.query_group_size
                    head = kv_h * traits.query_group_size + (
                        row - query_pos * traits.query_group_size
                    )
                    offset = (
                        (seq * traits.query_length + query_pos) * stride_q_row
                        + head * stride_q_head
                        + unit % row_units * 8
                    )
                    fx.ptr_store(
                        fx.full(8, 0.0, traits.OUT_DTYPE), fx.add_offset(out, offset)
                    )

        # Skip cleared padding records: no scratch writes or Q loads.
        # This CTA-uniform guard keeps every task barrier convergent.
        slot = fx.Int32(gpu.block_id("x"))
        work = fx.recast_iter(fx.Int32, work_info_ptr)
        if const_expr(traits.DIRECT and traits.kv_dtype != MXFP4_KV_DTYPE):
            # One-partition plan, capacity == batch: slot s runs packed task s
            # (any sequence's) and zeroes sequence s when it owns no task, as
            # the reducer would; pmax carries reduce_info. Both loads issue
            # together.
            task = fx.ptr_load(
                fx.add_offset(work, slot * 4),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
            info = fx.recast_iter(fx.Int32, pmax_ptr)
            count = fx.Int32(fx.ptr_load(fx.add_offset(info, slot * 2 + 1)))
            start = fx.Int32(task[1])
            end = fx.Int32(task[2])
            if start < end:
                _run_task(fx.Int32(task[0]), start, end, fx.Int32(task[3]))
            if count == 0:
                _zero_rows(slot)
        elif const_expr(traits.DIRECT):
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
                    out = fx.recast_iter(traits.OUT_DTYPE, pout_ptr)
                    offset = (
                        slot * stride_q_row
                        + (fx.Int32(gpu.block_id("y")) * traits.query_group_size + head)
                        * stride_q_head
                        + tid % (traits.head_dim // 8) * 8
                    )
                    fx.ptr_store(
                        fx.full(8, 0.0, traits.OUT_DTYPE), fx.add_offset(out, offset)
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

    if architecture == "gfx950":
        # Lead with what the task prologue reads first, so it fits the 14
        # dwords that preload into user SGPRs: the task record, the page ids,
        # Q, and pmax (reduce_info on one-partition plans). K/V and scale
        # pointers are first used after the page ids arrive.
        @flyc.kernel(name=kernel_name, known_block_size=(traits.CTA_THREADS, 1, 1))
        def pa_decode_tile_kernel(
            work_info_ptr: fx.Pointer,
            block_tables_ptr: fx.Pointer,
            query_ptr: fx.Pointer,
            max_blocks_per_seq: fx.Int32,
            num_sequences: fx.Int32,
            stride_q_row: fx.Int32,
            stride_q_head: fx.Int32,
            pmax_ptr: fx.Pointer,
            key_cache_ptr: fx.Pointer,
            value_cache_ptr: fx.Pointer,
            key_scale_ptr: fx.Pointer,
            value_scale_ptr: fx.Pointer,
            stride_ks_block: fx.Int32,
            stride_ks_head: fx.Int32,
            psum_ptr: fx.Pointer,
            pout_ptr: fx.Pointer,
        ):
            _pa_decode_tile_body(
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
                work_info_ptr,
                num_sequences,
            )

        def _tile_args(
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
        ):
            return (
                work_info,
                block_tables,
                query,
                max_blocks_per_seq,
                num_seqs,
                stride_q_row,
                stride_q_head,
                pmax,
                key_cache,
                value_cache,
                key_scale,
                value_scale,
                stride_ks_block,
                stride_ks_head,
                psum,
                pout,
            )

    else:

        @flyc.kernel(name=kernel_name, known_block_size=(traits.CTA_THREADS, 1, 1))
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
            _pa_decode_tile_body(
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
                work_info_ptr,
                num_sequences,
            )

        def _tile_args(*args):
            return args

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
        # Ambient contract permits FMAs; explicit per-op fastmath still wins.
        with CompilationContext.compile_hints({"fastmath": "contract"}):
            pa_decode_tile_kernel(
                *_tile_args(
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
                )
            ).launch(
                grid=(work_capacity, num_kv_heads * traits.query_splits, 1),
                block=(traits.CTA_THREADS, 1, 1),
                stream=stream,
            )

    if architecture == "gfx950":
        # The leading arguments preload into the free user SGPRs (14 dwords on
        # gfx950), so the task skips a kernarg round trip.
        pa_decode_tile_launch.compile_hints["llvm_options"] = {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": min(16, AITER_FLYDSL_KERNARG_PRELOAD_COUNT),
        }
    if traits.FP8_MTP4_LEAN:
        # Two CTAs per CU: hold the allocator to 256 registers per wave.
        pa_decode_tile_launch.compile_hints["waves_per_eu"] = 2
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
    plan_free=False,
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

    ``compute_dtype="mxfp4"`` (``Gfx1250Mxfp4Pipeline``) takes E2M1 K/V data and
    E8M0 scale views, head_dim 128, page 128, one query token and at most 16
    rows, and writes packed partials for the reducer or a cluster merge;
    ``plan_free`` instead reads ``work`` as the context lengths, runs sequence
    s whole in CTA s and stores its final rows (an empty sequence's are zeroed).
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
        plan_free,
    )
    if compute_dtype not in ("fp8", "bf16", "mxfp4"):
        raise ValueError(f"Unsupported gfx1250 PA compute dtype: {compute_dtype}")
    if plan_free and not t.mxfp4:
        raise ValueError("plan_free requires MXFP4 KV")
    if t.mxfp4 and not (
        head_dim == 128
        and block_size == 128
        and query_length == 1
        and 1 <= query_group_size <= 16
        and query_dtype in ("bf16", "fp8")
        and not per_token_kv
        and sliding_window == 0
        and query_splits == 1
        and not (direct_output or merge_partitions)
        and not (plan_free and cluster_merge)
    ):
        raise NotImplementedError(
            "gfx1250 MXFP4 PA decode supports head_dim 128, block 128, one query "
            "token, 1..16 query heads per KV head, a bf16 or e4m3 query, no "
            "window, and reducer, cluster-merged or plan-free output"
        )
    if query_dtype == "fp8" and not t.mxfp4:
        raise ValueError("An FP8 query needs MXFP4 KV")
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
    if t.total_bytes > 327680:
        raise NotImplementedError(
            "gfx1250 PA decode query/KV tiles exceed LDS capacity"
        )
    if t in _GFX1250_PA_DECODE_CACHE:
        return _GFX1250_PA_DECODE_CACHE[t]

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
        direct_output=direct_output or plan_free,
        merge_partitions=merge_partitions,
        cluster_merge=cluster_merge,
    )

    # Every argument that reducer-fed and direct-output kernels read fits in
    # the 30 user SGPRs (120 bytes) that preload kernel arguments, ahead of
    # the merge-only arguments.
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
        if const_expr(t.cluster_merge):
            # Cluster s runs sequence s's tasks on its leading CTAs.
            seq, part = task_slot // t.cluster_merge, task_slot % t.cluster_merge
            seq_tasks = fx.ptr_load(
                fx.add_offset(fx.recast_iter(fx.Int32, reduce_info), seq * 2),
                result_type=fx.Vector.make_type(2, fx.Int32),
            )
            first, count = fx.Int32(seq_tasks[0]), fx.Int32(seq_tasks[1])
            task_slot = (part < count).select(first + part, fx.Int32(0))
        if const_expr(t.direct_output):
            # One task per nonempty sequence; reduce_info supplies its packed slot.
            info = fx.recast_iter(fx.Int32, pmax)
            count = fx.Int32(info[task_slot * 2 + 1])
            # A trailing empty sequence's start can equal capacity. Its dummy
            # task read must stay in bounds even though no compute consumes it.
            task_slot = (count > 0).select(fx.Int32(info[task_slot * 2]), fx.Int32(0))
        if const_expr(t.plan_free):
            # CTA s runs sequence s: all tiles of its context length.
            context = fx.Int32(fx.recast_iter(fx.Int32, work)[task_slot])
            task = [
                task_slot,
                fx.Int32(0),
                (context + (KV_COMPUTE_BLOCK - 1)) // KV_COMPUTE_BLOCK,
                context,
            ]
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
                fx.Int32(task[2]),
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
            if const_expr(t.plan_free):
                ctx.output_ptr = output
            if const_expr(t.mxfp4):
                Gfx1250Mxfp4Pipeline(ctx).run()
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
        elif const_expr(t.plan_free):
            zero_output(task_slot, output)
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
        grid_x = capacity
        if const_expr(t.direct_output or t.plan_free):
            grid_x = sequences
        if const_expr(t.cluster_merge):
            grid_x = sequences * t.cluster_merge
        pa_decode_tile_kernel(
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
    if t.bf16_wave_query_tiles:
        # Iterative ILP scheduling leaves fewer VGPR bank switches per page.
        llvm_options["amdgpu-sched-strategy"] = "iterative-ilp"
    if not t.merged_output:
        # Argument loads that a preload leaves behind are slower than those of
        # an unpreloaded kernel (cluster merges ran 2-6% slower), so merging
        # kernels, whose arguments do not all fit, keep loading them.
        llvm_options["amdgpu-kernarg-preload"] = AITER_FLYDSL_KERNARG_PRELOAD
        llvm_options["amdgpu-kernarg-preload-count"] = (
            AITER_FLYDSL_KERNARG_PRELOAD_COUNT
        )
    if llvm_options:
        launch.compile_hints["llvm_options"] = llvm_options
    compiled = {"launch": launch, "kernel": pa_decode_tile_kernel}
    return _GFX1250_PA_DECODE_CACHE.setdefault(t, compiled)

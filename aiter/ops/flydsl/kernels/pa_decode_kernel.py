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

MXFP4 K/V (``kv_dtype=dtypes.fp4x2``, gfx950, head_dim 128, page 128, one query token a
row) share this scaffolding and have their own tile: QK is one scaled MFMA (fp4 K x
e4m3 Q), V is widened to bf16 through wave-private LDS. A wave owns 64 tokens of the
tile and all dims, so the tile loop has no barrier; the waves merge once after it.

NVFP4 K/V (``kv_dtype=dtypes.fp4x2`` with ``kv_scale_dtype=torch.float8_e4m3fn``,
gfx1250, head_dim 128, page 128) are e2m1 with one E4M3 scale per 16 values, stored
token-major per (page, KV head): [128, 64] payload and [128, 8] scale bytes. Tiles
of one query tile (<= 16 rows) run ``Gfx1250Nvfp4WavePipeline``: each wave owns 32
tokens of every page, DMAs them itself, computes QK and dequantizes V with
``v_wmma_scale16_f32_16x16x128_f8f6f4`` straight from the raw rows and keeps
softmax and P in registers, so the page loop has no CTA barrier. Wider tiles run
the BF16 page128 pipeline: tensor DMA double-buffers the raw page in LDS and the
CTA widens it (``v_cvt_scale_pk8_bf16_fp4`` then a packed BF16 multiply by the
block scale, exact) into the BF16 layouts that pipeline reads.

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

import os

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
    Gfx1250OutputMxQuant,
    Gfx1250OutputQuant,
)
from .kernels_common import create_llvm_ptr
from .pa_decode.pipeline_nvfp4 import Gfx1250Nvfp4WavePipeline
from .tensor_shim import buf_base_i64
from .pa_decode.pipeline import (
    Gfx1250Bf16Pipeline,
    Gfx1250Bf16WavePipeline,
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
# NVFP4 tiles of one query tile (<= 16 rows: M3.1 TP4 decode, MTP and prefill
# rows) run the wave pipeline (scaled-WMMA QK and V dequantization, no
# per-page CTA barrier). AITER_FLYDSL_NVFP4_PA_LEGACY=1 selects the CTA-shared
# BF16-widening pipeline instead (A/B benchmarks); wider tiles always use it.
# AITER_FLYDSL_NVFP4_PA_BUFFERS sets the wave pipeline's page ring (2-4).
NVFP4_LEGACY = os.environ.get("AITER_FLYDSL_NVFP4_PA_LEGACY", "0") == "1"
NVFP4_BUFFERS = int(os.environ.get("AITER_FLYDSL_NVFP4_PA_BUFFERS", "3"))
if NVFP4_BUFFERS not in (2, 3, 4):
    raise ValueError("AITER_FLYDSL_NVFP4_PA_BUFFERS must be 2, 3 or 4")


def nvfp4_wave_tile(kv_scale_dtype, rows):
    """Whether K/V of these block scales and query rows (query_length x GQA)
    run the NVFP4 wave tile, the tile whose epilogue writes the MXFP8 output."""
    return kv_scale_dtype == torch.float8_e4m3fn and not NVFP4_LEGACY and rows <= 16


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
    quant_output=False,
    store_output=True,
    quant_mx=False,
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
    if quant_output:
        tag = "_mxfp8" if quant_mx else "_quant"
        name += tag if store_output else f"{tag}only"
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
    quant_output: bool = False,
    store_output: bool = True,
    kv_scale_dtype: torch.dtype | None = None,
    quant_mx: bool = False,
    page_groups: int | None = None,
    xcd_rows: int = 0,
    xcds: int = 0,
    plan_free: bool = False,
    cluster_rows: int = 1,
):
    """Select and cache a PA-decode kernel and launch wrapper.

    ``kv_dtype`` is the KV cache's torch dtype: ``dtypes.fp8`` (the arch's e4m3
    flavor: FNUZ on gfx942, OCP elsewhere), ``torch.bfloat16``, or
    ``dtypes.fp4x2`` for MXFP4. ``dtypes.fp4x2`` is ``torch.uint8`` on torch
    builds without ``float4_e2m1fn_x2``, so a uint8 ``kv_dtype`` also selects
    MXFP4. ``query_dtype`` is the query's: f16 or bf16, or e4m3fn with MXFP4 KV.
    Non-torch dtypes, such as the former ``"fp8"`` strings, raise TypeError;
    unsupported dtypes and combinations raise ValueError.
    ``kv_scale_dtype=torch.float8_e4m3fn`` with an fp4x2/uint8 ``kv_dtype`` selects
    NVFP4 (gfx1250 only): E4M3 block scales of 16 values instead of MXFP4's E8M0
    of 32.

    ``quant_mx`` (with ``quant_output``) writes the MXFP8 output (``op_mxfp8``)
    instead of the per-token FP8 one. Only the NVFP4 wave tile has it
    (``nvfp4_wave_tile``); it takes every gfx1250 output mode that writes final
    rows (direct, last-arriver and cluster merges).

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
    nvfp4 = kv_scale_dtype == torch.float8_e4m3fn
    if kv_scale_dtype is not None and not nvfp4:
        raise ValueError(f"Unsupported PA KV scale dtype {kv_scale_dtype}")
    if nvfp4:
        kv_fx, query_fx = pa_fx_dtypes(kv_dtype, query_dtype)
        if architecture not in ("gfx1250", "gfx1250-strict"):
            raise NotImplementedError(
                f"NVFP4 PA decode supports gfx1250, not {architecture}"
            )
        if kv_fx != MXFP4_KV_DTYPE or query_fx not in (fx.Float16, fx.BFloat16):
            raise ValueError(
                "NVFP4 PA decode takes fp4x2/uint8 KV with an f16/bf16 query, "
                f"got {kv_dtype} and {query_dtype}"
            )
        wave = nvfp4_wave_tile(kv_scale_dtype, query_length * query_group_size) and (
            query_splits is None or query_splits == 1
        )
        return compile_gfx1250_pa_decode(
            head_dim=head_dim,
            query_group_size=query_group_size,
            block_size=block_size,
            softmax_scale=softmax_scale,
            query_dtype=_dtype_tag(query_fx),
            per_token_kv=False,
            query_length=query_length,
            trans_v=True,
            sliding_window=sliding_window,
            compute_dtype="nvfp4w" if wave else "nvfp4",
            nvfp4_buffers=NVFP4_BUFFERS if wave else 0,
            direct_output=direct_output and num_partitions == 1,
            query_splits=1 if query_splits is None else query_splits,
            merge_partitions=direct_output and num_partitions > 1,
            cluster_merge=cluster_merge,
            quant_output=quant_output,
            store_output=store_output,
            quant_mx=quant_mx,
            page_groups=(page_groups or 1) if wave else 1,
            xcd_rows=xcd_rows,
            xcds=xcds,
            plan_free=plan_free,
            cluster_rows=cluster_rows,
        )
    if quant_mx:
        raise ValueError("quant_mx (MXFP8 output) requires the NVFP4 wave tile")
    if architecture in ("gfx1250", "gfx1250-strict"):
        kv_fx, query_fx = pa_fx_dtypes(kv_dtype, query_dtype)
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
            quant_output=quant_output,
            store_output=store_output,
        )
    if architecture not in ("gfx942", "gfx950"):
        raise NotImplementedError(f"PA decode does not support {architecture}")
    if cluster_merge or quant_output or not store_output:
        raise NotImplementedError(
            "cluster_merge, quant_output and a skipped store require gfx1250"
        )
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
        buf: fx.Array[fx.Int32, traits.total_bytes // 4, 16]

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
                grid=(work_capacity, num_kv_heads * traits.query_splits, 1),
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
    quant_output=False,
    store_output=True,
    nvfp4_buffers=0,
    quant_mx=False,
    page_groups=1,
    xcd_rows=0,
    xcds=0,
    plan_free=False,
    cluster_rows=1,
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
        quant_output,
        store_output,
        nvfp4_buffers if compute_dtype == "nvfp4w" else 0,
        quant_mx,
        page_groups,
        xcd_rows if xcds > 1 else 0,
        xcds if xcd_rows and xcds > 1 else 0,
        plan_free,
        cluster_rows,
    )
    if compute_dtype not in ("fp8", "bf16", "nvfp4", "nvfp4w"):
        raise ValueError(f"Unsupported gfx1250 PA compute dtype: {compute_dtype}")
    if t.nvfp4 and not (
        head_dim == 128
        and block_size == 128
        and trans_v
        and not per_token_kv
        and sliding_window == 0
        and query_splits == 1
        and t.bf16_page128_packed
        and not t.bf16_wave_query_tiles
    ):
        # The widening pass maps one token (K) and one 8-token x 16-dim block (V)
        # to each of the 128 threads: head_dim 128 and page 128. Four query
        # tiles would select the BF16 wave-pair pipeline, which it does not feed.
        raise ValueError(
            "NVFP4 PA decode supports head_dim 128, page 128, at most 48 query rows "
            "(query_length x GQA), no sliding window or query split"
        )
    if t.nvfp4_wave and not (t.m_tiles == 1 and 2 <= nvfp4_buffers <= 4):
        raise ValueError("The NVFP4 wave pipeline takes one query tile, 2-4 buffers")
    if page_groups != 1 and not (
        t.nvfp4_wave
        and page_groups in (2, 4)
        and not merge_partitions
        and (quant_mx or not quant_output)
    ):
        # Only the first group stores: an epilogue with a CTA barrier (the
        # per-token FP8 maxima, the last-arriver merge) would diverge.
        raise ValueError(
            "page_groups 2/4 take the NVFP4 wave tile without the per-token FP8 "
            "output or the last-arriver merge"
        )
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
    if quant_mx and not (quant_output and t.nvfp4_wave):
        raise ValueError("quant_mx requires quant_output and the NVFP4 wave tile")
    if quant_mx and not (direct_output or merge_partitions or cluster_merge):
        raise ValueError("quant_mx requires an output mode that writes final rows")
    if (
        quant_output
        and not quant_mx
        and not (
            (
                direct_output
                and not merge_partitions
                and query_length == 1
                and query_splits == 1
            )
            or cluster_merge == 8
        )
    ):
        raise ValueError("quant_output requires the gfx1250 direct or cluster8 path")
    if quant_output and (
        query_group_size != 16
        or head_dim != 128
        or query_length != 1
        or block_size != (128 if t.nvfp4 else 16)
    ):
        raise ValueError(
            "quant_output requires query_length=1, GQA=16, head_dim=128 and page16 "
            "(page128 for NVFP4 K/V)"
        )
    if not (store_output or quant_output):
        raise ValueError("store_output=False requires quant_output")
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
        direct_output=direct_output,
        merge_partitions=merge_partitions,
        cluster_merge=cluster_merge,
        quant_output=quant_output,
        store_output=store_output,
        quant_mx=quant_mx,
    )
    if t.nvfp4_wave and nvfp4_buffers != 3:
        kernel_name += f"_ring{nvfp4_buffers}"
    if page_groups != 1:
        kernel_name += f"_pg{page_groups}"
    if t.xcd_rows:
        kernel_name += f"_xcd{t.xcd_rows}"
    if t.xcd_rows and (merge_partitions or not (direct_output or cluster_merge)):
        raise ValueError("xcd_rows remaps the direct and cluster output modes")
    if plan_free:
        if not (t.nvfp4_wave and (direct_output or cluster_merge)):
            raise ValueError(
                "plan_free takes the NVFP4 wave tile's direct or cluster output"
            )
        kernel_name += "_planfree"
    if cluster_rows != 1:
        if not (
            plan_free
            and cluster_merge
            and cluster_merge % cluster_rows == 0
            and cluster_merge // cluster_rows >= 2
        ):
            raise ValueError(
                "cluster_rows takes plan-free cluster merges of at least two "
                "CTAs a sequence"
            )
        kernel_name += f"_rows{cluster_rows}"

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
        quantized_output: fx.Pointer,
        output_scale: fx.Pointer,
        quant_rows: fx.Int32,
    ):

        # Optional FP8 output; the pipelines and cluster merge write it from
        # the final values in registers.
        quant = (Gfx1250OutputMxQuant if t.quant_mx else Gfx1250OutputQuant)(
            t, quantized_output, output_scale
        )

        def zero_output(seq, dst):
            # Empty sequences have no attention mass; clear this CTA's rows.
            if const_expr(t.quant_output):
                quant.store_zero(seq, sq_row, sq_head)
            rows = fx.recast_iter(t.q_dtype, dst)
            tid = fx.Int32(gpu.thread_id("x"))
            stores = -(-t.rows * t.head_dim // t.BLOCK_THREADS) if t.store_output else 0
            for i in range_constexpr(stores):
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

        def pad_output(row):
            # A padded token-bucket row past the sequences: FP8 zero bytes and a
            # unit descale. Called, not inlined, so the runtime branches that
            # use it do not carry ``quant`` as branch state.
            quant.store_pad(row, sq_row, sq_head)

        def xcd_remap(index):
            # A bijection of [0, grid) (padded to xcd_rows * xcds) that puts
            # xcd_rows consecutive rows (one request's MTP tokens, which share
            # most selected pages) on the XCD that dispatches them, so they
            # share its L2: dispatch index i runs on XCD i % xcds.
            xcd, slot = index % t.xcds, index // t.xcds
            return ((slot // t.xcd_rows) * t.xcds + xcd) * t.xcd_rows + slot % t.xcd_rows

        task_slot = fx.Int32(gpu.block_id("x"))
        row_block = task_slot
        page_ids = None

        lds_base = None
        if const_expr(t.plan_free):
            # The pipeline's LDS, taken here so the page ids can land in it
            # before the CTA branches on the context length.
            lds_base = fx.recast_iter(
                fx.Uint8,
                fx.SharedAllocator().allocate(SharedStorage).peek().buf.ptr,
            )

        def preload_pages(seq_id):
            # Lane j of each wave copies the sequence's page j (<= 32 pages)
            # into the wave's LDS slot with an async load: it has a side
            # effect, so it issues here, overlapping the context length load
            # the branch below waits for (a register load would sink into the
            # branch, a second round trip). The pipeline reads the slot.
            tid = fx.Int32(gpu.thread_id("x"))
            entry = seq_id * max_blocks + fx.min(tid % t.WAVE, max_blocks - 1)
            src = create_llvm_ptr(
                buf_base_i64(table) + fx.Int64(entry) * 4, address_space=1
            )
            dst = create_llvm_ptr(
                fx.Int64(fx.ptrtoint(lds_base))
                + fx.Int64(t.page_ids_offset + tid * 4),
                address_space=3,
            )
            fx.rocdl.global_load_async_to_lds_b32(src, dst, 0)
            return None

        # Tasks a sequence of the cluster merge: R = cluster_rows sequences
        # share a cluster of R x P CTAs (plan-free launches only).
        parts = t.cluster_merge // t.cluster_rows
        if const_expr(t.cluster_merge):
            # Cluster s runs sequence s's tasks on its leading CTAs.
            seq, part = task_slot // t.cluster_merge, task_slot % t.cluster_merge
            if const_expr(t.xcd_rows):
                seq = xcd_remap(seq)
            if const_expr(t.cluster_rows > 1):
                seq, part = seq * t.cluster_rows + part // parts, part % parts
            plan_seq = seq
            if const_expr(t.quant_output or t.xcd_rows or t.plan_free):
                # Clusters past the sequences write padded FP8 rows (below);
                # they read sequence 0's entry and run no task.
                pad_cluster = seq >= sequences
                plan_seq = pad_cluster.select(fx.Int32(0), seq)
            if const_expr(t.plan_free):
                # No plan: ``work`` holds the context lengths and the sequence's
                # min(P, tiles) tasks split its tiles evenly, task p in slot
                # seq * P + p of the [KV heads, sequences * P] partials.
                page_ids = preload_pages(plan_seq)
                context = pad_cluster.select(
                    fx.Int32(0), fx.Int32(fx.recast_iter(fx.Int32, work)[plan_seq])
                )
                tiles = (context + (KV_COMPUTE_BLOCK - 1)) // KV_COMPUTE_BLOCK
                count = fx.min(tiles, fx.Int32(parts))
                first = plan_seq * parts
                n_tasks = fx.max(count, fx.Int32(1))
                task = [
                    plan_seq,
                    part * tiles // n_tasks,
                    (part + 1) * tiles // n_tasks,
                    context,
                ]
                task_slot = first + part
            else:
                seq_tasks = fx.ptr_load(
                    fx.add_offset(fx.recast_iter(fx.Int32, reduce_info), plan_seq * 2),
                    result_type=fx.Vector.make_type(2, fx.Int32),
                )
                first, count = fx.Int32(seq_tasks[0]), fx.Int32(seq_tasks[1])
                if const_expr(t.quant_output or t.xcd_rows):
                    count = pad_cluster.select(fx.Int32(0), count)
                task_slot = (part < count).select(first + part, fx.Int32(0))
        if const_expr(t.plan_free and t.direct_output):
            # CTA = sequence: ``work`` holds the context lengths, no plan
            # lookups. The length and the sequence's page ids load together,
            # so the first page DMAs issue one memory round trip after launch.
            if const_expr(t.xcd_rows):
                task_slot = xcd_remap(task_slot)
            row_block = task_slot
            in_range = task_slot < sequences
            seq_id = in_range.select(task_slot, fx.Int32(0))
            page_ids = preload_pages(seq_id)
            context = in_range.select(
                fx.Int32(fx.recast_iter(fx.Int32, work)[seq_id]), fx.Int32(0)
            )
            tiles = (context + (KV_COMPUTE_BLOCK - 1)) // KV_COMPUTE_BLOCK
            task = [seq_id, fx.Int32(0), tiles, context]
            active = tiles > 0
        elif const_expr(t.direct_output):
            # One task per nonempty sequence; reduce_info supplies its packed slot.
            info = fx.recast_iter(fx.Int32, pmax)
            if const_expr(t.xcd_rows):
                task_slot = xcd_remap(task_slot)
                row_block = task_slot
            if const_expr(t.quant_output or t.xcd_rows):
                # CTAs past the sequences write the padded FP8 rows (below);
                # they read sequence 0's entry and run no task.
                pad_cta = task_slot >= sequences
                plan_slot = pad_cta.select(fx.Int32(0), task_slot)
                count = pad_cta.select(fx.Int32(0), fx.Int32(info[plan_slot * 2 + 1]))
            else:
                plan_slot = task_slot
                count = fx.Int32(info[task_slot * 2 + 1])
            # A trailing empty sequence's start can equal capacity. Its dummy
            # task read must stay in bounds even though no compute consumes it.
            task_slot = (count > 0).select(fx.Int32(info[plan_slot * 2]), fx.Int32(0))
        if const_expr(t.plan_free and t.cluster_merge):
            active = part < count
        elif const_expr(not t.plan_free):
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
            ctx.page_ids = page_ids
            ctx.lds_base = lds_base
            ctx.n_kv = kv_heads
            ctx.quant = quant
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
            if const_expr(t.nvfp4_wave):
                Gfx1250Nvfp4WavePipeline(ctx).run()
            elif const_expr(t.bf16_wave_query_tiles):
                Gfx1250Bf16WavePipeline(ctx).run()
            elif const_expr(t.bf16_page128_packed):
                Gfx1250Bf16Pipeline(ctx).run()
            elif const_expr(t.wave_query_tiles):
                Gfx1250WavePipeline(ctx).run()
            else:
                Gfx1250Pipeline(ctx).run()
        elif const_expr(t.plan_free and t.direct_output and not t.quant_output):
            if row_block < sequences:
                zero_output(row_block, pout)
        elif const_expr(t.direct_output):
            if const_expr(t.quant_output):
                if row_block < sequences:
                    zero_output(row_block, pout)
                elif row_block < quant_rows:
                    pad_output(row_block)
            elif const_expr(t.xcd_rows):
                if row_block < sequences:
                    zero_output(row_block, pout)
            else:
                zero_output(gpu.block_id("x"), pout)
        if const_expr(t.merge_partitions):
            # Empty sequences own no task. Capacity >= batch gives each one the
            # CTA at its index, which clears its rows.
            info = fx.recast_iter(fx.Int32, reduce_info)
            if task_slot < sequences:  # noqa: SIM102
                if fx.Int32(info[task_slot * 2 + 1]) == 0:
                    zero_output(task_slot, output)
        if const_expr(t.cluster_merge and t.cluster_rows > 1):
            # The cluster's sequences may need different merges, so every CTA
            # takes the barrier unconditionally: exactly once per CTA.
            merge_all = Gfx1250ClusterMerge(
                t,
                pmax,
                psum,
                pout,
                output,
                sq_row,
                sq_head,
                quant if t.quant_output else None,
            )
            merge_all.run(seq, part, first, count, capacity)
        if const_expr(t.cluster_merge):
            # Every CTA of a cluster reaches its barrier exactly once.
            if count > 1:
                if const_expr(t.cluster_rows == 1):
                    merge = Gfx1250ClusterMerge(
                        t,
                        pmax,
                        psum,
                        pout,
                        output,
                        sq_row,
                        sq_head,
                        quant if t.quant_output else None,
                    )
                    merge.run(seq, part, first, count, capacity)
            elif const_expr(t.quant_output):
                if seq >= sequences:
                    row = sequences + (seq - sequences) * parts + part
                    if row < quant_rows:
                        pad_output(row)
                elif (count == 0) & (part == 0):
                    zero_output(seq, output)
            elif const_expr(t.xcd_rows or t.plan_free):
                # Padded grids hold sequence slots past the sequences.
                if (count == 0) & (part == 0) & (seq < sequences):
                    zero_output(seq, output)
            elif (count == 0) & (part == 0):
                zero_output(seq, output)
        if const_expr(t.plan_free):
            # The page-id copy has landed before any wave leaves the CTA.
            fx.rocdl.s_wait_asynccnt(0)

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
        quantized_output: fx.Pointer,
        output_scale: fx.Pointer,
        quant_rows: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = capacity
        pad = t.xcd_rows * t.xcds
        if const_expr(t.direct_output):
            grid_x = sequences
            if const_expr(t.quant_output):
                # One CTA per padded FP8 row past the sequences.
                grid_x = quant_rows
            if const_expr(t.xcd_rows):
                grid_x = (grid_x + pad - 1) // pad * pad
        if const_expr(t.cluster_merge):
            parts = t.cluster_merge // t.cluster_rows
            slots = sequences
            if const_expr(t.quant_output):
                # One sequence slot (P CTAs) per P padded FP8 rows past them.
                slots = sequences + (quant_rows - sequences + parts - 1) // parts
            clusters = (slots + t.cluster_rows - 1) // t.cluster_rows
            if const_expr(t.xcd_rows):
                clusters = (clusters + pad - 1) // pad * pad
            grid_x = clusters * t.cluster_merge
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
            quantized_output,
            output_scale,
            quant_rows,
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
    if not t.merged_output or t.plan_free:
        # Argument loads that a preload leaves behind are slower than those of
        # an unpreloaded kernel (cluster merges ran 2-6% slower), so merging
        # kernels, whose arguments do not all fit, keep loading them. Plan-free
        # merges read their first arguments preloaded (+-0.1 us, MI455X).
        llvm_options["amdgpu-kernarg-preload"] = AITER_FLYDSL_KERNARG_PRELOAD
        llvm_options["amdgpu-kernarg-preload-count"] = (
            AITER_FLYDSL_KERNARG_PRELOAD_COUNT
        )
    if llvm_options:
        launch.compile_hints["llvm_options"] = llvm_options
    compiled = {"launch": launch, "kernel": pa_decode_tile_kernel}
    return _GFX1250_PA_DECODE_CACHE.setdefault(t, compiled)

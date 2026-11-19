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
caches run in wave pairs that each cover half a page and keep P in registers. Supported layouts
use TDM K/V transfers, with two LDS buffers where capacity permits.
One-partition plans can write final output directly. Plans with one task slot
per sequence merge a sequence's few partitions in its last-arriving CTA, and
plans whose sequences each fit one workgroup cluster merge within that
cluster; other plans use the shared partition reducer.
BF16 transposed page128 caches at D64/D128/D256 use a packed-page pipeline
with resident Q and fused PV accumulation; D256 is limited to 32 query rows.
Page16 BF16 caches at these dimensions retain vector-8 packing in LDS while
using the 64-token pipeline.

MXFP4 K/V (``kv_dtype=dtypes.fp4x2``, per-token KV, gfx950/gfx1250, head_dim 128, page
128; one query token on gfx950, MTP and sliding windows on gfx1250) share this
scaffolding and have their own tile: QK is one scaled MFMA/WMMA (fp4 K x e4m3 Q). gfx950
widens V to bf16 through wave-private LDS; gfx1250 dequantizes V with the scaled WMMA
against fp4 identity blocks, which leaves it transposed in registers for the bf16 PV
WMMA. A wave owns 64 tokens of the tile and all dims, so the tile loop has no barrier;
the waves merge once after it.

NVFP4 K/V (``kv_dtype=dtypes.fp4x2`` with ``kv_scale_dtype=torch.float8_e4m3fn``,
gfx1250, head_dim 128, page 128, one query tile of at most 16 rows) are e2m1 with one
E4M3 scale per 16 values, stored token-major per (page, KV head): [128, 64] payload
and [128, 8] scale bytes. ``Gfx1250Nvfp4WavePipeline`` runs them: each wave owns 32
tokens of every page, DMAs them itself, computes QK and dequantizes V with
``v_wmma_scale16_f32_16x16x128_f8f6f4`` straight from the raw rows and keeps softmax
and P in registers, so the page loop has no CTA barrier.

Logical layouts (not preshuffled):

* ``query``        [num_seqs * query_length, num_q_heads, head_dim]  f16/bf16
* ``key_cache``    [num_blocks, num_kv_heads, head_dim//x, block_size, x]
                   or NHD [num_blocks, 64, num_kv_heads, 256] (strided, head contiguous;
                   gfx950 FP8 head 256, page 64, at most two query M-tiles)
* ``value_cache``  [num_blocks, num_kv_heads, block_size//x, head_dim, x] (trans_v)
                   or [num_blocks, num_kv_heads, head_dim, block_size] (FP8 plain)
                   where x=16 for FP8 and x=8 for BF16; with NHD K, an NHD view of
                   the same shape: head_dim contiguous bytes per token, often the
                   head_dim bytes after K in one token row
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

import dataclasses
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import get_gfx_runtime

from .pa_decode.context import PaDecodeContext
from .pa_decode.op_merge import (
    MERGE_HEAD_DIM,
    MERGE_PARTITIONS,
    Gfx1250ClusterMerge,
    Gfx1250OutputQuant,
)
from .pa_decode.pipeline import (
    Gfx1250Bf16Pipeline,
    Gfx1250Bf16WavePipeline,
    Gfx1250Mxfp4Pipeline,
    Gfx1250Pipeline,
    Gfx1250WavePipeline,
    PaDecodePipeline,
)
from .pa_decode.pipeline_nvfp4 import Gfx1250Nvfp4WavePipeline
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
from .pa_decode_plan import CLUSTER_PARTITIONS, CLUSTER_STEP_TOKENS
from .tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

__all__ = [
    "CLUSTER_PARTITIONS",
    "CLUSTER_STEP_TOKENS",
    "FP8_PACK_K",
    "KV_COMPUTE_BLOCK",
    "LOG2E",
    "MERGE_HEAD_DIM",
    "MERGE_PARTITIONS",
    "MFMA_ACC_ELEMS",
    "MFMA_MNK",
    "WAVE",
    "compile_pa_decode_tile",
    "gfx1250_tile_traits",
]

# gfx1250 MXFP4: query rows (positions x GQA heads) a CTA keeps in registers by
# default, and at most.
MXFP4_GFX1250_ROWS = 32
MXFP4_GFX1250_MAX_ROWS = 64
# chip_info maps gfx1250-strict (MI450 A0) to gfx1250 only at dispatch sites
# validated on it. The MXFP4 kernel was not run on A0, nor its scaled WMMAs
# checked against A0's S_CLAUSE erratum.
MXFP4_GFX1250_STRICT_REFUSAL = (
    "MXFP4 pa_decode is not validated on gfx1250-strict (MI450 A0)"
)


def mxfp4_gfx1250_query_splits(query_length, query_group_size):
    """Default gfx1250 MXFP4 query splits: each 16-row M-tile adds its Q,
    scores and outputs to the registers, so split MTP positions until a CTA
    holds at most MXFP4_GFX1250_ROWS rows. Its M-tiles share the tile's V
    dequantization."""
    return next(
        (
            splits
            for splits in (1, 2, 4)
            if query_length % splits == 0
            and query_length // splits * query_group_size <= MXFP4_GFX1250_ROWS
        ),
        max(s for s in (1, 2, 4) if query_length % s == 0),
    )


def mxfp4_gfx1250_resident_ctas(rows, num_compute_units):
    """gfx1250 MXFP4 CTAs of ``rows`` query rows resident at once: two a CU up to
    MXFP4_GFX1250_ROWS rows (2 M-tiles, at most 441 VGPRs), one above (3 and 4
    M-tiles take 579 and 710 VGPRs)."""
    return num_compute_units * (2 if rows <= MXFP4_GFX1250_ROWS else 1)


# Key by selected specialization, not batch/head/CU scheduling inputs.
_PA_DECODE_TILE_CACHE = {}
_GFX1250_PA_DECODE_CACHE = {}
# Page slots in each wave's DMA ring of the NVFP4 wave pipeline (2-4; 3 measured
# within noise of 4 at less LDS and fewer VGPRs).
NVFP4_BUFFERS = int(os.environ.get("AITER_FLYDSL_NVFP4_PA_BUFFERS", "3"))
if NVFP4_BUFFERS not in (2, 3, 4):
    raise ValueError("AITER_FLYDSL_NVFP4_PA_BUFFERS must be 2, 3 or 4")


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
    fuse_tasks=1,
    wave_partials=False,
    wave_pairs=False,
    wave_dims=False,
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
        name += "_quant" if store_output else "_quantonly"
    if fuse_tasks > 1:
        name += f"_fuse{fuse_tasks}"
    if wave_partials:
        name += "_wp"
    if wave_pairs:
        name += "_pairs"
    if wave_dims:
        name += "_dims"
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
    cluster_seqs: int = 1,
    cluster_steps: bool = False,
    cluster_runs: bool = False,
    quant_output: bool = False,
    store_output: bool = True,
    fuse_tasks: int = 1,
    wave_partials: bool = False,
    wave_pairs: bool = False,
    wave_dims: bool = False,
    kv_scale_dtype: torch.dtype | None = None,
):
    """Select and cache a PA-decode kernel and launch wrapper.

    ``kv_dtype`` is the KV cache's torch dtype: ``dtypes.fp8`` (the arch's e4m3
    flavor: FNUZ on gfx942, OCP elsewhere), ``torch.bfloat16``, or
    ``dtypes.fp4x2`` for MXFP4. ``dtypes.fp4x2`` is ``torch.uint8`` on torch
    builds without ``float4_e2m1fn_x2``, so a uint8 ``kv_dtype`` also selects
    MXFP4. ``query_dtype`` is the query's: f16 or bf16, or e4m3fn with MXFP4 KV.
    ``kv_scale_dtype=torch.float8_e4m3fn`` with an fp4x2/uint8 ``kv_dtype`` selects
    NVFP4 (gfx1250 only): E4M3 block scales of 16 values instead of MXFP4's E8M0
    of 32.
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
    ``trans_k=False`` requires ``trans_v=False`` and selects NHD caches
    (gfx950 only).
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
    remain finite; NHD V is also zero-filled past the context by its DMA. The
    gfx1250 pipelines ignore the flag and always require it.
    ``wide_kv_addressing`` protects
    offsets at 2 GiB; ``kv_buffer_u32`` proves both FP8 caches are below 4 GiB
    and permits unsigned buffer offsets instead of i64. For NHD it also widens
    the head term, and the in-page term stays i32.
    """
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
        if not trans_k or not trans_v or per_token_kv:
            raise ValueError("NVFP4 PA decode takes HND K/V without per-token scales")
        t = gfx1250_tile_traits(
            head_dim=head_dim,
            query_group_size=query_group_size,
            block_size=block_size,
            num_seqs=num_seqs,
            num_kv_heads=num_kv_heads,
            num_compute_units=num_compute_units,
            work_capacity=work_capacity,
            num_partitions=num_partitions,
            softmax_scale=softmax_scale,
            query_dtype=query_dtype,
            kv_dtype=torch.bfloat16,
            query_length=query_length,
            query_splits=1,
            max_context_length=max_context_length,
            sliding_window=sliding_window,
            direct_output=direct_output,
            cluster_merge=cluster_merge,
            cluster_seqs=cluster_seqs,
            quant_output=quant_output,
            store_output=store_output,
        )
        # The wave pipeline stages its own pages: no BF16 step or staging
        # policy applies.
        return compile_gfx1250_pa_decode(
            dataclasses.replace(
                t,
                compute_dtype="nvfp4w",
                nvfp4_buffers=NVFP4_BUFFERS,
                wide_bf16_page16=False,
                wide_fp8_page128=False,
                wide_bf16_page128=False,
                narrow_fp8_page16=False,
            )
        )
    if architecture in ("gfx1250", "gfx1250-strict"):
        if not trans_k:
            raise NotImplementedError("NHD pa_decode requires gfx950")
        kv_fx, query_fx = pa_fx_dtypes(kv_dtype, query_dtype)
        if kv_fx == MXFP4_KV_DTYPE:
            if architecture == "gfx1250-strict":
                raise NotImplementedError(MXFP4_GFX1250_STRICT_REFUSAL)
            if query_fx not in (fx.BFloat16, fx.Float8E4M3FN):
                raise ValueError("MXFP4 KV requires a BF16 or FP8 query")
        elif query_fx not in (fx.Float16, fx.BFloat16) or kv_fx not in (
            fx.Float8E4M3FN,
            fx.BFloat16,
        ):
            raise ValueError(
                "gfx1250 PA decode takes an f16/bf16 query and e4m3fn/bf16 KV, "
                f"got {query_dtype} and {kv_dtype}"
            )
        return compile_gfx1250_pa_decode(
            gfx1250_tile_traits(
                head_dim=head_dim,
                query_group_size=query_group_size,
                block_size=block_size,
                num_seqs=num_seqs,
                num_kv_heads=num_kv_heads,
                num_compute_units=num_compute_units,
                work_capacity=work_capacity,
                num_partitions=num_partitions,
                softmax_scale=softmax_scale,
                query_dtype=query_dtype,
                kv_dtype=kv_dtype,
                per_token_kv=per_token_kv,
                query_length=query_length,
                trans_v=trans_v,
                query_splits=query_splits,
                max_context_length=max_context_length,
                sliding_window=sliding_window,
                direct_output=direct_output,
                cluster_merge=cluster_merge,
                cluster_seqs=cluster_seqs,
                cluster_steps=cluster_steps,
                cluster_runs=cluster_runs,
                quant_output=quant_output,
                store_output=store_output,
                fuse_tasks=fuse_tasks,
                wave_partials=wave_partials,
                wave_pairs=wave_pairs,
                wave_dims=wave_dims,
            )
        )
    if architecture not in ("gfx942", "gfx950"):
        raise NotImplementedError(f"PA decode does not support {architecture}")
    if cluster_merge or quant_output or not store_output:
        raise NotImplementedError(
            "cluster_merge, quant_output and a skipped store require gfx1250"
        )
    if fuse_tasks != 1 or wave_partials or wave_pairs or wave_dims:
        raise NotImplementedError(
            "fuse_tasks, wave_partials, wave_pairs and wave_dims require gfx1250 MXFP4"
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
        stride_k_block: fx.Int32,
        stride_k_token: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_block: fx.Int32,
        stride_v_token: fx.Int32,
        stride_v_head: fx.Int32,
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
            stride_k_block=stride_k_block,
            stride_k_token=stride_k_token,
            stride_k_head=stride_k_head,
            stride_v_block=stride_v_block,
            stride_v_token=stride_v_token,
            stride_v_head=stride_v_head,
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
        stride_k_block: fx.Int32,
        stride_k_token: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_block: fx.Int32,
        stride_v_token: fx.Int32,
        stride_v_head: fx.Int32,
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
                stride_k_block,
                stride_k_token,
                stride_k_head,
                stride_v_block,
                stride_v_token,
                stride_v_head,
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
            kv_query = fx.Int32(gpu.block_id("x" if traits.SPLIT_MAJOR else "y"))
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
        slot = fx.Int32(gpu.block_id("y" if traits.SPLIT_MAJOR else "x"))
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
            stride_k_block: fx.Int32,
            stride_k_token: fx.Int32,
            stride_k_head: fx.Int32,
            stride_v_block: fx.Int32,
            stride_v_token: fx.Int32,
            stride_v_head: fx.Int32,
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
                stride_k_block,
                stride_k_token,
                stride_k_head,
                stride_v_block,
                stride_v_token,
                stride_v_head,
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
            stride_k_block,
            stride_k_token,
            stride_k_head,
            stride_v_block,
            stride_v_token,
            stride_v_head,
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
                stride_k_block,
                stride_k_token,
                stride_k_head,
                stride_v_block,
                stride_v_token,
                stride_v_head,
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
            stride_k_block: fx.Int32,
            stride_k_token: fx.Int32,
            stride_k_head: fx.Int32,
            stride_v_block: fx.Int32,
            stride_v_token: fx.Int32,
            stride_v_head: fx.Int32,
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
                stride_k_block,
                stride_k_token,
                stride_k_head,
                stride_v_block,
                stride_v_token,
                stride_v_head,
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
        stride_k_block: fx.Int32,
        stride_k_token: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_block: fx.Int32,
        stride_v_token: fx.Int32,
        stride_v_head: fx.Int32,
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
                    stride_k_block,
                    stride_k_token,
                    stride_k_head,
                    stride_v_block,
                    stride_v_token,
                    stride_v_head,
                    work_info,
                    num_seqs,
                )
            ).launch(
                grid=(
                    (num_kv_heads * traits.query_splits, work_capacity, 1)
                    if traits.SPLIT_MAJOR
                    else (work_capacity, num_kv_heads * traits.query_splits, 1)
                ),
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
    if traits.FP8_MTP4_LEAN or traits.WAVES_PER_EU_2:
        # Two CTAs per CU: hold the allocator to 256 registers per wave.
        pa_decode_tile_launch.compile_hints["waves_per_eu"] = 2
    compiled = {
        "launch": pa_decode_tile_launch,
        "kernel": pa_decode_tile_kernel,
    }
    return _PA_DECODE_TILE_CACHE.setdefault(traits.cache_key, compiled)


def gfx1250_tile_traits(
    *,
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
    query_splits: int | None = None,
    max_context_length: int | None = None,
    sliding_window: int = 0,
    direct_output: bool = False,
    cluster_merge: int = 0,
    cluster_seqs: int = 1,
    cluster_steps: bool = False,
    cluster_runs: bool = False,
    quant_output: bool = False,
    store_output: bool = True,
    fuse_tasks: int = 1,
    wave_partials: bool = False,
    wave_pairs: bool = False,
    wave_dims: bool = False,
) -> Gfx1250Traits:
    """Select the gfx1250 tile ``compile_pa_decode_tile`` compiles.

    Arguments are those of ``compile_pa_decode_tile``; host-known grid bounds
    choose step widths and staging. Selection neither validates nor compiles.
    MXFP4 KV (``kv_dtype=dtypes.fp4x2``) takes ``direct_output`` as its
    plan-free form (``plan_free``).
    """
    kv_fx, query_fx = pa_fx_dtypes(kv_dtype, query_dtype)
    compute_dtype = _dtype_tag(kv_fx)
    mxfp4 = compute_dtype == "mxfp4"
    plan_free = False
    if mxfp4:
        # direct_output is the plan-free form: one task per sequence. MXFP4
        # KV is per-token KV, its e8m0 bytes being each row's scales; as on
        # gfx950 (PaDecodeTraits.create), the kernel keeps the f32 per-token
        # scale path off, so either per_token_kv selects the same kernel.
        plan_free, direct_output, per_token_kv = direct_output, False, False
        sliding_window = max(sliding_window, 0)
        if wave_pairs:
            # One CTA for 33-64 rows: wave pairs split the rows, not the CTA.
            query_splits = 1
        if query_splits is None:
            query_splits = mxfp4_gfx1250_query_splits(query_length, query_group_size)
    if query_splits is None:
        # Split the measured spilling BF16 specialization into one Q tile
        # per CTA. Other shapes retain fused queries and their KV reuse.
        query_splits = (
            4
            if compute_dtype == "bf16"
            and head_dim == 1024
            and query_group_size == 16
            and query_length == 4
            and block_size == 128
            and trans_v
            and sliding_window == 0
            else 1
        )
    grid_sequences = (
        num_seqs
        if direct_output and num_partitions == 1
        else (
            -(-num_seqs // cluster_seqs) * cluster_seqs * cluster_merge
            if cluster_merge
            else work_capacity
        )
    )
    grid_ctas = grid_sequences * num_kv_heads * query_splits
    one_cta_grid = grid_ctas <= num_compute_units
    # Two-page FP8 steps pay off with one CTA per CU, or when CTAs run several
    # tiles; two-CTA grids of one-tile tasks lose overlap.
    wide_pages = one_cta_grid or (
        grid_ctas <= 2 * num_compute_units
        and max_context_length is not None
        and num_seqs * -(-max_context_length // KV_COMPUTE_BLOCK) >= 3 * grid_sequences
    )
    return Gfx1250Traits(
        head_dim,
        query_group_size,
        block_size,
        head_dim**-0.5 if softmax_scale is None else softmax_scale,
        _dtype_tag(query_fx),
        per_token_kv,
        query_length,
        trans_v,
        sliding_window,
        compute_dtype,
        direct_output and num_partitions == 1,
        query_splits,
        direct_output and num_partitions > 1,
        cluster_merge,
        # Wide page16 stages leave room for two CTAs per CU: larger grids
        # (sparse prefill) keep narrower stages.
        grid_ctas <= 2 * num_compute_units
        and compute_dtype == "bf16"
        and block_size == 16,
        wide_pages and compute_dtype == "fp8" and block_size == 128,
        # Step-split clusters run 128-token steps, which their runs and
        # preloaded page ids assume.
        one_cta_grid and compute_dtype == "bf16" and not cluster_steps,
        grid_ctas > 4 * num_compute_units
        and compute_dtype == "fp8"
        and block_size == 16,
        cluster_seqs=cluster_seqs if cluster_merge else 1,
        # MXFP4 tiles stage no page16 copies: one kernel serves every grid.
        latency_grid=one_cta_grid and not mxfp4,
        cluster_steps=cluster_steps,
        cluster_runs=cluster_runs,
        quant_output=quant_output,
        store_output=store_output,
        plan_free=plan_free,
        fuse_tasks=fuse_tasks,
        wave_partials=wave_partials,
        wave_pairs=wave_pairs,
        wave_dims=wave_dims,
        resident_ctas=(
            mxfp4_gfx1250_resident_ctas(
                query_length * query_group_size // query_splits, num_compute_units
            )
            if fuse_tasks > 1
            else 0
        ),
    )


def compile_gfx1250_pa_decode(t):
    """Compile a gfx1250 tile kernel; merging partitions writes final output.

    ``t`` holds the ``gfx1250_tile_traits`` selection.
    ``merge_partitions`` plans keep packed partials and count arrivals per
    (sequence, KV head) in the plan's zeroed ``arrival_counters``. A positive
    ``cluster_merge`` P instead launches P CTAs per sequence as one workgroup
    cluster: a sequence with n tasks splits its tiles over its first min(n, P)
    CTAs, as the planner splits them, and the cluster merges after a cluster
    barrier. ``work`` then holds the context lengths. ``cluster_steps`` instead
    gives CTA p steps [p * 16 / P, (p + 1) * 16 / P) of block tables at most
    16 steps wide, ignoring the plan's task counts, with partials at slot
    sequence * P + p. ``cluster_runs`` (page128) likewise gives CTA p tiles
    [p * run_tiles, (p + 1) * run_tiles), ``run_tiles`` sized by the host from
    the length bound. In all modes a sequence's single task stores final rows
    directly, and empty sequences clear their rows.
    ``merge_partitions`` partial storage must be eight-byte aligned for its
    device-scope atomic accesses; the host wrapper falls back for offset views.

    ``compute_dtype="mxfp4"`` (``Gfx1250Mxfp4Pipeline``) takes E2M1 K/V data and
    E8M0 scale views (per-token KV), head_dim 128, page 128, MTP, sliding windows
    and at most 64 rows a CTA, and writes packed partials for the reducer or a
    cluster merge;
    ``plan_free`` instead reads ``work`` as the context lengths, runs sequence
    s whole in CTA s and stores its final rows (an empty sequence's are zeroed).
    ``fuse_tasks=2``
    (MXFP4 partials) halves the grid. When the plan's live tasks need more CTAs
    than ``resident_ctas``, CTA b runs plan slots 2 b and 2 b + 1: adjacent
    tiles of one sequence run as one task whose partial is slot 2 b's (slot
    2 b + 1's holds no mass), others run one after the other. Otherwise CTA b
    runs slot b alone.
    """
    if t.query_splits not in (1, 2, 4) or t.query_length % t.query_splits:
        raise ValueError("query_splits must be 1, 2, or 4 and divide query_length")
    if t.compute_dtype not in ("fp8", "bf16", "mxfp4", "nvfp4w"):
        raise ValueError(f"Unsupported gfx1250 PA compute dtype: {t.compute_dtype}")
    if t.plan_free and not t.mxfp4:
        raise ValueError("plan_free requires MXFP4 KV")
    if t.wave_partials and not (
        t.mxfp4 and t.cluster_merge and t.cluster_merge * t.NWARP <= t.WAVE
    ):
        raise ValueError(
            "wave_partials requires an MXFP4 cluster merge of at most "
            f"{32 // 4} CTAs (a lane per partial)"
        )
    if t.wave_pairs and not (
        t.mxfp4
        and t.query_splits == 1
        and 32 < t.rows <= 64
        and t.fuse_tasks == 1
        and not t.wave_partials
    ):
        raise ValueError(
            "wave_pairs requires one split of 33-64 MXFP4 rows, without fused "
            "tasks or wave partials"
        )
    if t.wave_dims and not (
        t.mxfp4
        and t.rows <= 16
        and t.fuse_tasks == 1
        and not t.wave_partials
        and not t.wave_pairs
    ):
        raise ValueError(
            "wave_dims requires one M-tile of MXFP4 rows, without fused tasks, "
            "wave partials or wave pairs"
        )
    if t.fuse_tasks not in (1, 2) or (
        t.fuse_tasks > 1 and (not t.mxfp4 or t.plan_free or t.cluster_merge)
    ):
        raise ValueError(
            "fuse_tasks is 1, or 2 for MXFP4 tasks that write reducer partials"
        )
    if t.mxfp4 and not (
        t.head_dim == 128
        and t.block_size == 128
        and 1 <= t.query_group_size <= 16
        and t.rows <= MXFP4_GFX1250_MAX_ROWS
        and t.query_dtype in ("bf16", "fp8")
        and not t.per_token_kv
        and not (t.direct_output or t.merge_partitions)
        and not (t.plan_free and t.cluster_merge)
        and not (t.cluster_steps or t.cluster_seqs > 1)
        and not t.quant_output
        and t.store_output
    ):
        raise NotImplementedError(
            "gfx1250 MXFP4 PA decode supports head_dim 128, block 128, 1..16 query "
            f"heads per KV head, at most {MXFP4_GFX1250_MAX_ROWS} query rows a CTA "
            "(query positions / query_splits x heads), a bf16 or e4m3 query, its "
            "in-cache e8m0 scales (no f32 per-token tensors) and reducer, "
            "cluster-merged (one sequence a cluster, plan or run tiles) or "
            "plan-free bf16 output (no FP8 output quantization)"
        )
    if t.query_dtype == "fp8" and not t.mxfp4:
        raise ValueError("An FP8 query needs MXFP4 KV")
    if t.nvfp4_wave and not (
        t.head_dim == 128
        and t.block_size == 128
        and t.trans_v
        and not t.per_token_kv
        and t.sliding_window == 0
        and t.query_splits == 1
        and t.m_tiles == 1
        and not (t.cluster_steps or t.cluster_runs)
        and 2 <= t.nvfp4_buffers <= 4
    ):
        # A wave owns 32 tokens of a 128-token page and all 16 query rows.
        raise ValueError(
            "NVFP4 PA decode supports head_dim 128, page 128, one query tile "
            "(query_length x GQA <= 16), no sliding window, query split or "
            "step/run clusters, and a 2-4 page ring"
        )
    if t.bf16 and t.per_token_kv:
        raise ValueError("BF16 PA decode does not use quantization scales")
    if t.merge_partitions and (t.direct_output or t.head_dim > MERGE_HEAD_DIM):
        raise ValueError(
            f"merge_partitions excludes direct_output and head_dim > {MERGE_HEAD_DIM}"
        )
    if t.cluster_merge and (
        t.direct_output
        or t.merge_partitions
        or t.head_dim > MERGE_HEAD_DIM
        or not 2 <= t.cluster_merge <= CLUSTER_PARTITIONS
    ):
        raise ValueError(
            f"cluster_merge must be in [2, {CLUSTER_PARTITIONS}] and excludes other "
            f"output modes and head_dim > {MERGE_HEAD_DIM}"
        )
    if t.cluster_steps and not (
        t.cluster_merge
        and CLUSTER_PARTITIONS % t.cluster_merge == 0
        and t.block_size == 16
        and t.query_length == 1
        and t.sliding_window == 0
        and t.TOKENS == CLUSTER_STEP_TOKENS
    ):
        raise ValueError(
            "cluster_steps requires a cluster dividing "
            f"{CLUSTER_PARTITIONS} CTAs over page16 decode rows without a "
            f"sliding window and {CLUSTER_STEP_TOKENS}-token steps"
        )
    if t.quant_output and (
        not (t.direct_output or t.cluster_merge)
        or t.merge_partitions
        or t.query_splits != 1
        or t.cluster_seqs != 1
    ):
        raise ValueError(
            "quant_output requires the direct or cluster_merge output, one "
            "query split and one sequence per cluster"
        )
    if not (t.store_output or t.quant_output):
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
        kv_dtype=t.compute_dtype,
        query_dtype=t.query_dtype,
        head_dim=t.head_dim,
        block_size=t.block_size,
        query_group_size=t.query_group_size,
        query_length=t.query_length,
        per_token_kv=t.per_token_kv,
        trans_v=t.trans_v,
        sliding_window=t.sliding_window,
        query_splits=t.query_splits,
        direct_output=t.direct_output or t.plan_free,
        merge_partitions=t.merge_partitions,
        cluster_merge=t.cluster_merge,
        quant_output=t.quant_output,
        store_output=t.store_output,
        fuse_tasks=t.fuse_tasks,
        wave_partials=t.wave_partials,
        wave_pairs=t.wave_pairs,
        wave_dims=t.wave_dims,
    )
    if t.nvfp4_wave and t.nvfp4_buffers != 3:
        kernel_name += f"_ring{t.nvfp4_buffers}"

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
        run_tiles: fx.Int32,
        quantized_output: fx.Pointer,
        output_scale: fx.Pointer,
        quant_rows: fx.Int32,
    ):

        # Optional FP8 output; the pipelines and cluster merge write it from
        # the final values in registers.
        quant = Gfx1250OutputQuant(t, quantized_output, output_scale)

        def zero_output(seq, dst):
            # Empty sequences have no attention mass; clear this CTA's rows.
            if const_expr(t.quant_output):
                quant.store_empty(
                    seq * t.query_length,
                    t.query_length,
                    fx.Float32(0.0),
                    sq_row,
                    sq_head,
                )
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

        def run_task(slot, slot_task):
            slot_ctx = PaDecodeContext(
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
                slot_task[0],
                slot_task[1],
                slot_task[2],
                slot_task[3],
            )
            slot_ctx.task_slot, slot_ctx.capacity = slot, capacity
            slot_ctx.n_kv = kv_heads
            Gfx1250Mxfp4Pipeline(slot_ctx).run()

        def empty_partial(slot):
            # The partial of a slot whose tiles the previous slot's task ran: no
            # mass (max -inf, sum 0, zero rows) for the reducer.
            tid = fx.Int32(gpu.thread_id("x"))
            kv_query = fx.Int32(gpu.block_id("y"))
            slot_rows = t.rows * t.query_splits
            slot_first = (
                fx.Int64(kv_query // t.query_splits) * fx.Int64(capacity)
                + fx.Int64(slot)
            ) * slot_rows + (kv_query % t.query_splits) * t.rows
            slot_out = fx.recast_iter(t.q_dtype, pout)
            # Views bound before the guards keep the writes SSA operands of them.
            slot_max = fx.recast_iter(fx.Float32, pmax)
            slot_sum = fx.recast_iter(fx.Float32, psum)
            for i in range_constexpr(-(-t.rows * t.head_dim // t.BLOCK_THREADS)):
                index = i * t.BLOCK_THREADS + tid
                if index < t.rows * t.head_dim:
                    slot_out[slot_first * t.head_dim + index] = t.q_dtype(0.0)
            if tid < t.rows:
                slot_max[slot_first + tid] = fx.Float32(float("-inf"))
                slot_sum[slot_first + tid] = fx.Float32(0.0)

        def pad_output(row):
            # A padded token-bucket row past the sequences' tokens: FP8 zero
            # bytes and a unit descale. Called, not inlined, so the runtime
            # branches that use it do not carry ``quant`` as branch state.
            quant.store_empty(
                row + sequences * t.query_length, 1, fx.Float32(1.0), sq_row, sq_head
            )

        if const_expr(t.fuse_tasks > 1):
            # The grid holds (capacity + 1) // 2 CTAs. The host's tile bound
            # cannot see real lengths (a table-capacity bound), so the plan's
            # live tasks decide: CTA b runs slot b alone when one slot a CTA
            # fits one round of resident CTAs, else slots 2 b and 2 b + 1.
            # Scalar loads return out of order, so all of them issue before
            # any is used.
            fuse_block = fx.Int32(gpu.block_id("x"))
            pair = fuse_block * 2
            last_tasks = fx.ptr_load(
                fx.add_offset(fx.recast_iter(fx.Int32, reduce_info), sequences * 2 - 2),
                result_type=fx.Vector.make_type(2, fx.Int32),
            )
            pair_tasks = [
                fx.ptr_load(
                    fx.add_offset(
                        fx.recast_iter(fx.Int32, work),
                        fx.min(slot, capacity - 1) * 4,
                    ),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
                for slot in (pair, pair + 1, fuse_block)
            ]
            fx.rocdl.sched_barrier(0)
            # Packed tasks: the last sequence's first slot plus its count.
            fuse_live = fx.Int32(last_tasks[0]) + fx.Int32(last_tasks[1])
            fuse_pairs = (fuse_live * kv_heads * t.query_splits > t.resident_ctas) | (
                fuse_live > (capacity + 1) // 2
            )
            pair = fuse_pairs.select(pair, fuse_block)
            pair_fields = [
                [
                    fuse_pairs.select(
                        fx.Int32(pair_tasks[0][f]), fx.Int32(pair_tasks[2][f])
                    )
                    for f in range_constexpr(4)
                ],
                [fx.Int32(pair_tasks[1][f]) for f in range_constexpr(4)],
            ]
            pair_active = [
                pair_fields[0][1] < pair_fields[0][2],
                fuse_pairs
                & (pair + 1 < capacity)
                & (pair_fields[1][1] < pair_fields[1][2]),
            ]
            # Adjacent tiles of one sequence: one task over both ranges.
            pair_fuse = (
                pair_active[0]
                & pair_active[1]
                & (pair_fields[0][0] == pair_fields[1][0])
                & (pair_fields[0][2] == pair_fields[1][1])
            )
            fuse_runs = fuse_pairs.select(
                2 - pair_fuse.select(fx.Int32(1), fx.Int32(0)), fx.Int32(1)
            )
            for pair_sub_i in range(0, fuse_runs, 1):
                pair_sub = fx.Int32(pair_sub_i)
                pair_first = pair_sub == 0
                pair_task = [
                    pair_first.select(pair_fields[0][f], pair_fields[1][f])
                    for f in range_constexpr(4)
                ]
                pair_task[2] = (pair_first & pair_fuse).select(
                    pair_fields[1][2], pair_task[2]
                )
                if pair_first.select(pair_active[0], pair_active[1]):
                    run_task(pair + pair_sub, pair_task)
                # The next task reuses the merge LDS.
                gpu.barrier()
            if pair_fuse:
                empty_partial(pair + 1)
            # The fused grid runs only plan slots, never the paths below.
            return

        task_slot = fx.Int32(gpu.block_id("x"))
        if const_expr(t.cluster_merge):
            # Cluster s runs sequence s's tasks on its leading CTAs. Their
            # tiles follow from the context length and task count exactly as
            # plan_pa_decode splits them: ``work`` holds the context lengths,
            # read alongside reduce_info instead of after it.
            seq, part = task_slot // t.cluster_merge, task_slot % t.cluster_merge
            # The last cluster pads the grid past the batch; with quant_output,
            # whole clusters past it write padded FP8 rows (below).
            pad_grid = t.cluster_seqs > 1 or t.quant_output
            if const_expr(pad_grid):
                present = seq < sequences
                seq = present.select(seq, fx.Int32(0))
            task_context = fx.Int32(fx.recast_iter(fx.Int32, work)[seq])
            step_bounds = None
            if const_expr(t.cluster_steps):
                # Block tables hold at most CLUSTER_PARTITIONS steps, so CTA p
                # takes steps [p chunk, (p + 1) chunk). Its first page ids
                # depend on no load, so they load with the context length
                # rather than after it: the wave's pages, then token 0's for
                # masked tail slices. The plan's tasks are not read.
                chunk = CLUSTER_PARTITIONS // t.cluster_merge
                first_step = part * chunk
                wave_pages = t.TOKENS // t.block_size // t.NWARP
                tbl = fx.recast_iter(fx.Int32, table)
                row = seq * max_blocks
                page = (
                    first_step * (t.TOKENS // t.block_size)
                    + fx.Int32(fx.rocdl.wave_id()) * wave_pages
                )
                # Only the active branch reads the ids, and the compiler would
                # sink their loads into it, after the length's wait; the
                # readfirstlanes are convergent, so they keep the loads here.
                stage_pages = [
                    fx.Int32(
                        fx.rocdl.readfirstlane(
                            T.i32, tbl[row + fx.min(page + j, max_blocks - 1)]
                        )
                    )
                    for j in range_constexpr(wave_pages)
                ] + [fx.Int32(fx.rocdl.readfirstlane(T.i32, tbl[row]))]
                # The scheduler would still place the loads after the wait.
                fx.rocdl.sched_barrier(0)
                length = fx.max(task_context, fx.Int32(0))
                steps = (length + t.TOKENS - 1) // t.TOKENS
                count = fx.min((steps + chunk - 1) // chunk, fx.Int32(t.cluster_merge))
                if const_expr(pad_grid):
                    count = present.select(count, fx.Int32(0))
                first = seq * t.cluster_merge
                task_slot = first + part
                step_bounds = (first_step * t.TOKENS, (first_step + chunk) * t.TOKENS)
                task_seq = seq
                task_begin = step_bounds[0] // KV_COMPUTE_BLOCK
                task_end = (step_bounds[1] + KV_COMPUTE_BLOCK - 1) // KV_COMPUTE_BLOCK
                active = part < count
            elif const_expr(t.cluster_runs):
                # CTA p takes tiles [p run_tiles, (p + 1) run_tiles), a run the
                # host sizes from the length bound, so its first stage's page
                # ids load with the context length rather than after it. Only
                # the active branch reads them, and the compiler would sink
                # their loads into it; readfirstlane keeps them here.
                first_tile = part * run_tiles
                run_end = first_tile + run_tiles
                if const_expr(t.mxfp4):
                    # MXFP4 attends a context longer than the bound in full:
                    # the cluster's last CTA also runs the tiles past the runs.
                    # Chosen before the length's wait, which only the min follows.
                    run_end = (part == t.cluster_merge - 1).select(
                        fx.Int32(2**30), run_end
                    )
                tbl = fx.recast_iter(fx.Int32, table)
                row = seq * max_blocks
                page = first_tile * (KV_COMPUTE_BLOCK // t.block_size)
                stage_pages = [
                    fx.Int32(
                        fx.rocdl.readfirstlane(
                            T.i32, tbl[row + fx.min(page + j, max_blocks - 1)]
                        )
                    )
                    # BF16 wave pipelines stage two pages up front.
                    for j in range_constexpr(max(t.TOKENS // t.block_size, 2))
                ] + [fx.Int32(fx.rocdl.readfirstlane(T.i32, tbl[row]))]
                # Keep the loads ahead of the length's wait.
                fx.rocdl.sched_barrier(0)
                length = fx.max(task_context, fx.Int32(0))
                tiles = (length + KV_COMPUTE_BLOCK - 1) // KV_COMPUTE_BLOCK
                count = fx.min(
                    (tiles + run_tiles - 1) // run_tiles, fx.Int32(t.cluster_merge)
                )
                first = seq * t.cluster_merge
                task_slot = first + part
                task_seq = seq
                task_begin = first_tile
                task_end = fx.min(run_end, tiles)
                active = first_tile < tiles
                if const_expr(pad_grid):
                    count = present.select(count, fx.Int32(0))
                    active = active & present
            else:
                # Scalar loads return out of order, so a use waits for every
                # load issued before it: issue both before either is used.
                seq_tasks = fx.ptr_load(
                    fx.add_offset(fx.recast_iter(fx.Int32, reduce_info), seq * 2),
                    result_type=fx.Vector.make_type(2, fx.Int32),
                )
                fx.rocdl.sched_barrier(0)
                # A sequence with more tasks than the cluster has CTAs splits its
                # tiles over the cluster, in its first task slots.
                first = fx.Int32(seq_tasks[0])
                if const_expr(t.wave_partials):
                    # A wave per slot: sequence s's waves take slots from
                    # NWARP * s * cluster_merge, which the host bounds by the
                    # scratch. The plan's first slot runs ahead of s *
                    # cluster_merge once a cluster is smaller than its tasks.
                    first = seq * t.cluster_merge
                count = fx.min(fx.Int32(seq_tasks[1]), fx.Int32(t.cluster_merge))
                if const_expr(pad_grid):
                    count = present.select(count, fx.Int32(0))
                task_slot = (part < count).select(first + part, fx.Int32(0))
                # Tiles of an int32 length, times at most 16 parts, fit Int32.
                length = fx.max(task_context, fx.Int32(0))
                first_tile = fx.Int32(0)
                if const_expr(t.sliding_window > 0):
                    window = min(t.query_length - 1 + t.sliding_window, 2**31 - 1)
                    first_tile = (
                        fx.max(length - window, fx.Int32(0)) // KV_COMPUTE_BLOCK
                    )
                tiles = (
                    length // KV_COMPUTE_BLOCK
                    + fx.Int32(length % KV_COMPUTE_BLOCK != 0)
                    - first_tile
                )
                parts = fx.max(count, fx.Int32(1))
                task_seq = seq
                task_begin = first_tile + part * tiles // parts
                task_end = first_tile + (part + 1) * tiles // parts
                active = part < count
        else:
            if const_expr(t.direct_output):
                # One task per nonempty sequence; reduce_info supplies its slot.
                # Later sequences start first: sparse prefill rows grow with
                # their position, so the longest tasks no longer form a tail.
                task_slot = sequences - 1 - task_slot
                if const_expr(t.quant_output):
                    # CTAs past the sequences write padded FP8 rows (below);
                    # they read sequence 0's entry and run no task.
                    task_slot = fx.max(task_slot, fx.Int32(0))
                direct_seq = task_slot
                info = fx.recast_iter(fx.Int32, pmax)
                count = fx.Int32(info[task_slot * 2 + 1])
                if const_expr(t.quant_output):
                    count = (fx.Int32(gpu.block_id("x")) < sequences).select(
                        count, fx.Int32(0)
                    )
                # A trailing empty sequence's start can equal capacity. Its dummy
                # task read must stay in bounds even though no compute consumes it.
                task_slot = (count > 0).select(
                    fx.Int32(info[task_slot * 2]), fx.Int32(0)
                )
            if const_expr(t.plan_free):
                # CTA s runs sequence s: the tiles its query positions can see.
                task_context = fx.Int32(fx.recast_iter(fx.Int32, work)[task_slot])
                task_seq, task_begin = task_slot, fx.Int32(0)
                if const_expr(t.sliding_window > 0):
                    first_visible = task_context - t.query_length + 1 - t.sliding_window
                    task_begin = (first_visible > 0).select(
                        first_visible // KV_COMPUTE_BLOCK, fx.Int32(0)
                    )
                task_end = (task_context + (KV_COMPUTE_BLOCK - 1)) // KV_COMPUTE_BLOCK
            else:
                task = fx.ptr_load(
                    fx.add_offset(fx.recast_iter(fx.Int32, work), task_slot * 4),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
                task_seq, task_begin = fx.Int32(task[0]), fx.Int32(task[1])
                task_end, task_context = fx.Int32(task[2]), fx.Int32(task[3])
            active = task_begin < task_end
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
                task_seq,
                task_begin,
                task_end,
                task_context,
            )
            # Packed partials are [KV heads, capacity, rows] by task slot.
            ctx.task_slot, ctx.capacity = task_slot, capacity
            ctx.n_kv = kv_heads
            ctx.quant = quant
            if const_expr(t.merge_partitions):
                # Issued early: only the epilogue reads the sequence's tasks.
                tasks = fx.ptr_load(
                    fx.add_offset(fx.recast_iter(fx.Int32, reduce_info), task_seq * 2),
                    result_type=fx.Vector.make_type(2, fx.Int32),
                )
                ctx.first_task = fx.Int32(tasks[0])
                ctx.task_count = fx.Int32(tasks[1])
                ctx.output_ptr = output
                ctx.counters_ptr = counters
            if const_expr(t.cluster_merge):
                ctx.task_count = count
                ctx.output_ptr = output
                ctx.step_bounds = step_bounds
                if const_expr(t.cluster_steps or t.cluster_runs):
                    ctx.stage_pages = stage_pages
            if const_expr(t.plan_free):
                ctx.output_ptr = output
            if const_expr(t.mxfp4):
                if const_expr(t.wave_pairs):
                    Gfx1250Mxfp4Pipeline(ctx).run_pairs()
                elif const_expr(t.wave_dims):
                    Gfx1250Mxfp4Pipeline(ctx).run_dims()
                else:
                    Gfx1250Mxfp4Pipeline(ctx).run()
            elif const_expr(t.nvfp4_wave):
                Gfx1250Nvfp4WavePipeline(ctx).run()
            elif const_expr(t.bf16_wave_query_tiles):
                Gfx1250Bf16WavePipeline(ctx).run()
            elif const_expr(t.bf16_page128_packed):
                Gfx1250Bf16Pipeline(ctx).run()
            elif const_expr(t.wave_query_tiles):
                Gfx1250WavePipeline(ctx).run()
            else:
                Gfx1250Pipeline(ctx).run()
        elif const_expr(t.direct_output):
            if const_expr(t.quant_output):
                block = fx.Int32(gpu.block_id("x"))
                if block < sequences:
                    zero_output(direct_seq, pout)
                else:
                    pad_output(block - sequences)
            else:
                zero_output(direct_seq, pout)
        elif const_expr(t.plan_free):
            zero_output(task_slot, output)
        if const_expr(t.merge_partitions):
            # Empty sequences own no task. Capacity >= batch gives each one the
            # CTA at its index, which clears its rows.
            info = fx.recast_iter(fx.Int32, reduce_info)
            if task_slot < sequences:  # noqa: SIM102
                if fx.Int32(info[task_slot * 2 + 1]) == 0:
                    zero_output(task_slot, output)
        if const_expr(t.cluster_merge and t.cluster_seqs > 1):
            # Sequences sharing a cluster need not agree on their task counts:
            # every CTA reaches the barrier, and only split sequences merge.
            Gfx1250ClusterMerge(t, pmax, psum, pout, output, sq_row, sq_head).sync()
            if count > 1:
                Gfx1250ClusterMerge(t, pmax, psum, pout, output, sq_row, sq_head).merge(
                    seq, part, first, count, capacity
                )
            elif (count == 0) & (part == 0) & present:
                zero_output(seq, output)
        elif const_expr(t.cluster_merge):
            # Every CTA of a cluster reaches its barrier exactly once. With
            # wave partials even a one-task sequence merges its waves here.
            waves = t.NWARP if t.wave_partials else 1
            if count > (0 if t.wave_partials else 1):
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
                merge.run(seq, part, first * waves, count * waves, capacity)
            elif const_expr(t.quant_output):
                pad_row = fx.Int32(gpu.block_id("x")) - sequences * t.cluster_merge
                if pad_row >= 0:
                    if pad_row < quant_rows - sequences * t.query_length:
                        pad_output(pad_row)
                elif (count == 0) & (part == 0):
                    zero_output(seq, output)
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
        run_tiles: fx.Int32,
        quantized_output: fx.Pointer,
        output_scale: fx.Pointer,
        quant_rows: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = capacity
        if const_expr(t.fuse_tasks > 1):
            grid_x = (capacity + 1) // 2
        if const_expr(t.direct_output or t.plan_free):
            grid_x = sequences
            if const_expr(t.quant_output):
                # One CTA per padded FP8 row past the sequences' tokens.
                grid_x = quant_rows - sequences * (t.query_length - 1)
        if const_expr(t.cluster_merge):
            # Whole clusters of cluster_seqs sequences.
            grid_x = ((sequences + t.cluster_seqs - 1) // t.cluster_seqs) * (
                t.cluster_seqs * t.cluster_merge
            )
            if const_expr(t.quant_output):
                # One cluster per t.cluster_merge padded FP8 rows past the
                # sequences' tokens.
                pads = quant_rows - sequences * t.query_length
                grid_x = (
                    sequences + (pads + t.cluster_merge - 1) // t.cluster_merge
                ) * t.cluster_merge
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
            run_tiles,
            quantized_output,
            output_scale,
            quant_rows,
            value_attrs=(
                {"rocdl.cluster_dims": f"{t.cluster_merge * t.cluster_seqs},1,1"}
                if t.cluster_merge
                else None
            ),
        ).launch(
            grid=(grid_x, kv_heads * t.query_splits, 1),
            block=(t.BLOCK_THREADS, 1, 1),
            stream=stream,
            cluster=(
                (t.cluster_merge * t.cluster_seqs, 1, 1) if t.cluster_merge else None
            ),
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

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""FP8, BF16 or MXFP4 paged-attention decode on gfx942/gfx950/gfx1250.

Cache layouts are logical, not preshuffled. K/V use E4M3 FNUZ on gfx942 and
OCP on gfx950/gfx1250. See ``kernels.pa_decode_kernel`` for Q/P quantization and
MFMA/WMMA specialization details. Native BF16 uses unscaled, 8-element packed
K/V caches. gfx1250 also accepts FP16 queries and plain BF16 V caches. MXFP4 K/V
(e2m1 data, e8m0 scale per 32 elements) run on gfx950 and gfx1250, see ``pa_decode``.
"""

import math
from functools import partial

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.utility import dtypes

from .kernels.pa_decode.traits import (
    d256_page64_pipe,
    nhd_max_query_length,
    require_nhd_pipe,
)
from .kernels.pa_decode_kernel import (
    CLUSTER_PARTITIONS,
    CLUSTER_STEP_TOKENS,
    KV_COMPUTE_BLOCK,
    MERGE_HEAD_DIM,
    MERGE_PARTITIONS,
    MXFP4_GFX1250_MAX_ROWS,
    MXFP4_GFX1250_STRICT_REFUSAL,
    compile_pa_decode_tile,
    gfx1250_tile_traits,
    mxfp4_gfx1250_query_splits,
    mxfp4_gfx1250_resident_ctas,
)
from .kernels.pa_decode_plan import PADecodePlan

# Plans with at most this many partitions per sequence merge within a cluster
# of at most CLUSTER_PARTITIONS CTAs. Above it, sequences would give each CTA
# more tiles than the reducer costs.
CLUSTER_GROUPED_PARTITIONS = 2 * CLUSTER_PARTITIONS
from .kernels.pa_decode_plan import plan_pa_decode as plan_pa_decode  # noqa: PLC0414
from .kernels.pa_decode_reduce import (
    DUAL_REGISTER_PARTITIONS,
    REGISTER_PARTITIONS,
    SPLIT_GROUPS,
    SPLIT_PARTITIONS,
    compile_pa_decode_ps_reduce,
)
from .kernels.tensor_shim import (
    _run_compiled,
    get_dtype_str,
    ptr_arg,
    torch_to_fx_dtype,
)

_CONSTANT_SCALES: dict[tuple[torch.device, float], torch.Tensor] = {}


def _constant_scale(device: torch.device, value: float) -> torch.Tensor:
    """One-element float32 scale, kept per (device, value) once built eagerly.

    Building it per call adds a fill or host copy to every call and to every
    captured CUDA graph: two extra dispatches per scalar-scale FP8 call. A
    scale first built during graph capture is not cached: its fill has not run
    yet, and its memory belongs to that graph's pool.
    """
    key = (device, value)
    scale = _CONSTANT_SCALES.get(key)
    if scale is None:
        scale = torch.full((1,), value, dtype=torch.float32, device=device)
        if not torch.cuda.is_current_stream_capturing() and len(_CONSTANT_SCALES) < 64:
            torch.cuda.current_stream(device).synchronize()  # readable from any stream
            _CONSTANT_SCALES[key] = scale
    return scale


def _kv_fx_dtype(dtype: torch.dtype):
    """FlyDSL type of BF16 or ``dtypes.fp8`` KV (as ``pa_decode`` requires), or None."""
    if dtype not in (torch.bfloat16, dtypes.fp8):
        return None
    return torch_to_fx_dtype(dtype)


def _default_workgroup_budget(
    arch: str,
    num_compute_units: int,
    kv_dtype: torch.dtype,
    head_dim: int,
    block_size: int,
    query_length: int,
    query_group_size: int,
    sliding_window: int,
) -> int:
    kv_fx = _kv_fx_dtype(kv_dtype)
    one_cta_per_cu = (
        kv_fx is not None
        and sliding_window <= 0
        and d256_page64_pipe(
            arch, kv_fx, head_dim, block_size, query_length, query_group_size
        )
    )
    return num_compute_units if one_cta_per_cu else 2 * num_compute_units


def pa_decode_nhd_max_query_length(
    device: torch.device,
    *,
    kv_dtype: torch.dtype,
    head_dim: int,
    block_size: int,
    query_group_size: int,
) -> int:
    """Longest uniform query length ``pa_decode`` takes with NHD K/V caches.

    NHD caches are 4D, ``[num_blocks, block_size, num_kv_heads, head_dim]``,
    and ``pa_decode`` selects them by ``key_cache.dim() == 4``. Returns 0 when
    NHD cannot serve the device and configuration at all; ``pa_decode`` raises
    ``NotImplementedError`` for any longer query, including MTP verify steps.
    Sliding windows do not change the limit. Reads device properties only, so
    callers can decide per layer before allocating caches or capturing graphs.
    Per-call strides, alignment and scales are validated by ``pa_decode``.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return 0
    arch = get_gfx_runtime()
    kv_fx = _kv_fx_dtype(kv_dtype)
    if kv_fx is None:
        return 0
    return nhd_max_query_length(arch, kv_fx, head_dim, block_size, query_group_size)


def pa_decode_default_workgroup_budget(
    device: torch.device,
    *,
    kv_dtype: torch.dtype,
    head_dim: int,
    block_size: int,
    query_length: int,
    query_group_size: int,
    sliding_window: int = 0,
) -> int:
    """Untuned ``plan_pa_decode`` work budget for this device and shape.

    The budget ``prepare_pa_decode_plan`` uses when no tuned result is cached:
    the CU count for dense page-64 shapes of the gfx950 FP8 head-256 pipe,
    packed or NHD, and twice the CU count otherwise. ``plan_pa_decode``
    defaults to twice the CU count because it does not see the cache shape.
    """
    device = torch.device(device)
    return _default_workgroup_budget(
        get_gfx_runtime(),
        torch.cuda.get_device_properties(device).multi_processor_count,
        kv_dtype,
        head_dim,
        block_size,
        query_length,
        query_group_size,
        sliding_window,
    )


def _nhd_strides(view: torch.Tensor) -> tuple[int, int, int]:
    """Block, token and head strides; size-1 axes are never stepped, so 0."""
    shape, stride = view.shape, view.stride()
    return tuple(0 if shape[d] == 1 else stride[d] for d in range(3))


def _view_span(view: torch.Tensor) -> int:
    """Elements from the view's first element through its last, inclusive."""
    return 1 + sum(
        (size - 1) * stride
        for size, stride in zip(view.shape, view.stride())
        if size > 1
    )


def _fuses_quant_output(
    tile_options: dict, *, cluster_merge: int, cluster_seqs: int
) -> bool:
    """Whether the gfx1250 tile quantizes the FP8 output in its epilogue.

    ``tile_options`` are the ``gfx1250_tile_traits`` arguments ``pa_decode``
    selects. A token's descale needs the max over all of its values. With one
    KV head a CTA, or the cluster merging its tasks, holds every row of its
    tokens, so the one-partition direct and the cluster-merge outputs quantize
    in their epilogue. More KV heads spread a token over separate CTAs (grid
    y); the last-arriver merge, clusters shared by several sequences
    (``cluster_seqs > 1``) and query-split tiles are not fused.
    """
    return bool(
        tile_options["num_kv_heads"] == 1
        and (
            (tile_options["direct_output"] and tile_options["num_partitions"] == 1)
            or (cluster_merge and cluster_seqs == 1)
        )
        and gfx1250_tile_traits(**tile_options).query_splits == 1
    )


def launch_pa_decode_ps_reduce(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    logits: torch.Tensor,
    sink_token: torch.Tensor | None,
    stride_output_bs: int,
    stride_output_len: int,
    stride_output_kv_head: int,
    stride_output_group_size: int,
    stride_exp_sums_head: int,
    stride_exp_sums_part: int,
    stride_logits_head: int,
    stride_logits_part: int,
    stride_logits_group: int,
    *,
    query_seq_len: int,
    query_group_size: int,
    head_size: int,
    max_partitions: int,
    stream: torch.cuda.Stream,
    reduce_info: torch.Tensor,
) -> None:
    """Reduce packed partials; every sequence has at most ``max_partitions``."""
    props = torch.cuda.get_device_properties(output.device)
    if max_partitions > props.multi_processor_count:
        raise ImportError(
            "FlyDSL pa_decode reduce supports at most "
            f"{props.multi_processor_count} partitions"
        )
    arch = get_gfx_runtime()
    wave = props.warp_size
    register_partitions = min(REGISTER_PARTITIONS, wave)
    max_cta_waves = 1024 // wave
    batch, kv_heads = output.shape[0], output.shape[2]
    rows = query_seq_len * query_group_size
    # Exact bounds keep the register ladder short; above a wave a power of two
    # bounds the specialization count. Per-sequence buffer bounds zero-fill
    # partitions beyond the live count, so rounding past the bound is safe.
    partitions = max_partitions
    if partitions > register_partitions:
        partitions = min(
            props.multi_processor_count, 1 << (partitions - 1).bit_length()
        )
    # A per-sequence span <= INT32_MAX keeps every attempted byte offset,
    # including inactive partitions, below 2**32, so out-of-bounds reads
    # cannot wrap back into valid data.
    if not (
        stride_logits_group == head_size
        and stride_logits_part == rows * head_size
        and 0 <= stride_logits_head <= 2**31 - 1
        and partitions * stride_logits_part * logits.element_size() <= 2**31 - 1
    ):
        raise NotImplementedError(
            "FlyDSL pa_decode reduce requires packed partial rows spanning at "
            "most 2**31 - 1 bytes per sequence"
        )
    # Paired loads need dword alignment; scalar output stores do not. The
    # register-only kernel pairs only D128/D256: at larger heads, halving its
    # waves costs more latency hiding than the wider loads save.
    dual = partitions > register_partitions
    pairs = (
        (head_size % 128 == 0 if dual else head_size in (128, 256))
        and logits.dtype in (torch.bfloat16, torch.float16)
        and logits.data_ptr() % 4 == 0
        and stride_logits_head % 2 == 0
    )
    ctas = batch * kv_heads * rows
    groups, num_xcds = 1, 0
    if (
        not dual
        and arch in ("gfx1250", "gfx1250-strict")
        and partitions > SPLIT_PARTITIONS
    ):
        # Register rungs above 16 partitions split their loads over groups,
        # within the same wave budget as the dual-mode CTA.
        group_waves = head_size // wave // (2 if pairs else 1)
        budget = max_cta_waves * props.multi_processor_count
        while (
            2 * groups <= SPLIT_GROUPS
            and 2 * groups * group_waves <= max_cta_waves
            and 2 * groups * group_waves * ctas <= budget
        ):
            groups *= 2
    if dual:
        # Partition groups fill the CTA while the grid stays within 16 waves
        # per CU. D128 sequences averaging at most 32 partitions keep one, or
        # on gfx1250 the four that split their 32-partition rung.
        group_waves = head_size // wave // (2 if pairs else 1)
        budget = max_cta_waves * props.multi_processor_count
        max_groups = max_cta_waves // group_waves
        if head_size == 128 and exp_sums.shape[1] <= DUAL_REGISTER_PARTITIONS * batch:
            max_groups = SPLIT_GROUPS if arch in ("gfx1250", "gfx1250-strict") else 1
        while 2 * groups <= max_groups and 2 * groups * group_waves * ctas <= budget:
            groups *= 2
        per_seq = rows * kv_heads
        if (
            arch not in ("gfx1250", "gfx1250-strict")
            and ctas * group_waves >= budget
            and batch * per_seq * per_seq <= 2**30
        ):
            num_xcds = get_num_xcds()
    num_xcds = num_xcds if num_xcds > 1 else 0
    compiled = compile_pa_decode_ps_reduce(
        architecture=arch,
        max_context_partition_num=partitions,
        head_size=head_size,
        output_dtype_str=get_dtype_str(output.dtype),
        logits_dtype_str=get_dtype_str(logits.dtype),
        sink_dtype_str=get_dtype_str(
            output.dtype if sink_token is None else sink_token.dtype
        ),
        use_sinks=sink_token is not None,
        query_group_size=query_group_size,
        pairs=pairs,
        parallel_groups=groups,
        num_xcds=num_xcds,
    )
    sink_ptr = (
        flyc.from_c_void_p(torch_to_fx_dtype(output.dtype), 0)
        if sink_token is None
        else ptr_arg(sink_token, torch_to_fx_dtype(sink_token.dtype))
    )
    _run_compiled(
        compiled["launch"],
        ptr_arg(output, torch_to_fx_dtype(output.dtype)),
        ptr_arg(exp_sums, fx.Float32),
        ptr_arg(max_logits, fx.Float32),
        ptr_arg(logits, torch_to_fx_dtype(logits.dtype)),
        sink_ptr,
        stride_output_bs,
        stride_output_len,
        stride_output_kv_head,
        stride_output_group_size,
        stride_exp_sums_head,
        stride_exp_sums_part,
        stride_logits_head,
        stride_logits_part,
        stride_logits_group,
        query_seq_len,
        batch,
        kv_heads,
        ptr_arg(reduce_info, fx.Int32),
        -(-(1 << 30) // rows),
        -(-(1 << 30) // (rows * kv_heads)),
        stream,
    )


def _pa_decode_mxfp4(
    output,
    query,
    k_data,
    v_data,
    k_scale,
    v_scale,
    context_lengths,
    block_tables,
    softmax_scale,
    exp_sums,
    max_logits,
    temporary_output,
    work_plan,
    max_context_length,
    query_length=1,
    sliding_window=0,
):
    """MXFP4 K/V on gfx950 and gfx1250: ``kv_dtype=dtypes.fp4x2`` of the tile kernel.

    gfx1250 also takes MTP query lengths and sliding windows; gfx950 callers
    refuse both before this point.
    """
    arch = get_gfx_runtime()
    if arch == "gfx1250-strict":
        raise NotImplementedError(MXFP4_GFX1250_STRICT_REFUSAL)
    gfx1250 = arch == "gfx1250"
    if arch != "gfx950" and not gfx1250:
        raise NotImplementedError("MXFP4 pa_decode supports gfx950 and gfx1250")
    # fp4x2 data and fp8_e8m0 scale views are the same bytes as uint8 ones.
    k_data, v_data, k_scale, v_scale = (
        (
            t.view(torch.uint8)
            if isinstance(t, torch.Tensor)
            and t.dtype in (dtypes.fp4x2, dtypes.fp8_e8m0)
            else t
        )
        for t in (k_data, v_data, k_scale, v_scale)
    )
    dev = query.device
    num_seqs = context_lengths.shape[0]
    _, num_q_heads, head_dim = query.shape
    num_kv_heads = k_data.shape[1]
    group = num_q_heads // num_kv_heads
    if head_dim != 128 or num_q_heads % num_kv_heads or group > 16:
        raise NotImplementedError(
            "MXFP4 pa_decode needs head_dim 128 and 1..16 query heads per KV head, "
            f"got query {tuple(query.shape)} over {num_kv_heads} KV heads"
        )
    if query.dtype not in (torch.float8_e4m3fn, torch.bfloat16):
        raise TypeError(
            f"MXFP4 pa_decode needs an e4m3 or bf16 query, got {query.dtype}"
        )
    if output.dtype != torch.bfloat16 or output.shape != query.shape:
        raise TypeError("MXFP4 pa_decode needs a bf16 output shaped like the query")
    if output.device != dev:
        raise ValueError(f"output must be on the query device ({dev})")
    if any(t.stride(2) != 1 for t in (query, output)) or any(
        x * t.element_size() % 16 for t in (query, output) for x in t.stride()[:2]
    ):
        raise ValueError(
            "MXFP4 pa_decode needs 16 B query and output row/head strides and a "
            "contiguous head_dim axis"
        )
    if gfx1250 and (query.data_ptr() % 16 or output.data_ptr() % 16):
        # gfx1250 loads query rows and stores output rows in 16 B units.
        raise ValueError("MXFP4 pa_decode needs 16 B aligned query and output")
    # The kernel adds row and head offsets in i32 (query bytes, output elements);
    # keep the end of the last head_dim chunk within 2**31 - 1 bytes for both.
    for name, t in (("query", query), ("output", output)):
        if (
            (t.shape[0] - 1) * t.stride(0) + (t.shape[1] - 1) * t.stride(1) + t.shape[2]
        ) * t.element_size() > 2**31 - 1:
            raise ValueError(
                f"MXFP4 pa_decode {name} must span at most 2**31 - 1 bytes"
            )
    # K and V are data views [pages, kv_heads, 128, 64] and e8m0 scale views
    # [pages, kv_heads, 128, 4] of one page; the four share page and head strides.
    page_head_strides = k_data.stride()[:2]
    if any(x % 16 or x >= 2**31 for x in page_head_strides):
        raise ValueError(
            "MXFP4 page and head strides must be multiples of 16 and fit i32"
        )
    for name, t, tail in (
        ("key_cache", k_data, 64),
        ("value_cache", v_data, 64),
        ("key_scale", k_scale, 4),
        ("value_scale", v_scale, 4),
    ):
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                f"{name} must be a uint8 torch.Tensor, got {type(t).__name__}"
            )
        if (
            t.dtype != torch.uint8
            or t.device != dev
            or t.shape[0] != k_data.shape[0]
            or tuple(t.shape[1:]) != (num_kv_heads, 128, tail)
            or t.stride() != (*page_head_strides, tail, 1)
        ):
            raise ValueError(
                f"{name} must be a uint8 [{k_data.shape[0]} pages, {num_kv_heads}, "
                f"128, {tail}] view (or {'fp4x2' if tail == 64 else 'fp8_e8m0'}) "
                f"on {dev} with the key cache's page count, page and head strides, "
                f"got {t.dtype} {tuple(t.shape)} {t.stride()}"
            )
        # mxfp4_init loads data as 16 B and scale as 4 B aligned.
        align = 16 if tail == 64 else 4
        if t.data_ptr() % align:
            raise ValueError(f"{name} base pointer must be {align} B aligned")
    if block_tables.dtype != torch.int32 or context_lengths.dtype != torch.int32:
        raise TypeError("block_tables and context_lengths must be int32")
    if query.shape[0] != num_seqs * query_length or block_tables.shape[0] != num_seqs:
        raise ValueError(
            f"query.shape[0] ({query.shape[0]}) must match context_lengths.shape[0] "
            f"* query_length ({num_seqs} * {query_length}) and "
            f"block_tables.shape[0] ({block_tables.shape[0]}) must match "
            f"{num_seqs}"
        )
    for name, t in (
        ("block_tables", block_tables),
        ("context_lengths", context_lengths),
    ):
        if t.device != dev or not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the query device ({dev})")
    if num_seqs == 0:
        return
    max_blocks_per_seq = block_tables.shape[1]
    schedule_context_length = max_blocks_per_seq * 128
    if max_context_length is not None:
        if not 0 <= max_context_length <= schedule_context_length:
            raise ValueError("max_context_length must be within block-table capacity")
        schedule_context_length = -(-max_context_length // 128) * 128
    same_strides = query.stride()[:2] == output.stride()[:2]
    if work_plan is None and not same_strides:
        raise ValueError("without a work_plan the output needs the query's strides")
    if work_plan is not None:
        work_plan.validate(num_seqs, num_kv_heads, dev)
        if work_plan.sliding_window != sliding_window:
            raise ValueError("sliding_window must match work_plan.sliding_window")
        if sliding_window > 0 and work_plan.query_length != query_length:
            raise ValueError("query_length must match work_plan.query_length")
    # One task per sequence needs neither the plan nor the reducer.
    direct = work_plan is None or (work_plan.max_partitions == 1 and same_strides)
    capacity = num_seqs if direct else work_plan.capacity
    num_compute_units = torch.cuda.get_device_properties(dev).multi_processor_count
    # Tiles a sequence can own: the length bound's, within the MTP window as on
    # the FP8/BF16 path.
    seq_tiles = max(1, -(-schedule_context_length // KV_COMPUTE_BLOCK))
    if sliding_window > 0:
        seq_tiles = min(
            seq_tiles,
            (sliding_window + query_length - 2 + KV_COMPUTE_BLOCK - 1)
            // KV_COMPUTE_BLOCK
            + 1,
        )
    # As for FP8/BF16 on gfx1250: a sequence's tasks, at most
    # CLUSTER_GROUPED_PARTITIONS of them, merge within a cluster of at most
    # CLUSTER_PARTITIONS CTAs instead of the reducer, while the clusters stay
    # resident (15 per 16 CUs). Larger bounds split their tiles over the
    # cluster. Every query split is a cluster of its own, so both bounds count
    # the splits (G16 x MTP4 at B32 x 8k would launch 1024 cluster CTAs, two
    # rounds at two a CU: 16.4 us against 14.4 us with the reducer), and MTP
    # clusters of 3 and 4 M-tile CTAs, one a CU, stay within the CUs.
    #
    # Single-token decode merges on chip with one tile a cluster CTA, or with
    # clusters over a quarter of the CUs. Below that the plan's one-tile
    # tasks and the reducer win (G16 B1 x 8k: 5.5 us against 5.8 us for 16
    # two-tile CTAs; B4 x 8k: 6.0 against 5.8 us; 8 KV heads B1 x 8k: 6.7
    # against 5.9 us). MTP tasks cost more, so they merge on chip whenever
    # the clusters fit (G16 x MTP4 at B3 x 8k: 7.4 against 7.1 us).
    splits = mxfp4_gfx1250_query_splits(query_length, group) if gfx1250 else 1
    rows = query_length * group // splits
    resident_ctas = mxfp4_gfx1250_resident_ctas(rows, num_compute_units)
    partitions = 1 if direct else min(work_plan.max_partitions, capacity, seq_tiles)
    # Split 41-64-row MTP (G16 x MTP4) runs whole, one CTA of 3-4 M-tiles a
    # CU dequantizing V once rather than once a split, when its clusters fit
    # one round with at least 8 tiles a CTA (B64 x 8k: 17.5 -> 15.9 us;
    # B128 x 8k, 2-CTA clusters instead of the reducer: 29.9 -> 25.9 us; 2 KV
    # heads at B32 x 8k: 17.7 -> 16.5 us). With 4 tiles a CTA the splits'
    # shorter tiles win (B32 x 8k: 11.5 against 11.7 us).
    if (
        gfx1250
        and splits > 1
        and query_length > 1
        and 40 < query_length * group <= MXFP4_GFX1250_MAX_ROWS
    ):
        whole_ctas = -(-num_compute_units // num_kv_heads)
        whole_size = min(partitions, CLUSTER_PARTITIONS)
        while whole_size > 2 and num_seqs * whole_size > whole_ctas:
            whole_size //= 2
        if (
            same_strides
            and 2 <= partitions <= CLUSTER_GROUPED_PARTITIONS
            and num_seqs * whole_size <= whole_ctas
            and num_seqs * num_kv_heads <= num_compute_units * 15 // 16
            and seq_tiles >= 8 * whole_size
        ):
            splits, rows = 1, query_length * group
            resident_ctas = mxfp4_gfx1250_resident_ctas(rows, num_compute_units)
    cluster_size = min(partitions, CLUSTER_PARTITIONS)
    cluster_ctas = max(capacity, -(-2 * num_compute_units // num_kv_heads))
    if rows > 32:
        cluster_ctas = -(-resident_ctas // num_kv_heads)
    if query_length > 1:
        # MTP clusters shrink until their CTAs fit one round: each CTA runs
        # more tiles, but the call merges on chip without the reducer (G16 x
        # MTP4 at B32 x 8k: 14.2 -> 11.7 us with 8-CTA clusters).
        while cluster_size > 2 and num_seqs * cluster_size * splits > cluster_ctas:
            cluster_size //= 2
    # One-M-tile CTAs whose clusters run in one round split their waves by
    # dims (``run_dims``) and merge on chip faster than the reducer (G16 B2 x
    # 8k: 5.3 -> 5.1 us), so single-token decode then takes the cluster too.
    # Long runs a CTA lose the merge's gain to the per-tile exchange (B1 x
    # 128k, an 8-CTA cluster of a max_partitions=8 plan: 48.6 against 64.3
    # us), so dims take clusters of at most 4 tiles a CTA.
    dims_cluster = (
        rows <= 16
        and num_seqs * num_kv_heads * splits * cluster_size <= num_compute_units
        and seq_tiles <= 4 * cluster_size
    )
    cluster_merge = (
        cluster_size
        if gfx1250
        and same_strides
        and 2 <= partitions <= CLUSTER_GROUPED_PARTITIONS
        and (
            query_length > 1
            or partitions <= CLUSTER_PARTITIONS
            or 4 * num_seqs * num_kv_heads * cluster_size >= num_compute_units
            or dims_cluster
        )
        and num_seqs * cluster_size * splits <= cluster_ctas
        and num_seqs * num_kv_heads * splits <= num_compute_units * 15 // 16
        else 0
    )
    # A gfx1250 MTP CTA keeping several M-tiles fits two to a CU, or one with
    # 3 or 4 M-tiles. When the plan's tasks times the KV heads and query
    # splits exceed the resident CTAs, a CTA runs two plan slots, so a
    # sequence's adjacent tiles fuse into one longer task. The kernel decides
    # from the plan's live tasks; the tiles the bound and window allow only
    # decide whether it may.
    max_tasks = min(capacity, num_seqs * seq_tiles)
    fuse_tasks = (
        2
        if gfx1250
        and not direct
        and not cluster_merge
        and rows > 16
        and max_tasks * num_kv_heads * splits > resident_ctas
        else 1
    )
    # 41-64 query rows that would fuse instead run whole in one CTA, two a CU,
    # whose wave pairs split the M-tiles and the tokens of each 128-token step
    # and swap their dequantized V halves (``run_pairs``): half the V dequant
    # of two query splits (G16 x MTP4 at B3 x 32k: 10.1 -> 9.4 us, B3 x 225k:
    # 27.0 -> 23.3 us). Where clusters merge, or with 40 rows, more and shorter
    # CTAs win (B3 x 8k: 7.0 against 9.1 us; G8 x MTP5 at B8 x 32k: 11.4
    # fused against 12.3 us).
    wave_pairs = fuse_tasks == 2 and 40 < query_length * group <= MXFP4_GFX1250_MAX_ROWS
    if wave_pairs:
        fuse_tasks = 1
    # Clusters of at most 8 CTAs whose scratch holds a slot per wave merge
    # each wave's partial directly, skipping the CTA's LDS merge of its four
    # waves and its barrier (G16 B1 x 1k decode: 4.6 -> 3.8 us). From 8 CTAs,
    # wave dims are faster where they apply (B4 x 2k: 4.28 -> 4.17 us).
    wave_partials = bool(
        gfx1250
        and cluster_merge
        and cluster_merge * 4 <= 32
        and 4 * num_seqs * cluster_merge <= capacity
        and not (dims_cluster and cluster_merge >= 8)
    )
    # Wave dims: wave w dequantizes dim block w of the whole tile and runs
    # its PV against every wave's P, so no wave merges another's outputs
    # (G16 decode B1 x 4k: 4.8 -> 4.1 us, B8 x 8k: 5.8 -> 5.3 us). Their
    # 520-odd VGPRs fit one CTA a CU, so only calls that fit one round use
    # them (B32 x 8k: 7.7 against 10.5 us).
    wave_dims = bool(gfx1250 and dims_cluster and cluster_merge and not wave_partials)
    # As on the FP8/BF16 path: clusters given a length bound give CTA p tiles
    # [p * run, (p + 1) * run), runs splitting the bound evenly, so each CTA
    # loads its first page ids with the context length rather than after it.
    # Single-token decode takes runs with wave dims only (G16 B16 x 8k 6.2 ->
    # 6.3 us without them; B1 x 4k 4.12 -> 4.01 us with them); MTP4 is faster
    # with runs (B3 x 8k 7.4 -> 6.9 us).
    bound_tiles = -(-schedule_context_length // KV_COMPUTE_BLOCK)
    cluster_runs = bool(
        cluster_merge
        and (query_length > 1 or wave_dims)
        and max_context_length is not None
        and sliding_window == 0
        and num_seqs * cluster_merge <= capacity
        and bound_tiles <= 4 * cluster_merge
    )
    run_tiles = -(-bound_tiles // cluster_merge) if cluster_runs else 0
    with torch.cuda.device(dev):
        compiled = compile_pa_decode_tile(
            head_dim=head_dim,
            query_group_size=group,
            block_size=128,
            num_seqs=num_seqs,
            num_kv_heads=num_kv_heads,
            num_compute_units=num_compute_units,
            num_partitions=1 if direct else work_plan.max_partitions,
            softmax_scale=softmax_scale,
            query_dtype=query.dtype,
            kv_dtype=dtypes.fp4x2,
            # MXFP4 KV is per-token KV: each row carries its own e8m0 scales.
            per_token_kv=True,
            query_length=query_length,
            sliding_window=sliding_window,
            work_capacity=capacity,
            max_context_length=schedule_context_length,
            direct_output=direct,
            cluster_merge=cluster_merge,
            cluster_runs=cluster_runs,
            fuse_tasks=fuse_tasks,
            wave_partials=wave_partials,
            wave_pairs=wave_pairs,
            wave_dims=wave_dims,
            query_splits=splits if gfx1250 else None,
        )
    pmax = psum = pout = output
    if not direct:
        shape = (num_kv_heads, capacity, query_length * group)
        scratch = []
        for name, t, expected, dtype in (
            ("max_logits", max_logits, shape, torch.float32),
            ("exp_sums", exp_sums, shape, torch.float32),
            ("temporary_output", temporary_output, (*shape, head_dim), output.dtype),
        ):
            if t is None:
                t = torch.empty(expected, dtype=dtype, device=dev)
            if (
                tuple(t.shape) != expected
                or t.dtype != dtype
                or t.device != dev
                or not t.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous {dtype} {expected} on {dev}"
                )
            scratch.append(t)
        pmax, psum, pout = scratch
    with torch.cuda.device(dev):
        stream = torch.cuda.current_stream(dev)
        operands = (
            ptr_arg(pmax, fx.Float32),
            ptr_arg(psum, fx.Float32),
            ptr_arg(pout, torch_to_fx_dtype(pout.dtype)),
            ptr_arg(query, torch_to_fx_dtype(query.dtype)),
            ptr_arg(k_data, fx.Float4E2M1FN),
            ptr_arg(v_data, fx.Float4E2M1FN),
            ptr_arg(block_tables, fx.Int32),
            ptr_arg(k_scale, fx.Float8E8M0FNU),
            ptr_arg(v_scale, fx.Float8E8M0FNU),
        )
        # Plan-free and cluster grids derive their tiles from the context lengths.
        work_info = ptr_arg(
            context_lengths if direct or cluster_merge else work_plan.work_info,
            fx.Int32,
        )
        if gfx1250:
            _run_compiled(
                compiled["launch"],
                *operands,
                max_blocks_per_seq,
                num_seqs,
                num_kv_heads,
                *page_head_strides,
                query.stride(0),
                query.stride(1),
                work_info,
                capacity,
                ptr_arg(output, torch_to_fx_dtype(output.dtype)),
                (
                    flyc.from_c_void_p(fx.Int32, 0)
                    if direct
                    else ptr_arg(work_plan.reduce_info, fx.Int32)
                ),
                flyc.from_c_void_p(fx.Int32, 0),
                run_tiles,
                # No FP8 output (pa_decode refuses it for MXFP4 KV).
                flyc.from_c_void_p(fx.Float8E4M3FN, 0),
                flyc.from_c_void_p(fx.Float32, 0),
                num_seqs,
                stream,
            )
            if cluster_merge:
                return
        else:
            _run_compiled(
                compiled["launch"],
                *operands,
                max_blocks_per_seq,
                num_seqs,
                num_kv_heads,
                *page_head_strides,
                query.stride(0),
                query.stride(1),
                # NHD K/V strides of the gfx942/gfx950 launch (#5949): unused.
                *((0,) * 6),
                work_info,
                capacity,
                stream,
            )
        if direct:
            return
        output_5d = output.reshape(
            num_seqs, query_length, num_kv_heads, group, head_dim
        )
        launch_pa_decode_ps_reduce(
            output_5d,
            psum,
            pmax,
            pout,
            None,
            *output_5d.stride()[:4],
            pmax.stride(0),
            pmax.stride(1),
            pout.stride(0),
            pout.stride(1),
            pout.stride(2),
            query_seq_len=query_length,
            query_group_size=group,
            head_size=head_dim,
            # Not bounded by max_context_length: the plan splits the real length,
            # so a context longer than the bound still has all its partials.
            max_partitions=min(work_plan.max_partitions, capacity),
            stream=stream,
            reduce_info=work_plan.reduce_info,
        )


# Launches of the gfx1250 tile kernel that writes the final output (direct or
# cluster merge, no reducer, no fallback quant), as a checked call set them up.
# `[_launch_options]` is a token for options and a plan that recorded a launch
# and `[token, _launch_operands]` is the launch: a later call with both keys
# launches on its own addresses, skipping the checks and the dispatch, and a
# call whose options and plan recorded none builds no second key. The keys
# hold what the checks and the dispatch read of a call: each scalar option and
# plan field with its type, the shape, strides, dtype and device of each
# operand and plan tensor, the partials' address modulo 8 and the current
# device. Not keyed: the architecture, which a process has one of, the other
# address bits and what the tensors hold. None until a checked call could
# record a launch: no call builds a key before that, so none does on the other
# architectures. Bounded: a full table is cleared.
_VALIDATED_LAUNCHES: dict | None = None
_VALIDATED_LAUNCHES_MAX = 1024


def _tensor_signature(t):
    if t is None:
        return None
    if type(t) is not torch.Tensor:
        raise TypeError
    return (t.shape, t.stride(), t.dtype, t.get_device())


def _launch_options(work_plan, *scalars):
    """First key of a call: its scalar options and its plan; no operand's
    signature goes into it. TypeError for a plan that is not a PADecodePlan
    of plain tensors."""
    if type(work_plan) is not PADecodePlan:
        raise TypeError
    # The plan's fields, not its identity: a per-forward plan of the same shape
    # (ATOM's sparse prefill builds one per forward) reuses the launch, which
    # reads the plan's tensors and capacity at call time.
    scalars += (
        work_plan.num_kv_heads,
        work_plan.max_partitions,
        work_plan.sliding_window,
        work_plan.query_length,
        work_plan.max_batch_size,
    )
    return (
        scalars,
        # The checks tell True from 1 and 0.0 from 0, which compare equal.
        tuple(map(type, scalars)),
        # Its shape is the capacity, which with the operand shapes picks the
        # cluster mode (cluster_seqs, cluster_steps, cluster_runs).
        _tensor_signature(work_plan.work_info),
        _tensor_signature(work_plan.reduce_info),
        _tensor_signature(work_plan.arrival_counters),
    )


def _launch_operands(operands):
    """Second key of a call: its operands, of which the partials, the output
    and the query come first. None for operands that are never recorded."""
    try:
        partials, output, query = map(_tensor_signature, operands[:3])
        if output[1] != query[1]:
            return None  # Other strides: the reducer writes the output.
        return (
            partials,
            output,
            query,
            *map(_tensor_signature, operands[3:]),
            # The last-arriver merge requires 8-byte aligned partials.
            operands[0].data_ptr() % 8,
            torch.cuda.current_device(),
        )
    except (TypeError, RuntimeError):  # Not plain tensors with storage.
        return None


def _record_launch(options, operands, launch):
    """Record the launch of a checked call for later calls with its keys."""
    global _VALIDATED_LAUNCHES
    if _VALIDATED_LAUNCHES is None:
        # This call built no key; calls build one from now on.
        _VALIDATED_LAUNCHES = {}
    elif options is not None and (key := _launch_operands(operands)) is not None:
        # Single dict operations only: calls that record at once may lose an
        # entry to each other's clear, and nothing else.
        if len(_VALIDATED_LAUNCHES) >= _VALIDATED_LAUNCHES_MAX:
            _VALIDATED_LAUNCHES.clear()
        token = _VALIDATED_LAUNCHES.setdefault(options, object())
        _VALIDATED_LAUNCHES[token, key] = launch


def pa_decode(
    output: torch.Tensor,  # [num_seqs * query_length, num_query_heads, head_size]
    query: torch.Tensor,  # [num_seqs * query_length, num_query_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x] or NHD [num_blocks, 64, num_kv_heads, 256]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, kv_block_size] or [num_blocks, num_kv_heads, kv_block_size // x, head_size, x] or NHD like key_cache
    context_lengths: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    softmax_scale: float,
    query_length: int,
    *,
    context_partition_size: int = 256,
    compute_type: torch.dtype | None = None,
    query_scale: torch.Tensor = None,
    key_scale: torch.Tensor = None,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale: torch.Tensor = None,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    exp_sums: torch.Tensor = None,  # [num_kv_heads, capacity, query_length * query_group_size]
    max_logits: torch.Tensor = None,  # [num_kv_heads, capacity, query_length * query_group_size]
    temporary_output: torch.Tensor = None,  # [num_kv_heads, capacity, query_length * query_group_size, head_size]
    quantized_output: torch.Tensor = None,  # optional FP8, output's shape + quantized_pad_rows rows
    output_scale: torch.Tensor = None,  # optional FP32 [num_seqs * query_length + quantized_pad_rows, 1]
    alibi_slopes: torch.Tensor = None,
    sinks: torch.Tensor = None,
    sliding_window: int = 0,
    work_plan: PADecodePlan | None = None,
    max_context_length: int | None = None,
    mask_unowned_kv: bool = True,
    store_output: bool = True,
    quantized_pad_rows: int = 0,
) -> None:
    """Decode FP8 or BF16 K/V using 256-token compute tiles.

    Supports page sizes 16/64/128. BF16 head dimensions are multiples of 64
    up to 1024; FP8 supports 64 and multiples of 128 up to 1024.
    K/V scales are [1] or [num_blocks, num_kv_heads, block_size, 1].

    The gfx950 FP8 head-256 page-16/64 pipe supports both packed V layouts,
    per-tensor or per-token scales, and sliding windows, with at most 32
    query rows (query_length times the GQA group size).

    A 4D ``key_cache`` selects the NHD layout: K and V are equally shaped
    [num_blocks, 64, num_kv_heads, 256] views with a contiguous head axis and
    non-negative other strides, e.g. both halves of one 512-byte token row. NHD
    requires gfx950, FP8 KV, head 256, page 64, and a uniform query_length
    whose rows (query_length times the GQA group size) fit two 16-row M-tiles,
    e.g. query_length 1-4 at GQA 8. Both scale modes and positive sliding
    windows are supported. Each view must be
    16-byte aligned with 16-byte multiple strides, its token rows must not
    overlap, and its strides and in-page span must fit int32; heads-outermost
    storage with several KV heads is thus limited to 2 GiB per head.
    ``pa_decode_nhd_max_query_length`` reports the longest query a
    configuration takes.

    ``compute_type=None`` infers compute from the cache dtype. BF16 K/V requires
    no K/V scales and uses x=8; FP8 uses x=16. gfx942/gfx950 require BF16 queries
    and vectorized 5D BF16 caches. gfx1250 also supports FP16 queries and plain
    BF16 V caches, with head dimensions 64 or multiples of 128 up to 1024.
    BF16 Q/P operands stay
    in BF16 and softmax statistics and MFMA accumulation stay in FP32.
    gfx942/gfx950 BF16 automatically splits query positions to keep each CTA
    within 64 KiB of LDS; a single query's GQA rows must fit that limit. gfx1250
    splits BF16 D1024/G16/MTP4 with transposed page128 caches and full attention
    into separate query CTAs. Q/K and V/output
    head dimensions must match.
    ALiBi and externally quantized queries are unsupported. ``softmax_scale``
    must be finite and positive, and so must FP8 ``key_scale``/``value_scale``
    values: host scalars are checked, device tensors are not read back.

    ``quantized_output`` and ``output_scale`` (gfx1250) add a per-token FP8
    output. They must be supplied together. It is OCP FP8 in the same logical
    layout as ``output`` with one contiguous FP32 descale per flattened query
    token, shape ``[num_seqs * query_length, 1]``, each with
    ``quantized_pad_rows`` extra rows (default 0); dequantization is
    ``quantized_output.float() * output_scale``. The tile kernel's epilogue
    quantizes the output when there is one KV head, the tile does not split
    its query rows over CTAs, and the plan writes either the one-partition
    direct output or a cluster merge with one sequence per cluster
    (``cluster_seqs == 1``). Every other plan writes the BF16/FP16 output and
    then runs the standalone per-token quant, with identical bytes: several KV
    heads, the last-arriver direct merge, clusters shared by several sequences
    (more than 15 sequence-KV-head pairs per 16 CUs), the reducer (sinks included) and the
    query-split BF16 D1024/G16/MTP4 page128 tile. Calls that omit both tensors
    retain the existing output and dispatch unchanged.
    With the quantized output, ``store_output=False`` skips the BF16/FP16
    store: ``output`` then only fixes the layout and its contents are left
    unspecified. ``quantized_pad_rows`` extra rows of ``quantized_output`` and
    ``output_scale`` past the query rows (a token bucket padded past the
    scheduled tokens) are written as FP8 zeros with a unit descale, so a
    consumer of the whole bucket reads finite, inert rows. The fused epilogue
    writes them in the PA launch; the fallback fills them with ``zero_()`` and
    ``fill_(1.0)`` after the standalone quant.

    MTP lengths include the query tokens and use dense causal masking.
    Independently selected sparse queries need separate table rows and
    query_length=1. Positive ``sliding_window`` includes each query's own
    position; nonpositive values disable it.

    ``work_plan`` is required and must be built with ``plan_pa_decode`` before
    calling decode. ``pa_decode_tuning.prepare_pa_decode_plan`` uses native
    FlyDSL autotune to select its budget before graph capture, and documents the
    default budget; ``tune=True`` forces a search during preparation.
    Decode and graph replay use the explicit plan without accessing autotune. Set its
    partition cap with ``plan_pa_decode(max_partitions=...)``. On gfx950 and
    gfx1250, when sinks are absent and query/output strides match, a cap of one
    writes output directly, without the reducer. On gfx1250, so do plans with one task slot per sequence (capacity equal to
    the batch, at most 8 partitions per sequence, head_dim <= 512): sequences
    that still own several tasks merge them in the tile kernel using the plan's
    arrival counters when partial scratch is eight-byte aligned. Otherwise,
    when P = min(partition bound, max_partitions)
    is in [2, 32], C = min(P, 16) CTAs per sequence fit the plan capacity or
    two per CU, and the sequences times KV heads stay within 15 per 16 CUs
    (head_dim <= 512), each sequence runs as one cluster of C workgroups that
    split its tiles and merge them on chip. Page16 D128 decode batches within
    one 16-CTA cluster per CU whose plan holds 16 slots per sequence (see
    ``plan_pa_decode``) instead split each sequence into 16 step-sized parts.
    Other plans use packed scratch and the separate reducer.

    ``sinks`` is a contiguous [num_query_heads] BF16/FP16/FP32 tensor on the
    query device: unscaled zero-value logits shared across batch/MTP positions.
    Each contributes once, independently of the window; -inf disables it and
    +inf suppresses the head's output.

    MXFP4 K/V (gfx950, gfx1250): ``key_cache``/``value_cache`` are data views
    [pages, kv_heads, 128, 64] (``dtypes.fp4x2`` or uint8) and
    ``key_scale``/``value_scale`` e8m0 views [pages, kv_heads, 128, 4]
    (``dtypes.fp8_e8m0`` or uint8) of the same pages (the four share page and head
    byte strides, multiples of 16). MXFP4 KV is per-token KV: those e8m0 bytes are
    each row's scales. Its kernels are requested with ``per_token_kv=True`` and,
    on gfx950 and gfx1250 alike, read only those bytes (no f32 per-token tensors).
    ``query`` is float8_e4m3fn, used as stored, or bf16, quantized to e4m3 per
    head row in the kernel; ``output`` is bf16. Supported: head_dim 128, block size
    128, at most 16 query heads per KV head. gfx950 takes ``query_length`` 1 and no
    sliding window; gfx1250 also takes MTP query lengths and sliding windows, with
    at most 64 query rows (query positions / query splits times GQA heads) a CTA:
    by default the positions split by 1, 2 or 4 so that a CTA keeps at most 32
    rows, or else by the largest of those dividing ``query_length`` (MTP3 x G16
    keeps 48 rows, MTP5 x G12 60; MTP5 x G16 is refused).
    Refused: other block sizes, head dims and output dtypes, and sinks
    (``mask_unowned_kv`` has no effect: tokens past the context are always masked).
    Scale bytes 0 and 255 are not supported in valid rows. V is widened to bf16,
    so a V element above the bf16 maximum (3.39e38: e8m0 byte 254 with an e2m1
    magnitude of 2 or more, byte 253 with 4 or more) widens to infinity even when
    the normalized output is finite. ``work_plan=None`` is accepted for MXFP4
    only: one task per sequence writes its rows directly (the output needs the
    query's row and head strides, a context-0 row is zero-filled) and nothing is
    specialized on ``num_seqs``; a plan with ``max_partitions == 1`` is served
    the same way. On gfx950 a task may span at most 4096 pages in
    ``max_context_length``: pass a smaller one otherwise. A real context longer
    than that bound is still fully attended (up to 2**19 tokens a task, the bound
    of the fixed PV shift, unchecked on the GPU). gfx1250 reads page ids from the
    block table, without that bound, and needs 16 B aligned query and output base
    pointers.

    Refresh ``work_plan`` on the current stream after changing lengths. Its
    max_partitions is bounded by device CUs. Match sliding_window and, for
    windowed plans, query_length. Scratch is [KV heads, capacity, query rows
    (, D)], with query rows = query_length * query_group_size.

    ``max_context_length`` selects the scheduling bound, rounded up to
    whole pages as in tuning. This keeps the selected specialization independent
    of extra block-table padding. It is the caller's contract, including for
    graph replay: every context length that will run must be at most
    ``max_context_length``. The reducer and in-kernel merges size their
    partition bound from it, so a smaller value can drop partitions and
    silently truncate attention, including on graph replay.

    ``mask_unowned_kv`` (default True) zeroes, on gfx942/gfx950, V bytes and
    K/V scales of tokens at or past each sequence's context length before PV,
    so pinned block-0 pages and the unwritten tail of the last page may hold
    any bytes, including NaN. False skips that masking (slightly faster on
    tiny boundary-tile-only calls); the caller must then keep those bytes and
    scales finite. gfx1250 ignores it and always requires finite bytes.

    GPU lengths and table entries are not checked. Callers must ensure
    0 <= context_lengths[i] <= block_tables.shape[1] * block_size and used
    block indices in [0, min(key_cache.shape[0], value_cache.shape[0])).
    """
    options = operands = None
    if (
        _VALIDATED_LAUNCHES is not None
        # Never recorded: sinks (the reducer applies them), partials that
        # pa_decode allocates and operands it refuses.
        and sinks is None
        and exp_sums is not None
        and max_logits is not None
        and temporary_output is not None
        and query_scale is None
        and alibi_slopes is None
    ):
        operands = (
            temporary_output,
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            key_scale,
            value_scale,
            max_logits,
            exp_sums,
            quantized_output,
            output_scale,
        )
        try:
            options = _launch_options(
                work_plan,
                softmax_scale,
                query_length,
                context_partition_size,
                compute_type,
                sliding_window,
                max_context_length,
                mask_unowned_kv,
                store_output,
                quantized_pad_rows,
                # Whether the call has an FP8 output, and its kind of scales:
                # the quant fallback and constant scales are never recorded,
                # and so share no key with a call that is.
                quantized_output is None,
                type(key_scale),
            )
            token = _VALIDATED_LAUNCHES.get(options)
        except (TypeError, RuntimeError):
            # Not a PADecodePlan of plain tensors, or an unhashable option.
            options = token = None
        if token is not None:
            launch = _VALIDATED_LAUNCHES.get((token, _launch_operands(operands)))
            if launch is not None:
                launch(torch.cuda.current_stream(), work_plan, *operands)
                return

    mxfp4 = isinstance(key_cache, torch.Tensor) and key_cache.dtype in (
        torch.uint8,
        dtypes.fp4x2,
    )
    if not isinstance(work_plan, PADecodePlan) and not (mxfp4 and work_plan is None):
        raise TypeError("work_plan must be a PADecodePlan")
    if context_partition_size != KV_COMPUTE_BLOCK:
        raise NotImplementedError(
            "pa_decode only supports context_partition_size=256, "
            f"got {context_partition_size}"
        )
    quant_output = quantized_output is not None or output_scale is not None
    if (quantized_output is None) != (output_scale is None):
        raise ValueError(
            "quantized_output and output_scale must either both be provided or both "
            "be None"
        )
    if not isinstance(store_output, bool):
        raise TypeError(
            f"store_output must be a bool, got {type(store_output).__name__}"
        )
    if not (store_output or quant_output):
        raise ValueError("store_output=False requires quantized_output")
    if (
        isinstance(quantized_pad_rows, bool)
        or not isinstance(quantized_pad_rows, int)
        or quantized_pad_rows < 0
    ):
        raise ValueError("quantized_pad_rows must be a nonnegative int")
    if quantized_pad_rows and not quant_output:
        raise ValueError("quantized_pad_rows requires quantized_output")
    if query_scale is not None:
        raise NotImplementedError(
            "pa_decode does not support externally quantized FP8 queries"
        )
    if alibi_slopes is not None:
        raise NotImplementedError("pa_decode does not support ALiBi")
    # Masked scores are scaled after masking on some paths, so a negative or
    # zero scale would give masked tokens weight.
    if not (math.isfinite(softmax_scale) and softmax_scale > 0):
        raise ValueError(
            f"softmax_scale must be finite and positive, got {softmax_scale}"
        )
    if not isinstance(sliding_window, int):
        raise TypeError("sliding_window must be an int")
    sliding_window = max(sliding_window, 0)
    if not isinstance(query_length, int):
        raise TypeError("query_length must be an int")
    if query_length < 1:
        raise ValueError(f"query_length must be positive, got {query_length}")

    required_tensors = (
        ("output", output),
        ("query", query),
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("context_lengths", context_lengths),
        ("block_tables", block_tables),
    )
    quant_tensors = (
        (("quantized_output", quantized_output), ("output_scale", output_scale))
        if quant_output
        else ()
    )
    for name, tensor in required_tensors + quant_tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor, got {type(tensor).__name__}"
            )

    expected_ranks = (
        ("output", output, (3,)),
        ("query", query, (3,)),
        ("key_cache", key_cache, (4, 5)),
        ("value_cache", value_cache, (4, 5)),
        ("context_lengths", context_lengths, (1,)),
        ("block_tables", block_tables, (2,)),
    )
    for name, tensor, ranks in expected_ranks:
        if tensor.dim() not in ranks:
            expected = " or ".join(f"{rank}D" for rank in ranks)
            raise ValueError(
                f"{name} must be {expected}, got shape {tuple(tensor.shape)}"
            )

    if mxfp4:
        if get_gfx_runtime() == "gfx1250-strict":
            raise NotImplementedError(MXFP4_GFX1250_STRICT_REFUSAL)
        if get_gfx_runtime() != "gfx1250":
            if query_length != 1:
                raise NotImplementedError(
                    f"MXFP4 KV supports query_length 1, got {query_length}"
                )
            if sliding_window:
                raise NotImplementedError(
                    "MXFP4 KV supports no sliding window, "
                    f"got sliding_window={sliding_window}"
                )
        if sinks is not None:
            raise NotImplementedError("MXFP4 KV supports no sinks, got sinks")
        if quant_output:
            raise NotImplementedError("MXFP4 KV supports no quantized output")
        if compute_type not in (None, torch.uint8, dtypes.fp4x2):
            raise NotImplementedError(
                "MXFP4 KV takes compute_type None, uint8 or fp4x2, "
                f"got {compute_type}"
            )
        return _pa_decode_mxfp4(
            output,
            query,
            key_cache,
            value_cache,
            key_scale,
            value_scale,
            context_lengths,
            block_tables,
            softmax_scale,
            exp_sums,
            max_logits,
            temporary_output,
            work_plan,
            max_context_length,
            query_length,
            sliding_window,
        )

    num_seqs = context_lengths.shape[0]
    total_q_rows, num_q_heads, head_dim = query.shape
    if query.device.type != "cuda":
        raise ValueError(f"query must be on a CUDA device, got {query.device}")
    if num_q_heads < 1:
        raise ValueError(f"query must contain at least one head, got {num_q_heads}")
    if total_q_rows != num_seqs * query_length:
        raise ValueError(
            f"query.shape[0] ({total_q_rows}) must equal "
            f"context_lengths.shape[0] * query_length ({num_seqs} * {query_length})"
        )
    if output.shape != query.shape:
        raise ValueError(
            f"output shape {tuple(output.shape)} must match "
            f"query shape {tuple(query.shape)}"
        )
    if block_tables.shape[0] != num_seqs:
        raise ValueError(
            f"block_tables.shape[0] ({block_tables.shape[0]}) must match "
            f"context_lengths.shape[0] ({num_seqs})"
        )

    trans_k = key_cache.dim() == 5
    trans_v = value_cache.dim() == 5
    nhd_layout = not trans_k
    if nhd_layout:
        num_blocks, block_size, num_kv_heads, k_head = key_cache.shape
        if k_head != head_dim:
            raise ValueError(
                "NHD key_cache shape must be "
                "[num_blocks, block_size, num_kv_heads, head_dim], "
                f"got {tuple(key_cache.shape)} for head_dim={head_dim}"
            )
    else:
        num_blocks, num_kv_heads, num_hgroups, block_size, hgroup_width = (
            key_cache.shape
        )
    if num_kv_heads < 1:
        raise ValueError(
            f"key_cache must contain at least one KV head, got {num_kv_heads}"
        )

    arch = get_gfx_runtime()
    if arch not in ("gfx942", "gfx950", "gfx1250"):
        raise NotImplementedError(
            f"pa_decode only supports gfx942, gfx950 and gfx1250, got {arch}"
        )
    is_bf16_kv = key_cache.dtype == torch.bfloat16
    expected_kv_dtype = torch.bfloat16 if is_bf16_kv else dtypes.fp8
    compute_type = key_cache.dtype if compute_type is None else compute_type
    if compute_type != expected_kv_dtype:
        raise NotImplementedError(
            f"pa_decode requires compute_type={expected_kv_dtype} for the KV cache "
            f"on {arch}, got {compute_type}"
        )
    if nhd_layout:
        require_nhd_pipe(
            arch,
            torch_to_fx_dtype(expected_kv_dtype),
            head_dim,
            block_size,
            query_length,
            num_q_heads // num_kv_heads,
        )
    vector_width = 8 if is_bf16_kv else 16

    # Enforce tile Q-load constraints and the reducer's 1024-thread limit.
    q_chunk = head_dim // 16
    if not (
        64 <= head_dim <= 1024
        and head_dim % 64 == 0
        and (
            (is_bf16_kv and arch not in ("gfx1250", "gfx1250-strict"))
            or q_chunk <= 8
            or q_chunk % 8 == 0
        )
    ):
        raise NotImplementedError(
            f"pa_decode does not support head_dim={head_dim}; supported values "
            + (
                "are multiples of 64 in [64, 1024] for BF16 KV"
                if is_bf16_kv and arch not in ("gfx1250", "gfx1250-strict")
                else "are 64 and multiples of 128 in [128, 1024]"
            )
        )
    if not nhd_layout and (
        num_hgroups != head_dim // vector_width or hgroup_width != vector_width
    ):
        raise ValueError(
            "key_cache shape must be "
            f"[num_blocks, num_kv_heads, head_dim // {vector_width}, "
            f"block_size, {vector_width}], "
            f"got {tuple(key_cache.shape)} for head_dim={head_dim}"
        )
    if block_size not in (16, 64, 128):
        raise ValueError(
            f"pa_decode only supports block_size in (16, 64, 128), got {block_size}"
        )

    if is_bf16_kv and arch not in ("gfx1250", "gfx1250-strict") and not trans_v:
        raise ValueError("BF16 KV requires the vectorized 5D value_cache layout")
    # The gfx1250 launcher has no NHD stride parameters.
    nhd_strides = () if arch in ("gfx1250", "gfx1250-strict") else (0,) * 6
    if nhd_layout:
        if value_cache.shape != key_cache.shape:
            raise ValueError(
                "NHD value_cache shape must match key_cache, "
                f"got {tuple(value_cache.shape)} vs {tuple(key_cache.shape)}"
            )
        if key_cache.stride(-1) != 1 or value_cache.stride(-1) != 1:
            raise ValueError("NHD key and value head axes must be contiguous")
        nhd_strides = _nhd_strides(key_cache) + _nhd_strides(value_cache)
        for name, view, strides in (
            ("key_cache", key_cache, nhd_strides[:3]),
            ("value_cache", value_cache, nhd_strides[3:]),
        ):
            if any(stride < 0 for stride in strides):
                raise ValueError(
                    f"NHD {name} stepped strides must be non-negative, got "
                    f"{view.stride()}"
                )
            # K loads and V LDS-DMAs move 16-byte chunks.
            if view.data_ptr() % 16 or any(stride % 16 for stride in strides):
                raise ValueError(
                    f"NHD {name} needs a 16-byte aligned base and block, token "
                    f"and head strides that are multiples of 16, got data_ptr "
                    f"{view.data_ptr():#x} and strides {view.stride()}"
                )
            # The V tail bound is live_tokens * token_stride bytes.
            if strides[1] < head_dim:
                raise ValueError(
                    f"NHD {name} token rows must not overlap, got token stride "
                    f"{view.stride(1)} for head_dim={head_dim}"
                )
        # Strides are i32 kernel arguments, and the in-page offset and V tail
        # bound stay i32 even in wide mode.
        k_block, k_token, k_head, v_block, v_token, v_head = nhd_strides
        page_span = block_size * max(k_token, v_token)
        if max(k_block, k_head, v_block, v_head, page_span) >= 2**31:
            raise NotImplementedError(
                "NHD pa_decode requires int32 block and head strides and "
                "block_size * token_stride < 2**31, got strides "
                f"{key_cache.stride()} and {value_cache.stride()}"
            )
        v_num_blocks, v_num_kv_heads = num_blocks, num_kv_heads
    elif trans_v:
        v_num_blocks, v_num_kv_heads = value_cache.shape[:2]
        expected_v_tail = (block_size // vector_width, head_dim, vector_width)
        if tuple(value_cache.shape[2:]) != expected_v_tail:
            raise ValueError(
                "transposed value_cache shape must be "
                f"[num_blocks, num_kv_heads, block_size // {vector_width}, "
                f"head_dim, {vector_width}], "
                f"got {tuple(value_cache.shape)} for block_size={block_size}, "
                f"head_dim={head_dim}"
            )
    else:
        v_num_blocks, v_num_kv_heads, v_head_dim, v_block_size = value_cache.shape
        if v_head_dim != head_dim or v_block_size != block_size:
            raise ValueError(
                "value_cache shape must be "
                "[num_blocks, num_kv_heads, head_dim, block_size], "
                f"got {tuple(value_cache.shape)} for block_size={block_size}, "
                f"head_dim={head_dim}"
            )
    # Packed V is a shifted view and may span fewer blocks than K.
    if v_num_blocks > num_blocks:
        raise ValueError(
            f"value_cache must not span more blocks than key_cache, "
            f"got {num_blocks} and {v_num_blocks}"
        )
    if v_num_kv_heads != num_kv_heads:
        raise ValueError(
            "key_cache and value_cache must have the same number of KV heads, "
            f"got {num_kv_heads} and {v_num_kv_heads}"
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads})"
        )

    if block_tables.dtype != torch.int32:
        raise TypeError(f"block_tables must be int32, got {block_tables.dtype}")
    if context_lengths.dtype != torch.int32:
        raise TypeError(f"context_lengths must be int32, got {context_lengths.dtype}")
    query_group_size = num_q_heads // num_kv_heads
    max_blocks_per_seq = block_tables.shape[1]
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"pa_decode only supports f16/bf16 query, got {query.dtype}")
    if (
        is_bf16_kv
        and arch not in ("gfx1250", "gfx1250-strict")
        and query.dtype != torch.bfloat16
    ):
        raise NotImplementedError("BF16 KV requires bfloat16 queries")
    if output.dtype != query.dtype:
        raise TypeError(
            "pa_decode requires output.dtype == query.dtype, "
            f"got {output.dtype} vs {query.dtype}"
        )
    if quant_output:
        if arch != "gfx1250":
            raise NotImplementedError("quantized PA output requires gfx1250")
        expected_quant_shape = (
            output.shape[0] + quantized_pad_rows,
            *output.shape[1:],
        )
        if tuple(quantized_output.shape) != expected_quant_shape:
            raise ValueError(
                "quantized_output shape must be output's plus quantized_pad_rows "
                f"rows, got {tuple(quantized_output.shape)} vs "
                f"{expected_quant_shape}"
            )
        if quantized_output.dtype != dtypes.fp8:
            raise TypeError(
                f"quantized_output must use OCP FP8 ({dtypes.fp8}), "
                f"got {quantized_output.dtype}"
            )
        expected_scale_shape = (total_q_rows + quantized_pad_rows, 1)
        if output_scale.shape != expected_scale_shape:
            raise ValueError(
                f"output_scale shape {tuple(output_scale.shape)} must be "
                f"{expected_scale_shape}"
            )
        if output_scale.dtype != torch.float32:
            raise TypeError(f"output_scale must be float32, got {output_scale.dtype}")
        if not (
            query.is_contiguous()
            and output.is_contiguous()
            and quantized_output.is_contiguous()
        ):
            raise ValueError(
                "quantized PA output requires contiguous query, output and "
                "quantized_output tensors"
            )
        if not output_scale.is_contiguous():
            raise ValueError("output_scale must be contiguous")

    if key_cache.dtype != expected_kv_dtype:
        raise TypeError(
            f"pa_decode requires {expected_kv_dtype} key cache on {arch}, "
            f"got {key_cache.dtype}"
        )
    if value_cache.dtype != expected_kv_dtype:
        raise TypeError(
            f"pa_decode requires {expected_kv_dtype} value cache on {arch}, "
            f"got {value_cache.dtype}"
        )

    if query.stride(2) != 1:
        raise ValueError(
            "pa_decode requires a contiguous query head_dim axis, "
            f"got strides {query.stride()}"
        )
    if output.stride(2) != 1:
        raise ValueError(
            "pa_decode requires a contiguous output head_dim axis, "
            f"got strides {output.stride()}"
        )

    dev = query.device
    if sinks is not None:
        if not isinstance(sinks, torch.Tensor):
            raise TypeError("sinks must be a torch.Tensor")
        if sinks.shape != (num_q_heads,):
            raise ValueError("sinks must have shape [num_query_heads]")
        if sinks.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError("sinks must have dtype bfloat16, float16, or float32")
        if sinks.device != dev:
            raise ValueError("sinks must be on the same device as query")
        if not sinks.is_contiguous():
            raise ValueError("sinks must be contiguous")
    for name, tensor in (
        ("output", output),
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("block_tables", block_tables),
        ("context_lengths", context_lengths),
        *quant_tensors,
    ):
        if tensor.device != dev:
            raise ValueError(
                f"{name} must be on the same device as query ({dev}), "
                f"got {tensor.device}"
            )
    for name, tensor in (
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("block_tables", block_tables),
        ("context_lengths", context_lengths),
    ):
        if nhd_layout and name in ("key_cache", "value_cache"):
            continue
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def normalize_scale(scale, name):
        if scale is None:
            return _constant_scale(dev, 1.0)
        if not isinstance(scale, torch.Tensor):
            # Per-tensor scores are masked before the scale is applied, which
            # needs a positive scale. Device tensors are not read back (no sync).
            if not (math.isfinite(float(scale)) and float(scale) > 0):
                raise ValueError(f"{name} must be finite and positive, got {scale}")
            return _constant_scale(dev, float(scale))
        if scale.numel() == 1:
            return scale.reshape(1)
        if scale.dim() == 4:
            if scale.shape[-1] != 1:
                raise ValueError(
                    f"{name} must have a trailing singleton dimension, "
                    f"got shape {tuple(scale.shape)}"
                )
            return scale.squeeze(-1)
        if scale.dim() != 3:
            raise ValueError(
                f"{name} must be scalar or have shape "
                "[num_blocks, num_kv_heads, block_size, 1]"
            )
        return scale

    if is_bf16_kv:
        if key_scale is not None or value_scale is not None:
            raise ValueError(
                "BF16 KV is unscaled; key_scale and value_scale must be None"
            )
        # The native BF16 specialization never dereferences scale pointers.
        # Null pointers avoid allocating dummy scales during graph capture.
        key_scale_t = value_scale_t = None
        per_token_kv = False
        stride_ks_block = stride_ks_head = 0
    else:
        if (key_scale is None) != (value_scale is None):
            raise ValueError(
                "key_scale and value_scale must either both be provided or both be None"
            )
        key_scale_t = normalize_scale(key_scale, "key_scale")
        value_scale_t = normalize_scale(value_scale, "value_scale")
        per_token_kv = key_scale_t.numel() > 1
        if per_token_kv != (value_scale_t.numel() > 1):
            raise ValueError(
                "key_scale and value_scale must both be per-tensor or both be per-token"
            )
        if per_token_kv:
            if key_scale_t.shape != value_scale_t.shape:
                raise ValueError(
                    "key_scale/value_scale shape mismatch: "
                    f"{tuple(key_scale_t.shape)} vs {tuple(value_scale_t.shape)}"
                )
            expected_scale_shape = (num_blocks, num_kv_heads, block_size)
            if key_scale_t.shape != expected_scale_shape:
                raise ValueError(
                    "per-token key_scale/value_scale must be "
                    "[num_blocks, num_kv_heads, block_size] matching the KV cache, "
                    f"got {tuple(key_scale_t.shape)}"
                )
            stride_ks_block = int(key_scale_t.stride(0))
            stride_ks_head = int(key_scale_t.stride(1))
            if key_scale_t.stride(2) != 1:
                raise ValueError(
                    "per-token key_scale token dimension must be contiguous, "
                    f"got strides {key_scale_t.stride()}"
                )
            if value_scale_t.stride() != key_scale_t.stride():
                raise ValueError(
                    "per-token key_scale and value_scale must have matching strides, "
                    f"got {key_scale_t.stride()} vs {value_scale_t.stride()}"
                )
        else:
            stride_ks_block = 0
            stride_ks_head = 0
        for name, scale in (("key_scale", key_scale_t), ("value_scale", value_scale_t)):
            if scale.dtype != torch.float32:
                raise TypeError(f"{name} tensor must be float32, got {scale.dtype}")
            if scale.device != dev:
                raise ValueError(
                    f"{name} tensor must be on the same device as query ({dev}), "
                    f"got {scale.device}"
                )

    schedule_context_length = int(max_blocks_per_seq) * int(block_size)
    if max_context_length is not None:
        if (
            isinstance(max_context_length, bool)
            or not isinstance(max_context_length, int)
            or not 0 <= max_context_length <= max_blocks_per_seq * block_size
        ):
            raise ValueError(
                "max_context_length must be a nonnegative host integer within "
                "block-table capacity"
            )
        schedule_context_length = (
            (max_context_length + block_size - 1) // block_size * block_size
        )
    work_plan.validate(num_seqs, num_kv_heads, dev)
    if work_plan.sliding_window != sliding_window:
        raise ValueError("sliding_window must match work_plan.sliding_window")
    if sliding_window > 0 and work_plan.query_length != query_length:
        raise ValueError("query_length must match work_plan.query_length")
    num_partitions = work_plan.max_partitions
    pmax = max_logits
    psum = exp_sums
    pout = temporary_output
    # Bound this call's partitions without shrinking the reusable plan or scratch.
    max_partitions = min(
        num_partitions,
        work_plan.capacity,
        max(1, -(-schedule_context_length // KV_COMPUTE_BLOCK)),
    )
    if sliding_window > 0:
        # Bound the unaligned MTP window without shrinking the reusable plan.
        window_tiles = (
            sliding_window + query_length - 2 + KV_COMPUTE_BLOCK - 1
        ) // KV_COMPUTE_BLOCK + 1
        max_partitions = min(max_partitions, window_tiles)
    # gfx1250 tile kernels write final output for one-partition plans and for
    # plans with one task slot per sequence: single tasks store directly, and
    # sequences given slots freed by empty ones merge in their last-arriving
    # CTA. That merge costs two device-scope round trips (sparse decode 64
    # rows: 17.7 us vs 10.3 us for tile plus reducer), so wider plans merge
    # within workgroup clusters instead.
    # gfx942/gfx950 one-partition plans (capacity == batch) store final rows
    # from the tile kernel too, without the reducer launch.
    direct_output = (
        arch in ("gfx942", "gfx950")
        and num_partitions == 1
        and sinks is None
        and output.stride() == query.stride()
    ) or (
        arch in ("gfx1250", "gfx1250-strict")
        and sinks is None
        and output.stride() == query.stride()
        and (
            num_partitions == 1
            or (
                work_plan.capacity == num_seqs
                and work_plan.arrival_counters is not None
                and max_partitions <= MERGE_PARTITIONS
                and head_dim <= MERGE_HEAD_DIM
                # Last-arriver partials use naturally aligned 64-bit atomics.
                # Internal torch allocations are aligned; supplied contiguous
                # views may have only element alignment.
                and (
                    pout is None
                    or (isinstance(pout, torch.Tensor) and pout.data_ptr() % 8 == 0)
                )
            )
        )
    )
    one_partition_output = direct_output and num_partitions == 1
    num_compute_units = torch.cuda.get_device_properties(dev).multi_processor_count
    # Other gfx1250 plans merge each sequence's tasks, at most P of them, in a
    # cluster of P CTAs. Its idle CTAs wait at the cluster barrier, so the grid
    # stays within the plan's or the default two-per-CU budget. gfx1250 keeps
    # 15 clusters resident per 16 CUs (240 on 256 CUs) whatever their size;
    # later clusters run as a second wave, which is slower than the reducer.
    # A cluster of at most CLUSTER_PARTITIONS CTAs splits each sequence's
    # tiles among its CTAs, so a 32-partition bound (8K contexts at the
    # default cap) still merges on chip.
    cluster_size = min(max_partitions, CLUSTER_PARTITIONS)
    # gfx1250 keeps 15 clusters resident per 16 CUs; batches with more
    # sequences share each cluster between several of them.
    cluster_seqs = 1
    while (
        -(-num_seqs // cluster_seqs) * num_kv_heads > num_compute_units * 15 // 16
        and 2 * cluster_seqs * cluster_size <= CLUSTER_PARTITIONS
    ):
        cluster_seqs *= 2
    cluster_merge = (
        cluster_size
        if (
            arch in ("gfx1250", "gfx1250-strict")
            and not direct_output
            and sinks is None
            and output.stride() == query.stride()
            and 2 <= max_partitions <= CLUSTER_GROUPED_PARTITIONS
            and num_seqs * cluster_size
            <= max(work_plan.capacity, -(-2 * num_compute_units // num_kv_heads))
            and -(-num_seqs // cluster_seqs) * num_kv_heads
            <= num_compute_units * 15 // 16
            and head_dim <= MERGE_HEAD_DIM
        )
        else 0
    )
    # Page16 D128 decode clusters of P CTAs over block tables of at most 16
    # steps give CTA p steps [p * 16 / P, (p + 1) * 16 / P), so it loads its
    # first page ids with the context length rather than after it. For rows of
    # 16 steps the runs are the planner's tiles. Batches within
    # one cluster per CU split each row into single steps, one per CTA,
    # whatever the partition bound (sparse decode at 4 rows: 5.3 -> 4.6 us);
    # their plans reserve the steps' partial slots.
    step_split = bool(
        cluster_merge
        and block_size == 16
        and head_dim == 128
        and query_length == 1
        and sliding_window == 0
        and max_blocks_per_seq * block_size <= CLUSTER_PARTITIONS * CLUSTER_STEP_TOKENS
    )
    step_cluster = (
        CLUSTER_PARTITIONS
        if step_split
        and cluster_seqs == 1
        and num_seqs * num_kv_heads * CLUSTER_PARTITIONS <= num_compute_units
        and num_seqs * CLUSTER_PARTITIONS <= work_plan.capacity
        else cluster_merge
    )
    tile_options = {
        "head_dim": head_dim,
        "query_group_size": query_group_size,
        "block_size": int(block_size),
        "num_seqs": num_seqs,
        "num_kv_heads": num_kv_heads,
        "num_compute_units": num_compute_units,
        "num_partitions": num_partitions,
        "softmax_scale": softmax_scale,
        "query_dtype": query.dtype,
        "kv_dtype": expected_kv_dtype,
        "per_token_kv": per_token_kv,
        "query_length": query_length,
        "trans_v": trans_v,
        "work_capacity": work_plan.capacity,
        "max_context_length": schedule_context_length,
        "sliding_window": sliding_window,
        "direct_output": direct_output,
    }
    # The runs and preloaded page ids assume the tile's 128-token steps. Tiles
    # that select 64-token steps (two query tiles, plain BF16 V, wider grids)
    # keep plan-derived clusters.
    cluster_steps = (
        step_split
        and CLUSTER_PARTITIONS % step_cluster == 0
        and num_seqs * step_cluster <= work_plan.capacity
        and gfx1250_tile_traits(
            **tile_options,
            cluster_merge=step_cluster,
            cluster_seqs=cluster_seqs,
            cluster_steps=True,
        ).TOKENS
        == CLUSTER_STEP_TOKENS
    )
    if cluster_steps:
        cluster_merge = step_cluster
    # Page128 clusters given a length bound instead give CTA p tiles
    # [p * run, (p + 1) * run), runs splitting the bound evenly, so they too
    # load their first page ids with the context length (decode at 8K, 1-8
    # rows: 3-5% faster). The bound's longest rows split as the planner
    # would; shorter rows use fewer CTAs. Runs beyond four tiles gain nothing.
    bound_tiles = -(-schedule_context_length // KV_COMPUTE_BLOCK)
    cluster_runs = bool(
        cluster_merge
        and not cluster_steps
        and cluster_seqs == 1
        and block_size == 128
        and max_context_length is not None
        and sliding_window == 0
        and num_seqs * cluster_merge <= work_plan.capacity
        and bound_tiles <= 4 * cluster_merge
    )
    run_tiles = -(-bound_tiles // cluster_merge) if cluster_runs else 0
    # Unfused plans write the BF16 output as usual and quantize it after with
    # the standalone per-token kernel, which is exactly the unfused path.
    quant_after_output = quant_output and not _fuses_quant_output(
        tile_options, cluster_merge=cluster_merge, cluster_seqs=cluster_seqs
    )
    if quant_after_output:
        quant_output = False
        if not store_output:
            # Keep the caller's BF16 buffer untouched, as the fused path does.
            output = torch.empty_like(output)

    # Bound byte offsets as well as element indices for 16-bit BF16 caches. A
    # strided NHD view can span more bytes than numel(), and its head stride
    # may scale with num_blocks, so measure the highest byte offset.
    if nhd_layout:
        cache_extent = (
            max(_view_span(key_cache), _view_span(value_cache))
            * key_cache.element_size()
        )
    else:
        cache_extent = max(
            key_cache.numel() * key_cache.element_size(),
            value_cache.numel() * value_cache.element_size(),
        )
    wide_kv_addressing = cache_extent >= 2**31
    kv_buffer_u32 = cache_extent < 2**32

    with torch.cuda.device(dev):
        compiled = compile_pa_decode_tile(
            architecture=arch,
            **tile_options,
            wide_kv_addressing=wide_kv_addressing,
            kv_buffer_u32=kv_buffer_u32,
            trans_k=trans_k,
            mask_unowned_kv=mask_unowned_kv,
            cluster_merge=cluster_merge,
            cluster_seqs=cluster_seqs,
            cluster_steps=cluster_steps,
            cluster_runs=cluster_runs,
            **(
                {"quant_output": True, "store_output": store_output}
                if quant_output
                else {}
            ),
        )

    total_rows = query_length * query_group_size
    expected_scalar_shape = (num_kv_heads, work_plan.capacity, total_rows)
    if pmax is None:
        pmax = torch.empty(*expected_scalar_shape, dtype=torch.float32, device=dev)
    if psum is None:
        psum = torch.empty(*expected_scalar_shape, dtype=torch.float32, device=dev)
    if pout is None:
        pout = torch.empty(
            *expected_scalar_shape, head_dim, dtype=output.dtype, device=dev
        )
    for name, tensor in (
        ("max_logits", pmax),
        ("exp_sums", psum),
        ("temporary_output", pout),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor, got {type(tensor).__name__}"
            )
    if pmax.shape != expected_scalar_shape:
        raise ValueError(
            f"max_logits shape {tuple(pmax.shape)} != {expected_scalar_shape}"
        )
    if psum.shape != expected_scalar_shape:
        raise ValueError(
            f"exp_sums shape {tuple(psum.shape)} != {expected_scalar_shape}"
        )
    expected_output_shape = (*expected_scalar_shape, head_dim)
    if pout.shape != expected_output_shape:
        raise ValueError(
            f"temporary_output shape {tuple(pout.shape)} != {expected_output_shape}"
        )
    if pmax.dtype != torch.float32:
        raise TypeError(f"max_logits must be float32, got {pmax.dtype}")
    if psum.dtype != torch.float32:
        raise TypeError(f"exp_sums must be float32, got {psum.dtype}")
    if pout.dtype != output.dtype:
        raise TypeError(
            f"temporary_output dtype {pout.dtype} must match output dtype "
            f"{output.dtype}"
        )
    for name, tensor in (
        ("max_logits", pmax),
        ("exp_sums", psum),
        ("temporary_output", pout),
    ):
        if tensor.device != dev:
            raise ValueError(
                f"{name} must be on the same device as query ({dev}), "
                f"got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    # What the checks and the dispatch fix of the launch, as `_tile_launch`
    # unpacks it.
    launch_spec = (
        compiled["launch"],
        arch in ("gfx1250", "gfx1250-strict"),
        one_partition_output,
        direct_output and not one_partition_output,  # merged
        bool(cluster_merge),
        quant_output,
        is_bf16_kv,
        # FlyDSL types: of the query, which the output and the partials share,
        # of the caches and of the FP8 output.
        torch_to_fx_dtype(query.dtype),
        torch_to_fx_dtype(key_cache.dtype),
        torch_to_fx_dtype(quantized_output.dtype) if quant_output else None,
        (
            int(max_blocks_per_seq),
            int(num_seqs),
            int(num_kv_heads),
            stride_ks_block,
            stride_ks_head,
            int(query.stride(0)),
            int(query.stride(1)),
            *nhd_strides,
        ),
        (
            run_tiles,
            # Rows of the FP8 output, padded ones included.
            int(total_q_rows + quantized_pad_rows) if quant_output else int(num_seqs),
        ),
    )
    if (
        arch == "gfx1250"
        and (direct_output or cluster_merge)
        # Not recorded: the standalone-quant fallback and constant FP8 KV
        # scales (None or host scalars), which are not operands.
        and not quant_after_output
        and (is_bf16_kv or isinstance(key_scale, torch.Tensor))
        and dev.index == torch.cuda.current_device()
    ):
        _record_launch(options, operands, partial(_tile_launch, launch_spec))
    with torch.cuda.device(dev):
        s = torch.cuda.current_stream(dev)
        _tile_launch(
            launch_spec,
            s,
            work_plan,
            pout,
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            key_scale_t,
            value_scale_t,
            pmax,
            psum,
            quantized_output,
            output_scale,
        )
        if not (direct_output or cluster_merge):
            output_5d = output.reshape(
                num_seqs, query_length, num_kv_heads, query_group_size, head_dim
            )
            launch_pa_decode_ps_reduce(
                output_5d,
                psum,
                pmax,
                pout,
                sinks,
                output_5d.stride(0),
                output_5d.stride(1),
                output_5d.stride(2),
                output_5d.stride(3),
                pmax.stride(0),
                pmax.stride(1),
                pout.stride(0),
                pout.stride(1),
                pout.stride(2),
                query_seq_len=query_length,
                query_group_size=query_group_size,
                head_size=head_dim,
                max_partitions=max_partitions,
                stream=s,
                reduce_info=work_plan.reduce_info,
            )
        if quant_after_output:
            from ..quant import dynamic_per_token_scaled_quant

            rows = output.shape[0]
            dynamic_per_token_scaled_quant(
                quantized_output[:rows].view(rows, -1),
                output.view(rows, -1),
                output_scale[:rows],
            )
            if quantized_pad_rows:
                quantized_output[rows:].zero_()
                output_scale[rows:].fill_(1.0)


# Launch arguments that a specialization does not read.
_NULL_I32 = flyc.from_c_void_p(fx.Int32, 0)
_NULL_F32 = flyc.from_c_void_p(fx.Float32, 0)
_NULL_FP8 = flyc.from_c_void_p(fx.Float8E4M3FN, 0)


def _tile_launch(
    spec,
    stream,
    work_plan,
    pout,
    output,
    query,
    key_cache,
    value_cache,
    block_tables,
    context_lengths,
    key_scale,
    value_scale,
    pmax,
    psum,
    quantized_output,
    output_scale,
):
    """Launch the tile kernel that ``spec`` describes on these operands and
    ``stream``. ``pa_decode`` records ``partial(_tile_launch, spec)``, so a
    repeated gfx1250 call builds its launch arguments as the checked call
    did."""
    (
        launch_fn,
        gfx1250,
        one_partition_output,
        merged,
        cluster_merge,
        quant_output,
        is_bf16_kv,
        out_fx,
        kv_fx,
        quant_fx,
        ints,
        tail,
    ) = spec
    merge_args = ()
    if gfx1250:
        merge_args = (
            ptr_arg(output, out_fx),
            ptr_arg(work_plan.reduce_info, fx.Int32),
            ptr_arg(work_plan.arrival_counters, fx.Int32) if merged else _NULL_I32,
            tail[0],
            ptr_arg(quantized_output, quant_fx) if quant_output else _NULL_FP8,
            ptr_arg(output_scale, fx.Float32) if quant_output else _NULL_F32,
            tail[1],
        )
    _run_compiled(
        launch_fn,
        (
            ptr_arg(work_plan.reduce_info, fx.Float32)
            if one_partition_output
            else ptr_arg(pmax, fx.Float32)
        ),
        ptr_arg(psum, fx.Float32),
        ptr_arg(output if one_partition_output else pout, out_fx),
        ptr_arg(query, out_fx),
        ptr_arg(key_cache, kv_fx),
        ptr_arg(value_cache, kv_fx),
        ptr_arg(block_tables, fx.Int32),
        _NULL_F32 if is_bf16_kv else ptr_arg(key_scale, fx.Float32),
        _NULL_F32 if is_bf16_kv else ptr_arg(value_scale, fx.Float32),
        *ints,
        # Cluster grids derive their tiles from the context lengths.
        ptr_arg(context_lengths if cluster_merge else work_plan.work_info, fx.Int32),
        work_plan.capacity,
        *merge_args,
        stream,
    )

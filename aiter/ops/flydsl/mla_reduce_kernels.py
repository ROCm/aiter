# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL MLA decode reduce API.

Drop-in alternative for the HIP ``aiter.mla_reduce_v1`` (csrc/kernels/mla/reduce.cu):
same signature and in/out contract. Opt-in via ``AITER_MLA_REDUCE_FLYDSL=1``;
production keeps the HIP kernel by default. Kernels are compiled per
(H, Dv, out_dtype, tier, output_lse) and ``lru_cache``-backed; ``Tier.ALL`` (one
kernel, device-side per-tile tier selection) mirrors HIP.
"""

from __future__ import annotations

import collections

import torch

from .kernels.mla_reduce import (
    Tier,
    _get_splitk_scratch,
    compile_mla_reduce,
    compile_mla_reduce_splitk,
    derive_actual_max_splits,
    plan_splitk_capture_safe,
    should_use_persistent_launch,
    waves_per_eu_from_env,
)

__all__ = [
    "flydsl_mla_reduce_v1",
]


# Low-direct-pmap threshold: when actual_max_splits <= this, read
# reduce_partial_map directly instead of staging it to LDS + a barrier.
_LOW_DIRECT_PMAP_THR = 8


def _out_dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    raise ValueError(
        f"flydsl_mla_reduce_v1 only supports bf16/fp16 output, got {dtype!r}"
    )


# Cache mapping reduce_indptr buffer identity -> derived actual_max_splits.
_ACTUAL_MAX_SPLITS_CACHE: collections.OrderedDict = collections.OrderedDict()
_ACTUAL_MAX_SPLITS_CACHE_CAP = 512


def _resolve_actual_max_splits(reduce_indptr: torch.Tensor) -> int | None:
    """Resolve ``actual_max_splits`` (true ``max_t(n_splits)``) capture-safely.

    Deriving the value needs a device read that is illegal under CUDA-graph capture.
    The eager/warmup pass reads the CSR and caches by buffer identity; under capture
    it returns the cached value, or ``None`` on a miss (caller falls back to the
    ``num_kv_splits`` gate). Never syncs during capture.
    """
    key = (reduce_indptr.data_ptr(), int(reduce_indptr.numel()))
    if torch.cuda.is_current_stream_capturing():
        return _ACTUAL_MAX_SPLITS_CACHE.get(key)
    val = derive_actual_max_splits(reduce_indptr)
    cache = _ACTUAL_MAX_SPLITS_CACHE
    if key in cache:
        cache.move_to_end(key)
    cache[key] = val
    if len(cache) > _ACTUAL_MAX_SPLITS_CACHE_CAP:
        cache.popitem(last=False)
    return val


def flydsl_mla_reduce_v1(
    partial_output: torch.Tensor,  # fp32 [rows, H, Dv] contiguous
    partial_lse: torch.Tensor,  # fp32 [rows, H] contiguous
    reduce_indptr: torch.Tensor,  # i32  [num_reduce_tile + 1]
    reduce_final_map: torch.Tensor | None,  # i32 [num_reduce_tile, 2] (optional)
    reduce_partial_map: torch.Tensor,  # i32 [reduce_indptr[-1]]
    max_seqlen_q: int,
    final_output: torch.Tensor,  # bf16/fp16 [bs, H, Dv]
    final_lse: torch.Tensor | None = None,  # fp32 [bs, H] (optional)
    num_kv_splits: int = 0,  # signature parity; grid derived from num_cu + CSR width
    actual_max_splits: int | None = None,  # true max tile width; gates DA split-K
    *,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """FlyDSL drop-in replacement for ``aiter.mla_reduce_v1`` (write-in-place).

    Results land in ``final_output`` (and ``final_lse`` if provided); no return.

    Args:
        partial_output / partial_lse: fp32 per-split partials.
        reduce_indptr / reduce_partial_map: CSR over splits (+ gather map).
        reduce_final_map: optional [tile, 2] {q_start, q_end}; ``None`` derives the
            q-range from the partial map (HIP ``use_reduce_final_map = false`` path).
        max_seqlen_q: q-positions per decode token group; drives grid-y (NTG).
        final_output: output tensor (bf16/fp16); strides read at runtime.
        final_lse: optional merged LSE output (fp32).
        num_kv_splits: HIP-signature parity; the split-K engagement budget when
            ``actual_max_splits`` is unresolved.
        actual_max_splits: true ``max_t(n_splits)`` gating split-K + low-direct-pmap.
            Normally ``None`` -> auto-resolved from ``reduce_indptr``
            (:func:`_resolve_actual_max_splits`); an explicit value is used as-is.
        stream: launch stream; defaults to the device's current stream.
    """
    if reduce_indptr.numel() < 2:
        return

    if actual_max_splits is None:
        actual_max_splits = _resolve_actual_max_splits(reduce_indptr)

    H = partial_output.size(-2)
    Dv = final_output.size(-1)
    out_dtype_str = _out_dtype_str(final_output.dtype)
    num_reduce_tile = reduce_indptr.numel() - 1
    output_lse = final_lse is not None
    use_reduce_final_map = reduce_final_map is not None

    if final_lse is None:
        final_lse = torch.empty(1, dtype=torch.float32, device=final_output.device)
    if reduce_final_map is None:
        reduce_final_map = torch.empty(1, dtype=torch.int32, device=final_output.device)
    if stream is None:
        stream = torch.cuda.current_stream(final_output.device)

    num_cu = torch.cuda.get_device_properties(
        final_output.device.index
    ).multi_processor_count

    use_persistent = should_use_persistent_launch(
        H=H,
        max_seqlen_q=max_seqlen_q,
        num_reduce_tile=num_reduce_tile,
        num_cu=num_cu,
    )

    # Device-adaptive split-K: for low-tile/high-split decode, split each active
    # tile's reduction across K cooperating blocks then combine. Needs
    # reduce_final_map for the combine's per-tile q-range.
    if use_reduce_final_map:
        engage_sk, sk_K, sk_slots = plan_splitk_capture_safe(
            num_final_rows=int(final_output.size(0)),
            H=H,
            max_seqlen_q=max_seqlen_q,
            num_kv_splits=int(num_kv_splits),
            num_cu=num_cu,
            actual_max_splits=actual_max_splits,
        )
        if engage_sk:
            launch_partial, launch_combine = compile_mla_reduce_splitk(
                H=H,
                Dv=Dv,
                out_dtype=out_dtype_str,
                K=sk_K,
                output_lse=output_lse,
                waves_per_eu=waves_per_eu_from_env(),
            )
            sk_acc, sk_ml = _get_splitk_scratch(
                sk_slots, sk_K, Dv, final_output.device.index
            )
            launch_partial(
                partial_output,
                partial_lse,
                reduce_indptr,
                reduce_partial_map,
                sk_acc,
                sk_ml,
                int(partial_output.size(0)),
                sk_slots * sk_K,
                stream,
            )
            launch_combine(
                reduce_final_map,
                sk_acc,
                sk_ml,
                final_output,
                final_lse,
                int(final_output.stride(-3)),
                int(final_output.stride(-2)),
                int(final_output.size(0)),
                sk_slots,
                stream,
            )
            return

    # Tier.ALL: one kernel with a device-side per-tile tier branch (always correct).
    tier = Tier.ALL

    # Adaptive launch: on a sparse grid, launch one block per active
    # (tile, head, q-group) instead of the persistent grid-stride kernel.
    # Multi-tile decode only (single-tile is split-K's domain).
    num_final_rows = int(final_output.size(0))
    use_adaptive = (
        use_reduce_final_map
        and num_final_rows > 1
        and num_final_rows < num_reduce_tile
        and num_final_rows * H * max_seqlen_q <= 4 * num_cu
    )

    low_direct_pmap_thr = (
        _LOW_DIRECT_PMAP_THR
        if use_adaptive
        and actual_max_splits is not None
        and int(actual_max_splits) <= _LOW_DIRECT_PMAP_THR
        else 0
    )

    kernel = compile_mla_reduce(
        H=H,
        Dv=Dv,
        out_dtype=out_dtype_str,
        tier=tier,
        persistent=use_persistent and not use_adaptive,
        output_lse=output_lse,
        use_reduce_final_map=use_reduce_final_map,
        waves_per_eu=waves_per_eu_from_env(),
        adaptive=use_adaptive,
        low_direct_pmap_thr=low_direct_pmap_thr,
    )

    kernel(
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_partial_map,
        reduce_final_map,
        final_output,
        final_lse,
        int(final_output.stride(-3)),
        int(final_output.stride(-2)),
        int(num_cu),
        int(num_reduce_tile),
        int(max_seqlen_q),
        int(partial_output.size(0)),
        int(final_output.size(0)),
        stream,
    )

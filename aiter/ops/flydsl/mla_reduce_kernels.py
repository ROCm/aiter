# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL MLA decode reduce API.

Drop-in fallback for the HIP ``aiter.mla_reduce_v1`` (csrc/kernels/mla/reduce.cu).
Same signature, same input/output contract, so production paths can swap it in.

The FlyDSL kernel is compiled per (H, Dv, out_dtype, tier, output_lse) and cached
(``compile_mla_reduce`` is ``lru_cache``-backed). Production uses ``Tier.ALL``:
one kernel with device-side runtime tier selection per tile (mirrors HIP).

This is an OPT-IN fallback. Production code (``aiter/mla.py``) keeps calling the
HIP ``aiter.mla_reduce_v1`` by default; set ``AITER_MLA_REDUCE_FLYDSL=1`` to route
through this wrapper instead.
"""

from __future__ import annotations

import collections
import os

import torch

from .kernels.mla_reduce import (
    LDS_MAX_SPLITS,
    Tier,
    _get_splitk_scratch,
    compile_mla_reduce,
    compile_mla_reduce_splitk,
    derive_actual_max_splits,
    plan_splitk_capture_safe,
    select_tier,
    should_use_persistent_launch,
    waves_per_eu_from_env,
)

__all__ = [
    "flydsl_mla_reduce_v1",
]


def _adaptive_launch_enabled() -> bool:
    """Gate for the adaptive (active_tiles x H) launch.

    Default-on for multi-tile sparse decode (``num_final_rows > 1`` and
    ``num_final_rows < num_reduce_tile``). Set
    ``AITER_MLA_REDUCE_ADAPTIVE_LAUNCH=0`` to force the persistent grid-stride
    launch. Single-tile decode (bs=1) always uses persistent or split-K.
    """
    return os.environ.get("AITER_MLA_REDUCE_ADAPTIVE_LAUNCH", "1") == "1"


def _host_tier_dispatch_enabled() -> bool:
    """Opt-in gate for host-visible per-tier dispatch (capture-safe).

    Default (unset/0): ``Tier.ALL`` — one kernel with device-side per-tile tier
    branch; correct for any data, but its VGPR footprint is set by the heaviest
    body (M256/MLDS), which drags the M64 mid-tail to occupancy 1.

    When set (``AITER_MLA_REDUCE_HOST_TIER=1``): the wrapper picks the tier on the
    HOST from ``num_kv_splits`` (via :func:`_safe_host_tier`) and compiles that
    single per-tier body. It is capture-safe because ``num_kv_splits`` is a pure
    host scalar (no device read/sync) and, for a fixed CUDA-graph capture
    configuration, is constant.

    CORRECTNESS PRECONDITION: a per-tier body reduces a tile's actual ``n_splits``
    correctly ONLY up to its fixed LSE cap (M64=64, M256=256; SIMPLE/MLDS loop or
    cover all 304). So ``select_tier(num_kv_splits)`` is correct ONLY IF
    ``num_kv_splits`` is a true upper bound on EVERY active tile's ``n_splits``.

    WHY THIS STAYS OPT-IN (exp3 finding, 2026-07-06): the ``num_kv_splits`` that
    reaches this wrapper on the real dispatch (``aiter/mla.py`` ->
    ``_mla_reduce_v1_dispatch``) is the metadata's ``max_split_per_batch`` -- a
    per-BATCH split *budget*, NOT a per-tile upper bound. The metadata's greedy
    load balancer can concentrate up to ``min(num_clusters, max_split_per_batch *
    num_batches)`` splits on a single skewed tile (measured: a per-tile n_splits of
    171 at ``max_split_per_batch=32``), which silently overflows a baked M64/M256
    body. So ``select_tier(num_kv_splits)`` is NOT safe in general; only the
    ``num_kv_splits=None`` dispatch path (which fills
    ``get_mla_decode_fwd_max_splits`` = ``num_clusters``) is a true bound, and that
    selects MLDS (no occupancy win over ``Tier.ALL``). :func:`_safe_host_tier`
    therefore guards against under-baking. The mid-tail per-tier win is not
    realizable through this host scalar; it needs the CUDA-graph conditional-node
    split (see the traversal results log).
    """
    return os.environ.get("AITER_MLA_REDUCE_HOST_TIER", "0") == "1"


# Low-split direct-pmap path threshold. For adaptive multi-tile decode whose
# true per-tile max split (``actual_max_splits``) is <= this, the reduce reads
# reduce_partial_map directly instead of staging it to LDS + a barrier first
# (measured faster on the low-split serving shapes; high-split captures keep the
# vectorized LDS pmap path via a separate compiled kernel). Always on.
_LOW_DIRECT_PMAP_THR = 8


def _safe_host_tier(num_kv_splits: int) -> Tier:
    """Capture-safe host tier from ``num_kv_splits``, GUARDED against under-baking.

    The true per-tile split count can reach the kernel's hard cap
    ``LDS_MAX_SPLITS`` (= ``num_clusters`` on the real metadata) and is NOT bounded
    by ``num_kv_splits`` (= ``max_split_per_batch``, a per-batch budget) -- see
    :func:`_host_tier_dispatch_enabled`. A fixed-nlse M64/M256 body silently drops
    splits beyond its LSE cap, so it is only safe to bake a sub-MLDS tier when
    ``num_kv_splits`` already provably covers the worst case. Since that cannot be
    verified capture-safely from ``num_kv_splits`` alone, only ``Tier.MLDS``
    (nlse=5 -> covers 320 >= LDS_MAX_SPLITS) is trusted; every smaller selection
    falls back to the always-correct device-side ``Tier.ALL`` (which reduces each
    tile's actual ``n_splits`` per-tile). This makes the opt-in flag incapable of
    silently corrupting output for any input.
    """
    assert LDS_MAX_SPLITS <= 320, "MLDS body (nlse=5) covers 320 splits"
    tier = select_tier(num_kv_splits)
    return tier if tier is Tier.MLDS else Tier.ALL


def _out_dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    raise ValueError(
        f"flydsl_mla_reduce_v1 only supports bf16/fp16 output, got {dtype!r}"
    )


# ---- actual_max_splits resolution (capture-safe, warmup-populated) ----------
# actual_max_splits (true max_t(n_splits) over active tiles) gates two host-side
# decisions that must be BAKED at CUDA-graph capture time: split-K engage/K
# (plan_splitk_capture_safe) and the low-direct-pmap kernel variant. Deriving it
# needs a device read (.item()), which is illegal under graph capture.
#
# The standard graph pattern is warmup(eager) -> capture -> replay. We derive the
# true value on the eager/warmup pass (sync allowed) and cache it keyed by the
# reduce_indptr BUFFER identity. Graph capture requires static input tensors, so
# a bucket's reduce_indptr is a fixed buffer -- warmup and capture see the same
# pointer, and capture reuses the warmup-derived value. A capture-time miss
# returns None, which safely degrades to the num_kv_splits-gated legacy behavior
# (correctness is unaffected: split-K is device-adaptive per tile). This mirrors
# the opus-workspace warmup-then-capture idiom in aiter/tuned_gemm.py.
_ACTUAL_MAX_SPLITS_CACHE: collections.OrderedDict = collections.OrderedDict()
_ACTUAL_MAX_SPLITS_CACHE_CAP = 512


def _resolve_actual_max_splits(reduce_indptr: torch.Tensor) -> int | None:
    """Capture-safe ``actual_max_splits``: derive+cache eager, reuse on capture.

    Returns the true per-tile max split width for ``reduce_indptr``. On the
    eager/warmup pass it reads the CSR (host sync) and caches by buffer identity;
    under CUDA-graph capture it returns the cached warmup value (or ``None`` on a
    miss, degrading to the ``num_kv_splits`` gate). Never syncs during capture.
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
    """FlyDSL drop-in replacement for ``aiter.mla_reduce_v1``.

    Matches the HIP kernel's signature and write-in-place contract: results land
    in ``final_output`` (and ``final_lse`` if provided). No return value.

    Args:
        partial_output / partial_lse: fp32 per-split partials.
        reduce_indptr / reduce_partial_map: CSR over splits (+ gather map).
        reduce_final_map: optional [tile, 2] {q_start, q_end}; when ``None`` the
            q-range is derived from the partial map (uniform qo-len), mirroring
            the HIP ``use_reduce_final_map = false`` path.
        max_seqlen_q: number of q-positions per decode token group; drives grid-y
            (NTG), so multi-token decode (decode_qlen > 1) is handled correctly.
        final_output: output tensor (bf16/fp16); strides are read at runtime.
        final_lse: optional merged LSE output (fp32).
        num_kv_splits: HIP-signature parity AND, when ``AITER_MLA_REDUCE_HOST_TIER=1``,
            the host tier hint (guarded by :func:`_safe_host_tier`; see
            :func:`_host_tier_dispatch_enabled`). Default (flag off): unused -- the
            split count is read per-tile from the CSR map, grid from num_cu.
        actual_max_splits: optional explicit override of the true
            ``max_t(n_splits)`` over active tiles. Normally left ``None``: the
            wrapper auto-resolves it from ``reduce_indptr`` via
            :func:`_resolve_actual_max_splits` (capture-safe warmup cache), so
            callers (incl. ``mla_decode_fwd``) need not thread it. When set, the
            explicit value is used as-is (tests / advanced callers). It gates
            device-adaptive split-K engagement + the low-direct-pmap variant
            instead of the loose ``num_kv_splits`` budget.
        stream: launch stream; defaults to the current stream of the device.
    """
    if reduce_indptr.numel() < 2:
        return

    # Auto-resolve the split-K gate input when not explicitly overridden. Uses a
    # warmup-populated cache so it stays correct AND capture-safe (no device sync
    # under CUDA-graph capture); see _resolve_actual_max_splits.
    if actual_max_splits is None:
        actual_max_splits = _resolve_actual_max_splits(reduce_indptr)

    H = partial_output.size(-2)
    Dv = final_output.size(-1)
    out_dtype_str = _out_dtype_str(final_output.dtype)
    num_reduce_tile = reduce_indptr.numel() - 1
    output_lse = final_lse is not None
    use_reduce_final_map = reduce_final_map is not None

    num_cu = torch.cuda.get_device_properties(
        final_output.device.index
    ).multi_processor_count

    use_persistent = should_use_persistent_launch(
        H=H,
        max_seqlen_q=max_seqlen_q,
        num_reduce_tile=num_reduce_tile,
        num_cu=num_cu,
    )

    # -------- Device-adaptive, capture-safe split-K (default-able) ----------
    # For low-active-tile / high-split decode the persistent grid is mostly idle
    # (few (tile,head) slots vs num_cu); split each active tile's reduction across
    # K cooperating partial blocks, then combine. The plan is HOST-only (no device
    # read/sync) so it is safe under CUDA-graph capture and on by default; the
    # per-tile K allocation is device-adaptive so a single capture stays correct
    # as the per-tile split counts vary across replays. Needs reduce_final_map
    # (the combine reads each tile's q-range).
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
            if final_lse is None:
                final_lse = torch.empty(
                    1, dtype=torch.float32, device=final_output.device
                )
            if stream is None:
                stream = torch.cuda.current_stream(final_output.device)
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

    # Capture-safe tier selection. Default: Tier.ALL (device-side per-tile
    # branch), correct for any data. Opt-in: host per-tier dispatch, but GUARDED
    # (_safe_host_tier) because num_kv_splits (= max_split_per_batch) is not a
    # per-tile upper bound -- see _host_tier_dispatch_enabled / _safe_host_tier.
    if _host_tier_dispatch_enabled() and num_kv_splits > 0:
        tier = _safe_host_tier(int(num_kv_splits))
    else:
        tier = Tier.ALL

    # Adaptive launch: when the decode grid is sparse (active tiles <<
    # num_reduce_tile) launch one block per active (tile, head, q-group)
    # instead of the persistent grid-stride kernel. num_final_rows is the
    # host-known active-tile count (decode batch, CSR prefix).
    #
    # Gate: multi-tile batches only (num_final_rows > 1). bs=1 / single-tile
    # decode is split-K's domain (b1_s128); adaptive regresses there (+1µs).
    num_final_rows = int(final_output.size(0))
    use_adaptive = (
        _adaptive_launch_enabled()
        and use_reduce_final_map
        and num_final_rows > 1
        and num_final_rows < num_reduce_tile
        and num_final_rows * H * max_seqlen_q <= 4 * num_cu
    )

    low_direct_pmap_thr = 0
    if (
        use_adaptive
        and actual_max_splits is not None
        and int(actual_max_splits) <= _LOW_DIRECT_PMAP_THR
    ):
        low_direct_pmap_thr = _LOW_DIRECT_PMAP_THR

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

    if final_lse is None:
        final_lse = torch.empty(1, dtype=torch.float32, device=final_output.device)
    if reduce_final_map is None:
        reduce_final_map = torch.empty(1, dtype=torch.int32, device=final_output.device)

    if stream is None:
        stream = torch.cuda.current_stream(final_output.device)

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

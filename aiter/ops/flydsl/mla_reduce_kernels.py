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

import os

import torch

from .kernels.mla_reduce import (
    LDS_MAX_SPLITS,
    Tier,
    compile_mla_reduce,
    select_tier,
    should_use_persistent_launch,
    waves_per_eu_from_env,
)

__all__ = [
    "flydsl_mla_reduce_v1",
]


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
        stream: launch stream; defaults to the current stream of the device.
    """
    if reduce_indptr.numel() < 2:
        return

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

    # Capture-safe tier selection. Default: Tier.ALL (device-side per-tile
    # branch), correct for any data. Opt-in: host per-tier dispatch, but GUARDED
    # (_safe_host_tier) because num_kv_splits (= max_split_per_batch) is not a
    # per-tile upper bound -- see _host_tier_dispatch_enabled / _safe_host_tier.
    if _host_tier_dispatch_enabled() and num_kv_splits > 0:
        tier = _safe_host_tier(int(num_kv_splits))
    else:
        tier = Tier.ALL

    kernel = compile_mla_reduce(
        H=H,
        Dv=Dv,
        out_dtype=out_dtype_str,
        tier=tier,
        persistent=use_persistent,
        output_lse=output_lse,
        use_reduce_final_map=use_reduce_final_map,
        waves_per_eu=waves_per_eu_from_env(),
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

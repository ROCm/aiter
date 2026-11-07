# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Input contract of plan.prepare/run (small and large sp_qkv plans): the properties the kernels ASSUME and cannot check on the device.

The kernels hardwire K = 6144 and a compile-time N (9216 dense / 9856 indexer), read A with a per-launch row stride, read W as an exact (16,16)-preshuffled [N, 6144]
(no K stride argument), read the scales as flat pointers and, for the small plans, read positions[t] / slot_mapping[t] for every t < T = 4*M. A violation is a silent wrong
result or an out-of-bounds read, so it is rejected here with ContractError BEFORE any binding or launch. The profile row match already rejects a different M, N or K; these
checks protect against a caller that hands different TENSORS to a matched plan (a K-padded or unshuffled W, a padded x, short token arrays, non-contiguous scales).

Cost: pure Python attribute compares. They run in prepare() (once, outside capture), in run() while a graph is being captured (once, at capture) and in eager run()
(every call); graph replays execute no Python, so replays pay nothing.

What is deliberately ALLOWED (aiter native gemm_a8w8_bpreshuffle accepts it, or ATOM produces it):
  * x as a row-strided view (stride(0) >= K, stride(1) == 1): strideA is a per-launch argument;
  * w with MORE rows than N (ATOM pads N to 128: the first N rows are layout-compatible, the dense build already reads a prefix of the indexer weight);
  * w_scale as [N] or [N,1] (and longer than N for the padded case), x_scale as [M] or [M,1];
  * slot_mapping entries of -1 (the kernel skips cache writes for slot < 0): only the LENGTH of positions/slot_mapping is checked.
"""

from __future__ import annotations

from aiter.gemm_fusion.match import ContractError

K_HIDDEN = 6144


def _fail(owner: str, msg: str):
    raise ContractError(f"{owner}: {msg}")


def _is(t, torch_mod, name: str) -> bool:
    want = getattr(torch_mod, name, None)
    return want is not None and t.dtype == want


def check_activation(owner: str, x, x_scale, *, M: int, K: int = K_HIDDEN) -> None:
    """x: fp8 e4m3fn [M, K], K contiguous (a row-strided view is fine); x_scale: fp32 with exactly M elements, contiguous ([M] or [M,1])."""
    import torch

    if x.dim() != 2:
        _fail(owner, f"x must be 2-D [{M}, {K}], got shape {tuple(x.shape)}")
    if x.shape[0] != M:
        _fail(owner, f"x has {x.shape[0]} rows, the plan's local M is {M}")
    if x.shape[1] != K:
        _fail(
            owner,
            f"x K={x.shape[1]} != {K}: the kernels hardwire K={K} (a K-padded activation is not supported; pass x[:, :{K}] as a row-strided view)",
        )
    if x.stride(1) != 1 or x.stride(0) < K:
        _fail(
            owner,
            f"x must be K-contiguous with row stride >= {K}, got strides ({x.stride(0)}, {x.stride(1)})",
        )
    if not _is(x, torch, "float8_e4m3fn"):
        _fail(
            owner,
            f"x dtype {x.dtype} is not float8_e4m3fn (the input must already be quantized per token)",
        )
    if not _is(x_scale, torch, "float32"):
        _fail(owner, f"x_scale dtype {x_scale.dtype} is not float32")
    if x_scale.numel() != M or not x_scale.is_contiguous():
        _fail(
            owner,
            f"x_scale must be contiguous with exactly {M} elements, got shape {tuple(x_scale.shape)} (contiguous={x_scale.is_contiguous()})",
        )


def check_weight_scale(owner: str, w_scale, *, N: int) -> None:
    """w_scale: fp32, contiguous, [N] or [N,1] (at least N rows: ATOM's N padding appends ones)."""
    import torch

    if not _is(w_scale, torch, "float32"):
        _fail(owner, f"w_scale dtype {w_scale.dtype} is not float32")
    shp = tuple(w_scale.shape)
    rows_ok = (len(shp) == 1 and shp[0] >= N) or (
        len(shp) == 2 and shp[1] == 1 and shp[0] >= N
    )
    if not rows_ok or not w_scale.is_contiguous():
        _fail(
            owner,
            f"w_scale must be contiguous [N] or [N,1] with N >= {N}, got shape {shp} (contiguous={w_scale.is_contiguous()})",
        )


def check_weight(owner: str, w, *, N: int, K: int = K_HIDDEN) -> None:
    """w: fp8 e4m3fn, (16,16)-preshuffled, contiguous [rows >= N, K == 6144]. A K-padded weight (aiter shuffle_weight(pad_k_to=...)) or an unshuffled one would be
    read with the wrong addressing: silently wrong, so it is rejected."""
    import torch

    if w.dim() != 2:
        _fail(owner, f"w must be 2-D [N, {K}], got shape {tuple(w.shape)}")
    if w.shape[1] != K:
        _fail(
            owner,
            f"w K={w.shape[1]} != {K}: a K-padded weight (shuffle_weight pad_k_to) is not supported, the kernels address the preshuffled weight with K={K}",
        )
    padded_k, orig_k = getattr(w, "aiter_padded_k", None), getattr(
        w, "aiter_original_k", None
    )
    if (padded_k is not None and padded_k != K) or (orig_k is not None and orig_k != K):
        _fail(
            owner,
            f"w was shuffled with K padding (aiter_original_k={orig_k}, aiter_padded_k={padded_k}); the kernels require K={K} exactly",
        )
    if getattr(w, "is_shuffled", None) is False:
        _fail(
            owner,
            "w is not (16,16)-preshuffled (is_shuffled=False): the kernels read the preshuffled layout",
        )
    if w.shape[0] < N:
        _fail(owner, f"w has {w.shape[0]} rows < N={N}")
    if not _is(w, torch, "float8_e4m3fn"):
        _fail(owner, f"w dtype {w.dtype} is not float8_e4m3fn")
    if not w.is_contiguous():
        _fail(
            owner,
            "w must be contiguous (the preshuffled layout has no stride argument)",
        )


def check_call_id(owner: str, call_id) -> int:
    """R12: the 31-bit decision key that tags every handshake of this launch. Must be an int in [1, 2^31): 0 is refused (a zero tag equals the content of a zeroed window)."""
    if (
        isinstance(call_id, bool)
        or not isinstance(call_id, int)
        or not 1 <= call_id < (1 << 31)
    ):
        _fail(
            owner,
            f"call_id must be an int in [1, 2^31) (the 31-bit decision key of site/tokens/vid/n_fused), got {call_id!r}",
        )
    return call_id


def check_token_arrays_v3(owner: str, *, T: int, positions, slot_mapping, outs) -> int:
    """R12 small plans: positions / slot_mapping are PER-LAUNCH kernargs now and the kernel pads: it reads only t < t_valid and treats rows t >= t_valid as (position 0,
    slot -1). So a SHORT array is legal (the pad rows run as ordinary rows whose cache writes are skipped); what is required is int64, contiguous and at least one
    element. t_valid = min(numel(positions), numel(slot_mapping), T). The outputs still need T rows (the kernel writes q/k/v/index_q for every row).
    """
    import torch

    for name, t in (("positions", positions), ("slot_mapping", slot_mapping)):
        if int(t.numel()) < 1:
            _fail(owner, f"{name} is empty")
        if not _is(t, torch, "int64"):
            _fail(
                owner, f"{name} dtype {t.dtype} is not int64 (the kernel reads int64)"
            )
        if not t.is_contiguous():
            _fail(owner, f"{name} must be contiguous")
    for name, t in (outs or {}).items():
        if name in ("q", "k", "v", "index_q") and t.shape[0] < T:
            _fail(
                owner,
                f"output {name!r} has {t.shape[0]} rows but the kernel writes {T}",
            )
    return min(int(positions.numel()), int(slot_mapping.numel()), int(T))


def check_token_arrays(owner: str, *, T: int, positions, slot_mapping, outs) -> None:
    """Small plans: positions[t] / slot_mapping[t] (int64) are read for every t < T = 4*M, and q/k/v/index_q are written for every t < T. Padded rows are supported: the
    caller stages T-long arrays whose pad rows are (position 0, slot_mapping -1); the kernel runs norm/rope on them like any row (in-bounds, discarded by the caller) and skips
    every cache write for slot < 0 (KV and index cache). A SHORTER array means the caller did NOT stage the pad rows (positions carries only the real token count while x has
    the padded rows): an out-of-bounds read/write, rejected. A longer persistent buffer is safe. Only lengths are checked, never values.
    """
    import torch

    for name, t in (("positions", positions), ("slot_mapping", slot_mapping)):
        n = int(t.numel())
        if n < T:
            _fail(
                owner,
                f"{name} has {n} entries but the kernel reads T = 4 x local M = {T}: stage a T-long array whose pad rows are position 0 / slot_mapping -1 "
                f"(the kernel skips cache writes for slot < 0); a shorter array is an out-of-bounds read",
            )
        if not _is(t, torch, "int64"):
            _fail(
                owner, f"{name} dtype {t.dtype} is not int64 (the kernel reads int64)"
            )
    for name, t in (outs or {}).items():
        if name in ("q", "k", "v", "index_q") and t.shape[0] < T:
            _fail(
                owner,
                f"output {name!r} has {t.shape[0]} rows but the kernel writes {T}",
            )

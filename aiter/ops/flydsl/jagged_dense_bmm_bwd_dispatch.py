# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host wrapper for the FlyDSL ``jagged_dense_bmm`` BACKWARD kernels.

Companion to the forward's ``jagged_dense_bmm_dispatch_v2.py``. It folds all the
host-side glue the backward needs (previously inlined in every bench / example /
recsys-harness driver) into one place:

    - ``configure_dim(D)`` + a **single-D-per-process guard** (the backward
      snapshots the square dense dim D = K = N as a compile-time constant on the
      first launch and rejects later drift);
    - the ``GJ_STAGES_A`` / ``COARSEN_M`` **schedule-knob snapshot guard** (both
      are compile-time module globals snapshotted on the first grad_jagged
      launch, and the FlyDSL artifact cache key does NOT include them, so a
      second, different value in the same process would silently reuse a stale
      compiled kernel — we reject it loudly instead);
    - fp32 split-reduction scratch allocation (cached per (n_groups, D, device));
    - the ``BLOCK_M``-padded ``dJagged`` output (so a partial tail-tile store
      stays in-bounds) and the ``[:L]`` view returned to the caller;
    - the dense reshape to the backward's **plain ``(n_groups*K, N)`` K-major**
      layout (the documented backward contract; the forward instead consumes a
      tall pre-transposed ``(n_groups*N, K)`` buffer);
    - ``mark_layout_dynamic`` on the dlpack views (mirrors the validated bench);
    - stream defaulting;
    - calling ``grad_jagged`` then the fused ``grad_dense_bias``.

Backward-only, D fixed per process. Per-shape scheduling is resolved from a JSON
dispatch table (``jagged_dense_bmm_bwd_dispatch.json``, arch-keyed-v1, same loader
shape as the forward's ``jagged_dense_bmm_dispatch_v2``): the tunables are
``split`` (None → the kernel's ``2 if D<=256 else 1`` rule), ``coarsen_m`` (None →
module default 2), and ``gj_stages_a`` (None → heuristic ``1 if D<=256 else 2`` —
the per-D winner from ``2026-07-03_mvonstra-amd_bench.md``). Resolution order is
explicit kwarg > exact shape-id winner > D-bucketed heuristic. There is no MFMA
"atom" knob: the experimental fp16 32×32×8 path was removed, so bf16 16×16×16 is
the only path.

Usage::

    from aiter.ops.flydsl import jagged_dense_bmm_bwd_dispatched
    d_jagged, d_dense, d_bias = jagged_dense_bmm_bwd_dispatched(
        jagged, dense, d_out, seq_offsets,
        n_groups=B, max_seq_len=Mi, stream=st,
    )
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch

from .kernels import jagged_dense_bmm_bwd as _bwd

__all__ = [
    "jagged_dense_bmm_bwd_dispatched",
    "resolve_config",
    "shape_id",
]

# --------------------------------------------------------------------------- #
# Per-shape JSON dispatch (mirrors jagged_dense_bmm_dispatch_v2's loader shape). #
# --------------------------------------------------------------------------- #

# Config schema. Every key is optional; None means "use the kernel/heuristic
# default", so a winner that omits a key doesn't override it. No "atom" key: the
# fp16 path is gone, bf16 16x16x16 is the only schedule.
_SCHEMA_DEFAULTS = {
    "split": None,         # None -> D-derived (2 if D<=256 else 1)
    "coarsen_m": None,     # None -> module default (2)
    "gj_stages_a": None,   # None -> heuristic (1 if D<=256 else 2)
}

_DISPATCH_TABLE: Optional[dict] = None


def _dispatch_json_paths() -> tuple[Path, ...]:
    env = os.environ.get("FLYDSL_JDBBA_BWD_DISPATCH_JSON")
    if env:
        return (Path(env),)
    return (Path(__file__).resolve().parent / "jagged_dense_bmm_bwd_dispatch.json",)


def _detect_arch() -> Optional[str]:
    """Detected ROCm arch (e.g. ``gfx942``), or None. Shares the forward's arch
    override env so a test can target a non-native table."""
    env = os.environ.get("FLYDSL_JAGGED_DENSE_BMM_ARCH")
    if env:
        return env
    try:
        from flydsl.runtime.device import get_rocm_arch

        return get_rocm_arch()
    except Exception:
        return None


def _select_arch_section(data: dict, arch: Optional[str]) -> dict:
    """Pick the per-arch sub-table from an ``arch-keyed-v1`` JSON; a flat (legacy)
    JSON is returned as-is (matches the forward loader)."""
    by_arch = data.get("by_arch")
    if not isinstance(by_arch, dict):
        return data
    if arch and arch in by_arch:
        return by_arch[arch]
    if arch:
        for sect in by_arch.values():
            if sect.get("gfx") == arch:
                return sect
    if len(by_arch) == 1:
        return next(iter(by_arch.values()))
    return {}


def _load_dispatch_table() -> dict:
    global _DISPATCH_TABLE
    if _DISPATCH_TABLE is not None:
        return _DISPATCH_TABLE
    arch = _detect_arch()
    for path in _dispatch_json_paths():
        if path.is_file():
            data = json.loads(path.read_text())
            section = _select_arch_section(data, arch)
            _DISPATCH_TABLE = {
                "gfx": section.get("gfx"),
                "winners": dict(section.get("winners") or {}),
                "fallback": dict(section.get("fallback") or {}),
            }
            return _DISPATCH_TABLE
    _DISPATCH_TABLE = {"gfx": None, "winners": {}, "fallback": {}}
    return _DISPATCH_TABLE


def shape_id(*, n_groups: int, reduction_k: int, output_n: int, max_seq_len: int) -> str:
    """Per-shape key, same format as the forward. For the backward K == N == D, so
    ``reduction_k == output_n == D``."""
    return f"B{n_groups}D{reduction_k}K{output_n}N{max_seq_len}"


def _opt_int(v):
    return None if v is None else int(v)


def _normalize_cfg(cfg: dict) -> dict:
    """Coerce a raw (JSON or kwarg) config to the full schema; unknown keys dropped,
    missing keys stay None so kernel/heuristic defaults still apply."""
    return {k: _opt_int(cfg.get(k, default)) for k, default in _SCHEMA_DEFAULTS.items()}


def _heuristic_cfg(*, reduction_k: int) -> dict:
    """D-bucketed fallback. Reads the JSON ``fallback`` if present, else the
    built-in per-D default: gj_stages_a = 1 (D<=256) else 2; split/coarsen_m left
    None (kernel D-derived / module default)."""
    rules = _load_dispatch_table().get("fallback") or {}
    base = dict(rules.get("global") or {})
    by_bucket = rules.get("by_d_bucket") or {}
    bucket = "d_le_256" if reduction_k <= 256 else "d_gt_256"
    if isinstance(by_bucket.get(bucket), dict):
        base.update(by_bucket[bucket].get("config") or {})
    cfg = _normalize_cfg(base)
    if cfg["gj_stages_a"] is None:
        cfg["gj_stages_a"] = 1 if reduction_k <= 256 else 2
    return cfg


def resolve_config(
    *,
    n_groups: int,
    reduction_k: int,
    output_n: int,
    max_seq_len: int,
    split: Optional[int] = None,
    coarsen_m: Optional[int] = None,
    gj_stages_a: Optional[int] = None,
) -> dict:
    """Full schedule config for a shape: explicit kwarg > table winner > heuristic.

    Returns a normalized dict with keys split / coarsen_m / gj_stages_a (each an
    int or None, where None means "let the kernel/module default stand").
    """
    explicit = {"split": split, "coarsen_m": coarsen_m, "gj_stages_a": gj_stages_a}
    table = _load_dispatch_table()
    sid = shape_id(n_groups=n_groups, reduction_k=reduction_k, output_n=output_n, max_seq_len=max_seq_len)
    entry = table["winners"].get(sid)
    cfg = _normalize_cfg(entry) if entry is not None else _heuristic_cfg(reduction_k=reduction_k)
    for k, v in explicit.items():
        if v is not None:
            cfg[k] = int(v)
    return cfg

# Process-wide first-launch snapshot state. D and the schedule knobs (SPLIT,
# GJ_STAGES_A, COARSEN_M) are compile-time constants the kernels snapshot on their
# first launch, so the whole process is pinned to the first values seen here.
_CONFIGURED_D: Optional[int] = None
_CONFIGURED_KNOBS: Optional[tuple[int, int, int]] = None  # (SPLIT, GJ_STAGES_A, COARSEN_M)

# fp32 split-reduction scratch, cached per (n_groups, D, device). The partials
# passes fully overwrite every slot they later reduce (empty groups included), so
# reuse across calls is safe without re-zeroing.
_SCRATCH_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _configure_once(D: int, cfg: dict) -> None:
    """Set D + the resolved schedule knobs before the first launch and guard drift.

    ``cfg`` carries split / coarsen_m / gj_stages_a (each int or None → keep the
    kernel/module default). On the first call this rebinds the compile-time
    constants; on every later call it only *validates* that D and the knobs match
    the pinned values, raising loudly (never silently mutating) otherwise.
    """
    global _CONFIGURED_D, _CONFIGURED_KNOBS
    D = int(D)

    if _CONFIGURED_D is None:
        # configure_dim first (it sets the D-derived SPLIT), then apply overrides.
        _bwd.configure_dim(D)
        if cfg.get("split") is not None:
            _bwd.SPLIT = int(cfg["split"])
        if cfg.get("gj_stages_a") is not None:
            _bwd.GJ_STAGES_A = int(cfg["gj_stages_a"])
        if cfg.get("coarsen_m") is not None:
            _bwd.COARSEN_M = int(cfg["coarsen_m"])
        _CONFIGURED_D = D
        _CONFIGURED_KNOBS = (_bwd.SPLIT, _bwd.GJ_STAGES_A, _bwd.COARSEN_M)
        return

    # Validate-only: intended knob values default to the pinned ones when a cfg
    # entry is None, so an unspecified knob always matches.
    want = (
        int(cfg["split"]) if cfg.get("split") is not None else _bwd.SPLIT,
        int(cfg["gj_stages_a"]) if cfg.get("gj_stages_a") is not None else _bwd.GJ_STAGES_A,
        int(cfg["coarsen_m"]) if cfg.get("coarsen_m") is not None else _bwd.COARSEN_M,
    )
    if _CONFIGURED_D != D:
        raise RuntimeError(
            f"jagged_dense_bmm_bwd_dispatched: this process is pinned to D={_CONFIGURED_D} "
            f"(K = N = the square dense dim), but was called with D={D}. The backward "
            "snapshots D as a compile-time constant on its first launch and cannot serve a "
            "second D in the same process. Run one D per process (e.g. a fresh subprocess "
            "per D). Removing this constraint is Phase 4 of the aiter integration plan "
            "(runtime-derived D, like the forward gen kernel)."
        )
    if _CONFIGURED_KNOBS != want:
        raise RuntimeError(
            f"jagged_dense_bmm_bwd_dispatched: schedule knobs are pinned to "
            f"(SPLIT, GJ_STAGES_A, COARSEN_M)={_CONFIGURED_KNOBS} for this process but were "
            f"changed to {want}. These are compile-time globals snapshotted on the first launch "
            "AND the FlyDSL artifact cache key ignores GJ_STAGES_A/COARSEN_M, so changing them "
            "in-process would silently reuse the stale compiled kernel. Pick one config per "
            "process (clear ~/.flydsl/cache and use a fresh process to switch)."
        )


def _get_scratch(n_groups: int, K: int, N: int, split: int, device: torch.device):
    """Cached fp32 (dense_partials, bias_partials) for the split reduction.

    SPLIT >= 2 (e.g. D=256): real (n_groups*SPLIT*K, N) / (n_groups*SPLIT, N)
    scratch the reduce passes sum. SPLIT == 1 (e.g. D=512): the partials pass
    writes dDense/dBias directly and never reads this scratch, so we pass a tiny
    correctly-typed placeholder instead of ~1 GB of unused fp32 (the grad_dense_bias
    launcher still needs valid tensor args for the const-false SPLIT>=2 branch).
    """
    key = (n_groups, K, N, split, device.type, device.index)
    cached = _SCRATCH_CACHE.get(key)
    if cached is None:
        if split >= 2:
            dense_partials = torch.empty(n_groups * split * K, N, dtype=torch.float32, device=device)
            bias_partials = torch.empty(n_groups * split, N, dtype=torch.float32, device=device)
        else:
            dense_partials = torch.empty(1, N, dtype=torch.float32, device=device)
            bias_partials = torch.empty(1, N, dtype=torch.float32, device=device)
        _SCRATCH_CACHE[key] = (dense_partials, bias_partials)
        cached = _SCRATCH_CACHE[key]
    return cached


def jagged_dense_bmm_bwd_dispatched(
    jagged: torch.Tensor,        # (L, K)          bf16, packed rows
    dense: torch.Tensor,         # (n_groups, K, N) bf16
    d_out: torch.Tensor,         # (L, N)          bf16, upstream gradient
    seq_offsets: torch.Tensor,   # (n_groups + 1,) int32, prefix-sum row offsets
    n_groups: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    stream=None,
    *,
    split: Optional[int] = None,
    gj_stages_a: Optional[int] = None,
    coarsen_m: Optional[int] = None,
):
    """Dispatched backward: returns ``(d_jagged, d_dense, d_bias)``.

    Given the forward ``Out[s:e] = Jagged[s:e] @ Dense[b] + Bias[b]`` per group b,
    computes for the upstream gradient ``d_out``:

        d_jagged[s:e] = d_out[s:e] @ Dense[b].T   (L, K)
        d_dense[b]    = Jagged[s:e].T @ d_out[s:e] (n_groups, K, N)
        d_bias[b]     = sum_m d_out[s:e]           (n_groups, N)

    The square dense dim ``D = K = N`` is a compile-time constant pinned per
    process (see the single-D guard). ``max_seq_len`` sizes the grad_jagged M
    grid envelope; if omitted it is derived from ``seq_offsets`` (costs one
    device→host sync — pass it to avoid that). The schedule knobs
    ``split`` / ``gj_stages_a`` / ``coarsen_m`` are normally resolved per-shape
    from the JSON dispatch table; passing any of them here forces that value
    (must be constant per process).
    """
    if n_groups is None:
        n_groups = dense.shape[0]
    n_groups = int(n_groups)

    K = dense.shape[1]
    N = dense.shape[2]
    if K != N:
        raise ValueError(
            f"backward requires a square dense dim (K == N == D); got dense (K={K}, N={N})."
        )
    D = int(K)

    if dense.shape[0] != n_groups:
        raise ValueError(f"dense.shape[0]={dense.shape[0]} != n_groups={n_groups}.")
    if seq_offsets.numel() != n_groups + 1:
        raise ValueError(
            f"seq_offsets must have n_groups+1={n_groups + 1} entries, got {seq_offsets.numel()}."
        )
    if seq_offsets.dtype != torch.int32:
        raise ValueError(f"seq_offsets must be int32, got {seq_offsets.dtype}.")

    if max_seq_len is None:
        # One device→host sync: envelope = longest group. Pass max_seq_len to skip.
        max_seq_len = int((seq_offsets[1:] - seq_offsets[:-1]).max().item())
    max_seq_len = int(max_seq_len)

    # Resolve per-shape schedule (explicit kwarg > JSON winner > D-bucketed
    # heuristic), then pin it for the process. For the backward K == N == D.
    cfg = resolve_config(
        n_groups=n_groups, reduction_k=D, output_n=D, max_seq_len=max_seq_len,
        split=split, gj_stages_a=gj_stages_a, coarsen_m=coarsen_m,
    )
    _configure_once(D, cfg)

    # Read the (now-pinned) tiling constants back from the kernel module.
    BLOCK_M = _bwd.BLOCK_M
    SPLIT = _bwd.SPLIT

    import flydsl.compiler as flyc

    device = jagged.device
    total_rows = jagged.shape[0]
    if stream is None:
        stream = torch.cuda.current_stream()

    tDOut = flyc.from_dlpack(d_out).mark_layout_dynamic(leading_dim=1, divisibility=8)

    # Grad outputs are allocated fresh per call (mirrors how torch.autograd invokes
    # a backward — it allocates the returned grads each step rather than reusing
    # buffers). They use torch.empty, NOT torch.zeros: the kernels fully overwrite
    # every returned element (grad_jagged writes all L packed rows over all K cols;
    # grad_dense_bias writes every (b,k,n)/(b,n), storing an explicit 0 for empty
    # groups), so a zero-init memset (multi-GB at the North-Star shape) would be
    # pure overhead. The d_jagged padding rows [L:L+BLOCK_M] are never returned.

    # --- dJagged: RHS is Dense[b] in its plain (K, N) layout, flattened tall. ---
    # reshape is a view; .contiguous() is a no-op for the usual contiguous (B,K,N)
    # weight (no copy), so this adds no real per-call cost.
    dense_kn = dense.reshape(n_groups * K, N).contiguous()
    d_jagged = torch.empty(total_rows + BLOCK_M, K, dtype=torch.bfloat16, device=device)
    tDJ = flyc.from_dlpack(d_jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
    _bwd.grad_jagged(tDJ, tDOut, dense_kn, seq_offsets, n_groups, max_seq_len, stream=stream)

    # --- dDense (+ fused dBias): split-reduction over the sequence axis m. ---
    d_dense = torch.empty(n_groups, K, N, dtype=torch.bfloat16, device=device)
    d_bias = torch.empty(n_groups, N, dtype=torch.bfloat16, device=device)
    dense_partials, bias_partials = _get_scratch(n_groups, K, N, SPLIT, device)
    tJagged = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
    _bwd.grad_dense_bias(
        d_dense.view(n_groups * K, N), d_bias, tJagged, tDOut, seq_offsets,
        dense_partials, bias_partials, n_groups, max_seq_len, stream=stream,
    )

    return d_jagged[:total_rows], d_dense, d_bias

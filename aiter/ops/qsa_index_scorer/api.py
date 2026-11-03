# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public op: QSA index scorer (paged MQA logits, compressed page 4) -- FlyDSL gfx1250.

Drop-in for ATOM's ``qsa_paged_mqa_logits`` (same 12 leading parameters, same
names/order/defaults, see :data:`ATOM_SCORER_SIGNATURE`) for the call classes
listed in a profile. **Off unless explicitly enabled**: a profile must be named
(``profile=`` or ``$AITER_QSA_INDEX_SCORER_PROFILE``); with neither, every call
is ``Reason.DISABLED`` and neither FlyDSL nor the kernel module is imported.

Caller protocol (native fallback is always the caller's own scorer)::

    ok, reason = eligibility(q, cache, table, t2r, pos, ctx, 4, None, cols,
                             compressed_k_scale=scale, profile=DEFAULT_PROFILE)
    if ok:
        logits, visible = qsa_paged_mqa_logits(..., profile=DEFAULT_PROFILE)
    else:
        logits, visible = atom_native_scorer(...)   # log `reason` once per (site, reason)

or ``out, decision = try_qsa_paged_mqa_logits(...)`` (``out is None`` => native).
:func:`qsa_paged_mqa_logits` itself raises :class:`QSAIndexScorerUnsupported`
(``.reason``) for any uncovered call: never a silent wrong answer.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import math
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any

import torch

from . import profiles as P
from .profiles import Reason

__all__ = [
    "ATOM_SCORER_SIGNATURE", "Decision", "QSAIndexScorerUnsupported", "decide", "eligibility",
    "layout_eligibility", "predict_route", "qsa_paged_mqa_logits", "try_qsa_paged_mqa_logits",
    "selected_profile",
]

#: ATOM ``atom/model_ops/qwen4_exp/ops/qsa.py::qsa_paged_mqa_logits`` (frozen
#: sha256 aaa78f43...): (name, default) of every parameter, in order. Our op
#: takes exactly these positionally/by keyword; extras are keyword-only.
ATOM_SCORER_SIGNATURE: tuple[tuple[str, Any], ...] = (
    ("q", inspect.Parameter.empty),
    ("compressed_k_cache", inspect.Parameter.empty),
    ("page_table", inspect.Parameter.empty),
    ("token_to_request", inspect.Parameter.empty),
    ("query_positions", inspect.Parameter.empty),
    ("context_lens", inspect.Parameter.empty),
    ("compress_ratio", 4),
    ("score_divisor", None),
    ("max_columns", None),
    ("row_starts", None),
    ("allow_flydsl", False),
    ("compressed_k_scale", None),
)
ATOM_SCORER_SOURCE_SHA256 = "aaa78f43255cf950bfbdf5d618246f9a7a671f0262cff2398b2ca4ea703810ee"

_DEFAULT_DIVISOR = math.sqrt(P.HEAD_DIM)
#: Device types a call may live on (CPU tests widen this; production: GPU only).
_ALLOWED_DEVICE_TYPES = ("cuda",)
_FP8 = getattr(torch, "float8_e4m3fn", None)

_LOCK = threading.Lock()
_KERNEL = None                 # lazily imported kernel module
_KERNEL_ENV: dict | None = None  # effective QSA_* switches when the kernel was imported
_KERNEL_SHA: dict = {}         # path -> sha256
_DEVICE_INFO: dict = {}        # device index -> (arch, cu)


class QSAIndexScorerUnsupported(RuntimeError):
    """The call is not covered by the selected profile; run the native scorer."""

    def __init__(self, decision: "Decision"):
        super().__init__(f"qsa_index_scorer: {decision.reason.value}: {decision.detail}")
        self.decision = decision
        self.reason = decision.reason


@dataclass(frozen=True)
class Decision:
    ok: bool
    reason: Reason
    detail: str = ""
    profile: str | None = None
    cell: str | None = None
    vid: str | None = None
    columns: int | None = None

    def __iter__(self):            # ``ok, reason = decision``
        return iter((self.ok, self.reason.value))

    @property
    def static(self) -> bool:
        """True when the reason depends only on persistent state (prepare-time)."""
        return self.reason in P.STATIC_REASONS


def _no(reason: Reason, detail: str = "", **kw) -> Decision:
    return Decision(False, reason, detail, **kw)


# ---------------------------------------------------------------------------
# Profile selection / kernel import (both lazy, both cached)
# ---------------------------------------------------------------------------
def selected_profile(profile: str | None = None) -> str | None:
    """Explicit profile argument, else ``$AITER_QSA_INDEX_SCORER_PROFILE``, else None (off)."""
    if profile:
        return profile
    env = os.environ.get(P.PROFILE_ENV, "").strip()
    return env or None


def _load_kernel():
    """Import the kernel module once (package import, else by file path)."""
    global _KERNEL, _KERNEL_ENV
    if _KERNEL is not None:
        return _KERNEL
    with _LOCK:
        if _KERNEL is None:
            env = P.effective_kernel_env()
            if __name__.startswith("aiter."):
                mod = importlib.import_module(P.KERNEL_MODULE)
            else:  # standalone (file-loaded) package: load the sibling kernel file
                spec = importlib.util.spec_from_file_location(
                    "qsa_index_scorer_kernel", P.KERNEL_PATH)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            _KERNEL_ENV, _KERNEL = env, mod
    return _KERNEL


def _kernel_env() -> dict:
    """Switches the kernel was (or will be) built with: import-time snapshot."""
    return dict(_KERNEL_ENV) if _KERNEL_ENV is not None else P.effective_kernel_env()


def _kernel_sha256() -> str:
    path = getattr(_KERNEL, "__file__", None) or P.KERNEL_PATH
    sha = _KERNEL_SHA.get(path)
    if sha is None:
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        _KERNEL_SHA[path] = sha
    return sha


def _device_info(device: torch.device) -> tuple[str, int]:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    info = _DEVICE_INFO.get(idx)
    if info is None:
        props = torch.cuda.get_device_properties(idx)
        arch = str(getattr(props, "gcnArchName", "")).split(":")[0]
        info = (arch, int(props.multi_processor_count))
        _DEVICE_INFO[idx] = info
    return info


# ---------------------------------------------------------------------------
# Route prediction (reads the kernel's own router functions and constants)
# ---------------------------------------------------------------------------
def predict_route(K, cache_dtype: str, rows: int, requests: int, columns: int,
                  kv_ptr: int) -> tuple[str, dict]:
    """(vid, params) the kernel wrapper launches for this call (default-path routing)."""
    is_fp8 = cache_dtype == "fp8"
    if K._DRAFT_PATH:
        d = K.draft_route(rows, requests, is_fp8, kv_ptr)
        if d is not None:
            return "qsa.page4.draft.v1", {"k_load": d[0], "wave_cols": int(d[1])}
    if K._DECODE_GROUP != "0":
        w = K.decode_group_route(rows, requests, is_fp8, kv_ptr)
        if w is not None:
            return "qsa.page4.decg.v1", {"k_load": "tdm", "wave_cols": int(w),
                                         "barrier": False}
    if not is_fp8 and K._BF16_PATH == "wmma":
        rpb = K.wmma_bf16_rows_per_block(rows, requests)
        k_load = K._resolve_bf16_k_load(rpb)
        if k_load == "tdm" and kv_ptr % 128:
            rpb, k_load = 16, "raw"
        if (rpb == 32 and k_load == "tdm" and K.TDM_PERSIST and K._ACCESS == "raw"
                and columns >= K.WMMA_COLS * K.TDM_PERSIST_CHUNKS):
            return "qsa.page4.r32_persist.v1", {"k_load": "tdm", "rows_per_block": 32,
                                                "persist_cols": int(K.TDM_PERSIST_CHUNKS),
                                                "buffers": int(K.TDM_PERSIST_BUFFERS)}
        if rpb == 32:
            return "qsa.page4.r32.v1", {"k_load": k_load, "rows_per_block": 32}
        return f"qsa.page4.r{rpb}.v1", {"k_load": k_load, "rows_per_block": rpb}
    if is_fp8 and K._FP8_PATH == "wmma":
        rpb = K.wmma_rows_per_block(rows, requests)
        k_load = K._resolve_fp8_k_load(rpb)
        if k_load == "tdm" and kv_ptr % 128:
            rpb, k_load = 16, "raw"
        if rpb == 32:
            return "qsa.page4.r32_fp8.v1", {"k_load": k_load, "rows_per_block": 32,
                                            "cvt": K._FP8_CVT}
        return "qsa.page4.r16_fp8.v1", {"k_load": k_load, "rows_per_block": 16}
    return "qsa.page4.scalar.v0", {}


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
def _vec_ok(t, n: int) -> bool:
    return (isinstance(t, torch.Tensor) and t.ndim == 1 and int(t.shape[0]) == n
            and t.dtype in (torch.int32, torch.int64) and t.is_contiguous())


def _cache_format(cache: torch.Tensor, scale) -> str:
    if cache.dtype == torch.bfloat16:
        return "bf16" if scale is None else "bf16+scale"
    if _FP8 is not None and cache.dtype == _FP8:
        if scale is None:
            return "fp8-noscale"
        if scale.dtype == torch.float32 and scale.ndim == 3:
            return "fp8"
        return "mxfp8" if scale.dtype == torch.uint8 else "fp8-badscale"
    if cache.dtype == torch.uint8:
        return "fp4"
    return str(cache.dtype).replace("torch.", "")


def _layout_decision(cache, scale, device, prof: P.Profile) -> Decision | str:
    """Static (prepare-time) cache checks; returns the cache dtype tag on success."""
    if not isinstance(cache, torch.Tensor):
        return _no(Reason.CACHE_SHAPE, "compressed_k_cache is not a tensor")
    if cache.device != device:
        return _no(Reason.DEVICE_MISMATCH, f"compressed_k_cache on {cache.device}, q on {device}")
    fmt = _cache_format(cache, scale)
    if fmt not in P.CACHE_DTYPES:
        return _no(Reason.CACHE_FORMAT, f"index cache format {fmt} (supported: bf16, fp8 E4M3 "
                                        "+ fp32 per-row scale)")
    if cache.ndim != 4 or tuple(cache.shape[1:]) != (P.PAGE_SIZE, 1, P.HEAD_DIM) or cache.shape[0] < 1:
        return _no(Reason.CACHE_SHAPE, f"compressed_k_cache {tuple(cache.shape)} != "
                                       f"[pages>=1, {P.PAGE_SIZE}, 1, {P.HEAD_DIM}]")
    if not cache.is_contiguous():
        return _no(Reason.CACHE_LAYOUT, "compressed_k_cache not contiguous")
    align = int(prof.data["contract"].get("cache_align_bytes", P.CACHE_ALIGN_BYTES))
    if cache.data_ptr() % align:
        return _no(Reason.CACHE_ALIGNMENT, f"compressed_k_cache base {cache.data_ptr():#x} not "
                                           f"{align}-byte aligned (TDM routes need it)")
    if fmt == "fp8":
        if (scale.device != device or tuple(scale.shape) != (int(cache.shape[0]), P.PAGE_SIZE, 1)
                or not scale.is_contiguous() or scale.data_ptr() % 4):
            return _no(Reason.SCALE_LAYOUT, f"compressed_k_scale {tuple(scale.shape)} "
                                            f"{scale.dtype} must be contiguous fp32 [pages, 4, 1]")
    return fmt


def _resolve(profile: str | None) -> P.Profile | Decision:
    name = selected_profile(profile)
    if name is None:
        return _no(Reason.DISABLED, f"no profile selected (pass profile= or set {P.PROFILE_ENV})")
    try:
        return P.load_profile(name)
    except P.ProfileError as exc:
        return _no(exc.reason, exc.detail, profile=name)


def layout_eligibility(compressed_k_cache: torch.Tensor,
                       compressed_k_scale: torch.Tensor | None = None, *,
                       profile: str | None = None) -> Decision:
    """Prepare-time (rank-uniform) checks: profile, kernel build, device, cache layout.

    Evaluate once per bucket/cache allocation; a failure here holds for every
    call on that cache, so the call site can pin the native scorer.
    """
    prof = _resolve(profile)
    if isinstance(prof, Decision):
        return prof
    dev = compressed_k_cache.device
    if dev.type not in _ALLOWED_DEVICE_TYPES:
        return _no(Reason.DEVICE_NOT_GPU, f"cache on {dev}", profile=prof.name)
    fmt = _layout_decision(compressed_k_cache, compressed_k_scale, dev, prof)
    if isinstance(fmt, Decision):
        return Decision(False, fmt.reason, fmt.detail, profile=prof.name)
    static = _static_decision(prof, dev)
    if static is not None:
        return static
    return Decision(True, Reason.OK, fmt, profile=prof.name)


def _static_decision(prof: P.Profile, dev: torch.device) -> Decision | None:
    arch, cu = _device_info(dev)
    if arch not in prof.archs():
        return _no(Reason.ARCH_UNSUPPORTED, f"device arch {arch!r}", profile=prof.name)
    if cu not in prof.cus(arch):
        return _no(Reason.CU_UNSUPPORTED, f"device cu {cu}", profile=prof.name)
    diff = P.kernel_env_diff(prof, _kernel_env())
    if diff:
        return _no(Reason.KERNEL_ENV_MISMATCH,
                   ", ".join(f"{k}={v[1]!r} (profiled {v[0]!r})" for k, v in sorted(diff.items())),
                   profile=prof.name)
    sha = _kernel_sha256()
    if sha != prof.kernel_sha256:
        return _no(Reason.PROFILE_KERNEL_MISMATCH,
                   f"kernel sha256 {sha[:12]} != profiled {prof.kernel_sha256[:12]}",
                   profile=prof.name)
    return None


def decide(q, compressed_k_cache, page_table, token_to_request, query_positions, context_lens,
           compress_ratio=4, score_divisor=None, max_columns=None, row_starts=None,
           allow_flydsl=False, compressed_k_scale=None, *, logits=None, visible_groups=None,
           profile: str | None = None, allow_unverified: bool = False) -> Decision:
    """Full decision for one call (``Decision.ok`` / typed ``Decision.reason``).

    ``allow_flydsl`` is ATOM's switch for its own page-16 FlyDSL prefill
    scorer; it does not apply to this (page-4) kernel and is ignored.
    ``allow_unverified`` is for :mod:`gen_profile` sweeps only.
    """
    prof = _resolve(profile)
    if isinstance(prof, Decision):
        return prof
    name = prof.name

    def no(reason, detail=""):
        return _no(reason, detail, profile=name)

    if int(compress_ratio) != P.COMPRESS_RATIO:
        return no(Reason.COMPRESS_RATIO, f"compress_ratio {compress_ratio} != {P.COMPRESS_RATIO}")
    if not isinstance(q, torch.Tensor) or q.ndim != 3 or tuple(q.shape[1:]) != (P.INDEX_HEADS, P.HEAD_DIM):
        return no(Reason.Q_SHAPE, f"q {tuple(getattr(q, 'shape', ()))} != [rows, "
                                  f"{P.INDEX_HEADS}, {P.HEAD_DIM}]")
    if q.dtype != torch.bfloat16:
        return no(Reason.Q_DTYPE, f"q dtype {q.dtype} (bf16 only)")
    rows = int(q.shape[0])
    if rows == 0:
        return no(Reason.EMPTY, "rows == 0")
    if rows > P.MAX_ROWS:
        return no(Reason.ROWS_RULE, f"rows {rows} > {P.MAX_ROWS}")
    if not q.is_contiguous() or q.data_ptr() % 16:
        return no(Reason.Q_LAYOUT, "q must be contiguous and 16-byte aligned")
    dev = q.device
    if dev.type not in _ALLOWED_DEVICE_TYPES:
        return no(Reason.DEVICE_NOT_GPU, f"q on {dev}")
    fmt = _layout_decision(compressed_k_cache, compressed_k_scale, dev, prof)
    if isinstance(fmt, Decision):
        return no(fmt.reason, fmt.detail)
    if (not isinstance(page_table, torch.Tensor) or page_table.ndim != 2
            or page_table.shape[0] < 1 or page_table.shape[1] < 1
            or page_table.dtype not in (torch.int32, torch.int64) or not page_table.is_contiguous()):
        return no(Reason.METADATA, "page_table must be contiguous int32/int64 [requests>=1, width>=1]")
    requests, width = int(page_table.shape[0]), int(page_table.shape[1])
    for nm, t, n in (("token_to_request", token_to_request, rows),
                     ("query_positions", query_positions, rows),
                     ("context_lens", context_lens, requests)):
        if not _vec_ok(t, n):
            return no(Reason.METADATA, f"{nm} must be contiguous int32/int64 [{n}]")
    for nm, t in (("page_table", page_table), ("token_to_request", token_to_request),
                  ("query_positions", query_positions), ("context_lens", context_lens)):
        if t.device != dev:
            return no(Reason.DEVICE_MISMATCH, f"{nm} on {t.device}, q on {dev}")
    if score_divisor is not None and float(score_divisor) != _DEFAULT_DIVISOR:
        return no(Reason.DIVISOR, f"score_divisor {score_divisor} (golden covers sqrt(128) only)")
    if max_columns is not None and int(max_columns) < 1:
        return no(Reason.MAX_COLUMNS, f"max_columns {max_columns} < 1 (native clamps to 1)")
    columns = width * P.PAGE_SIZE if max_columns is None else min(width * P.PAGE_SIZE, int(max_columns))
    if logits is not None and (tuple(logits.shape) != (rows, columns) or logits.dtype != torch.float32
                               or not logits.is_contiguous() or logits.device != dev):
        return no(Reason.OUTPUT_LAYOUT, f"logits must be contiguous fp32 {(rows, columns)} on {dev}")
    for nm, t in (("visible_groups", visible_groups), ("row_starts", row_starts)):
        if t is not None and (t.numel() != rows or t.dtype != torch.int32
                              or not t.is_contiguous() or t.device != dev):
            return no(Reason.OUTPUT_LAYOUT, f"{nm} must be contiguous int32 [{rows}] on {dev}")
    elem = 1 if fmt == "fp8" else 2
    pages = int(compressed_k_cache.shape[0])
    for nm, nbytes in (("q", rows * P.INDEX_HEADS * P.HEAD_DIM * 2),
                       ("compressed_k_cache", pages * P.PAGE_SIZE * P.HEAD_DIM * elem),
                       ("compressed_k_scale", pages * P.PAGE_SIZE * 4 if fmt == "fp8" else 4),
                       ("page_table", requests * width * 4), ("logits", rows * columns * 4)):
        if nbytes > P.MAX_PLANE_BYTES:
            return no(Reason.PLANE_TOO_LARGE, f"{nm} {nbytes} B > int32 byte addressing")
    static = _static_decision(prof, dev)
    if static is not None:
        return static
    arch, cu = _device_info(dev)
    m = P.match(prof, arch=arch, cu=cu, cache_dtype=fmt, rows=rows, requests=requests,
                columns=columns, table_width=width, allow_unverified=allow_unverified)
    if not m.ok:
        return Decision(False, m.reason, m.detail, profile=name,
                        cell=m.row.cell if m.row else None, columns=columns)
    try:
        K = _load_kernel()
    except Exception as exc:  # noqa: BLE001 -- any import failure means "run native"
        return no(Reason.KERNEL_UNAVAILABLE, f"{type(exc).__name__}: {exc}")
    diff = P.kernel_env_diff(prof, _kernel_env())   # import-time snapshot (may differ now)
    if diff:
        return no(Reason.KERNEL_ENV_MISMATCH, ", ".join(sorted(diff)))
    vid, params = predict_route(K, fmt, rows, requests, columns, compressed_k_cache.data_ptr())
    if vid != m.row.vid or params != dict(m.row.params):
        return Decision(False, Reason.ROUTE_MISMATCH, f"router {vid} {params} != row "
                        f"{m.row.vid} {dict(m.row.params)}", profile=name, cell=m.row.cell,
                        columns=columns)
    return Decision(True, Reason.OK, m.detail, profile=name, cell=m.row.cell, vid=vid,
                    columns=columns)


def eligibility(q, compressed_k_cache, page_table, token_to_request, query_positions,
                context_lens, compress_ratio=4, score_divisor=None, max_columns=None,
                row_starts=None, allow_flydsl=False, compressed_k_scale=None, *, logits=None,
                visible_groups=None, profile: str | None = None) -> tuple[bool, str]:
    """``(ok, reason)`` for one call; ``reason`` is a :class:`Reason` value."""
    d = decide(q, compressed_k_cache, page_table, token_to_request, query_positions,
               context_lens, compress_ratio, score_divisor, max_columns, row_starts,
               allow_flydsl, compressed_k_scale, logits=logits, visible_groups=visible_groups,
               profile=profile)
    return d.ok, d.reason.value


def _launch(decision: Decision, q, compressed_k_cache, page_table, token_to_request,
            query_positions, context_lens, score_divisor, max_columns, row_starts,
            compressed_k_scale, logits, visible_groups):
    K = _load_kernel()
    return K.qsa_paged_mqa_logits_page4(
        q, compressed_k_cache, page_table, token_to_request, query_positions, context_lens,
        compress_ratio=P.COMPRESS_RATIO, divisor=score_divisor, max_columns=max_columns,
        compressed_k_scale=compressed_k_scale, logits=logits, visible_groups=visible_groups,
        row_starts=row_starts)


def qsa_paged_mqa_logits(q: torch.Tensor, compressed_k_cache: torch.Tensor,
                         page_table: torch.Tensor, token_to_request: torch.Tensor,
                         query_positions: torch.Tensor, context_lens: torch.Tensor,
                         compress_ratio: int = 4, score_divisor: float | None = None,
                         max_columns: int | None = None, row_starts: torch.Tensor | None = None,
                         allow_flydsl: bool = False,
                         compressed_k_scale: torch.Tensor | None = None, *,
                         logits: torch.Tensor | None = None,
                         visible_groups: torch.Tensor | None = None,
                         profile: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused scorer for a profiled call; raises :class:`QSAIndexScorerUnsupported` otherwise.

    Returns ``(logits [rows, columns] fp32, visible_groups [rows] int32)`` with
    ATOM's semantics (``-inf`` outside ``c < visible`` and on invalid pages;
    ``row_starts``, if given, is zero-filled).
    """
    d = decide(q, compressed_k_cache, page_table, token_to_request, query_positions,
               context_lens, compress_ratio, score_divisor, max_columns, row_starts,
               allow_flydsl, compressed_k_scale, logits=logits, visible_groups=visible_groups,
               profile=profile)
    if not d.ok:
        raise QSAIndexScorerUnsupported(d)
    return _launch(d, q, compressed_k_cache, page_table, token_to_request, query_positions,
                   context_lens, score_divisor, max_columns, row_starts, compressed_k_scale,
                   logits, visible_groups)


def try_qsa_paged_mqa_logits(*args, **kwargs):
    """``(out, decision)``: ``out`` is ``(logits, visible)`` or ``None`` (run native)."""
    d = decide(*args, **kwargs)
    if not d.ok:
        return None, d
    bound = inspect.signature(decide).bind(*args, **kwargs)
    bound.apply_defaults()
    a = bound.arguments
    return _launch(d, a["q"], a["compressed_k_cache"], a["page_table"], a["token_to_request"],
                   a["query_positions"], a["context_lens"], a["score_divisor"],
                   a["max_columns"], a["row_starts"], a["compressed_k_scale"], a["logits"],
                   a["visible_groups"]), d


def _kernel_loaded() -> bool:
    """True once the kernel module (and hence FlyDSL) has been imported by this API."""
    return _KERNEL is not None or P.KERNEL_MODULE in sys.modules

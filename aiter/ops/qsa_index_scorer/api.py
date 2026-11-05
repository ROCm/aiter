# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public op: QSA index scorer (paged MQA logits, index page 4 or 16) -- FlyDSL gfx1250.

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

API 1.1 (zero-overhead serving path, additive)::

    prewarm(profile=DEFAULT_PROFILE)          # once at model load, before graph capture
    d = decide(<call>, profile=DEFAULT_PROFILE)   # once per call class (bucket)
    logits, visible = launch(d, <call>)       # every step: no decide, no tensor reads/sync

:func:`qsa_paged_mqa_logits` is exactly ``decide`` + ``launch`` (shared code).

R2 additions (all opt-in / keyword-only; API 1.1 callers unaffected):

* **Token-dimension buckets** (ROUND-R2E: ON by default; ``decide(..., buckets=False)`` or
  ``$AITER_QSA_INDEX_SCORER_BUCKETS=0`` disables; only ``pass`` profile buckets whose padded target
  shape is served by a ``pass`` row/rule). ``group``: MTP verify bench3 (rows = 3 x
  requests) padded to the decode-group kernel's 4-row group through pre-allocated staging
  planes (allocated at ``decide``; per call 3 ``index_select`` in + kernel + 2
  ``index_select`` out; nothing allocated when outputs are supplied, no host sync); the
  caller sees exactly its real rows (bit-identical per-row results, caller buffers never
  overrun). R2-MERGE default: the ``verify3_g3_*`` bucket (``params.group_rows`` 3,
  listed first) launches the kernel's native 3-row groups on the caller's own tensors
  (:func:`_group_kernel_launch`: one launch, zero copy, no staging) whenever
  :func:`kernel_real_rows_per_group` has 3; the ``verify3_pad4_*`` staging bucket is the
  fallback. Pinned launch knobs (``P.LAUNCH_KNOBS``: ``wg_waves``, bucket ``group_rows``)
  are builder keywords; ``route_mismatch`` compares the variant without them. ``rows``: tail-band rows rounded up to
  the decode band: the decode-group kernel runs with the real row count as its row bound
  (no padding, no writes beyond the real rows); one decision serves the bucket's rows.
* **Warm-cache time**: ``prewarm(compressed_k_cache=..., pos_dtypes=(torch.int64,))`` builds
  only what that cache can select; FlyDSL's persistent JIT cache is directed to
  :func:`compile_cache_dir` (``$AITER_QSA_INDEX_SCORER_CACHE_DIR`` > ``$FLYDSL_RUNTIME_CACHE_DIR``
  > ``~/.cache/aiter/qsa_index_scorer/<kernel sha16>``) around our compiles only;
  ``jobs > 1`` compiles in no-device worker processes (FlyDSL tracing is not thread-safe);
  ``compile_only=True`` populates the cache on a no-device host; :func:`prewarm_stats`.
* **Page 8** (KV block 32): profile page size; served only when the kernel reports it in
  ``SUPPORTED_PAGE_SIZES`` (:func:`kernel_page_sizes`), else ``cache_shape_unsupported``.
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
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from . import profiles as P
from .profiles import Reason

__all__ = [
    "ATOM_SCORER_SIGNATURE", "Decision", "QSAIndexScorerUnsupported", "decide", "eligibility",
    "launch", "layout_eligibility", "predict_route", "prewarm", "qsa_paged_mqa_logits",
    "try_qsa_paged_mqa_logits", "selected_profile", "prewarm_plan", "prewarm_stats",
    "compile_cache_dir", "kernel_page_sizes",
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
    #: private launch plan of an ``ok`` decision (variant, geometry, profile binding)
    _plan: Any = field(default=None, repr=False, compare=False)

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


_ENV_MEMO: dict = {}   # id(profile) -> (profile, env snapshot, diff)


def _env_diff(prof: P.Profile) -> dict:
    """``kernel_env_diff`` against the import-time snapshot, memoised per (profile,
    snapshot) identity (the snapshot is assigned once by ``_load_kernel``, never
    mutated); before the kernel import the live environment is read every time."""
    env = _KERNEL_ENV
    if env is None:
        return P.kernel_env_diff(prof, P.effective_kernel_env())
    memo = _ENV_MEMO.get(id(prof))
    if memo is None or memo[0] is not prof or memo[1] is not env:
        memo = (prof, env, P.kernel_env_diff(prof, env))
        _ENV_MEMO[id(prof)] = memo
    return memo[2]


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
                  kv_ptr: int, page_size: int = P.PAGE_SIZE) -> tuple[str, dict]:
    """(vid, params) the kernel wrapper launches for this call (default-path routing).

    ROUND-PAGE16: the router is page-size independent; the vid namespace is
    ``qsa.page<P>`` (``P.variant_prefix``)."""
    vid, params = _predict_route4(K, cache_dtype, rows, requests, columns, kv_ptr, page_size)
    params.update(_env_knobs(K, vid, rows, requests, cache_dtype))
    if int(page_size) != P.PAGE_SIZE:
        vid = P.variant_prefix(page_size) + vid[len("qsa.page4"):]
    return vid, params


def _env_knobs(K, vid: str, rows: int, requests: int, cache_dtype: str = "bf16") -> dict:
    """ROUND-R2 wrapper knobs the import-time env applies to this launch (non-default
    only, as ``P.LAUNCH_KNOBS`` params; ``{}`` under the profiled default env)."""
    kind = P.variant_kind(vid)
    if kind == "draft":
        if hasattr(K, "draft_wg_waves"):           # ROUND-R2E: per-dtype 'auto' (fp8 2)
            w = int(K.draft_wg_waves(cache_dtype == "fp8"))
        else:
            w = int(getattr(K, "_DRAFT_WG_WAVES", 4))
        return {} if w == P.KNOB_DEFAULTS["wg_waves"] else {"wg_waves": w}
    if kind == "decg" and hasattr(K, "decode_group_knobs"):
        g, mode, w = K.decode_group_knobs(rows, requests)
        out = {}
        if g != P.KNOB_DEFAULTS["group_rows"]:
            out["group_rows"] = int(g)
        if w != P.KNOB_DEFAULTS["wg_waves"]:
            out["wg_waves"] = int(w)
        if mode != ("dedup" if g == 3 else "base"):
            out["row_mode"] = mode            # never a profile knob: always route_mismatch
        return out
    return {}


def _knob_kw(vid: str, params) -> dict:
    """Builder keywords of a variant's pinned launch knobs -- exactly the keywords the
    kernel wrapper passes for the same knobs (same ``lru_cache`` entry / launcher)."""
    knobs = P.split_knobs(vid, params)[1]
    if not knobs:
        return {}
    kind = P.variant_kind(vid)
    w = int(knobs.get("wg_waves", P.KNOB_DEFAULTS["wg_waves"]))
    if kind == "draft":
        return {"wg_waves": w}
    g = int(knobs.get("group_rows", P.KNOB_DEFAULTS["group_rows"]))
    return {"group_rows": g, "row_mode": "dedup" if g == 3 else "base", "wg_waves": w}


def _route_ok(vid: str, params, want_vid: str, want_params) -> bool:
    """Router (vid, params) vs a row/rule/bucket: same variant; launch knobs are the
    entry's to pin, but an env-applied knob must equal the pinned one."""
    if vid != want_vid:
        return False
    rb, rk = P.split_knobs(vid, params)
    wb, wk = P.split_knobs(want_vid, want_params)
    return rb == wb and (not rk or rk == wk)


def _predict_route4(K, cache_dtype: str, rows: int, requests: int, columns: int,
                    kv_ptr: int, page_size: int = P.PAGE_SIZE) -> tuple[str, dict]:
    is_fp8 = cache_dtype == "fp8"
    pk8 = "asm" if getattr(K, "_FP8_CVT_ASM", False) else "intrin"
    if K._DRAFT_PATH:
        d = K.draft_route(rows, requests, is_fp8, kv_ptr)
        if d is not None:
            params = {"k_load": d[0], "wave_cols": int(d[1])}
            if is_fp8:
                params.update(sched=K.draft_fp8_sched(), pk8=pk8)
            return "qsa.page4.draft.v1", params
    if K._DECODE_GROUP != "0":
        if hasattr(K, "decode_group_knobs"):     # R2 kernel: the wrapper's group_rows
            w = K.decode_group_route(rows, requests, is_fp8, kv_ptr,
                                     K.decode_group_knobs(rows, requests)[0])
        else:
            w = K.decode_group_route(rows, requests, is_fp8, kv_ptr)
        if w is not None:
            params = {"k_load": "tdm", "wave_cols": int(w), "barrier": False}
            if is_fp8:
                params.update(sched=K.decg_fp8_sched(), pk8=pk8)
            return "qsa.page4.decg.v1", params
    if not is_fp8 and K._BF16_PATH == "wmma":
        rpb = K.wmma_bf16_rows_per_block(rows, requests)
        k_load = K._resolve_bf16_k_load(rpb)
        if k_load == "tdm" and kv_ptr % 128:
            rpb, k_load = 16, "raw"
        if rpb == 32 and k_load == "tdm" and K._ACCESS == "raw" and hasattr(K, "prefill_bf16_route"):
            # ROUND-R2B: the kernel's data-driven route table (default 'legacy' == below).
            pk, pkw = K.prefill_bf16_route(rows, requests, columns, page_size)
            if pk == "persist":
                return "qsa.page4.r32_persist.v1", {"k_load": "tdm", "rows_per_block": 32,
                                                    "persist_cols": int(K.TDM_PERSIST_CHUNKS),
                                                    "buffers": int(K.TDM_PERSIST_BUFFERS)}
            if pk == "ktile":                        # ROUND-R2C (route variant kt<RB>x<C>)
                kp = {"k_load": "tdm", "rows_per_block": 32, "kt_rows": int(pkw["kt_rows"]),
                      "col_tiles": int(pkw["col_tiles"])}
                if pkw.get("lean"):                  # ROUND-R2D (route variant kt<RB>x<C>l)
                    kp["lean"] = True
                if pkw.get("ipf"):                   # ROUND-R2D (route variant ...p<KB>)
                    kp["ipf"] = int(pkw["ipf"])
                return "qsa.page4.r32_kt.v1", kp
            if pk == "kstat":
                kp = {"k_load": "tdm", "rows_per_block": 32, "row_tiles": int(pkw["row_tiles"]),
                      "col_tiles": int(pkw["col_tiles"])}
                if pkw.get("early"):                 # ROUND-R2C (route variant kstat<R>x<C>e)
                    kp["early"] = True
                return "qsa.page4.r32_kstat.v1", kp
        elif (rpb == 32 and k_load == "tdm" and K.TDM_PERSIST and K._ACCESS == "raw"
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
        if (rpb == 32 and k_load == "tdm" and K._ACCESS == "raw"
                and hasattr(K, "prefill_fp8_route")):
            # ROUND-R2E: the kernel's data-driven fp8 route table (QSA_FP8_PREFILL_ROUTE;
            # QSA_FP8_KT / QSA_FP8_EARLY=1 win inside the kernel router, as in the wrapper)
            fk, fkw = K.prefill_fp8_route(rows, requests, columns, page_size)
            if fk == "ktile":
                kp = {"k_load": "tdm", "rows_per_block": 32, "kt_rows": int(fkw["kt_rows"]),
                      "col_tiles": int(fkw["col_tiles"]), "cvt": K._FP8_CVT, "pk8": pk8}
                if fkw.get("lean"):
                    kp["lean"] = True
                if fkw.get("kconv"):
                    kp["kconv"] = True
                if fkw.get("ipf"):
                    kp["ipf"] = int(fkw["ipf"])
                return "qsa.page4.r32_kt_fp8.v1", kp
            fp = {"k_load": k_load, "rows_per_block": 32, "cvt": K._FP8_CVT, "pk8": pk8}
            if fk == "early":
                fp["early"] = True
            return "qsa.page4.r32_fp8.v1", fp
        if (rpb == 32 and k_load == "tdm" and K._ACCESS == "raw"
                and getattr(K, "FP8_KT", None) is not None):     # ROUND-R2C (QSA_FP8_KT)
            return "qsa.page4.r32_kt_fp8.v1", {"k_load": "tdm", "rows_per_block": 32,
                                               "kt_rows": int(K.FP8_KT[0]),
                                               "col_tiles": int(K.FP8_KT[1]),
                                               "cvt": K._FP8_CVT, "pk8": pk8}
        if rpb == 32:
            fp = {"k_load": k_load, "rows_per_block": 32, "cvt": K._FP8_CVT, "pk8": pk8}
            if (getattr(K, "FP8_R32_EARLY", False) and k_load == "tdm"
                    and K._ACCESS == "raw"):            # ROUND-R2C (QSA_FP8_EARLY=1)
                fp["early"] = True
            return "qsa.page4.r32_fp8.v1", fp
        return "qsa.page4.r16_fp8.v1", {"k_load": k_load, "rows_per_block": 16, "pk8": pk8}
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
    if (cache.ndim != 4 or int(cache.shape[1]) not in P.PAGE_SIZES
            or tuple(cache.shape[2:]) != (1, P.HEAD_DIM) or cache.shape[0] < 1):
        return _no(Reason.CACHE_SHAPE, f"compressed_k_cache {tuple(cache.shape)} != "
                                       f"[pages>=1, P in {P.PAGE_SIZES}, 1, {P.HEAD_DIM}]")
    if int(cache.shape[1]) not in kernel_page_sizes():   # R2 capability probe (page 8)
        return _no(Reason.CACHE_SHAPE, f"index page size {int(cache.shape[1])} not implemented "
                                       f"by the kernel (SUPPORTED_PAGE_SIZES {kernel_page_sizes()})")
    if not cache.is_contiguous():
        return _no(Reason.CACHE_LAYOUT, "compressed_k_cache not contiguous")
    align = int(prof.data["contract"].get("cache_align_bytes", P.CACHE_ALIGN_BYTES))
    if cache.data_ptr() % align:
        return _no(Reason.CACHE_ALIGNMENT, f"compressed_k_cache base {cache.data_ptr():#x} not "
                                           f"{align}-byte aligned (TDM routes need it)")
    if fmt == "fp8":
        page = int(cache.shape[1])
        if (scale.device != device or tuple(scale.shape) != (int(cache.shape[0]), page, 1)
                or not scale.is_contiguous() or scale.data_ptr() % 4):
            return _no(Reason.SCALE_LAYOUT, f"compressed_k_scale {tuple(scale.shape)} "
                                            f"{scale.dtype} must be contiguous fp32 [pages, {page}, 1]")
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
    diff = _env_diff(prof)
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
           profile: str | None = None, allow_unverified: bool = False,
           buckets: bool | None = None) -> Decision:
    """Full decision for one call (``Decision.ok`` / typed ``Decision.reason``).

    ``allow_flydsl`` is ATOM's switch for its own page-16 FlyDSL prefill
    scorer; it does not apply to this kernel and is ignored. ROUND-PAGE16: the
    index page size is read from ``compressed_k_cache.shape[1]`` (4 or 16) and
    only rows/rules of that page size match.
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
    page = int(compressed_k_cache.shape[1])          # validated by _layout_decision
    columns = width * page if max_columns is None else min(width * page, int(max_columns))
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
                       ("compressed_k_cache", pages * page * P.HEAD_DIM * elem),
                       ("compressed_k_scale", pages * page * 4 if fmt == "fp8" else 4),
                       ("page_table", requests * width * 4), ("logits", rows * columns * 4)):
        if nbytes > P.MAX_PLANE_BYTES:
            return no(Reason.PLANE_TOO_LARGE, f"{nm} {nbytes} B > int32 byte addressing")
    static = _static_decision(prof, dev)
    if static is not None:
        return static
    arch, cu = _device_info(dev)
    if prof.buckets and _buckets_enabled(buckets):
        bd = _decide_bucket(prof, arch, cu, fmt, rows, requests, columns, width, page, pages,
                            compressed_k_cache, allow_unverified, dev, query_positions.dtype,
                            token_to_request.dtype)
        if bd is not None:
            return bd
    m = P.match(prof, arch=arch, cu=cu, cache_dtype=fmt, rows=rows, requests=requests,
                columns=columns, table_width=width, allow_unverified=allow_unverified,
                page_size=page)
    if not m.ok:
        return Decision(False, m.reason, m.detail, profile=name,
                        cell=m.row.cell if m.row else None, columns=columns)
    try:
        K = _load_kernel()
    except Exception as exc:  # noqa: BLE001 -- any import failure means "run native"
        return no(Reason.KERNEL_UNAVAILABLE, f"{type(exc).__name__}: {exc}")
    diff = _env_diff(prof)   # import-time snapshot (may differ now)
    if diff:
        return no(Reason.KERNEL_ENV_MISMATCH, ", ".join(sorted(diff)))
    if page not in kernel_page_sizes(K):     # R2: page 8 before the page-8 kernel lands
        return no(Reason.CACHE_SHAPE, f"index page size {page} not implemented by the kernel "
                                      f"(SUPPORTED_PAGE_SIZES {kernel_page_sizes(K)})")
    vid, params = predict_route(K, fmt, rows, requests, columns, compressed_k_cache.data_ptr(),
                                page)
    if not _route_ok(vid, params, m.row.vid, m.row.params):
        return Decision(False, Reason.ROUTE_MISMATCH, f"router {vid} {params} != row "
                        f"{m.row.vid} {dict(m.row.params)}", profile=name, cell=m.row.cell,
                        columns=columns)
    plan = _Plan(K, prof, vid, dict(m.row.params), fmt == "fp8", compressed_k_cache.dtype, rows,
                 requests, width, columns, pages, page)
    return Decision(True, Reason.OK, m.detail, profile=name, cell=m.row.cell, vid=vid,
                    columns=columns, _plan=plan)


def _decide_bucket(prof, arch, cu, fmt, rows, requests, columns, width, page, pages, cache,
                   allow_unverified, dev=None, pos_dtype=None, t2r_dtype=None) -> Decision | None:
    """R2 bucket decision, or None (no bucket applies: the call is matched unpadded).

    R2-MERGE: a kernel-native group bucket (``group.kernel_group_rows`` 3,
    ``params.group_rows`` 3) launches the decode-group kernel's 3-row groups on the
    caller's own tensors (plan rows = the real rows, no staging); it is skipped when
    the kernel lacks the capability, so the staging bucket (pad 3 -> 4) applies."""
    try:
        K = _load_kernel()
    except Exception:  # noqa: BLE001 -- the unpadded path reports kernel_unavailable
        return None
    if _env_diff(prof) or page not in kernel_page_sizes(K):
        return None
    bm = P.match_bucket(prof, arch=arch, cu=cu, cache_dtype=fmt, rows=rows, requests=requests,
                        columns=columns, table_width=width, allow_unverified=allow_unverified,
                        page_size=page, native_group_rows=kernel_real_rows_per_group(K))
    if bm is None:
        return None
    t = bm.target_rows
    if (t * columns * 4 > P.MAX_PLANE_BYTES
            or t * P.INDEX_HEADS * P.HEAD_DIM * 2 > P.MAX_PLANE_BYTES or t > P.MAX_ROWS):
        return None
    vid, params = predict_route(K, fmt, t, requests, columns, cache.data_ptr(), page)
    b = bm.bucket.raw
    kg0 = (b.get("group") or {}).get("kernel_group_rows") if b["kind"] == "group" else None
    if kg0 is not None and P.variant_kind(vid) == "decg":
        # ROUND-R2E: a kernel-native group launch uses the router's slab for group_rows
        # (decode_group_route(..., group_rows), checked again below on the real rows)
        w3 = K.decode_group_route(rows, requests, fmt == "fp8", cache.data_ptr(), int(kg0))
        if w3 is not None:
            params = dict(params, wave_cols=int(w3))
    if not _route_ok(vid, params, bm.bucket.vid, bm.bucket.params):
        return None
    launch_rows, call_rows = t, bm.call_rows
    if b["kind"] == "group":
        g = b["group"]
        kg = g.get("kernel_group_rows")
        bp = _BucketPlan(b["id"], "group", g["real"], g["padded"], kg is not None)
        if kg is not None:
            # the kernel's own router for the real-rows launch (rows % group_rows, r16
            # regime, 128 B TDM base) must give the bucket's slab width
            w = K.decode_group_route(rows, requests, fmt == "fp8", cache.data_ptr(), int(kg))
            if w is None or int(w) != int(bm.bucket.params.get("wave_cols", -1)):
                return None
            launch_rows, call_rows = rows, (rows, rows)
    else:
        bp = _BucketPlan(b["id"], "rows")
    plan = _Plan(K, prof, vid, dict(bm.bucket.params), fmt == "fp8", cache.dtype, launch_rows,
                 requests, width, columns, pages, page, bucket=bp, call_rows=call_rows)
    if plan.kind != "decg":
        return None
    if bp.kind == "group" and not bp.real_rows_kernel and dev is not None:
        _group_stage(plan, dev, rows, pos_dtype, t2r_dtype)   # allocate before any capture
    return Decision(True, Reason.OK, f"{bm.bucket.cell} -> rows {t} ({bm.target.detail})",
                    profile=prof.name, cell=f"{bm.bucket.cell}>{bm.target.row.cell}", vid=vid,
                    columns=columns, _plan=plan)


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
    """Generic path: the kernel module's own wrapper (re-validates and re-routes).

    Used only for decisions whose variant has no direct launch plan (none of the
    profiled ``pass`` variants; kept for completeness and as the GPU reference).
    """
    K = _load_kernel()
    return K.qsa_paged_mqa_logits_page4(
        q, compressed_k_cache, page_table, token_to_request, query_positions, context_lens,
        compress_ratio=P.COMPRESS_RATIO, divisor=score_divisor, max_columns=max_columns,
        compressed_k_scale=compressed_k_scale, logits=logits, visible_groups=visible_groups,
        row_starts=row_starts)


# ---------------------------------------------------------------------------
# Launch plans (API 1.1): the kernel wrapper's default-env launch, precomputed
# ---------------------------------------------------------------------------
# A plan is built once by ``decide`` for an ``ok`` call and replays exactly the
# launch the kernel wrapper ``qsa_paged_mqa_logits_page4`` performs for that call
# class (default kernel env, which ``decide`` enforces): the same builder with the
# same positional/keyword arguments (hence the same ``lru_cache`` entry, launcher
# object and FlyDSL compiled function), the same grid and the same ``_run``
# arguments. Everything shape-, dtype- or route-dependent is resolved here; the
# per-call work is output allocation (when not supplied), zero-copy views, the
# int64 metadata narrowing the wrapper also does, and the launch itself.
_I32, _I64, _U8, _F32, _BF16 = torch.int32, torch.int64, torch.uint8, torch.float32, torch.bfloat16
_PROFILE_CACHE = P._CACHE   # identity of the loaded profile objects (O(1) binding check)


def _variant_spec(vid: str, params, is_fp8: bool):
    """(kind, builder name, builder positional args) the wrapper uses for a variant.

    Mirrors ``qsa_paged_mqa_logits_page4`` (draft / decode-group / bf16 WMMA incl.
    persist / fp8 WMMA branches); ``pos64`` is added per call. ``None`` = no direct
    plan (scalar or unknown variant): the generic wrapper path is used.
    """
    div = _DEFAULT_DIVISOR                 # == float(math.sqrt(HEAD_DIM)) in the wrapper
    is_fp8 = bool(is_fp8)
    for pg in P.PAGE_SIZES:                # ROUND-PAGE16: same builders, page_size keyword
        pre = P.variant_prefix(pg) + "."
        if pg != P.PAGE_SIZE and vid.startswith(pre):
            vid = "qsa.page4." + vid[len(pre):]
    if vid == "qsa.page4.draft.v1":
        return "draft", "_build_draft_kernel", (is_fp8, div, str(params["k_load"]),
                                                int(params["wave_cols"]))
    if vid == "qsa.page4.decg.v1":
        return "decg", "_build_decode_group_kernel", (is_fp8, div, int(params["wave_cols"]))
    if not is_fp8 and vid == "qsa.page4.r32_persist.v1":
        return "wmma", "_build_wmma_kernel", (div, 32, "persist")
    if not is_fp8 and vid == "qsa.page4.r32_kstat.v1":    # ROUND-R2B (knobs: build_kw)
        return "wmma", "_build_wmma_kernel", (div, 32, "kstat")
    if not is_fp8 and vid == "qsa.page4.r32_kt.v1":       # ROUND-R2C (knobs: build_kw)
        return "wmma", "_build_wmma_kernel", (div, 32, "ktile")
    if not is_fp8 and vid in ("qsa.page4.r16.v1", "qsa.page4.r32.v1", "qsa.page4.r64.v1"):
        return "wmma", "_build_wmma_kernel", (div, int(params["rows_per_block"]),
                                              str(params["k_load"]))
    if is_fp8 and vid == "qsa.page4.r32_kt_fp8.v1":         # ROUND-R2C (knobs: build_kw)
        return "wmma_fp8", "_build_wmma_fp8_kernel", (div, 32, "ktile")
    if is_fp8 and vid in ("qsa.page4.r16_fp8.v1", "qsa.page4.r32_fp8.v1"):
        return "wmma_fp8", "_build_wmma_fp8_kernel", (div, int(params["rows_per_block"]),
                                                      str(params["k_load"]))
    return None


class _Plan:
    """Launch plan of one decided call class (see the section comment)."""

    __slots__ = ("kernel", "profile", "path", "vid", "params", "is_fp8", "cache_dtype", "rows",
                 "requests", "width", "columns", "pages", "pt_shape", "kind", "builder",
                 "build_args", "scaled", "exes", "page_size", "build_kw", "bucket", "call_lo",
                 "call_hi", "warm")

    def __init__(self, K, prof, vid, params, is_fp8, cache_dtype, rows, requests, width,
                 columns, pages, page_size=P.PAGE_SIZE, bucket=None, call_rows=None):
        self.bucket = bucket
        # caller rows the plan serves: exactly `rows`, or a bucket's real-row range
        lo_hi = (int(rows), int(rows)) if call_rows is None else call_rows
        self.call_lo, self.call_hi = int(lo_hi[0]), int(lo_hi[1])
        self.page_size = int(page_size)
        # P = 4: no keyword (the wrapper's exact lru_cache entry); P = 16: page_size=16
        self.build_kw = {} if self.page_size == P.PAGE_SIZE else {"page_size": self.page_size}
        # R2-MERGE: pinned launch knobs (group_rows / wg_waves) = the wrapper's keywords
        self.build_kw.update(_knob_kw(vid, params))
        if P.variant_kind(vid) == "r32_kstat":       # ROUND-R2B: structural builder keywords
            self.build_kw.update(row_tiles=int(params["row_tiles"]),
                                 col_tiles=int(params["col_tiles"]))
        if P.variant_kind(vid) in ("r32_kt", "r32_kt_fp8"):   # ROUND-R2C: structural keywords
            self.build_kw.update(kt_rows=int(params["kt_rows"]), col_tiles=int(params["col_tiles"]))
            if params.get("lean"):                       # ROUND-R2D structural keywords
                self.build_kw["lean"] = True
            if params.get("ipf"):
                self.build_kw["ipf"] = int(params["ipf"])
            if params.get("kconv"):                      # fp8 only (builder validates)
                self.build_kw["kconv"] = True
        if params.get("early") and P.variant_kind(vid) in ("r32_kstat", "r32_fp8"):
            self.build_kw["early"] = True               # ROUND-R2C structural keyword
        self.kernel, self.profile, self.path = K, prof, prof.path
        self.vid, self.params = vid, dict(params)
        self.is_fp8, self.cache_dtype = bool(is_fp8), cache_dtype
        self.rows, self.requests, self.width = int(rows), int(requests), int(width)
        self.columns, self.pages = int(columns), int(pages)
        self.pt_shape = (self.requests, self.width)
        spec = _variant_spec(vid, params, is_fp8)
        if spec is None or getattr(K, "_QSA_STRICT_I32", False):
            spec = (None, None, ())           # generic wrapper path
        self.kind, self.builder, self.build_args = spec
        self.scaled = self.kind in ("draft", "decg", "wmma_fp8")   # kernels taking a scale plane
        self.exes = [None, None]              # [int32 positions, int64 positions]
        self.warm = set()                     # launchers this plan has launched (R2 cache scope)

    def build(self, pos64: bool):
        """(launcher, grid_x, grid_y) for the position dtype (builder lru_cache hit)."""
        K = self.kernel
        fn = getattr(K, self.builder)
        launcher = (fn(*self.build_args, pos64=True, **self.build_kw) if pos64
                    else fn(*self.build_args, **self.build_kw))
        gx, gy = self.grid(self.rows, launcher)
        ent = (launcher, gx, gy)
        self.exes[1 if pos64 else 0] = ent
        return ent

    def grid(self, rows: int, launcher=None) -> tuple[int, int]:
        """The wrapper's grid for ``rows`` rows (pure arithmetic)."""
        K, columns, a, kw = self.kernel, self.columns, self.build_args, self.build_kw
        if self.kind == "draft":
            if "wg_waves" in kw:
                return K.draft_grid(rows, columns, a[3], kw["wg_waves"])
            return K.draft_grid(rows, columns, a[3])
        if self.kind == "decg":
            if "group_rows" in kw:
                return K.decode_group_grid(rows, columns, a[2], kw["group_rows"], kw["wg_waves"])
            return K.decode_group_grid(rows, columns, a[2])
        if self.kind == "wmma":
            if launcher is None:
                launcher = (self.exes[0] or self.exes[1])[0]
            if hasattr(K, "wmma_launch_grid"):           # ROUND-R2B: kstat R x C spans
                return K.wmma_launch_grid(rows, columns, launcher)
            return K.wmma_grid(rows, columns, a[1], launcher.cols_per_block)
        if "kt_rows" in kw:                              # ROUND-R2C fp8 ktile
            return K.kt_grid(rows, columns, kw["kt_rows"], kw["col_tiles"])
        return K.wmma_grid(rows, columns, a[1])


def _execute(plan: _Plan, q, compressed_k_cache, page_table, token_to_request,
             query_positions, context_lens, score_divisor, max_columns, row_starts,
             compressed_k_scale, logits, visible_groups):
    """Run a decided call: the wrapper's launch for this class, without re-validation."""
    if plan.kind is None:
        return _launch(None, q, compressed_k_cache, page_table, token_to_request,
                       query_positions, context_lens, score_divisor, max_columns, row_starts,
                       compressed_k_scale, logits, visible_groups)
    if plan.bucket is not None:
        return _execute_bucket(plan, q, compressed_k_cache, page_table, token_to_request,
                               query_positions, context_lens, row_starts, compressed_k_scale,
                               logits, visible_groups)
    exe, args = _kernel_args(plan, q, compressed_k_cache, page_table, token_to_request,
                             query_positions, context_lens, row_starts, compressed_k_scale,
                             logits, visible_groups)
    _kernel_run(plan, exe, args, q.device)
    return args[-10], args[-9]          # logits, visible_groups (flat)


def _kernel_run(plan: _Plan, exe, args, dev) -> None:
    """Kernel adapter: one launch (``K._run``) on the current stream. First compile of a
    launcher (no ``_cf`` yet) runs with the persistent cache directory scoped in (R2)."""
    K = plan.kernel
    stream = K._launch_stream(dev)
    if exe not in plan.warm:          # first launch of this launcher through this plan
        if getattr(exe, "_cf", None) is None:
            with _scoped_cache_dir(compile_cache_dir(plan.profile.kernel_sha256)):
                K._run(exe, *args, stream)
        else:
            K._run(exe, *args, stream)
        plan.warm.add(exe)
        return
    K._run(exe, *args, stream)


def _kernel_args(plan: _Plan, q, compressed_k_cache, page_table, token_to_request,
                 query_positions, context_lens, row_starts, compressed_k_scale, logits,
                 visible_groups, rows: int | None = None):
    """(launcher, positional args without the stream) of one launch -- the wrapper's
    exact argument list. ``rows`` (R2 row buckets): the real row count of the call;
    the grid is derived from it (O(1) arithmetic), the launcher is the plan's."""
    pos64 = query_positions.dtype is _I64
    ent = plan.exes[pos64]
    if ent is None:
        ent = plan.build(pos64)
    exe, gx, gy = ent
    K = plan.kernel
    if rows is None or rows == plan.rows:
        rows = plan.rows
    else:
        gx, gy = plan.grid(rows)
    dev = q.device
    # same allocation order / views as the wrapper
    if logits is None:
        logits = torch.empty((rows, plan.columns), dtype=_F32, device=dev)
    vg = (torch.empty(rows, dtype=_I32, device=dev) if visible_groups is None
          else visible_groups.view(-1))
    rs = torch.empty(rows, dtype=_I32, device=dev) if row_starts is None else row_starts.view(-1)
    pt = page_table if page_table.dtype is _I32 else K._to_i32_saturated(page_table)
    t2r = (token_to_request if token_to_request.dtype is _I32
           else K._to_i32_saturated(token_to_request))
    cl = context_lens if context_lens.dtype is _I32 else K._to_i32_saturated(context_lens)
    qp = query_positions
    if t2r.stride(0) != 1:
        t2r = t2r.contiguous().view(-1)
    if qp.stride(0) != 1:
        qp = qp.contiguous().view(-1)
    if pos64:
        qp = qp.view(_I32)                  # [2*rows] (lo, hi) dwords, zero copy
    if cl.stride(0) != 1:
        cl = cl.contiguous().view(-1)
    pages = plan.pages
    kv_u8 = compressed_k_cache.view(_U8).view(pages, -1)
    q_u8 = q.view(_U8).view(rows, -1)
    if plan.scaled:
        ks = (compressed_k_scale.view(pages, plan.page_size, 1) if plan.is_fp8
              else K._dummy_scale(dev))
        return exe, (q_u8, kv_u8, ks, pt, t2r, qp, cl, logits, vg, rs, gx, gy, rows,
                     plan.columns, pages, plan.requests, plan.width)
    return exe, (q_u8, kv_u8, pt, t2r, qp, cl, logits, vg, rs, gx, gy, rows,
                 plan.columns, pages, plan.requests, plan.width)


# ---------------------------------------------------------------------------
# R2: token-dimension buckets. See profiles.py "Buckets". ROUND-R2E: ON by default (only
# ``pass`` buckets are ever selected); ``decide(buckets=False)`` or
# ``$AITER_QSA_INDEX_SCORER_BUCKETS=0`` disables them.
# ---------------------------------------------------------------------------
#: switch when ``decide(buckets=None)``: unset / "1" = buckets on, "0" (off/false/no) = off
BUCKETS_ENV = "AITER_QSA_INDEX_SCORER_BUCKETS"
BUCKETS_DEFAULT = True
#: (device, kind, call rows, padded rows, columns, pos dtype, t2r dtype) -> _GroupStage
_STAGES: dict = {}


def _buckets_enabled(buckets) -> bool:
    if buckets is not None:
        return bool(buckets)
    v = os.environ.get(BUCKETS_ENV, "").strip().lower()
    if not v:
        return BUCKETS_DEFAULT
    return v in ("1", "true", "on", "yes")


class _BucketPlan:
    """Bucket part of a plan: ``kind`` 'rows' (real rows <= bucket rows; the kernel's
    row bound masks the rest: nothing padded, nothing written beyond the real rows)
    or 'group' (``real`` rows per request padded to ``padded`` = the decode-group
    kernel's 4-row group)."""

    __slots__ = ("id", "kind", "real", "padded", "real_rows_kernel")

    def __init__(self, bid, kind, real=0, padded=0, real_rows_kernel=False):
        self.id, self.kind = bid, kind
        self.real, self.padded = int(real), int(padded)
        self.real_rows_kernel = bool(real_rows_kernel)


def kernel_real_rows_per_group(K=None) -> tuple:
    """Capability probe (kernel worker merge point): real rows per decode group the
    decode-group kernel can mask natively (``DECG_REAL_ROWS_PER_GROUP``, e.g. (3, 4)).
    ``()`` = not supported: the op pads through private staging buffers instead."""
    K = _KERNEL if K is None else K
    got = getattr(K, "DECG_REAL_ROWS_PER_GROUP", None)
    if got is None and K is not None:
        # ROUND-R2 kernel (d438d2b3): the decode-group builder's ``group_rows`` keyword,
        # validated by ``_check_decg_knobs`` (pure Python), is the capability
        chk = getattr(K, "_check_decg_knobs", None)
        got = []
        for g in (3, 4):
            try:
                if callable(chk) and int(chk(g, "base", 4)[0]) == g:
                    got.append(g)
            except (TypeError, ValueError):
                pass
    return tuple(int(x) for x in got or ())


class _GroupStage:
    """Pre-allocated, reused staging planes of a 'group' bucket call class.

    ``gather`` [padded rows]: padded slot -> source real row (the pad slot repeats the
    group's last real row, so its request id / position are those of a real row and a
    request-uniform group stays uniform: the decode-group kernel keeps its single-pass
    path). ``scatter`` [real rows]: real row -> padded slot. Allocated once per class
    (first decide/launch of the class, i.e. before graph capture); per call only
    ``index_select(out=)`` into these planes and into the caller's outputs.
    Not safe for concurrent use of one class from two streams.
    """

    __slots__ = ("gather", "scatter", "q", "t2r", "pos", "logits", "vis", "rs")

    def __init__(self, dev, rows, prows, columns, real, padded, pos_dtype, t2r_dtype):
        slot = torch.arange(prows, dtype=torch.int64)
        grp, k = slot // padded, slot % padded
        self.gather = (grp * real + torch.clamp(k, max=real - 1)).to(dev)
        r = torch.arange(rows, dtype=torch.int64)
        self.scatter = ((r // real) * padded + r % real).to(dev)
        self.q = _aligned_zeros((prows, P.INDEX_HEADS, P.HEAD_DIM), _BF16, dev, 16)
        self.t2r = torch.zeros(prows, dtype=t2r_dtype, device=dev)
        self.pos = torch.zeros(prows, dtype=pos_dtype, device=dev)
        self.logits = torch.empty((prows, columns), dtype=_F32, device=dev)
        self.vis = torch.empty(prows, dtype=_I32, device=dev)
        self.rs = torch.empty(prows, dtype=_I32, device=dev)


def _group_stage(plan: _Plan, dev, rows, pos_dtype, t2r_dtype) -> _GroupStage:
    b = plan.bucket
    key = (str(dev), plan.rows, rows, plan.columns, b.real, b.padded, pos_dtype, t2r_dtype)
    st = _STAGES.get(key)
    if st is None:
        st = _STAGES[key] = _GroupStage(dev, rows, plan.rows, plan.columns, b.real, b.padded,
                                        pos_dtype, t2r_dtype)
    return st


def _group_kernel_launch(plan: _Plan, q, cache, pt, t2r, qp, cl, row_starts, scale, logits,
                         visible_groups):
    """Kernel-native group bucket (R2-MERGE): ONE decode-group launch with
    ``group_rows`` = the bucket's real rows per request (the plan's build keywords),
    on the caller's own tensors -- each 3-row group is one request, one pass; the
    kernel writes exactly the real rows (logits / visible_groups / row_starts) straight
    into the caller's outputs. Zero copy, no staging, no allocation when outputs are
    supplied (the wrapper's QSA_DECG_ROWS3=1 launch, same launcher)."""
    exe, args = _kernel_args(plan, q, cache, pt, t2r, qp, cl, row_starts, scale, logits,
                             visible_groups)
    _kernel_run(plan, exe, args, q.device)
    return args[-10], args[-9]


def _execute_bucket(plan: _Plan, q, cache, pt, t2r, qp, cl, row_starts, scale, logits,
                    visible_groups):
    b = plan.bucket
    rows = q.shape[0]
    dev = q.device
    if b.kind == "rows":          # kernel row bound = real rows; grid from the real rows
        exe, args = _kernel_args(plan, q, cache, pt, t2r, qp, cl, row_starts, scale, logits,
                                 visible_groups, rows=rows)
        _kernel_run(plan, exe, args, dev)
        return args[-10], args[-9]
    if b.real_rows_kernel:
        return _group_kernel_launch(plan, q, cache, pt, t2r, qp, cl, row_starts, scale, logits,
                                    visible_groups)
    st = _group_stage(plan, dev, rows, qp.dtype, t2r.dtype)
    torch.index_select(q, 0, st.gather, out=st.q)
    torch.index_select(t2r, 0, st.gather, out=st.t2r)
    torch.index_select(qp, 0, st.gather, out=st.pos)
    exe, args = _kernel_args(plan, st.q, cache, pt, st.t2r, st.pos, cl, st.rs, scale,
                             st.logits, st.vis)
    _kernel_run(plan, exe, args, dev)
    if logits is None:
        logits = torch.empty((rows, plan.columns), dtype=_F32, device=dev)
    vg = (torch.empty(rows, dtype=_I32, device=dev) if visible_groups is None
          else visible_groups.view(-1))
    torch.index_select(st.logits, 0, st.scatter, out=logits)
    torch.index_select(st.vis, 0, st.scatter, out=vg)
    if row_starts is not None:
        row_starts.view(-1).zero_()
    return logits, vg


def _refusal(decision, reason: Reason | None = None, detail: str = "") -> QSAIndexScorerUnsupported:
    if reason is None and not getattr(decision, "ok", False):
        if isinstance(decision, Decision):
            return QSAIndexScorerUnsupported(decision)
        reason, detail = Reason.ROUTE_MISMATCH, f"not a Decision: {type(decision).__name__}"
    if reason is None:
        reason = Reason.PROFILE_KERNEL_MISMATCH
        detail = ("decision is not bound to the currently loaded profile/kernel "
                  "(re-run decide)")
    return QSAIndexScorerUnsupported(Decision(False, reason, detail,
                                              profile=getattr(decision, "profile", None),
                                              cell=getattr(decision, "cell", None)))


def launch(decision: Decision, q: torch.Tensor, compressed_k_cache: torch.Tensor,
           page_table: torch.Tensor, token_to_request: torch.Tensor,
           query_positions: torch.Tensor, context_lens: torch.Tensor,
           compress_ratio: int = 4, score_divisor: float | None = None,
           max_columns: int | None = None, row_starts: torch.Tensor | None = None,
           allow_flydsl: bool = False, compressed_k_scale: torch.Tensor | None = None, *,
           logits: torch.Tensor | None = None,
           visible_groups: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the variant fixed in an ``ok`` :class:`Decision` WITHOUT re-running ``decide``.

    Caller contract (API 1.1): ``decision`` came from :func:`decide` for a call of
    the identical class key -- shapes, dtypes, strides/contiguity, devices,
    alignment classes, ``compress_ratio``/``score_divisor``/``max_columns``,
    ``row_starts``/``logits``/``visible_groups`` presence and layout, cache
    tensor layout. ``compress_ratio``, ``score_divisor``, ``max_columns`` and
    ``allow_flydsl`` are taken from the decision (not re-read). Only O(1) guards:
    decision ok and bound to the loaded profile + kernel, q dtype, and the
    rows / page-table shape / cache dtype the plan was built for (memory safety).
    No tensor reads, no host sync, no environment reads. The result is bitwise
    the one :func:`qsa_paged_mqa_logits` returns (it is ``decide`` + this).
    Raises :class:`QSAIndexScorerUnsupported` when a guard fails.
    """
    plan = getattr(decision, "_plan", None)
    if (plan is None or plan.kernel is not _KERNEL
            or _PROFILE_CACHE.get(plan.path) is not plan.profile):
        raise _refusal(decision)
    if q.dtype is not _BF16:
        raise _refusal(decision, Reason.Q_DTYPE, f"q dtype {q.dtype} (bf16 only)")
    if (not plan.call_lo <= q.shape[0] <= plan.call_hi or page_table.shape != plan.pt_shape
            or compressed_k_cache.dtype is not plan.cache_dtype):
        raise _refusal(decision, Reason.ROUTE_MISMATCH,
                       "call does not match the decision's class key (rows / page_table "
                       "shape / cache dtype): re-run decide")
    return _execute(plan, q, compressed_k_cache, page_table, token_to_request, query_positions,
                    context_lens, score_divisor, max_columns, row_starts, compressed_k_scale,
                    logits, visible_groups)


# ---------------------------------------------------------------------------
# Prewarm (API 1.1): build + JIT-compile every variant the profile can select
# ---------------------------------------------------------------------------
#: tiny private problem per variant: (rows, requests, table_width); the kernel
#: router must select the variant for it (checked before launching)
_PREWARM_GEOMETRY = {
    "qsa.page4.draft.v1": (1, 1, 16),
    "qsa.page4.decg.v1": (4, 1, 16),
    "qsa.page4.r16.v1": (5, 1, 16),
    "qsa.page4.r16_fp8.v1": (5, 1, 16),
    "qsa.page4.r32.v1": (32, 1, 16),
    # ROUND-R2E: route presets r2e -> r32 fp8 / persist only above 2048 columns, lean ktile
    # (bf16 RB32 x C2; fp8 x C2 for rows <= 2048, x C4 above) below
    "qsa.page4.r32_fp8.v1": (32, 1, 1024),
    "qsa.page4.r32_persist.v1": (32, 1, 1024),
    "qsa.page4.r32_kt.v1": (32, 1, 16),
    "qsa.page4.r32_kt_fp8.v1": (32, 1, 16),
}
#: ROUND-R2E: (vid, param) -> geometry overrides (fp8 ktile C4 = the rows > 2048 band)
_PREWARM_GEOMETRY_PARAMS = {("qsa.page4.r32_kt_fp8.v1", ("col_tiles", 4)): (2080, 1, 16)}
_PREWARM_PAGES = 4
_PREWARM_LOCK = threading.Lock()
#: (profile path, kernel sha, device, tag, params) -> seconds the build + first launch took
_PREWARMED: dict = {}
#: tag -> {"seconds", "cached" (already warm in-process), "compile_only"} of the last prewarm
_PREWARM_STATS: dict = {}


def _vid_page(vid: str) -> int:
    """Index page size encoded in a variant id (``qsa.page<P>.*``)."""
    pre = vid.split(".", 2)[1] if vid.count(".") >= 2 else ""
    return int(pre[4:]) if pre.startswith("page") and pre[4:].isdigit() else P.PAGE_SIZE


def kernel_page_sizes(K=None) -> tuple:
    """Index page sizes the (loaded or given) kernel module implements.

    R2 capability probe: ``SUPPORTED_PAGE_SIZES`` of the kernel module (4 / 16
    today; the page-8 kernel adds 8). Without a loaded kernel: the same constant read
    from the kernel source text (:func:`_source_page_sizes`, no import).
    """
    K = _KERNEL if K is None else K
    got = None if K is None else getattr(K, "SUPPORTED_PAGE_SIZES", None)
    if got is None:
        return _source_page_sizes()
    return got if type(got) is tuple else tuple(int(p) for p in got)


_SRC_PAGES: dict = {}


def _source_page_sizes(path: str | None = None) -> tuple:
    """``SUPPORTED_PAGE_SIZES`` read from the kernel source text (no import, no FlyDSL):
    the capability probe before the kernel module is loaded. Memoised per path."""
    import re  # noqa: PLC0415
    path = path or P.KERNEL_PATH
    got = _SRC_PAGES.get(path)
    if got is None:
        got = (P.PAGE_SIZE,)
        try:
            with open(path, encoding="utf-8") as fh:
                m = re.search(r"^SUPPORTED_PAGE_SIZES\s*=\s*\(([0-9,\s]+)\)", fh.read(), re.M)
            if m:
                got = tuple(int(x) for x in m.group(1).replace(" ", "").split(",") if x)
        except OSError:
            pass
        _SRC_PAGES[path] = got
    return got


def _profile_variants(prof: P.Profile, cache_dtypes, arch: str | None = None,
                      cu: int | None = None, page_sizes=None) -> list[tuple[str, str, tuple]]:
    """Sorted unique (cache_dtype, vid, params items) over the ``pass`` rows, rules and
    bucket targets, optionally restricted to ``page_sizes``.

    The page size is part of ``vid`` (``qsa.page4.*`` / ``qsa.page8.*`` / ``qsa.page16.*``)."""
    out = set()
    for r in [*prof.rows, *prof.rules]:
        raw = r.raw
        if not r.verified or raw["cache_dtype"] not in cache_dtypes:
            continue
        if (arch is not None and raw["arch"] != arch) or (cu is not None and int(raw["cu"]) != cu):
            continue
        if page_sizes is not None and P.page_of(raw) not in page_sizes:
            continue
        out.add((raw["cache_dtype"], r.vid, tuple(sorted(dict(r.params).items()))))
    # R2-MERGE: verified buckets that pin launch knobs (kernel-native 3-row groups)
    # launch their own specialization (other buckets launch their target's variant)
    for b in prof.buckets:
        raw = b.raw
        if not b.verified or raw["cache_dtype"] not in cache_dtypes \
                or not P.split_knobs(b.vid, b.params)[1]:
            continue
        if (arch is not None and raw["arch"] != arch) or (cu is not None and int(raw["cu"]) != cu):
            continue
        if page_sizes is not None and P.page_of(raw) not in page_sizes:
            continue
        out.add((raw["cache_dtype"], b.vid, tuple(sorted(dict(b.params).items()))))
    return sorted(out)


# ---------------------------------------------------------------------------
# Persistent compile cache (R2): FlyDSL's on-disk JIT cache, scoped to our kernels
# ---------------------------------------------------------------------------
#: our knob: a directory, or ``off`` (leave FlyDSL's own default / env alone)
CACHE_DIR_ENV = "AITER_QSA_INDEX_SCORER_CACHE_DIR"
FLYDSL_CACHE_ENV = "FLYDSL_RUNTIME_CACHE_DIR"


def compile_cache_dir(kernel_sha256: str | None = None) -> str | None:
    """Directory FlyDSL persists this kernel's compiled variants in (or None = untouched).

    Precedence: ``$AITER_QSA_INDEX_SCORER_CACHE_DIR`` (``off``/``0``/``none`` = do not
    redirect) > ``$FLYDSL_RUNTIME_CACHE_DIR`` (honoured as set by the user/image) >
    ``~/.cache/aiter/qsa_index_scorer/<kernel sha256[:16]>``. A kernel source change
    gets a fresh directory (FlyDSL's own key also covers the source, this just keeps
    stale artifacts apart and easy to delete).
    """
    ours = os.environ.get(CACHE_DIR_ENV)
    if ours is not None and ours.strip().lower() in ("", "0", "off", "none", "false"):
        return None
    if ours:
        return os.path.abspath(os.path.expanduser(ours))
    if os.environ.get(FLYDSL_CACHE_ENV):
        return os.environ[FLYDSL_CACHE_ENV]
    home = os.path.expanduser("~")
    if not home or home == "~" or not os.path.isabs(home):
        return None              # no home directory: leave FlyDSL's own setting alone
    sha = (kernel_sha256 or _kernel_sha256())[:16]
    return os.path.join(home, ".cache", "aiter", "qsa_index_scorer", sha)


class _scoped_cache_dir:
    """Point ``$FLYDSL_RUNTIME_CACHE_DIR`` at ``path`` while our kernels compile.

    FlyDSL binds a JIT function to its disk-cache directory on the function's
    first compile (``_ensure_cache_manager``), so setting the variable only
    around our own first compiles scopes the directory to this kernel module
    and leaves every other FlyDSL user of the process on its own setting.
    Not re-entrant across threads that compile other FlyDSL kernels meanwhile
    (prewarm runs at model load, single-threaded by contract).
    """

    def __init__(self, path: str | None):
        self.path = path

    def __enter__(self):
        self.old = os.environ.get(FLYDSL_CACHE_ENV)
        if self.path and self.path != self.old:
            try:
                os.makedirs(self.path, exist_ok=True)
            except OSError:      # unwritable: leave FlyDSL on its own setting
                self.path = None
                return self
            os.environ[FLYDSL_CACHE_ENV] = self.path
        else:
            self.path = None
        return self

    def __exit__(self, *exc):
        if self.path:
            if self.old is None:
                os.environ.pop(FLYDSL_CACHE_ENV, None)
            else:
                os.environ[FLYDSL_CACHE_ENV] = self.old
        return False


def _pos_tag(dtype) -> str:
    if dtype is _I32:
        return "i32"
    if dtype is _I64:
        return "i64"
    raise ValueError(f"pos_dtypes: {dtype} (int32 / int64 only)")


def _aligned_zeros(shape, dtype, dev, align: int = P.CACHE_ALIGN_BYTES):
    n = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    raw = torch.zeros(n + align, dtype=_U8, device=dev)
    off = (-raw.data_ptr()) % align
    return raw[off:off + n].view(dtype).view(*shape)


def _compile_only_run(K, plan: "_Plan", exe, args) -> None:
    """No-device compile of one launch (``COMPILE_ONLY=1``, ``FLYDSL_GPU_ARCH=gfx1250``):
    the very ``flyc.compile`` the kernel's ``_run`` issues, with a null stream."""
    import flydsl.compiler as flyc   # noqa: PLC0415 -- lazy, like the kernel import
    import flydsl.expr as fx         # noqa: PLC0415
    if getattr(exe, "_cf", None) is None:
        flyc.compile(exe, *args, fx.Stream(None))


def _prewarm_one(K, prof, dev, cache_dtype: str, vid: str, params: dict, pos_dtype,
                 compile_only: bool = False) -> None:
    """Build + launch one variant on a private dummy problem (compiles it).

    ``compile_only``: CPU tensors, compile without a device or launch (timing /
    cache-population mode for no-device hosts)."""
    page = _vid_page(vid)
    v4 = "qsa.page4." + vid.split(".", 2)[2]
    rows, requests, width = _PREWARM_GEOMETRY[v4]
    for (gv, (pk, pv)), geo in _PREWARM_GEOMETRY_PARAMS.items():
        if gv == v4 and params.get(pk) == pv:
            rows, requests, width = geo
    knobs = P.split_knobs(vid, params)[1]
    if "group_rows" in knobs:          # kernel-native group bucket: one 3-row request
        rows = int(knobs["group_rows"]) * requests
    if page != P.PAGE_SIZE:
        width = max(1, width * P.PAGE_SIZE // page)   # same columns, fewer (larger) pages
    columns, pages = width * page, _PREWARM_PAGES
    is_fp8 = cache_dtype == "fp8"
    cdt = _FP8 if is_fp8 else _BF16
    q = _aligned_zeros((rows, P.INDEX_HEADS, P.HEAD_DIM), _BF16, dev)
    cache = _aligned_zeros((pages, page, 1, P.HEAD_DIM), cdt, dev)
    scale = torch.ones((pages, page, 1), dtype=_F32, device=dev) if is_fp8 else None
    got = predict_route(K, cache_dtype, rows if "group_rows" not in knobs else 4 * requests,
                        requests, columns, cache.data_ptr(), page)
    if "group_rows" in knobs and P.variant_kind(got[0]) == "decg":
        # ROUND-R2E: the kernel-native group launch's slab (fp8 g3 w32), as _decide_bucket
        w3 = K.decode_group_route(rows, requests, is_fp8, cache.data_ptr(), int(knobs["group_rows"]))
        if w3 is not None:
            got = (got[0], dict(got[1], wave_cols=int(w3)))
    if not _route_ok(*got, vid, params) or (
            "group_rows" in knobs and K.decode_group_route(
                rows, requests, is_fp8, cache.data_ptr(), int(knobs["group_rows"])) is None):
        raise RuntimeError(f"prewarm geometry {(rows, requests, width)} routes to {got}, "
                           f"not {vid} {dict(params)}")
    pt = (torch.arange(requests * width, dtype=_I32, device=dev) % pages).view(requests, width)
    t2r = torch.zeros(rows, dtype=_I32, device=dev)
    qp = torch.arange(rows, dtype=pos_dtype, device=dev) + 4 * width - rows
    cl = torch.full((requests,), 4 * width, dtype=_I32, device=dev)
    plan = _Plan(K, prof, vid, params, is_fp8, cdt, rows, requests, width, columns, pages, page)
    if plan.kind is None:
        raise RuntimeError(f"prewarm: no direct launch plan for {vid}")
    if compile_only:
        exe, args = _kernel_args(plan, q, cache, pt, t2r, qp, cl, None, scale, None, None)
        _compile_only_run(K, plan, exe, args)
        return
    _execute(plan, q, cache, pt, t2r, qp, cl, None, None, None, scale, None, None)


def prewarm_plan(profile: str | None = None, *, cache_dtypes=("bf16",),
                 pos_dtypes=(torch.int32, torch.int64), page_sizes=None,
                 compressed_k_cache=None, compressed_k_scale=None, arch: str | None = None,
                 cu: int | None = None, kernel=None) -> list[tuple[str, str, tuple, Any]]:
    """The (tag, cache dtype, params items, pos dtype) list :func:`prewarm` builds (pure;
    no device, no kernel import unless ``kernel`` is given for the page capability probe).

    Scoping (R2): a bound ``compressed_k_cache`` fixes the page size and cache dtype
    (only variants that cache can select); ``page_sizes`` restricts explicitly; page
    sizes the kernel does not implement (:func:`kernel_page_sizes`) are skipped.
    """
    prof = _resolve(profile)
    if isinstance(prof, Decision):
        return []
    if compressed_k_cache is not None:
        fmt = _cache_format(compressed_k_cache, compressed_k_scale)
        if fmt not in P.CACHE_DTYPES or compressed_k_cache.ndim != 4:
            return []
        cache_dtypes = (fmt,)
        page_sizes = (int(compressed_k_cache.shape[1]),)
    caps = set(kernel_page_sizes(kernel)) if kernel is not None else None
    tags = [(_pos_tag(d), d) for d in pos_dtypes]
    out = []
    for cache_dtype, vid, items in _profile_variants(prof, tuple(cache_dtypes), arch, cu,
                                                     None if page_sizes is None
                                                     else {int(p) for p in page_sizes}):
        if caps is not None and _vid_page(vid) not in caps:
            continue
        for ptag, pd in tags:
            out.append((f"{vid}/{cache_dtype}/{ptag}", cache_dtype, items, pd))
    return out


def prewarm_stats() -> dict:
    """Per-variant seconds of the last :func:`prewarm` call(s) (R2 report):
    ``{tag: {"seconds", "cached", "compile_only"}}``; ``cached`` = already warm in-process."""
    return {k: dict(v) for k, v in _PREWARM_STATS.items()}


def prewarm(profile: str | None = None, *, cache_dtypes=("bf16",),
            pos_dtypes=(torch.int32, torch.int64), device=None, page_sizes=None,
            compressed_k_cache=None, compressed_k_scale=None, cache_dir: str | None = None,
            jobs: int = 1, compile_only: bool = False) -> list[str]:
    """Build and JIT-compile every kernel variant a ``pass`` row/rule of the profile can select.

    For each (cache dtype in ``cache_dtypes``) x (variant) x (position dtype in
    ``pos_dtypes``): calls the kernel module's builder with exactly the
    arguments the launch path uses (same ``lru_cache`` entry) and launches it
    once on a tiny private zero problem on ``device`` (FlyDSL compiles on the
    first launch, not at build time), then synchronizes. Never touches user
    data. Idempotent (a variant already warmed for this profile/kernel/device is
    skipped). Call once at model load, before warmup / CUDA-graph capture.

    R2 (warm-cache time, all keyword-only and backward compatible):

    * ``compressed_k_cache`` (+ ``compressed_k_scale``): build only the variants that
      cache can select (its page size and dtype); ``page_sizes`` restricts by hand.
      ATOM passes int64 positions only -> ``pos_dtypes=(torch.int64,)`` halves the work.
    * persistent compile cache: FlyDSL's on-disk JIT cache, directed to
      :func:`compile_cache_dir` (``cache_dir`` overrides) while our kernels compile;
      a restart with the same kernel is a disk hit (no LLVM run).
    * ``jobs > 1``: compile variants on a thread pool (FlyDSL compiles under per-key
      file locks; see ``prewarm_time_qsa_index_scorer.py`` for the measured effect).
    * ``compile_only``: no device, no launch: CPU dummy tensors and ``flyc.compile``
      (needs ``COMPILE_ONLY=1 FLYDSL_GPU_ARCH=gfx1250``); populates the persistent cache
      and measures compile seconds on a no-device host. Device/arch checks are skipped.
    * :func:`prewarm_stats` reports per-variant seconds.

    Returns the warmed variant tags ``"<vid>/<cache dtype>/<i32|i64>"`` (sorted).
    Returns ``[]`` when the profile is off/unavailable or the device is not one
    the profile covers (calls would run native anyway).
    """
    for d in pos_dtypes:
        _pos_tag(d)
    prof = _resolve(profile)
    if isinstance(prof, Decision):
        return []
    if compile_only:
        dev = torch.device("cpu")
        arch = cu = None
    else:
        if device is None and compressed_k_cache is not None:
            device = compressed_k_cache.device
        if device is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        else:
            dev = torch.device(device)
            if dev.type == "cuda" and dev.index is None:
                dev = torch.device("cuda", torch.cuda.current_device())
        if dev.type not in _ALLOWED_DEVICE_TYPES or _static_decision(prof, dev) is not None:
            return []
    try:
        K = _load_kernel()
    except Exception:  # noqa: BLE001 -- same policy as decide: kernel unavailable => native
        return []
    if _env_diff(prof):
        return []
    if not compile_only:
        arch, cu = _device_info(dev)
    work = prewarm_plan(prof.path, cache_dtypes=cache_dtypes, pos_dtypes=pos_dtypes,
                        page_sizes=page_sizes, compressed_k_cache=compressed_k_cache,
                        compressed_k_scale=compressed_k_scale, arch=arch, cu=cu, kernel=K)
    cdir = cache_dir if cache_dir is not None else compile_cache_dir(prof.kernel_sha256)
    out = []

    def one(tag, cache_dtype, items, pos_dtype):
        key = (prof.path, prof.kernel_sha256, str(dev), tag, items, bool(compile_only))
        if key in _PREWARMED:
            _PREWARM_STATS[tag] = {"seconds": 0.0, "cached": True, "compile_only": compile_only}
            return tag
        t0 = time.perf_counter()
        _prewarm_one(K, prof, dev, cache_dtype, tag.split("/")[0], dict(items), pos_dtype,
                     compile_only)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        dt = time.perf_counter() - t0
        _PREWARMED[key] = dt
        _PREWARM_STATS[tag] = {"seconds": dt, "cached": False, "compile_only": compile_only}
        return tag

    with _PREWARM_LOCK, _scoped_cache_dir(cdir):
        todo = [w for w in work
                if (prof.path, prof.kernel_sha256, str(dev), w[0], w[2], bool(compile_only))
                not in _PREWARMED]
        if int(jobs) > 1 and len(todo) > 1 and cdir:
            # FlyDSL tracing is not thread-safe (one SharedAllocator per kernel: a thread pool
            # fails); compile in no-device worker processes into the shared persistent
            # cache, then the in-process pass below is a disk hit per variant.
            _compile_in_workers(prof.path, todo, cdir, int(jobs))
        out = [one(*w) for w in work]
    return sorted(set(out))


#: relative compile cost (recipe image, compile-only, seconds) for worker partitioning
_COMPILE_COST = {"r32_persist": 12.5, "r32": 1.8, "r32_fp8": 1.8, "r16": 1.1, "r16_fp8": 1.1,
                 "decg": 0.6, "draft": 0.2, "r32_kt": 2.0, "r32_kt_fp8": 2.0}


def _compile_in_workers(profile_path: str, work, cdir: str, jobs: int) -> list[int]:
    """Compile ``work`` in ``jobs`` no-device subprocesses (``COMPILE_ONLY=1``) sharing the
    persistent cache ``cdir``; returns their exit codes (failures are not fatal: the
    in-process pass compiles whatever is missing)."""
    import json  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    def c(w):
        return _COMPILE_COST.get(w[0].split("/")[0].split(".")[2], 1.0)

    # ONE variant per fresh worker process (measured, recipe image): FlyDSL's disk-cache
    # key of this kernel module depends on what the process compiled before (a worker
    # compiling r32 -> r16 -> decg -> draft produced decg/r32 keys the fresh in-process
    # pass then missed), while a fresh process per variant reproduces exactly the keys of
    # a sequential prewarm. Longest first; at most ``jobs`` at a time.
    queue = sorted(work, key=lambda w: -c(w))
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, COMPILE_ONLY="1", FLYDSL_RUNTIME_CACHE_DIR=cdir)
    env.setdefault("FLYDSL_GPU_ARCH", "gfx1250")
    env.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    running, codes = [], []
    while queue or running:
        while queue and len(running) < jobs:
            t, dt, items, pd = queue.pop(0)
            arg = json.dumps({"profile": profile_path, "cache_dir": cdir,
                              "work": [[t, dt, [list(x) for x in items], _pos_tag(pd)]]})
            running.append(subprocess.Popen(
                [sys.executable, "-B", os.path.join(here, "_prewarm_worker.py"), arg], env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        p = running.pop(0)
        codes.append(p.wait())
    return codes


def _compile_work(profile: str, work, cache_dir: str | None) -> list[str]:
    """Worker side of :func:`_compile_in_workers`: compile-only build of the given
    ``[tag, cache dtype, params items, i32|i64]`` entries (no device)."""
    prof = _resolve(profile)
    if isinstance(prof, Decision):
        return []
    K = _load_kernel()
    done = []
    with _scoped_cache_dir(cache_dir):
        for tag, cache_dtype, items, ptag in work:
            pd = _I64 if ptag == "i64" else _I32
            _prewarm_one(K, prof, torch.device("cpu"), cache_dtype, tag.split("/")[0],
                         {k: v for k, v in items}, pd, True)
            done.append(tag)
    return done


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
    return _execute(d._plan, q, compressed_k_cache, page_table, token_to_request,
                    query_positions, context_lens, score_divisor, max_columns, row_starts,
                    compressed_k_scale, logits, visible_groups)


def _execute_decided(d, q, compressed_k_cache, page_table, token_to_request, query_positions,
                     context_lens, compress_ratio=4, score_divisor=None, max_columns=None,
                     row_starts=None, allow_flydsl=False, compressed_k_scale=None, *,
                     logits=None, visible_groups=None, profile=None, allow_unverified=False,
                     buckets=None):
    """``decide``'s argument binding, then the shared launch (no ``inspect`` per call)."""
    return _execute(d._plan, q, compressed_k_cache, page_table, token_to_request,
                    query_positions, context_lens, score_divisor, max_columns, row_starts,
                    compressed_k_scale, logits, visible_groups)


def try_qsa_paged_mqa_logits(*args, **kwargs):
    """``(out, decision)``: ``out`` is ``(logits, visible)`` or ``None`` (run native)."""
    d = decide(*args, **kwargs)
    if not d.ok:
        return None, d
    return _execute_decided(d, *args, **kwargs), d


def _kernel_loaded() -> bool:
    """True once the kernel module (and hence FlyDSL) has been imported by this API."""
    return _KERNEL is not None or P.KERNEL_MODULE in sys.modules

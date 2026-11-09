# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Native runtime of the neutral ``sp_qkv_fused`` v1 op (``aiter.ops.gemm_fusion.sp_qkv``).

This module is the **AITER-owned native implementation** behind the public op
``aiter.sp_qkv_fused`` (see ``op.py`` and ``CONTRACT.md``).  It is deliberately
*not* a public interface: the op calls :func:`sp_qkv_native` with one flat payload
dict, and every policy that used to be caller state in the ``gemm_fusion``
framework lives here:

* **profile selection** -- environment ``AITER_GEMM_FUSION_PROFILE`` (a profile
  name under ``aiter/configs/gemm_fusion/`` or a path to a profile JSON); default
  ``minimax_m3_sp4_gfx950_v1``.  The alternate r19 profile (16K band) is reachable
  *only* this way: it is never an op argument.
* **resource session** -- one ``FusionResources`` per
  ``(device, rank, world, group id)``, created on the first eager call (or by
  :func:`open_session`) and released by :func:`close_session`.  None of the comm
  / arena / ERR-page machinery is caller state.
* **plan selection** -- the immutable profile is matched with
  ``aiter.gemm_fusion.match.match_padded``; the plan is prepared/bound once per
  matched bucket, outside graph capture.
* **call ids** -- a per-process monotonic counter, masked to the low 18 bits the
  device tag can carry (R12 tag contract).  See :func:`_next_call_id` for the
  "all ranks issue the same call sequence" assumption that makes this safe.
* **error protocol** -- the R12 per-comm ERR page is read back asynchronously
  once per call; a non-zero word raises :class:`GemmFusionDeviceError` (a typed
  ``RuntimeError``) instead of silently succeeding.

Coverage (option B, coverage-limited native path)
-------------------------------------------------
The native path serves the **fused prologue**: the ``sp_qkv.small.v1``
two-stage plans (GEMM + SP exchange + head Gemma-RMSNorm + partial NeoX RoPE +
KV/index-cache writes).  For those cells ``plan.run`` writes the contract's
``q``/``k``/``v``/``index_q`` outputs and the caches in place, exactly the
preconditions of the framework's native post-prologue stage
(``stages_native.attend_after_prologue``: attention over ``q``/``k``/``v`` with
the just-written caches).  A caller that registered ``native_stage=`` on
:func:`open_session` gets that **consumer continuation** invoked after a
successful launch, on the same stream.  It is explicitly optional and is *not
part of the op contract*: the runtime never imports ATOM, the default is
``None``, and without it the caller runs its own attention tail after the op
returns.

**Output staging (binding rule):** the runtime always owns bucket-sized
output staging and binds *those* (never the caller's tensors).  After every
launch it copies the real ``T = 4*real local M`` rows back to the caller's public
outputs on the same stream (device-side, capturable).  Consequences: caller
output buffers, views and row ranges are irrelevant to the binding -- layers may
share one bucket-sized buffer (or pass disjoint row-range views) and only the
per-call row count matters; one binding per ``(mode, bucket M)`` per session is
enough, and it covers exact buckets and pad hits alike.  The staging cost is
bounded and lazy: it exists only for matched buckets that are actually prepared
(small-band T_bucket <= 1024 rows; the Phase-2 large band is not served).

Calls outside this fused coverage -- the **one-stage** plan family
(``sp_qkv.large.v1``: GEMM + SP exchange only, whose remaining prologue/attention
is not part of this payload contract) and the small/large gaps of the profile --
raise ``aiter.gemm_fusion.match.ContractError`` listing what was searched and
what the framework coverage is.  There is **no unfused fallback** in this
runtime: option A (the unfused composition owned by native) is the named
follow-up in ``CONTRACT.md``.  DECIDED with the contract owner (Lead): option
(a) -- ``CONTRACT.md`` §8 narrows its *fused native* coverage to exactly these
prologue cells; the one-stage large/r19 band is named **Phase-2**: it needs an
aiter-owned native post-exchange prologue op (or a contract split), which is
*not* part of this runtime and must not be added here, so it stays a coverage
error.

The same rule is exposed as a cheap static query, :func:`sp_qkv_covers`: it
answers "would a fused prologue plan match this call?" without creating a
session, allocating, building or launching anything (safe from a consumer's
``__init__``).

Graph capture
-------------
Creating a session or preparing/binding a bucket is only legal outside capture;
a first call inside a capture raises ``RuntimeError``.  Once a bucket is
prepared, the same call may be captured/replayed: ``plan.run`` is graph-safe
(no Python on replay).  Pointers of the bound tensors (outputs, caches, scales,
norm weights, RoPE tables) must not move after ``prepare``; a detected move
raises instead of writing through the stale binding.

Unverified on CPU
-----------------
These tests are CPU-only with fake plans/resources.  The real GPU path (CCO
comm creation, arena registration, kernel launch, byte-exactness against the
``#6689`` golden cells) is *not* exercised here; see ``CONTRACT.md`` and the
native-parity follow-up.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "GemmFusionDeviceError",
    "check_errors",
    "close_session",
    "open_session",
    "profile_name",
    "sp_qkv_available",
    "sp_qkv_covers",
    "sp_qkv_native",
]

PROFILE_ENV = "AITER_GEMM_FUSION_PROFILE"
DEFAULT_PROFILE = "minimax_m3_sp4_gfx950_v1"
MODES = ("dense", "indexer", "reuse")

#: The exact flat payload handed over by ``op.py`` (contract argument names, in
#: signature order).  Optional tensors are present with ``None``.
PAYLOAD_KEYS = (
    "q",
    "x",
    "x_scale",
    "w",
    "w_scale",
    "q_norm_w",
    "k_norm_w",
    "cos_cache",
    "sin_cache",
    "positions",
    "slot_mapping",
    "k_cache",
    "v_cache",
    "k_scale",
    "v_scale",
    "mode",
    "k",
    "v",
    "index_q",
    "index_q_norm_w",
    "index_k_norm_w",
    "index_cache",
    "eps",
    "group",
    "stream",
)

# The covered fused cells are the small-plan prologue: 2 stages.
_PROLOGUE_STAGES = 2
_WORLD = 4
_HIDDEN_K = 6144
_HEAD_DIM = 128
_Q_HEADS = 16
_KV_HEADS = 1
_NORM_W_ELEMS = 128
_ROTARY_PAIRS = 32
_N = {"dense": 9216, "indexer": 9856, "reuse": 9856}
_KV_PAGE = {"dense": 128, "indexer": 16, "reuse": 16}

# R12 tag: the device image can carry only the LOW call_id bits in the handshake
# tag (14-bit m_real + 18-bit call_id for the r19 ABI-7 header; the older large
# headers use 19 bits, so 18 is the common safe width).
_TAG_CALLID_BITS = 18
_TAG_CALLID_MASK = (1 << _TAG_CALLID_BITS) - 1


class GemmFusionDeviceError(RuntimeError):
    """A non-zero R12 per-comm ERR word was observed by the native runtime.

    Raised from the next :func:`sp_qkv_native` call (or :func:`check_errors` /
    :func:`close_session`) for the step whose asynchronous readback returned a
    non-zero word.  The fused kernels are fail-stop; the runtime never retries
    and never hides this as a "success".
    """


@dataclass
class _Bucket:
    """One prepared (plan, bucket) binding of a session."""

    plan: Any
    n_stages: int
    bucket_m: int
    prepared: bool = False
    outs: dict[str, Any] = field(default_factory=dict)
    staging: Any = None
    abi: int = 3
    bound_ptrs: tuple = ()
    prepared_once: bool = False


@dataclass
class _Session:
    """Per-``(device, rank, world, group)`` resource session."""

    key: tuple
    rank: int
    world: int
    device: int
    group: Any
    profile: Any
    profile_name: str
    res: Any
    native_stage: Callable[..., Any] | None = None
    buckets: dict = field(default_factory=dict)
    readback_seq: int = 0
    readback_done: int = 0


_SESSIONS: dict[tuple, _Session] = {}
_LOCK = threading.RLock()
_CALL_ID_COUNTER = 0
_CALL_ID_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Lazy environment seams (kept module-level so CPU tests can replace them)
# ---------------------------------------------------------------------------
def _gf() -> Any:
    """``aiter.gemm_fusion`` (imported lazily: keep this module import-light)."""

    from aiter import gemm_fusion

    return gemm_fusion


def _detect_arch() -> str:
    return _gf().detect_arch()


def _detect_cu() -> int:
    return int(_gf().detect_cu())


def _detect_capabilities() -> frozenset:
    return frozenset(_gf().detect_capabilities())


def _load_profile(name: str) -> Any:
    return _gf().load_profile(name)


def _match_padded(spec: Any, ctx: Any, profile: Any) -> tuple:
    return _gf().match_padded(spec, ctx, profile)


def _contract_error() -> type:
    return _gf().ContractError


def _spec_for(mode: str, world: int, eps: float) -> Any:
    from . import specs

    return specs.sp_qkv_spec(mode, world=world, eps=eps)


def _world_rank(group: Any) -> tuple[int, int]:
    """(rank, world) of ``group`` (or of the default process group when None)."""

    import torch.distributed as dist

    if group is None:
        return int(dist.get_rank()), int(dist.get_world_size())
    return int(dist.get_rank(group)), int(dist.get_world_size(group))


def _create_resources(
    rank: int,
    world: int,
    group: Any,
    device: int,
    caps: frozenset,
    profile: Any,
) -> Any:
    """Create the CCO resources for one session (outside graph capture)."""

    import torch.distributed as dist

    from aiter.gemm_fusion.resources import FusionResources

    return FusionResources(
        rank,
        world,
        group=group,
        dist=dist,
        device=device,
        caps=caps,
        profile=profile,
    )


def _is_capturing() -> bool:
    """True while a CUDA graph capture is active on the current stream."""

    try:
        import torch

        return bool(
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
    except Exception:  # noqa: BLE001 - no torch / no GPU: eager
        return False


def _new_tensor(shape: tuple, dtype_name: str, device: Any) -> Any:
    """Allocate a staging tensor (lazily imports torch)."""

    import torch

    dtype = getattr(torch, dtype_name)
    return torch.empty(shape, dtype=dtype, device=device)


def _copy_into(dst: Any, src: Any) -> None:
    dst.copy_(src)


# ---------------------------------------------------------------------------
# Payload validation (duck-typed: CPU tests use lightweight fake tensors)
# ---------------------------------------------------------------------------
def _shape(t: Any) -> tuple:
    return tuple(int(v) for v in getattr(t, "shape", ()))


def _dtype_name(t: Any) -> str:
    d = getattr(t, "dtype", None)
    name = getattr(d, "__name__", None)
    if name:
        return str(name)
    return str(d).rsplit(".", 1)[-1]


def _numel(t: Any) -> int:
    if hasattr(t, "numel"):
        return int(t.numel())
    n = 1
    for d in _shape(t):
        n *= d
    return n


def _is_contiguous(t: Any) -> bool:
    fn = getattr(t, "is_contiguous", None)
    return True if fn is None else bool(fn())


def _stride(t: Any, dim: int) -> int | None:
    fn = getattr(t, "stride", None)
    if fn is None:
        return None
    try:
        return int(fn(dim))
    except Exception:  # noqa: BLE001 - fakes without strides
        return None


def _fail(msg: str) -> None:
    raise _contract_error()(f"sp_qkv_fused native runtime: {msg}")


def _require(t: Any, name: str) -> None:
    if t is None:
        _fail(
            f"payload[{name!r}] is None; the fused native path needs the tensor "
            "(mode-inconsistent payloads are rejected by op.py before this point)"
        )


def _check_dtype(t: Any, name: str, want: str) -> None:
    got = _dtype_name(t)
    if got != want:
        _fail(f"payload[{name!r}] dtype {got!r} is not {want!r}")


def _check_dim(t: Any, name: str, dim: int) -> None:
    got = len(_shape(t))
    if got != dim:
        _fail(f"payload[{name!r}] must be {dim}-D, got shape {_shape(t)}")


def _validate_payload(payload: Any) -> tuple[str, int]:
    """Validate the flat contract payload; returns ``(mode, M)``.

    Structural problems (unknown/missing keys, bad ``mode``) raise ``ValueError``
    (caller/op bugs); tensor contract violations raise ``ContractError`` (the
    native-path contract of ``CONTRACT.md`` §3/§7).
    """

    if not isinstance(payload, Mapping):
        raise TypeError(f"sp_qkv_native expects a payload dict, got {type(payload)!r}")
    missing = [k for k in PAYLOAD_KEYS if k not in payload]
    if missing:
        raise ValueError(f"sp_qkv native payload is missing {missing}")
    extra = [k for k in payload if k not in PAYLOAD_KEYS]
    if extra:
        raise ValueError(
            f"sp_qkv native payload has unknown keys {extra} (allowed: {list(PAYLOAD_KEYS)})"
        )
    mode = payload["mode"]
    if mode not in MODES:
        raise ValueError(f"payload mode must be one of {MODES}, got {mode!r}")

    for name in ("q", "x", "x_scale", "w", "w_scale", "q_norm_w", "k_norm_w"):
        _require(payload[name], name)
    # x / M
    x = payload["x"]
    _check_dim(x, "x", 2)
    if _shape(x)[1] != _HIDDEN_K:
        _fail(f"payload['x'] K={_shape(x)[1]} != {_HIDDEN_K} (hardwired)")
    _check_dtype(x, "x", "float8_e4m3fn")
    m = int(_shape(x)[0])
    if m < 1:
        _fail("payload['x'] has no rows (local M must be >= 1)")
    t_rows = _WORLD * m

    if len(_shape(payload["x_scale"])) not in (1, 2) or _numel(payload["x_scale"]) != m:
        _fail(
            "payload['x_scale'] must be fp32 with exactly M elements ([M] or [M,1]), "
            f"got shape {_shape(payload['x_scale'])}"
        )
    _check_dtype(payload["x_scale"], "x_scale", "float32")

    _check_dim(payload["w"], "w", 2)
    if _shape(payload["w"])[1] != _HIDDEN_K:
        _fail(f"payload['w'] K={_shape(payload['w'])[1]} != {_HIDDEN_K}")
    if _shape(payload["w"])[0] < _N[mode]:
        _fail(
            f"payload['w'] rows {_shape(payload['w'])[0]} < N={_N[mode]} "
            f"(the kernels read an N-row prefix of the preshuffled weight)"
        )
    if not _is_contiguous(payload["w"]):
        _fail("payload['w'] must be contiguous ((16,16)-preshuffled weights)")
    _check_dtype(payload["w"], "w", "float8_e4m3fn")

    w_scale = payload["w_scale"]
    if len(_shape(w_scale)) not in (1, 2) or _numel(w_scale) < _N[mode]:
        _fail(
            f"payload['w_scale'] must be fp32 with >= N={_N[mode]} rows, got "
            f"shape {_shape(w_scale)}"
        )
    _check_dtype(w_scale, "w_scale", "float32")

    for name in ("q_norm_w", "k_norm_w"):
        _check_dim(payload[name], name, 1)
        if _numel(payload[name]) != _NORM_W_ELEMS:
            _fail(f"payload[{name!r}] must have {_NORM_W_ELEMS} elements")
        _check_dtype(payload[name], name, "float32")

    for name in ("cos_cache", "sin_cache"):
        _check_dim(payload[name], name, 2)
        if _shape(payload[name])[1] != _ROTARY_PAIRS:
            _fail(
                f"payload[{name!r}] must be [max_pos, {_ROTARY_PAIRS}], got "
                f"{_shape(payload[name])}"
            )
        _check_dtype(payload[name], name, "bfloat16")

    for name in ("positions", "slot_mapping"):
        _check_dim(payload[name], name, 1)
        if _numel(payload[name]) < 1:
            _fail(f"payload[{name!r}] has no entries")
        _check_dtype(payload[name], name, "int64")

    # the local token count must be a whole number of SP rows
    if m * _WORLD != t_rows:  # pragma: no cover - definitionally true
        _fail("internal: T is not 4*M")

    # outputs / mode-specific tensors
    _validate_output(payload["q"], "q", t_rows, _Q_HEADS)
    if mode == "indexer":
        for name in (
            "index_q",
            "index_q_norm_w",
            "index_k_norm_w",
            "index_cache",
        ):
            _require(payload[name], name)
        _validate_output(payload["index_q"], "index_q", t_rows, _KV_HEADS)
        for name, want in (
            ("index_q_norm_w", None),
            ("index_k_norm_w", None),
        ):
            t = payload[name]
            _check_dim(t, name, 1)
            if _numel(t) != _NORM_W_ELEMS:
                _fail(f"payload[{name!r}] must have {_NORM_W_ELEMS} elements")
            _check_dtype(t, name, "float32")
        _check_dtype(payload["index_cache"], "index_cache", "float8_e4m3fn")
        if _numel(payload["index_cache"]) % _HEAD_DIM:
            _fail(
                f"payload['index_cache'] numel {_numel(payload['index_cache'])} is not "
                f"a multiple of {_HEAD_DIM} (flat page-128 layout)"
            )
    else:
        for name in ("k", "v"):
            _require(payload[name], name)
            _validate_output(payload[name], name, t_rows, _KV_HEADS)

    # caches (the kernels write through raw pointers: a layout mismatch is a
    # silent wrong write, so reject it before any binding)
    for name in ("k_cache", "v_cache"):
        t = payload[name]
        _require(t, name)
        _check_dtype(t, name, "float8_e4m3fn")
        if len(_shape(t)) != 5:
            _fail(
                f"payload[{name!r}] must be the 5-D golden SHUFFLE cache, got "
                f"shape {_shape(t)}"
            )
        page_dim = 3 if name == "k_cache" else 2
        if _shape(t)[page_dim] != _KV_PAGE[mode]:
            _fail(
                f"payload[{name!r}] page {_shape(t)[page_dim]} != {_KV_PAGE[mode]} "
                f"for mode {mode!r}"
            )
    for name in ("k_scale", "v_scale"):
        t = payload[name]
        if t is None:
            _fail(f"payload[{name!r}] is None: the fused cells are fp8 caches")
        _check_dtype(t, name, "float32")
        if _numel(t) < 1:
            _fail(f"payload[{name!r}] has no elements")

    eps = payload["eps"]
    if not isinstance(eps, (int, float)) or not float(eps) > 0:
        raise ValueError(f"payload eps must be a positive float, got {eps!r}")
    return str(mode), m


def _validate_output(t: Any, name: str, t_rows: int, heads: int) -> None:
    _check_dim(t, name, 3)
    shape = _shape(t)
    if shape[0] < t_rows or shape[1:] != (heads, _HEAD_DIM):
        _fail(
            f"payload[{name!r}] must be bf16 [>=T={t_rows}, {heads}, {_HEAD_DIM}], "
            f"got shape {shape}"
        )
    _check_dtype(t, name, "bfloat16")


# ---------------------------------------------------------------------------
# Profile / session lifecycle
# ---------------------------------------------------------------------------
def profile_name() -> str:
    """The active profile name/path: ``AITER_GEMM_FUSION_PROFILE`` or the default."""

    return os.environ.get(PROFILE_ENV) or DEFAULT_PROFILE


def _match_ctx(
    *, arch: str, cu: int, world: int, m: int, caps: frozenset, graph_mode: str
) -> Any:
    """Build the planner's ``MatchCtx`` (pure: no device/resource state)."""

    return _gf().MatchCtx(
        arch=arch,
        cu=cu,
        world=world,
        m=m,
        available=caps,
        graph_mode=graph_mode,
    )


def sp_qkv_available() -> bool:
    """Whether the native fused path is usable in this process (CPU-checkable).

    True when the framework is importable and the selected profile loads.  It
    does not touch a device, MORI or the compiled kernels: a first call may
    still fail at resource/kernel setup, which is reported as an exception.
    """

    try:
        _load_profile(profile_name())
        _gf()
    except Exception:  # noqa: BLE001 - unavailability is a query, not an error
        return False
    return True


def sp_qkv_covers(payload: Mapping) -> bool:
    """Static, session-independent coverage query for the neutral op contract.

    Answers exactly the planner's question -- "would a fused prologue plan match
    this call?" -- using the same payload validation, spec, ``MatchCtx``,
    profile and ``match_padded`` as :func:`sp_qkv_native`, but with **no
    resource creation, no allocation, no build and no launch**, and without
    touching the session cache.  It is safe to call from a consumer's
    ``__init__`` and on every call: it **never raises** -- a missing framework/
    MORI/profile, an invalid payload or an uncovered call returns ``False``.
    ``True`` means "the planner would match a prologue plan"; the actual call can
    still fail later at resource/kernel setup.

    Tensor placeholders are enough: this query reads only shape/dtype/numel and
    ``w.is_contiguous`` from the payload -- never data, device or pointers -- so
    the per-call tensors may be structurally valid meta/empty tensors while the
    bound weights/norm-weights/caches are real.  What it *does* need is a live
    GPU/ROCm/MORI context (``_detect_arch``/``_detect_cu``/``_detect_capabilities``)
    and a real SP ``group`` (or the initialized default process group).  The
    answer is for the row count of ``x``: pass a placeholder whose first dimension
    is the local M being asked about (>= 1; a 0-row placeholder is rejected), or
    query distinct M/buckets -- a placeholder sized only to the deployment's
    largest local rows answers that largest M, not every smaller call.
    """

    try:
        if not isinstance(payload, Mapping):
            return False
        missing = [k for k in PAYLOAD_KEYS if k not in payload]
        extra = [k for k in payload if k not in PAYLOAD_KEYS]
        if missing or extra:
            return False
        mode = payload["mode"]
        if mode not in MODES:
            return False
        _validate_payload(payload)  # structural + tensor contract; ffi-free
        _, world = _world_rank(payload["group"])
        if int(world) != _WORLD:
            return False
        profile = _load_profile(profile_name())
        m = int(_shape(payload["x"])[0])
        spec = _spec_for(mode, int(world), float(payload["eps"]))
        ctx = _match_ctx(
            arch=_detect_arch(),
            cu=_detect_cu(),
            world=int(world),
            m=m,
            caps=_detect_capabilities(),
            graph_mode="eager",
        )
        plan, n_stages, bucket_m = _match_padded(spec, ctx, profile)
        if plan is None or int(n_stages) < _PROLOGUE_STAGES:
            return False
        return not (
            bucket_m is not None
            and int(bucket_m) != m
            and int(getattr(plan, "abi", 3)) < 3
        )
    except Exception:  # noqa: BLE001 - a coverage query must never raise
        return False


def _device_index_of(payload: Mapping) -> int:
    for name in ("q", "x"):
        t = payload.get(name)
        dev = getattr(t, "device", None)
        if dev is None:
            continue
        idx = getattr(dev, "index", None)
        if isinstance(idx, int):
            return int(idx)
        text = str(dev)
        suffix = text.rsplit(":", 1)[-1]
        if suffix.isdigit():
            return int(suffix)
    return 0


def _next_call_id() -> int:
    """The per-process, rank-uniform call id handed to the launch.

    A single monotonic counter, masked to the low 18 bits the R12 tag carries
    (non-zero).  The counter is per *process*: every rank must issue the same
    call sequence (the framework contract already required "all ranks run
    identical launch counts"; the same sequence follows from a rank-uniform
    control flow -- no per-rank conditionals around fused calls).  The mask only
    needs distinct low bits within one device epoch: distinct ids within any
    contiguous 2^18 calls are guaranteed, and the device STEP word separates
    epochs.
    """

    global _CALL_ID_COUNTER
    with _CALL_ID_LOCK:
        _CALL_ID_COUNTER += 1
        value = (_CALL_ID_COUNTER - 1) & _TAG_CALLID_MASK
    return value + 1


def _session_key(group: Any, device: int) -> tuple:
    rank, world = _world_rank(group)
    group_id = 0 if group is None else id(group)
    return (int(device), int(rank), int(world), int(group_id))


def _default_device() -> int:
    """The current HIP device index (0 without torch/GPU: CPU tests)."""

    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
    except Exception:  # noqa: BLE001, S110 - no torch / no GPU
        pass
    return 0


def _get_session(
    payload: Mapping,
    group: Any,
    device: int | None,
    *,
    create: bool,
    native_stage: Callable[..., Any] | None = None,
) -> _Session | None:
    dev = _device_index_of(payload) if device is None else int(device)
    rank, world = _world_rank(group)
    key = _session_key(group, dev)
    with _LOCK:
        sess = _SESSIONS.get(key)
        if sess is not None:
            if native_stage is not None and sess.native_stage is None:
                sess.native_stage = native_stage
            return sess
        if not create:
            return None
        if _is_capturing():
            raise RuntimeError(
                "sp_qkv native runtime: no FusionResources session for "
                f"(device={dev}, rank={rank}, world={world}) and a graph capture is "
                "active. Sessions and plan bindings must be created outside capture: "
                "call sp_qkv_native once eagerly (or open_session) before capturing."
            )
        name = profile_name()
        profile = _load_profile(name)
        res = _create_resources(
            rank, world, group, dev, _detect_capabilities(), profile
        )
        sess = _Session(
            key=key,
            rank=rank,
            world=world,
            device=dev,
            group=group,
            profile=profile,
            profile_name=name,
            res=res,
            native_stage=native_stage,
        )
        _SESSIONS[key] = sess
        return sess


def open_session(
    *,
    group: Any = None,
    device: int | None = None,
    native_stage: Callable[..., Any] | None = None,
) -> _Session:
    """Create (or fetch) the resource session for ``(device, rank, world, group)``.

    Optional warm-up hook for frameworks: creating the CCO comm/arena is heavy
    and collective, so doing it here (outside capture, all ranks in the same
    order) keeps the first fused call cheap.  ``native_stage`` is an explicitly
    optional **consumer continuation**, not part of the op contract: a callback
    invoked after a successful prologue launch -- the caller's native
    post-prologue stage (the framework's ``attend_after_prologue`` semantics:
    attention over ``q``/``k``/``v`` with the caches the launch just wrote).  The
    runtime does not import ATOM and defaults to no callback.
    """

    return _get_session(
        {},
        group,
        _default_device() if device is None else int(device),
        create=True,
        native_stage=native_stage,
    )  # type: ignore[return-value]


def close_session(*, group: Any = None, device: int | None = None) -> None:
    """Release the session of ``(device, rank, world, group)`` (idempotent).

    Pending ERR readbacks are consumed first (a latched device error surfaces
    here rather than being lost); release is refused during graph capture.
    """

    rank, world = _world_rank(group)
    keys = []
    with _LOCK:
        for key, sess in list(_SESSIONS.items()):
            if key[1] == rank and key[2] == world:
                if device is not None and key[0] != int(device):
                    continue
                if (group is None and key[3] != 0) or (
                    group is not None and key[3] != id(group)
                ):
                    continue
                keys.append(key)
    if not keys:
        return
    if _is_capturing():
        raise RuntimeError(
            "sp_qkv native runtime: close_session is not allowed during graph capture"
        )
    for key in keys:
        with _LOCK:
            sess = _SESSIONS.pop(key, None)
        if sess is None:  # pragma: no cover - single owner by lock
            continue
        _check_errors(sess)
        release = getattr(sess.res, "release", None)
        if callable(release):
            release()


def _reset_for_testing() -> None:
    """Drop the module cache/counter (tests only; never for production code)."""

    global _CALL_ID_COUNTER
    with _LOCK:
        _SESSIONS.clear()
    with _CALL_ID_LOCK:
        _CALL_ID_COUNTER = 0


# ---------------------------------------------------------------------------
# ERR-page readback
# ---------------------------------------------------------------------------
def _describe_err(word: int) -> str:
    try:
        from aiter.gemm_fusion.errpage import decode_err

        info = decode_err(int(word))
        return f"ERR word 0x{int(word):08x} ({info.describe()})"
    except Exception:  # noqa: BLE001 - diagnostics must not mask the error
        return f"ERR word 0x{int(word):08x}"


def _check_errors(session: _Session) -> None:
    """Consume the pending asynchronous ERR readback; raise on a non-zero word."""

    res = session.res
    consume = getattr(res, "readback_consume", None)
    if not callable(consume) or session.readback_seq <= session.readback_done:
        return
    word = consume(session.readback_done)
    if word is None:
        return  # the copy has not completed yet; checked again on the next call
    session.readback_done += 1
    if word:
        raise GemmFusionDeviceError(
            f"sp_qkv native runtime: rank {session.rank} observed {_describe_err(word)} "
            f"for the step completed by the previous sp_qkv_fused call "
            f"(profile={session.profile_name!r}); the fused kernels are fail-stop and "
            "the runtime never retries. Recover with the collective reset protocol "
            "(the caller owns the engine-level decision)."
        )


def _enqueue_readback(session: _Session, plan: Any, stream: Any) -> None:
    """Enqueue one async ERR-word readback after a successful launch (R12 plans)."""

    if int(getattr(plan, "abi", 3)) < 3:
        return  # pre-R12 plans have no per-comm ERR page
    res = session.res
    enqueue = getattr(res, "readback_enqueue", None)
    if not callable(enqueue):
        return
    enqueue(session.readback_seq, stream)
    session.readback_seq += 1


def check_errors(*, group: Any = None, device: int | None = None) -> None:
    """Consume the pending ERR readback of the session (typed error on failure).

    The per-call path already checks before every launch; this helper lets a
    caller surface the *last* call's error without issuing another op call.
    """

    rank, world = _world_rank(group)
    with _LOCK:
        sessions = [
            sess
            for key, sess in _SESSIONS.items()
            if key[1] == rank
            and key[2] == world
            and (device is None or key[0] == int(device))
            and (
                (group is None and key[3] == 0)
                or (group is not None and key[3] == id(group))
            )
        ]
    for sess in sessions:
        _check_errors(sess)


# ---------------------------------------------------------------------------
# Prepare / run
# ---------------------------------------------------------------------------
def _outs_of(payload: Mapping) -> dict:
    mode = payload["mode"]
    if mode == "indexer":
        return {"q": payload["q"], "index_q": payload["index_q"]}
    return {"q": payload["q"], "k": payload["k"], "v": payload["v"]}


def _staging_outs(payload: Mapping, bucket_m: int) -> dict:
    """Bucket-sized runtime output staging, always the bound output tensors.

    The plan's geometry is the *bucket* M (a launch with a smaller real m still
    writes up to the bucket's T rows), while the public contract's outputs are
    caller-sized (T = 4*real M).  The runtime binds these buffers once per
    ``(mode, bucket)`` (outside capture) and copies the real rows back to the
    caller's tensors after every launch (device-side, capturable), so caller
    output buffers/views/row ranges never take part in the binding.
    """

    dev = getattr(payload["q"], "device", None)
    t_rows = _WORLD * bucket_m
    if payload["mode"] == "indexer":
        return {
            "q": _new_tensor((t_rows, _Q_HEADS, _HEAD_DIM), "bfloat16", dev),
            "index_q": _new_tensor((t_rows, _KV_HEADS, _HEAD_DIM), "bfloat16", dev),
        }
    return {
        "q": _new_tensor((t_rows, _Q_HEADS, _HEAD_DIM), "bfloat16", dev),
        "k": _new_tensor((t_rows, _KV_HEADS, _HEAD_DIM), "bfloat16", dev),
        "v": _new_tensor((t_rows, _KV_HEADS, _HEAD_DIM), "bfloat16", dev),
    }


def _copy_rows(dst: Any, src: Any, rows: int) -> None:
    """Copy the real output rows of a launch back to a caller output tensor."""

    dst[:rows].copy_(src[:rows])


def _meta_of(payload: Mapping, staging: Any = None) -> dict:
    """``Plan.run(meta=...)``: the framework's sp_qkv key contract."""

    positions = payload["positions"] if staging is None else staging["pos"]
    slot_mapping = payload["slot_mapping"] if staging is None else staging["slot"]
    meta = {
        "positions": positions,
        "slot_mapping": slot_mapping,
        "k_cache": payload["k_cache"],
        "v_cache": payload["v_cache"],
        "k_scale": payload["k_scale"],
        "v_scale": payload["v_scale"],
        "cos_cache": payload["cos_cache"],
        "sin_cache": payload["sin_cache"],
        "q_norm_w": payload["q_norm_w"],
        "k_norm_w": payload["k_norm_w"],
    }
    if payload["mode"] == "indexer":
        meta["index_cache"] = payload["index_cache"]
        meta["index_q_norm_w"] = payload["index_q_norm_w"]
        meta["index_k_norm_w"] = payload["index_k_norm_w"]
    return meta


def _make_staging(payload: Mapping, bucket_m: int) -> dict:
    """Persistent staging tensors for plans that bind the small per-call inputs.

    ABI < 3 plans (r17/r18) bind ``x_scale``/``positions``/``slot_mapping`` by
    pointer, so the runtime owns address-stable buffers and copies into them each
    call (device-side, capturable).  ABI >= 3 plans (r19) take those as
    per-launch kernargs; no staging is needed.
    """

    dev = getattr(payload["x"], "device", None)
    xs = _new_tensor((bucket_m, 1), "float32", dev)
    pos = _new_tensor((_WORLD * bucket_m,), "int64", dev)
    slot = _new_tensor((_WORLD * bucket_m,), "int64", dev)
    pos.zero_()
    slot.fill_(-1)
    return {"xs": xs, "pos": pos, "slot": slot, "m": bucket_m}


def _make_prepare_x(payload: Mapping, bucket_m: int) -> Any:
    """The bucket-sized activation passed to ``prepare`` (bindings bake in the
    bucket M: the plan's own bucket may be larger than this call's real local M).
    """

    if int(_shape(payload["x"])[0]) == bucket_m:
        return payload["x"]
    return _new_tensor(
        (bucket_m, _HIDDEN_K),
        "float8_e4m3fn",
        getattr(payload["x"], "device", None),
    )


def _stage_inputs(staging: dict, payload: Mapping) -> None:
    """Copy the per-call token arrays into the bound staging buffers."""

    n = int(_numel(payload["positions"]))
    t_rows = _WORLD * int(staging["m"])
    _copy_into(staging["xs"], payload["x_scale"])
    _copy_into(staging["pos"][:n], payload["positions"][:n])
    _copy_into(staging["slot"][:n], payload["slot_mapping"][:n])
    if n < t_rows:
        staging["pos"][n:].zero_()
        staging["slot"][n:].fill_(-1)


def _ptr_tuple(
    payload: Mapping, *, include_outputs: bool = True, outs: Any = None
) -> tuple:
    """Pointers of every tensor a plan binds (detect a moved binding).

    ``include_outputs=False`` checks the staging outputs a padded bucket binds
    instead of the caller's (caller-sized) output tensors.
    """

    names = (
        "w_scale",
        "k_cache",
        "v_cache",
        "k_scale",
        "v_scale",
        "q_norm_w",
        "k_norm_w",
        "cos_cache",
        "sin_cache",
        "index_q_norm_w",
        "index_k_norm_w",
        "index_cache",
    )
    out = []
    if include_outputs:
        names = ("q", "k", "v", "index_q") + names
    else:
        for name in ("q", "k", "v", "index_q"):
            t = (outs or {}).get(name)
            fn = getattr(t, "data_ptr", None)
            out.append(0 if fn is None else int(fn()))
    for name in names:
        t = payload.get(name)
        fn = getattr(t, "data_ptr", None)
        out.append(0 if fn is None else int(fn()))
    return tuple(out)


def _bound(res: Any) -> Any:
    return getattr(res, "bound", res)


def _prepare_bucket(
    session: _Session,
    plan: Any,
    n_stages: int,
    bucket_m: int,
    payload: Mapping,
    m: int,
) -> _Bucket:
    """Prepare/bind one bucket outside graph capture (idempotent per bucket)."""

    if _is_capturing():
        raise RuntimeError(
            "sp_qkv native runtime: the first call for this bucket reached graph "
            "capture before prepare/bind. The fused plan binds pointers outside "
            "capture; call this cell once eagerly (or open_session + a warm-up call) "
            "before capturing."
        )
    log_finalize = getattr(session.res, "finalize", None)
    if callable(log_finalize):
        log_finalize()
    abi = int(getattr(plan, "abi", 3))
    staging = None
    if abi < 3:
        staging = _make_staging(payload, bucket_m)
        _stage_inputs(staging, payload)
    if m > bucket_m:  # pragma: no cover - match_padded never returns this
        raise RuntimeError(
            f"sp_qkv native runtime: real local M={m} exceeds the matched bucket "
            f"M={bucket_m}"
        )
    bind_outs = _staging_outs(payload, bucket_m)
    meta = _meta_of(payload, staging)
    prep = getattr(plan, "prepare", None)
    if callable(prep):
        prep(
            x=_make_prepare_x(payload, bucket_m),
            x_scale=staging["xs"] if staging is not None else payload["x_scale"],
            w_scale=payload["w_scale"],
            outs=bind_outs,
            meta=meta,
            res=_bound(session.res),
        )
    bucket = _Bucket(
        plan=plan,
        n_stages=n_stages,
        bucket_m=bucket_m,
        prepared=True,
        outs=bind_outs,
        staging=staging,
        abi=abi,
        bound_ptrs=_ptr_tuple(payload, include_outputs=False, outs=bind_outs),
        prepared_once=True,
    )
    return bucket


def _coverage_message(
    *,
    session: _Session,
    mode: str,
    m: int,
    arch: str,
    cu: int,
    world: int,
    caps: frozenset,
    extra: str = "",
) -> str:
    return (
        "sp_qkv_fused v1 native runtime is coverage-limited (CONTRACT.md §8, "
        "option B): no fused prologue plan covers this call; there is no unfused "
        f"fallback. searched profile={session.profile_name!r} mode={mode!r} "
        f"local M={m} (T={_WORLD * m}) arch={arch!r} cu={cu} world={world} "
        f"available_capabilities={sorted(caps)}." + (f" {extra}" if extra else "")
    )


def _run_plan(
    session: _Session,
    bucket: _Bucket,
    payload: Mapping,
) -> dict:
    """Launch the prepared plan; returns the meta used for the native stage."""

    if bucket.abi < 3:
        staging = bucket.staging
        assert staging is not None
        _stage_inputs(staging, payload)
        x_scale = staging["xs"]
        meta = _meta_of(payload, staging)
    else:
        x_scale = payload["x_scale"]
        meta = _meta_of(payload, None)
    call_id = _next_call_id()
    bucket.plan.run(
        x=payload["x"],
        x_scale=x_scale,
        w=payload["w"],
        w_scale=payload["w_scale"],
        outs=bucket.outs,
        meta=meta,
        res=_bound(session.res),
        stream=payload["stream"],
        call_id=call_id,
    )
    rows = _WORLD * int(_shape(payload["x"])[0])
    if rows > _WORLD * bucket.bucket_m:  # pragma: no cover - prepare guard covers it
        raise RuntimeError(
            f"sp_qkv native runtime: real T={rows} exceeds the bucket's "
            f"T={_WORLD * bucket.bucket_m}"
        )
    for name, dst in _outs_of(payload).items():
        _copy_rows(dst, bucket.outs[name], rows)
    return meta


def _run_native_stage(
    session: _Session,
    payload: Mapping,
    bucket: _Bucket,
    meta: dict,
) -> None:
    """Invoke the optional caller-registered consumer continuation.

    This is explicitly *not part of the op contract*: it is the seam for the
    framework's ``stages_native.attend_after_prologue``,
    it consumes exactly the tensors the fused prologue just wrote (``q``/``k``/
    ``v``/``index_q`` and the caches) on the same stream.  The native runtime
    cannot import ATOM, so without a registered callback the caller runs its own
    attention tail after :func:`sp_qkv_native` returns.  The callback receives
    the bucket-sized runtime staging outputs (the first ``T = 4*real local M``
    rows are the real ones); the copy-back to the caller's public outputs has
    already been enqueued on the same stream.
    """

    callback = session.native_stage
    if callback is None:
        return
    callback(
        payload=payload, outs=dict(bucket.outs), meta=meta, stream=payload["stream"]
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def sp_qkv_native(payload: Mapping) -> None:
    """Execute one fused SP4 QKV prologue step (the native ``sp_qkv_fused`` path).

    The payload keys are exactly :data:`PAYLOAD_KEYS` (contract names, signature
    order; optional tensors present with ``None``).  Returns ``None``; every
    output/cache tensor is written in place by the kernels.

    Raises ``ValueError`` for a malformed payload; ``ContractError`` for a tensor
    contract violation, an uncovered call (coverage-limited native path), or a
    moved binding; ``GemmFusionDeviceError`` when the R12 ERR page reports a
    device-side failure; ``RuntimeError`` for a prepare/bind attempted inside
    graph capture.
    """

    mode, m = _validate_payload(payload)
    group = payload["group"]
    capturing = _is_capturing()
    session = _get_session(payload, group, None, create=True)
    assert session is not None
    if session.world != _WORLD:
        _fail(
            f"SP world {session.world} is not covered: sp_qkv_fused v1 is certified "
            f"for world={_WORLD} (CONTRACT.md §8)"
        )

    spec = _spec_for(mode, session.world, float(payload["eps"]))
    arch = _detect_arch()
    cu = _detect_cu()
    caps = _detect_capabilities()
    ctx = _match_ctx(
        arch=arch,
        cu=cu,
        world=session.world,
        m=m,
        caps=caps,
        graph_mode="capture" if capturing else "eager",
    )
    plan, n_stages, bucket_m = _match_padded(spec, ctx, session.profile)
    if plan is None:
        raise _contract_error()(
            _coverage_message(
                session=session,
                mode=mode,
                m=m,
                arch=arch,
                cu=cu,
                world=session.world,
                caps=caps,
            )
        )
    if int(n_stages) < _PROLOGUE_STAGES:
        raise _contract_error()(
            _coverage_message(
                session=session,
                mode=mode,
                m=m,
                arch=arch,
                cu=cu,
                world=session.world,
                caps=caps,
                extra=(
                    f"The matched plan {getattr(plan, 'vid', type(plan).__name__)!r} "
                    f"covers only {int(n_stages)} stage (GEMM + SP exchange): this "
                    "payload contract carries the prologue outputs q/k/v/index_q and "
                    "the cache writes. The one-stage large/16K band is Phase-2: it "
                    "needs an aiter-owned native post-exchange prologue op (or a "
                    "contract split), which is not part of this runtime, so it is "
                    "not served by the native prologue path."
                ),
            )
        )

    if bucket_m is not None and int(bucket_m) != m and int(getattr(plan, "abi", 3)) < 3:
        raise _contract_error()(
            _coverage_message(
                session=session,
                mode=mode,
                m=m,
                arch=arch,
                cu=cu,
                world=session.world,
                caps=caps,
                extra=(
                    f"The matched plan is the pad bucket M={int(bucket_m)} (real local "
                    f"M={m}) of a legacy (ABI < 3) source; pad-M launches are only "
                    "supported by the per-launch-kernarg (ABI >= 3) plans."
                ),
            )
        )

    # Prepare/bind once per matched bucket, outside capture.
    bkey = (mode, int(bucket_m))
    bucket = session.buckets.get(bkey)
    if bucket is None:
        bucket = _prepare_bucket(
            session, plan, int(n_stages), int(bucket_m), payload, m
        )
        session.buckets[bkey] = bucket
    elif bucket.plan is not plan:
        # match_padded builds a new plan object per call; a re-prepared bucket
        # would be a new, unbound binding inside capture. Keep the bound instance,
        # but never mix two variants under one prepared bucket.
        if getattr(bucket.plan, "vid", None) != getattr(plan, "vid", None):
            _fail(
                "the profile resolved a different variant for an already prepared "
                f"bucket ({getattr(bucket.plan, 'vid', None)!r} -> "
                f"{getattr(plan, 'vid', None)!r}); start a new session for a new "
                "profile"
            )
        plan = bucket.plan

    # A moved binding is a silent wrong write: detect it before the launch.
    # (Caller output tensors are deliberately not part of `bound_ptrs`: the
    # runtime binds its own staging and copies real rows into whatever the caller
    # passed, so output buffers/views/rows are free to vary.)
    if (
        _ptr_tuple(payload, include_outputs=False, outs=bucket.outs)
        != bucket.bound_ptrs
    ):
        _fail(
            "the tensors bound at prepare moved (a bound output/cache/scale/norm "
            "pointer changed). The planned binding must stay address-stable; start "
            "a new session/prepare (e.g. re-open after a KV-pool re-allocation)."
        )

    # Surface the previous launch's device error before issuing another op.
    _check_errors(session)
    meta = _run_plan(session, bucket, payload)
    _enqueue_readback(session, bucket.plan, payload["stream"])
    _run_native_stage(session, payload, bucket, meta)

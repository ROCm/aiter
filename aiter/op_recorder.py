# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Record aiter op calls made by a framework so they can be replayed standalone.

Recording is off unless ``AITER_RECORD`` is set when aiter is imported:

``AITER_RECORD=<dir>``
    Turn recording on. Each process writes one bundle under ``<dir>``.
``AITER_RECORD_OPS=fused_moe,...``
    Ops to record (default: every op that supports recording).
``AITER_RECORD_SKIP_CALLS=N``
    Ignore the first N calls of each op, e.g. warm-up (default 0).
``AITER_RECORD_MAX_CALLS=N``
    Stop recording an op after N recorded calls (default 100000).
``AITER_RECORD_EVERY=N``
    Record one call in N after the skipped ones (default 1).

A bundle contains::

    session.json   process-wide state: versions, GPU, AITER_* env, patches
    calls.jsonl    one JSON object per recorded call
    payloads/      contents of the small tensors that drive the work
                   (routing ids/weights, masks, lengths), torch.save chunks
    configs/       copies of the tuned config files that were in effect

Only tensor metadata (shape, dtype, strides, layout attributes) is recorded
for large floating point tensors; replay fills them synthetically. Integer and
bool tensors are recorded in full, plus any argument an op lists as a payload.

Recording copies payload tensors to the host, which synchronizes the device,
so run the framework in eager mode (no CUDA/HIP graphs). Calls made while a
graph is being captured are still written, without payloads and flagged
``capturing``; they show the shapes and kernels that graph replay uses.

This module depends only on the standard library and torch so it can be
imported (and tested) without the rest of aiter.
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import dataclasses
import enum
import functools
import hashlib
import importlib
import inspect
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("aiter")

FORMAT_VERSION = 1

ENV_RECORD = "AITER_RECORD"
ENV_OPS = "AITER_RECORD_OPS"
ENV_SKIP = "AITER_RECORD_SKIP_CALLS"
ENV_MAX = "AITER_RECORD_MAX_CALLS"
ENV_EVERY = "AITER_RECORD_EVERY"

DEFAULT_MAX_CALLS = 100_000
# Integer/bool tensors up to this many elements are saved in full.
MAX_PAYLOAD_NUMEL = 1 << 22
# Calls are written together with their payloads, in chunks, so calls.jsonl
# never references a payload that is not on disk yet.
FLUSH_EVERY_CALLS = 32
FLUSH_EVERY_SECONDS = 1.0

# Environment variables saved in session.json besides AITER_*.
_EXTRA_ENV = ("CU_NUM",)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"[aiter record] ignoring {name}={raw!r}: not an integer")
        return default


@dataclasses.dataclass(frozen=True)
class RecordSettings:
    root: str
    ops: frozenset[str] | None  # None: every op that supports recording
    skip_calls: int = 0
    max_calls: int = DEFAULT_MAX_CALLS
    every: int = 1

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> RecordSettings | None:
        root = environ.get(ENV_RECORD, "").strip()
        if not root:
            return None
        ops_raw = environ.get(ENV_OPS, "").strip()
        ops = None
        if ops_raw and ops_raw.lower() != "all":
            ops = frozenset(o.strip() for o in ops_raw.split(",") if o.strip())
        return cls(
            root=root,
            ops=ops,
            skip_calls=max(0, _env_int(environ, ENV_SKIP, 0)),
            max_calls=max(0, _env_int(environ, ENV_MAX, DEFAULT_MAX_CALLS)),
            every=max(1, _env_int(environ, ENV_EVERY, 1)),
        )

    def wants(self, op: str) -> bool:
        return self.ops is None or op in self.ops


@dataclasses.dataclass(frozen=True)
class OpSpec:
    """How to record one op.

    name:
        Op name used in bundles and in ``AITER_RECORD_OPS``.
    payload_args:
        Floating point arguments whose contents are saved too (integer and
        bool tensors are always saved).
    session_info:
        Returns op-specific JSON data for session.json (e.g. which functions
        a framework monkeypatched).
    config_files:
        Returns ``{env_var: path}`` for the tuned config files the op reads.
        They are copied into the bundle; replay points ``env_var`` at the copy.
    """

    name: str
    payload_args: tuple[str, ...] = ()
    session_info: Callable[[], dict] | None = None
    config_files: Callable[[], dict[str, str]] | None = None


# ---------------------------------------------------------------------------
# Value encoding
# ---------------------------------------------------------------------------

_JSON_SCALARS = (bool, int, float, str, type(None))


def dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def dtype_from_str(name: str) -> torch.dtype:
    dtype = getattr(torch, name.removeprefix("torch."), None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"unknown torch dtype {name!r}")


def _alignment(ptr: int, cap: int = 4096) -> int:
    """Largest power of two (up to ``cap``) that divides ``ptr``."""
    if ptr == 0:
        return cap
    return min(ptr & -ptr, cap)


def tensor_attrs(tensor: torch.Tensor) -> dict[str, Any]:
    """Python attributes set on a tensor (e.g. ``is_shuffled``)."""
    try:
        attrs = vars(tensor)
    except TypeError:
        return {}
    return {
        k: v
        for k, v in attrs.items()
        if isinstance(k, str) and isinstance(v, _JSON_SCALARS)
    }


def tensor_spec(tensor: torch.Tensor) -> dict[str, Any]:
    ptr = tensor.data_ptr() if tensor.numel() else 0
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "dtype": dtype_to_str(tensor.dtype),
        "device": tensor.device.type,
        "contiguous": tensor.is_contiguous(),
        "data_ptr": ptr,
        "align": _alignment(ptr),
        "attrs": tensor_attrs(tensor),
    }


def _is_payload_dtype(dtype: torch.dtype) -> bool:
    return dtype == torch.bool or not (dtype.is_floating_point or dtype.is_complex)


def _is_enum_like(value: Any) -> bool:
    """Python enums and pybind11 enums (aiter's ActivationType, QuantType)."""
    if isinstance(value, enum.Enum):
        return True
    cls = type(value)
    return hasattr(cls, "__members__") and hasattr(value, "name")


class _Encoder:
    def __init__(self, store_payload: Callable[[torch.Tensor], str] | None):
        self._store_payload = store_payload

    def encode(self, value: Any, save_contents: bool = False) -> Any:
        if isinstance(value, torch.Tensor):
            out: dict[str, Any] = {"__tensor__": tensor_spec(value), "payload": None}
            want = save_contents or (
                _is_payload_dtype(value.dtype) and value.numel() <= MAX_PAYLOAD_NUMEL
            )
            if want and self._store_payload is not None:
                out["payload"] = self._store_payload(value)
            return out
        if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
            return value
        if isinstance(value, float):
            return value
        if isinstance(value, torch.dtype):
            return {"__dtype__": dtype_to_str(value)}
        if isinstance(value, torch.device):
            return {"__device__": str(value)}
        if _is_enum_like(value):
            cls = type(value)
            enc = {
                "__enum__": cls.__name__,
                "module": getattr(cls, "__module__", ""),
                "name": value.name,
            }
            with contextlib.suppress(Exception):
                enc["value"] = (
                    value.value if isinstance(value, enum.Enum) else int(value)
                )
            return enc
        if isinstance(value, tuple):
            return {"__tuple__": [self.encode(v) for v in value]}
        if isinstance(value, list):
            return [self.encode(v) for v in value]
        if isinstance(value, dict) and all(isinstance(k, str) for k in value):
            return {"__dict__": {k: self.encode(v) for k, v in value.items()}}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            cls = type(value)
            return {
                "__object__": f"{cls.__module__}.{cls.__qualname__}",
                "fields": {
                    f.name: self.encode(getattr(value, f.name))
                    for f in dataclasses.fields(value)
                },
            }
        cls = type(value)
        return {"__unsupported__": f"{cls.__module__}.{cls.__qualname__}"}


def find_unsupported(encoded: Any, path: str = "") -> list[str]:
    """Paths of encoded values that replay cannot rebuild."""
    found: list[str] = []
    if isinstance(encoded, dict):
        if "__unsupported__" in encoded or "__object__" in encoded:
            kind = encoded.get("__unsupported__") or encoded.get("__object__")
            return [f"{path or '<value>'} ({kind})"]
        if "__tensor__" in encoded:
            return []
        inner = encoded.get("__dict__", encoded.get("__tuple__", encoded))
        items = inner.items() if isinstance(inner, dict) else enumerate(inner)
        for k, v in items:
            found += find_unsupported(v, f"{path}.{k}" if path else str(k))
    elif isinstance(encoded, list):
        for i, v in enumerate(encoded):
            found += find_unsupported(v, f"{path}[{i}]")
    return found


# ---------------------------------------------------------------------------
# Session information
# ---------------------------------------------------------------------------


def _git_info(root: str) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            res = subprocess.run(
                ["git", "-C", root, *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return res.stdout if res.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {}
    status = git("status", "--porcelain", "--untracked-files=no") or ""
    diff = git("diff", "HEAD") or ""
    return {
        "commit": commit.strip(),
        "dirty": bool(status.strip()),
        "diff_sha1": hashlib.sha1(diff.encode()).hexdigest() if diff else None,
    }


def _package_version(*names: str) -> str | None:
    from importlib import metadata

    for name in names:
        with contextlib.suppress(metadata.PackageNotFoundError):
            return metadata.version(name)
    return None


def _distributed_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        k: os.environ[k]
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        if k in os.environ
    }
    with contextlib.suppress(Exception):
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            info["rank"] = dist.get_rank()
            info["world_size"] = dist.get_world_size()
    return info


def _process_rank() -> int | None:
    info = _distributed_info()
    for key in ("rank", "RANK"):
        if key in info:
            with contextlib.suppress(ValueError):
                return int(info[key])
    return None


def capture_env(environ: Mapping[str, str] = os.environ) -> dict[str, str]:
    return {
        k: v
        for k, v in sorted(environ.items())
        if (k.startswith("AITER_") or k in _EXTRA_ENV) and not k.startswith(ENV_RECORD)
    }


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    # Ops that record are part of aiter, so aiter is loaded; never import it
    # from here (this module must stay usable without aiter).
    chip_info = sys.modules.get("aiter.jit.utils.chip_info")
    if chip_info is not None:
        with contextlib.suppress(Exception):
            info["gfx"] = chip_info.get_gfx()
            info["cu_num"] = chip_info.get_cu_num()
    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name()
    return info


def aiter_info() -> dict[str, Any]:
    """aiter install location, version and git state."""
    root = str(Path(__file__).resolve().parent.parent)
    return {
        "root": root,
        "version": _package_version("amd-aiter", "aiter"),
        **_git_info(root),
    }


_FRAMEWORKS = ("vllm", "sglang", "atom", "transformer_engine")


def _framework_info() -> dict[str, str | None]:
    found = {}
    for name in _FRAMEWORKS:
        mod = sys.modules.get(name)
        if mod is not None:
            found[name] = getattr(mod, "__version__", None)
    return found


# ---------------------------------------------------------------------------
# Bundle writer
# ---------------------------------------------------------------------------

_CURRENT_CALL: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "aiter_record_current_call", default=None
)
_DISPATCH_LISTENERS: list[Callable[[dict], None]] = []
_HOOK_ACTIVE = False


def dispatch_hook_active() -> bool:
    """Cheap check ops use before describing their dispatch decision."""
    return _HOOK_ACTIVE


def note_dispatch(info: dict) -> None:
    """Attach an op's dispatch decision (kernels chosen) to the current call."""
    call = _CURRENT_CALL.get()
    if call is not None:
        call["dispatch"].append(info)
    for listener in _DISPATCH_LISTENERS:
        listener(info)


@contextlib.contextmanager
def collect_dispatch() -> Iterator[list[dict]]:
    """Collect dispatch decisions made inside the block (used by replay)."""
    global _HOOK_ACTIVE
    collected: list[dict] = []
    _DISPATCH_LISTENERS.append(collected.append)
    previous = _HOOK_ACTIVE
    _HOOK_ACTIVE = True
    try:
        yield collected
    finally:
        _DISPATCH_LISTENERS.remove(collected.append)
        _HOOK_ACTIVE = previous or bool(_DISPATCH_LISTENERS)


def _is_capturing() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001 - no usable device: not capturing
        return False


@dataclasses.dataclass
class _OpCounters:
    seen: int = 0
    recorded: int = 0
    capturing: int = 0
    limit_logged: bool = False


class BundleWriter:
    """Writes one bundle for the current process. Thread-safe."""

    def __init__(self, settings: RecordSettings):
        self.settings = settings
        self.path: Path | None = None
        self._lock = threading.RLock()
        self._specs: dict[str, OpSpec] = {}
        self._counters: dict[str, _OpCounters] = {}
        self._pending_calls: list[str] = []
        self._pending_payloads: dict[str, torch.Tensor] = {}
        self._payload_keys: set[str] = set()
        self._chunk = 0
        self._last_flush = time.monotonic()
        self._next_call_id = 0
        self._session: dict[str, Any] | None = None
        self._closed = False

    # -- bundle setup -------------------------------------------------------

    def _open(self) -> None:
        if self.path is not None:
            return
        rank = _process_rank()
        host = socket.gethostname().split(".")[0]
        name = f"{host}-pid{os.getpid()}"
        if rank is not None:
            name = f"rank{rank}-{name}"
        path = Path(self.settings.root).expanduser() / name
        (path / "payloads").mkdir(parents=True, exist_ok=True)
        (path / "configs").mkdir(exist_ok=True)
        self.path = path
        self._session = {
            "format_version": FORMAT_VERSION,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "flydsl": _package_version("flydsl"),
            "aiter": aiter_info(),
            "gpu": _gpu_info(),
            "env": capture_env(),
            "frameworks": _framework_info(),
            "distributed": _distributed_info(),
            "settings": dataclasses.asdict(self.settings)
            | {"ops": sorted(self.settings.ops) if self.settings.ops else None},
            "ops": {},
        }
        self._write_session()
        logger.info(f"[aiter record] recording to {path}")

    def _register_op(self, spec: OpSpec) -> None:
        """First recorded call of an op: save its session info and configs."""
        if spec.name in self._specs:
            return
        self._specs[spec.name] = spec
        assert self.path is not None and self._session is not None
        op_info: dict[str, Any] = {"configs": {}}
        if spec.session_info is not None:
            try:
                op_info.update(spec.session_info())
            except Exception as exc:  # noqa: BLE001 - recording must not break the op
                op_info["session_info_error"] = repr(exc)
        if spec.config_files is not None:
            try:
                files = spec.config_files()
            except Exception as exc:  # noqa: BLE001 - recording must not break the op
                files = {}
                op_info["config_error"] = repr(exc)
            for env_var, src in files.items():
                if not src or not os.path.isfile(src):
                    continue
                rel = Path("configs") / f"{spec.name}.{env_var}{Path(src).suffix}"
                shutil.copyfile(src, self.path / rel)
                op_info["configs"][env_var] = {"file": str(rel), "source": src}
        self._session["ops"][spec.name] = op_info
        self._write_session()

    def _write_session(self) -> None:
        assert self.path is not None and self._session is not None
        session = dict(self._session)
        session["counters"] = {
            op: dataclasses.asdict(c) for op, c in self._counters.items()
        }
        tmp = self.path / "session.json.tmp"
        tmp.write_text(json.dumps(session, indent=2, default=str))
        os.replace(tmp, self.path / "session.json")

    # -- per call -----------------------------------------------------------

    def _store_payload(self, tensor: torch.Tensor) -> str:
        host = tensor.detach().to("cpu", copy=True).contiguous()
        digest = hashlib.sha1()
        digest.update(f"{host.dtype}|{tuple(host.shape)}|".encode())
        try:
            digest.update(host.reshape(-1).view(torch.uint8).numpy().tobytes())
        except Exception:  # noqa: BLE001 - no numpy / unviewable dtype: slow path
            digest.update(repr(host.tolist()).encode())
        key = digest.hexdigest()[:20]
        if key not in self._payload_keys:
            self._payload_keys.add(key)
            self._pending_payloads[key] = host
        return key

    def begin(
        self,
        spec: OpSpec,
        signature: inspect.Signature,
        args: tuple,
        kwargs: dict,
    ) -> dict | None:
        """Encode a call's arguments, or return None if it is not sampled."""
        with self._lock:
            if self._closed:
                return None
            counters = self._counters.setdefault(spec.name, _OpCounters())
            counters.seen += 1
            index = counters.seen - 1 - self.settings.skip_calls
            if index < 0 or index % self.settings.every:
                return None
            if counters.recorded >= self.settings.max_calls:
                if not counters.limit_logged:
                    counters.limit_logged = True
                    logger.info(
                        f"[aiter record] {spec.name}: reached "
                        f"{ENV_MAX}={self.settings.max_calls}, no longer recording"
                    )
                return None
            try:
                self._open()
                self._register_op(spec)
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                capturing = _is_capturing()
                encoder = _Encoder(None if capturing else self._store_payload)
                encoded_args = {
                    name: encoder.encode(value, name in spec.payload_args)
                    for name, value in bound.arguments.items()
                }
            except Exception as exc:  # noqa: BLE001 - recording must not break the op
                logger.warning(
                    f"[aiter record] failed to record a {spec.name} call: {exc!r}"
                )
                return None
            counters.recorded += 1
            counters.capturing += int(capturing)
            call = {
                "call_id": self._next_call_id,
                "op": spec.name,
                "op_seq": counters.seen - 1,
                "time": time.time(),
                "capturing": capturing,
                "args": encoded_args,
                "dispatch": [],
            }
            self._next_call_id += 1
            return call

    def end(self, call: dict) -> None:
        line = json.dumps(call, default=str)
        with self._lock:
            if self._closed:
                return
            self._pending_calls.append(line)
            if (
                len(self._pending_calls) >= FLUSH_EVERY_CALLS
                or time.monotonic() - self._last_flush >= FLUSH_EVERY_SECONDS
            ):
                self._flush_locked()

    def _flush_locked(self) -> None:
        if self.path is None:
            return
        if self._pending_payloads:
            target = self.path / "payloads" / f"chunk_{self._chunk:05d}.pt"
            tmp = target.with_suffix(".tmp")
            torch.save(self._pending_payloads, tmp)
            os.replace(tmp, target)
            self._chunk += 1
            self._pending_payloads = {}
        if self._pending_calls:
            with open(self.path / "calls.jsonl", "a") as fp:
                fp.write("\n".join(self._pending_calls) + "\n")
            self._pending_calls = []
        self._last_flush = time.monotonic()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()
            if self.path is not None:
                self._write_session()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.flush()
            self._closed = True
            self._warn_if_empty()

    def _warn_if_empty(self) -> None:
        if any(c.recorded for c in self._counters.values()):
            return
        # Only processes that used the GPU are expected to run aiter ops.
        with contextlib.suppress(Exception):
            if not torch.cuda.is_initialized():
                return
        wanted = ", ".join(sorted(self.settings.ops)) if self.settings.ops else "any op"
        logger.warning(
            f"[aiter record] {ENV_RECORD} is set but no {wanted} call was recorded "
            "in this process. The framework may not route these ops to aiter "
            "(vLLM: VLLM_ROCM_USE_AITER=1, SGLang: SGLANG_USE_AITER=1), or the "
            "model uses another kernel path."
        )


_WRITER: BundleWriter | None = None
_WRITER_LOCK = threading.Lock()


def _writer(settings: RecordSettings) -> BundleWriter:
    global _WRITER
    with _WRITER_LOCK:
        if _WRITER is None:
            _WRITER = BundleWriter(settings)
            atexit.register(_WRITER.close)
        return _WRITER


def flush() -> None:
    """Write pending calls to disk (also done periodically and at exit)."""
    if _WRITER is not None:
        _WRITER.flush()


def record_op(spec: OpSpec, environ: Mapping[str, str] | None = None):
    """Decorator that records calls of an aiter op entry point.

    When recording is off for ``spec.name`` the function is returned unchanged,
    so there is no overhead. The decision is made when the module defining the
    op is imported.
    """
    settings = RecordSettings.from_env(os.environ if environ is None else environ)

    def decorator(func):
        global _HOOK_ACTIVE
        if settings is None or not settings.wants(spec.name):
            return func
        writer = _writer(settings)
        signature = inspect.signature(func)
        _HOOK_ACTIVE = True

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            call = writer.begin(spec, signature, args, kwargs)
            if call is None:
                return func(*args, **kwargs)
            token = _CURRENT_CALL.set(call)
            try:
                return func(*args, **kwargs)
            except BaseException as exc:
                call["error"] = f"{type(exc).__name__}: {exc}"[:1000]
                raise
            finally:
                _CURRENT_CALL.reset(token)
                writer.end(call)

        wrapper.__aiter_record_spec__ = spec
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Reading bundles and rebuilding arguments (used by replay scripts)
# ---------------------------------------------------------------------------


def load_session(bundle: str | os.PathLike) -> dict:
    return json.loads((Path(bundle) / "session.json").read_text())


def iter_calls(bundle: str | os.PathLike, op: str | None = None) -> Iterator[dict]:
    path = Path(bundle) / "calls.jsonl"
    if not path.exists():
        return
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            call = json.loads(line)
            if op is None or call.get("op") == op:
                yield call


def load_payloads(bundle: str | os.PathLike) -> dict[str, torch.Tensor]:
    payloads: dict[str, torch.Tensor] = {}
    for chunk in sorted((Path(bundle) / "payloads").glob("chunk_*.pt")):
        payloads.update(torch.load(chunk, map_location="cpu", weights_only=True))
    return payloads


def _resolve_enum(enc: dict) -> Any:
    candidates = []
    if enc.get("module"):
        with contextlib.suppress(Exception):
            candidates.append(importlib.import_module(enc["module"]))
    # aiter's pybind enums live in a compiled module; aiter re-exports them.
    if "aiter" in sys.modules:
        candidates.append(sys.modules["aiter"])
    for mod in candidates:
        cls = getattr(mod, enc["__enum__"], None)
        if cls is not None and hasattr(cls, enc["name"]):
            return getattr(cls, enc["name"])
    raise ValueError(f"cannot resolve enum {enc['__enum__']}.{enc['name']}")


def _fill_synthetic(base: torch.Tensor, name: str, generator: torch.Generator) -> None:
    """Fill a flat tensor with values that are cheap and valid for its role."""
    dtype = base.dtype
    is_scale = "scale" in name
    if dtype == getattr(torch, "float8_e8m0fnu", None) or (
        dtype == torch.uint8 and is_scale
    ):
        base.view(torch.uint8).fill_(127)  # e8m0 exponent 127 == 1.0
    elif base.element_size() == 1 and not (
        dtype.is_floating_point and dtype.itemsize == 1 and "float8" in str(dtype)
    ):
        # packed fp4 / int8 / uint8 data: random bytes
        base.view(torch.uint8).random_(0, 256, generator=generator)
    elif dtype.is_floating_point:
        if is_scale:
            base.fill_(1.0)
        else:
            tmp = torch.randn(
                base.shape, generator=generator, device=base.device, dtype=torch.float32
            )
            base.copy_(tmp.mul_(0.1))
    elif dtype == torch.bool:
        base.fill_(True)
    else:
        base.zero_()


class ArgumentFactory:
    """Rebuilds recorded arguments on a device.

    Tensors without payload are allocated with the recorded shape, strides,
    storage offset and attributes, and filled synthetically. A recorded
    buffer (same data_ptr and layout) maps to the same replay buffer, so
    per-layer weights stay distinct while being allocated once.
    Arguments named in ``ignore_ptr_for`` share one buffer per layout instead
    (e.g. one weight set for all layers).
    """

    def __init__(
        self,
        payloads: Mapping[str, torch.Tensor],
        device: str | torch.device = "cuda",
        seed: int = 0,
        ignore_ptr_for: tuple[str, ...] | frozenset[str] = (),
    ):
        self.payloads = payloads
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.ignore_ptr_for = frozenset(ignore_ptr_for)
        self._buffers: dict[tuple, torch.Tensor] = {}
        self._payload_tensors: dict[str, torch.Tensor] = {}
        self.warnings: list[str] = []

    def build_args(self, encoded_args: Mapping[str, Any]) -> dict[str, Any]:
        return {name: self.build(value, name) for name, value in encoded_args.items()}

    def build(self, enc: Any, name: str = "") -> Any:
        if isinstance(enc, list):
            return [self.build(v, name) for v in enc]
        if not isinstance(enc, dict):
            return enc
        if "__tensor__" in enc:
            return self._tensor(enc["__tensor__"], enc.get("payload"), name)
        if "__dtype__" in enc:
            return dtype_from_str(enc["__dtype__"])
        if "__device__" in enc:
            return self.device
        if "__enum__" in enc:
            return _resolve_enum(enc)
        if "__tuple__" in enc:
            return tuple(self.build(v, name) for v in enc["__tuple__"])
        if "__dict__" in enc:
            return {k: self.build(v, k) for k, v in enc["__dict__"].items()}
        problems = find_unsupported(enc, name)
        raise ValueError(f"cannot rebuild argument {', '.join(problems) or name}")

    def _tensor(self, spec: dict, payload: str | None, name: str) -> torch.Tensor:
        dtype = dtype_from_str(spec["dtype"])
        shape, stride = tuple(spec["shape"]), tuple(spec["stride"])
        offset = int(spec.get("storage_offset", 0))
        if payload is not None:
            if payload not in self.payloads:
                raise ValueError(f"{name}: payload {payload} missing from bundle")
            tensor = self._payload_tensors.get(payload)
            if tensor is None:
                tensor = self.payloads[payload].to(self.device)
                self._payload_tensors[payload] = tensor
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name}: payload shape {tuple(tensor.shape)} != {shape}"
                )
            if tensor.stride() != stride:
                tensor = torch.empty_strided(
                    shape, stride, dtype=dtype, device=self.device
                ).copy_(tensor)
        else:
            ptr = None if name in self.ignore_ptr_for else spec.get("data_ptr")
            key = (name, ptr, spec["dtype"], shape, stride, offset)
            tensor = self._buffers.get(key)
            if tensor is None:
                if _is_payload_dtype(dtype) and int(torch.Size(shape).numel()):
                    self.warnings.append(
                        f"{name}: integer tensor recorded without contents, using zeros"
                    )
                tensor = self._allocate(shape, stride, offset, dtype, name)
                self._buffers[key] = tensor
        for attr, value in spec.get("attrs", {}).items():
            setattr(tensor, attr, value)
        return tensor

    def _allocate(self, shape, stride, offset, dtype, name) -> torch.Tensor:
        if any(s == 0 for s in shape):
            span = 0
        else:
            span = 1 + sum((s - 1) * st for s, st in zip(shape, stride))
        base = torch.empty(offset + span, dtype=dtype, device=self.device)
        if base.numel():
            _fill_synthetic(base, name, self.generator)
        return base.as_strided(shape, stride, offset)

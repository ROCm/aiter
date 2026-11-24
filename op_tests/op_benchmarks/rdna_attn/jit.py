# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Standalone builds of the RDNA attention kernels, called through ctypes.

One shared object per build, of any knobs, compiled straight from
csrc/kernels/rdna_attn: the points tune.py searches are not in the tables.
After `install()`, this module picks and loads builds as
`aiter.ops.rdna_attn` does (it is a `kernels` of harness.py), with the knobs
of `override()` on top of the tables' entry.
"""

import contextlib
import ctypes
import dataclasses
import functools
import hashlib
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from aiter.ops import rdna_attn as R

_CSRC = Path(__file__).resolve().parents[3] / "csrc" / "kernels" / "rdna_attn"
_COMMON = (
    _CSRC / "rdna_attn_common.cuh",
    _CSRC / "rdna_attn_launch.h",
)
# Not the configs headers: their entries do not change a build.
_SOURCES = (_CSRC / "rdna_decode_attn.cuh", *_COMMON)
_PREFILL_SOURCES = (_CSRC / "rdna_prefill_attn.cuh", *_COMMON)


@functools.cache
def _arch() -> str:
    """The target: RDNA_JIT_ARCH, else the current device's."""
    arch = os.environ.get("RDNA_JIT_ARCH")
    return arch or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


_WRAPPER = """#include "rdna_decode_attn.cuh"
using namespace rdna;
extern "C" __attribute__((visibility("default"))) void rdna_decode_launch(
    const LaunchArgs* a) {{
  launch_decode<DecodeConfig{{{fields}}}, {batched}>(*a);
}}
"""
_PREFILL_WRAPPER = """#include "rdna_prefill_attn.cuh"
using namespace rdna;
extern "C" __attribute__((visibility("default"))) void rdna_prefill_launch(
    const PrefillArgs* a) {{
  launch_prefill<PrefillConfig{{{fields}}}>(*a);
}}
"""


def cpp_fields(fields: list[int], names: tuple[str, ...]) -> str:
    """A config's fields, named by `names`, as C++ initializers: the head size
    and dtype as their HeadDim and Dtype enumerators
    (rdna_attn_types.h)."""
    out = []
    for name, x in zip(names, fields, strict=True):
        if name == "head_size":
            out.append(f"HeadDim::k{x}")
        elif name == "dtype":
            out.append("Dtype::kBf16" if x else "Dtype::kFp16")
        else:
            out.append(str(x))
    return ", ".join(out)


def parse_cpp_fields(text: str) -> list[int]:
    """The fields of C++ initializers written by `cpp_fields`."""
    enums = {"Dtype::kFp16": 0, "Dtype::kBf16": 1}
    return [
        enums[t] if t in enums else int(t.removeprefix("HeadDim::k"))
        for t in map(str.strip, text.split(","))
    ]


class _LaunchArgs(ctypes.Structure):
    """Mirrors rdna::LaunchArgs (rdna_attn_launch.h)."""

    _fields_ = [
        ("q", ctypes.c_void_p),
        ("kv", ctypes.c_void_p),
        ("bt", ctypes.c_void_p),
        ("acc", ctypes.c_void_p),
        ("m", ctypes.c_void_p),
        ("l", ctypes.c_void_p),
        ("cnt", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("seq_lens", ctypes.c_void_p),
        ("bt_width", ctypes.c_int),
        ("scale", ctypes.c_float),
        ("stream", ctypes.c_void_p),
        ("mm_ranges", ctypes.c_void_p),
        ("mm_capacity", ctypes.c_int),
        ("nseq", ctypes.c_int),
        ("bt_stride", ctypes.c_int),
        ("acc_stride", ctypes.c_int),
        ("ml_stride", ctypes.c_int),
        ("cnt_stride", ctypes.c_int),
        ("mm_stride", ctypes.c_int),
    ]


class _PrefillArgs(ctypes.Structure):
    """Mirrors rdna::PrefillArgs (rdna_attn_launch.h)."""

    _fields_ = [
        ("q", ctypes.c_void_p),
        ("kv", ctypes.c_void_p),
        ("bt", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("seq_lens", ctypes.c_void_p),
        ("partial_o", ctypes.c_void_p),
        ("partial_ml", ctypes.c_void_p),
        ("counters", ctypes.c_void_p),
        ("num_tokens", ctypes.c_int),
        ("max_seq_len", ctypes.c_int),
        ("scale", ctypes.c_float),
        ("stream", ctypes.c_void_p),
        ("mm_ranges", ctypes.c_void_p),
        ("num_mm_ranges", ctypes.c_int),
        ("plan", ctypes.c_void_p),
        ("query_start", ctypes.c_void_p),
        ("num_entries", ctypes.c_int),
        ("bt_stride", ctypes.c_int),
        ("head_group", ctypes.c_int),
        ("v", ctypes.c_void_p),
        ("q_stride", ctypes.c_int),
        ("k_stride", ctypes.c_int),
        ("v_stride", ctypes.c_int),
    ]


_cache: Path | None = None
_ablate = 0
_sealed = False
_overrides: dict = {}
_variant_memo: dict = {}
_loaded: dict = {}
_failed: dict[R.KernelVariant | R.PrefillVariant, str] = {}
_lock = threading.Lock()


def source_digest(prefill: bool = False) -> str:
    """A digest of the kernel sources."""
    h = hashlib.sha256()
    for p in _PREFILL_SOURCES if prefill else _SOURCES:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


@functools.cache
def _toolchain() -> tuple[list[str], str]:
    """(clang command, HIP runtime rpath). The pip ROCm SDK gets a shim
    --rocm-path with the unversioned libamdhip64.so.
    """
    sdk = Path(torch.__file__).resolve().parents[1] / "_rocm_sdk_core"
    if sdk.is_dir():
        shim = _cache / "rocm"
        if not (shim / "lib" / "libamdhip64.so").exists():
            tmp = _cache / f"rocm.{os.getpid()}.{threading.get_ident()}"
            (tmp / "lib").mkdir(parents=True)
            for name in ("bin", "include", "share", "libexec", "etc"):
                if (sdk / name).exists():
                    (tmp / name).symlink_to(sdk / name)
            for entry in (sdk / "lib").iterdir():
                (tmp / "lib" / entry.name).symlink_to(entry)
            hip = next((sdk / "lib").glob("libamdhip64.so.*"))
            (tmp / "lib" / "libamdhip64.so").symlink_to(hip)
            try:
                os.rename(tmp, shim)
            except OSError:  # made by another process
                shutil.rmtree(tmp)
        bitcode = sdk / "lib" / "llvm" / "amdgcn" / "bitcode"
        return [
            str(sdk / "lib" / "llvm" / "bin" / "clang++"),
            f"--rocm-path={shim}",
            f"--rocm-device-lib-path={bitcode}",
        ], str(sdk / "lib")
    from torch.utils.cpp_extension import ROCM_HOME

    if ROCM_HOME is None:
        raise RuntimeError("no ROCm found: neither the pip SDK nor ROCM_HOME")
    rocm = Path(ROCM_HOME)
    return [str(rocm / "lib" / "llvm" / "bin" / "clang++"), f"--rocm-path={rocm}"], (
        str(rocm / "lib")
    )


def _build(v: R.KernelVariant | R.PrefillVariant) -> Path:
    """Compile one variant, cached by a hash of the sources and command."""
    prefill = isinstance(v, R.PrefillVariant)
    if prefill:
        fields = cpp_fields(R.prefill_config_fields(v), R.PREFILL_CONFIG_FIELDS)
        tag = f"{v.name}_ab{_ablate}"
        kernel, template = "prefill_attn", _PREFILL_WRAPPER
        batched = "false"
    else:
        fields = cpp_fields(R.config_fields(v), R.DECODE_CONFIG_FIELDS)
        tag = f"{v.name}_l{v.layout}_ab{_ablate}"
        kernel, template = "decode_attn", _WRAPPER
        batched = "true" if v.batched else "false"
    wrapper = _cache / f"{tag}.hip"
    _write(wrapper, template.format(fields=fields, batched=batched))
    clang, rpath = _toolchain()
    cmd = [
        *clang,
        "-x",
        "hip",
        f"--offload-arch={_arch()}",
        "-fno-gpu-rdc",
        "-O3",
        "-std=c++20",
        "-fPIC",
        "-shared",
        "-Wno-unused-variable",
        # Builds loaded together need distinct kernel names.
        f"-D{kernel}={kernel}_{hashlib.sha256(tag.encode()).hexdigest()[:16]}",
        f"-DABLATE={_ablate}",
        f"-I{_CSRC}",
        f"-Wl,-rpath,{rpath}",
        str(wrapper),
    ]
    h = hashlib.sha256((source_digest(prefill) + " ".join(cmd)).encode()).hexdigest()
    so = _cache / f"{tag}-{h[:16]}.so"
    if so.is_file():
        return so
    tmp = so.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    proc = subprocess.run([*cmd, "-o", str(tmp)], capture_output=True, text=True)
    if proc.returncode:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(proc.stderr.strip()[-2000:] or f"clang {proc.returncode}")
    os.replace(tmp, so)
    return so


class Jit:
    """A decode build, launched as rdna_attn.load's."""

    def __init__(self, v: R.KernelVariant, path: Path):
        self.variant = v
        self._launch = ctypes.CDLL(str(path)).rdna_decode_launch
        self._launch.argtypes = [ctypes.POINTER(_LaunchArgs)]
        self._launch.restype = None

    def decode_attn(
        self,
        q,
        kv_cache,
        block_table,
        out,
        acc,
        m,
        ls,
        cnt,
        seq_lens,
        scale,
        mm_ranges=None,
    ):
        v = self.variant
        if not (q.is_contiguous() and out.is_contiguous()):
            raise RuntimeError("q and out must be contiguous")
        if not q.dtype == kv_cache.dtype == out.dtype == v.dtype:
            raise RuntimeError("dtype does not match the build")
        nseq = q.size(0) // v.max_query_len
        if nseq * v.max_query_len != q.size(0) or (not v.batched and nseq != 1):
            raise RuntimeError(f"q has {q.size(0)} tokens for {v.name}")
        rows = mm_ranges is not None and mm_ranges.dim() == 2
        if mm_ranges is not None and not (
            mm_ranges.dtype == torch.int32
            and mm_ranges.is_contiguous()
            and mm_ranges.dim() in (1, 2)
            and mm_ranges.size(-1) % 2 == 1
            and (mm_ranges.size(0) >= nseq if rows else nseq == 1)
            and mm_ranges.device == q.device
        ):
            raise RuntimeError(
                "mm_ranges must be a contiguous int32 [count, start0, end0, ...], "
                f"or one such row per sequence, on {q.device}"
            )
        args = _LaunchArgs(
            q.data_ptr(),
            kv_cache.data_ptr(),
            block_table.data_ptr(),
            acc.data_ptr(),
            m.data_ptr(),
            ls.data_ptr(),
            cnt.data_ptr(),
            out.data_ptr(),
            seq_lens.data_ptr(),
            block_table.size(-1),
            scale,
            torch.cuda.current_stream(q.device).cuda_stream,
            mm_ranges.data_ptr() if mm_ranges is not None else None,
            mm_ranges.size(-1) // 2 if mm_ranges is not None else 0,
            nseq,
            block_table.stride(0) if block_table.dim() == 2 else 0,
            acc.stride(0) if v.batched else 0,
            m.stride(0) if v.batched else 0,
            cnt.stride(0) if v.batched else 0,
            mm_ranges.stride(0) if rows and v.batched else 0,
        )
        self._launch(ctypes.byref(args))


class JitPrefill:
    """A prefill build, launched as rdna_attn.load_prefill's."""

    def __init__(self, v: R.PrefillVariant, path: Path):
        self.variant = v
        self._launch = ctypes.CDLL(str(path)).rdna_prefill_launch
        self._launch.argtypes = [ctypes.POINTER(_PrefillArgs)]
        self._launch.restype = None

    def prefill_attn(
        self,
        q,
        kv_cache,
        block_table,
        out,
        partial_o,
        partial_ml,
        counters,
        seq_lens,
        max_seq_len,
        scale,
        mm_ranges=None,
        plan=None,
        query_start_loc=None,
        head_group=0,
    ):
        v = self.variant
        if not (q.is_contiguous() and out.is_contiguous()):
            raise RuntimeError("q and out must be contiguous")
        if not q.dtype == kv_cache.dtype == out.dtype == v.dtype:
            raise RuntimeError("dtype does not match the build")
        if (plan is None) != (query_start_loc is None):
            raise RuntimeError("a batch takes both a plan and query_start_loc")
        dim = 2 if plan is None else 3
        if mm_ranges is not None and not (
            mm_ranges.dtype == torch.int32
            and mm_ranges.is_contiguous()
            and mm_ranges.dim() == dim
            and mm_ranges.size(-1) == 2
            and mm_ranges.device == q.device
        ):
            raise RuntimeError(
                f"mm_ranges must be contiguous int32 of {dim} dims (.., 2) on "
                f"{q.device}, got {mm_ranges.dtype} {tuple(mm_ranges.shape)} "
                f"on {mm_ranges.device}"
            )
        args = _PrefillArgs(
            q.data_ptr(),
            kv_cache.data_ptr(),
            block_table.data_ptr(),
            out.data_ptr(),
            seq_lens.data_ptr(),
            partial_o.data_ptr(),
            partial_ml.data_ptr(),
            counters.data_ptr(),
            q.size(0),
            max_seq_len,
            scale,
            torch.cuda.current_stream(q.device).cuda_stream,
            mm_ranges.data_ptr() if mm_ranges is not None else None,
            mm_ranges.size(-2) if mm_ranges is not None else 0,
            plan.data_ptr() if plan is not None else None,
            query_start_loc.data_ptr() if plan is not None else None,
            plan.size(0) if plan is not None else 0,
            block_table.stride(0) if block_table.dim() == 2 else 0,
            head_group,
        )
        self._launch(ctypes.byref(args))

    def encoder_attn(
        self,
        q,
        k,
        v,
        out,
        partial_o,
        partial_ml,
        counters,
        seq_lens,
        max_seq_len,
        scale,
        plan=None,
        query_start_loc=None,
        head_group=0,
    ):
        var = self.variant
        if var.causal:
            raise RuntimeError("a causal build runs through prefill_attn")
        for x in (q, k, v):
            if x.stride(2) != 1 or x.stride(1) != var.head_size:
                raise RuntimeError("each token's heads must be contiguous")
        if not (out.is_contiguous() and q.dtype == k.dtype == v.dtype == var.dtype):
            raise RuntimeError("out must be contiguous, all of the build's dtype")
        if (plan is None) != (query_start_loc is None):
            raise RuntimeError("a batch takes both a plan and query_start_loc")
        args = _PrefillArgs(
            q=q.data_ptr(),
            kv=k.data_ptr(),
            out=out.data_ptr(),
            seq_lens=seq_lens.data_ptr(),
            partial_o=partial_o.data_ptr(),
            partial_ml=partial_ml.data_ptr(),
            counters=counters.data_ptr(),
            num_tokens=q.size(0),
            max_seq_len=max_seq_len,
            scale=scale,
            stream=torch.cuda.current_stream(q.device).cuda_stream,
            plan=plan.data_ptr() if plan is not None else None,
            query_start=query_start_loc.data_ptr() if plan is not None else None,
            num_entries=plan.size(0) if plan is not None else 0,
            head_group=head_group,
            v=v.data_ptr(),
            q_stride=q.stride(0),
            k_stride=k.stride(0),
            v_stride=v.stride(0),
        )
        self._launch(ctypes.byref(args))


def load(v: R.KernelVariant | R.PrefillVariant) -> Jit | JitPrefill | None:
    """The variant's build, or None if it fails (see `failure`).

    Raises:
        RuntimeError: Sealed and not built yet.

    """
    if v in _loaded:
        return _loaded[v]
    if isinstance(v, R.PrefillVariant):
        # MIN_QUERY_LEN does not change the build.
        step = v
        v = dataclasses.replace(v, min_query_len=R.PREFILL_MIN_M)
        if v in _loaded:
            _loaded[step] = _loaded[v]
            return _loaded[v]
    if v in _failed:
        return None
    if _sealed:
        raise RuntimeError(
            f"{v.name} was not precompiled; building it now would land in the "
            "timed region"
        )
    try:
        path = _build(v)
        mod = JitPrefill(v, path) if isinstance(v, R.PrefillVariant) else Jit(v, path)
    except Exception as exc:  # noqa: BLE001 - a refused shape is a normal outcome
        with _lock:
            _failed[v] = f"{v.name} does not build: {exc}"
        return None
    with _lock:
        _loaded[v] = mod
    return mod


# harness.py loads prefill builds through this name.
load_prefill = load


def failure(v: R.KernelVariant | R.PrefillVariant) -> str | None:
    if isinstance(v, R.PrefillVariant):
        v = dataclasses.replace(v, min_query_len=R.PREFILL_MIN_M)
    return _failed.get(v)


def precompile(variants) -> int:
    """Build variants in parallel and load them, sealed or not; the number
    built."""
    global _sealed
    todo = [
        (
            dataclasses.replace(v, min_query_len=R.PREFILL_MIN_M)
            if isinstance(v, R.PrefillVariant)
            else v
        )
        for v in variants
    ]
    todo = [v for v in dict.fromkeys(todo) if v not in _loaded]
    sealed, _sealed = _sealed, False
    try:
        with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 8) - 2)) as pool:
            list(pool.map(load, todo))
    finally:
        _sealed = sealed
    return len(todo)


def seal() -> None:
    """Refuse further builds, which would disturb timing."""
    global _sealed
    _sealed = True


def variant_for(hq, hkv, d, m, page_size, layout, window, dtype, batch=False):
    """variant_for with the `override()` knobs applied."""
    base = R.variant_for(hq, hkv, d, m, page_size, layout, window, dtype, batch)
    if base is None:
        base = R.KernelVariant(
            d,
            hq,
            hkv,
            m,
            page_size,
            layout,
            dtype=dtype,
            window=window,
            batched=int(batch),
        )
    return _with_overrides(base)


def _with_overrides(base):
    """`base` with the overridden knobs it has (memoized)."""
    key = (base, tuple(sorted(_overrides.items())))
    hit = _variant_memo.get(key)
    if hit is None:
        names = {f.name for f in dataclasses.fields(base)}
        own = {k: v for k, v in _overrides.items() if k in names}
        hit = _variant_memo[key] = dataclasses.replace(base, **own) if own else base
    return hit


# Prefill knobs for shapes without a row.
def _prefill_defaults(d: int) -> dict:
    return {
        "waves": 8,
        "key_tile": 64 if d <= 128 else 16,
        "head_dim_split": max(1, d // 128),
        "v_t": 1,
        "double_buffer": int(d == 256),
        "key_waves": 1,
        "max_segments": 8,
        "head_group": 8,
    }


def prefill_variant_for(hq, hkv, d, page_size, layout, window, dtype, num_tokens):
    """prefill_variant_for with the `override()` knobs applied."""
    base = R.prefill_variant_for(
        hq, hkv, d, page_size, layout, window, dtype, num_tokens
    )
    if base is None:
        if layout != 1 or num_tokens < R.PREFILL_MIN_M:
            return None
        base = R.PrefillVariant(
            d,
            hq,
            hkv,
            window,
            page_size,
            R.PREFILL_MIN_M,
            **_prefill_defaults(d),
            dtype=dtype,
        )
    return _with_overrides(base)


def prefill_batch_variant_for(hq, hkv, d, page_size, layout, window, dtype, *batch):
    """prefill_batch_variant_for with the `override()` knobs applied."""
    v, head_group = R.prefill_batch_variant_for(
        hq, hkv, d, page_size, layout, window, dtype, *batch
    )
    if v is None:
        v = prefill_variant_for(
            hq, hkv, d, page_size, layout, window, dtype, R.PREFILL_MIN_M
        )
        return v, 0
    return _with_overrides(v), head_group


@contextlib.contextmanager
def override(knobs: dict):
    """Force these launch knobs (after `install`)."""
    _overrides.clear()
    _overrides.update(knobs)
    try:
        yield
    finally:
        _overrides.clear()


def install(ablate: int = 0) -> None:
    """Set where the builds are kept: RDNA_JIT_CACHE, by default
    ~/.cache/aiter/rdna_jit.

    Args:
        ablate: ABLATE of every build (profiling only, wrong results).

    """
    global _cache, _ablate
    _ablate = ablate
    root = os.environ.get("RDNA_JIT_CACHE") or Path.home() / ".cache/aiter/rdna_jit"
    _cache = Path(root).resolve()
    _cache.mkdir(parents=True, exist_ok=True)


def _write(path: Path, text: str) -> None:
    """Write `text` to `path` atomically, unless it is already there."""
    if not path.is_file() or path.read_text() != text:
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, path)

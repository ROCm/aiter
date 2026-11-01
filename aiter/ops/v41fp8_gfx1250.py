# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 backend of ``mla_sparse_prefill_v41`` (DSV4.1 MXFP8 sparse MLA attention).

Warp-specialized persistent HIP kernel (``v41fp8_gfx1250.hip``): TDM gather of the sparse rows,
native MX-FP8 WMMA for QK^T, BF16 WMMA for PV. Enabled from
``mla_sparse_prefill._opus_sparse_prefill_v41`` when ``AITER_V41_FP8_GFX1250=1`` (default off).
The kernel is built with hipcc on first use and cached in ``AITER_V41_FP8_CACHE``
(default ``~/.cache/aiter_v41fp8``), keyed by the source hash.

Constraints (checked): H == 64, D == 512, int32 CSR indices, MXFP8 rows with the E8M0 scales inline
at +512 of a 528-byte row (ATOM's V4.1 pool / ring layout), q contiguous along D.
"""

import ctypes
import functools
import hashlib
import os
import subprocess

import torch

H, D, ROWB = 64, 512, 528
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v41fp8_gfx1250.hip")
_HIPCC = os.environ.get("HIPCC", "/opt/rocm/bin/hipcc")


class _Params(ctypes.Structure):
    _fields_ = [
        ("q", ctypes.c_void_p),
        ("q_st", ctypes.c_longlong),
        ("q_sh", ctypes.c_longlong),
        ("qs", ctypes.c_void_p),
        ("qs_st", ctypes.c_longlong),
        ("qs_sh", ctypes.c_longlong),
        ("p_base", ctypes.c_void_p),
        ("p_rows", ctypes.c_longlong),
        ("e_base", ctypes.c_void_p),
        ("e_rows", ctypes.c_longlong),
        ("p_idx", ctypes.c_void_p),
        ("p_ptr", ctypes.c_void_p),
        ("e_idx", ctypes.c_void_p),
        ("e_ptr", ctypes.c_void_p),
        ("sink", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("o_st", ctypes.c_longlong),
        ("o_sh", ctypes.c_longlong),
        ("scale_log2", ctypes.c_float),
        ("T", ctypes.c_int),
        ("G", ctypes.c_int),
        ("minus1", ctypes.c_void_p),
        ("prof", ctypes.c_void_p),
        ("n_pidx", ctypes.c_longlong),
        ("n_eidx", ctypes.c_longlong),  # start-up index prefetch (0 = off)
    ]


@functools.cache
def _lib():
    with open(_SRC, "rb") as f:
        src = f.read()
    tag = hashlib.sha1(src + b"m1").hexdigest()[:16]
    cache = os.environ.get(
        "AITER_V41_FP8_CACHE", os.path.expanduser("~/.cache/aiter_v41fp8")
    )
    os.makedirs(cache, exist_ok=True)
    so = os.path.join(cache, f"v41fp8_gfx1250_{tag}.so")
    if not os.path.exists(so):
        tmp = so + f".tmp{os.getpid()}"
        cmd = [
            _HIPCC,
            "-O3",
            "--offload-arch=gfx1250",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-DQKMODE=1",
            _SRC,
            "-o",
            tmp,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"hipcc failed to build {_SRC}:\n{r.stderr[-4000:]}")
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.v41fp8_gfx1250_launch.argtypes = [ctypes.POINTER(_Params), ctypes.c_void_p]
    lib.v41fp8_gfx1250_launch.restype = ctypes.c_int
    lib.v41fp8_gfx1250_max_T.restype = ctypes.c_int
    return lib


@functools.cache
def _minus1(dev: str):
    return torch.full((64,), -1, dtype=torch.int32, device=dev)


def _check_rows(name, v, s):
    if v.shape[0] == 0:
        return
    if v.dim() != 2 or v.shape[1] != D or s.shape[-1] != D // 32:
        raise RuntimeError(f"{name}: expected [rows, 512] values and [rows, 16] scales")
    if (
        v.stride(1) != 1
        or v.stride(0) != ROWB
        or s.stride(0) != ROWB
        or s.data_ptr() != v.data_ptr() + D
    ):
        raise RuntimeError(
            f"{name}: gfx1250 kernel needs inline MXFP8 rows (528-byte stride, scales at +512)"
        )
    if v.shape[0] >= 2**31:
        raise RuntimeError(f"{name}: too many rows")


def sparse_prefill_v41_gfx1250(
    q,
    q_scale,
    unified_kv,
    unified_kv_scale,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv,
    kv_scale,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
    out,
):
    T, h, d = q.shape
    if h != H or d != D:
        raise RuntimeError(
            f"gfx1250 mla_sparse_prefill_v41 supports H=64, D=512, got H={h}, D={d}"
        )
    if q.stride(2) != 1 or q_scale.stride(2) != 1:
        raise RuntimeError("q / q_scale must be contiguous along the last dim")
    for t in (kv_indices_prefix, kv_indptr_prefix, kv_indices_extend, kv_indptr_extend):
        if t.dtype != torch.int32:
            raise RuntimeError("CSR indices / indptr must be int32")
    _check_rows("unified_kv", unified_kv, unified_kv_scale)
    _check_rows("kv", kv, kv_scale)
    if out is None:
        out = torch.empty(T, H, D, dtype=torch.bfloat16, device=q.device)
    elif out.shape != (T, H, D) or out.dtype != torch.bfloat16 or out.stride(2) != 1:
        raise RuntimeError("out must be [T, 64, 512] bf16, contiguous along D")
    if T == 0:
        return out
    lib = _lib()
    if T > lib.v41fp8_gfx1250_max_T():
        raise RuntimeError(
            f"gfx1250 mla_sparse_prefill_v41: T={T} exceeds {lib.v41fp8_gfx1250_max_T()}"
        )
    p = _Params()
    p.q, p.q_st, p.q_sh = (
        q.data_ptr(),
        q.stride(0) * q.element_size(),
        q.stride(1) * q.element_size(),
    )
    p.qs, p.qs_st, p.qs_sh = (
        q_scale.data_ptr(),
        q_scale.stride(0) * q_scale.element_size(),
        q_scale.stride(1) * q_scale.element_size(),
    )
    p.p_base, p.p_rows = unified_kv.data_ptr(), unified_kv.shape[0]
    p.e_base, p.e_rows = kv.data_ptr(), kv.shape[0]
    p.p_idx, p.p_ptr = kv_indices_prefix.data_ptr(), kv_indptr_prefix.data_ptr()
    p.e_idx, p.e_ptr = kv_indices_extend.data_ptr(), kv_indptr_extend.data_ptr()
    if attn_sink is not None:
        if attn_sink.dtype != torch.float32 or attn_sink.numel() != H:
            raise RuntimeError("attn_sink must be [64] fp32")
        p.sink = attn_sink.data_ptr()
    else:
        p.sink = None
    p.out, p.o_st, p.o_sh = out.data_ptr(), out.stride(0) * 2, out.stride(1) * 2
    p.scale_log2 = float(softmax_scale) * 1.4426950408889634
    p.T = T
    p.G = torch.cuda.get_device_properties(q.device).multi_processor_count
    p.minus1 = _minus1(str(q.device)).data_ptr()
    p.prof = None
    p.n_pidx, p.n_eidx = kv_indices_prefix.numel(), kv_indices_extend.numel()
    err = lib.v41fp8_gfx1250_launch(
        ctypes.byref(p),
        ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream),
    )
    if err:
        raise RuntimeError(f"v41fp8_gfx1250 launch failed: hipError {err}")
    return out

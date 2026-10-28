# SPDX-License-Identifier: MIT
"""Ahead-of-time m3_step_kernel binaries (VLLM_M3_GLUON_AOT_DIR): code objects compiled offline
(port/offline_compile.py: Triton 3.8 Gluon, gfx950) and launched with hipModuleLaunchKernel, so an
image whose own Triton cannot build the kernel (vLLM v0.27.1: Triton 3.6) runs the same binary.

``m3_step_w{S}_q{QL}.hsaco`` + ``.json`` (its constexprs and LDS bytes) per width. The kernel's
parameters are the 31 of m3_step_kernel plus Triton's two trailing scratch pointers (null: the
kernel has no global / profile scratch)."""

from __future__ import annotations

import ctypes
import json
import os

import torch

# m3_step_kernel's runtime parameters in order: ctypes type of each (pointers as uint64)
_ARGS = [ctypes.c_uint64, ctypes.c_int32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
         ctypes.c_int32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
         ctypes.c_uint64, ctypes.c_int64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
         ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
         ctypes.c_int32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
         ctypes.c_uint64, ctypes.c_uint64]  # (.. moe_out, qkv_out, attn_in)
_N_SCRATCH = 2

_hip = None


def _lib():
    global _hip
    if _hip is None:
        _hip = ctypes.CDLL("libamdhip64.so")
        _hip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        _hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        _hip.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + [
            ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        _hip.hipGetErrorString.restype = ctypes.c_char_p
    return _hip


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f"Gluon M3 AOT: {what} failed: {_lib().hipGetErrorString(rc).decode()} ({rc})")


class AotKernel:
    """One width's binary, loaded on the current device."""

    def __init__(self, aot_dir: str, S: int, QL: int, consts: dict):
        tag = f"w{S}_q{QL}"
        path = os.path.join(aot_dir, f"m3_step_{tag}.hsaco")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        meta = json.load(open(os.path.join(aot_dir, f"m3_step_{tag}.json")))
        want = {k: consts[k] for k in ("S", "HALF", "PHALF", "FFN_OFF", "QL", "STAGES", "LOOP")}
        got = {k: meta["consts"][k] for k in want}
        if got != want:
            raise RuntimeError(f"Gluon M3 AOT {tag}: built with {got}, the runner wants {want}")
        self.shared = int(meta["shared"])
        self.meta = meta
        self._image = open(path, "rb").read()  # kept alive: the module references it
        lib = _lib()
        self._mod = ctypes.c_void_p()
        _check(lib.hipModuleLoadData(ctypes.byref(self._mod), self._image), f"hipModuleLoadData({tag})")
        self._fn = ctypes.c_void_p()
        _check(lib.hipModuleGetFunction(ctypes.byref(self._fn), self._mod, meta["name"].encode()),
               f"hipModuleGetFunction({tag})")

    def __call__(self, grid: int, args) -> None:
        vals = []
        if len(args) != len(_ARGS):
            raise RuntimeError(f"Gluon M3 AOT: {len(args)} launch arguments, the binary takes {len(_ARGS)}")
        for t, a in zip(_ARGS, args):
            if isinstance(a, torch.Tensor):
                a = a.data_ptr()
            vals.append(t(a))
        vals += [ctypes.c_uint64(0)] * _N_SCRATCH
        params = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
        stream = torch.cuda.current_stream().cuda_stream
        _check(_lib().hipModuleLaunchKernel(self._fn, grid, 1, 1, 512, 1, 1, self.shared, ctypes.c_void_p(stream),
                                            params, None), "hipModuleLaunchKernel")

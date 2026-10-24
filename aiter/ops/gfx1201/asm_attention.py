# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Precompiled gfx1201 SageAttention code object for the specialized sequence."""

import ctypes
import os

import torch

from aiter.jit.core import AITER_ASM_DIR

_runtime = None
_modules = {}


def _check(status):
    if status:
        raise RuntimeError(f"HIP module API failed: {status}")


def _function():
    global _runtime
    if _runtime is None:
        _runtime = ctypes.CDLL("libamdhip64.so")
        _runtime.hipModuleLoad.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
        ]
        _runtime.hipModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        _runtime.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [
            ctypes.c_uint
        ] * 7 + [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
    code_object = os.path.join(
        AITER_ASM_DIR, "gfx1201", "sage_attention", "attention_pv_early.hsaco"
    )
    key = (torch.cuda.current_device(), code_object)
    if key not in _modules:
        module = ctypes.c_void_p()
        function = ctypes.c_void_p()
        _check(_runtime.hipModuleLoad(ctypes.byref(module), os.fsencode(key[1])))
        _check(
            _runtime.hipModuleGetFunction(
                ctypes.byref(function), module, b"sage_hip_attn_bm128_bn32"
            )
        )
        _modules[key] = (module, function)
    return _modules[key][1]


def launch_hip_sage_core(
    q_int8,
    k_int8,
    v_fp8,
    q_scale,
    k_scale,
    v_scale,
    out,
    batch_size,
    padded_seq_len,
    valid_seq_len,
    num_heads,
):
    if (
        min(batch_size, valid_seq_len, num_heads) <= 0
        or padded_seq_len != (valid_seq_len + 31) // 32 * 32
    ):
        raise ValueError(
            "Expected positive shape and sequence padded to a multiple of 32"
        )
    if out.dtype != torch.bfloat16 or out.device != q_int8.device:
        raise ValueError("Expected BF16 output on the input device")
    with torch.cuda.device(q_int8.device):
        function = _function()
        arguments = [
            ctypes.c_void_p(tensor.data_ptr())
            for tensor in (q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, out)
        ]
        arguments += [
            ctypes.c_int(value)
            for value in (batch_size, padded_seq_len, valid_seq_len, num_heads)
        ]
        parameters = (ctypes.c_void_p * len(arguments))(
            *[ctypes.addressof(argument) for argument in arguments]
        )
        grid = batch_size * ((valid_seq_len + 511) // 512) * num_heads
        _check(
            _runtime.hipModuleLaunchKernel(
                function,
                grid,
                1,
                1,
                512,
                1,
                1,
                0,
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                parameters,
                None,
            )
        )

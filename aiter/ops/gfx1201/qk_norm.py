# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

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
        _runtime.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        _runtime.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        _runtime.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    code_object = os.path.join(
        AITER_ASM_DIR, "gfx1201", "sage_attention", "qk_norm_native.hsaco"
    )
    key = (torch.cuda.current_device(), code_object)
    if key not in _modules:
        module = ctypes.c_void_p()
        function = ctypes.c_void_p()
        _check(_runtime.hipModuleLoad(ctypes.byref(module), os.fsencode(key[1])))
        _check(_runtime.hipModuleGetFunction(ctypes.byref(function), module, b"_ZN2at6native12_GLOBAL__N_128vectorized_layer_norm_kernelIN3c108BFloat16EfLb1EEEviT0_PKT_S8_S8_PS5_S9_PS6_"))
        _modules[key] = (module, function)
    return _modules[key][1]



def apply(tensor, weight, eps=1e-5):
    if tensor.ndim < 1 or tensor.numel() == 0 or tensor.dtype != torch.bfloat16 or tensor.shape[-1] != 128 or not tensor.is_contiguous():
        raise ValueError("Contiguous BF16 D128 required")
    if weight.shape != (128,) or weight.dtype != tensor.dtype or weight.device != tensor.device or not weight.is_contiguous():
        raise ValueError("Same-device BF16 weight [128] required")
    if not tensor.is_cuda or tensor.requires_grad or weight.requires_grad or float(eps) != 1e-5:
        raise ValueError("Inference-only GPU input and eps=1e-5 required")
    if not torch.cuda.get_device_properties(tensor.device).gcnArchName.startswith("gfx1201"):
        raise ValueError("Native QK norm code object requires gfx1201")
    rows = tensor.numel() // 128
    if tensor.numel() >= 2**31:
        raise ValueError("QK norm requires int32 indexing")
    output = torch.empty_like(tensor)
    mean = torch.empty((rows,), device=tensor.device, dtype=torch.float32)
    reciprocal = torch.empty_like(mean)
    with torch.cuda.device(tensor.device):
        function = _function()
        arguments = [ctypes.c_int(128), ctypes.c_float(1e-5)]
        arguments += [ctypes.c_void_p(value) for value in
                      (tensor.data_ptr(), weight.data_ptr(), 0, mean.data_ptr(), reciprocal.data_ptr(), output.data_ptr())]
        parameters = (ctypes.c_void_p * len(arguments))(*[ctypes.addressof(argument) for argument in arguments])
        _check(_runtime.hipModuleLaunchKernel(
            function, rows, 1, 1, 32, 1, 1, 128,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), parameters, None))
    return output

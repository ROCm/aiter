# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
from packaging import version
from packaging.version import Version
import importlib


aiter_lib = None


def is_torch_equal_or_newer(target: str) -> bool:
    """Check if the installed torch version is >= the target version.

    Args:
        target: a version string, like "2.6.0".

    Returns:
        Whether the condition meets.
    """
    import torch

    try:
        return _is_torch_equal_or_newer(str(torch.__version__), target)
    except Exception:
        # Fallback to PKG-INFO to load the package info, needed by the doc gen.
        return Version(importlib.metadata.version("torch")) >= Version(target)


# Helper function used in testing.
def _is_torch_equal_or_newer(torch_version: str, target: str) -> bool:
    torch_version = version.parse(torch_version)
    return torch_version >= version.parse(target)


def torch_compile_guard(mutates_args: list[str] = [], device: str = "cpu"):
    def decorator(func):
        try:
            import torch
            from torch.library import Library
            import inspect
        except ImportError:

            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        global aiter_lib
        aiter_lib = Library("aiter", "FRAGMENT") if aiter_lib is None else aiter_lib
        op_name = func.__name__
        sig = inspect.signature(func)
        return_annotation = sig.return_annotation
        return_non_tensor = False
        # Only return int/bool/float will cause graph breaks
        if return_annotation in [int, bool, float]:
            return_non_tensor = True

        def outer_wrapper(*args, **kwargs):
            dummy = torch.empty(1, device=device)
            if return_non_tensor:
                result = getattr(torch.ops.aiter, op_name)(dummy, *args, **kwargs)
                _, int_value = result
                return int_value
            return getattr(torch.ops.aiter, op_name)(dummy, *args, **kwargs)

        if hasattr(torch.ops.aiter, func.__name__):
            return outer_wrapper
        if hasattr(torch.library, "infer_schema"):
            schema_str = torch.library.infer_schema(func, mutates_args=mutates_args)
        else:
            # for pytorch 2.4
            import torch._custom_op.impl

            schema_str = torch._custom_op.impl.infer_schema(
                func, mutates_args=mutates_args
            )

        input_part, output_part = schema_str.split("->", 1)
        if not sig.parameters:
            new_input = "(Tensor dummy)"
        else:
            new_input = "(Tensor dummy, " + input_part[1:]

        output_part = output_part.strip()
        if not return_non_tensor:
            new_output = output_part
        else:
            # return only int will cause graph breaks and we add dummy_out
            new_output = "(Tensor, " + output_part + ")"
        schema_str = f"{new_input} -> {new_output}".strip()

        def custom_impl(dummy_tensor, *args, **kwargs):
            out = torch.empty(1, device=device)
            if not return_non_tensor:
                return func(*args, **kwargs)
            return out, func(*args, **kwargs)

        def fake_impl(dummy_tensor, *args, **kwargs):
            out = torch.empty(1, device=device)
            if not return_non_tensor:
                return func(*args, **kwargs)
            default_values = {int: 0, bool: True, float: 0.0}

            default_value = default_values.get(return_annotation, None)
            return out, default_value

        if is_torch_equal_or_newer("2.8.0"):
            tags = ()
        else:
            tags = (torch.Tag.needs_fixed_stride_order,)

        my_lib = aiter_lib
        my_lib.define(op_name + schema_str, tags=tags)
        my_lib.impl(op_name, custom_impl, dispatch_key="CUDA")
        my_lib.impl(op_name, custom_impl, dispatch_key="CPU")
        my_lib._register_fake(op_name, fake_impl)

        return outer_wrapper

    return decorator

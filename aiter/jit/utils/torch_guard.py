# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
from packaging import version
from packaging.version import Version
import importlib
from typing import Any, Callable, Optional, Union, List, get_args, get_origin

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


MANUAL_SCHEMA_OPS = [
    "register_graph_buffers",
    "module_moe_ck2stages",
    "mha_fwd",
    "fmha_v3_fwd",
    "mha_varlen_fwd",
    "mha_bwd",
    "fmha_v3_bwd",
    "mha_varlen_bwd",
    "fmha_v3_varlen_bwd",
    "fmha_v3_varlen_fwd",
    "mha_batch_prefill",
    "hipb_findallsols",
    "rocb_findallsols",
    "_ActivationType",
    "_QuantType",
    "init_custom_ar",
    "greedy_sample",
    "random_sample",
    "mixed_sample",
    "exponential",
]


NONE_WRAPPED_OP = [
    # "hipb_create_extension",
    # "hipb_destroy_extension",
    "getHipblasltKernelName",
    # "rocb_create_extension",
    # "rocb_destroy_extension",
    "get_graph_buffer_ipc_meta",
    "_ActivationType",
    "_QuantType",
    # "dispose",
    # "meta_size",
    # "get_padded_m",
    "compile_mha_fwd",
    "compile_mha_bwd",
    "init_custom_qr",
    "qr_max_size",
    "qr_destroy",
    "qr_open_handles",
    "qr_get_handle",
]


def generate_schema(func, mutates_args: Union[list[str], str] = "unknown") -> str:
    import inspect

    import torch

    sig = inspect.signature(func)
    parameters = []
    for idx, (name, param) in enumerate(sig.parameters.items()):
        param_type = param.annotation
        flag = True
        is_mutates = True
        if mutates_args != "unknown" and name not in mutates_args:
            is_mutates = False

        if param_type is torch.Tensor:
            if is_mutates:
                type_str = f"Tensor(a{idx}!)"
            else:
                type_str = "Tensor"
        elif param_type == Optional[torch.Tensor]:
            if is_mutates:
                type_str = f"Tensor(a{idx}!)?"
            else:
                type_str = "Tensor?"
        elif get_origin(param_type) is Union and torch.Tensor in get_args(param_type):
            if is_mutates:
                type_str = f"Tensor(a{idx}!)?"
            else:
                type_str = "Tensor?"
        elif param_type in (torch.SymInt, int):
            type_str = "SymInt"
        elif param_type in (float, bool, str):
            type_str = param_type.__name__
        elif param_type == Optional[torch.Generator]:
            type_str = "Generator?"
        elif (
            get_origin(param_type) in (list, List)
            and get_args(param_type)[0] is torch.Tensor
        ):
            if is_mutates:
                type_str = f"Tensor(a{idx}!)[]"
            else:
                type_str = "Tensor[]"
        elif get_origin(param_type) in (list, List) and get_args(param_type)[0] is int:
            type_str = "int[]"
        elif param_type == Optional[torch.dtype]:
            type_str = "ScalarType?"
        else:
            type_str = "*"
            flag = False
        if flag:
            param_str = f"{type_str} {name}"

            if param.default != inspect.Parameter.empty:
                if param.default is None:
                    param_str += "=None"
                else:
                    param_str += f"={param.default}"
        else:
            param_str = f"{type_str} "

        parameters.append(param_str)
    return_annotation = sig.return_annotation
    return_type = ""
    if return_annotation is type(None) or return_annotation is None:
        return_type = "()"
    elif return_annotation is torch.Tensor:
        return_type = "Tensor"
    elif (
        get_origin(return_annotation) is list and get_args(return_annotation)[0] is int
    ):
        return_type = "int[]"
    elif return_annotation is int:
        return_type = "int"
    elif return_annotation is float:
        return_type = "float"
    elif return_annotation is bool:
        return_type = "bool"
    elif (
        get_origin(return_annotation) is list
        and get_args(return_annotation)[0] is torch.Tensor
    ):
        return_type = "Tensor[]"
    elif get_origin(return_annotation) is tuple:
        args = get_args(return_annotation)
        type_strings = []
        for arg in args:
            if arg is torch.Tensor:
                type_strings.append("Tensor")
            elif arg is int:
                type_strings.append("int")
            elif arg is float:
                type_strings.append("float")
            elif arg is bool:
                type_strings.append("bool")
        return_type = f"({', '.join(type_strings)})"
    else:
        return_type = "Any"

    schema = f"({', '.join(parameters)}) -> {return_type}"

    return schema


def torch_compile_guard(
    mutates_args: Union[list[str], str] = "unknown",
    device: str = "cpu",
    calling_func_: Optional[Callable[..., Any]] = None,
    gen_fake: Optional[Callable[..., Any]] = None,
):
    def decorator(func):
        # In core.py, we calling wrapper, but actually we need use aiter.op func
        calling_func = calling_func_ if calling_func_ is not None else func

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        try:
            import torch
            from torch.library import Library
            import inspect
        except ImportError:
            return wrapper

        if calling_func.__name__ in NONE_WRAPPED_OP:
            return wrapper

        def wrapper_register(calling_func):
            import inspect

            import torch
            import torch.library
            from torch.library import Library

            global aiter_lib
            aiter_lib = Library("aiter", "FRAGMENT") if aiter_lib is None else aiter_lib
            schema = ""
            if calling_func.__name__ in MANUAL_SCHEMA_OPS:
                schema = generate_schema(calling_func)
            else:
                sig = inspect.signature(calling_func)
                if hasattr(torch.library, "infer_schema"):
                    schema = torch.library.infer_schema(
                        calling_func, mutates_args=mutates_args
                    )
                else:
                    # for pytorch 2.4
                    import torch._custom_op.impl

                    # torch 2.4 not support mutates "unknown" for inplace all param
                    if mutates_args == "unknown":
                        mutates_args_custom = []

                        for param_name, param in sig.parameters.items():
                            if param.annotation == torch.Tensor:
                                mutates_args_custom.append(param_name)

                    schema = torch._custom_op.impl.infer_schema(
                        calling_func, mutates_args_custom
                    )
            return schema

        schema = wrapper_register(calling_func)

        sig = inspect.signature(calling_func)
        input_is_tensor = False
        parameters = list(sig.parameters.values())

        if parameters:
            first_param = parameters[0]
            if (
                first_param.annotation is not inspect.Parameter.empty
                and first_param.annotation is torch.Tensor
            ):
                input_is_tensor = True

        input_part, output_part = schema.split("->", 1)
        if input_is_tensor:
            new_input = input_part
        else:
            if not sig.parameters:
                new_input = "(Tensor dummy)"
            else:
                new_input = "(Tensor dummy, " + input_part[1:]

        return_non_tensor = False
        return_annotation = sig.return_annotation
        if return_annotation in [int, bool, float]:
            output_part = "(Tensor, " + output_part + ")"
            return_non_tensor = True

        schema = f"{new_input} -> {output_part}".strip()

        loadName = calling_func.__name__

        def wrapper_custom(*args, **kwargs):
            result = (
                getattr(torch.ops.aiter, f"{loadName}")(*args, **kwargs)
                if input_is_tensor
                else getattr(torch.ops.aiter, f"{loadName}")(
                    torch.empty(1, device=device), *args, **kwargs
                )
            )
            return result[1] if return_non_tensor else result

        if hasattr(torch.ops.aiter, loadName):
            return wrapper_custom

        def abstract_impl(*args, **kwargs):
            if gen_fake is not None:
                if return_non_tensor:
                    return torch.empty(1, device=device), gen_fake(*args, **kwargs)
                else:
                    return gen_fake(*args, **kwargs)
            if return_non_tensor:
                return torch.empty(1, device=device), calling_func(*args, **kwargs)
            return calling_func(*args, **kwargs)

        def outer_wrapper(*args, **kwargs):
            return (
                wrapper(*args, **kwargs)
                if not return_non_tensor
                else (torch.empty(1, device=device), wrapper(*args, **kwargs))
            )

        def abstract_impl_dummy(dummy, *args, **kwargs):
            if gen_fake is not None:
                if return_non_tensor:
                    return torch.empty(1, device=device), gen_fake(*args, **kwargs)
                else:
                    return gen_fake(*args, **kwargs)
            if return_non_tensor:
                return torch.empty(1, device=device), calling_func(*args, **kwargs)
            return calling_func(*args, **kwargs)

        def outer_wrapper_dummy(dummy, *args, **kwargs):
            return (
                wrapper(*args, **kwargs)
                if not return_non_tensor
                else (torch.empty(1, device=device), wrapper(*args, **kwargs))
            )

        custom_func = outer_wrapper
        fake_func = abstract_impl
        if not input_is_tensor:
            custom_func = outer_wrapper_dummy
            fake_func = abstract_impl_dummy

        if not hasattr(torch.ops.aiter, calling_func.__name__):
            if is_torch_equal_or_newer("2.8.0"):
                tags = ()
            else:
                tags = (torch.Tag.needs_fixed_stride_order,)
            op_schema = f"aiter::{loadName}" + schema
            aiter_lib.define(op_schema, tags=tags)
            aiter_lib.impl(f"aiter::{loadName}", custom_func, dispatch_key="CUDA")
            aiter_lib.impl(f"aiter::{loadName}", custom_func, dispatch_key="CPU")
            aiter_lib._register_fake(f"{loadName}", fake_func)

        return wrapper_custom

    return decorator

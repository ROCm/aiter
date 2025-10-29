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

# We default all args are inplace, you can define inplace args for specific op
SPECIAL_OPS_MUTATES_ARGS = {}


def generate_schema(func) -> str:
    import inspect

    import torch

    sig = inspect.signature(func)
    parameters = []
    mutates_args = SPECIAL_OPS_MUTATES_ARGS.get(func.__name__, [])
    for idx, (name, param) in enumerate(sig.parameters.items()):
        param_type = param.annotation
        flag = True
        is_mutates = True
        if len(mutates_args) > 0 and name not in mutates_args:
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
    mutates_args: list[str] = [],
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
                mutates_args = SPECIAL_OPS_MUTATES_ARGS.get(
                    calling_func.__name__, "unknown"
                )
                if hasattr(torch.library, "infer_schema"):
                    sig = torch.library.infer_schema(
                        calling_func, mutates_args=mutates_args
                    )
                else:
                    # for pytorch 2.4
                    import torch._custom_op.impl

                    # torch 2.4 not support mutates "unknown" for inplace all param
                    if mutates_args == "unknown":
                        mutates_args = []

                        for param_name, param in sig.parameters.items():
                            if param.annotation == torch.Tensor:
                                mutates_args.append(param_name)

                    sig = torch._custom_op.impl.infer_schema(calling_func, mutates_args)
                schema = f"{sig}"
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

        return_int = False
        return_annotation = sig.return_annotation
        if return_annotation is int:
            output_part = "(Tensor, " + output_part + ")"
            return_int = True

        schema = f"{new_input} -> {output_part}".strip()

        loadName = calling_func.__name__

        def abstract_impl(*args, custom_build_args={}, **kwargs):
            if return_int:
                return torch.empty(1, device="cuda"), 1
            if gen_fake is not None:
                return gen_fake(*args, **kwargs)
            return func(*args, **kwargs)

        def outer_wrapper(*args, **kwargs):
            return (
                wrapper(*args, **kwargs)
                if not return_int
                else (torch.empty(1, device="cuda"), wrapper(*args, **kwargs))
            )

        def abstract_impl_dummy(dummy, *args, custom_build_args={}, **kwargs):
            if return_int:
                return torch.empty(1, device="cuda"), 1
            if gen_fake is not None:
                return gen_fake(*args, **kwargs)
            return func(*args, **kwargs)

        def outer_wrapper_dummy(dummy, *args, **kwargs):
            return (
                wrapper(*args, **kwargs)
                if not return_int
                else (torch.empty(1, device="cuda"), wrapper(*args, **kwargs))
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

        def wrapper_custom(*args, custom_build_args={}, **kwargs):
            result = (
                getattr(torch.ops.aiter, f"{loadName}")(*args, **kwargs)
                if input_is_tensor
                else getattr(torch.ops.aiter, f"{loadName}")(
                    torch.empty(1, device="cuda"), *args, **kwargs
                )
            )
            return result[1] if return_int else result

        return wrapper_custom

    return decorator

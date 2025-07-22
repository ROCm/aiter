import torch
import ctypes
from torch.library import Library
from typing import Callable, Optional, Tuple
from csrc.cpp_itfs.utils import AITER_LOG_MORE
from aiter.test_common import log_args


ctypes_map = {
    int: ctypes.c_int,
    float: ctypes.c_float,
    bool: ctypes.c_bool,
    str: ctypes.c_char_p,
}

aiter_lib = Library("aiter", "FRAGMENT")


def torch_to_c_types(*args):
    c_args = []
    for arg in args:
        if arg is None:
            c_args.append(ctypes.POINTER(ctypes.c_int)())
        elif isinstance(arg, torch.Tensor):
            c_args.append(ctypes.cast(arg.data_ptr(), ctypes.c_void_p))
        elif isinstance(arg, torch.cuda.Stream):
            c_args.append(ctypes.cast(arg.cuda_stream, ctypes.c_void_p))
        else:
            if type(arg) not in ctypes_map:
                raise ValueError(f"Unsupported type: {type(arg)}")
            c_args.append(ctypes_map[type(arg)](arg))
    return c_args


hip_types_map = {
    torch.bfloat16: "__hip_bfloat16",
    torch.float16: "_Float16",
    torch.int8: "int8_t",
    torch.uint8: "uint8_t",
    torch.float8_e4m3fnuz: "__hip_fp8_e4m3_fnuz",
    torch.uint32: "uint32_t",
    torch.int32: "int32_t",
    torch.uint16: "uint16_t",
    torch.int16: "int16_t",
    torch.float: "float",
    torch.float32: "float",
}


def torch_to_hip_types(*types):
    return [hip_types_map[t] for t in types]


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str],
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
    dispatch_key: str = "CUDA",
    tags: Tuple[torch.Tag, ...] = (),
):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.

    By default, the custom op is registered to the vLLM library. If you
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: the lifetime of the operator is tied to the lifetime of the
    library object. If you want to bind the operator to a different library,
    make sure the library object is alive when the operator is used.
    """
    import torch.library

    def _op_func(*args, **kwargs):
        if AITER_LOG_MORE >= 2:
            log_args(op_func, *args, **kwargs)
        return op_func(*args, **kwargs)

    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl

        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)
    my_lib = target_lib or aiter_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, _op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)

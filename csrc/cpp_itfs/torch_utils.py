import torch
import ctypes

ctypes_map = {
    int: ctypes.c_int,
    float: ctypes.c_float,
    bool: ctypes.c_bool,
    str: ctypes.c_char_p,
}


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

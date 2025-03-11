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
        if isinstance(arg, torch.Tensor):
            c_args.append(ctypes.cast(arg.data_ptr(), ctypes.c_void_p))
        else:
            if type(arg) not in ctypes_map:
                raise ValueError(f"Unsupported type: {type(arg)}")
            c_args.append(ctypes_map[type(arg)](arg))
    return c_args
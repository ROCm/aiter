import torch
import triton

# For now, there is 1-to-1 correspondence between arch and device
_ARCH_TO_DEVICE = {
    "gfx942": "MI300X",
    "gfx950": "MI350X",
}

def get_arch():
    return triton.runtime.driver.active.get_current_target().arch

def get_device():
    return _ARCH_TO_DEVICE[get_arch()]

def arch_supports_fp4():
    return get_arch() in ("gfx950")

def arch_supports_fp8():
    return get_arch() in ("gfx942", "gfx950")

def is_fp4_avail():
    return get_arch() in ("gfx950")

def is_fp8_avail():
    return get_arch() in ("gfx942", "gfx950")

def get_fp8_dtypes():
    if get_arch() in ("gfx950"):
        e5m2_dtype = torch.float8_e5m2
        e4m3_dtype = torch.float8_e4m3fn
    else:
        e5m2_dtype = torch.float8_e5m2fnuz
        e4m3_dtype = torch.float8_e4m3fnuz

    return e5m2_dtype, e4m3_dtype


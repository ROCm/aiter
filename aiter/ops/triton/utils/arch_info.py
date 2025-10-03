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


def get_fp8_e4m3_dtype():
    if get_arch() in ("gfx950"):
        e4m3_dtype = torch.float8_e4m3fn
    else:
        e4m3_dtype = torch.float8_e4m3fnuz

    return e4m3_dtype


def get_num_sms():
    # Returns the Compute Unit count of the current device
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    num_sms = current_device.multi_processor_count
    return num_sms

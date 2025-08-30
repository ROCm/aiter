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

import triton

# For now, there is 1-to-1 correspondence between arch and device
ARCH_TO_DEVICE = {
    "gfx942": "MI300X",
    "gfx950": "MI350X",
}


def get_arch():
    return triton.runtime.driver.active.get_current_target().arch


def get_device():
    return ARCH_TO_DEVICE[get_arch()]


def arch_supports_fp4():
    return get_arch() in ("gfx950")


def arch_supports_fp8():
    return get_arch() in ("gfx942", "gfx950")

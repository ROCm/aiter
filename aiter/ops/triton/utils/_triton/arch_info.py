import triton
from functools import lru_cache

# For now, there is 1-to-1 correspondence between arch and device
_ARCH_TO_DEVICE = {
    "gfx942": "MI300X",
    "gfx950": "MI350X",
}


@lru_cache(maxsize=1)
def get_arch():
    try:
        arch = (
            triton.runtime.driver.active.get_current_target().arch
        )  # If running with torch
    except RuntimeError:  # else running with JAX
        from jax._src.lib import gpu_triton as triton_kernel_call_lib

        arch = triton_kernel_call_lib.get_arch_details("0")
        arch = arch.split(":")[0]

    return arch


@lru_cache(maxsize=1)
def get_device():
    return _ARCH_TO_DEVICE[get_arch()]


def is_fp4_avail():
    return get_arch() in ("gfx950")


def is_fp8_avail():
    return get_arch() in ("gfx942", "gfx950")

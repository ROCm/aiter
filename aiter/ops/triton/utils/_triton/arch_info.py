import triton
from functools import lru_cache


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


def is_fp4_avail():
    return get_arch() in ("gfx950")


def is_fp8_avail():
    return get_arch() in ("gfx942", "gfx950")

import functools
import os

import triton

try:
    _CACHED_ARCH = triton.runtime.driver.active.get_current_target().arch
except RuntimeError:
    from jax._src.lib import gpu_triton as triton_kernel_call_lib

    _CACHED_ARCH = triton_kernel_call_lib.get_arch_details("0").split(":")[0]


def get_arch():
    return _CACHED_ARCH


@functools.lru_cache(maxsize=1)
def _get_device_cu_count():
    try:
        driver = triton.runtime.driver.active
        props = driver.utils.get_device_properties(driver.get_current_device())
        return int(props["multiprocessor_count"])
    except Exception:  # noqa: BLE001
        return None


def get_cu_count():
    """CU count of the current device (CU_NUM overrides), or None if unknown."""
    cu_num = int(os.environ.get("CU_NUM", "0") or 0)
    if cu_num > 0:
        return cu_num
    return _get_device_cu_count()


def is_gluon_avail():
    return get_arch() in ("gfx950", "gfx1250")


def is_fp4_avail():
    return get_arch() in ("gfx950", "gfx1250")


def is_fp8_avail():
    return get_arch() in ("gfx942", "gfx950", "gfx1250", "gfx1200", "gfx1201")


def is_mx_scale_preshuffling_avail():
    return get_arch() in ("gfx950", "gfx1250")


def is_tdm_avail():
    return get_arch() in ("gfx1250",)


_LDS_CAP_BYTES = {"gfx1250": 327680, "gfx950": 163840, "gfx942": 65536}

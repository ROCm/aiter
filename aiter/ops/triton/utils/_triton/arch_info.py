import contextlib

import triton

try:
    _CACHED_ARCH = triton.runtime.driver.active.get_current_target().arch
except RuntimeError:
    from jax._src.lib import gpu_triton as triton_kernel_call_lib

    _CACHED_ARCH = triton_kernel_call_lib.get_arch_details("0").split(":")[0]


def get_arch():
    return _CACHED_ARCH


@contextlib.contextmanager
def no_async_copy_on_gfx1250():
    # On gfx1250, async copy lowers masked loads to global_load_async_to_lds,
    # which has no bounds check and faults the GPU. The knob is part of Triton's
    # compile cache key, so kernels launched outside this scope keep async copy.
    amd_knobs = getattr(triton.knobs, "amd", None)
    if get_arch() != "gfx1250" or not hasattr(amd_knobs, "use_async_copy"):
        yield
        return
    with amd_knobs.scope():
        amd_knobs.use_async_copy = False
        yield


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

try:
    import triton
except ImportError:
    triton = None


def _detect_arch():
    if triton is not None:
        try:
            return triton.runtime.driver.active.get_current_target().arch
        except Exception:
            pass

    try:
        from jax._src.lib import gpu_triton as triton_kernel_call_lib

        return triton_kernel_call_lib.get_arch_details("0").split(":")[0]
    except Exception:
        pass

    return None


_CACHED_ARCH = _detect_arch()


def get_arch():
    return _CACHED_ARCH


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

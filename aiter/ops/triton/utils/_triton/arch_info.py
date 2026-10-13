import triton


def _probe_arch():
    try:
        return triton.runtime.driver.active.get_current_target().arch
    except RuntimeError:
        from jax._src.lib import gpu_triton as triton_kernel_call_lib

        return triton_kernel_call_lib.get_arch_details("0").split(":")[0]


# Probed at import so torch.compile sees a constant instead of tracing into the
# driver. Modules that only import this one (the flash-attention backend on a
# GPU-less host, for one) must not need a driver, so a failed probe is retried
# on first use, where the error surfaces.
try:
    _CACHED_ARCH = _probe_arch()
except Exception:  # noqa: BLE001
    _CACHED_ARCH = None


def get_arch():
    global _CACHED_ARCH
    if _CACHED_ARCH is None:
        _CACHED_ARCH = _probe_arch()
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

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


def is_sw_emulated_fp8_avail():
    # No fp8 WMMA or cvt instructions: Triton converts fp8 operands to fp16
    # in software and runs the fp16 WMMA. The archs match get_fp8_dtypes()
    # in utils/types.py, and each FP8 entry point opts in separately once
    # it has configs and a passing test.
    return get_arch() in ("gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151")


def is_any_fp8_avail():
    # FP8 through hardware or through software emulation.
    return is_fp8_avail() or is_sw_emulated_fp8_avail()


def is_mx_scale_preshuffling_avail():
    return get_arch() in ("gfx950", "gfx1250")


def is_tdm_avail():
    return get_arch() in ("gfx1250",)


_LDS_CAP_BYTES = {"gfx1250": 327680, "gfx950": 163840, "gfx942": 65536}

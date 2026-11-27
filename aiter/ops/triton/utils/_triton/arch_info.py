import os

import triton

_KFD_NODES = "/sys/class/kfd/kfd/topology/nodes"


def _probe_kfd_arch():
    # KFD sysfs gives the arch without starting HIP, so the process can still
    # fork. Use it only when all GPUs have one arch, because then the device
    # order and the visible devices do not change the result.
    if os.environ.get("TRITON_OVERRIDE_ARCH") or os.environ.get(
        "HSA_OVERRIDE_GFX_VERSION"
    ):
        return None
    archs = set()
    try:
        for node in os.listdir(_KFD_NODES):
            with open(os.path.join(_KFD_NODES, node, "properties")) as f:
                props = dict(line.split() for line in f if line.strip())
            ver = int(props.get("gfx_target_version", 0))
            if ver:
                archs.add(f"gfx{ver // 10000}{ver // 100 % 100}{ver % 100:x}")
    except (OSError, ValueError):
        return None
    # The sysfs version cannot tell gfx1250 from gfx1250-strict.
    if len(archs) != 1 or "gfx1250" in archs:
        return None
    return archs.pop()


def _probe_arch():
    arch = _probe_kfd_arch()
    if arch is not None:
        return arch
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
    return get_arch() in ("gfx950", "gfx1250", "gfx1250-strict")


def is_fp4_avail():
    return get_arch() in ("gfx950", "gfx1250", "gfx1250-strict")


def is_sw_emulated_fp4_avail():
    # No native FP4 instructions: Triton lowers tl.dot_scaled by upcasting the
    # operands to bf16. Only the Triton backend runs here, and each FP4 entry
    # point opts in separately once it has configs and a passing test.
    return get_arch() in ("gfx1101", "gfx1150", "gfx1151")


def is_any_fp4_avail():
    # FP4 through hardware or through software emulation.
    return is_fp4_avail() or is_sw_emulated_fp4_avail()


def is_fp8_avail():
    return get_arch() in (
        "gfx942",
        "gfx950",
        "gfx1250",
        "gfx1250-strict",
        "gfx1200",
        "gfx1201",
    )


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
    return get_arch() in ("gfx950", "gfx1250", "gfx1250-strict")


def is_tdm_avail():
    return get_arch() in ("gfx1250", "gfx1250-strict")


def is_cdna3():
    return get_arch() == "gfx942"


def is_cdna4():
    return get_arch() == "gfx950"


_LDS_CAP_BYTES = {
    "gfx1250": 327680,
    "gfx1250-strict": 327680,
    "gfx950": 163840,
    "gfx942": 65536,
}

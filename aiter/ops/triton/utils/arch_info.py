import functools

import triton

try:
    _CACHED_ARCH = triton.runtime.driver.active.get_current_target().arch
except RuntimeError:
    from jax._src.lib import gpu_triton as triton_kernel_call_lib

    _CACHED_ARCH = triton_kernel_call_lib.get_arch_details("0").split(":")[0]


def get_arch():
    return _CACHED_ARCH


CDNA_ARCHS = ("gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950")
# NOTE: gfx1250 is intentionally not listed here; it has its own tuning paths.
RDNA_ARCHS = (
    "gfx1030",
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1150",
    "gfx1151",
    "gfx1200",
    "gfx1201",
)

# LDS (shared memory) a single workgroup can allocate, in bytes.
# amdclang++ enforces per target.
_LDS_CAP_BYTES = {
    # CDNA
    "gfx908": 65536,
    "gfx90a": 65536,
    "gfx940": 65536,
    "gfx941": 65536,
    "gfx942": 65536,
    "gfx950": 163840,
    # RDNA
    "gfx1030": 65536,
    "gfx1100": 65536,
    "gfx1101": 65536,
    "gfx1102": 65536,
    "gfx1150": 65536,
    "gfx1151": 65536,
    "gfx1200": 65536,
    "gfx1201": 65536,
    "gfx1250": 327680,
}


def is_cdna():
    return get_arch() in CDNA_ARCHS


def is_rdna():
    return get_arch() in RDNA_ARCHS


def is_fp4_avail():
    return get_arch() in ("gfx950", "gfx1250")


def is_fp8_avail():
    return get_arch() in ("gfx942", "gfx950", "gfx1250", "gfx1200", "gfx1201")


def is_mx_scale_preshuffling_avail():
    return get_arch() in ("gfx950", "gfx1250")


def is_tdm_avail():
    return get_arch() in ("gfx1250",)


def get_lds_cap_bytes(arch=None):
    """LDS bytes one workgroup can allocate on `arch` (default: the current arch)."""
    arch = get_arch() if arch is None else arch
    if arch not in _LDS_CAP_BYTES:
        raise ValueError(
            f"No LDS capacity defined for arch {arch!r}; add it to _LDS_CAP_BYTES."
        )
    return _LDS_CAP_BYTES[arch]


@functools.lru_cache(maxsize=1)
def get_num_sms():
    # Returns the Compute Unit count of the device.
    #
    # Prefer chip_info.get_cu_num(): it honors the CU_NUM env override and is the
    # same value the tuning dispatch keys (gfx, cu_num, M, N, K) are built from,
    # so grid/segment sizing stays consistent with the selected tuned configs.
    # Fall back to torch's multi_processor_count when get_cu_num() is unavailable
    # (e.g. rocminfo missing/unparseable).
    try:
        from aiter.jit.utils.chip_info import get_cu_num

        return get_cu_num()
    except Exception as chip_info_err:  # noqa: BLE001
        try:
            import torch
        except ImportError as torch_err:
            raise RuntimeError(
                "Cannot determine the device CU count: chip_info.get_cu_num() failed "
                f"({chip_info_err!r}) and torch is not installed."
            ) from torch_err

        current_device_index = torch.cuda.current_device()
        current_device = torch.cuda.get_device_properties(current_device_index)
        return current_device.multi_processor_count


def get_num_xcds():
    # Currently, you can't query this programmatically. For gfx942/gfx950 it's 8, so we hardcode that here.
    return 8

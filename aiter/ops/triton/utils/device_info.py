import functools

# hipDeviceAttributeNumberOfXccs. AMD-specific attributes start at
# hipDeviceAttributeAmdSpecificBegin (10000); this one is 10018.
# ROCm before 7 does not expose it.
_HIP_DEVICE_ATTRIBUTE_NUMBER_OF_XCCS = 10018


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
    except Exception:  # noqa: BLE001
        import torch

        current_device_index = torch.cuda.current_device()
        current_device = torch.cuda.get_device_properties(current_device_index)
        return current_device.multi_processor_count


# Kernels use the XCD count to keep workgroups that share data on one L2.
# ROCm 7+ reports the live count through hipDeviceAttributeNumberOfXccs,
# including partition mode. Older runtimes do not, so those releases keep
# this table. Parts that are not listed still return 8.
_XCDS_BY_GFX_AND_CU_NUM = {
    ("gfx942", 304): 8,
    ("gfx942", 228): 6,
    ("gfx942", 80): 4,
    ("gfx950", 256): 8,
}


def _xcds_for_part(gfx: str, cu_num: int) -> int:
    return _XCDS_BY_GFX_AND_CU_NUM.get((gfx, cu_num), 8)


def _xcds_from_known_part() -> int:
    try:
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

        return _xcds_for_part(get_gfx_runtime(), get_cu_num())
    except Exception:  # noqa: BLE001
        return 8


def _rocm_major():
    try:
        from aiter.dist.device_communicators.rocm_version import get_rocm_version

        version = get_rocm_version()
    except Exception:  # noqa: BLE001
        return None
    if not version:
        return None
    return int(version[0])


def _current_device_id() -> int:
    import torch

    return int(torch.cuda.current_device())


@functools.lru_cache(maxsize=8)
def _query_num_xccs(device_id: int) -> int:
    import ctypes

    from aiter.jit.utils.chip_info import _load_hip_runtime

    libhip = _load_hip_runtime()
    xcc_count = ctypes.c_int()
    err = libhip.hipDeviceGetAttribute(
        ctypes.byref(xcc_count),
        _HIP_DEVICE_ATTRIBUTE_NUMBER_OF_XCCS,
        int(device_id),
    )
    if err != 0:
        raise RuntimeError(
            f"hipDeviceGetAttribute(NumberOfXccs) failed with error {err}"
        )
    return int(xcc_count.value)


def get_num_xcds(device_id=None) -> int:
    """Return the number of XCDs visible on ``device_id``.

    On ROCm 7 and newer this is ``hipDeviceAttributeNumberOfXccs``. Earlier
    releases use the architecture and CU count, and return 8 when that part
    is not in the table. An undetected ROCm version tries the query first.
    """
    major = _rocm_major()
    if major is None or major >= 7:
        try:
            if device_id is None:
                device_id = _current_device_id()
            count = _query_num_xccs(int(device_id))
        except Exception:  # noqa: BLE001
            count = 0
        if count > 0:
            return count
    return _xcds_from_known_part()

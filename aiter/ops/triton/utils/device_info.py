import functools

# hipDeviceAttributeNumberOfXccs. The C enum is not available to ctypes, so
# the value is the one from hip_runtime_api.h: AMD-specific attributes begin
# at 10000, and hipDeviceAttributeClockInstructionRate shares that value, so
# this enumerator is 10018. On ROCm 7.2.4 the query returns the XCC count.
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
# this table. Every other arch and every unlisted CU count returns 1.
# 0 is not a usable count: callers divide and modulo by it.
_XCDS_BY_GFX_AND_CU_NUM = {
    ("gfx942", 304): 8,  # MI300X, MI325X: 8 XCDs x 38 CUs
    ("gfx942", 228): 6,  # MI300A: 6 XCDs x 38 CUs
    ("gfx950", 256): 8,  # MI350X, MI355X: 8 XCDs x 32 CUs
    ("gfx950", 128): 4,  # MI350P: 4 XCDs x 32 CUs
    ("gfx1250", 256): 8,  # MI455X: 8 XCDs x 32 WGPs
}


def _xcds_for_part(gfx: str, cu_num: int) -> int:
    return _XCDS_BY_GFX_AND_CU_NUM.get((gfx, cu_num), 1)


def _xcds_from_known_part() -> int:
    try:
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

        return _xcds_for_part(get_gfx_runtime(), get_cu_num())
    except Exception:  # noqa: BLE001
        return 1


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


def _hip_num_xccs(device_id: int) -> int:
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


@functools.cache
def _query_num_xccs(device_id: int) -> int:
    # Cache a rejection as 0. functools.cache does not store exceptions, and this
    # function is called from kernel launch paths. A runtime that rejects the
    # attribute will reject it for the rest of the process, so repeating the
    # driver call on every launch is wasted work. The following table lookup
    # is already cached. A successful count is cached the same way.
    try:
        count = _hip_num_xccs(int(device_id))
    except Exception:  # noqa: BLE001
        return 0
    return max(0, count)


def get_num_xcds(device_id=None) -> int:
    """Return the number of XCDs visible on ``device_id``.

    On ROCm 7 and newer this is ``hipDeviceAttributeNumberOfXccs``. Earlier
    releases use the architecture and CU count. Only the CDNA3, CDNA4, and
    CDNA5 parts in the table have a known XCD count. Every other case returns
    1, because callers divide by the count and one L2 domain is an identity
    remap. An undetected ROCm version tries the query first.
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

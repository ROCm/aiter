import functools

import torch


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


def get_num_xcds(device_index: int | None = None) -> int:
    """Return the XCC count for a HIP device, defaulting to the current device."""
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _get_num_xcds(device_index)


@functools.cache
@torch.compiler.assume_constant_result
def _get_num_xcds(device_index: int) -> int:
    import ctypes

    try:
        libhip = ctypes.CDLL("libamdhip64.so")
    except OSError:
        # Python ROCm SDK packages expose the versioned HIP 7 library.
        libhip = ctypes.CDLL("libamdhip64.so.7")
    get_attribute = libhip.hipDeviceGetAttribute
    get_attribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    get_attribute.restype = ctypes.c_int
    # hipDeviceAttributeNumberOfXccs, available since ROCm 7.0.
    num_xccs_attribute = 10018
    num_xccs = ctypes.c_int()
    error = get_attribute(ctypes.byref(num_xccs), num_xccs_attribute, device_index)
    if error != 0:
        raise RuntimeError(
            "hipDeviceGetAttribute(NumberOfXccs) failed for device "
            f"{device_index} with error {error}"
        )
    if num_xccs.value < 1:
        raise RuntimeError(
            "hipDeviceGetAttribute(NumberOfXccs) returned an invalid count "
            f"{num_xccs.value} for device {device_index}"
        )
    return num_xccs.value

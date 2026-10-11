import functools


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


# XCDs (accelerator complex dies, each with its own L2) per part. ROCm has no query for
# the count, but the CU count identifies the part: MI300X 304, MI300A 228, MI308X 80,
# MI350X/MI355X 256. Kernels use it to keep workgroups sharing data on one L2.
_XCDS_BY_GFX_AND_CU_NUM = {
    ("gfx942", 304): 8,
    ("gfx942", 228): 6,
    ("gfx942", 80): 4,
    ("gfx950", 256): 8,
}


def get_num_xcds() -> int:
    try:
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

        part = (get_gfx_runtime(), get_cu_num())
    except Exception:  # noqa: BLE001
        return 8
    return _XCDS_BY_GFX_AND_CU_NUM.get(part, 8)

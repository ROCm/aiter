import functools

# Re-exported so existing Triton importers keep working.
from aiter.jit.utils.chip_info import get_num_xcds  # noqa: F401


@functools.cache
def _num_sms_from_torch(device_id: int) -> int:
    import torch

    return torch.cuda.get_device_properties(device_id).multi_processor_count


def get_num_sms() -> int:
    # Returns the Compute Unit count of the device this thread is bound to.
    #
    # Prefer chip_info.get_cu_num(): it honors the CU_NUM env override and is the
    # same value the tuning dispatch keys (gfx, cu_num, M, N, K) are built from,
    # so grid/segment sizing stays consistent with the selected tuned configs.
    # It is memoized there, and deliberately not memoized again here: AOT clears
    # that cache after overriding CU_NUM, and a second copy would go stale.
    # Fall back to torch's multi_processor_count when get_cu_num() is unavailable
    # (e.g. rocminfo missing/unparseable, or agents that disagree on the count).
    try:
        from aiter.jit.utils.chip_info import get_cu_num

        return get_cu_num()
    except Exception:  # noqa: BLE001
        import torch

        return _num_sms_from_torch(torch.cuda.current_device())

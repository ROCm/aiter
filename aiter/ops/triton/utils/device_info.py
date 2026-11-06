import functools
import os


@functools.cache
def _chip_info_num_xcds():
    # This import must happen lazily: chip_info needs aiter/jit/utils on
    # sys.path, which aiter.jit.core adds, and AITER_TRITON_ONLY=1 does not
    # import aiter.jit.core.
    try:
        from aiter.jit.utils.chip_info import get_num_xcds
    except ImportError:
        return None
    return get_num_xcds


def get_num_xcds(default: int | None = 8) -> int:
    """XCD count of the current device, as chip_info.get_num_xcds reports it.

    Returns the default when chip_info cannot be imported, or raises if None.
    """
    chip_info_num_xcds = _chip_info_num_xcds()
    if chip_info_num_xcds is None:
        if default is None:
            raise RuntimeError(
                "chip_info cannot be imported, so the XCD count is unknown."
            )
        return default
    return chip_info_num_xcds(default)


_cu_num_failed: set[str | None] = set()


@functools.cache
def _num_sms_from_torch(device_id: int) -> int:
    import torch

    return torch.cuda.get_device_properties(device_id).multi_processor_count


def get_num_sms() -> int:
    # Returns the Compute Unit count of the device.
    #
    # Prefer chip_info.get_cu_num(): it honors the CU_NUM env override and is the
    # same value the tuning dispatch keys (gfx, cu_num, M, N, K) are built from,
    # so grid/segment sizing stays consistent with the selected tuned configs.
    # AOT clears chip_info's cache after CU_NUM overrides, so a count it returns
    # is not cached here; only a failure is.
    # Fall back to torch's multi_processor_count when get_cu_num() is unavailable
    # (e.g. rocminfo missing/unparseable).
    if _cu_num_failed and os.getenv("CU_NUM") in _cu_num_failed:
        import torch

        return _num_sms_from_torch(torch.cuda.current_device())
    try:
        from aiter.jit.utils.chip_info import get_cu_num

        return get_cu_num()
    except Exception:  # noqa: BLE001
        _cu_num_failed.add(os.getenv("CU_NUM"))
        import torch

        return _num_sms_from_torch(torch.cuda.current_device())

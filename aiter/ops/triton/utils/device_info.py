import functools


@functools.lru_cache(maxsize=1)
def get_num_sms():
    # Returns the Compute Unit count of the device.
    # Prefer chip_info.get_cu_num() (honors CU_NUM env override, matches the tuning
    # dispatch keys). Fall back to torch's multi_processor_count when unavailable
    # (e.g. rocminfo missing/unparseable).
    try:
        from aiter.jit.utils.chip_info import get_cu_num

        return get_cu_num()
    except Exception:
        import torch

        current_device_index = torch.cuda.current_device()
        current_device = torch.cuda.get_device_properties(current_device_index)
        return current_device.multi_processor_count


def get_num_xcds():
    # Currently, you can't query this programmatically. For gfx942/gfx950 it's 8, so we hardcode that here.
    return 8

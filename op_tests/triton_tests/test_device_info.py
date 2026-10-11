import pytest

from aiter.jit.utils import chip_info
from aiter.ops.triton.utils.device_info import get_num_xcds


@pytest.mark.parametrize(
    "gfx, cu_num, xcds",
    [
        ("gfx942", 304, 8),  # MI300X
        ("gfx942", 228, 6),  # MI300A
        ("gfx942", 80, 4),  # MI308X
        ("gfx950", 256, 8),  # MI350X / MI355X
        ("gfx1250", 256, 8),  # unknown part keeps the historical default
    ],
)
def test_get_num_xcds_follows_the_part(monkeypatch, gfx, cu_num, xcds):
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: gfx)
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: cu_num)
    assert get_num_xcds() == xcds


def test_get_num_xcds_defaults_to_8_when_the_gpu_cannot_be_queried(monkeypatch):
    def unavailable():
        raise RuntimeError("rocminfo not found")

    monkeypatch.setattr(chip_info, "get_cu_num", unavailable)
    assert get_num_xcds() == 8

import pytest
import torch

from aiter.ops.triton.utils.shuffle import shuffle_scale_moe

# rows must be a multiple of scale_kwidth (8), cols a multiple of
# preshuffle_factor (32) after shuffle_scale_moe's internal transpose --
# see _shuffle_scale_tile_gfx950 / _shuffle_scale_tile_gfx1250.
_ROWS, _COLS = 8, 32

_NON_CAPABLE_ARCHS = ["gfx942", "gfx90a", "gfx1201"]
_CAPABLE_ARCHS = [("gfx950", "CDNA4_SCALE"), ("gfx1250", "GFX1250_SCALE")]


def _make_data():
    return torch.arange(_ROWS * _COLS, dtype=torch.uint8).reshape(_ROWS, _COLS)


@pytest.mark.parametrize("arch", _NON_CAPABLE_ARCHS)
def test_shuffle_scale_moe_is_noop_on_non_capable_arch(arch):
    """Archs with no native preshuffled MX scale layout (i.e. not
    gfx950/gfx1250) must pass ``data`` through unchanged instead of crashing
    with UnboundLocalError (the bug this test guards against)."""
    data = _make_data()

    scale, layout = shuffle_scale_moe(data, arch=arch, return_layout=True)
    assert layout is None
    assert scale is data

    scale_only = shuffle_scale_moe(data, arch=arch, return_layout=False)
    assert scale_only is data


@pytest.mark.parametrize("arch,expected_layout", _CAPABLE_ARCHS)
def test_shuffle_scale_moe_still_shuffles_on_capable_arch(arch, expected_layout):
    """Regression guard: the non-capable-arch no-op branch must not swallow
    the real gfx950/gfx1250 preshuffle path."""
    data = _make_data()

    scale, layout = shuffle_scale_moe(data, arch=arch, return_layout=True)
    assert layout == expected_layout
    assert scale.numel() == data.numel()
    assert not torch.equal(scale.reshape(-1), data.reshape(-1))

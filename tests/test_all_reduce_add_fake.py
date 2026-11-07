"""The fake `all_reduce_add_` gives torch.compile describes the output the
eager path returns. Where the Gluon fused add does not apply, that path
all-reduces `input_ + other`, so shape and dtype follow the sum's broadcasting
and type promotion. CPU only: the device communicator is a stand-in."""

from types import SimpleNamespace

import pytest
import torch

from aiter.dist import parallel_state


def _group():
    group = object.__new__(parallel_state.GroupCoordinator)
    group.world_size = 2
    group.unique_name = "group"
    # No `all_reduce_add`: the group takes the add-then-reduce fallback.
    group.device_communicator = SimpleNamespace(all_reduce=lambda x, *args: x)
    return group


@pytest.mark.parametrize(
    "a, b",
    [
        ((4, 8, torch.bfloat16), (4, 8, torch.bfloat16)),
        ((4, 8, torch.bfloat16), (4, 8, torch.float32)),
        ((4, 8, torch.bfloat16), (1, 8, torch.bfloat16)),
        ((1, 8, torch.bfloat16), (4, 8, torch.float32)),
    ],
)
def test_fake_matches_the_fallback_output(a, b):
    tensor = torch.ones(a[:2], dtype=a[2])
    other = torch.ones(b[:2], dtype=b[2])
    eager = _group()._all_reduce_add_out_place(tensor, other, True, False)
    fake = parallel_state.all_reduce_add_fake(tensor, other, "group", True, False)
    assert fake.shape == eager.shape
    assert fake.dtype == eager.dtype
    assert fake.device == eager.device

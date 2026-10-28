import torch

from aiter.dist.device_communicators.custom_all_reduce import is_weak_contiguous


def test_custom_all_reduce_accepts_contiguous_storage_offset_view():
    base = torch.empty(17)
    view = base[1:]

    assert view.storage_offset() == 1
    assert view.is_contiguous()
    assert is_weak_contiguous(view)


def test_custom_all_reduce_rejects_full_storage_transpose():
    view = torch.empty((8, 16)).transpose(0, 1)

    assert view.untyped_storage().nbytes() == view.numel() * view.element_size()
    assert not view.is_contiguous()
    assert not is_weak_contiguous(view)

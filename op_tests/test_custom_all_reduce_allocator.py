import pytest

from aiter.dist.device_communicators import custom_all_reduce


@pytest.mark.parametrize("value", [True, False])
def test_expandable_segments_uses_allocator_snapshot(monkeypatch, value):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    monkeypatch.setattr(
        custom_all_reduce.torch.cuda.memory,
        "_snapshot",
        lambda: {"allocator_settings": {"expandable_segments": value}},
    )

    assert custom_all_reduce._expandable_segments_enabled() is value


@pytest.mark.parametrize(
    ("name", "config", "expected"),
    [
        ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True", True),
        (
            "PYTORCH_HIP_ALLOC_CONF",
            "max_split_size_mb:64, expandable_segments:True",
            True,
        ),
        ("PYTORCH_ALLOC_CONF", "expandable_segments:False", False),
        (
            "PYTORCH_ALLOC_CONF",
            "expandable_segments:False,expandable_segments:True",
            True,
        ),
    ],
)
def test_expandable_segments_falls_back_to_environment(
    monkeypatch, name, config, expected
):
    for variable in (
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_HIP_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(name, config)
    monkeypatch.setattr(
        custom_all_reduce.torch.cuda.memory,
        "_snapshot",
        lambda: (_ for _ in ()).throw(RuntimeError),
    )

    assert custom_all_reduce._expandable_segments_enabled() is expected


def test_cuda_allocator_config_has_pytorch_precedence(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    monkeypatch.setenv("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setattr(
        custom_all_reduce.torch.cuda.memory,
        "_snapshot",
        lambda: (_ for _ in ()).throw(RuntimeError),
    )

    assert custom_all_reduce._expandable_segments_enabled() is False


def test_custom_allreduce_stays_disabled_for_expandable_segments(monkeypatch, caplog):
    monkeypatch.setattr(custom_all_reduce, "custom_ar", True)
    monkeypatch.setattr(custom_all_reduce, "_is_gfx1250", False)
    monkeypatch.setattr(custom_all_reduce, "_use_vmm", False)
    monkeypatch.setattr(custom_all_reduce, "_expandable_segments_enabled", lambda: True)
    monkeypatch.setattr(
        custom_all_reduce.CustomAllreduce, "_select_ops", lambda self: None
    )

    communicator = custom_all_reduce.CustomAllreduce(group=None, device=0)

    assert communicator.disabled is True
    assert "falling back to RCCL" in caplog.text

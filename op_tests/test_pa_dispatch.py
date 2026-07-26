import torch

from aiter.ops import attention


def test_paged_attention_asm_dispatch_matches_packaged_architectures(monkeypatch):
    args = (256, 4, 128, torch.bfloat16, 1)

    for arch in ("gfx942", "gfx950"):
        monkeypatch.setattr(attention, "get_gfx", lambda arch=arch: arch)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda _: type("Properties", (), {"multi_processor_count": 104})(),
        )
        assert attention._should_use_asm_kernel(*args)

    for arch in ("gfx90a", "gfx1100", "gfx1201", "gfx1250"):
        monkeypatch.setattr(attention, "get_gfx", lambda arch=arch: arch)
        assert not attention._should_use_asm_kernel(*args)

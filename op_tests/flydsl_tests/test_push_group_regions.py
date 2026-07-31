import torch

from aiter.ops.flydsl.dispatch_combine_v2.dispatch_combine_op import (
    EpDispatchCombineConfig,
    _push_group_regions,
)


def _cfg(**ov):
    base = dict(
        rank=0,
        world_size=2,
        hidden_dim=512,
        max_num_inp_token_per_rank=128,
        num_experts_per_rank=4,
        num_experts_per_token=2,
        data_type=torch.float8_e4m3fn,
    )
    base.update(ov)
    return EpDispatchCombineConfig(**base)


def test_push_group_defaults_off(monkeypatch):
    monkeypatch.delenv("AITER_EP_PUSH_GROUP", raising=False)
    assert _cfg().push_group is False
    assert _push_group_regions(_cfg(), tile_m=64) == []


def test_push_group_cap_default_and_align(monkeypatch):
    monkeypatch.setenv("AITER_EP_PUSH_GROUP", "1")
    monkeypatch.delenv("AITER_EP_PUSH_GROUP_CAP", raising=False)
    cfg = _cfg()  # world_size*mtpr = 2*128 = 256, align_up(tile_m=64) -> 256
    assert cfg.push_group is True
    assert cfg.push_group_cap % 64 == 0
    assert cfg.push_group_cap >= 2 * 128


def test_push_group_cap_env_override_aligns(monkeypatch):
    monkeypatch.setenv("AITER_EP_PUSH_GROUP", "1")
    monkeypatch.setenv("AITER_EP_PUSH_GROUP_CAP", "100")
    cfg = _cfg()
    assert cfg.push_group_cap == 128  # align_up(100, 64)


def test_push_group_running_region(monkeypatch):
    monkeypatch.setenv("AITER_EP_PUSH_GROUP", "1")
    regs = dict(_push_group_regions(_cfg(), tile_m=64))
    assert regs["pg_running"] == 4 * 4  # num_experts_per_rank * 4 bytes

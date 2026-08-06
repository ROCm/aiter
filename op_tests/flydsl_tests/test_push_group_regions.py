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


def test_push_group_defaults_off():
    # explicit switch (was AITER_EP_PUSH_GROUP env); default off.
    assert _cfg().push_group is False
    assert _push_group_regions(_cfg(), tile_m=64) == []


def test_cap_default_is_worst_case_aligned():
    # cap_per_expert=0 (default) => worst-case ws*mtp = 2*128 = 256, align_up(64) -> 256.
    cfg = _cfg(push_group=True)
    assert cfg.push_group is True
    assert cfg.effective_cap_per_expert % 64 == 0
    assert cfg.effective_cap_per_expert == 2 * 128


def test_cap_pin_aligns():
    # caller-pinned capacity, aligned up to tile_m.
    cfg = _cfg(push_group=True, cap_per_expert=100)
    assert cfg.effective_cap_per_expert == 128  # align_up(100, 64)


def test_push_group_running_region():
    regs = dict(_push_group_regions(_cfg(push_group=True), tile_m=64))
    assert regs["pg_running"] == 4 * 4  # num_experts_per_rank * 4 bytes

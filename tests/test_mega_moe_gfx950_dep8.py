# SPDX-License-Identifier: MIT
"""mega-moe-provider on DEP8: the default-off DSpark draft-layout switch, the [1, tokens, hidden]
FFN input ATOM's V4.1 backbone hands the seam, the once-only decline log, and the small-input
tuning switches. CPU only."""

import importlib
from types import SimpleNamespace


def _ffn(n, experts=384, local=48, topk=6):
    qm = SimpleNamespace(
        fused_experts="native",
        is_guinterleave=True,
        use_triton=False,
        is_gfx1250=False,
        hidden_size=5120,
        intermediate_size=2304,
    )
    layer = SimpleNamespace(
        quant_method=qm,
        w13_weight=SimpleNamespace(is_shuffled=True),
        global_num_experts=experts,
        local_num_experts=local,
        top_k=topk,
        swiglu_limit=10.0,
    )
    return SimpleNamespace(experts=layer, n=n)


# A measured EP8 small-input override.
_DEP8_OVERRIDE = (
    "32+64+128:s1.use_tile_resource=1,fs1.epi_evec=8,fs1.epi_lean=3,fs1.head_uniform=1,"
    "fs1.g2_epi_lean=2,fs1.expert_cached_w=2"
)
_EP8 = {"world_size": 8, "experts_per_rank": 48, "model_dim": 5120, "inter_dim": 2304}


def test_small_m_override_replaces_only_the_listed_buckets(monkeypatch):
    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_config as config

    stock = {b: config.select_fused_config(b, 128, **_EP8) for b in (16, 32, 64, 128)}
    monkeypatch.setenv("AITER_MEGA_SMALL_M_OVERRIDE", _DEP8_OVERRIDE)
    switched = importlib.reload(config)
    try:
        for bucket in (32, 64, 128):
            knobs = switched.select_fused_config(bucket, 128, **_EP8).fused_stage1
            got = (
                knobs.epi_evec,
                knobs.epi_lean,
                knobs.g2_epi_lean,
                knobs.expert_cached_w,
            )
            assert got == (8, 3, 2, 2)
            assert knobs.head_uniform is True
        assert repr(switched.select_fused_config(16, 128, **_EP8)) == repr(stock[16])
    finally:
        monkeypatch.delenv("AITER_MEGA_SMALL_M_OVERRIDE")
        restored = importlib.reload(config)
    for bucket, before in stock.items():
        assert repr(restored.select_fused_config(bucket, 128, **_EP8)) == repr(before)


def test_fused_draft_layout_needs_its_switch(monkeypatch):
    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_config as config

    draft = (8, 5120, 2304, 128, 3)  # (EP world, hidden, inter, experts, top-k)
    shape = {"world_size": 8, "model_dim": 5120, "inter_dim": 2304, "experts": 128}
    assert draft not in config.FUSED_LAYOUTS
    monkeypatch.setenv("AITER_MEGA_FUSED_DRAFT_LAYOUT", "1")
    switched = importlib.reload(config)
    try:
        assert draft in switched.FUSED_LAYOUTS
        assert switched.fused_layout_small_only(**shape)
    finally:
        monkeypatch.delenv("AITER_MEGA_FUSED_DRAFT_LAYOUT")
        importlib.reload(config)

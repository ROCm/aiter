# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEV2 config selection for EPLB redundant-expert profiles -- no GPU needed.

mega_moe_config imports only the standard library and its sibling envs module,
so it is loaded here on its own, without aiter/__init__.py (torch, JIT build).
"""

import importlib.util
import pathlib
import sys
import types

import pytest

_MEGA_MOE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "aiter"
    / "ops"
    / "flydsl"
    / "kernels"
    / "mega_moe"
)
_PACKAGE = "_test_mega_moe_v2_config_pkg"


def _load_mega_moe_config():
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_MEGA_MOE)]
    sys.modules[_PACKAGE] = package
    name = f"{_PACKAGE}.mega_moe_config"
    spec = importlib.util.spec_from_file_location(
        name, _MEGA_MOE / "mega_moe_config.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mc = _load_mega_moe_config()

# DeepSeek-V4.1-Flash: 384 routed experts, EP8, d5120 / i2304.
FLASH = {"model_dim": 5120, "inter_dim": 2304, "world_size": 8}
V4_PRO = {"model_dim": mc.V4_MODEL_DIM, "inter_dim": mc.V4_INTER_DIM, "world_size": 8}
MTPRS = (256, 512, 1024, 4096, 8192, 16384, 32768)


def _select(tokens, mtpr, epr, shape, a_dtype=mc.ACTIVATION_FP8):
    return mc.select_mega_moe_config(
        tokens, mtpr, a_dtype=a_dtype, experts_per_rank=epr, **shape
    )


def _cases():
    # Compact dispatch only: fixed-slot (mtpr <= FIXED_SLOT_MAX_MTPR) is
    # reserved for epr48, so an epr48 config there is not comparable.
    return [
        (bucket, mtpr)
        for mtpr in MTPRS
        if mtpr > mc.FIXED_SLOT_MAX_MTPR
        for bucket in mc.TOKEN_BUCKETS
        if bucket <= mtpr
    ]


def test_flash_ep8_eplb_epr49_matches_flash_ep8_profile():
    for epr in (48, *mc.V41_FLASH_EP8_EPLB_EXPERTS_PER_RANKS):
        profile = mc._match_network(epr, FLASH["model_dim"], FLASH["inter_dim"])
        assert profile is not None and profile.name == "v41_flash_ep8"
    # Other expert counts and other shapes stay unmatched.
    assert mc._match_network(50, FLASH["model_dim"], FLASH["inter_dim"]) is None
    assert mc._match_network(49, mc.V4_MODEL_DIM, mc.V4_INTER_DIM) is None


@pytest.mark.parametrize("bucket,mtpr", _cases())
def test_flash_ep8_eplb_epr49_gets_epr48_tuning(bucket, mtpr):
    # The V4.1-Flash EP8 tuner depends only on dtype, world size, mtpr, bucket
    # and fixed-slot, so with compact dispatch epr49 selects what epr48 does.
    assert _select(bucket, mtpr, 49, FLASH) == _select(bucket, mtpr, 48, FLASH)


def test_flash_ep8_eplb_epr49_is_tuned():
    # Points where the V4.1-Flash EP8 tuner moves off the base config; an
    # untuned epr (50, no profile) shows the base config.
    tuned = _select(16384, 32768, 49, FLASH)
    base = _select(16384, 32768, 50, FLASH)
    assert tuned.stage2.deep_a_pipeline and tuned.stage2.b2stage
    assert not tuned.stage2.aligned_pair
    assert tuned != base
    assert _select(256, 256, 49, FLASH).stage1.num_dispatch_cu == 224
    assert _select(128, 8192, 49, FLASH).stage1.num_dispatch_cu == 160


def test_flash_ep8_eplb_epr49_fp4_untouched():
    # The tuner is A8-only and the Flash EP8 row has no FP4 prelude.
    for bucket, mtpr in ((128, 8192), (16384, 32768)):
        assert _select(bucket, mtpr, 49, FLASH, mc.ACTIVATION_FP4) == _select(
            bucket, mtpr, 50, FLASH, mc.ACTIVATION_FP4
        )


def test_flash_ep8_eplb_epr49_bundle_plan_matches_epr48():
    for mtpr in (8192, 32768):
        plan49 = mc.build_mega_moe_bundle_plan(mtpr, experts_per_rank=49, **FLASH)
        plan48 = mc.build_mega_moe_bundle_plan(mtpr, experts_per_rank=48, **FLASH)
        assert not plan49.fixed_slot_dispatch
        assert plan49.entries == plan48.entries
        assert plan49.stage1_variants == plan48.stage1_variants
        assert plan49.stage2_variants == plan48.stage2_variants


@pytest.mark.parametrize("epr", mc.V4_EPLB_EXPERTS_PER_RANKS)
def test_v4_pro_eplb_family_keeps_base_tuning(epr):
    # V4-Pro's EPLB family still gets only the FP4 prelude, not the epr48 tuner.
    profile = mc._match_network(epr, mc.V4_MODEL_DIM, mc.V4_INTER_DIM)
    assert profile is not None and profile.name == "v4_pro"
    assert not profile.tune_family

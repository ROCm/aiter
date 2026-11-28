# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""With AITER_FLYDSL_STAGE2_FP8=1 or AITER_MXFP4_INTERMEDIATE=1, a row of the
stage2 FP8 table (tuned_stage2_fp8_fmoe) replaces the main row of its shape.

Qwen3.8 MXFP4 TP8, E 513: the main rows at 8192 to 32768 tokens are one-stage
post-reduce kernels; the stage2 FP8 rows are FlyDSL `reduce` rows.
"""

import os
import subprocess
import sys

import pytest

import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import AsmMoeMode, get_2stage_cfgs
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

pytestmark = pytest.mark.skipif(
    get_gfx_runtime() != "gfx950" or get_cu_num() != 256,
    reason="the Qwen3.8 MXFP4 rows are tuned for gfx950 with 256 CUs",
)


def _key(token):
    return (
        fused_moe_module.get_gfx_runtime(),
        fused_moe_module.get_cu_num(),
        token,
        8192,
        256,
        513,
        11,
        str(ActivationType.Silu),
        str(dtypes.bf16),
        str(dtypes.fp4x2),
        str(dtypes.fp4x2),
        str(QuantType.per_1x32),
        True,
        False,
    )


SWITCHES = ("AITER_FLYDSL_STAGE2_FP8", "AITER_MXFP4_INTERMEDIATE")


def _dispatch(
    monkeypatch, token, fp8, activation=ActivationType.Silu, switch=SWITCHES[0]
):
    for name in SWITCHES:
        monkeypatch.setenv(name, "1" if fp8 and name == switch else "0")
    get_2stage_cfgs.cache_clear()
    try:
        return get_2stage_cfgs(
            fused_moe_module.get_padded_M(token),
            8192,
            256,
            513,
            11,
            dtypes.bf16,
            dtypes.fp4x2,
            dtypes.fp4x2,
            QuantType.per_1x32,
            True,
            activation,
            False,
            0,
            0,
            is_shuffled=True,
            opus_weights_shuffled=True,
        )
    finally:
        get_2stage_cfgs.cache_clear()


@pytest.mark.parametrize("switch", SWITCHES)
@pytest.mark.parametrize(("token", "block_m"), [(8192, 64), (16384, 128), (32768, 64)])
def test_stage2_fp8_row_replaces_post_reduce_row(monkeypatch, token, block_m, switch):
    off = _dispatch(monkeypatch, token, fp8=False)
    assert off.run_1stage and off.flat == AsmMoeMode.POST_REDUCE

    on = _dispatch(monkeypatch, token, fp8=True, switch=switch)
    assert not on.run_1stage and on.flat == AsmMoeMode.SORTED
    assert on.block_m == block_m


def test_shape_without_stage2_fp8_row_keeps_main_row(monkeypatch):
    off = _dispatch(monkeypatch, 4096, fp8=False)
    on = _dispatch(monkeypatch, 4096, fp8=True)
    assert (on.run_1stage, on.block_m) == (off.run_1stage, off.block_m)


def test_tier_fallback_uses_stage2_fp8_row_of_same_tier(monkeypatch):
    # Both tables fall back from 131072 to their 32768 rows.
    assert not _dispatch(monkeypatch, 131072, fp8=True).run_1stage


def test_stage2_fp8_row_of_smaller_tier_keeps_main_row(monkeypatch):
    _dispatch(monkeypatch, 16384, fp8=False)
    primary, fallback = fused_moe_module.cfg_2stages
    row_32768 = primary[_key(32768)]
    monkeypatch.setattr(
        fused_moe_module,
        "cfg_2stages",
        ({**primary, _key(131072): row_32768}, fallback),
    )
    # The main table has a 131072 row; the stage2 FP8 row is for 32768.
    assert _dispatch(monkeypatch, 131072, fp8=True).run_1stage


def test_stage2_fp8_row_keeps_main_row_of_other_activation(monkeypatch):
    _dispatch(monkeypatch, 8192, fp8=False)
    primary, fallback = fused_moe_module.cfg_2stages
    situv2_key = _key(8192)[:7] + (str(ActivationType.Situv2),) + _key(8192)[8:]
    monkeypatch.setattr(
        fused_moe_module,
        "cfg_2stages",
        ({**primary, situv2_key: primary[_key(8192)]}, fallback),
    )
    # The main table has a SiTUv2 row; the stage2 FP8 row is for Silu.
    on = _dispatch(monkeypatch, 8192, fp8=True, activation=ActivationType.Situv2)
    assert on.run_1stage and on.flat == AsmMoeMode.POST_REDUCE


@pytest.mark.parametrize("switch", SWITCHES)
def test_aot_replays_stage2_fp8_rows_only_with_switch(switch):
    # A fresh process: aiter.aot.flydsl.moe reads the switch at import.
    check = f"""
import importlib, os
import aiter.aot.flydsl.moe as aot_moe
from aiter.jit.core import AITER_CONFIGS
fp8_csv = AITER_CONFIGS.AITER_CONFIG_FMOE_STAGE2_FP8_FILE
assert fp8_csv not in aot_moe.DEFAULT_CSVS
os.environ["{switch}"] = "1"
aot_moe = importlib.reload(aot_moe)
assert fp8_csv in aot_moe.DEFAULT_CSVS
jobs = {{(int(j["expert"]), int(j["token"])) for j in aot_moe.parse_csv(fp8_csv)}}
assert {{(513, 8192), (513, 16384), (513, 32768)}} <= jobs, jobs
"""
    env = {**os.environ, **dict.fromkeys(SWITCHES, "0")}
    subprocess.run([sys.executable, "-c", check], env=env, check=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

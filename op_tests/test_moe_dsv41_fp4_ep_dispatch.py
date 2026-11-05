# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DSV4.1 Flash EP16 config selection and local/shared expert numerics.

The numerical test runs one EP rank's compute on a single GPU. It exercises
global expert IDs and masking; it does not run inter-node communication.
"""

import csv
import importlib
from pathlib import Path

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes, get_gfx
from aiter.fused_moe import (
    fused_moe,
    get_2stage_cfgs,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle

CONFIG = (
    Path(__file__).resolve().parents[1]
    / "aiter/configs/model_configs/dsv41_fp4_tuned_fmoe.csv"
)
TOKENS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 131072)


@pytest.fixture
def dsv41_config(monkeypatch):
    module = importlib.import_module("aiter.fused_moe")
    monkeypatch.setenv("AITER_CONFIG_FMOE", str(CONFIG))
    monkeypatch.setattr(module, "get_cu_num", lambda: 256)
    monkeypatch.setattr(module, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(module, "cfg_2stages", None)
    module.AITER_CONFIGS.get_config_file.cache_clear()
    get_2stage_cfgs.cache_clear()
    yield module
    get_2stage_cfgs.cache_clear()
    module.AITER_CONFIGS.get_config_file.cache_clear()


def _expected_row(token, experts, topk):
    with CONFIG.open() as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if (
                row["gfx"],
                row["token"],
                row["model_dim"],
                row["inter_dim"],
                row["expert"],
                row["topk"],
                row["q_dtype_a"],
            )
            == (
                "gfx950",
                str(token),
                "5120",
                "2304",
                str(experts),
                str(topk),
                "torch.float4_e2m1fn_x2",
            )
        ]
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize("token", TOKENS)
@pytest.mark.parametrize(("experts", "topk"), ((24, 6), (25, 7)))
def test_ep16_selects_tuned_kernels(dsv41_config, token, experts, topk):
    row = _expected_row(token, experts, topk)
    meta = get_2stage_cfgs(
        token,
        5120,
        2304,
        experts,
        topk,
        dtypes.bf16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        is_shuffled=True,
        opus_weights_shuffled=True,
        is_ep=True,
    )
    assert meta.stage1.keywords["kernelName"] == row["kernelName1"]
    assert meta.stage2.keywords["kernelName"] == row["kernelName2"]
    assert meta.fuse_quant == "fp4"
    assert meta.block_m == int(row["block_m"])


@pytest.mark.parametrize("token", TOKENS)
@pytest.mark.parametrize("shared", (False, True))
def test_ep16_mask_and_shared_numerics(dsv41_config, shared, token):
    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        pytest.skip("requires a gfx950 GPU")
    torch.manual_seed(7389)
    hidden, inter = 5120, 2304
    local_experts, topk = (25, 7) if shared else (24, 6)
    # Last rank. Without fusion routed IDs 360..383 map to weights 0..23.
    # With local replication, ATOM remaps the 384 logical IDs into 16 blocks
    # of 25 physical IDs, including one shared slot in each block (400 total).
    start = 375 if shared else 360
    mask = torch.zeros(400 if shared else 385, dtype=torch.int32, device="cuda")
    mask[start : start + local_experts] = 1
    ids = (
        (
            torch.arange(token, device="cuda")[:, None] * 6
            + torch.arange(6, device="cuda")[None, :]
        )
        % 336
    ).to(torch.int32)
    # Every row has a local route, plus non-local routes that must be discarded.
    ids[:, 0] = 360 + torch.arange(token, device="cuda") % 24
    if shared:
        ids = (ids // 24) * 25 + ids % 24
    else:
        ids[0, 1] = 384  # masked sentinel, outside the routed ID range
    if shared:
        ids = torch.cat(
            (ids, torch.full((token, 1), 399, device="cuda", dtype=torch.int32)), dim=1
        )
    weights = torch.full((token, topk), 1 / 6, dtype=torch.float32, device="cuda")
    if shared:
        weights[:, -1] = 1.0  # shared is always active, not router-selected
    x = torch.randn((token, hidden), dtype=dtypes.bf16, device="cuda") * 0.1
    w1 = (
        torch.randn(
            (local_experts, 2 * inter, hidden), dtype=dtypes.bf16, device="cuda"
        )
        * 0.1
    )
    w2 = (
        torch.randn((local_experts, hidden, inter), dtype=dtypes.bf16, device="cuda")
        * 0.1
    )
    w1q, s1 = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2, shuffle=False)
    w2q, s2 = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2, shuffle=False)
    del w1, w2
    w1s = shuffle_weight(w1q, layout=(16, 16))
    w2s = shuffle_weight(w2q, layout=(16, 16))
    w1s.is_shuffled = w2s.is_shuffled = True
    out = fused_moe(
        x,
        w1s,
        w2s,
        weights,
        ids,
        expert_mask=mask,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        quant_dtype_a=dtypes.fp4x2,
        w1_scale=e8m0_shuffle(s1),
        w2_scale=e8m0_shuffle(s2),
        swiglu_limit=10.0,
    )
    torch.cuda.synchronize()
    # Independent Torch reference on sampled rows, including both quantization
    # boundaries and the same global-to-local mapping. Sample across all 24
    # local experts; raw validation tensors are not written to disk.
    sample = torch.arange(32, device="cuda")
    local_ids = torch.where(mask[ids[sample].long()].bool(), ids[sample] - start, -1)
    xq, xs = per_1x32_f4_quant(x[sample], quant_dtype=dtypes.fp4x2, shuffle=False)
    a2 = torch_moe_stage1(
        xq,
        w1q,
        w2q,
        weights[sample],
        local_ids,
        dtype=dtypes.bf16,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        a1_scale=xs,
        w1_scale=s1,
        swiglu_limit=10.0,
    )
    a2q, a2s = per_1x32_f4_quant(
        a2.reshape(-1, inter), quant_dtype=dtypes.fp4x2, shuffle=False
    )
    ref = torch_moe_stage2(
        a2q.view(32, topk, inter // 2),
        w1q,
        w2q,
        weights[sample],
        local_ids,
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=s2,
        a2_scale=a2s,
        doweight=True,
    )
    assert torch.isfinite(out).all()
    cosine_error = (
        1
        - torch.nn.functional.cosine_similarity(
            ref.float(), out[sample].float(), dim=-1
        )
        .mean()
        .item()
    )
    print(
        f"EP16 token={token} local_experts={local_experts} topk={topk} "
        f"shared={shared} cosine_error={cosine_error:.8f}"
    )
    assert cosine_error < 0.01

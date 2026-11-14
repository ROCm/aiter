# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
import os
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn.functional as F

from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.moe.fused_moe_a16w16 import (
    fused_moe_a16w16_stage1,
    fused_moe_a16w16_stage2,
    parse_triton_a16w16_kernel,
    triton_a16w16_kernel_name,
)

DEV = "cuda"


def make_inputs(M, model_dim, inter_dim, E, topk, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(M, model_dim, device=DEV, generator=g).to(torch.bfloat16)
    w1 = (
        torch.randn(E, 2 * inter_dim, model_dim, device=DEV, generator=g)
        / model_dim**0.5
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(E, model_dim, inter_dim, device=DEV, generator=g) / inter_dim**0.5
    ).to(torch.bfloat16)
    logits = torch.randn(M, E, device=DEV, generator=g)
    topk_weights, topk_ids = torch.topk(logits.softmax(-1), topk, dim=-1)
    return x, w1, w2, topk_weights, topk_ids.to(torch.int32)


def torch_moe(x, w1, w2, topk_weights, topk_ids):
    inter_dim = w2.shape[2]
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float32, device=DEV)
    for slot in range(topk_ids.shape[1]):
        for e in topk_ids[:, slot].unique().tolist():
            rows = (topk_ids[:, slot] == e).nonzero().flatten()
            h = x[rows].float() @ w1[e].float().t()
            a = F.silu(h[:, :inter_dim]) * h[:, inter_dim:]
            a = a.to(torch.bfloat16).float()
            out[rows] += topk_weights[rows, slot, None] * (a @ w2[e].float().t())
    return out


def cos_diff(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return 1.0 - 2.0 * float((a * b).sum()) / float((a * a + b * b).sum())


def test_kernel_names_round_trip():
    name = triton_a16w16_kernel_name(1, 32, 128, 64, 8, 4, 2, we=2)
    assert name == "triton_moe1_a16w16_bm32_bn128_bk64_gm8_w4_s2_we2"
    assert parse_triton_a16w16_kernel(name) == {
        "stage": 1,
        "bm": 32,
        "bn": 128,
        "bk": 64,
        "gm": 8,
        "warps": 4,
        "stages": 2,
        "we": 2,
    }
    with pytest.raises(ValueError):
        parse_triton_a16w16_kernel("triton_moe1_a16w16_bm32")


@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize(
    "M, model_dim, inter_dim, E, topk, block_m",
    [
        (1, 512, 256, 8, 2, 16),
        (33, 1024, 512, 16, 4, 32),
        (300, 4096, 512, 32, 2, 32),
        (1024, 1024, 768, 8, 1, 64),
        (2048, 512, 256, 8, 8, 128),
    ],
)
@pytest.mark.parametrize("cfg", [(64, 64, 1, 4, 2, 0), (128, 64, 8, 8, 2, 2)])
def test_fused_moe_a16w16_stages(
    M, model_dim, inter_dim, E, topk, block_m, shuffled, cfg
):
    x, w1, w2, topk_weights, topk_ids = make_inputs(M, model_dim, inter_dim, E, topk)
    expected = torch_moe(x, w1, w2, topk_weights, topk_ids)
    if shuffled:
        w1, w2 = shuffle_weight(w1, (16, 16)), shuffle_weight(w2, (16, 16))
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out = moe_sorting(
        topk_ids, topk_weights, E, model_dim, torch.bfloat16, block_m
    )
    inter = torch.empty(M, topk, inter_dim, dtype=torch.bfloat16, device=DEV)
    fused_moe_a16w16_stage1(
        x,
        w1,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        inter,
        topk,
        block_m,
        triton_a16w16_kernel_name(1, block_m, *cfg),
    )
    fused_moe_a16w16_stage2(
        inter,
        w2,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        out,
        topk,
        block_m,
        triton_a16w16_kernel_name(2, block_m, *cfg),
    )
    assert cos_diff(out.float(), expected) < 1e-4
    torch.testing.assert_close(out.float(), expected, atol=3e-2, rtol=3e-2)


def test_fused_moe_a16w16_block_m_mismatch():
    x, w1, _, topk_weights, topk_ids = make_inputs(16, 512, 256, 8, 2)
    sorted_ids, _, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, 8, 512, torch.bfloat16, 32
    )
    inter = torch.empty(16, 2, 256, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(ValueError, match="block_m"):
        fused_moe_a16w16_stage1(
            x,
            w1,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            inter,
            2,
            32,
            triton_a16w16_kernel_name(1, 16, 64, 64, 1, 4, 2),
        )


_DISPATCH_CHILD = textwrap.dedent(
    """
    import sys
    import torch
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight
    sys.path.insert(0, {test_dir!r})
    from test_fused_moe_a16w16 import cos_diff, make_inputs, torch_moe

    for M in {tokens}:
        x, w1, w2, tw, tid = make_inputs(M, {model_dim}, {inter_dim}, {E}, {topk})
        expected = torch_moe(x, w1, w2, tw, tid)
        w1s, w2s = shuffle_weight(w1, (16, 16)), shuffle_weight(w2, (16, 16))
        out = fused_moe(
            x, w1s, w2s, tw, tid,
            activation=ActivationType.Silu, quant_type=QuantType.No,
        )
        d = cos_diff(out.float(), expected)
        print(f"M={{M}} cos_diff={{d:.3e}}", flush=True)
        assert d < 1e-4, d
    print("DISPATCH_OK")
    """
)


def test_fused_moe_dispatches_triton_a16w16_rows(tmp_path):
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    model_dim, inter_dim, E, topk = 1024, 512, 16, 2
    tokens = [1, 7, 64, 300]
    rows = []
    for t in [1 << i for i in range(17)]:
        bm = 16 if t < 256 else 32
        rows.append(
            {
                "gfx": get_gfx(),
                "cu_num": get_cu_num(),
                "token": t,
                "model_dim": model_dim,
                "inter_dim": inter_dim,
                "expert": E,
                "topk": topk,
                "act_type": "ActivationType.Silu",
                "dtype": "torch.bfloat16",
                "q_dtype_a": "torch.bfloat16",
                "q_dtype_w": "torch.bfloat16",
                "q_type": "QuantType.No",
                "use_g1u1": 1,
                "doweight_stage1": 0,
                "block_m": bm,
                "ksplit": 0,
                "us1": 0,
                "kernelName1": triton_a16w16_kernel_name(1, bm, 128, 64, 1, 4, 2),
                "err1": "0.0%",
                "us2": 0,
                "kernelName2": triton_a16w16_kernel_name(2, bm, 128, 64, 8, 4, 2),
                "err2": "0.0%",
                "us": 0,
                "run_1stage": 0,
            }
        )
    path = tmp_path / "triton_a16w16_fmoe.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    code = _DISPATCH_CHILD.format(
        test_dir=os.path.dirname(os.path.abspath(__file__)),
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
    )
    env = {**os.environ, "AITER_CONFIG_FMOE": str(path), "AITER_LOG_LEVEL": "INFO"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0 and "DISPATCH_OK" in log, log[-4000:]
    assert "kernelName1='triton_moe1_a16w16_" in log, log[-4000:]

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dynamic routing, packed K384 weights and graph replay for MXFP8 prefill."""

import functools
import importlib

import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.mxfp8_moe_8wave import kernel_name, stage1, stage2
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility import fp4_utils

fm = importlib.import_module("aiter.fused_moe")
pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="requires gfx950 scaled MFMA"
)


def quant(x):
    blocks = x.float().reshape(-1, 32)
    exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-30) / 448.0))
    scale = torch.exp2(exponent)
    q = (blocks / scale[:, None]).to(torch.float8_e4m3fn).reshape(x.shape)
    s = (exponent + 127).to(torch.uint8).reshape(-1, x.shape[-1] // 32)
    deq = (q.float().reshape(-1, 32) * scale[:, None]).reshape(x.shape)
    return q, s, deq


def reference(x, w1, w2, ids, weights, limit):
    xf = quant(x)[2]
    act = torch.empty((*ids.shape, w2.shape[-1]), device=x.device, dtype=torch.bfloat16)
    for e in range(w1.shape[0]):
        rows, slots = torch.where(ids == e)
        gate, up = (xf[rows] @ w1[e].T).chunk(2, -1)
        gate = gate.clamp(max=limit)
        act[rows, slots] = (
            gate * torch.sigmoid(1.702 * gate) * (up.clamp(-limit, limit) + 1)
        ).bfloat16()
    af = quant(act)[2]
    partial = torch.empty(
        (*ids.shape, x.shape[-1]), device=x.device, dtype=torch.bfloat16
    )
    for e in range(w2.shape[0]):
        rows, slots = torch.where(ids == e)
        partial[rows, slots] = (af[rows, slots] @ w2[e].T).bfloat16()
    return (partial.float() * weights[..., None]).sum(1).bfloat16()


@pytest.mark.parametrize("tile", [(256, 256), (128, 512)])
@pytest.mark.parametrize("ep", [False, True])
def test_dynamic_routes_graph_and_packed_k384(tile, ep):
    torch.manual_seed(813)
    tokens, hidden, inter, experts, topk = 257, 512, 384, 7, 3
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    w1, s1, r1 = quant(torch.randn(experts, inter * 2, hidden, device="cuda") * 0.1)
    w2, s2, r2 = quant(torch.randn(experts, hidden, inter, device="cuda") * 0.1)
    mask = None
    local_experts = experts
    if ep:
        local = [0, 2, 4, 6]
        local_experts = len(local)
        mask = torch.tensor([1, 0, 1, 0, 1, 0, 1], device="cuda", dtype=torch.int32)
        w1, w2 = w1[local], w2[local]
        s1 = s1.view(experts, 2 * inter, -1)[local].reshape(-1, hidden // 32)
        s2 = s2.view(experts, hidden, -1)[local].reshape(-1, inter // 32)
    w1, w2 = shuffle_weight_a16w4(w1, 16, True), shuffle_weight_a16w4(w2, 16, False)
    s1, s2 = shuffle_scale_a16w4(s1, local_experts, True), fp4_utils.e8m0_shuffle(s2)
    ids = torch.rand(tokens, experts, device="cuda").topk(topk, -1).indices.int()
    weights = torch.rand(tokens, topk, device="cuda").softmax(-1)
    limit = 5.0
    meta = fm.MOEMetadata(
        functools.partial(stage1, kernelName=kernel_name(1, *tile)),
        functools.partial(stage2, kernelName=kernel_name(2, swizzle=3)),
        256,
        0,
        prequant=False,
        fuse_quant="fp8",
        skip_inter_quant=True,
    )

    def forward():
        return fm._fused_moe_impl(
            x,
            w1,
            w2,
            weights,
            ids,
            w1_scale=s1,
            w2_scale=s2,
            activation=aiter.ActivationType.Swiglu.value,
            quant_type=aiter.QuantType.per_1x32.value,
            gate_mode="interleave",
            swiglu_limit=limit,
            expert_mask=mask,
            _metadata_transform=lambda _: meta,
        )

    def check(out):
        ref_weights = weights if mask is None else weights * mask[ids.long()]
        ref = reference(x, r1, r2, ids, ref_weights, limit)
        assert out.isfinite().all()
        a, b = out.float(), ref.float()
        error = (a - b).square().sum() / (a.square() + b.square()).sum().clamp_min(
            1e-30
        )
        assert error < 2e-5, error

    output = forward().clone()
    check(output)
    for _ in range(3):
        assert torch.equal(forward(), output)
    # All tensor shapes stay fixed while sorting's valid padded row count changes.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    graph.replay()
    check(captured)
    concentrated = torch.arange(topk, device="cuda", dtype=torch.int32)
    if ep:
        concentrated = concentrated * 2 + 1  # All routes go to remote experts.
    ids.copy_(concentrated.expand_as(ids))
    graph.replay()
    check(captured)
    assert torch.equal(forward(), captured)


def test_tuned_config_preserves_other_moe_paths(monkeypatch, tmp_path):
    import csv
    from types import SimpleNamespace

    kwargs = {
        "token": 32768,
        "model_dim": 6144,
        "inter_dim": 384,
        "expert": 129,
        "topk": 5,
        "dtype": aiter.dtypes.bf16,
        "q_dtype_a": aiter.dtypes.fp8,
        "q_dtype_w": aiter.dtypes.fp8,
        "q_type": aiter.QuantType.per_1x32,
        "activation": aiter.ActivationType.Swiglu,
        "use_g1u1": True,
        "doweight_stage1": False,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "gate_mode": "interleave",
    }
    row = {k: str(int(v) if isinstance(v, bool) else v) for k, v in kwargs.items()}
    row["act_type"] = row.pop("activation")
    row.update(
        gfx="gfx950",
        cu_num=256,
        block_m=256,
        ksplit=0,
        kernelName1=kernel_name(1),
        kernelName2=kernel_name(2),
    )
    csv_path = tmp_path / "tuned.csv"
    with csv_path.open("w") as f:
        writer = csv.DictWriter(f, row.keys())
        writer.writeheader()
        writer.writerow(row)
    monkeypatch.setattr(
        fm, "AITER_CONFIGS", SimpleNamespace(AITER_CONFIG_FMOE_FILE=str(csv_path))
    )
    monkeypatch.setattr(fm, "cfg_2stages", None)
    fm.get_2stage_cfgs.cache_clear()
    try:
        selected = fm.get_2stage_cfgs(**kwargs)
        assert selected.stage1.func is stage1 and selected.stage2.func is stage2
        for override in (
            {"activation": aiter.ActivationType.Silu},
            {"gate_mode": "separated"},
            {"input_dtype": aiter.dtypes.fp8},
            {"has_stage1_bias": True},
            {"has_stage2_bias": True},
            {"hidden_pad": 256},
        ):
            call_kwargs = {**kwargs, **override}
            actual = fm.get_2stage_cfgs(**call_kwargs)
            assert getattr(actual.stage1, "func", actual.stage1) is not stage1
            with monkeypatch.context() as bypass:
                bypass.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
                fm.get_2stage_cfgs.cache_clear()
                expected = fm.get_2stage_cfgs(**call_kwargs)
            fm.get_2stage_cfgs.cache_clear()
            assert actual.block_m == expected.block_m
            assert actual.ksplit == expected.ksplit
            for stage_name in ("stage1", "stage2"):
                a, b = getattr(actual, stage_name), getattr(expected, stage_name)
                assert a.func is b.func and a.keywords == b.keywords
    finally:
        fm.get_2stage_cfgs.cache_clear()

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dynamic routing, packed K384 weights and graph replay for MXFP8/A8W4 prefill."""

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


def quant_weight(x, dtype):
    if dtype == "fp8":
        return quant(x)
    q, s = aiter.get_torch_quant(aiter.QuantType.per_1x32)(
        x, quant_dtype=aiter.dtypes.fp4x2
    )
    q = q.view(*x.shape[:-1], x.shape[-1] // 2)
    deq = fp4_utils.mxfp4_to_f32(q).reshape(-1, 32)
    deq *= fp4_utils.e8m0_to_f32(s).reshape(-1, 1)
    return q, s, deq.reshape(x.shape)


def reference(x, w1, w2, ids, weights, limit, b_dtype="fp8"):
    xf = quant(x)[2]
    act = torch.empty((*ids.shape, w2.shape[-1]), device=x.device, dtype=torch.bfloat16)
    for e in range(w1.shape[0]):
        rows, slots = torch.where(ids == e)
        gate, up = (xf[rows] @ w1[e].T).chunk(2, -1)
        if b_dtype == "fp4":
            if limit:
                gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
            act[rows, slots] = (torch.nn.functional.silu(gate) * up).bfloat16()
        else:
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


@pytest.mark.parametrize(
    "tile,waves,b_dtype,limit,persistent",
    [
        (tile, 8, b_dtype, limit, False)
        for tile in [(256, 256), (128, 512)]
        for b_dtype, limit in [("fp8", 5.0), ("fp8", 0.0), ("fp4", None)]
    ]
    + [((128, 256), 4, "fp4", None, False)]
    + [
        ((256, 256), 8, "fp8", limit, repeats)
        for repeats in (2, 4)
        for limit in (0.0, 5.0)
    ],
)
@pytest.mark.parametrize("ep", [False, True])
def test_dynamic_routes_graph_and_packed_k384(
    tile, waves, ep, b_dtype, limit, persistent
):
    torch.manual_seed(813)
    tokens, hidden, inter, experts, topk = 257, 512, 384, 7, 3
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    w1, s1, r1 = quant_weight(
        torch.randn(experts, inter * 2, hidden, device="cuda") * 0.1, b_dtype
    )
    w2, s2, r2 = quant_weight(
        torch.randn(experts, hidden, inter, device="cuda") * 0.1, b_dtype
    )
    mask = None
    local_experts = experts
    if ep:
        local = [0, 2, 4, 6]
        local_experts = len(local)
        mask = torch.tensor([1, 0, 1, 0, 1, 0, 1], device="cuda", dtype=torch.int32)
        w1 = w1.view(torch.uint8)[local].view(w1.dtype)
        w2 = w2.view(torch.uint8)[local].view(w2.dtype)
        s1 = s1.view(experts, 2 * inter, -1)[local].reshape(-1, hidden // 32)
        s2 = s2.view(experts, hidden, -1)[local].reshape(-1, inter // 32)
    w1, w2 = shuffle_weight_a16w4(w1, 16, True), shuffle_weight_a16w4(w2, 16, False)
    s1 = shuffle_scale_a16w4(s1, local_experts, True)
    s2 = (
        shuffle_scale_a16w4(s2, local_experts, False)
        if b_dtype == "fp4"
        else fp4_utils.e8m0_shuffle(s2)
    )
    ids = torch.rand(tokens, experts, device="cuda").topk(topk, -1).indices.int()
    weights = torch.rand(tokens, topk, device="cuda").softmax(-1)
    meta = fm.MOEMetadata(
        functools.partial(
            stage1, kernelName=kernel_name(1, *tile, b_dtype=b_dtype, waves=waves)
        ),
        functools.partial(
            stage2,
            kernelName=kernel_name(
                2,
                tile_m=128 if waves == 4 else 256,
                swizzle=3,
                b_dtype=b_dtype,
                waves=waves,
                persistent_tiles=persistent,
            ),
        ),
        128 if waves == 4 else 256,
        0,
        prequant=False,
        fuse_quant="fp8",
        skip_inter_quant=True,
    )
    assert fm.stage2_uses_route_reduce(meta.stage2)

    def forward():
        return fm._fused_moe_impl(
            x,
            w1,
            w2,
            weights,
            ids,
            w1_scale=s1,
            w2_scale=s2,
            activation=(
                aiter.ActivationType.Silu
                if b_dtype == "fp4"
                else aiter.ActivationType.Swiglu
            ).value,
            quant_type=aiter.QuantType.per_1x32.value,
            gate_mode="interleave",
            swiglu_limit=limit,
            expert_mask=mask,
            _metadata_transform=lambda _: meta,
        )

    def check(out):
        ref_weights = weights if mask is None else weights * mask[ids.long()]
        ref = reference(x, r1, r2, ids, ref_weights, limit, b_dtype)
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


@pytest.mark.parametrize(
    "b_dtype,waves,persistent",
    [("fp8", 8, 0), ("fp4", 8, 0), ("fp4", 4, 0), ("fp8", 8, 2), ("fp8", 8, 4)],
)
def test_tuned_config_preserves_other_moe_paths(
    monkeypatch, tmp_path, b_dtype, waves, persistent
):
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
        "q_dtype_w": aiter.dtypes.fp4x2 if b_dtype == "fp4" else aiter.dtypes.fp8,
        "q_type": aiter.QuantType.per_1x32,
        "activation": (
            aiter.ActivationType.Silu
            if b_dtype == "fp4"
            else aiter.ActivationType.Swiglu
        ),
        "use_g1u1": True,
        "doweight_stage1": False,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "gate_mode": "interleave",
    }
    row = {k: str(int(v) if isinstance(v, bool) else v) for k, v in kwargs.items()}
    row["act_type"] = row.pop("activation")
    block_m = 128 if waves == 4 else 256
    row.update(
        gfx="gfx950",
        cu_num=256,
        block_m=block_m,
        ksplit=0,
        kernelName1=kernel_name(1, tile_m=block_m, b_dtype=b_dtype, waves=waves),
        kernelName2=kernel_name(
            2,
            tile_m=block_m,
            b_dtype=b_dtype,
            waves=waves,
            persistent_tiles=persistent,
        ),
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
        assert selected.block_m == block_m
        assert fm.stage2_uses_route_reduce(selected.stage2)
        for override in (
            {
                "activation": (
                    aiter.ActivationType.Swiglu
                    if b_dtype == "fp4"
                    else aiter.ActivationType.Silu
                )
            },
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


def test_four_wave_stage1_exact_padding_and_dirty_graph(monkeypatch):
    from aiter.ops.flydsl import mxfp8_moe_8wave as adapter

    torch.manual_seed(813)
    tokens, hidden, inter, experts, topk = 129, 256, 384, 3, 2
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    w1, s1, _ = quant_weight(
        torch.randn(experts, 2 * inter, hidden, device="cuda") * 0.1, "fp4"
    )
    w1 = shuffle_weight_a16w4(w1, 16, True)
    s1 = shuffle_scale_a16w4(s1, experts, True)
    w2 = torch.empty(experts, hidden, inter // 2, device="cuda", dtype=torch.int8)
    ids = torch.rand(tokens, experts, device="cuda").topk(topk, -1).indices.int()
    weights = torch.rand(tokens, topk, device="cuda").softmax(-1)
    records = {}
    run = adapter._run

    def record(kind, args, **kw):
        if kind == "gemm" and kw.get("num_waves") == 4:
            records.update(args=args, kwargs=kw)
        return run(kind, args, **kw)

    monkeypatch.setattr(adapter, "_run", record)
    results = {}
    for waves, block_m in [(8, 256), (4, 128)]:
        sorted_ids, _, expert_ids, valid, _ = fm.moe_sorting(
            ids,
            weights,
            experts,
            hidden,
            torch.bfloat16,
            block_size=block_m,
            accumulate=False,
        )
        q, s = stage1(
            x,
            w1,
            w2,
            sorted_ids,
            expert_ids,
            valid,
            None,
            topk,
            block_m=block_m,
            kernelName=kernel_name(1, tile_m=block_m, b_dtype="fp4", waves=waves),
            w1_scale=s1,
        )
        count = int(valid[0].item())
        rid = sorted_ids[:count].long()
        active = (rid & 0xFFFFFF) < tokens
        pos = torch.arange(count, device="cuda")[active]
        routes = (rid[active] & 0xFFFFFF) * topk + (rid[active] >> 24)
        results[waves] = q, s, pos, routes, torch.arange(count, device="cuda")[~active]
    oldq, olds, oldpos, oldroutes, _ = results[8]
    q, s, pos, routes, pad = results[4]
    inverse = torch.empty(tokens * topk, device="cuda", dtype=torch.int64)
    inverse[oldroutes] = oldpos
    refpos = inverse[routes]
    kg = torch.arange(q.shape[1] // 32, device="cuda")[None, :]

    def scale_index(rows):
        rows = rows[:, None]
        return (
            ((rows // 32 * 2 + kg // 8) * 64 + kg % 4 * 16 + rows % 16) * 4
            + kg // 4 % 2 * 2
            + rows // 16 % 2
        )

    def check():
        assert torch.equal(q[pos], oldq[refpos])
        assert torch.equal(
            s.flatten()[scale_index(pos)], olds.flatten()[scale_index(refpos)]
        )
        assert torch.count_nonzero(q[pad]).item() == 0
        assert torch.all(s.flatten()[scale_index(pad)] == 19).item()
        assert (
            torch.count_nonzero(q[: int(pos.numel() + pad.numel()), inter:]).item() == 0
        )

    def launch():
        args = records["args"]
        run("gemm", (*args[:-1], torch.cuda.current_stream()), **records["kwargs"])

    for value in (37, 83):
        records["args"][2].fill_(value)
        launch()
        check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for value in (19, 101):
        records["args"][2].fill_(value)
        graph.replay()
        check()


@pytest.mark.parametrize("stage", [1, 2])
def test_four_wave_aot(stage):
    from aiter.aot.flydsl.moe import compile_one_config
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    name = kernel_name(stage, tile_m=128, b_dtype="fp4", waves=4)
    params = get_flydsl_kernel_params(name)
    assert params["sort_block_m"] == 128 and params["num_waves"] == 4
    result = compile_one_config(
        name,
        model_dim=512,
        inter_dim=384,
        experts=7,
        topk=3,
        cu_num=256,
        token_num=257,
        **params,
    )
    assert result["compile_time"] is not None
    # AOT must also suppress dispatch when this launcher was warmed by runtime.
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "name",
    [
        "flydsl_moe1_mxfp8_4w_t128x256_xcd1",
        "flydsl_moe2_a8w4_4w_t256x256_xcd1",
        "flydsl_moe1_a8w4_8w_t128x256_xcd1",
        "flydsl_moe1_mxfp8_8w_t256x256_persistent2_xcd3",
        "flydsl_moe2_a8w4_8w_t256x256_persistent2_xcd3",
    ],
)
def test_prefill_rejects_incompatible_geometry(name):
    from aiter.ops.flydsl.mxfp8_moe_8wave import kernel_params

    with pytest.raises(ValueError):
        kernel_params(name)


@pytest.mark.parametrize("repeats", [2, 4])
def test_persistent_stage2_aot(repeats):
    from aiter.aot.flydsl.moe import compile_one_config
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    name = kernel_name(2, persistent_tiles=repeats, swizzle=3)
    params = get_flydsl_kernel_params(name)
    assert params["persistent_tiles"] == repeats and params["sort_block_m"] == 256
    result = compile_one_config(
        name,
        model_dim=512,
        inter_dim=384,
        experts=7,
        topk=3,
        cu_num=256,
        token_num=257,
        **params,
    )
    assert result["compile_time"] is not None
    torch.cuda.synchronize()

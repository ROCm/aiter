# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Determinism coverage for the MXFP4 MoE stage2 epilogue selection."""

import functools
import importlib
from pathlib import Path

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    _apply_mxfp4_stage2_determinism_policy,
    _flydsl_v2_stage2_wrapper,
    _is_flydsl_v2_layout_stage2,
    _mxfp4_a4w4_stage2_fw,
    get_2stage_cfgs,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.mxfp4_kname import (
    deterministic_v2_gemm2_name,
    parse_flydsl_v2_gemm2_kernel,
)

# The numerics of the swapped epilogue can only be exercised on the arch whose
# tuned rows select it; the host-side tests above cover the selection itself.
_GFX950_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="gfx950 GPU required",
)

MXMOE_G1 = "flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt"
MOE1_G1 = "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3"
OPUS_G1 = "opus_moe1_afp8_wfp4_bf16_t32x384_pair_kw7_m3_fp8"
ATOMIC_STAGE2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_sbm16"
REDUCE_STAGE2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_reduce_sbm16"
# dsv3_fp4_tuned_fmoe.csv pairs flydsl_moe1_afp4_wfp4_bf16 with this pair
# (token=32, inter_dim=512). The decode-shape row at inter_dim=256 is an MXMOE
# stage-1 now, and the 256-tile stage-2 was retuned to a 128-tile one.
DSV3_G1 = "flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3"
DSV3_ATOMIC_STAGE2 = (
    "flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_persist_nt_sbm32"
)
DSV3_REDUCE_STAGE2 = (
    "flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_reduce_persist_nt_sbm32"
)
# dsv4_fp8fp4_tuned_fmoe.csv pairs opus_moe1_afp8_wfp4_bf16 with this pair
OPUS_ATOMIC_STAGE2 = "flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_nt_sbm32"
OPUS_REDUCE_STAGE2 = "flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_reduce_nt_sbm32"


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        (ATOMIC_STAGE2, REDUCE_STAGE2),
        (
            "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_persist_sbm16",
            "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_reduce_persist_sbm16",
        ),
        (
            "flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_atomic_persist_nt_sbm64",
            "flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_persist_nt_sbm64",
        ),
        (
            "flydsl_moe2_layout_afp8_wfp4_fp4_t32x128x256_atomic_sbm32_bf16lds",
            "flydsl_moe2_layout_afp8_wfp4_fp4_t32x128x256_reduce_sbm32_bf16lds",
        ),
        (
            "flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_atomic_persist_nt_sbm64_bf16lds_sp2",
            "flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_persist_nt_sbm64_bf16lds_sp2",
        ),
    ),
)
def test_atomic_stage2_has_a_reduce_sibling(name, expected):
    selected = deterministic_v2_gemm2_name(name)

    assert selected == expected
    assert "_atomic" not in selected
    original = parse_flydsl_v2_gemm2_kernel(name)
    substituted = parse_flydsl_v2_gemm2_kernel(selected)
    assert substituted is not None, "substituted name must stay in the grammar"
    assert {**substituted, "epilog": "atomic"} == original


@pytest.mark.parametrize(
    "name",
    (
        REDUCE_STAGE2,
        # a layout-v2 row, but with the scatter epilogue: no reduce sibling
        "flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_scatter_sbm16",
        # native mxmoe GEMM2 and GEMM1 are different name families
        "flydsl_mxmoe_g2_a4w4_16x128x128",
        MXMOE_G1,
        "flydsl_moe1_afp4_wfp4_fp4_t16x256x256_nt",
        "",
        None,
    ),
)
def test_non_atomic_stage2_names_are_left_alone(name):
    assert deterministic_v2_gemm2_name(name) is None


@pytest.mark.parametrize(
    ("stage1", "atomic", "reduce"),
    (
        (MXMOE_G1, ATOMIC_STAGE2, REDUCE_STAGE2),
        # The epilogue belongs to the stage-2 kernel, so every stage-1 family
        # that can select a flydsl_moe2_layout_ GEMM2 is rewritten too.
        (MOE1_G1, DSV3_ATOMIC_STAGE2, DSV3_REDUCE_STAGE2),
        (OPUS_G1, OPUS_ATOMIC_STAGE2, OPUS_REDUCE_STAGE2),
    ),
)
def test_policy_rewrites_atomic_stage2_without_mutating_the_row(
    monkeypatch, stage1, atomic, reduce
):
    monkeypatch.delenv("AITER_MXFP4_ATOMIC_STAGE2", raising=False)
    row = {"kernelName1": stage1, "kernelName2": atomic, "ksplit": 0}

    selected = _apply_mxfp4_stage2_determinism_policy(row)

    assert selected is not row
    assert selected["kernelName2"] == reduce
    assert selected["kernelName1"] == stage1
    assert selected["ksplit"] == 0
    # the cached tuning row keeps the tuned epilogue
    assert row["kernelName2"] == atomic


def test_policy_allows_atomic_stage2_only_through_the_rollback_opt_in(monkeypatch):
    monkeypatch.setenv("AITER_MXFP4_ATOMIC_STAGE2", "1")
    row = {"kernelName1": MOE1_G1, "kernelName2": DSV3_ATOMIC_STAGE2}

    assert _apply_mxfp4_stage2_determinism_policy(row) is row


@pytest.mark.parametrize(
    "row",
    (
        {"kernelName1": MXMOE_G1, "kernelName2": REDUCE_STAGE2},
        {"kernelName1": MXMOE_G1, "kernelName2": "flydsl_mxmoe_g2_a4w4_16x128x128"},
        # the native flydsl_moe2 stage2 family is a different kernel and is
        # outside this policy
        {
            "kernelName1": MOE1_G1,
            "kernelName2": "flydsl_moe2_abf16_wfp4_bf16_t16x128x128_atomic_bnt2",
        },
        {"kernelName1": MXMOE_G1, "kernelName2": ""},
        {"kernelName1": MXMOE_G1},
        {},
    ),
)
def test_policy_is_a_noop_for_unaffected_rows(row):
    assert _apply_mxfp4_stage2_determinism_policy(row) is row


def test_policy_passes_through_a_missing_config():
    assert _apply_mxfp4_stage2_determinism_policy(None) is None


@pytest.fixture
def gfx950_tuned_config(monkeypatch):
    """Load one model config as if running on a 256-CU gfx950.

    The tuned rows are keyed by the host fp8 dtype, so the gfx950 spelling is
    pinned instead of inheriting whatever arch this test host reports, and the
    rollback env var is cleared so the default-policy assertions hold when the
    suite is launched with the opt-in exported.
    """
    fused_moe = importlib.import_module("aiter.fused_moe")
    monkeypatch.setattr(fused_moe, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fused_moe, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(dtypes, "fp8", torch.float8_e4m3fn)
    monkeypatch.delenv("AITER_MXFP4_ATOMIC_STAGE2", raising=False)

    def _load(filename):
        monkeypatch.setenv("AITER_CONFIG_FMOE", str(_model_config(filename)))
        monkeypatch.setattr(fused_moe, "cfg_2stages", None)
        monkeypatch.setattr(fused_moe, "cfg_2stages_by_file", {})
        fused_moe.AITER_CONFIGS.get_config_file.cache_clear()
        get_2stage_cfgs.cache_clear()
        return fused_moe

    yield _load
    get_2stage_cfgs.cache_clear()
    fused_moe.AITER_CONFIGS.get_config_file.cache_clear()


def _model_config(filename):
    return (
        Path(__file__).resolve().parents[1] / "aiter/configs/model_configs" / filename
    )


# token, model_dim, inter_dim, expert, topk, (q_dtype_a, q_dtype_w) per family.
# The dtype names resolve against the fixture-pinned `dtypes` at call time so
# an fp8 host never freezes its own spelling into the lookup key.
_DECODE_SHAPES = {
    "glm5": (1, 6144, 2048, 257, 9, "fp4x2", "fp4x2"),
    # inter_dim=512 is the remaining dsv3 row whose stage-1 is flydsl_moe1_;
    # the 256 one was retuned to the MXMOE front-end upstream.
    "dsv3": (32, 7168, 512, 257, 9, "fp4x2", "fp4x2"),
    "dsv4": (1, 7168, 384, 384, 6, "fp8", "fp4x2"),
}


def _decode_cfg(model, **overrides):
    """Decode-shape lookup for one tuned MXFP4 model config."""
    token, model_dim, inter_dim, expert, topk, a_dtype, w_dtype = _DECODE_SHAPES[model]
    return get_2stage_cfgs(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        expert=expert,
        topk=topk,
        dtype=dtypes.bf16,
        q_dtype_a=getattr(dtypes, a_dtype),
        q_dtype_w=getattr(dtypes, w_dtype),
        q_type=QuantType.per_1x32,
        use_g1u1=True,
        activation=ActivationType.Silu,
        doweight_stage1=False,
        hidden_pad=0,
        intermediate_pad=0,
        is_shuffled=True,
        opus_weights_shuffled=True,
        **overrides,
    )


def test_tuned_atomic_row_dispatches_the_reduce_epilogue(gfx950_tuned_config):
    gfx950_tuned_config("glm5_fp4_tuned_fmoe.csv")

    meta = _decode_cfg("glm5")

    assert meta.stage1.keywords["kernelName1"] == MXMOE_G1
    assert meta.stage2.keywords["kernelName2"] == REDUCE_STAGE2


def test_dispatch_keeps_the_tuned_row_and_honours_the_opt_in(
    gfx950_tuned_config, monkeypatch
):
    fused_moe = gfx950_tuned_config("glm5_fp4_tuned_fmoe.csv")
    _decode_cfg("glm5")
    tuned_rows = fused_moe.cfg_2stages[0]
    assert any(row["kernelName2"] == ATOMIC_STAGE2 for row in tuned_rows.values())

    monkeypatch.setenv("AITER_MXFP4_ATOMIC_STAGE2", "1")
    get_2stage_cfgs.cache_clear()
    meta = _decode_cfg("glm5")

    assert meta.stage2.keywords["kernelName2"] == ATOMIC_STAGE2


@pytest.mark.parametrize(
    ("config", "model", "stage1", "atomic", "reduce"),
    (
        (
            "dsv3_fp4_tuned_fmoe.csv",
            "dsv3",
            DSV3_G1,
            DSV3_ATOMIC_STAGE2,
            DSV3_REDUCE_STAGE2,
        ),
        (
            "dsv4_fp8fp4_tuned_fmoe.csv",
            "dsv4",
            OPUS_G1,
            OPUS_ATOMIC_STAGE2,
            OPUS_REDUCE_STAGE2,
        ),
    ),
)
def test_other_stage1_families_dispatch_the_reduce_epilogue(
    gfx950_tuned_config, config, model, stage1, atomic, reduce
):
    fused_moe = gfx950_tuned_config(config)

    meta = _decode_cfg(model)

    assert meta.stage1.keywords["kernelName"] == stage1
    assert meta.stage2.keywords["kernelName"] == reduce
    tuned_rows = fused_moe.cfg_2stages[0]
    assert any(row["kernelName2"] == atomic for row in tuned_rows.values())


def test_v2_layout_stage2_detection_covers_both_dispatch_spellings():
    """Both spellings reach the same v2 GEMM2, so both take the v2 arguments."""
    assert _is_flydsl_v2_layout_stage2(_flydsl_v2_stage2_wrapper)
    assert _is_flydsl_v2_layout_stage2(
        functools.partial(_mxfp4_a4w4_stage2_fw, kernelName2=REDUCE_STAGE2)
    )
    # a native mxmoe GEMM2 and an unrelated wrapper are different kernels
    assert not _is_flydsl_v2_layout_stage2(
        functools.partial(
            _mxfp4_a4w4_stage2_fw,
            kernelName2="flydsl_mxmoe_g2_a4w4_16x128x128",
        )
    )
    assert not _is_flydsl_v2_layout_stage2(None)


@pytest.mark.parametrize("use_valid_token_count", [False, True])
def test_mxmoe_frontend_forwards_the_route_ids_to_the_v2_wrapper(
    monkeypatch, use_valid_token_count
):
    """The front-end has to hand the ids to the v2 wrapper, or they are lost."""
    fused_moe = importlib.import_module("aiter.fused_moe")
    seen = {}

    def fake_wrapper(**kwargs):
        seen.update(kwargs)
        return kwargs["out"]

    monkeypatch.setattr(fused_moe, "_flydsl_v2_stage2_wrapper", fake_wrapper)
    out = torch.zeros((2, 128), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1], [2, -1]], dtype=torch.int32)
    expert_mask = torch.ones(4, dtype=torch.int32)

    result = fused_moe._mxfp4_a4w4_stage2_fw(
        torch.zeros((32, 64), dtype=torch.uint8),
        torch.zeros((4, 256), dtype=torch.uint8),
        torch.zeros((4, 128, 64), dtype=torch.uint8),
        torch.zeros(32, dtype=torch.int32),
        torch.zeros(32, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        out,
        2,
        kernelName2=REDUCE_STAGE2,
        expert_mask=expert_mask,
        topk_ids=topk_ids,
        use_valid_token_count=use_valid_token_count,
    )

    assert result is out
    assert seen["kernelName"] == REDUCE_STAGE2
    assert seen["topk_ids"] is topk_ids
    assert seen["expert_mask"] is expert_mask
    assert seen["use_valid_token_count"] is use_valid_token_count


def test_aot_precompiles_the_kernel_the_runtime_dispatches(
    gfx950_tuned_config, monkeypatch
):
    """AOT has to key the cache on the rewritten kernel, not the tuned row.

    Under ``FLYDSL_RUNTIME_RUN_ONLY=1`` a mismatch is a hard failure instead of
    a JIT compile, so the precompiled name has to be the dispatched one.
    """
    from aiter.aot.flydsl import mxfp4_moe

    monkeypatch.setattr(mxfp4_moe, "_ATOMIC_STAGE2", False)
    gfx950_tuned_config("glm5_fp4_tuned_fmoe.csv")
    jobs = mxfp4_moe.parse_csv(str(_model_config("glm5_fp4_tuned_fmoe.csv")))
    v2_jobs = [job for job in jobs if job.get("v2_stage2")]

    assert v2_jobs, "glm5 must yield layout-v2 stage2 jobs"
    assert all(job["epilog"] != "atomic" for job in v2_jobs)

    dispatched = _decode_cfg("glm5").stage2.keywords["kernelName2"]
    assert dispatched == REDUCE_STAGE2
    assert dispatched in {job["kernel_name"] for job in v2_jobs}


@_GFX950_GPU
def test_reduce_skips_dropped_routes_and_repeats_bitwise():
    """A route with a negative router id must not leak into the reduction.

    ``moe_sorting`` drops it and GEMM2 never writes its slot, so the buffer
    holds uninitialized memory there. The same launch twice must also return
    identical bits, which is what MTP and prefix reuse compare.
    """
    from aiter.ops.flydsl.moe_kernels import _run_moe_reduction

    token, topk, model_dim = 2, 4, 256
    target = torch.full(
        (token, topk, model_dim), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    for route, value in enumerate((1.0, 2.0, 3.0)):
        target[:, route].fill_(value)
    topk_ids = torch.tensor([[0, 1, 2, -1]] * token, dtype=torch.int32, device="cuda")

    outs = []
    for _ in range(2):
        out = torch.empty((token, model_dim), dtype=torch.bfloat16, device="cuda")
        _run_moe_reduction(target, out, token, topk, model_dim, topk_ids=topk_ids)
        outs.append(out.clone())
    torch.cuda.synchronize()

    assert not outs[0].isnan().any(), "dropped route leaked into the reduction"
    torch.testing.assert_close(outs[0], torch.full_like(outs[0], 6.0))
    assert torch.equal(outs[0], outs[1]), "reduce epilogue is not bitwise repeatable"


@_GFX950_GPU
@pytest.mark.parametrize("is_fp8", [False, True])
def test_reduce_stops_at_the_device_valid_token_count(is_fp8):
    """EP padding must not be reduced, and must not be gathered either.

    ``moe_sorting`` only sorts the first ``num_local_tokens`` rows, so the
    dispatch-buffer tail keeps whatever garbage ids the caller left there. A
    positive one would index past ``expert_mask``; the count guard has to stop
    the row before either load.
    """
    from aiter.ops.flydsl.kernels.mxfp4_gemm_common import fp8out_row_bytes
    from aiter.ops.flydsl.moe_kernels import _run_moe_reduction

    token, topk, model_dim, valid = 4, 4, 256, 2
    target = torch.full(
        (token, topk, model_dim), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    target[:valid].fill_(1.0)
    if is_fp8:
        packed = torch.empty(
            (token * topk, fp8out_row_bytes(model_dim)),
            dtype=torch.uint8,
            device="cuda",
        )
        packed[:, :model_dim] = (
            target.reshape(-1, model_dim).to(dtypes.fp8).view(torch.uint8)
        )
        packed[:, model_dim:] = 127  # E8M0 scale 1.
        target = packed
    # Poisoned tail: a valid-looking expert id far outside the mask.
    topk_ids = torch.full((token, topk), 1 << 20, dtype=torch.int32, device="cuda")
    topk_ids[:valid] = 0
    topk_ids[valid::2] = 0  # In-range padding still needs the token-count guard.
    expert_mask = torch.ones(4, dtype=torch.int32, device="cuda")
    valid_tokens = torch.tensor([valid], dtype=torch.int32, device="cuda")

    out = torch.empty((token, model_dim), dtype=torch.bfloat16, device="cuda")

    def launch():
        _run_moe_reduction(
            target,
            out,
            token,
            topk,
            model_dim,
            expert_mask=expert_mask,
            topk_ids=topk_ids,
            valid_tokens=valid_tokens,
            is_fp8=is_fp8,
        )

    launch()
    torch.cuda.synchronize()

    assert not out.isnan().any(), "padded row leaked into the reduction"
    torch.testing.assert_close(out[valid:], torch.zeros_like(out[valid:]))
    torch.testing.assert_close(out[:valid], torch.full_like(out[:valid], float(topk)))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for count in (1, 0, valid):
        valid_tokens.fill_(count)
        graph.replay()
        expected = torch.zeros_like(out)
        expected[:count] = float(topk)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@_GFX950_GPU
def test_v2_reduce_stage2_matches_the_reference_and_repeats():
    """End-to-end numerics of the stage2 kernel the policy dispatches.

    Several experts and topk >= 4 so each output row sums competing routed
    contributions with non-uniform weights -- the case the atomic epilogue
    cannot reproduce bitwise. Both eager launches and CUDA graph replays must
    return identical bits.
    """
    from aiter.fused_moe import moe_sorting, torch_moe_stage2
    from aiter.ops.quant import (
        mxfp4_moe_sort_fwd,
        per_1x32_f4_quant,
        per_1x32_f8_scale_f8_quant,
    )
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
    from aiter.test_common import checkAllclose
    from aiter.utility import fp4_utils

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    token, model_dim, inter_dim, experts, topk, block_m = 32, 128, 384, 8, 4, 32

    score = torch.randn((token, experts), dtype=torch.bfloat16, device="cuda")
    topk_weights, topk_ids = torch.topk(
        torch.softmax(score.float(), dim=-1), topk, dim=-1
    )
    # Renormalize so the routes of one token compete with unequal weights.
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    a2 = torch.randn((token, topk, inter_dim), dtype=torch.bfloat16, device="cuda") / 4
    w1 = torch.zeros(
        (experts, inter_dim * 2, model_dim), dtype=torch.bfloat16, device="cuda"
    )
    w2 = (
        torch.randn(
            (experts, model_dim, inter_dim), dtype=torch.bfloat16, device="cuda"
        )
        / 4
    )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, experts, model_dim, torch.bfloat16, block_m
    )
    # Exercise the extra sorted tiles beyond the token-major activation rows.
    assert num_valid_ids[0].item() > token * topk

    a2_q, a2_scale = per_1x32_f8_scale_f8_quant(
        a2, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    w1_q, _ = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(experts, inter_dim * 2, model_dim // 2)
    w2_q = w2_q.view(experts, model_dim, inter_dim // 2)

    a2_dequant = (
        a2_q.float().view(token, topk, inter_dim // 32, 32)
        * fp4_utils.e8m0_to_f32(a2_scale).view(token, topk, inter_dim // 32, 1)
    ).view(token, topk, inter_dim)
    reference = torch_moe_stage2(
        a2_dequant,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=torch.bfloat16,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=None,
        doweight=True,
    )
    a2_scale_sorted = mxfp4_moe_sort_fwd(
        a2_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        cols=inter_dim,
    )
    w2_shuffled = shuffle_weight_a16w4(w2_q, 16, False)
    w2_scale_shuffled = shuffle_scale_a16w4(w2_scale, experts, False)

    def launch(out):
        _flydsl_v2_stage2_wrapper(
            inter_states=a2_q.reshape(token * topk, inter_dim),
            w1=None,
            w2=w2_shuffled,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            out=out,
            topk=topk,
            kernelName=OPUS_REDUCE_STAGE2,
            model_dim=model_dim,
            inter_dim=inter_dim,
            num_experts=experts,
            w2_scale=w2_scale_shuffled,
            a2_scale=a2_scale_sorted,
            sorted_weights=sorted_weights,
            topk_ids=topk_ids,
            block_m=block_m,
        )
        return out

    eager = []
    for _ in range(4):
        out = torch.empty((token, model_dim), dtype=torch.bfloat16, device="cuda")
        launch(out)
        eager.append(out.clone())
    torch.cuda.synchronize()

    for run in eager[1:]:
        assert torch.equal(eager[0], run), "eager reduce epilogue is not repeatable"

    assert not eager[0].isnan().any()
    err = checkAllclose(
        reference.float(),
        eager[0].float(),
        rtol=1e-2,
        atol=1e-2,
        msg="v2 reduce stage2 vs reference",
    )
    assert err < 0.05, f"v2 reduce stage2 mismatch ratio {err:.3f}"

    # CUDA graph replays: the same static output, so any order dependence shows
    # up as differing bytes between replays.
    graph_out = torch.empty((token, model_dim), dtype=torch.bfloat16, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            launch(graph_out)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(graph_out)
    replays = []
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        replays.append(graph_out.clone())

    for run in replays[1:]:
        assert torch.equal(replays[0], run), "graph reduce epilogue is not repeatable"
    assert torch.equal(replays[0], eager[0]), "graph and eager disagree"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

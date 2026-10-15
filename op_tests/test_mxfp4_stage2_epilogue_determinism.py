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
DSV3_ATOMIC_STAGE2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32"
DSV3_REDUCE_STAGE2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_reduce_nt_sbm32"
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
    assert name.replace("_atomic_", "_reduce_", 1) == selected
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

    selected = _apply_mxfp4_stage2_determinism_policy(row, keys=("policy",))

    assert selected is not row
    assert selected["kernelName2"] == reduce
    assert selected["kernelName1"] == stage1
    assert selected["ksplit"] == 0
    # the cached tuning row keeps the tuned epilogue
    assert row["kernelName2"] == atomic


def test_policy_allows_atomic_stage2_only_through_the_rollback_opt_in(monkeypatch):
    monkeypatch.setenv("AITER_MXFP4_ATOMIC_STAGE2", "1")
    row = {"kernelName1": MOE1_G1, "kernelName2": DSV3_ATOMIC_STAGE2}

    assert _apply_mxfp4_stage2_determinism_policy(row, keys=("optin",)) is row


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
    assert _apply_mxfp4_stage2_determinism_policy(row, keys=("noop",)) is row


def test_policy_passes_through_a_missing_config():
    assert _apply_mxfp4_stage2_determinism_policy(None, keys=("none",)) is None


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
    "dsv3": (32, 7168, 256, 257, 9, "fp4x2", "fp4x2"),
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
            MOE1_G1,
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


def test_reduce_kernel_name_separates_the_dropped_route_guard():
    """The compiled kernel name is the AOT disk-cache key, so the guard shows."""
    from aiter.ops.flydsl.kernels.moe_reduce import compile_moe_reduction

    def kernel_name(launcher):
        kernel = next(
            cell.cell_contents
            for cell in launcher.func.__closure__
            if type(cell.cell_contents).__name__ == "KernelFunction"
        )
        return getattr(kernel, "_name", None) or kernel._kernel_name

    plain = compile_moe_reduction(topk=4, model_dim=256, dtype_str="bf16")
    guarded = compile_moe_reduction(
        topk=4, model_dim=256, dtype_str="bf16", skip_invalid_ids=True
    )

    assert plain is not guarded
    assert "_i0" in kernel_name(plain)
    assert "_i1" in kernel_name(guarded)
    # the same Constexpr set still hits the launcher cache
    assert compile_moe_reduction(topk=4, model_dim=256, dtype_str="bf16") is plain


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


def _call_v2_reduce_wrapper(monkeypatch, *, topk_ids):
    """Run the v2 reduce wrapper with GEMM2 and the reduction stubbed out.

    ``torch.empty`` is poisoned so the per-route buffer still carries NaN where
    GEMM2 (mocked to write nothing) left it: the dropped-route slots.
    """
    fused_moe = importlib.import_module("aiter.fused_moe")
    from aiter.ops.flydsl import moe_kernels
    from aiter.ops.flydsl.kernels import mxmoe_dispatcher

    seen = {}

    def fake_gemm2(**kwargs):
        seen["gemm2_out"] = kwargs["out"]

    def fake_reduction(target, out, *args, **kwargs):
        seen["target"] = target.clone()
        seen["reduction_ids"] = kwargs.get("topk_ids")

    real_empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        tensor = real_empty(*args, **kwargs)
        if tensor.dtype.is_floating_point:
            tensor.fill_(float("nan"))
        return tensor

    with monkeypatch.context() as patch:
        patch.setattr(mxmoe_dispatcher, "mxfp4_moe_gemm2", fake_gemm2)
        patch.setattr(moe_kernels, "_run_moe_reduction", fake_reduction)
        patch.setattr(torch, "empty", poisoned_empty)
        fused_moe._flydsl_v2_stage2_wrapper(
            inter_states=torch.zeros((32, 64), dtype=torch.uint8),
            w1=None,
            w2=torch.zeros((2, 128, 64), dtype=torch.uint8),
            sorted_token_ids=torch.zeros(32, dtype=torch.int32),
            sorted_expert_ids=torch.zeros(32, dtype=torch.int32),
            num_valid_ids=torch.zeros(1, dtype=torch.int32),
            out=torch.zeros((2, 128), dtype=torch.bfloat16),
            topk=2,
            kernelName=REDUCE_STAGE2,
            model_dim=128,
            inter_dim=64,
            num_experts=2,
            block_m=16,
            topk_ids=topk_ids,
        )
    return seen


def test_reduce_target_is_zeroed_only_when_the_route_ids_are_unavailable(
    monkeypatch,
):
    """Dropped routes are the kernel's job once it can read the router ids."""
    ids = torch.zeros((2, 2), dtype=torch.int32)
    with_ids = _call_v2_reduce_wrapper(monkeypatch, topk_ids=ids)
    assert with_ids["reduction_ids"] is ids
    assert torch.isnan(
        with_ids["target"]
    ).all(), "the zero-fill is dead weight when the kernel skips invalid routes"

    without_ids = _call_v2_reduce_wrapper(monkeypatch, topk_ids=None)
    assert without_ids["reduction_ids"] is None
    assert not torch.isnan(without_ids["target"]).any()
    assert (without_ids["target"] == 0).all()


def test_reduce_derives_the_dropped_route_guard_from_the_ids(monkeypatch):
    """The specialization is a pure function of the arguments, so the AOT
    precompile and the runtime call site cannot disagree about it."""
    from aiter.ops.flydsl import moe_kernels
    from aiter.ops.flydsl.kernels import moe_reduce

    calls = []
    with monkeypatch.context() as patch:
        patch.setattr(
            moe_reduce, "compile_moe_reduction", lambda **kwargs: calls.append(kwargs)
        )
        patch.setattr(moe_kernels, "_run_compiled", lambda exe, args: None)
        target = torch.zeros((2, 2, 8), dtype=torch.bfloat16)
        out = torch.empty((2, 8), dtype=torch.bfloat16)
        ids = torch.zeros((2, 2), dtype=torch.int32)

        moe_kernels._run_moe_reduction(target, out, 2, 2, 8, stream=0)
        moe_kernels._run_moe_reduction(target, out, 2, 2, 8, topk_ids=ids, stream=0)

    assert [call["skip_invalid_ids"] for call in calls] == [False, True]
    assert [call["use_mask"] for call in calls] == [False, False]


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
def test_v2_reduce_stage2_matches_the_reference_and_repeats():
    """End-to-end numerics of the stage2 kernel the policy dispatches."""
    from aiter.fused_moe import moe_sorting, torch_moe_stage2
    from aiter.ops.quant import (
        mxfp4_moe_sort_fwd,
        per_1x32_f4_quant,
        per_1x32_f8_scale_f8_quant,
    )
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
    from aiter.utility import fp4_utils

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    token, model_dim, inter_dim, experts, topk, block_m = 32, 128, 384, 1, 1, 32
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
    topk_ids = torch.zeros((token, topk), dtype=torch.int32, device="cuda")
    topk_weights = torch.ones((token, topk), dtype=torch.float32, device="cuda")
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, experts, model_dim, torch.bfloat16, block_m
    )

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

    outs = []
    for _ in range(2):
        out = torch.empty((token, model_dim), dtype=torch.bfloat16, device="cuda")
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
        outs.append(out.clone())
    torch.cuda.synchronize()

    assert not outs[0].isnan().any()
    torch.testing.assert_close(outs[0], reference, atol=1.0, rtol=0.05)
    assert torch.equal(outs[0], outs[1]), "reduce epilogue is not bitwise repeatable"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

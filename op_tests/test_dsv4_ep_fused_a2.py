# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Focused DSV4 EP validation for FlyDSL stage1 fused FP8 A2 quantization.

The production-shape validation is intentionally an explicit script rather than
part of the default pytest collection:

    HIP_VISIBLE_DEVICES=0 python op_tests/test_dsv4_ep_fused_a2.py
"""

import json
import os
import time

import pytest
import torch

import aiter
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

MODEL_DIM = 7168
INTER_DIM = 3072
LOCAL_E = 48
TOPK = 6
TIERS = (512, 1024, 4096, 8192, 16384, 32768)
EXPECTED = {
    512: (
        "flydsl_moe1_afp8_wfp4_bf16_t64x256x256_w3_gui_fp8",
        "flydsl_moe2_afp8_wfp4_bf16_t64x256x256_atomic_bnt2_persist",
    ),
    1024: (
        "flydsl_moe1_afp8_wfp4_bf16_t128x256x256_gui_fp8",
        "flydsl_moe2_afp8_wfp4_bf16_t64x128x256_atomic_persist_sbm128",
    ),
    4096: (
        "flydsl_moe1_afp8_wfp4_bf16_t128x256x256_gui_fp8",
        "flydsl_moe2_afp8_wfp4_bf16_t64x256x256_atomic_sbm128",
    ),
    8192: (
        "flydsl_moe1_afp8_wfp4_bf16_t128x256x256_w3_bnt0_gui_fp8",
        "flydsl_moe2_afp8_wfp4_bf16_t64x256x256_atomic_persist_sbm128",
    ),
    16384: (
        "flydsl_moe1_afp8_wfp4_bf16_t128x256x256_bnt0_gui_fp8",
        "flydsl_moe2_afp8_wfp4_bf16_t64x128x256_atomic_persist_sbm128",
    ),
    32768: (
        "flydsl_moe1_afp8_wfp4_bf16_t128x256x256_bnt0_gui_fp8",
        "flydsl_moe2_afp8_wfp4_bf16_t64x128x256_atomic_persist_sbm128",
    ),
}


def _metadata(token: int, *, no_fake_expert: bool):
    return fm.get_2stage_cfgs(
        token,
        MODEL_DIM,
        INTER_DIM,
        LOCAL_E,
        TOPK,
        torch.bfloat16,
        dtypes.fp8,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        GateMode.INTERLEAVE.value,
        is_ep=True,
        ep_has_fake_expert=not no_fake_expert,
    )


def _kernel_name(stage) -> str:
    return getattr(stage, "keywords", {}).get("kernelName", "")


def _reset_default_config():
    os.environ.pop("AITER_CONFIG_FMOE", None)
    AITER_CONFIGS.get_config_file.cache_clear()
    fm.get_2stage_cfgs.cache_clear()
    fm.cfg_2stages = None


def test_production_metadata_exact_rows(monkeypatch):
    """CPU/config assertion: the default merge selects every no-fake row."""
    monkeypatch.delenv("AITER_CONFIG_FMOE", raising=False)
    monkeypatch.setattr(fm, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fm, "get_gfx_runtime", lambda: "gfx950")
    old_cfg = fm.cfg_2stages
    _reset_default_config()
    try:
        for token in TIERS:
            production = _metadata(token, no_fake_expert=True)
            assert (
                _kernel_name(production.stage1),
                _kernel_name(production.stage2),
            ) == EXPECTED[token]
            assert production.fuse_quant == "fp8"
            assert "_fp8" in _kernel_name(production.stage1).split("_t")[-1]
    finally:
        fm.get_2stage_cfgs.cache_clear()
        fm.cfg_2stages = old_cfg
        AITER_CONFIGS.get_config_file.cache_clear()


def test_prefill_padded_lookup_uses_32768_row(monkeypatch):
    """M in [32768, 131071] uses the production 32768 no-fake row."""
    monkeypatch.delenv("AITER_CONFIG_FMOE", raising=False)
    monkeypatch.setenv("AITER_FLYDSL_EP_NO_FAKE_EXPERT", "1")
    monkeypatch.setattr(fm, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fm, "get_gfx_runtime", lambda: "gfx950")
    old_cfg = fm.cfg_2stages
    _reset_default_config()
    try:
        for runtime_m in (32768, 32769, 65535, 65536, 131071):
            padded_m = fm.get_padded_M(runtime_m)
            assert padded_m == 32768
            metadata = _metadata(padded_m, no_fake_expert=True)
            assert (
                _kernel_name(metadata.stage1),
                _kernel_name(metadata.stage2),
            ) == EXPECTED[32768]
            assert metadata.fuse_quant == "fp8"
    finally:
        fm.get_2stage_cfgs.cache_clear()
        fm.cfg_2stages = old_cfg
        AITER_CONFIGS.get_config_file.cache_clear()


def _make_weights():
    # Deterministic packed FP4 payloads and finite 2^-6 E8M0 scales.  Creating
    # quantized tensors directly avoids materializing >6 GiB of BF16 sources.
    gen = torch.Generator(device="cuda").manual_seed(20260728)
    w1 = torch.randint(
        0,
        256,
        (LOCAL_E, INTER_DIM * 2, MODEL_DIM // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=gen,
    ).view(dtypes.fp4x2)
    w2 = torch.randint(
        0,
        256,
        (LOCAL_E, MODEL_DIM, INTER_DIM // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=gen,
    ).view(dtypes.fp4x2)
    w1_scale = (
        torch.full(
            (LOCAL_E * INTER_DIM * 2, MODEL_DIM // 32),
            121,
            dtype=torch.uint8,
            device="cuda",
        )
        .view(dtypes.fp8_e8m0)
        .contiguous()
    )
    w2_scale = (
        torch.full(
            (LOCAL_E * MODEL_DIM, INTER_DIM // 32),
            121,
            dtype=torch.uint8,
            device="cuda",
        )
        .view(dtypes.fp8_e8m0)
        .contiguous()
    )
    return (
        shuffle_weight_a16w4(w1, 16, True),
        shuffle_weight_a16w4(w2, 16, False),
        shuffle_scale_a16w4(w1_scale, LOCAL_E, True),
        shuffle_scale_a16w4(w2_scale, LOCAL_E, False),
    )


def _make_routes(token: int):
    # Deliberately skew the 48 local experts across a 96-expert global space.
    # Only 12 local experts receive work, leaving 36 local experts empty.
    mask = torch.zeros(96, dtype=dtypes.i32, device="cuda")
    local_global_ids = torch.tensor(
        [(i * 17 + 3) % 96 for i in range(LOCAL_E)],
        dtype=dtypes.i32,
        device="cuda",
    ).unique()
    # The modular sequence is unique for i<48 because gcd(17, 96)==1.
    assert local_global_ids.numel() == LOCAL_E
    mask[local_global_ids] = 1
    nonlocal_ids = torch.where(mask == 0)[0].to(dtypes.i32)
    active_local = local_global_ids[::4]

    row = torch.arange(token, device="cuda")[:, None]
    slot = torch.arange(TOPK, device="cuda")[None, :]
    local_route = active_local[(row * 5 + slot * 3) % active_local.numel()]
    nonlocal_route = nonlocal_ids[(row * 7 + slot * 11) % nonlocal_ids.numel()]
    # First quarter is local-only; remaining rows alternate local/nonlocal.
    use_local = (row < token // 4) | ((row + slot) % 2 == 0)
    ids = torch.where(use_local, local_route, nonlocal_route).to(dtypes.i32)
    weights = (0.05 + ((row * 13 + slot * 7) % 19).to(torch.float32) / 50).contiguous()
    return mask, ids.contiguous(), weights, use_local


def _run_one(
    hidden,
    w1,
    w2,
    w1_scale,
    w2_scale,
    mask,
    ids,
    weights,
    *,
    no_fake_expert: bool,
):
    if no_fake_expert:
        os.environ["AITER_FLYDSL_EP_NO_FAKE_EXPERT"] = "1"
    else:
        os.environ.pop("AITER_FLYDSL_EP_NO_FAKE_EXPERT", None)
    fm.get_2stage_cfgs.cache_clear()
    metadata = _metadata(hidden.shape[0], no_fake_expert=no_fake_expert)

    captured = {}
    original_stage2 = metadata.stage2

    def stage2_probe(*args, **kwargs):
        captured["a2"] = args[0]
        captured["sorted_token_capacity"] = args[3].numel()
        captured["num_valid_ids"] = args[5].detach().cpu()
        captured["a2_scale"] = kwargs.get("a2_scale")
        return original_stage2(*args, **kwargs)

    metadata.stage2 = stage2_probe
    quant_calls = 0
    original_quant = fm.fused_dynamic_mxfp8_quant_moe_sort

    def quant_probe(*args, **kwargs):
        nonlocal quant_calls
        quant_calls += 1
        return original_quant(*args, **kwargs)

    fm.fused_dynamic_mxfp8_quant_moe_sort = quant_probe
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        out = fm.fused_moe(
            hidden,
            w1,
            w2,
            weights,
            ids,
            expert_mask=mask,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            gate_mode=GateMode.INTERLEAVE.value,
        )
        torch.cuda.synchronize()
    finally:
        fm.fused_dynamic_mxfp8_quant_moe_sort = original_quant
        metadata.stage2 = original_stage2
    runtime_ms = (time.perf_counter() - start) * 1e3
    peak_memory_bytes = torch.cuda.max_memory_allocated()

    assert captured["a2"].dtype == dtypes.fp8
    assert captured["a2"].shape == (hidden.shape[0], TOPK, INTER_DIM)
    assert captured["a2"].is_contiguous()
    assert captured["a2_scale"].dtype == dtypes.fp8_e8m0
    assert captured["a2_scale"].ndim == 2
    assert captured["a2_scale"].shape[1] == INTER_DIM // 32
    assert captured["a2_scale"].stride(1) == 1
    assert quant_calls == (1 if no_fake_expert else 2)
    return out, metadata, captured, runtime_ms, peak_memory_bytes, quant_calls


def _errors(production, baseline):
    production = production.float()
    baseline = baseline.float()
    abs_err = (production - baseline).abs()
    rel_err = abs_err / baseline.abs().clamp_min(1e-6)
    close = torch.isclose(production, baseline, rtol=1e-2, atol=1e-2)
    dot_denom = (production.square() + baseline.square()).sum()
    logits_diff = 1 - 2 * (production * baseline).sum() / dot_denom
    return {
        "max_abs": abs_err.max().item(),
        "mean_abs": abs_err.mean().item(),
        "max_rel": rel_err.max().item(),
        "mean_rel": rel_err.mean().item(),
        "outlier_ratio_rtol1e-2_atol1e-2": (~close).float().mean().item(),
        "logits_diff": logits_diff.item(),
    }


def _print_summary(rows, verdict, limitation):
    payload = {
        "verdict": verdict,
        "limitation": limitation,
        "environment": {
            "AITER_CONFIG_FMOE": "unset (normal model-config merge)",
            "production_AITER_FLYDSL_EP_NO_FAKE_EXPERT": "1",
            "reference_AITER_FLYDSL_EP_NO_FAKE_EXPERT": "unset",
        },
        "shape_contract": {
            "gfx": "gfx950",
            "model_dim": MODEL_DIM,
            "inter_dim": INTER_DIM,
            "local_experts": LOCAL_E,
            "global_experts": 96,
            "routed_topk": TOPK,
            "fake_topk_slot": False,
        },
        "tiers": rows,
    }
    print(json.dumps(payload, indent=2))


@pytest.mark.skip(reason="production-shape GPU validation; execute this file directly")
def test_production_gpu_validation():
    run_validation()


def run_validation():
    assert torch.cuda.is_available()
    assert aiter.get_gfx() == "gfx950"
    assert aiter.get_cu_num() == 256
    assert "AITER_CONFIG_FMOE" not in os.environ
    _reset_default_config()
    torch.manual_seed(20260728)
    rows = []
    verdict = "FAIL"
    limitation = ""
    try:
        w1, w2, w1_scale, w2_scale = _make_weights()
        for token in TIERS:
            hidden = (
                torch.randn(token, MODEL_DIM, dtype=torch.bfloat16, device="cuda") / 8
            )
            mask, ids, weights, use_local = _make_routes(token)
            (
                production,
                production_meta,
                cap,
                production_ms,
                production_peak_memory,
                production_qcalls,
            ) = _run_one(
                hidden,
                w1,
                w2,
                w1_scale,
                w2_scale,
                mask,
                ids,
                weights,
                no_fake_expert=True,
            )
            (
                baseline,
                baseline_meta,
                _,
                baseline_ms,
                baseline_peak_memory,
                baseline_qcalls,
            ) = _run_one(
                hidden,
                w1,
                w2,
                w1_scale,
                w2_scale,
                mask,
                ids,
                weights,
                no_fake_expert=False,
            )
            errors = _errors(production, baseline)
            assert torch.isfinite(production).all()
            assert torch.isfinite(baseline).all()
            assert errors["outlier_ratio_rtol1e-2_atol1e-2"] <= 0.05
            assert errors["logits_diff"] <= 0.01

            # Atomic stage2 is not bitwise deterministic, so establish its
            # repeat envelope before perturbing nonlocal slots.  The masked
            # perturbation must remain within the established A8W4 tolerance
            # and cannot materially exceed ordinary repeat noise.
            repeated, _, _, _, _, _ = _run_one(
                hidden,
                w1,
                w2,
                w1_scale,
                w2_scale,
                mask,
                ids,
                weights,
                no_fake_expert=True,
            )
            repeat_errors = _errors(repeated, production)
            changed_weights = weights.clone()
            changed_weights[~use_local] = changed_weights[~use_local] * 1000 + 17
            invariant, _, _, _, _, _ = _run_one(
                hidden,
                w1,
                w2,
                w1_scale,
                w2_scale,
                mask,
                ids,
                changed_weights,
                no_fake_expert=True,
            )
            invariance_errors = _errors(invariant, production)
            assert invariance_errors["outlier_ratio_rtol1e-2_atol1e-2"] <= max(
                0.05,
                repeat_errors["outlier_ratio_rtol1e-2_atol1e-2"] + 0.01,
            )
            assert invariance_errors["logits_diff"] <= max(
                0.01, repeat_errors["logits_diff"] * 2 + 1e-6
            )

            production_names = (
                _kernel_name(production_meta.stage1),
                _kernel_name(production_meta.stage2),
            )
            assert production_names == EXPECTED[token]
            assert production_meta.fuse_quant == "fp8"
            valid_counts = cap["num_valid_ids"].tolist()
            # Opus stores padded valid rows in [0] and the original token
            # boundary in [1].  The first may exceed token*topk due to expert
            # block padding, but must remain inside sorted-token capacity.
            assert 0 <= valid_counts[0] <= cap["sorted_token_capacity"]
            assert valid_counts[1] == token
            rows.append(
                {
                    "token": token,
                    "production_stage1": production_names[0],
                    "production_stage2": production_names[1],
                    "baseline_stage1": _kernel_name(baseline_meta.stage1),
                    "baseline_stage2": _kernel_name(baseline_meta.stage2),
                    "production_fuse_quant": production_meta.fuse_quant,
                    "baseline_fuse_quant": baseline_meta.fuse_quant,
                    "a2_dtype": str(cap["a2"].dtype),
                    "a2_shape": list(cap["a2"].shape),
                    "scale_dtype": str(cap["a2_scale"].dtype),
                    "scale_shape": list(cap["a2_scale"].shape),
                    "scale_stride": list(cap["a2_scale"].stride()),
                    "num_valid_ids": valid_counts,
                    "production_quant_calls": production_qcalls,
                    "baseline_quant_calls": baseline_qcalls,
                    "production_runtime_ms": production_ms,
                    "baseline_runtime_ms": baseline_ms,
                    "production_peak_memory_bytes": production_peak_memory,
                    "baseline_peak_memory_bytes": baseline_peak_memory,
                    "errors": errors,
                    "atomic_repeat_errors": repeat_errors,
                    "masked_slot_invariance_errors": invariance_errors,
                    "verdict": "PASS",
                }
            )
            del (
                hidden,
                mask,
                ids,
                weights,
                production,
                baseline,
                repeated,
                invariant,
                cap,
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        verdict = "PASS"
    except Exception as exc:
        limitation = f"hard stop: {type(exc).__name__}: {exc}"
        raise
    finally:
        _print_summary(rows, verdict, limitation)
        fm.get_2stage_cfgs.cache_clear()
        fm.cfg_2stages = None
        AITER_CONFIGS.get_config_file.cache_clear()
        os.environ.pop("AITER_FLYDSL_EP_NO_FAKE_EXPERT", None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_validation()

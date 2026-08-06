# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter import fused_moe as fused_moe_module
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.moe_sorting_kernel import moe_sorting_flydsl
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
)
from aiter.ops.flydsl.mxfp4_gemm2_kernels import flydsl_mxfp4_gemm2
from aiter.ops.flydsl.mxfp4_v2_tune_utils import (
    populate_v2_intermediate_from_ref,
)
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import shuffle_scale, shuffle_weight
from aiter.test_common import checkAllclose
from aiter.utility import fp4_utils
from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import (
    FmoeTuner,
    Mxfp4FlydslTuner,
)

_SKIP_GFX950_FLYDSL = pytest.mark.skipif(
    get_gfx() != "gfx950" or not is_flydsl_available(),
    reason="gfx950 FlyDSL required",
)


def _check_close(ref, out, label):
    assert torch.isfinite(out).all(), f"{label}: output contains NaN or Inf"
    ref_f32 = ref.float().reshape(1, -1)
    out_f32 = out.float().reshape(1, -1)
    cosine = torch.nn.functional.cosine_similarity(
        ref_f32, out_f32, dim=1, eps=1e-12
    ).item()
    assert cosine >= 0.999, (
        f"{label}: cosine similarity {cosine:.6f} is below 0.999"
    )
    err = checkAllclose(ref, out, msg=label, atol=0.01, rtol=0.05)
    assert err == 0 or err <= 0.05, f"{label}: error ratio {err} exceeds 0.05"


def test_check_close_rejects_all_zero_output():
    ref = torch.tensor((0.6, -0.4, 0.2), dtype=torch.float32)
    out = torch.zeros_like(ref)
    with pytest.raises(AssertionError, match="cosine similarity"):
        _check_close(ref, out, "all_zero")


def _dequant_fp4(value, scale, cols):
    values = fp4_utils.mxfp4_to_f32(value).view(*value.shape[:-1], cols // 32, 32)
    scales = fp4_utils.e8m0_to_f32(scale).view(*value.shape[:-1], cols // 32, 1)
    return (values * scales).view(*value.shape[:-1], cols)


def prepare_direct_stage2_case(
    inter_dim, bk, *, bm=32, use_nt=False, epilog="atomic", seed=123
):
    token = 33
    if epilog == "atomic":
        model_dim, expert, topk = 256, 2, 2
    else:
        # The generated aux scatter kernels' smallest production key is
        # MiniMax's (H=3072, TOPK=8). The direct sort/GEMM inputs can still use
        # a small expert count while exercising the real reduction path.
        model_dim, expert, topk = 3072, 8, 8
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    token_idx = torch.arange(token, dtype=torch.int32, device=device)
    slot_idx = torch.arange(topk, dtype=torch.int32, device=device)
    topk_ids = (token_idx[:, None] + slot_idx[None, :]) % expert
    topk_weights = torch.arange(
        topk, 0, -1, dtype=torch.float32, device=device
    ).repeat(token, 1)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    max_padded = token * topk + expert * bm - topk
    max_sorted = (max_padded + bm - 1) // bm * bm
    sorted_ids = torch.empty(max_sorted, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(max_sorted, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(max_sorted // bm, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    moe_buf = torch.empty((token, model_dim), dtype=torch.bfloat16, device=device)
    moe_sorting_flydsl(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        expert,
        bm,
    )
    torch.cuda.synchronize()
    n_sorted = int(num_valid_ids[0].item())
    sorted_row = torch.arange(max_sorted, dtype=torch.int32, device=device)
    packed_token = sorted_ids & 0x00FFFFFF
    packed_slot = (sorted_ids >> 24) & 0xFF
    valid_row = (
        (sorted_row < n_sorted) & (packed_token < token) & (packed_slot < topk)
    )
    reverse_sorted = torch.full(
        (token * topk,), -1, dtype=torch.int32, device=device
    )
    route = packed_token[valid_row] * topk + packed_slot[valid_row]
    reverse_sorted[route.long()] = sorted_row[valid_row]
    assert (reverse_sorted >= 0).all(), "not every routed slot was sorted"

    if epilog == "nonatomic_mxfp4":
        # The f4out path intentionally quantizes each route before reduction.
        # Use exactly representable values so the strict final-output checker
        # tests kernel correctness rather than expected second-quantization loss,
        # while touching every K32 scale group and both signs across N.
        inter = torch.zeros(
            (token, topk, inter_dim), dtype=torch.bfloat16, device=device
        )
        inter[..., ::32] = 1
        w2 = torch.zeros(
            (expert, model_dim, inter_dim), dtype=torch.bfloat16, device=device
        )
        signs = torch.where(
            torch.arange(model_dim, device=device) % 2 == 0,
            1,
            -1,
        ).to(torch.bfloat16)
        w2[:, :, ::32] = signs[None, :, None]
    else:
        inter = (
            torch.randn(
                (token, topk, inter_dim), dtype=torch.bfloat16, device=device
            )
            / 10
        )
        w2 = (
            torch.randn(
                (expert, model_dim, inter_dim), dtype=torch.bfloat16, device=device
            )
            / 10
        )

    inter_q, inter_scale = per_1x32_f4_quant(
        inter.view(token * topk, inter_dim),
        quant_dtype=dtypes.fp4x2,
        shuffle=False,
    )
    w2_q, w2_scale = per_1x32_f4_quant(
        w2, quant_dtype=dtypes.fp4x2, shuffle=False
    )

    layout = {
        "sti": sorted_ids,
        "swt": sorted_weights,
        "sei": sorted_expert_ids,
        "cumsum": num_valid_ids,
        "n": n_sorted,
        "max_sorted": max_sorted,
    }
    # Reconstruct the native GEMM1 sorted-row payload and E8M0 layout. The v2
    # helper is byte-compatible when sort block and compute block are both BM32.
    populate_v2_intermediate_from_ref(
        {"ref1": inter, "stage2_adtype": "fp4"},
        layout,
        token,
        topk,
        bm,
    )

    inter_dequant = _dequant_fp4(inter_q, inter_scale, inter_dim).view(
        token, topk, inter_dim
    )
    w2_dequant = _dequant_fp4(w2_q, w2_scale, inter_dim)
    ref = torch.zeros((token, model_dim), dtype=torch.float32, device=device)
    for slot in range(topk):
        expert_weight = w2_dequant[topk_ids[:, slot].long()]
        route = torch.bmm(
            expert_weight, inter_dequant[:, slot].unsqueeze(-1)
        ).squeeze(-1)
        ref += route * topk_weights[:, slot, None]

    atomic = epilog == "atomic"
    mxfp4out = epilog == "nonatomic_mxfp4"
    cshuffle = epilog == "nonatomic_cshuffle"
    out = torch.zeros((token, model_dim), dtype=torch.bfloat16, device=device)
    flat_out_scale = None
    if atomic:
        flat_out = out
    elif mxfp4out:
        flat_out = torch.full(
            (max_sorted, model_dim // 2),
            0xFF,
            dtype=torch.uint8,
            device=device,
        )
        flat_out_scale = torch.full(
            (max_sorted, model_dim // 32),
            0xFF,
            dtype=torch.uint8,
            device=device,
        )
    else:
        flat_out = torch.empty(
            (max_sorted, model_dim), dtype=torch.bfloat16, device=device
        )
    launch_kwargs = {
        "inter_sorted_quant": layout["isq"],
        "inter_sorted_shuffled_scale": layout["iss"],
        "w2_u8": shuffle_weight(
            w2_q, (16, 16), is_guinterleave=False, gate_up=False
        ),
        "w2_scale_u8": shuffle_scale(
            w2_scale.view(expert * model_dim, inter_dim // 32),
            expert,
            is_guinterleave=False,
            gate_up=False,
        ),
        "sorted_expert_ids": sorted_expert_ids,
        "cumsum_tensor": num_valid_ids,
        "sorted_token_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "flat_out": flat_out,
        "M_logical": token,
        "max_sorted": max_sorted,
        "BM": bm,
        "use_nt": use_nt,
        "atomic": atomic,
        "mxfp4out": mxfp4out,
        "NE": expert,
        "D_HIDDEN": model_dim,
        "D_INTER": inter_dim,
        "topk": topk,
        "flat_out_scale": flat_out_scale,
        "cshuffle": cshuffle,
        "BN": 256,
        "BK": bk,
    }
    return {
        "ref": ref.to(torch.bfloat16),
        "out": out,
        "flat_out": flat_out,
        "flat_out_scale": flat_out_scale,
        "reverse_sorted": reverse_sorted,
        "kwargs": launch_kwargs,
    }


def run_direct_stage2(case):
    case["out"].zero_()
    flydsl_mxfp4_gemm2(**case["kwargs"])
    kwargs = case["kwargs"]
    if kwargs["mxfp4out"]:
        valid_rows = case["reverse_sorted"].long()
        assert (case["flat_out"][valid_rows] != 0xFF).any(), (
            "f4out payload was not written"
        )
        assert (case["flat_out_scale"][valid_rows] != 0xFF).any(), (
            "f4out scale was not written"
        )
        aiter.mxfp4_moe_scatter_reduce_q(
            flat_out_q=case["flat_out"],
            flat_out_scale=case["flat_out_scale"],
            reverse_sorted=case["reverse_sorted"],
            sorted_weights=kwargs["sorted_weights"],
            out=case["out"],
            NE=kwargs["NE"],
            TOPK=kwargs["topk"],
            D_HIDDEN=kwargs["D_HIDDEN"],
            MB=kwargs["BM"],
        )
    elif not kwargs["atomic"]:
        aiter.mxfp4_moe_scatter_reduce(
            flat_out=case["flat_out"],
            reverse_sorted=case["reverse_sorted"],
            sorted_weights=kwargs["sorted_weights"],
            out=case["out"],
            NE=kwargs["NE"],
            TOPK=kwargs["topk"],
            D_HIDDEN=kwargs["D_HIDDEN"],
            MB=kwargs["BM"],
        )
    return case["out"]


_NATIVE_VARIANTS = [
    pytest.param(16, False, "atomic", id="bm16-atomic"),
    pytest.param(16, True, "atomic", id="bm16-atomic-nt"),
    pytest.param(32, False, "atomic", id="bm32-atomic"),
    pytest.param(32, True, "atomic", id="bm32-atomic-nt"),
    pytest.param(64, False, "atomic", id="bm64-atomic"),
    pytest.param(64, True, "atomic", id="bm64-atomic-nt"),
    pytest.param(128, False, "nonatomic", id="bm128-nonatomic"),
    pytest.param(128, False, "nonatomic_mxfp4", id="bm128-f4out"),
    pytest.param(32, False, "nonatomic_cshuffle", id="bm32-cshuffle"),
    pytest.param(64, False, "nonatomic_cshuffle", id="bm64-cshuffle"),
    pytest.param(128, False, "nonatomic_cshuffle", id="bm128-cshuffle"),
]

_K_CASES = [
    pytest.param(384, 128, id="k384-bk128"),
    pytest.param(512, 128, id="k512-bk128"),
    pytest.param(512, 256, id="k512-bk256"),
]


@pytest.mark.parametrize(("bm", "use_nt", "epilog"), _NATIVE_VARIANTS)
@pytest.mark.parametrize(("inter_dim", "bk"), _K_CASES)
@_SKIP_GFX950_FLYDSL
def test_native_full_variant_matrix(bm, use_nt, epilog, inter_dim, bk):
    case = prepare_direct_stage2_case(
        inter_dim,
        bk,
        bm=bm,
        use_nt=use_nt,
        epilog=epilog,
        seed=1000 + bm + inter_dim + bk,
    )
    out = run_direct_stage2(case)
    _check_close(
        case["ref"].float(),
        out.float(),
        f"native_{epilog}_bm{bm}_nt{int(use_nt)}_k{inter_dim}_bk{bk}",
    )


@_SKIP_GFX950_FLYDSL
def test_native_k384_bk128_high_level_smoke(monkeypatch):
    # DSV4-EP8 contributes the generated aux key (NE=48, H=7168, TOPK=6).
    # Aux sorting ignores D_INTER, so use the real K384 native BK128 contraction.
    token, model_dim, inter_dim, expert, topk, bm = 1, 7168, 384, 48, 6, 32
    activation = ActivationType.Silu
    hidden = torch.ones(
        (token, model_dim), dtype=torch.bfloat16, device="cuda"
    )
    a1_qt, a1_scale = per_1x32_f4_quant(
        hidden, quant_dtype=dtypes.fp4x2, shuffle=False
    )
    w1 = torch.zeros(
        (expert, inter_dim * 2, model_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    w1[:, :, 0] = 1
    w1_qt, w1_scale = per_1x32_f4_quant(
        w1, quant_dtype=dtypes.fp4x2, shuffle=False
    )
    w1_a16 = shuffle_weight(
        w1_qt, (16, 16), is_guinterleave=False, gate_up=True
    )
    w1s_a16 = shuffle_scale(
        w1_scale, expert, is_guinterleave=False, gate_up=True
    )
    del w1

    w2 = torch.zeros(
        (expert, model_dim, inter_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    signs = torch.where(
        torch.arange(model_dim, device="cuda") % 2 == 0,
        1,
        -1,
    ).to(torch.bfloat16)
    w2[:, :, ::32] = signs[None, :, None]
    w2_qt, w2_scale = per_1x32_f4_quant(
        w2, quant_dtype=dtypes.fp4x2, shuffle=False
    )
    w2_a16 = shuffle_weight(
        w2_qt, (16, 16), is_guinterleave=False, gate_up=False
    )
    w2s_a16 = shuffle_scale(
        w2_scale, expert, is_guinterleave=False, gate_up=False
    )
    del w2

    topk_ids = torch.arange(
        topk, dtype=torch.int32, device="cuda"
    ).view(1, topk)
    route_weights = torch.arange(
        topk, 0, -1, dtype=torch.float32, device="cuda"
    ).view(1, topk)
    topk_weights = route_weights / route_weights.sum()

    kn1 = Mxfp4FlydslTuner._g1_kname(bm, False, False)
    kn2 = Mxfp4FlydslTuner._g2_kname(bm, False, "atomic", 128)
    metadata = fused_moe_module.MOEMetadata(
        stage1=functools.partial(
            fused_moe_module._mxfp4_a4w4_stage1_fw,
            kernelName1=kn1,
            interleave=False,
        ),
        stage2=functools.partial(
            fused_moe_module._mxfp4_a4w4_stage2_fw,
            kernelName2=kn2,
        ),
        block_m=bm,
        ksplit=0,
        fuse_quant="fp4",
        output_aux=True,
        prequant=False,
    )
    monkeypatch.setattr(
        fused_moe_module,
        "get_2stage_cfgs",
        lambda *args, **kwargs: metadata,
    )

    out = fused_moe_module.fused_moe(
        hidden,
        w1_a16,
        w2_a16,
        topk_weights,
        topk_ids,
        activation=activation,
        quant_type=QuantType.per_1x32,
        w1_scale=w1s_a16,
        w2_scale=w2s_a16,
        dtype=dtypes.bf16,
        gate_mode="separated",
    )
    ref1 = FmoeTuner.run_torch_moe_stage1(
        a1_qt,
        w1_qt[:topk],
        w2_qt[:topk],
        topk_weights,
        topk_ids,
        a1_scale,
        w1_scale[: topk * inter_dim * 2],
        dtype=dtypes.bf16,
        activation=activation,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        topk=topk,
        situ_beta=DEFAULT_SITUV2_BETA,
        situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
    )
    inter_q, inter_scale = per_1x32_f4_quant(
        ref1.view(token * topk, inter_dim),
        quant_dtype=dtypes.fp4x2,
        shuffle=False,
    )
    inter_dequant = _dequant_fp4(inter_q, inter_scale, inter_dim).view(
        token, topk, inter_dim
    )
    w2_dequant = _dequant_fp4(
        w2_qt[:topk],
        w2_scale[: topk * model_dim].view(
            topk, model_dim, inter_dim // 32
        ),
        inter_dim,
    )
    ref = torch.zeros((token, model_dim), dtype=torch.float32, device="cuda")
    for slot in range(topk):
        expert_weight = w2_dequant[topk_ids[:, slot].long()]
        route = torch.bmm(
            expert_weight, inter_dequant[:, slot].unsqueeze(-1)
        ).squeeze(-1)
        ref += route * topk_weights[:, slot, None]
    _check_close(
        ref.to(dtypes.bf16).float(),
        out.float(),
        "native_high_level_k384_bk128",
    )


@pytest.mark.parametrize(
    ("inter_dim", "bk"),
    [
        pytest.param(384, 128, id="k384-bk128"),
        pytest.param(512, 256, id="k512-bk256"),
    ],
)
@_SKIP_GFX950_FLYDSL
def test_native_atomic_bm32_direct_stage2(inter_dim, bk):
    case = prepare_direct_stage2_case(inter_dim, bk)
    out = run_direct_stage2(case)
    _check_close(
        case["ref"].float(), out.float(), f"native_bm32_k{inter_dim}_bk{bk}"
    )


@pytest.mark.parametrize(
    "inter_dim",
    [
        pytest.param(512, id="k512-4tiles"),
        pytest.param(1024, id="k1024-8tiles"),
    ],
)
@_SKIP_GFX950_FLYDSL
def test_native_atomic_bm16_bk128_deep_k(inter_dim):
    case = prepare_direct_stage2_case(inter_dim, 128, bm=16)
    out = run_direct_stage2(case)
    _check_close(
        case["ref"].float(), out.float(), f"native_bm16_k{inter_dim}_bk128"
    )

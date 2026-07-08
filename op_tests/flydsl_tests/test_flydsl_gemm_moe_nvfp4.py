from __future__ import annotations

import csv
import time
from dataclasses import dataclass

import pytest
import torch

from aiter.fused_moe import (
    fused_moe,
    get_2stage_cfgs,
    get_padded_M,
    moe_sorting,
    torch_moe,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.enum import ActivationType, QuantType
from aiter.ops.flydsl.moe_kernels import (
    flydsl_moe_stage1,
    flydsl_moe_stage2,
    get_flydsl_kernel_params,
)
from aiter.utility.fp4_utils import (
    _quantize_nvfp4_weight_for_moe,
    shuffle_nvfp4_weight_for_flydsl,
)


DTYPE = torch.bfloat16
NVFP4_BF16_QDTYPE = "nvfp4_bf16"
MODEL_DIMS = [128, 192, 256, 512, 1024, 1536, 2048]
INTER_DIMS = [128, 192, 256, 512, 1024, 1536, 2048]
TOKENS = [1, 16, 234]



@dataclass
class Nvfp4MoeCase:
    hidden: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    w1_packed_flydsl: torch.Tensor
    w1_scale_flydsl: torch.Tensor
    w1_global_scale: torch.Tensor
    w1_bf16_qdq: torch.Tensor
    w2_packed_flydsl: torch.Tensor
    w2_scale_flydsl: torch.Tensor
    w2_global_scale: torch.Tensor
    w2_bf16_qdq: torch.Tensor
    experts: int
    model_dim: int
    inter_dim: int
    topk: int


def _assert_close(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    max_abs = diff.max().item()
    max_rel = (diff / expected_f.abs().clamp_min(1e-12)).max().item()
    print(f"{label}: max_abs={max_abs:.6g}, max_rel={max_rel:.6g}")
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _make_case(
    tokens: int = 16,
    experts: int = 8,
    model_dim: int = 256,
    inter_dim: int = 128,
    topk: int = 2,
) -> Nvfp4MoeCase:
    torch.manual_seed(123)
    device = torch.device("cuda")

    hidden = (
        torch.randn((tokens, model_dim), device=device, dtype=DTYPE) * 0.2
    ).contiguous()
    router_logits = torch.randn((tokens, experts), device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=1)
    topk_weights = torch.softmax(topk_weights, dim=1).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()

    w1 = (
        torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=DTYPE)
        * 0.1
    ).contiguous()
    w2 = (
        torch.randn((experts, model_dim, inter_dim), device=device, dtype=DTYPE) * 0.1
    ).contiguous()

    w1_packed, w1_scale, w1_global_scale = _quantize_nvfp4_weight_for_moe(w1)
    w2_packed, w2_scale, w2_global_scale = _quantize_nvfp4_weight_for_moe(w2)

    w1_packed_flydsl = shuffle_nvfp4_weight_for_flydsl(w1_packed).contiguous()
    w2_packed_flydsl = shuffle_nvfp4_weight_for_flydsl(w2_packed).contiguous()
    setattr(w1_packed_flydsl, "is_shuffled", True)
    setattr(w2_packed_flydsl, "is_shuffled", True)

    return Nvfp4MoeCase(
        hidden=hidden,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w1_packed_flydsl=w1_packed_flydsl,
        w1_scale_flydsl=w1_scale.permute(0, 2, 1).contiguous(),
        w1_global_scale=w1_global_scale.contiguous(),
        w1_bf16_qdq=w1_scale.dequantized_weight.contiguous(),
        w2_packed_flydsl=w2_packed_flydsl,
        w2_scale_flydsl=w2_scale.permute(0, 2, 1).contiguous(),
        w2_global_scale=w2_global_scale.contiguous(),
        w2_bf16_qdq=w2_scale.dequantized_weight.contiguous(),
        experts=experts,
        model_dim=model_dim,
        inter_dim=inter_dim,
        topk=topk,
    )


def _has_invalid_nvfp4_flydsl_weight_layout(model_dim: int, inter_dim: int) -> bool:
    w1_n_out = 2 * inter_dim
    w1_packed_k = model_dim // 2
    w2_n_out = model_dim
    w2_packed_k = inter_dim // 2
    return (
        w1_n_out % 16 != 0
        or w1_packed_k % 32 != 0
        or w2_n_out % 16 != 0
        or w2_packed_k % 32 != 0
    )


def _make_case_or_assert_invalid_layout(
    tokens: int,
    experts: int,
    model_dim: int,
    inter_dim: int,
    topk: int,
) -> Nvfp4MoeCase | None:
    if _has_invalid_nvfp4_flydsl_weight_layout(model_dim, inter_dim):
        with pytest.raises(ValueError, match="NVFP4 FlyDSL weight requires"):
            _make_case(
                tokens=tokens,
                experts=experts,
                model_dim=model_dim,
                inter_dim=inter_dim,
                topk=topk,
            )
        pytest.skip("unsupported case")

    return _make_case(
        tokens=tokens,
        experts=experts,
        model_dim=model_dim,
        inter_dim=inter_dim,
        topk=topk,
    )


def _route(case: Nvfp4MoeCase, block_m: int):
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        case.topk_ids,
        case.topk_weights,
        case.experts,
        case.model_dim,
        DTYPE,
        block_m,
    )
    num_valid_ids = num_valid_ids[:1].contiguous()
    valid_count = int(num_valid_ids.item())
    valid_blocks = (valid_count + block_m - 1) // block_m
    valid_elems = valid_blocks * block_m
    return (
        sorted_ids[:valid_elems].contiguous(),
        sorted_weights[:valid_elems].contiguous(),
        sorted_expert_ids[:valid_blocks].contiguous(),
        num_valid_ids,
    )


def _nvfp4_metadata(case: Nvfp4MoeCase):
    return get_2stage_cfgs(
        case.hidden.shape[0],
        case.model_dim,
        case.inter_dim,
        case.experts,
        case.topk,
        DTYPE,
        DTYPE,
        NVFP4_BF16_QDTYPE,
        QuantType.No,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
    )


def _flydsl_params(stage_func):
    kernel_name = getattr(stage_func, "keywords", {}).get("kernelName", "")
    params = get_flydsl_kernel_params(kernel_name)
    assert params is not None, kernel_name
    assert params["a_dtype"] == "bf16", params
    assert params["b_dtype"] == "nvfp4", params
    assert params["out_dtype"] == "bf16", params
    return params


def _is_unsupported_nvfp4_stage1_shape(
    model_dim: int,
    inter_dim: int,
    params: dict,
) -> bool:
    tile_n = int(params["tile_n"])
    tile_k = int(params["tile_k"])
    k_tiles = model_dim // tile_k
    return inter_dim % tile_n != 0 or k_tiles < 2 or k_tiles % 2 != 0


def _is_unsupported_nvfp4_stage2_shape(model_dim: int, params: dict) -> bool:
    tile_n = int(params["tile_n"])
    return model_dim % tile_n != 0


def test_online_tuning(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    tune_file = tmp_path / "tuned_fmoe.csv"
    tune_file.write_text(
        "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,"
        "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,"
        "us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw,_tag\n"
    )
    monkeypatch.setenv("AITER_CONFIG_FMOE", str(tune_file))
    monkeypatch.setenv("AITER_ONLINE_TUNE", "1")

    def tuned_tokens() -> list[int]:
        with tune_file.open(newline="") as f:
            return [
                int(row["token"])
                for row in csv.DictReader(f)
                if row["q_dtype_w"] == NVFP4_BF16_QDTYPE
                and "flydsl_moe" in row["kernelName1"]
            ]

    def get_metadata(m: int):
        metadata = get_2stage_cfgs(
            get_padded_M(m),
            256,
            128,
            128,
            8,
            DTYPE,
            DTYPE,
            NVFP4_BF16_QDTYPE,
            QuantType.No,
            True,
            ActivationType.Silu,
            False,
            0,
            0,
            True,
        )
        assert metadata.stage1 is not None
        assert metadata.stage2 is not None
        return metadata

    for token in [1, 2, 8]:
        assert token not in tuned_tokens()
        get_metadata(token)
        assert token in tuned_tokens()

    before_csv = tune_file.read_text()
    start = time.perf_counter()
    get_metadata(6)
    assert time.perf_counter() - start < 2
    assert tune_file.read_text() == before_csv

    assert 16 not in tuned_tokens()
    get_metadata(12)
    assert 16 in tuned_tokens()

    before_csv = tune_file.read_text()
    get_metadata(16)
    assert tune_file.read_text() == before_csv


@pytest.mark.parametrize("model_dim", MODEL_DIMS)
@pytest.mark.parametrize("inter_dim", INTER_DIMS)
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("experts", [128])
@pytest.mark.parametrize("tokens", TOKENS)
def test_stage1_correctness(
    tokens: int,
    experts: int,
    model_dim: int,
    inter_dim: int,
    topk: int,
) -> None:
    case = _make_case_or_assert_invalid_layout(
        tokens=tokens,
        experts=experts,
        model_dim=model_dim,
        inter_dim=inter_dim,
        topk=topk,
    )

    metadata = get_2stage_cfgs(
        case.hidden.shape[0],
        case.model_dim,
        case.inter_dim,
        case.experts,
        case.topk,
        DTYPE,
        DTYPE,
        NVFP4_BF16_QDTYPE,
        QuantType.No,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
    )
    params = _flydsl_params(metadata.stage1)
    block_m = int(metadata.block_m)
    sorted_ids, _, sorted_expert_ids, num_valid_ids = _route(case, block_m)
    out = torch.empty(
        (case.hidden.shape[0], case.topk, case.inter_dim),
        dtype=DTYPE,
        device=case.hidden.device,
    )
    if _is_unsupported_nvfp4_stage1_shape(case.model_dim, case.inter_dim, params):
        with pytest.raises(ValueError, match="FlyDSL nvfp4_bf16 stage1 requires"):
            flydsl_moe_stage1(
                a=case.hidden,
                w1=case.w1_packed_flydsl,
                sorted_token_ids=sorted_ids,
                sorted_expert_ids=sorted_expert_ids,
                num_valid_ids=num_valid_ids,
                out=out,
                topk=case.topk,
                tile_m=params["tile_m"],
                tile_n=params["tile_n"],
                tile_k=params["tile_k"],
                a_dtype="bf16",
                b_dtype="nvfp4",
                out_dtype="bf16",
                w1_scale=case.w1_scale_flydsl,
                global_scale=case.w1_global_scale,
                sorted_weights=None,
                use_async_copy=True,
                k_batch=params.get("k_batch", 1),
                waves_per_eu=params.get("waves_per_eu", 3),
                b_nt=params.get("b_nt", 2),
                gate_mode=params.get("gate_mode", "separated"),
                xcd_swizzle=params.get("xcd_swizzle", 0),
            )
        pytest.skip("unsupported case")

    ref = torch_moe_stage1(
        case.hidden,
        case.w1_bf16_qdq,
        case.w2_bf16_qdq,
        case.topk_weights,
        case.topk_ids,
        dtype=DTYPE,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
        doweight=False,
    )

    out = flydsl_moe_stage1(
        a=case.hidden,
        w1=case.w1_packed_flydsl,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        topk=case.topk,
        tile_m=params["tile_m"],
        tile_n=params["tile_n"],
        tile_k=params["tile_k"],
        a_dtype="bf16",
        b_dtype="nvfp4",
        out_dtype="bf16",
        w1_scale=case.w1_scale_flydsl,
        global_scale=case.w1_global_scale,
        sorted_weights=None,
        use_async_copy=True,
        k_batch=params.get("k_batch", 1),
        waves_per_eu=params.get("waves_per_eu", 3),
        b_nt=params.get("b_nt", 2),
        gate_mode=params.get("gate_mode", "separated"),
        xcd_swizzle=params.get("xcd_swizzle", 0),
    )
    torch.cuda.synchronize()
    _assert_close("stage1", out, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("model_dim", MODEL_DIMS)
@pytest.mark.parametrize("inter_dim", INTER_DIMS)
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("experts", [128])
@pytest.mark.parametrize("tokens", TOKENS)
def test_stage2_correctness(
    tokens: int,
    experts: int,
    model_dim: int,
    inter_dim: int,
    topk: int,
) -> None:
    case = _make_case_or_assert_invalid_layout(
        tokens=tokens,
        experts=experts,
        model_dim=model_dim,
        inter_dim=inter_dim,
        topk=topk,
    )

    metadata = get_2stage_cfgs(
        case.hidden.shape[0],
        case.model_dim,
        case.inter_dim,
        case.experts,
        case.topk,
        DTYPE,
        DTYPE,
        NVFP4_BF16_QDTYPE,
        QuantType.No,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
    )
    params = _flydsl_params(metadata.stage2)
    block_m = int(metadata.block_m)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = _route(case, block_m)
    torch.manual_seed(456)
    inter_states = (
        torch.randn(
            (case.hidden.shape[0], case.topk, case.inter_dim),
            device=case.hidden.device,
            dtype=DTYPE,
        )
        * 0.2
    ).contiguous()
    out = torch.zeros(
        (case.hidden.shape[0], case.model_dim),
        dtype=DTYPE,
        device=case.hidden.device,
    )
    if _is_unsupported_nvfp4_stage2_shape(case.model_dim, params):
        with pytest.raises(ValueError, match="FlyDSL nvfp4_bf16 stage2 requires"):
            flydsl_moe_stage2(
                inter_states=inter_states,
                w2=case.w2_packed_flydsl,
                sorted_token_ids=sorted_ids,
                sorted_expert_ids=sorted_expert_ids,
                num_valid_ids=num_valid_ids,
                out=out,
                topk=case.topk,
                tile_m=params["tile_m"],
                tile_n=params["tile_n"],
                tile_k=params["tile_k"],
                a_dtype="bf16",
                b_dtype="nvfp4",
                out_dtype="bf16",
                mode=params.get("mode", "atomic"),
                w2_scale=case.w2_scale_flydsl,
                global_scale=case.w2_global_scale,
                a2_scale=None,
                sorted_weights=sorted_weights,
                sort_block_m=params.get("sort_block_m", 0),
                persist=params.get("persist", None),
                b_nt=params.get("b_nt", 0),
                xcd_swizzle=params.get("xcd_swizzle", 0),
            )
        pytest.skip("unsupported case")

    ref = torch_moe_stage2(
        inter_states,
        case.w1_bf16_qdq,
        case.w2_bf16_qdq,
        case.topk_weights,
        case.topk_ids,
        dtype=DTYPE,
        quant_type=QuantType.No,
        doweight=True,
    )

    out = torch.zeros_like(ref)
    flydsl_moe_stage2(
        inter_states=inter_states,
        w2=case.w2_packed_flydsl,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        topk=case.topk,
        tile_m=params["tile_m"],
        tile_n=params["tile_n"],
        tile_k=params["tile_k"],
        a_dtype="bf16",
        b_dtype="nvfp4",
        out_dtype="bf16",
        mode=params.get("mode", "atomic"),
        w2_scale=case.w2_scale_flydsl,
        global_scale=case.w2_global_scale,
        a2_scale=None,
        sorted_weights=sorted_weights,
        sort_block_m=params.get("sort_block_m", 0),
        persist=params.get("persist", None),
        b_nt=params.get("b_nt", 0),
        xcd_swizzle=params.get("xcd_swizzle", 0),
    )
    torch.cuda.synchronize()
    _assert_close("stage2", out, ref, atol=1.5e-1, rtol=1.5e-1)


@pytest.mark.parametrize("model_dim", MODEL_DIMS)
@pytest.mark.parametrize("inter_dim", INTER_DIMS)
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("experts", [128])
@pytest.mark.parametrize("tokens", TOKENS)
def test_fused_moe_2stages_correctness(
    tokens: int,
    experts: int,
    model_dim: int,
    inter_dim: int,
    topk: int,
) -> None:
    case = _make_case_or_assert_invalid_layout(
        tokens=tokens,
        experts=experts,
        model_dim=model_dim,
        inter_dim=inter_dim,
        topk=topk,
    )

    metadata = get_2stage_cfgs(
        case.hidden.shape[0],
        case.model_dim,
        case.inter_dim,
        case.experts,
        case.topk,
        DTYPE,
        DTYPE,
        NVFP4_BF16_QDTYPE,
        QuantType.No,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
    )
    params = _flydsl_params(metadata.stage1)
    if _is_unsupported_nvfp4_stage1_shape(case.model_dim, case.inter_dim, params):
        with pytest.raises(ValueError, match="FlyDSL nvfp4_bf16 stage1 requires"):
            fused_moe(
                case.hidden,
                case.w1_packed_flydsl,
                case.w2_packed_flydsl,
                case.topk_weights,
                case.topk_ids,
                activation=ActivationType.Silu,
                quant_type=QuantType.No,
                dtype=DTYPE,
                w1_scale=case.w1_scale_flydsl,
                w2_scale=case.w2_scale_flydsl,
                w1_global_scale=case.w1_global_scale,
                w2_global_scale=case.w2_global_scale,
            )
        pytest.skip("unsupported case")

    ref = torch_moe(
        case.hidden,
        case.w1_bf16_qdq,
        case.w2_bf16_qdq,
        case.topk_weights,
        case.topk_ids,
        activation=ActivationType.Silu,
    )

    out = fused_moe(
        case.hidden,
        case.w1_packed_flydsl,
        case.w2_packed_flydsl,
        case.topk_weights,
        case.topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
        dtype=DTYPE,
        w1_scale=case.w1_scale_flydsl,
        w2_scale=case.w2_scale_flydsl,
        w1_global_scale=case.w1_global_scale,
        w2_global_scale=case.w2_global_scale,
    )
    torch.cuda.synchronize()
    _assert_close("fused_moe_2stages", out, ref, atol=2e-1, rtol=2e-1)

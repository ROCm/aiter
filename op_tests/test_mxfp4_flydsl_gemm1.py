# SPDX-License-Identifier: MIT
import pytest
import torch


def test_port_module_imports_and_constants():
    """端口内核模块可导入，且对外暴露编译入口、grid 与 Kimi 常量。"""
    from aiter.ops.flydsl.kernels import mxfp4_gemm1 as port

    assert callable(port.compile_gemm1_a4w4_port)
    assert callable(port.gemm1_grid)
    assert (port.NE, port.K, port.INTER, port.TOPK) == (385, 7168, 512, 9)
    assert port.N_OUT == 1024


def test_guard_rejects_bad_shape():
    """非 Kimi 形状必须 fail-loud。"""
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import _assert_supported

    with pytest.raises(NotImplementedError, match="Kimi"):
        _assert_supported(
            NE=257, D_HIDDEN=7168, D_INTER=512, topk=9,
            BM=32, use_nt=True, inline_quant=False,
        )


def test_guard_rejects_bad_variant():
    """支持组合外的变体必须 fail-loud。"""
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import _assert_supported

    with pytest.raises(NotImplementedError, match="变体|variant"):
        _assert_supported(
            NE=385, D_HIDDEN=7168, D_INTER=512, topk=9,
            BM=64, use_nt=True, inline_quant=False,
        )


def test_guard_accepts_supported():
    """Kimi 形状 + 支持变体不抛异常（不触发编译）。"""
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import _assert_supported

    for BM, nt, iq in [(32, True, False), (32, False, False),
                       (128, False, False), (16, True, True)]:
        _assert_supported(
            NE=385, D_HIDDEN=7168, D_INTER=512, topk=9,
            BM=BM, use_nt=nt, inline_quant=iq,
        )


_HAS_CUDA = torch.cuda.is_available()


def _is_gfx950():
    if not _HAS_CUDA:
        return False
    try:
        name = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        name = ""
    return "gfx95" in name


_GFX950 = pytest.mark.skipif(
    not _is_gfx950(), reason="flydsl gemm1 需要 gfx950 (mfma_scale_f32_16x16x128_f8f6f4)"
)


def _build_kimi_mx(device, M, seed=2):
    import aiter
    from aiter import QuantType, dtypes
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    NE, H, INTER, TOPK = 385, 7168, 512, 9
    torch.manual_seed(seed)
    tq = aiter.get_torch_quant(QuantType.per_1x32)
    w1 = torch.randn((NE, 2 * INTER, H), dtype=dtypes.bf16, device=device) / 10
    w2 = torch.randn((NE, H, INTER), dtype=dtypes.bf16, device=device) / 10
    w1q, w1s = tq(w1, quant_dtype=dtypes.fp4x2)
    w2q, w2s = tq(w2, quant_dtype=dtypes.fp4x2)
    w = dict(
        w1=shuffle_weight_a16w4(w1q, 16, True),
        w2=shuffle_weight_a16w4(w2q, 16, False),
        w1_scale=shuffle_scale_a16w4(w1s, NE, True),
        w2_scale=shuffle_scale_a16w4(w2s, NE, False),
    )
    w["w1"].shuffle_kind = "mxfp4_moe"

    torch.manual_seed(seed + 1)
    hidden = torch.randn((M, H), dtype=dtypes.bf16, device=device) / 10
    g = torch.Generator(device=device).manual_seed(seed + 1)
    bias = torch.randn(NE - 1, generator=g, device=device) * 0.5
    scores = torch.randn(M, NE - 1, generator=g, device=device) + bias
    rw, rid = torch.topk(scores.softmax(-1), TOPK - 1, dim=-1)
    sid = torch.full((M, 1), NE - 1, device=device, dtype=rid.dtype)
    sw = torch.ones((M, 1), device=device, dtype=rw.dtype)
    topk_ids = torch.cat([sid, rid], dim=1).to(torch.int32)
    topk_weight = torch.cat([sw, rw], dim=1).to(torch.float32)
    return hidden, w, topk_ids, topk_weight


@_GFX950
@pytest.mark.parametrize("M", [64, 256])
def test_flydsl_gemm1_matches_hip_end_to_end(M):
    import aiter
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    device = torch.device("cuda")
    hidden, w, topk_ids, topk_weight = _build_kimi_mx(device, M)

    def run():
        return fused_moe(
            hidden, w["w1"], w["w2"], topk_weight, topk_ids,
            activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
            w1_scale=w["w1_scale"], w2_scale=w["w2_scale"],
        )

    out_hip = run()

    w["w1"].gemm1_backend = "flydsl"
    out_fly = run()

    cos = torch.nn.functional.cosine_similarity(
        out_hip.float().reshape(-1), out_fly.float().reshape(-1), dim=0
    ).item()
    assert cos > 0.99, f"M={M} cosine={cos:.5f}"

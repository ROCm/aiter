# SPDX-License-Identifier: MIT
import pytest
import torch


def test_port_module_imports_and_constants():
    """?????????,?????????? Kimi ???"""
    from aiter.ops.flydsl.kernels import mxfp4_gemm2 as port

    assert callable(port.compile_gemm2_a4w4_port)
    # gemm2 ?? K=512 (contraction = inter_dim),N_OUT = model_dim = 7168?
    assert (port.NE, port.K, port.N_OUT, port.TOPK) == (385, 512, 7168, 9)


def test_guard_rejects_bad_shape():
    """? Kimi/DSR ???? fail-loud?"""
    from aiter.ops.flydsl.mxfp4_gemm2_kernels import _assert_supported

    with pytest.raises(NotImplementedError, match="Kimi"):
        _assert_supported(
            NE=385,
            D_HIDDEN=7168,
            D_INTER=256,
            topk=9,
            BM=32,
            use_nt=False,
            atomic=True,
            mxfp4out=False,
        )


def test_guard_rejects_bad_variant():
    """?????????? fail-loud(atomic ??? BM16/32/64)?"""
    from aiter.ops.flydsl.mxfp4_gemm2_kernels import _assert_supported

    with pytest.raises(NotImplementedError, match="??|variant"):
        _assert_supported(
            NE=385,
            D_HIDDEN=7168,
            D_INTER=512,
            topk=9,
            BM=128,
            use_nt=False,
            atomic=True,
            mxfp4out=False,
        )


def test_guard_accepts_supported():
    """Kimi/DSR ?? + ????????(?????)?"""
    from aiter.ops.flydsl.mxfp4_gemm2_kernels import _assert_supported

    supported = [
        # atomic: BM?{16,32,64} x {ATOMIC, NT}
        (16, False, True, False),
        (16, True, True, False),
        (32, False, True, False),
        (32, True, True, False),
        (64, False, True, False),
        (64, True, True, False),
        # nonatomic bf16 flat (BM128)
        (128, False, False, False),
        # nonatomic mxfp4-out (BM128)
        (128, False, False, True),
    ]
    for NE in (257, 385):
        for BM, nt, atomic, mxfp4out in supported:
            _assert_supported(
                NE=NE,
                D_HIDDEN=7168,
                D_INTER=512,
                topk=9,
                BM=BM,
                use_nt=nt,
                atomic=atomic,
                mxfp4out=mxfp4out,
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
    not _is_gfx950(),
    reason="flydsl gemm2 ?? gfx950 (mfma_scale_f32_16x16x128_f8f6f4)",
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
def test_flydsl_gemm2_matches_hip_end_to_end(M):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    device = torch.device("cuda")
    hidden, w, topk_ids, topk_weight = _build_kimi_mx(device, M)

    def run():
        return fused_moe(
            hidden,
            w["w1"],
            w["w2"],
            topk_weight,
            topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=w["w1_scale"],
            w2_scale=w["w2_scale"],
        )

    out_hip = run()

    w["w2"].gemm2_backend = "flydsl"
    out_fly = run()

    cos = torch.nn.functional.cosine_similarity(
        out_hip.float().reshape(-1), out_fly.float().reshape(-1), dim=0
    ).item()
    assert cos > 0.99, f"M={M} cosine={cos:.5f}"

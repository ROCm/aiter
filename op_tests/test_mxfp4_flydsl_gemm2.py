# SPDX-License-Identifier: MIT
import pytest
import torch


def test_port_module_imports_and_constants():
    """?????????,?????????? Kimi ???"""
    from aiter.ops.flydsl.kernels import mxfp4_gemm2 as port

    assert callable(port.compile_gemm2_a4w4_port)
    # gemm2 ?? K=512 (contraction = inter_dim),N_OUT = model_dim = 7168?
    assert (port.NE, port.K, port.N_OUT) == (385, 512, 7168)
    # KIMI-default K-derived sizes via the *_for(512) helpers (byte-for-byte).
    assert port.K_HALF == port.k_half_for(512) == 256
    assert port.k_tiles_total_for(512) == 2
    assert port.kas_per_chunk_dw_for(512) == 128
    assert port.kbs_stride_n0_dw_for(512) == 128


def test_k_parametrized_helpers():
    """K(=inter_dim) parametrization: *_for(k) helpers + tile counts for a new K."""
    from aiter.ops.flydsl.kernels import mxfp4_gemm2 as port

    # 768 -> 3 K-tiles (streaming); 2048 -> 8 tiles.
    assert port.k_tiles_total_for(768) == 3
    assert port.k_tiles_total_for(2048) == 8
    assert port.kunroll_for(768) == 1  # K_TILES_TOTAL - kStages
    assert port.kunroll_for(2048) == 6
    assert port.k_half_for(768) == 384
    # B/scale byte sizes scale with K.
    assert port.bq_bytes_for(385, 7168, 768) == 385 * 7168 * 384
    assert port.aq_bytes_for(655360, 768) == 655360 * 384


def test_guard_rejects_bad_shape():
    """D_INTER ?? 256 ?? -> fail-loud (e.g. 384/192)?"""
    from aiter.ops.flydsl.mxfp4_gemm2_kernels import _assert_supported

    with pytest.raises(NotImplementedError, match="256"):
        _assert_supported(
            NE=385,
            D_HIDDEN=7168,
            D_INTER=384,
            topk=9,
            BM=32,
            use_nt=False,
            atomic=True,
            mxfp4out=False,
        )


def test_guard_rejects_bad_hidden():
    """D_HIDDEN(=model_dim/output) ? Kimi/DSR ???? fail-loud?"""
    from aiter.ops.flydsl.mxfp4_gemm2_kernels import _assert_supported

    with pytest.raises(NotImplementedError, match="Kimi"):
        _assert_supported(
            NE=385,
            D_HIDDEN=4096,
            D_INTER=512,
            topk=9,
            BM=32,
            use_nt=False,
            atomic=True,
            mxfp4out=False,
        )


def test_guard_accepts_new_inter():
    """??? inter_dim(768/2048, %256) ??? (D_INTER ??)?"""
    from aiter.ops.flydsl.mxfp4_gemm2_kernels import _assert_supported

    for D_INTER in (256, 512, 768, 1024, 2048):
        _assert_supported(
            NE=385,
            D_HIDDEN=7168,
            D_INTER=D_INTER,
            topk=9,
            BM=32,
            use_nt=False,
            atomic=True,
            mxfp4out=False,
        )


def test_guard_rejects_bad_variant():
    """?????????? fail-loud(atomic ??? BM16/32/64)?"""
    from aiter.ops.flydsl.mxfp4_gemm2_kernels import _assert_supported

    with pytest.raises(NotImplementedError, match="variant"):
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


@_GFX950
@pytest.mark.parametrize("D_INTER", [512, 768, 1024, 2048])
def test_flydsl_gemm2_parametrized_k_compiles(D_INTER):
    """K(=inter_dim) parametrization: gemm2 must COMPILE for the fast path
    (D_INTER=512, K_TILES_TOTAL=2) and the streaming path (>512). The full fused_moe
    end-to-end path can't drive a non-KIMI inter_dim, so compile coverage + the
    chained numeric test below are the achievable checks."""
    from aiter.ops.flydsl.kernels.mxfp4_gemm2 import compile_gemm2_a4w4_port

    assert D_INTER % 256 == 0
    for BM, nt in [(16, False), (32, False), (32, True), (64, False)]:
        launch = compile_gemm2_a4w4_port(
            BM=BM, use_nt=nt, epilog="atomic", D_INTER=D_INTER
        )
        assert callable(launch)
    # nonatomic (BM128) bf16 + mxfp4 flat epilogs also must compile for new K.
    for epilog in ("nonatomic", "nonatomic_mxfp4"):
        launch = compile_gemm2_a4w4_port(
            BM=128, use_nt=False, epilog=epilog, D_INTER=D_INTER
        )
        assert callable(launch)


@_GFX950
@pytest.mark.parametrize(
    "D_INTER",
    [
        512,  # KIMI fast path (sanity: chained gemm1->gemm2 known-good)
        768,  # new K -> streaming K-loop (3 tiles)
    ],
)
def test_flydsl_gemm2_parametrized_k_numeric(D_INTER):
    """Standalone numeric check of the parametrized gemm2 (BM32 atomic).

    Chains FlyDSL gemm1 -> FlyDSL gemm2 at the SAME (KIMI NE/H/TOPK, new INTER):
    gemm1 produces inter_sorted_quant + inter_sorted_shuffled_scale in exactly the
    layout gemm2 consumes (so no by-hand A-scale shuffle reconstruction needed). The
    reference dequantizes the gemm1 fp4 intermediate + w2, computes the per-token
    weighted topk-sum gemm2 (inter @ w2.T * sorted_weight), and compares to the
    atomic gemm2 output. The streaming K-loop (D_INTER>512) handles >2 K-tiles; the
    cosine ceiling reflects fp4 intermediate-quant error and matches the 512 (fast-
    path / known-good) score, so a comparable score at 768 validates the K stream."""
    import aiter
    from aiter import QuantType, dtypes
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1
    from aiter.ops.flydsl.mxfp4_gemm2_kernels import flydsl_mxfp4_gemm2
    from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32

    device = torch.device("cuda")
    NE, H, TOPK = 385, 7168, 9  # KIMI NE/H/TOPK; INTER (= gemm2 K) is parametrized.
    INTER = D_INTER
    BM, M, seed = 32, 256, 2
    tq = aiter.get_torch_quant(QuantType.per_1x32)

    # weights
    torch.manual_seed(seed)
    w1 = torch.randn((NE, 2 * INTER, H), dtype=dtypes.bf16, device=device) / 10
    w2 = torch.randn((NE, H, INTER), dtype=dtypes.bf16, device=device) / 10
    w1q, w1s = tq(w1, quant_dtype=dtypes.fp4x2)
    w2q, w2s = tq(w2, quant_dtype=dtypes.fp4x2)
    w1u8 = shuffle_weight_a16w4(w1q, 16, True)
    w1u8.shuffle_kind = "mxfp4_moe"
    w1_scale = shuffle_scale_a16w4(w1s, NE, True)
    w2u8 = shuffle_weight_a16w4(w2q, 16, False)
    w2_scale = shuffle_scale_a16w4(w2s, NE, False)

    def _u8(t):
        return (
            t.view(torch.uint8)
            if (t is not None and t.element_size() == 1 and t.dtype != torch.uint8)
            else t
        )

    w1u8v, w1sv = _u8(w1u8), _u8(w1_scale)
    w2u8v, w2sv = _u8(w2u8), _u8(w2_scale)

    # routing (KIMI: shared expert id NE-1 always, TOPK-1 routed)
    torch.manual_seed(seed + 1)
    hidden = torch.randn((M, H), dtype=dtypes.bf16, device=device) / 10
    g = torch.Generator(device=device).manual_seed(seed + 1)
    bias = torch.randn(NE - 1, generator=g, device=device) * 0.5
    scores = torch.randn(M, NE - 1, generator=g, device=device) + bias
    rw, rid = torch.topk(scores.softmax(-1), TOPK - 1, dim=-1)
    sid = torch.full((M, 1), NE - 1, device=device, dtype=rid.dtype)
    sw = torch.ones((M, 1), device=device, dtype=rw.dtype)
    topk_ids = torch.cat([sid, rid], 1).to(torch.int32)
    topk_weight = torch.cat([sw, rw], 1).to(torch.float32)

    # sort (KIMI shape -> HIP threestage sort)
    active = min(NE, M * TOPK)
    max_sorted = ((M * TOPK + active * (BM - 1) + BM - 1) // BM) * BM
    eb = lambda: torch.empty((0,), device=device, dtype=dtypes.bf16)
    sti = torch.empty((max_sorted,), device=device, dtype=dtypes.i32)
    sei = torch.empty((max_sorted // BM,), device=device, dtype=dtypes.i32)
    cumsum = torch.empty((1,), device=device, dtype=dtypes.i32)
    rev = torch.empty((M * TOPK,), device=device, dtype=dtypes.i32)
    swt = torch.empty((max_sorted,), device=device, dtype=dtypes.fp32)
    mm = torch.empty((NE,), device=device, dtype=dtypes.i32)
    mind = torch.empty((max_sorted,), device=device, dtype=dtypes.i32)
    aiter.mxfp4_moe_sort(
        topk_ids=topk_ids,
        topk_weight=topk_weight,
        sorted_token_ids=sti,
        sorted_expert_ids=sei,
        cumsum_tensor=cumsum,
        reverse_sorted=rev,
        sorted_weights=swt,
        masked_m=mm,
        m_indices=mind,
        bf16_zero_out=eb(),
        bf16_zero_workspace=eb(),
        M_logical=M,
        NE=NE,
        TOPK=TOPK,
        D_HIDDEN=H,
        D_INTER=D_INTER,
        MB=BM,
        prologue=1,
    )
    torch.cuda.synchronize()
    n = int(cumsum[0].item())

    # A-quant + a_scale_sorted_shuffled (reuse the validated gemm1-test machinery)
    from op_tests.test_mxfp4_flydsl_gemm1 import _torch_a_scale_sorted_shuffled

    aq, asc = tq(hidden, quant_dtype=dtypes.fp4x2)
    aq = aq.view(torch.uint8).view(M, H // 2).contiguous()
    asc = asc.view(torch.uint8).view(M, H // 32).contiguous()
    assh = _torch_a_scale_sorted_shuffled(asc, sti, cumsum, max_sorted, H, BM=BM)

    # gemm1 -> inter (fp4) + shuffled scale (== gemm2's A inputs)
    isq = torch.zeros((max_sorted, D_INTER // 2), device=device, dtype=torch.uint8)
    isc_cols = D_INTER // 32
    isr = (
        (((max_sorted * ((2 * INTER) // 64) * 4) + isc_cols - 1) // isc_cols + 31)
        // 32
        * 32
    )
    iss = torch.zeros((isr, isc_cols), device=device, dtype=torch.uint8)
    flydsl_mxfp4_gemm1(
        a_quant=aq,
        a_scale_sorted_shuffled=assh,
        w1_u8=w1u8v,
        w1_scale_u8=w1sv,
        sorted_expert_ids=sei,
        cumsum_tensor=cumsum,
        m_indices=mind,
        inter_sorted_quant=isq,
        inter_sorted_shuffled_scale=iss,
        hidden_states=hidden,
        n_tokens=M,
        BM=BM,
        use_nt=True,
        inline_quant=False,
        NE=NE,
        D_HIDDEN=H,
        D_INTER=D_INTER,
        topk=TOPK,
    )
    torch.cuda.synchronize()

    # gemm2 (BM32 atomic): inter x w2 -> per-token weighted topk-sum (flat_out)
    flat_out = torch.zeros((M, H), dtype=dtypes.bf16, device=device)
    flydsl_mxfp4_gemm2(
        inter_sorted_quant=isq,
        inter_sorted_shuffled_scale=iss,
        w2_u8=w2u8v,
        w2_scale_u8=w2sv,
        sorted_expert_ids=sei,
        cumsum_tensor=cumsum,
        sorted_token_ids=sti,
        sorted_weights=swt,
        flat_out=flat_out,
        M_logical=M,
        max_sorted=max_sorted,
        BM=BM,
        use_nt=False,
        atomic=True,
        mxfp4out=False,
        NE=NE,
        D_HIDDEN=H,
        D_INTER=D_INTER,
        topk=TOPK,
    )
    torch.cuda.synchronize()

    # reference: an INDEPENDENT full-precision-ish MoE (dequant A/w1 -> silu_mul ->
    # @ dequant w2, weighted per-token topk sum). The cosine vs the kernel output is
    # bounded by the fp4 intermediate-quant the kernel applies between gemm1 and
    # gemm2 (the same lossy step KIMI/512 incurs), which is why the 512 fast path and
    # the 768 streaming path land at a comparable ceiling.
    W2 = mxfp4_to_f32(w2q.view(torch.uint8))
    W2s = e8m0_to_f32(w2s.view(torch.uint8)).view(NE, H, INTER // 32)
    W2 = (W2.view(NE, H, INTER // 32, 32) * W2s.unsqueeze(-1)).view(NE, H, INTER)

    # Independent intermediate reference: A_deq @ w1 -> silu_mul (full-precision-ish).
    A = mxfp4_to_f32(aq.view(torch.uint8))
    Asc = e8m0_to_f32(asc.view(torch.uint8))
    A = (A.view(M, H // 32, 32) * Asc.unsqueeze(-1)).view(M, H)
    W1 = mxfp4_to_f32(w1q.view(torch.uint8))
    W1s = e8m0_to_f32(w1s.view(torch.uint8)).view(NE, 2 * INTER, H // 32)
    W1 = (W1.view(NE, 2 * INTER, H // 32, 32) * W1s.unsqueeze(-1)).view(
        NE, 2 * INTER, H
    )
    import torch.nn.functional as Fnn

    mind_c, sei_c, swt_c = mind[:n].cpu(), sei.cpu(), swt[:n].cpu()
    ref = torch.zeros((M, H), dtype=torch.float32, device=device)
    for r in range(n):
        tok = int(mind_c[r].item())
        if tok >= M:
            continue
        e = int(sei_c[r // BM].item())
        gate = A[tok] @ W1[e, :INTER].T
        up = A[tok] @ W1[e, INTER : 2 * INTER].T
        inter_r = Fnn.silu(gate) * up  # (INTER,)
        out_r = (inter_r @ W2[e].T) * float(swt_c[r].item())  # (H,)
        ref[tok] += out_r

    cos = Fnn.cosine_similarity(
        ref.reshape(-1), flat_out.float().reshape(-1), dim=0
    ).item()
    print(f"[gemm2 numeric] INTER={INTER} K_TILES={INTER // 256} cos={cos:.4f} (n={n})")
    # fp4 intermediate-quant ceiling; KIMI(512) fast path scores the same, so a
    # comparable score at the streaming (768) shape validates the K-loop.
    assert cos > 0.90, f"INTER={INTER} cos={cos:.4f} (n={n})"

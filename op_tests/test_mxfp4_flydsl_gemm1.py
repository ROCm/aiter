# SPDX-License-Identifier: MIT
import pytest
import torch


def test_port_module_imports_and_constants():
    """Port module imports cleanly and exposes the compile fn / grid + the Kimi constants."""
    from aiter.ops.flydsl.kernels import mxfp4_gemm1 as port

    assert callable(port.compile_gemm1_a4w4_port)
    assert callable(port.gemm1_grid)
    assert (port.NE, port.K, port.INTER, port.TOPK) == (385, 7168, 512, 9)
    assert port.N_OUT == 1024


def test_guard_accepts_non_kimi_ne_inter_topk():
    """OUTPUT side (NE/INTER/TOPK) is now parametrized: non-KIMI but
    divisibility-legal shapes (e.g. minimax NE=256/INTER=768/TOPK=8) are accepted."""
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import _assert_supported

    # minimax: NE=256, H=3072, INTER=768, TOPK=8 (2*768=1536 % 256 == 0).
    _assert_supported(
        NE=256,
        D_HIDDEN=3072,
        D_INTER=768,
        topk=8,
        BM=32,
        use_nt=True,
        inline_quant=False,
    )
    # arbitrary NE / TOPK are fine as long as divisibility holds.
    _assert_supported(
        NE=257,
        D_HIDDEN=7168,
        D_INTER=512,
        topk=5,
        BM=32,
        use_nt=True,
        inline_quant=False,
    )


def test_guard_rejects_bad_inter():
    """2*D_INTER (= N_OUT) must be a multiple of 256 (i.e. D_INTER % 128 == 0)."""
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import _assert_supported

    with pytest.raises(NotImplementedError, match="N_OUT|D_INTER"):
        _assert_supported(
            NE=385,
            D_HIDDEN=7168,
            D_INTER=500,  # 2*500=1000, not a multiple of 256
            topk=9,
            BM=32,
            use_nt=True,
            inline_quant=False,
        )


def test_guard_rejects_non_256_multiple_k():
    """K (= D_HIDDEN) must be a multiple of 256; otherwise the guard rejects it."""
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import _assert_supported

    with pytest.raises(NotImplementedError, match="256"):
        _assert_supported(
            NE=385,
            D_HIDDEN=7000,
            D_INTER=512,
            topk=9,
            BM=32,
            use_nt=True,
            inline_quant=False,
        )


def test_guard_accepts_parametrized_k():
    """Non-7168 K values that are multiples of 256 are accepted (parametrized K)."""
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import _assert_supported

    for H in (3072, 4096, 7168):
        _assert_supported(
            NE=385,
            D_HIDDEN=H,
            D_INTER=512,
            topk=9,
            BM=32,
            use_nt=True,
            inline_quant=False,
        )


def test_guard_rejects_bad_variant():
    """An unsupported variant fails loud."""
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import _assert_supported

    with pytest.raises(NotImplementedError, match="variant"):
        _assert_supported(
            NE=385,
            D_HIDDEN=7168,
            D_INTER=512,
            topk=9,
            BM=64,
            use_nt=True,
            inline_quant=False,
        )


def test_guard_accepts_supported():
    """Kimi shape + supported (BM, nt, inline_quant) combos pass the guard."""
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import _assert_supported

    for BM, nt, iq in [
        (32, True, False),
        (32, False, False),
        (128, False, False),
        (16, True, True),
    ]:
        _assert_supported(
            NE=385,
            D_HIDDEN=7168,
            D_INTER=512,
            topk=9,
            BM=BM,
            use_nt=nt,
            inline_quant=iq,
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
    reason="flydsl gemm1 requires gfx950 (mfma_scale_f32_16x16x128_f8f6f4)",
)


def _build_kimi_mx(device, M, seed=2, H=7168):
    import aiter
    from aiter import QuantType, dtypes
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    # NE/INTER/TOPK stay at the KIMI values; H (= D_HIDDEN, the gemm1 contraction
    # dim K) is parametrized to exercise the K-parametrized FlyDSL gemm1 port.
    NE, INTER, TOPK = 385, 512, 9
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

    w["w1"].gemm1_backend = "flydsl"
    out_fly = run()

    cos = torch.nn.functional.cosine_similarity(
        out_hip.float().reshape(-1), out_fly.float().reshape(-1), dim=0
    ).item()
    assert cos > 0.99, f"M={M} cosine={cos:.5f}"


@_GFX950
@pytest.mark.parametrize("H", [3072, 4096])
def test_flydsl_gemm1_parametrized_k_compiles(H):
    """K-parametrization: every supported variant must COMPILE for a non-7168 H.

    The full fused_moe end-to-end path can't drive a non-KIMI H (the tuned CSV has
    no mxfp4 pipeline row for model_dim!=7168, so fused_moe rejects the weight before
    gemm1), and the HIP gemm1/quant/sort_scales reference kernels are hard-templated
    to H=7168 -- so end-to-end / HIP-reference comparison is impossible for new H.
    Compile coverage + the standalone numeric test below are the achievable checks."""
    from aiter.ops.flydsl.kernels.mxfp4_gemm1 import compile_gemm1_a4w4_port

    assert H % 256 == 0
    for BM, nt, iq in [
        (32, True, False),
        (32, False, False),
        (128, False, False),
        (16, True, True),
    ]:
        launch = compile_gemm1_a4w4_port(BM=BM, use_nt=nt, inline_quant=iq, D_HIDDEN=H)
        assert callable(launch)


@_GFX950
@pytest.mark.parametrize(
    "NE,H,INTER,TOPK",
    [
        (385, 7168, 512, 9),  # KIMI (sanity: still compiles all 4 variants)
        (256, 3072, 768, 8),  # minimax: non-KIMI NE/INTER/TOPK + parametrized K
    ],
)
def test_flydsl_gemm1_parametrized_shape_compiles(NE, H, INTER, TOPK):
    """INTER/NE/TOPK-parametrization: every supported variant must COMPILE for a
    non-KIMI OUTPUT shape (minimax NE=256/H=3072/INTER=768/TOPK=8)."""
    from aiter.ops.flydsl.kernels.mxfp4_gemm1 import compile_gemm1_a4w4_port

    assert (2 * INTER) % 256 == 0 and H % 256 == 0
    for BM, nt, iq in [
        (32, True, False),
        (32, False, False),
        (128, False, False),
        (16, True, True),
    ]:
        launch = compile_gemm1_a4w4_port(
            BM=BM,
            use_nt=nt,
            inline_quant=iq,
            D_HIDDEN=H,
            D_INTER=INTER,
            NE=NE,
            TOPK=TOPK,
        )
        assert callable(launch)


@_GFX950
@pytest.mark.parametrize(
    "NE,H,INTER,TOPK",
    [
        (385, 7168, 512, 9),  # KIMI
        (256, 3072, 768, 8),  # minimax
    ],
)
def test_flydsl_gemm1_separated_compiles(NE, H, INTER, TOPK):
    from aiter.ops.flydsl.kernels.mxfp4_gemm1 import compile_gemm1_a4w4_port

    assert (2 * INTER) % 256 == 0 and H % 256 == 0
    for BM, nt, iq in [
        (32, True, False),
        (32, False, False),
        (128, False, False),
        (16, True, True),
    ]:
        launch = compile_gemm1_a4w4_port(
            BM=BM,
            use_nt=nt,
            inline_quant=iq,
            D_HIDDEN=H,
            D_INTER=INTER,
            NE=NE,
            TOPK=TOPK,
            interleave=False,
        )
        assert callable(launch)


def _torch_threestage_sort(topk_ids, topk_weight, M, NE, TOPK, BM, max_sorted):
    """Torch replica of moe_3stage_sort.cuh (stable expert-grouped sort, each
    expert region padded up to a multiple of BM). The HIP threestage sort is
    hard-templated to KIMI (NE=385/TOPK=9) only, so non-KIMI shapes need this to
    build the gemm1 inputs. Validated against the HIP sort at the KIMI shape (same
    sti/sei/cumsum/m_indices). Returns (sti, sei, cumsum, mind)."""
    device = topk_ids.device
    flat_e = topk_ids.reshape(-1)  # (M*TOPK,) expert per (token,topk) pair, row-major
    counts = torch.bincount(flat_e, minlength=NE)  # tokens per expert
    padded = ((counts + BM - 1) // BM) * BM
    starts = torch.zeros(NE + 1, dtype=torch.int64, device=device)
    starts[1:] = torch.cumsum(padded, 0)
    total = int(starts[NE].item())
    assert total <= max_sorted, f"total {total} > max_sorted {max_sorted}"

    sti = torch.full((max_sorted,), M & 0x00FFFFFF, dtype=torch.int32, device=device)
    mind = torch.full((max_sorted,), M & 0x00FFFFFF, dtype=torch.int32, device=device)
    cumsum = torch.tensor([total], dtype=torch.int32, device=device)
    sei = torch.zeros(max_sorted // BM, dtype=torch.int32, device=device)

    # sorted_expert_ids per BM-block (mirror sort_cumsum: blocks in [start/BM, end/BM)).
    for e in range(NE):
        b0 = int(starts[e].item()) // BM
        b1 = int(starts[e + 1].item()) // BM
        sei[b0:b1] = e

    # stable placement: iterate pairs in linear order, append into each expert's run.
    fill = starts[:NE].clone()  # next write position per expert
    flat_e_c = flat_e.cpu().tolist()
    fill_c = fill.cpu().tolist()
    sti_c = sti.cpu()
    mind_c = mind.cpu()
    for i, eid in enumerate(flat_e_c):
        sp = fill_c[eid]
        fill_c[eid] = sp + 1
        token_id = i // TOPK
        topk_id = i % TOPK
        sti_c[sp] = (token_id & 0x00FFFFFF) | ((topk_id & 0xFF) << 24)
        mind_c[sp] = token_id & 0x00FFFFFF
    sti = sti_c.to(device)
    mind = mind_c.to(device)
    return sti, sei, cumsum, mind


def _torch_a_scale_sorted_shuffled(asc, sti, cumsum, max_sorted, H, BM=32, BK=256):
    """Reconstruct a_scale_sorted_shuffled exactly as moe_sort_scales.cuh does, in
    torch (the HIP kernel is H-locked to 7168). Validated bit-for-bit at H=7168:
    feeding this reconstruction to gemm1 reproduces the HIP-prologue result."""
    device = asc.device
    H // 32
    MN_PACK = 2
    K_PACK = BK // 128
    C_M1 = BM // (16 * MN_PACK)
    C_K1 = (H // 32) // (4 * K_PACK)
    K_LANE, N_LANE = 4, 16
    DWORDS_PER_CHUNK = C_M1 * C_K1 * K_LANE * N_LANE
    n_chunks = max_sorted // BM
    actual_sorted = int(cumsum[0].item())
    actual_n_chunks = (actual_sorted + BM - 1) // BM
    total_work = n_chunks * DWORDS_PER_CHUNK
    sti_c = sti & 0x00FFFFFF
    out = torch.zeros((total_work, 4), dtype=torch.uint8, device=device)
    wid = torch.arange(total_work, device=device)
    r = wid.clone()
    n_lane = r % N_LANE
    r //= N_LANE
    k_lane = r % K_LANE
    r //= K_LANE
    ku = r % C_K1
    r //= C_K1
    mi = r % C_M1
    r //= C_M1
    chunk = r
    valid_chunk = chunk < actual_n_chunks
    M = asc.shape[0]
    for ikxdl in range(K_PACK):
        for im_a in range(MN_PACK):
            sorted_row = chunk * BM + (mi * MN_PACK + im_a) * 16 + n_lane
            rowok = (sorted_row < actual_sorted) & valid_chunk
            srow = torch.clamp(sorted_row, max=max_sorted - 1)
            stiv = sti_c[srow]
            tid = torch.where((stiv < M) & rowok, stiv, torch.zeros_like(stiv))
            k_idx = ku * K_PACK * 4 + ikxdl * 4 + k_lane
            byte = asc[tid.long(), k_idx.long()]
            out[:, ikxdl * MN_PACK + im_a] = torch.where(
                rowok, byte, torch.zeros_like(byte)
            )
    return out.reshape(-1).contiguous()


@_GFX950
@pytest.mark.parametrize(
    "interleave", [True, False], ids=["interleave", "separated"]
)
@pytest.mark.parametrize(
    "NE,H,INTER,TOPK",
    [
        (385, 7168, 512, 9),  # KIMI (known-good; ~0.88 fp4-output-quant ceiling)
        (385, 3072, 512, 9),  # parametrized K only
        (385, 4096, 512, 9),  # parametrized K only
        (256, 3072, 768, 8),  # minimax: parametrized INTER/NE/TOPK + K
    ],
)
def test_flydsl_gemm1_parametrized_shape_numeric(NE, H, INTER, TOPK, interleave):
    """Standalone numeric check of the parametrized gemm1 (BM32 non-inline).

    Uses the (shape-independent) threestage sort + torch per_1x32 A-quant + a torch
    reconstruction of a_scale_sorted_shuffled (the HIP quant/sort_scales kernels are
    H-locked). Compares each sorted output row (dequantized fp4) against a torch
    silu_mul reference on the SAME dequantized A/w1. The ~0.88 mean-row-cosine
    ceiling is the per-row fp4 OUTPUT-quant error, not a kernel defect: KIMI
    (the known-good shape) scores the same, so a comparable score at a new
    (NE,H,INTER,TOPK) validates the parametrized N/contraction handling."""
    import aiter
    from aiter import QuantType, dtypes
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1
    from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32
    import torch.nn.functional as Fnn

    device = torch.device("cuda")
    BM, M, seed = 32, 256, 2
    D_INTER, topk = INTER, TOPK
    tq = aiter.get_torch_quant(QuantType.per_1x32)

    torch.manual_seed(seed)
    w1 = torch.randn((NE, 2 * INTER, H), dtype=dtypes.bf16, device=device) / 10
    w1q, w1s = tq(w1, quant_dtype=dtypes.fp4x2)
    w1u8 = shuffle_weight_a16w4(w1q, 16, interleave)
    w1_scale = shuffle_scale_a16w4(w1s, NE, interleave)
    if w1u8.element_size() == 1 and w1u8.dtype != torch.uint8:
        w1u8 = w1u8.view(torch.uint8)
    w1_scale_u8 = (
        w1_scale.view(torch.uint8)
        if (
            w1_scale is not None
            and w1_scale.element_size() == 1
            and w1_scale.dtype != torch.uint8
        )
        else w1_scale
    )

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

    active = min(NE, M * topk)
    max_sorted = ((M * topk + active * (BM - 1) + BM - 1) // BM) * BM
    # The HIP threestage sort is hard-templated to KIMI (NE=385/TOPK=9). For other
    # shapes use the torch replica (validated vs HIP at the KIMI shape: identical
    # sti/sei/cumsum/m_indices). Both produce the exact layout gemm1 consumes.
    if (NE, topk) == (385, 9):

        def eb():
            return torch.empty((0,), device=device, dtype=dtypes.bf16)

        sti = torch.empty((max_sorted,), device=device, dtype=dtypes.i32)
        sei = torch.empty((max_sorted // BM,), device=device, dtype=dtypes.i32)
        cumsum = torch.empty((2,), device=device, dtype=dtypes.i32)
        rev = torch.empty((M * topk,), device=device, dtype=dtypes.i32)
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
            TOPK=topk,
            D_HIDDEN=H,
            D_INTER=D_INTER,
            MB=BM,
            prologue=1,
        )
        torch.cuda.synchronize()
    else:
        sti, sei, cumsum, mind = _torch_threestage_sort(
            topk_ids, topk_weight, M, NE, topk, BM, max_sorted
        )
    n = int(cumsum[0].item())

    aq, asc = tq(hidden, quant_dtype=dtypes.fp4x2)
    aq = aq.view(torch.uint8).view(M, H // 2).contiguous()
    asc = asc.view(torch.uint8).view(M, H // 32).contiguous()
    assh = _torch_a_scale_sorted_shuffled(asc, sti, cumsum, max_sorted, H, BM=BM)

    isq = torch.zeros((max_sorted, D_INTER // 2), device=device, dtype=torch.uint8)
    isc = D_INTER // 32
    N_OUT = 2 * INTER
    isr = (((max_sorted * (N_OUT // 64) * 4) + isc - 1) // isc + 31) // 32 * 32
    iss = torch.zeros((isr, isc), device=device, dtype=torch.uint8)
    flydsl_mxfp4_gemm1(
        a_quant=aq,
        a_scale_sorted_shuffled=assh,
        w1_u8=w1u8,
        w1_scale_u8=w1_scale_u8,
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
        topk=topk,
        interleave=interleave,
    )
    torch.cuda.synchronize()

    A = mxfp4_to_f32(aq.view(torch.uint8))
    Asc = e8m0_to_f32(asc.view(torch.uint8))
    A = (A.view(M, H // 32, 32) * Asc.unsqueeze(-1)).view(M, H)
    W = mxfp4_to_f32(w1q.view(torch.uint8))
    Ws = e8m0_to_f32(w1s.view(torch.uint8)).view(NE, 2 * INTER, H // 32)
    W = (W.view(NE, 2 * INTER, H // 32, 32) * Ws.unsqueeze(-1)).view(NE, 2 * INTER, H)
    fly_q = mxfp4_to_f32(isq[:n].view(torch.uint8))

    mind_c, sei_c = mind[:n].cpu(), sei.cpu()
    cossum, cnt = 0.0, 0
    for r in range(n):
        tok = int(mind_c[r].item())
        if tok >= M:
            continue
        e = int(sei_c[r // BM].item())
        gate = A[tok] @ W[e, :INTER].T
        up = A[tok] @ W[e, INTER : 2 * INTER].T
        ref = Fnn.silu(gate) * up
        cossum += Fnn.cosine_similarity(
            ref.reshape(-1), fly_q[r].float().reshape(-1), dim=0
        ).item()
        cnt += 1
    mean_cos = cossum / cnt
    mode = "interleave" if interleave else "separated"
    print(
        f"[gemm1 numeric] NE={NE} H={H} INTER={INTER} TOPK={TOPK} mode={mode} "
        f"mean_row_cos={mean_cos:.4f} (cnt={cnt})"
    )
    # 0.85 floor: the fp4 output-quant ceiling (~0.88) holds for KIMI too.
    assert mean_cos > 0.85, (
        f"NE={NE} H={H} INTER={INTER} TOPK={TOPK} mode={mode} "
        f"mean_row_cos={mean_cos:.4f} (cnt={cnt})"
    )

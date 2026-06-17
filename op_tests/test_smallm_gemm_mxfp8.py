# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for the decode small-M MX-FP8 GEMM ops (gfx950).

Each op (``mxfp8_gemv`` dense, ``smallm_mxfp8_moe_grouped_gemm`` MoE via the
``grouped_gemm_mxfp8`` wrapper) is compared against a PyTorch reference that
dequantizes the SAME e4m3/e8m0 inputs the kernel reads (so the only difference
is fp8 matrix-core accumulation vs bf16 matmul -> ~28 dB / cosine ~0.999).

gfx950-only: the kernels use mfma_scale_f32_16x16x128_f8f6f4 and a host
gfx950 guard; on other archs they raise, so the module is skipped there.
"""

import pytest
import torch

from aiter.ops.smallm_gemm_mxfp8 import grouped_gemm_mxfp8, mxfp8_gemv

DEVICE = "cuda"
FP8_MAX = 448.0  # e4m3 max magnitude


def _gcn_arch() -> str:
    try:
        return torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return ""


requires_gfx950 = pytest.mark.skipif(
    _gcn_arch().split(":")[0] != "gfx950",
    reason="decode small-M MX-FP8 GEMMs require a gfx950 (CDNA4) device.",
)


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / (b.norm() + 1e-8)).item()


# ── self-contained MX-FP8 (e4m3 data + e8m0 1x32 scales) quant / dequant ──────
def quant_mxfp8(x: torch.Tensor):
    """x [..., K] -> (q e4m3 [..., K], e8m0 uint8 [..., K//32])."""
    K = x.shape[-1]
    xb = x.float().reshape(*x.shape[:-1], K // 32, 32)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-20)
    exp = torch.ceil(torch.log2(amax / FP8_MAX)).clamp(-127, 127)
    scale = torch.exp2(exp)
    q = (xb / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    e8m0 = (exp + 127).to(torch.uint8)
    return (
        q.reshape(*x.shape[:-1], K).contiguous(),
        e8m0.squeeze(-1).reshape(*x.shape[:-1], K // 32).contiguous(),
    )


def dequant_mxfp8(q: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    K = q.shape[-1]
    qb = q.float().reshape(*q.shape[:-1], K // 32, 32)
    exp = e8m0.reshape(*e8m0.shape, 1).float() - 127.0
    return (qb * torch.exp2(exp)).reshape(*q.shape[:-1], K)


def moe_align(topk_ids: torch.Tensor, block_m: int, E: int):
    """Reference moe_align_block_size: per-expert token slots padded to block_m,
    pad slots marked with the sentinel ``M`` (== num_valid_tokens)."""
    flat = topk_ids.reshape(-1).to(torch.int32)
    M = flat.numel()
    sorted_ids, expert_ids = [], []
    for e in range(E):
        idx = (flat == e).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n == 0:
            continue
        npad = ((n + block_m - 1) // block_m) * block_m
        blk = torch.full((npad,), M, dtype=torch.int32, device=topk_ids.device)
        blk[:n] = idx.to(torch.int32)
        sorted_ids.append(blk)
        expert_ids.append(
            torch.full((npad // block_m,), e, dtype=torch.int32, device=topk_ids.device)
        )
    sorted_ids = torch.cat(sorted_ids)
    expert_ids = torch.cat(expert_ids)
    num_post = torch.tensor(
        [sorted_ids.numel()], dtype=torch.int32, device=topk_ids.device
    )
    return sorted_ids, expert_ids, num_post


# ── dense GEMV (+ MFMA crossover) ─────────────────────────────────────────────
# (K, N) are M3's qkv / o_proj / gate_up / down per-GPU shapes at TP1/2/4/8.
@requires_gfx950
@pytest.mark.parametrize(
    "K,N",
    [
        (6144, 2304),
        (2048, 6144),
        (6144, 1536),
        (1536, 6144),  # TP4 per-GPU
        (6144, 1152),
        (6144, 768),
        (1024, 6144),
        (768, 6144),  # TP8 per-GPU
        (6144, 9216),
        (6144, 6144),
        (8192, 6144),  # TP1 per-GPU
        (6144, 4608),
        (6144, 3072),
        (4096, 6144),
        (3072, 6144),  # TP2 per-GPU
    ],
)
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16, 32, 64])
@torch.inference_mode()
def test_mxfp8_gemv(K, N, M):
    from aiter.ops.smallm_gemm_mxfp8 import _tuned_cfg

    torch.manual_seed(0)
    x = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16) * 0.5
    w = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    xq, xs = quant_mxfp8(x)
    wq, ws = quant_mxfp8(w)

    got = mxfp8_gemv(xq, xs, wq, ws, torch.bfloat16)
    tuned = _tuned_cfg().get((K, N, M))  # (kernel, n_sub, k_splits, use_hip)
    if tuned is not None and not tuned[3]:
        # CSV explicitly routes this cell to Triton (use_hip=0): None is correct.
        assert got is None
        pytest.skip(f"({M},{K},{N}) use_hip=0 -> Triton")
    # Every tuned use_hip=1 cell MUST engage; a None here is a real regression,
    # not a fall-through to skip (which would mask HIP-path failures).
    assert got is not None, f"({M},{K},{N}) is use_hip=1 but mxfp8_gemv returned None"
    # Reference consumes the SAME quantized bits the kernel reads.
    ref = torch.nn.functional.linear(dequant_mxfp8(xq, xs), dequant_mxfp8(wq, ws))
    assert got.shape == (M, N)
    assert _relerr(got, ref) < 5e-2


@requires_gfx950
@torch.inference_mode()
def test_mxfp8_gemv_out_of_envelope_returns_none():
    # M outside the kernel envelope (>64, no GEMV/MFMA template) must fall back.
    M, K, N = 128, 4096, 4096
    x = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)
    xq, xs = quant_mxfp8(x)
    wq, ws = quant_mxfp8(w)
    assert mxfp8_gemv(xq, xs, wq, ws, torch.bfloat16) is None
    # A non-bf16 out is also unsupported.
    assert mxfp8_gemv(xq, xs, wq, ws, torch.float16) is None


@requires_gfx950
@torch.inference_mode()
def test_mxfp8_gemv_untuned_large_m_falls_back():
    # On an UNtuned shape, large M (32/64) can lose to Triton, so default-engage
    # is capped at M<=16: untuned M=64 must fall back (return None) rather than
    # risk a regression. (Tuned TP4/TP8 M=64 still engage via the CSV.)
    K, N = 4096, 4096  # not in CSV / _MFMA_CFG
    x = torch.randn(64, K, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)
    xq, xs = quant_mxfp8(x)
    wq, ws = quant_mxfp8(w)
    assert mxfp8_gemv(xq, xs, wq, ws, torch.bfloat16) is None


@requires_gfx950
@pytest.mark.parametrize("M", [1, 4, 16])
@torch.inference_mode()
def test_mxfp8_gemv_untuned_shape_engages(M):
    # Envelope dispatch: an in-envelope (K, N) with NO tuned/hand-table entry
    # (e.g. a TP2-ish 4096x4096) must still engage HIP and be correct in the
    # small-M regime -- this is the generalization the allowlist used to block.
    K, N = 4096, 4096
    x = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16) * 0.5
    w = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    xq, xs = quant_mxfp8(x)
    wq, ws = quant_mxfp8(w)
    got = mxfp8_gemv(xq, xs, wq, ws, torch.bfloat16)
    assert (
        got is not None
    ), "in-envelope shape should engage HIP under envelope dispatch"
    ref = torch.nn.functional.linear(dequant_mxfp8(xq, xs), dequant_mxfp8(wq, ws))
    assert got.shape == (M, N)
    assert _relerr(got, ref) < 5e-2


# ── MoE grouped GEMM ──────────────────────────────────────────────────────────
def _ref_grouped(a_deq, w_deq, sorted_ids, expert_ids, num_valid, a_div, block_m, wt):
    """Per-slot reference: out[slot] = a_deq[slot // a_div] @ w_deq[expert].T."""
    Np = sorted_ids.shape[0]
    N = w_deq.shape[1]
    out = torch.zeros(num_valid, N, device=a_deq.device, dtype=torch.float32)
    for blk in range(Np // block_m):
        e = int(expert_ids[blk])
        for s in range(blk * block_m, (blk + 1) * block_m):
            tok = int(sorted_ids[s])
            if tok >= num_valid:
                continue
            row = a_deq[tok // a_div]
            o = row.float() @ w_deq[e].float().T
            if wt is not None:
                o = o * float(wt[tok])
            out[tok] = o
    return out


@requires_gfx950
@pytest.mark.parametrize(
    "E,N,K,a_div,has_w,T,top_k",
    [
        # M3 = 256 experts, so E/GPU = 128/64/32/16 at EP2/4/8/16. allowlist is
        # E-agnostic for engagement, E-aware for the bound (gemm1 widens at E>=128).
        (128, 1536, 6144, 4, False, 4, 4),  # gemm1 @EP2 (E=128): M_routed=16
        (128, 6144, 768, 1, True, 8, 1),  # gemm2 @EP2 weighted: M_routed=8
        (128, 6144, 768, 1, False, 8, 1),  # gemm2 @EP2 no-combine
        (32, 1536, 6144, 4, False, 4, 4),  # gemm1 @EP8 (E=32)
        (64, 6144, 768, 1, True, 8, 1),  # gemm2 @EP4 (E=64)
        (128, 1536, 6144, 4, False, 16, 4),  # gemm1 @EP2: M_routed=64 (wide envelope)
    ],
)
@torch.inference_mode()
def test_grouped_gemm_mxfp8(E, N, K, a_div, has_w, T, top_k):
    torch.manual_seed(0)
    block_m = 64
    M = T * top_k  # num_valid_tokens
    M_act = T if a_div == top_k else M  # gemm1 reads per-token; gemm2 per-slot

    a = torch.randn(M_act, K, device=DEVICE, dtype=torch.bfloat16) * 0.5
    w = torch.randn(E, N, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    aq, asc = quant_mxfp8(a)
    wq, wsc = quant_mxfp8(w)

    topk_ids = torch.randint(0, E, (T, top_k), device=DEVICE, dtype=torch.int32)
    sorted_ids, expert_ids, num_post = moe_align(topk_ids, block_m, E)
    wt = None
    if has_w:
        wt = torch.rand(M, device=DEVICE, dtype=torch.float32)

    got = grouped_gemm_mxfp8(
        aq,
        asc,
        wq,
        wsc,
        sorted_ids,
        expert_ids,
        num_post,
        M,
        top_k,
        block_m,
        torch.bfloat16,
        a_div,
        mul_weight_by=wt,
        topk_ids=topk_ids,
    )
    assert got is not None, "shape should be in the HIP MoE envelope"
    ref = _ref_grouped(
        dequant_mxfp8(aq, asc),
        dequant_mxfp8(wq, wsc),
        sorted_ids,
        expert_ids,
        M,
        a_div,
        block_m,
        wt,
    )
    assert got.shape == (M, N)
    assert _relerr(got, ref) < 5e-2


@requires_gfx950
@torch.inference_mode()
def test_moe_large_E_exceeds_2gb_falls_back():
    # The raw-buffer voffset is a signed int32, so weight byte size E*N*K must
    # stay < 2GB. no-EP gemm1 on M3 (E=256, 1536x6144) is 2.4GB -> the wrapper
    # must return None (Triton) instead of launching and faulting the GPU.
    free, _ = torch.cuda.mem_get_info()
    if free < 5 * 1024**3:
        pytest.skip("needs >5GB free to allocate the 2.4GB test weight")
    E, N, K, block_m, M = 256, 1536, 6144, 64, 8
    aq = torch.zeros(M, K, dtype=torch.uint8, device=DEVICE)
    asc = torch.zeros(M, K // 32, dtype=torch.uint8, device=DEVICE)
    w = torch.zeros(E, N, K, dtype=torch.uint8, device=DEVICE)  # 2.4GB
    wsc = torch.zeros(E, N, K // 32, dtype=torch.uint8, device=DEVICE)
    sids = torch.zeros(block_m, dtype=torch.int32, device=DEVICE)
    eids = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    npp = torch.tensor([block_m], dtype=torch.int32, device=DEVICE)
    # Returns None at the >2GB preflight, before any kernel launch (no fault).
    assert (
        grouped_gemm_mxfp8(
            aq, asc, w, wsc, sids, eids, npp, M, 4, block_m, torch.bfloat16, 4
        )
        is None
    )


# ── input-validation guards (negative tests) ──────────────────────────────────
@requires_gfx950
@torch.inference_mode()
def test_gemv_rejects_non_multiple_of_32_K():
    # K not a multiple of the 1x32 MX scale block must be rejected, not truncated.
    from aiter.ops.smallm_gemm_mxfp8 import smallm_mxfp8_gemv

    M, K, N = 1, 48, 64  # 48 % 32 != 0
    kb = (K + 31) // 32
    Xq = torch.zeros(M, K, dtype=torch.uint8, device=DEVICE)
    Wq = torch.zeros(N, K, dtype=torch.uint8, device=DEVICE)
    Xs = torch.zeros(M, kb, dtype=torch.uint8, device=DEVICE)
    Ws = torch.zeros(N, kb, dtype=torch.uint8, device=DEVICE)
    with pytest.raises(RuntimeError, match="multiple of 32"):
        smallm_mxfp8_gemv(Xq, Xs, Wq, Ws, torch.bfloat16, 8)


@requires_gfx950
@torch.inference_mode()
def test_run_gemv_rejects_M_above_16():
    # The GEMV template only instantiates M-tiles up to 16; a larger M must be
    # rejected loudly rather than silently mis-padded.
    from aiter.ops.smallm_gemm_mxfp8 import _run_gemv

    M, K, N = 17, 6144, 2304
    xq, xs = quant_mxfp8(torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16))
    wq, ws = quant_mxfp8(torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="M<=16"):
        _run_gemv(xq, xs, wq, ws, torch.bfloat16, M, K)


def _valid_raw_moe_args():
    """A known-good raw smallm_mxfp8_moe_grouped_gemm positional arg list (the
    gemm2 K=768 config); negative tests mutate one entry to trip a guard."""
    torch.manual_seed(0)
    E, N, K, a_div, block_m, T, top_k = 128, 6144, 768, 1, 64, 8, 1
    M = T * top_k  # num_valid_tokens
    aq, asc = quant_mxfp8(torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16) * 0.5)
    wq, wsc = quant_mxfp8(
        torch.randn(E, N, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    )
    topk_ids = torch.randint(0, E, (T, top_k), device=DEVICE, dtype=torch.int32)
    sorted_ids, expert_ids, num_post = moe_align(topk_ids, block_m, E)
    out = torch.zeros(M, N, device=DEVICE, dtype=torch.bfloat16)
    return [
        aq.view(torch.uint8),
        asc,
        wq.view(torch.uint8),
        wsc,
        sorted_ids,
        expert_ids,
        num_post,
        out,
        E,
        N,
        K,
        M,
        M,
        a_div,
        block_m,
    ]


@requires_gfx950
@torch.inference_mode()
def test_moe_rejects_non_float32_weight():
    # mul_weight_by is dereferenced as raw on-device float32; a wrong-dtype
    # tensor must be rejected before the kernel reads it.
    from aiter.ops.smallm_gemm_mxfp8 import smallm_mxfp8_moe_grouped_gemm

    args = _valid_raw_moe_args()
    num_valid = args[11]
    bad_wt = torch.ones(num_valid, dtype=torch.int32, device=DEVICE)  # int32, not f32
    with pytest.raises(RuntimeError, match="float32"):
        smallm_mxfp8_moe_grouped_gemm(*args, bad_wt)


@requires_gfx950
@torch.inference_mode()
def test_moe_rejects_undersized_out():
    # out smaller than num_valid_tokens*N would OOB-write; must be rejected.
    from aiter.ops.smallm_gemm_mxfp8 import smallm_mxfp8_moe_grouped_gemm

    args = _valid_raw_moe_args()
    args[7] = torch.zeros(1, dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(RuntimeError, match="out has"):
        smallm_mxfp8_moe_grouped_gemm(*args, None)


if __name__ == "__main__":
    # CI runs op_tests via `python3 <file>`; without this the pytest tests are
    # only defined, never executed (CI would report a pass having run nothing).
    # On non-gfx95x the requires_gfx950 marker skips them cleanly.
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

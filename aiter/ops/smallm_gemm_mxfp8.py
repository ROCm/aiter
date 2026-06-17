# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Decode small-M MX-FP8 (e4m3 data + e8m0 1x32 K-scales) GEMM kernels for gfx950:
# smallm_mxfp8_gemv / smallm_mxfp8_mfma (dense) and smallm_mxfp8_moe_grouped_gemm
# (MoE). The public wrappers engage HIP for in-envelope shapes, else return None
# (caller falls back to Triton).
from functools import lru_cache
from typing import Optional

import torch

from ..jit.core import compile_ops


# ── raw kernel entry points (JIT-built on first call) ──────────────────────────
@compile_ops("module_smallm_mxfp8_dense")
def smallm_mxfp8_gemv(
    Xq: torch.Tensor,
    Xs: torch.Tensor,
    Wq: torch.Tensor,
    Ws: torch.Tensor,
    out_dtype: torch.dtype,
    block_n: int,
) -> torch.Tensor: ...


@compile_ops("module_smallm_mxfp8_dense_mfma")
def smallm_mxfp8_mfma(
    Xq: torch.Tensor,
    Xs: torch.Tensor,
    Wq: torch.Tensor,
    Ws: torch.Tensor,
    out_dtype: torch.dtype,
    n_sub: int,
    k_splits: int,
) -> torch.Tensor: ...


@compile_ops("module_smallm_mxfp8_moe")
def smallm_mxfp8_moe_grouped_gemm(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    out: torch.Tensor,
    E: int,
    N: int,
    K: int,
    num_valid_tokens: int,
    M_act: int,
    a_div: int,
    block_m: int,
    mul_weight_by: Optional[torch.Tensor],
    block_n: int = 8,
) -> torch.Tensor: ...


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


# ── dense GEMV: kernel envelope + MFMA crossover ───────────────────────────────
_SUPPORTED_M_TILES = (1, 2, 4, 8, 16)  # GEMV kernel template instantiations

# (K, N) -> {M: (k_splits, n_sub)} autotuned MFMA config (M in {8,16,32,64}).
_MFMA_CFG = {
    # TP4 per-GPU shapes
    (6144, 2304): {8: (8, 1), 16: (8, 1), 32: (8, 2), 64: (8, 2)},  # qkv
    (2048, 6144): {8: (1, 1), 16: (1, 1), 32: (1, 1), 64: (1, 2)},  # o_proj
    (6144, 1536): {8: (2, 1), 16: (8, 1), 32: (8, 2), 64: (4, 2)},  # mlp_gate_up
    (1536, 6144): {8: (1, 1), 16: (1, 1), 32: (1, 1), 64: (1, 2)},  # mlp_down
    # TP8 per-GPU shapes (autotuned on MI355X; (k_splits, n_sub))
    (6144, 1152): {8: (2, 1), 16: (2, 1), 32: (8, 1), 64: (8, 2)},  # qkv        @TP8
    (6144, 768): {8: (2, 1), 16: (2, 1), 32: (2, 1), 64: (8, 1)},  # mlp_gate_up @TP8
    (1024, 6144): {8: (1, 1), 16: (1, 2), 32: (1, 1), 64: (1, 2)},  # o_proj     @TP8
    (768, 6144): {8: (1, 1), 16: (1, 2), 32: (1, 2), 64: (1, 2)},  # mlp_down    @TP8
    # TP1 per-GPU shapes (full, unsharded). M=64 omitted where HIP loses -> Triton.
    (6144, 9216): {
        8: (4, 1),
        16: (8, 2),
        32: (4, 2),
    },  # qkv             @TP1 (M64->Triton)
    (6144, 6144): {8: (4, 1), 16: (8, 2), 32: (8, 2), 64: (1, 2)},  # gate_up/down @TP1
    (8192, 6144): {8: (4, 1), 16: (8, 2), 32: (4, 2), 64: (1, 2)},  # o_proj      @TP1
    # TP2 per-GPU shapes. M=64 omitted where HIP loses -> Triton.
    (6144, 4608): {8: (8, 1), 16: (8, 2), 32: (8, 2), 64: (4, 2)},  # qkv         @TP2
    (6144, 3072): {8: (8, 1), 16: (4, 1), 32: (4, 2), 64: (2, 2)},  # mlp_gate_up @TP2
    (4096, 6144): {
        8: (4, 1),
        16: (8, 2),
        32: (4, 2),
    },  # o_proj          @TP2 (M64->Triton)
    (3072, 6144): {
        8: (1, 1),
        16: (2, 1),
        32: (1, 1),
    },  # mlp_down        @TP2 (M64->Triton)
}
_MFMA_M_SET = frozenset({8, 16, 32, 64})
# Untuned-shape default-engage ceiling: HIP robustly beats Triton for M <= 16 on
# every measured shape, but can lose at M in {32,64} on untuned shapes -- those
# require a tuned CSV / _MFMA_CFG entry to engage.
_DEFAULT_ENGAGE_MAX_M = 16


def _next_supported_m(m: int) -> int:
    for t in _SUPPORTED_M_TILES:
        if m <= t:
            return t
    return _SUPPORTED_M_TILES[-1]


@lru_cache(maxsize=1)
def _is_gfx950() -> bool:
    # Match the C++ host guard exactly (get_gpu_arch()=="gfx950"): the device
    # reports "gfx950:sramecc+:xnack-", so compare the bare arch before the ':'.
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
        return arch.split(":")[0] == "gfx950"
    except Exception:
        return False


def _default_mfma_cfg(K: int):
    """Untuned-shape MFMA config (k_splits, n_sub). Split-K helps the tall-skinny
    decode reduction; pick the largest power-of-2 k_splits <= 4 that divides
    K/128, n_sub=1 (always valid). The CSV / _MFMA_CFG override this when tuned."""
    kiters = K // 128
    for ks in (4, 2):
        if kiters % ks == 0:
            return (ks, 1)
    return (1, 1)


def _run_mfma(xq, x_scale, wq, w_scale, out_dtype, n_sub, k_splits):
    # The raw-buffer kernels index every input as a contiguous row-major buffer;
    # a non-contiguous payload or scale tensor would mis-index silently.
    return smallm_mxfp8_mfma(
        _as_u8(xq).contiguous(),
        x_scale.contiguous(),
        _as_u8(wq).contiguous(),
        w_scale.contiguous(),
        out_dtype,
        n_sub,
        k_splits,
    )


def _run_gemv(xq, x_scale, wq, w_scale, out_dtype, M, K):
    block_n = 8  # locked: BLOCK_N=8 is the measured winner for M<=8
    if M > _SUPPORTED_M_TILES[-1]:
        raise ValueError(f"gemv supports M<={_SUPPORTED_M_TILES[-1]}, got M={M}")
    # Raw-buffer kernels require contiguous row-major payload and scale buffers.
    xq_u, wq_u = _as_u8(xq).contiguous(), _as_u8(wq).contiguous()
    x_scale, w_scale = x_scale.contiguous(), w_scale.contiguous()
    m_tile = _next_supported_m(M)
    if m_tile == M:
        return smallm_mxfp8_gemv(xq_u, x_scale, wq_u, w_scale, out_dtype, block_n)
    # Pad to a supported M_TILE (zero rows discarded by the [:M] slice);
    # deterministic shape keeps cuda-graph capture happy.
    xq_pad = torch.zeros((m_tile, K), dtype=xq_u.dtype, device=xq_u.device)
    xq_pad[:M].copy_(xq_u)
    xs_pad = torch.zeros(
        (m_tile, x_scale.shape[1]), dtype=x_scale.dtype, device=x_scale.device
    )
    xs_pad[:M].copy_(x_scale)
    out_pad = smallm_mxfp8_gemv(xq_pad, xs_pad, wq_u, w_scale, out_dtype, block_n)
    return out_pad[:M].contiguous()


@lru_cache(maxsize=1)
def _tuned_cfg():
    """Autotuned per-(K,N,M) config from op_tests/tune_smallm_mxfp8.py:
    (K,N,M) -> (kernel, n_sub, k_splits, use_hip). Empty when the CSV has not
    been generated, in which case mxfp8_gemv uses the hand-tuned tables below."""
    import csv
    import os

    path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "smallm_mxfp8_tuned.csv"
    )
    out = {}
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                out[(int(r["K"]), int(r["N"]), int(r["M"]))] = (
                    r["kernel"],
                    int(r["n_sub"]),
                    int(r["k_splits"]),
                    bool(int(r["use_hip"])),
                )
    except FileNotFoundError:
        pass
    return out


def mxfp8_gemv(
    xq: torch.Tensor,  # [M, K] fp8 e4m3fn (or uint8 view)
    x_scale: torch.Tensor,  # [M, K//32] uint8 (E8M0)
    wq: torch.Tensor,  # [N, K] fp8 e4m3fn
    w_scale: torch.Tensor,  # [N, K//32] uint8 (E8M0)
    out_dtype: torch.dtype = torch.bfloat16,
) -> Optional[torch.Tensor]:
    """Decode dense MX-FP8 linear (X @ W.T) -> [M, N], or None when the shape is
    outside the kernel envelope or a tuned entry routes it to Triton. The
    autotuned CSV is consulted first; absent an entry, an in-envelope shape still
    engages via the hand-tuned _MFMA_CFG or a default config."""
    if out_dtype != torch.bfloat16:
        return None
    if not _is_gfx950():
        return None
    M, K = xq.shape
    N = wq.shape[0]

    tuned = _tuned_cfg().get((K, N, M))
    if tuned is not None:
        kernel, n_sub, k_splits, use_hip = tuned
        if not use_hip:
            return None
        try:
            if kernel == "mfma":
                return _run_mfma(xq, x_scale, wq, w_scale, out_dtype, n_sub, k_splits)
            return _run_gemv(xq, x_scale, wq, w_scale, out_dtype, M, K)
        except Exception:
            return None

    # No tuned entry: engage in-envelope shapes via _MFMA_CFG or a default.
    # Untuned M in {32,64} can lose to Triton (measured), so require a tuned entry.
    try:
        if M in _MFMA_M_SET:
            cfg = _MFMA_CFG.get((K, N), {}).get(M)
            if cfg is None:
                if M > _DEFAULT_ENGAGE_MAX_M:
                    return None
                cfg = _default_mfma_cfg(K)
            k_splits, n_sub = cfg
            return _run_mfma(xq, x_scale, wq, w_scale, out_dtype, n_sub, k_splits)
        if M <= _SUPPORTED_M_TILES[-1]:
            return _run_gemv(xq, x_scale, wq, w_scale, out_dtype, M, K)
    except Exception:
        return None
    return None


# ── MoE grouped GEMM: shape envelope ───────────────────────────────────────────
# (N, K, a_div, has_weight) -> list of (M_routed_lo, M_routed_hi, E_min). HIP beats
# the Triton grouped GEMM when M_routed in [lo, hi] AND experts-per-GPU E >= E_min.
# Keyed WITHOUT E (engages under any expert-parallel degree), but the bucket bound
# is E-aware because the win-envelope widens with E: more experts -> more routing
# padding -> Triton wastes more, so HIP's relative position improves at higher
# M_routed. Measured on MI355X (block_m=64, top_k=4; M3 = 256 experts):
#   gemm1 (deep K=6144): wins to M_routed 16 at any E (EP16/8/4 = E 16/32/64), and
#                        to 64 once E>=128 (EP2). no-EP (E=256) exceeds the 2GB
#                        raw-buffer limit -> rejected (falls to Triton).
#   gemm2 (shallow K=768): wins to M_routed 8 at any E (incl. no-EP E=256, 1.2GB).
_MOE_ALLOWLIST = {
    (1536, 6144, 4, False): [(1, 16, 0), (17, 64, 128)],  # gemm1 gate_up (deep K)
    (6144, 768, 1, True): [(1, 8, 0)],  # gemm2 down weighted (shallow K)
    (6144, 768, 1, False): [(1, 8, 0)],  # gemm2 down no-combine
}

# Per-(N, K) BLOCK_N for the MoE kernel (autotuned on MI355X). The dense-GEMV tile
# (8) is the default and wins on deep-K gemm1; shallow-K / wide-N gemm2 prefers 16
# (~+17% at the decode operating point). Only {8, 16} are kernel-supported.
_MOE_BLOCK_N = {
    (6144, 768): 16,  # gemm2 down (K=768)
}


def grouped_gemm_mxfp8(
    a_q: torch.Tensor,  # [M, K] fp8 e4m3 (or uint8)
    a_scale: torch.Tensor,  # [M, K//32] uint8
    w: torch.Tensor,  # [E, N, K] fp8 e4m3 (or uint8)
    w_scale: torch.Tensor,  # [E, N, K//32] uint8
    sorted_token_ids: torch.Tensor,  # int32
    expert_ids: torch.Tensor,  # int32
    num_tokens_post_padded: torch.Tensor,  # int32 [1]
    num_valid_tokens: int,
    top_k: int,
    block_m: int,
    out_dtype: torch.dtype,
    a_div: int,
    mul_weight_by: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,  # accepted for caller-signature parity
) -> Optional[torch.Tensor]:
    """Decode MoE MX-FP8 grouped GEMM (sorted_token_ids layout, matches the Triton
    helper). Returns the [M_routed, N] output, or None when the shape is outside
    the measured-win envelope (caller falls back to Triton)."""
    M_act, K = a_q.shape
    E, N, K2 = w.shape
    M_routed = num_valid_tokens

    # Kernel preflight.
    if not _is_gfx950():
        return None
    if K != K2 or out_dtype != torch.bfloat16:
        return None
    if K % 1024 != 0 and K != 768:  # multiple of K_PER_WARP_STEP=1024 or known short-K
        return None
    if block_m % 4 != 0 or a_div not in (1, 4, 8):
        return None
    # Raw-buffer voffset is signed int32: a >=2GB byte offset wraps negative and
    # faults the kernel (uncatchable). Reject here so low-EP / large-E shapes
    # (e.g. no-EP gemm1 at E=256 = 2.4GB) fall back to Triton cleanly.
    if E * N * K > 0x7FFFFFFF or M_act * K > 0x7FFFFFFF:
        return None

    buckets = _MOE_ALLOWLIST.get((N, K, a_div, mul_weight_by is not None))
    if buckets is None or not any(
        lo <= M_routed <= hi and E >= e_min for lo, hi, e_min in buckets
    ):
        return None

    try:
        # Raw-buffer kernels index payload/scale tensors as contiguous row-major.
        aq_u = _as_u8(a_q).contiguous()
        w_u = _as_u8(w).contiguous()
        w_scale = w_scale.contiguous()
        a_scale_c = a_scale.contiguous() if a_scale.stride(-1) != 1 else a_scale
        # zeros so 0-token tiles stay zero on the output side (matches Triton).
        out = torch.zeros((M_routed, N), dtype=out_dtype, device=a_q.device)
        wt = None
        if mul_weight_by is not None:
            # Kernel reads this as on-device contiguous float32; .to(dtype) alone
            # would not move a CPU tensor onto the GPU.
            wt = mul_weight_by.to(device=a_q.device, dtype=torch.float32).contiguous()
        sti = (
            sorted_token_ids
            if sorted_token_ids.dtype == torch.int32
            else sorted_token_ids.to(torch.int32)
        )
        ei = (
            expert_ids
            if expert_ids.dtype == torch.int32
            else expert_ids.to(torch.int32)
        )
        ntp = (
            num_tokens_post_padded
            if num_tokens_post_padded.dtype == torch.int32
            else num_tokens_post_padded.to(torch.int32)
        )
        smallm_mxfp8_moe_grouped_gemm(
            aq_u,
            a_scale_c,
            w_u,
            w_scale,
            sti,
            ei,
            ntp,
            out,
            E,
            N,
            K,
            int(M_routed),
            int(M_act),
            int(a_div),
            int(block_m),
            wt,
            _MOE_BLOCK_N.get((N, K), 8),
        )
        return out
    except Exception:
        return None

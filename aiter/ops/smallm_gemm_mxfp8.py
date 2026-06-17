# SPDX-License-Identifier: MIT
# Decode small-M MX-FP8 (OCP, e4m3 data + e8m0 1x32 K-scales) GEMM kernels for
# MiniMax-M3 on AMD gfx950 (MI355X). Three custom kernels tuned for the decode
# regime (M small), each a standalone JIT module:
#   * smallm_mxfp8_gemv               dense GEMV (M in {1,2,4,8,16})
#   * smallm_mxfp8_mfma               dense MFMA crossover (M in {8,16,32,64}, split-K)
#   * smallm_mxfp8_moe_grouped_gemm   MoE grouped GEMM (sorted_token_ids layout)
#
# The public wrappers below carry the measured shape envelope (allowlist) and
# kernel-API mechanics (fp8->uint8 view, M-tile padding, MFMA config). They
# return None when the shape is outside the supported/measured set; the caller
# (e.g. vLLM's mxfp8_native_moe / rocm_native dispatchers) falls back to Triton.
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
) -> torch.Tensor: ...


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


# ── dense GEMV: shape envelope + MFMA crossover ────────────────────────────────
# (K, N) -> set of M for which the Lever-1 GEMV beats the Triton dot_scaled path.
_DENSE_ALLOWLIST = {
    (6144, 2304): set(range(1, 9)),   # qkv_proj:    M in {1..8}
    (6144, 1536): set(range(1, 9)),   # mlp_gate_up: M in {1..8}
    (2048, 6144): set(range(1, 5)),   # o_proj:      M in {1..4}
    (1536, 6144): set(range(1, 5)),   # mlp_down:    M in {1..4}
}
_SUPPORTED_M_TILES = (1, 2, 4, 8, 16)  # kernel template instantiations

# (K, N) -> {M: (k_splits, n_sub)} where the MFMA crossover beats bf16 (M in {8,16,32,64}).
_MFMA_CFG = {
    (6144, 2304): {8: (4, 1), 16: (8, 1), 32: (4, 1), 64: (2, 1)},   # qkv
    (2048, 6144): {8: (1, 1), 16: (1, 1), 32: (1, 1), 64: (1, 2)},   # o_proj
    (6144, 1536): {8: (4, 1), 16: (8, 1), 32: (4, 1), 64: (2, 1)},   # mlp_gate_up
    (1536, 6144): {8: (1, 1), 16: (1, 1), 32: (1, 1), 64: (1, 1)},   # mlp_down
}
_MFMA_M_SET = frozenset({8, 16, 32, 64})


def _next_supported_m(m: int) -> int:
    for t in _SUPPORTED_M_TILES:
        if m <= t:
            return t
    return _SUPPORTED_M_TILES[-1]


def _run_mfma(xq, x_scale, wq, w_scale, out_dtype, n_sub, k_splits):
    return smallm_mxfp8_mfma(
        _as_u8(xq), x_scale, _as_u8(wq), w_scale, out_dtype, n_sub, k_splits
    )


def _run_gemv(xq, x_scale, wq, w_scale, out_dtype, M, K):
    block_n = 8  # locked: BLOCK_N=8 is the measured winner for M<=8
    xq_u, wq_u = _as_u8(xq), _as_u8(wq)
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
    xq: torch.Tensor,       # [M, K] fp8 e4m3fn (or uint8 view)
    x_scale: torch.Tensor,  # [M, K//32] uint8 (E8M0)
    wq: torch.Tensor,       # [N, K] fp8 e4m3fn
    w_scale: torch.Tensor,  # [N, K//32] uint8 (E8M0)
    out_dtype: torch.dtype = torch.bfloat16,
) -> Optional[torch.Tensor]:
    """Decode dense MX-FP8 linear (X @ W.T). Returns [M, N], or None when the
    shape is outside the win envelope (caller falls back to Triton).

    Prefers the autotuned CSV (configs/smallm_mxfp8_tuned.csv) when present
    (config + HIP-vs-Triton decision per shape); else uses the hand-tuned
    _MFMA_CFG / _DENSE_ALLOWLIST below."""
    if out_dtype != torch.bfloat16:
        return None  # kernels only emit bf16
    M, K = xq.shape
    N = wq.shape[0]

    tuned = _tuned_cfg().get((K, N, M))
    if tuned is not None:
        kernel, n_sub, k_splits, use_hip = tuned
        if not use_hip:
            return None  # autotune found Triton faster for this shape
        try:
            if kernel == "mfma":
                return _run_mfma(xq, x_scale, wq, w_scale, out_dtype, n_sub, k_splits)
            return _run_gemv(xq, x_scale, wq, w_scale, out_dtype, M, K)
        except Exception:
            return None

    # ── hand-tuned fallback (no CSV entry for this shape) ──
    if M in _MFMA_M_SET:
        cfg = _MFMA_CFG.get((K, N))
        if cfg is not None and M in cfg:
            k_splits, n_sub = cfg[M]
            try:
                return _run_mfma(xq, x_scale, wq, w_scale, out_dtype, n_sub, k_splits)
            except Exception:
                pass
    allowed = _DENSE_ALLOWLIST.get((K, N))
    if allowed is None or M not in allowed:
        return None
    try:
        return _run_gemv(xq, x_scale, wq, w_scale, out_dtype, M, K)
    except Exception:
        return None


# ── MoE grouped GEMM: shape envelope ───────────────────────────────────────────
# (E, N, K, a_div, has_weight) -> list of (M_routed_lo, M_routed_hi) buckets the
# kernel wins. Measured on MI355X / gfx950 with E=128 / top_k=4, caller block_m=64.
_MOE_ALLOWLIST = {
    (128, 1536, 6144, 4, False): [(4, 16)],   # gemm1 (gate_up)
    (128, 6144,  768, 1, True): [(4, 8)],     # gemm2 weighted (K=768)
    (128, 6144,  768, 1, False): [(4, 8)],    # gemm2 no-combine (K=768)
}


def grouped_gemm_mxfp8(
    a_q: torch.Tensor,                       # [M, K] fp8 e4m3 (or uint8)
    a_scale: torch.Tensor,                   # [M, K//32] uint8
    w: torch.Tensor,                         # [E, N, K] fp8 e4m3 (or uint8)
    w_scale: torch.Tensor,                   # [E, N, K//32] uint8
    sorted_token_ids: torch.Tensor,          # int32
    expert_ids: torch.Tensor,                # int32
    num_tokens_post_padded: torch.Tensor,    # int32 [1]
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
    if K != K2 or out_dtype != torch.bfloat16:
        return None
    if K % 1024 != 0 and K != 768:  # multiple of K_PER_WARP_STEP=1024 or known short-K
        return None
    if block_m % 4 != 0 or a_div not in (1, 4, 8):
        return None

    buckets = _MOE_ALLOWLIST.get((E, N, K, a_div, mul_weight_by is not None))
    if buckets is None or not any(lo <= M_routed <= hi for lo, hi in buckets):
        return None

    try:
        aq_u = _as_u8(a_q).contiguous()
        w_u = _as_u8(w)
        a_scale_c = a_scale.contiguous() if a_scale.stride(-1) != 1 else a_scale
        # zeros so 0-token tiles stay zero on the output side (matches Triton).
        out = torch.zeros((M_routed, N), dtype=out_dtype, device=a_q.device)
        wt = None
        if mul_weight_by is not None:
            wt = mul_weight_by.to(torch.float32) if mul_weight_by.dtype != torch.float32 else mul_weight_by
            wt = wt.contiguous()
        sti = sorted_token_ids if sorted_token_ids.dtype == torch.int32 else sorted_token_ids.to(torch.int32)
        ei = expert_ids if expert_ids.dtype == torch.int32 else expert_ids.to(torch.int32)
        ntp = num_tokens_post_padded if num_tokens_post_padded.dtype == torch.int32 else num_tokens_post_padded.to(torch.int32)
        smallm_mxfp8_moe_grouped_gemm(
            aq_u, a_scale_c, w_u, w_scale, sti, ei, ntp, out,
            E, N, K, int(M_routed), int(M_act), int(a_div), int(block_m), wt,
        )
        return out
    except Exception:
        return None

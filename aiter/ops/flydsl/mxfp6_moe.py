"""MXFP6 grouped MoE (FlyDSL, gfx950): routed-expert span of one SwiGLU MoE layer.

    y[t] = sum_j w[t, j] * W2[e_tj] @ (silu(W1g[e_tj] @ x[t]) * (W1u[e_tj] @ x[t]))

Weights MXFP6 (E2M3 default, E3M2 via MX6_WFMT=e3m2); activations MXFP6 (W6A6, MX6_AFMT=e2m3|e3m2) or
MXFP8 E4M3 (W6A8, MX6_AFMT=e4m3); E8M0 scale per 32 along K. E2M3 quantization is bitwise identical to the
usual reference (scale 2^ceil(log2(amax / 7.5)), round-half-even onto the grid, clamp).

Plugin contract (`run`): called once with inp=None and {"w1": [E, 2I, H] bf16 (gate rows first),
"w2": [E, H, I] bf16} to prepare weights; then per forward with {"x": [T, H] bf16, "ids": [T, K] int32,
"w": [T, K] fp32}, returns y [T, H] bf16.

Pipeline per forward (all FlyDSL, no host syncs, bitwise deterministic):
  1. route: counting sort by expert (one block, or histogram/scan/scatter above 32k entries) -> gather
     index, tile -> expert map, used-tile count, pos[t, j] (slot order within an expert does not affect bits)
  2. quant_rows: x -> MX rows + scales
  3. stage 1 (gg6e "swiglu_q"): gathered x @ W1^T, SwiGLU, MX quantization of h in the epilogue
  4. stage 2 (gg6e "store"): h @ W2^T -> bf16 rows
  5. combine: y[t] = sum_j w[t, j] * o[pos[t, j]] in fixed j order (fp32), no atomics
Constraints: hidden and inter multiples of 128; any expert count, top-k and token count; weights > 4 GiB ok.
"""

from __future__ import annotations

import os

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from . import mxfp6_moe_quant as q
from .kernels import mxfp6_moe_aux as ak
from .kernels.mxfp6_moe_gemm import run_gg6e


def pick_bm(rows_per_expert: float) -> int:
    """256-row tiles waste most of the work when experts get ~100 rows (many experts / few tokens)."""
    return 128 if rows_per_expert <= 160 else 256


def stage_bn(N: int, a_fmt: str) -> int:
    """N tile of a stage. Fixed per stage at prepare time: the weights are preshuffled N-block-major for it.
    128 when N % 256 != 0 (hidden a multiple of 128 only) or for FP8 activations (256 x 256 with FP8 A rows
    does not fit three LDS stages), else 256."""
    return 128 if (N % 256 != 0 or a_fmt == "e4m3") else 256


def gemm_cfg(N: int, a_fmt: str, bm: int = 256):
    """gg6e (FM, FN, WGM, WGN), always 4 waves (2 x 2); block BM x stage_bn(N): wave tile (BM/2) x (BN/2)."""
    return (bm // 32, stage_bn(N, a_fmt) // 32, 2, 2)


A_FMT = os.environ.get("MX6_AFMT", "e2m3")
W_FMT = os.environ.get("MX6_WFMT", "e2m3")  # e2m3 | e3m2 (OCP MX FP6 variants)


def prepare(layer: dict) -> dict:
    w1, w2 = layer["w1"], layer["w2"]
    E, N1, H = w1.shape
    I = w2.shape[2]
    assert N1 == 2 * I and H % 128 == 0 and I % 128 == 0, "hidden and inter must be multiples of 128"
    dev = w1.device
    bn1, bn2 = stage_bn(N1, A_FMT), stage_bn(H, A_FMT)
    b1 = torch.empty(E, N1 // bn1, H // 128, bn1 // 16, 64 * 24, dtype=torch.uint8, device=dev)
    s1 = torch.empty(E, N1 // 16, q.kq(H), 64, dtype=torch.int32, device=dev)
    b2 = torch.empty(E, H // bn2, I // 128, bn2 // 16, 64 * 24, dtype=torch.uint8, device=dev)
    s2 = torch.empty(E, H // 16, q.kq(I), 64, dtype=torch.int32, device=dev)
    for e in range(E):  # per expert: the fp32 working copy of all experts does not fit
        c, s = q.quant_mx(w1[e : e + 1], W_FMT)
        b1[e : e + 1], s1[e : e + 1] = q.preshuffle_b(c, s, interleave_gate_up=True, bn=bn1)
        c, s = q.quant_mx(w2[e : e + 1], W_FMT)
        b2[e : e + 1], s2[e : e + 1] = q.preshuffle_b(c, s, bn=bn2)
    return {"E": E, "H": H, "I": I, "b1": b1, "s1": s1, "b2": b2, "s2": s2, "a_fmt": A_FMT, "w_fmt": W_FMT}


def route(ids: torch.Tensor, E: int, bm: int):
    """FlyDSL counting-sort router: ridx, pos [T, K], texp [+count], M. One block for small inputs
    (fewest launches), multi-block (histogram / scan / scatter) above 32k entries (~5x faster at 128k)."""
    if ids.numel() > 32768:
        return ak.route_mb(ids, E, bm)
    return ak.route(ids, E, bm)


def quant_rows(x: torch.Tensor, fmt: str):
    return ak.quant_rows(x, fmt)


def forward(inp: dict, st: dict) -> torch.Tensor:
    x, ids, w = inp["x"], inp["ids"], inp["w"]
    T, K = ids.shape
    E, H, I, af = st["E"], st["H"], st["I"], st["a_fmt"]
    bm = pick_bm(T * K / E)
    ridx, pos, texp, M = route(ids, E, bm)
    xa, xs = quant_rows(x, af)
    # stage 1 writes h already MX-quantized (stage-2 A rows + [M, KQ, 4] scales)
    ha = torch.empty(M, I * q.FORMATS[af][3] // 8, dtype=torch.uint8, device=x.device)
    hs = torch.empty(M, q.kq(I), 4, dtype=torch.int32, device=x.device)
    run_gg6e(xa, xs, ridx, texp, st["b1"], st["s1"], ha, N=2 * I, K=H, a_fmt=af, b_fmt=st["w_fmt"], epi="swiglu_q", gather=True, gm=8,
             cfg=gemm_cfg(2 * I, af, bm), out_scales=hs)
    o = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    run_gg6e(ha, hs, ridx, texp, st["b2"], st["s2"], o, N=H, K=I, a_fmt=af, b_fmt=st["w_fmt"], epi="store", gather=False, gm=2,
             cfg=gemm_cfg(H, af, bm))
    return ak.combine(o, pos, w)


SUPPORTED_GFX = ("gfx950",)  # scaled F8F6F4 MFMA + 160 KiB LDS


def run(inp, state):
    if get_gfx_runtime() not in SUPPORTED_GFX:
        raise RuntimeError(f"mxfp6_moe needs one of {SUPPORTED_GFX}, got {get_gfx_runtime()}")
    if inp is None:
        return prepare(state)
    return forward(inp, state)

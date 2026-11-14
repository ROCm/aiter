# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Muse Spark MoE on MXFP4 weights: FlyDSL grouped GEMMs + per-expert RMSNorm finalize.

Reproduces one expert-parallel rank of the Muse Spark MoE as the SGLang Muse
backend calls it (``muse_moe_flydsl``):

1. sort the (token, slot) pairs by local expert, BLOCK_M-padded
   (``sorted_ids`` hold ``token * topk + slot``; padding holds ``M * topk``);
2. ``launch_muse_moe_gemm`` stage 1: ``inter = SiLU(x @ gate^T) * (x @ up^T)``;
3. ``launch_muse_moe_gemm`` stage 2: ``y = inter @ down^T`` (one row per slot);
4. ``build_finalize``: per slot ``v = (gamma + gain) * y``, then
   ``out[t] = sum_s score_s * v_s / rms(v_s)`` over the slots on local experts.

Candidates: the same kernels on the checkpoint (native) weight layout, and on
aiter's a16w4 preshuffle (``shuffle_weight_a16w4(w, 16, False)``) with the
config ``FLYDSL_SHUFFLED_W``. The scales stay native in both. The preshuffled
output must be bit-identical to the native output.

Muse Spark Open (``config.json``): moe_hidden_size K = H = 4096,
moe_intermediate_size I = 4096, num_local_experts 256, num_experts_per_tok 8,
post_norm_eps 1e-8, post-expert norm gain center "one".
One rank holds 256 / EP experts: 32 at EP8, 64 at EP4 (``-e``).
Every rank sees all M tokens and computes only its local experts.
"""

import argparse
import functools
import itertools

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

torch.set_default_device("cuda")

# The kernels use the gfx950 v_permlane{16,32}_swap and the MXFP4 MFMA path.
SUPPORTED_GFX = ["gfx950"]

NUM_GLOBAL_EXPERTS = 256  # Muse Spark: num_local_experts in config.json
TOPK = 8  # num_experts_per_tok
EPS = 1e-8  # post_norm_eps
GAIN = 1.0  # post_norm_gain_center_type "one": gamma is zero-centered


def muse_mxfp4_config(M, topk=TOPK):
    """MXFP4 tile config of the SGLang Muse backend (``_pick_flydsl_config``).

    ``GEMM1_*`` / ``GEMM2_*`` keys override the shared value for one stage.
    """
    if M * topk > 1024:
        cfg = {"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 4}
        cfg["FLYDSL_K_UNROLL"] = 2 if M >= 4096 else 4
    else:
        cfg = {"BLOCK_M": 16, "BLOCK_N": 64 if M * topk <= 64 else 128, "num_warps": 4}
    if M <= 1024:
        # Weight streaming: two waves, a wide N tile and a 512-element K tile.
        cfg["BLOCK_M"] = 32 if M > 512 else 16
        cfg["BLOCK_N"] = 32 if M <= 4 else (64 if M <= 16 else 128)
        cfg["num_warps"] = 2
        cfg["FLYDSL_K_UNROLL"] = 16
        if M > 128:
            cfg["GEMM2_BLOCK_N"] = 256
    return cfg


def _stage_cfg(cfg, stage):
    out = dict(cfg)
    for key, value in cfg.items():
        if key.startswith(stage + "_"):
            out[key[len(stage) + 1 :]] = value
    return out


def moe_align_block_size(topk_ids, num_local_experts, local_start, block_m):
    """Sort (token, slot) pairs by local expert and pad each expert to ``block_m``.

    Returns ``sorted_ids`` [cap] (``token * topk + slot``, padding ``M * topk``),
    ``expert_ids`` [cap // block_m] and ``num_post_padded`` [1], all int32.
    Slots on other ranks' experts are dropped.
    """
    M, topk = topk_ids.shape
    E = num_local_experts
    n = M * topk
    local = topk_ids.reshape(-1).to(torch.int64) - local_start
    key = torch.where((local >= 0) & (local < E), local, torch.full_like(local, E))
    counts = torch.zeros(E + 1, dtype=torch.int64).scatter_add_(
        0, key, torch.ones_like(key)
    )
    padded = (counts[:E] + block_m - 1) // block_m * block_m
    cum_padded = torch.cumsum(padded, 0)
    order = torch.argsort(key, stable=True)
    sorted_key = key[order]
    rank = torch.arange(n) - (torch.cumsum(counts, 0) - counts)[sorted_key]
    cap = (n + E * (block_m - 1) + block_m - 1) // block_m * block_m
    start = torch.cat([cum_padded - padded, torch.full((1,), cap)])
    dest = torch.where(sorted_key < E, start[sorted_key] + rank, cap)
    sorted_ids = torch.full((cap + 1,), n, dtype=torch.int32)
    sorted_ids.scatter_(0, dest, order.to(torch.int32))
    block_start = torch.arange(cap // block_m) * block_m
    expert_ids = torch.searchsorted(cum_padded, block_start, right=True)
    expert_ids = torch.clamp(expert_ids, max=E - 1).to(torch.int32)
    return sorted_ids[:cap], expert_ids, cum_padded[-1:].to(torch.int32)


@functools.cache
def _finalize_launcher(H, topk):
    """One compiled finalize per shape, as the model caches it."""
    from aiter.ops.flydsl.muse_moe_finalize import build_finalize

    return build_finalize(H, topk, True, True, "bf16", 1)


def run_muse_moe(
    x, topk_ids, topk_scores, w13, s13, w2, s2, gamma, local_start, align, cfg
):
    """One rank of the Muse MoE, as ``muse_moe_flydsl`` launches it (MXFP4, no split-K).

    ``align`` is the output of ``moe_align_block_size``; the model computes it
    with a separate kernel, so it is not part of the timed region.
    """
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
    from aiter.ops.flydsl.muse_moe_gemm import launch_muse_moe_gemm

    M, K = x.shape
    topk = topk_ids.shape[1]
    E, two_i, _ = w13.shape
    I = two_i // 2
    H = w2.shape[1]
    num_valid = M * topk
    sorted_ids, expert_ids, num_post = align
    inter = torch.empty((num_valid, I), dtype=dtypes.bf16)
    launch_muse_moe_gemm(
        x, w13, s13, inter, sorted_ids, expert_ids, num_post,
        I, K, num_valid, topk, True, True, True, _stage_cfg(cfg, "GEMM1"),
    )  # fmt: skip
    y = torch.empty((num_valid, H), dtype=dtypes.bf16)
    launch_muse_moe_gemm(
        inter, w2, s2, y, sorted_ids, expert_ids, num_post,
        H, I, num_valid, topk, False, False, True, _stage_cfg(cfg, "GEMM2"),
    )  # fmt: skip
    out = torch.empty((M, H), dtype=dtypes.bf16)
    _run_compiled(
        _finalize_launcher(H, topk),
        y, gamma, topk_scores, topk_ids, out, local_start, E, EPS, GAIN, M,
        fx.Stream(torch.cuda.current_stream().cuda_stream),
    )  # fmt: skip
    return out


def _dequant(w, s):
    """Packed MXFP4 [N, K/2] + E8M0 [N, K/32] -> fp32 [N, K]."""
    w = mxfp4_to_f32(w)
    return w * e8m0_to_f32(s).repeat_interleave(32, dim=-1)


def run_torch(x, topk_ids, topk_scores, w13, s13, w2, s2, gamma, local_start):
    """fp32 reference with the two BF16 roundings the model keeps (inter and y)."""
    M = topk_ids.shape[0]
    E = w13.shape[0]
    I = w13.shape[1] // 2
    H = w2.shape[1]
    scale = (gamma.float() + GAIN).to(dtypes.bf16).float()
    out = torch.zeros((M, H), dtype=dtypes.fp32)
    for e in range(E):
        tok, slot = torch.nonzero(topk_ids == local_start + e, as_tuple=True)
        if tok.numel() == 0:
            continue
        h = x[tok].float() @ _dequant(w13[e], s13[e]).t()
        inter = (F.silu(h[:, :I]) * h[:, I:]).to(dtypes.bf16).float()
        y = (inter @ _dequant(w2[e], s2[e]).t()).to(dtypes.bf16).float()
        v = (y * scale).to(dtypes.bf16).float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)
        out.index_add_(0, tok, topk_scores[tok, slot, None] * v)
    return out.to(dtypes.bf16)


def _random_mxfp4(E, N, K):
    """Random packed E2M1 bytes and E8M0 exponents in 2**-8 .. 2**-5."""
    w = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8)
    s = torch.randint(119, 123, (E, N, K // 32), dtype=torch.uint8)
    return w, s


@benchmark()
def test_muse_moe(m, e, k, i):
    """m tokens, e local experts (256 / EP), k = moe_hidden_size, i = moe_intermediate_size."""
    from aiter.ops.shuffle import shuffle_weight_a16w4

    torch.manual_seed(0)  # same routing and weights in every run
    H = k
    local_start = 0
    x = torch.randn((m, k), dtype=dtypes.bf16)
    topk_ids = torch.rand((m, NUM_GLOBAL_EXPERTS)).topk(TOPK, dim=-1).indices
    topk_ids = topk_ids.to(torch.int32)
    topk_scores = torch.softmax(torch.randn((m, TOPK)), dim=-1)
    w13, s13 = _random_mxfp4(e, 2 * i, k)  # [gate; up] rows, as SGLang stacks them
    w2, s2 = _random_mxfp4(e, H, i)
    gamma = (0.1 * torch.randn(H)).to(dtypes.bf16)  # zero-centered
    ref = run_torch(x, topk_ids, topk_scores, w13, s13, w2, s2, gamma, local_start)

    # In-place preshuffle of the model (MUSE_MOE_SHUFFLE): weights only.
    w13_shuf, w2_shuf = (
        shuffle_weight_a16w4(w.view(dtypes.fp4x2), 16, False).view(torch.uint8)
        for w in (w13, w2)
    )
    cfg = muse_mxfp4_config(m)
    align = moe_align_block_size(topk_ids, e, local_start, cfg["BLOCK_M"])
    candidates = {
        "muse_native": (w13, w2, dict(cfg)),
        "muse_shuffled": (w13_shuf, w2_shuf, dict(cfg, FLYDSL_SHUFFLED_W=1)),
    }

    # Work of the local slots only; weight bytes of the local experts they hit.
    local = (topk_ids >= local_start) & (topk_ids < local_start + e)
    n_loc = int(local.sum())
    hit = int(torch.unique(topk_ids[local]).numel())
    flops = 2 * n_loc * k * 2 * i + 2 * n_loc * i * H
    w_bytes = hit * (2 * i * k + H * i) * (0.5 + 1 / 32)
    act_bytes = (m * k + 2 * n_loc * i + 2 * n_loc * H + m * H) * x.element_size()
    nbytes = w_bytes + act_bytes

    ret = {"gfx": get_gfx(), "local slots": n_loc, "BLOCK_M": cfg["BLOCK_M"]}
    outs = {}
    for name, (w13_c, w2_c, cfg_c) in candidates.items():
        out, us = run_perftest(
            run_muse_moe,
            x, topk_ids, topk_scores, w13_c, s13, w2_c, s2, gamma, local_start,
            align, cfg_c,
        )  # fmt: skip
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{name}: muse moe m={m} e={e}",
        )
        outs[name] = out
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    # The preshuffle changes only the weight byte address: the outputs must match bit for bit.
    checkAllclose(
        outs["muse_native"].to(dtypes.fp32),
        outs["muse_shuffled"].to(dtypes.fp32),
        rtol=0,
        atol=0,
        tol_err_ratio=0,
        msg=f"muse_shuffled vs muse_native (bit-exact): m={m} e={e}",
    )
    ret["shuffled == native"] = torch.equal(outs["muse_native"], outs["muse_shuffled"])
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("Muse FlyDSL MoE unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        nargs="*",
        # 1-16: decode and DFlash verify steps; 32-1024: batched decode / short
        # prefill; 4096-16384: prefill chunks (16k is the chunked-prefill size).
        default=[1, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384],
        help="""Number of tokens.
    e.g.: -m 16""",
    )
    parser.add_argument(
        "-e",
        "--experts",
        type=int,
        nargs="*",
        default=[32, 64],
        help="""Local experts on one rank: 256 / EP (32 = EP8, 64 = EP4).
    e.g.: -e 32""",
    )
    parser.add_argument(
        "-s",
        "--ki",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(4096, 4096)],
        help="""moe_hidden_size,moe_intermediate_size (Muse Spark: 4096,4096).
    e.g.: -s 4096,4096""",
    )
    args = parser.parse_args()

    df = []
    for e, (k, i), m in itertools.product(args.experts, args.ki, args.m):
        df.append(test_muse_moe(m, e, k, i))
    df = pd.DataFrame(df)
    aiter.logger.info("muse moe summary (markdown):\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()

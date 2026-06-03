"""Split-K (kb>=2) correctness for FlyDSL FP8 blockscale MoE stage1.

Compares kb=2 and kb=4 against a torch FP32 reference of UNACTIVATED
gate/up partials (the only thing the kb>=2 kernel writes). Also compares
kb=2 vs kb=4 directly to surface split-K self-consistency.

Why no kb=1 in the A/B: at kb=1 with gate_mode=interleave + out_dtype=bf16
the kernel fuses activation in the epilog and writes the activated value
into both gate and up interleave slots — that's a different output format
from kb>=2 (which writes raw unactivated partials), so a direct diff is
meaningless. We instead use the torch ref as the ground truth and report
how much split-K drifts from it.

Reports per (M, kb):
  vs_ref_max / vs_ref_mean / vs_ref_err%   — bf16 atomic accumulation error
                                              vs FP32 ground truth
  nondet_max                                — max diff across N repeats
                                              (atomic ordering jitter)

And per M:
  kb2_vs_kb4_max / mean                     — split-K self-consistency

Usage:
    HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/ab_splitk_correctness.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from test_flydsl_blockscale_moe import (  # noqa: E402
    DEVICE,
    _expand_blockscale,
    _launch_flydsl_stage1_splitk,
    _prepare_data,
    SCALE_BLOCK_N_DEFAULT,
    SCALE_BLOCK_K_DEFAULT,
)

MODEL_DIM = 7168
INTER_DIM = 256
EXPERTS   = 257
TOPK      = 9

M_VALUES = [1, 8, 16, 32, 64, 256, 1024]
KB_VALUES = [2, 4]
N_REPEATS = 3
TILE_M, TILE_N, TILE_K = 64, 128, 128


def _torch_stage1_partials_ref(data):
    """FP32 reference of UNACTIVATED gate/up — same shape as kb>=2 tmp_out
    after rearranging from interleave gui_layout.

    Returns (tokens, topk, 2*inter_dim) f32 where last-dim is [gate||up].
    """
    a_q = data["x_bq"]
    w1 = data["w1_bq"]
    topk_ids = data["topk_ids"]
    a_scale = data["x_bscale_ref"]
    w1_scale = data["w1_bscale_flat"]
    inter_dim = data["inter_dim"]
    blk_n, blk_k = SCALE_BLOCK_N_DEFAULT, SCALE_BLOCK_K_DEFAULT

    tokens, model_dim = a_q.shape
    topk = topk_ids.shape[1]
    expert = w1.shape[0]
    a = a_q.float()
    a = (a.view(tokens, -1, blk_k) * a_scale.unsqueeze(-1)).view(tokens, model_dim)
    w = w1.float() * _expand_blockscale(
        w1_scale, expert, (2 * inter_dim) // blk_n, model_dim // blk_k, blk_n, blk_k
    )

    a_rep = a.unsqueeze(1).expand(-1, topk, -1)
    gu = torch.zeros(tokens, topk, 2 * inter_dim, dtype=torch.float32, device=a.device)
    for e in range(expert):
        mask = topk_ids == e
        if mask.any():
            gu[mask] = a_rep[mask] @ w[e].T  # [..., 2*inter_dim] = [gate||up]
    return gu


def _interleave_to_split(tmp_out, tokens, topk, inter_dim):
    """gui_layout [g0..15,u0..15,g16..31,u16..31,...] -> [gate||up] (tokens,topk,2*inter_dim)."""
    t2 = tmp_out.view(tokens, topk, inter_dim // 16, 2, 16)
    gate = t2[..., 0, :].reshape(tokens, topk, inter_dim)
    up   = t2[..., 1, :].reshape(tokens, topk, inter_dim)
    return torch.cat([gate, up], dim=-1)


def _run_kb(data, kb):
    return _launch_flydsl_stage1_splitk(
        data, tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K, k_batch=kb, act="silu",
    )


def _diff_stats(a, b, rtol=1e-2, atol=1e-2):
    af = a.float(); bf = b.float()
    d = (af - bf).abs()
    return d.max().item(), d.mean().item(), (d > (atol + rtol * bf.abs())).float().mean().item()


def main():
    print(f"# FlyDSL blockscale stage1 split-K vs FP32 reference (unactivated partials)")
    print(f"# shape: model_dim={MODEL_DIM}, inter_dim={INTER_DIM}, "
          f"experts={EXPERTS}, topk={TOPK}")
    print(f"# tile={TILE_M}x{TILE_N}x{TILE_K}, repeats={N_REPEATS}")
    print(f"# tol: rtol=1e-2 atol=1e-2 (bf16 has ~3.9e-3 relative error inherent)")
    print()
    print(f"#  {'M':>4s} {'kb':>2s}   "
          f"{'vs_ref_max':>10s} {'vs_ref_mean':>11s} {'vs_ref_err%':>11s}   "
          f"{'nondet':>9s}")

    self_consistency = {}  # M -> (kb2_vs_kb4_max, mean)

    for M in M_VALUES:
        data = _prepare_data(M, MODEL_DIM, INTER_DIM, EXPERTS, TOPK)
        tokens = data["tokens"]
        inter_dim = data["inter_dim"]
        topk = data["topk"]

        ref = _torch_stage1_partials_ref(data)  # f32, [gate||up]

        kb_split = {}
        for kb in KB_VALUES:
            runs_tmp = [_run_kb(data, kb).clone() for _ in range(N_REPEATS)]
            # rearrange to [gate||up] to align with ref
            runs_split = [_interleave_to_split(t, tokens, topk, inter_dim) for t in runs_tmp]

            vs_ref_max, vs_ref_mean, vs_ref_err = _diff_stats(runs_split[0], ref)
            nondet = 0.0
            for i in range(len(runs_split)):
                for j in range(i + 1, len(runs_split)):
                    nondet = max(nondet, (runs_split[i] - runs_split[j]).abs().max().item())

            print(f"   {M:>4d} {kb:>2d}   "
                  f"{vs_ref_max:10.3e} {vs_ref_mean:11.3e} {100*vs_ref_err:10.3f}%   "
                  f"{nondet:9.3e}", flush=True)
            kb_split[kb] = runs_split[0]

        c = (kb_split[2] - kb_split[4]).abs()
        self_consistency[M] = (c.max().item(), c.mean().item())

        del data, ref, kb_split, runs_tmp, runs_split
        torch.cuda.empty_cache()

    print()
    print(f"# === split-K self-consistency (kb=2 vs kb=4) ===")
    print(f"# {'M':>5s}  {'kb2_vs_kb4_max':>15s} {'mean':>12s}")
    for M, (mx, mn) in self_consistency.items():
        print(f"  {M:>5d}  {mx:15.3e} {mn:12.3e}")

    print()
    print("# Interpretation:")
    print("#   vs_ref err is QUANT noise + atomic precision drift combined.")
    print("#   For bf16 stage1 partials, ~3.9e-3 relative error is the floor.")
    print("#   If kb=2 and kb=4 agree closely (kb2_vs_kb4_max ~ nondet), split-K")
    print("#   is internally consistent and atomic precision is fine.")
    print("#   If kb2_vs_kb4_max >> nondet, the kb-dependent drift points at an")
    print("#   accumulation-precision bug — case for moving to f32 atomic.")


if __name__ == "__main__":
    main()

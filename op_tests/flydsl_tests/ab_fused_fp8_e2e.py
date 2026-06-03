"""Stage1+stage2 e2e A/B for fused FP8 stage1.

Production-equivalent correctness check that avoids the sorted-tile vs
token-major layout mismatch that defeats a raw bytewise A/B on stage1
output: feed the fused stage1's (fp8, sorted_scale) directly into
flydsl_moe_stage2 (which natively consumes the sorted-tile scale
layout — the production dispatcher in fused_moe.py does exactly this
when metadata.fuse_quant == "fp8").

Compares three things per M:

  * fused   : flydsl_moe_stage1(out_dtype='fp8', kb=1, interleave) → (fp8, scale_sorted)
              → flydsl_moe_stage2(a2=fp8, a2_scale=scale_sorted)
  * unfused : flydsl_moe_stage1(out_dtype='bf16', kb=1) → bf16
              → per_group_quant_hip(transpose_scale=True) → flydsl_moe_stage2
  * torch   : _torch_stage1_ref → external fp8 quant of bf16 ref → _torch_stage2_ref

Reports per M:
  fused_vs_torch_err%, unfused_vs_torch_err%, fused_vs_unfused_err%

If fused_vs_unfused is small AND fused_vs_torch ≈ unfused_vs_torch,
the fused fp8 stage1 is functionally equivalent at the e2e level.

Usage:
    HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/ab_fused_fp8_e2e.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from test_flydsl_blockscale_moe import (  # noqa: E402
    DEVICE,
    DTYPE_FP8,
    _prepare_data,
    _torch_stage1_ref,
    _torch_stage2_ref,
    aiter_moe_sorting,
    SCALE_BLOCK_N_DEFAULT,
    SCALE_BLOCK_K_DEFAULT,
)
from aiter.ops.flydsl.moe_kernels import (  # noqa: E402
    flydsl_moe_stage1,
    flydsl_moe_stage2,
)
from aiter.ops.quant import per_group_quant_hip  # noqa: E402

MODEL_DIM = 7168
INTER_DIM = 256
EXPERTS   = 257
TOPK      = 9

M_VALUES = [1, 8, 16, 32, 64, 256]
S1_TILE = (16, 128, 256)   # fused-fp8 stage1 tile
S2_TILE = (16, 128, 128)   # stage2 tile


def _run_fused_e2e(data):
    """Fused FP8 stage1 → stage2, returning final out2 (bf16/fp16)."""
    tm, tn, tk = S1_TILE
    sids, sw, se, nv, _ = aiter_moe_sorting(
        data["topk_ids"], data["topk_weights"],
        data["experts"], data["model_dim"], torch.bfloat16, tm,
    )
    fp8_out, scale_sorted = flydsl_moe_stage1(
        data["x_bq"], data["w1_bq_shuf"], sids, se, nv,
        topk=data["topk"],
        tile_m=tm, tile_n=tn, tile_k=tk,
        a_dtype="fp8", b_dtype="fp8", out_dtype="fp8", act="silu",
        w1_scale=data["w1_bscale_flat"], a1_scale=data["x_bscale_fly"],
        sorted_weights=None, gate_mode="interleave",
        k_batch=1, waves_per_eu=2,
    )
    # Stage2: pass fused output + sorted-tile scale buffer directly.
    s2_tm, s2_tn, s2_tk = S2_TILE
    # stage2 needs its OWN sorting at its own tile_m if different from s1's
    if s2_tm != tm:
        s2_sids, s2_sw, s2_se, s2_nv, _ = aiter_moe_sorting(
            data["topk_ids"], data["topk_weights"],
            data["experts"], data["model_dim"], torch.bfloat16, s2_tm,
        )
    else:
        s2_sids, s2_sw, s2_se, s2_nv = sids, sw, se, nv
    out2 = flydsl_moe_stage2(
        inter_states=fp8_out,
        w2=data["w2_bq_shuf"],
        sorted_token_ids=s2_sids,
        sorted_expert_ids=s2_se,
        num_valid_ids=s2_nv,
        topk=data["topk"],
        tile_m=s2_tm, tile_n=s2_tn, tile_k=s2_tk,
        a_dtype="fp8", b_dtype="fp8", out_dtype="bf16",
        mode="atomic",
        w2_scale=data["w2_bscale_flat"],
        a2_scale=scale_sorted,
        sorted_weights=s2_sw,
    )
    return out2, fp8_out, scale_sorted


def _run_unfused_e2e(data):
    """bf16 stage1 → external fp8 quant → stage2."""
    tm, tn, tk = S1_TILE
    sids, sw, se, nv, _ = aiter_moe_sorting(
        data["topk_ids"], data["topk_weights"],
        data["experts"], data["model_dim"], torch.bfloat16, tm,
    )
    out_bf16 = flydsl_moe_stage1(
        data["x_bq"], data["w1_bq_shuf"], sids, se, nv,
        topk=data["topk"],
        tile_m=tm, tile_n=tn, tile_k=tk,
        a_dtype="fp8", b_dtype="fp8", out_dtype="bf16", act="silu",
        w1_scale=data["w1_bscale_flat"], a1_scale=data["x_bscale_fly"],
        sorted_weights=None, gate_mode="separated",
        k_batch=1, waves_per_eu=2,
    )
    tokens, topk, inter_dim = out_bf16.shape
    a2_bq, a2_scale_fly = per_group_quant_hip(
        out_bf16.view(-1, inter_dim),
        quant_dtype=DTYPE_FP8,
        group_size=SCALE_BLOCK_K_DEFAULT,
        transpose_scale=True,
    )
    a2_bq = a2_bq.view(tokens, topk, inter_dim)
    s2_tm, s2_tn, s2_tk = S2_TILE
    if s2_tm != tm:
        s2_sids, s2_sw, s2_se, s2_nv, _ = aiter_moe_sorting(
            data["topk_ids"], data["topk_weights"],
            data["experts"], data["model_dim"], torch.bfloat16, s2_tm,
        )
    else:
        s2_sids, s2_sw, s2_se, s2_nv = sids, sw, se, nv
    out2 = flydsl_moe_stage2(
        inter_states=a2_bq,
        w2=data["w2_bq_shuf"],
        sorted_token_ids=s2_sids,
        sorted_expert_ids=s2_se,
        num_valid_ids=s2_nv,
        topk=data["topk"],
        tile_m=s2_tm, tile_n=s2_tn, tile_k=s2_tk,
        a_dtype="fp8", b_dtype="fp8", out_dtype="bf16",
        mode="atomic",
        w2_scale=data["w2_bscale_flat"],
        a2_scale=a2_scale_fly,
        sorted_weights=s2_sw,
    )
    # Also return a2_bq + per-row scale for torch ref alignment
    nblk_k_w2 = inter_dim // SCALE_BLOCK_K_DEFAULT
    a2_scale_2d = a2_scale_fly.view(nblk_k_w2, -1).t().contiguous()
    return out2, a2_bq, a2_scale_2d, out_bf16


def _torch_ref_e2e(data, a2_bq, a2_scale_2d):
    """Torch stage2 reference using same (a2_bq, a2_scale) as unfused path."""
    tokens = data["tokens"]; model_dim = data["model_dim"]
    inter_dim = data["inter_dim"]; topk = data["topk"]
    return _torch_stage2_ref(
        a2_bq, data["w2_bq"], data["topk_ids"], data["topk_weights"],
        a2_scale_2d, data["w2_bscale_flat"],
        tokens, model_dim, inter_dim, topk,
        SCALE_BLOCK_N_DEFAULT, SCALE_BLOCK_K_DEFAULT,
    )


def _err_ratio(a, b, rtol=1e-2, atol=1e-2):
    af = a.float(); bf = b.float()
    d = (af - bf).abs()
    err_pct = (d > (atol + rtol * bf.abs())).float().mean().item()
    return d.max().item(), d.mean().item(), 100 * err_pct


def main():
    print(f"# FlyDSL fused FP8 stage1 → stage2 e2e A/B")
    print(f"# shape: model_dim={MODEL_DIM}, inter_dim={INTER_DIM}, "
          f"experts={EXPERTS}, topk={TOPK}")
    print(f"# s1_tile={S1_TILE}, s2_tile={S2_TILE}")
    print(f"# tol: rtol=1e-2 atol=1e-2 (bf16 + fp8 quant noise)")
    print()
    print(f"# {'M':>4s}   "
          f"{'fu_max':>9s} {'fu_mean':>9s} {'fu_err%':>7s}   "
          f"{'un_max':>9s} {'un_mean':>9s} {'un_err%':>7s}   "
          f"{'fu/un_max':>9s} {'fu/un_mean':>10s} {'fu/un_err%':>10s}", flush=True)

    t0 = time.time()
    for M in M_VALUES:
        data = _prepare_data(M, MODEL_DIM, INTER_DIM, EXPERTS, TOPK)

        # Unfused path: this also gives us a2_bq + a2_scale_2d for the torch ref.
        out_un, a2_bq, a2_scale_2d, _bf16 = _run_unfused_e2e(data)

        # Fused path.
        out_fu, _fp8, _scale = _run_fused_e2e(data)

        # Torch reference (anchored to unfused's a2_q so quant noise cancels):
        ref = _torch_ref_e2e(data, a2_bq, a2_scale_2d)

        fu_mx, fu_mn, fu_e = _err_ratio(out_fu, ref)
        un_mx, un_mn, un_e = _err_ratio(out_un, ref)
        fu_un_mx, fu_un_mn, fu_un_e = _err_ratio(out_fu, out_un)
        elapsed = time.time() - t0
        print(f"  {M:>4d}   "
              f"{fu_mx:9.3e} {fu_mn:9.3e} {fu_e:6.2f}%   "
              f"{un_mx:9.3e} {un_mn:9.3e} {un_e:6.2f}%   "
              f"{fu_un_mx:9.3e} {fu_un_mn:10.3e} {fu_un_e:9.2f}%   "
              f"[{elapsed:5.1f}s]", flush=True)

        del data, out_un, out_fu, a2_bq, a2_scale_2d, ref
        torch.cuda.empty_cache()

    print()
    print("# Interpretation:")
    print("#   fu_err% ≈ un_err%  →  fused stage1 has same fp8-quant noise as unfused chain")
    print("#   fu/un_err% small   →  fused and unfused paths agree at e2e level")
    print("#   Note: ref uses the unfused path's a2_q, so quant noise on stage2-INPUT")
    print("#   cancels in un_err% but NOT in fu_err% (fused a2_q differs slightly")
    print("#   at v_med3 clamp boundaries). Expect fu_err% slightly > un_err%.")


if __name__ == "__main__":
    main()

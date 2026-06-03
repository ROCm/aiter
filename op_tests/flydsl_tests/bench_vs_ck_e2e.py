"""FlyDSL fused-FP8 stage1+stage2 vs CK 2-stage at DSR1 TP=8 small M.

Reports per-stage and total us. Both paths include an externally-computed
requant from stage1 output → fp8 for stage2 input (factored out of timing
since both pay it identically in production-equivalent dispatch).

Usage:
    HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/bench_vs_ck_e2e.py
"""
from __future__ import annotations

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from test_flydsl_blockscale_moe import (  # noqa: E402
    DEVICE,
    SCALE_BLOCK_N_DEFAULT,
    SCALE_BLOCK_K_DEFAULT,
    _launch_flydsl_stage2,
    _prepare_data,
    aiter_moe_sorting,
)
from aiter import ActivationType, QuantType  # noqa: E402
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1  # noqa: E402
from aiter.ops.moe_op import ck_moe_stage1_fwd, ck_moe_stage2_fwd  # noqa: E402
from aiter.test_common import run_perftest  # noqa: E402

MODEL_DIM = int(os.environ.get("BENCH_MODEL_DIM", 7168))
INTER_DIM = int(os.environ.get("BENCH_INTER_DIM", 256))
EXPERTS   = int(os.environ.get("BENCH_EXPERTS", 257))
TOPK      = int(os.environ.get("BENCH_TOPK", 9))

M_VALUES = [1, 2, 4, 8, 16, 32, 64]
FLY_TILE = (16, 128, 256, 2)         # stage1 fused fp8 tile
FLY_S2_TILE = (16, 128, 128, 2)      # stage2 tile (matches existing tests)
CK_BLOCK_M = 32


def _bench_fly_s1(data, tile):
    tm, tn, tk, wpe = tile
    sorted_ids, sorted_w, sorted_e, num_valid, _ = aiter_moe_sorting(
        data["topk_ids"], data["topk_weights"],
        data["experts"], data["model_dim"], torch.bfloat16, tm,
    )

    def _run():
        return flydsl_moe_stage1(
            data["x_bq"], data["w1_bq_shuf"],
            sorted_ids, sorted_e, num_valid,
            topk=data["topk"],
            tile_m=tm, tile_n=tn, tile_k=tk,
            a_dtype="fp8", b_dtype="fp8", out_dtype="fp8",
            act="silu",
            w1_scale=data["w1_bscale_flat"], a1_scale=data["x_bscale_fly"],
            sorted_weights=None,
            gate_mode="interleave",
            k_batch=1,
            waves_per_eu=wpe,
        )

    res = _run()
    torch.cuda.synchronize()
    _, us = run_perftest(_run, num_iters=20, num_warmup=5)
    return us, res, sorted_ids, sorted_w, sorted_e, num_valid


def _bench_fly_s2(data, out1_bf16_proxy, *, tile, sorted_ids, sorted_w,
                  sorted_e, num_valid):
    tm, tn, tk, wpe = tile
    out2, _a2bq, _a2sc, us = _launch_flydsl_stage2(
        data, out1_bf16_proxy,
        tile_m=tm, tile_n=tn, tile_k=tk, waves_per_eu=wpe,
        sorted_ids=sorted_ids, sorted_w=sorted_w,
        sorted_e=sorted_e, num_valid=num_valid,
    )
    return us, out2


def _bench_ck_s1(data, block_m=CK_BLOCK_M):
    tokens = data["tokens"]; model_dim = data["model_dim"]
    inter_dim = data["inter_dim"]; experts = data["experts"]; topk = data["topk"]
    blk_n, blk_k = SCALE_BLOCK_N_DEFAULT, SCALE_BLOCK_K_DEFAULT

    sorted_ids, sorted_w, sorted_e, num_valid, _ = aiter_moe_sorting(
        data["topk_ids"], data["topk_weights"],
        experts, model_dim, torch.bfloat16, block_m,
    )
    out_ck = torch.zeros(tokens, topk, inter_dim, device=DEVICE, dtype=torch.float16)
    nblk_n_w1 = (2 * inter_dim) // blk_n
    nblk_k_w1 = model_dim // blk_k
    w1_scale_ck = data["w1_bscale_flat"].view(experts, nblk_n_w1, nblk_k_w1).contiguous()

    def _run():
        ck_moe_stage1_fwd(
            hidden_states=data["a1_bq"], w1=data["w1_bq_shuf"], w2=data["w2_bq_shuf"],
            sorted_token_ids=sorted_ids, sorted_expert_ids=sorted_e,
            num_valid_ids=num_valid, out=out_ck, topk=topk, kernelName="",
            w1_scale=w1_scale_ck, a1_scale=data["a1_bscale"],
            block_m=block_m, sorted_weights=None,
            quant_type=QuantType.per_1x128, activation=ActivationType.Silu,
            dst_type=torch.float16,
        )

    _run(); torch.cuda.synchronize()
    _, us = run_perftest(_run, num_iters=20, num_warmup=5)
    return us, out_ck, sorted_ids, sorted_w, sorted_e, num_valid


def _bench_ck_s2(data, out1, sorted_ids, sorted_w, sorted_e, num_valid,
                 block_m=CK_BLOCK_M):
    from aiter.ops.quant import per_group_quant_hip
    from test_flydsl_blockscale_moe import DTYPE_FP8 as _FP8
    tokens = data["tokens"]; model_dim = data["model_dim"]
    inter_dim = data["inter_dim"]; experts = data["experts"]; topk = data["topk"]
    blk_n, blk_k = SCALE_BLOCK_N_DEFAULT, SCALE_BLOCK_K_DEFAULT

    # Requantize stage1 output to fp8 (external; both paths pay this in real use)
    a2_bq, a2_scale_fly = per_group_quant_hip(
        out1.to(torch.bfloat16).view(-1, inter_dim),
        quant_dtype=_FP8,
        group_size=blk_k,
        transpose_scale=True,
    )
    nblk_k_w2 = inter_dim // blk_k
    a2_scale_2d = a2_scale_fly.view(nblk_k_w2, -1).t().contiguous()
    a2_for_ck = a2_bq.view(tokens, topk, inter_dim)
    nblk_n_w2 = model_dim // blk_n
    nblk_k_w2_w = inter_dim // blk_k
    w2_scale_ck = data["w2_bscale_flat"].view(experts, nblk_n_w2, nblk_k_w2_w).contiguous()
    out_s2 = torch.zeros(tokens, model_dim, device=DEVICE, dtype=torch.float16)

    def _run():
        out_s2.zero_()
        ck_moe_stage2_fwd(
            inter_states=a2_for_ck, w1=data["w1_bq_shuf"], w2=data["w2_bq_shuf"],
            sorted_token_ids=sorted_ids, sorted_expert_ids=sorted_e,
            num_valid_ids=num_valid, out=out_s2, topk=topk, kernelName="",
            w2_scale=w2_scale_ck, a2_scale=a2_scale_2d,
            block_m=block_m, sorted_weights=sorted_w,
            quant_type=QuantType.per_1x128, activation=ActivationType.Silu,
        )

    _run(); torch.cuda.synchronize()
    _, us = run_perftest(_run, num_iters=20, num_warmup=5)
    return us


def main():
    print(f"# FlyDSL fused FP8 (s1+s2) vs CK 2-stage")
    print(f"# shape: model_dim={MODEL_DIM}, inter_dim={INTER_DIM}, "
          f"experts={EXPERTS}, topk={TOPK}")
    print()
    hdr = (f"# {'M':>4s}  {'fly_s1':>7s} {'fly_s2':>7s} {'fly_tot':>7s}  "
           f"{'ck_s1':>7s} {'ck_s2':>7s} {'ck_tot':>7s}  "
           f"{'gap_tot':>8s}  {'fly/ck':>7s}")
    print(hdr)

    for M in M_VALUES:
        data = _prepare_data(M, MODEL_DIM, INTER_DIM, EXPERTS, TOPK)

        # FlyDSL stage1 (fused fp8 kb=1)
        fly_s1_us, (fp8_out, scale), sids, sw, se, nv = _bench_fly_s1(data, FLY_TILE)

        # FlyDSL stage2: feed bf16 proxy (existing path requants from bf16; the
        # production fused-fp8 chain would skip this requant, so this is a
        # *worst-case* number for FlyDSL stage2.)
        # Build a bf16 proxy by dequantizing the fp8 stage1 output.
        # For perf-only A/B we just feed zeros of the right shape (stage2 time
        # is dominated by GEMM, not by input values).
        tokens = data["tokens"]; inter_dim = data["inter_dim"]; topk = data["topk"]
        bf16_proxy = torch.zeros(tokens, topk, inter_dim,
                                 device=DEVICE, dtype=torch.bfloat16)
        fly_s2_us, _ = _bench_fly_s2(
            data, bf16_proxy, tile=FLY_S2_TILE,
            sorted_ids=sids, sorted_w=sw, sorted_e=se, num_valid=nv,
        )

        # CK
        ck_s1_us, out_ck_s1, csids, csw, cse, cnv = _bench_ck_s1(data)
        ck_s2_us = _bench_ck_s2(data, out_ck_s1, csids, csw, cse, cnv)

        fly_tot = fly_s1_us + fly_s2_us
        ck_tot = ck_s1_us + ck_s2_us
        gap = fly_tot - ck_tot
        ratio = fly_tot / ck_tot if ck_tot > 0 else float('nan')
        print(f"  {M:>4d}  {fly_s1_us:7.2f} {fly_s2_us:7.2f} {fly_tot:7.2f}  "
              f"{ck_s1_us:7.2f} {ck_s2_us:7.2f} {ck_tot:7.2f}  "
              f"{gap:+8.2f}  {ratio:6.2f}x", flush=True)

        del data, fp8_out, scale, bf16_proxy, out_ck_s1
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

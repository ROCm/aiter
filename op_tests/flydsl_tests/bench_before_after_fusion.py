"""Stage1 perf: before fusion vs after fusion vs CK at DSR1 TP=8 small M.

  * pre  = out_dtype='fp8', k_batch=2 — legacy path: splitK bf16 GEMM
           + silu_and_mul_fq + zero-fill (3 GPU kernels per call)
  * post = out_dtype='fp8', k_batch=1 — Phase B fused: single kernel
           does GEMM + silu+mul + fp8 quant + tiled scale-byte write
  * ck   = ck_moe_stage1_fwd (single CK kernel writing fp16)

Reports per-mode us and kernel count.
"""
from __future__ import annotations

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from test_flydsl_blockscale_moe import (  # noqa: E402
    DEVICE, SCALE_BLOCK_N_DEFAULT, SCALE_BLOCK_K_DEFAULT,
    _prepare_data, aiter_moe_sorting,
)
from aiter import ActivationType, QuantType  # noqa: E402
from aiter.ops.flydsl.moe_kernels import (  # noqa: E402
    flydsl_moe_stage1,
    _get_compiled_silu_fused,
    _run_compiled,
    _ptr_arg_safe,
)
from aiter.ops.moe_op import ck_moe_stage1_fwd  # noqa: E402
from aiter.test_common import run_perftest  # noqa: E402

MODEL_DIM, INTER_DIM, EXPERTS, TOPK = 7168, 256, 257, 9
M_VALUES = [1, 4, 8, 16, 32]
TILE = (16, 128, 256, 2)
CK_BLOCK_M = 32


def _fly_run_fn(data, tile, k_batch):
    tm, tn, tk, wpe = tile
    sids, sw, se, nv, _ = aiter_moe_sorting(
        data["topk_ids"], data["topk_weights"],
        data["experts"], data["model_dim"], torch.bfloat16, tm,
    )
    def _run():
        flydsl_moe_stage1(
            data["x_bq"], data["w1_bq_shuf"], sids, se, nv,
            topk=data["topk"],
            tile_m=tm, tile_n=tn, tile_k=tk,
            a_dtype="fp8", b_dtype="fp8", out_dtype="fp8", act="silu",
            w1_scale=data["w1_bscale_flat"], a1_scale=data["x_bscale_fly"],
            sorted_weights=None, gate_mode="interleave",
            k_batch=k_batch, waves_per_eu=wpe,
        )
    return _run


def _fly_bf16_kb1_run_fn(data, tile):
    """kb=1 bf16 output with Phase A silu+mul fusion (no fp8 quant).

    This is the GEMM-and-Phase-A baseline at kb=1; subtracting from fused-fp8
    kb=1 isolates the Phase-B fp8-quant epilog cost.
    """
    tm, tn, tk, wpe = tile
    sids, sw, se, nv, _ = aiter_moe_sorting(
        data["topk_ids"], data["topk_weights"],
        data["experts"], data["model_dim"], torch.bfloat16, tm,
    )
    def _run():
        flydsl_moe_stage1(
            data["x_bq"], data["w1_bq_shuf"], sids, se, nv,
            topk=data["topk"],
            tile_m=tm, tile_n=tn, tile_k=tk,
            a_dtype="fp8", b_dtype="fp8", out_dtype="bf16", act="silu",
            w1_scale=data["w1_bscale_flat"], a1_scale=data["x_bscale_fly"],
            sorted_weights=None, gate_mode="interleave",
            k_batch=1, waves_per_eu=wpe,
        )
    return _run


def _ck_run_fn(data, block_m=CK_BLOCK_M):
    tokens = data["tokens"]; model_dim = data["model_dim"]
    inter_dim = data["inter_dim"]; experts = data["experts"]; topk = data["topk"]
    blk_n, blk_k = SCALE_BLOCK_N_DEFAULT, SCALE_BLOCK_K_DEFAULT
    sids, sw, se, nv, _ = aiter_moe_sorting(
        data["topk_ids"], data["topk_weights"], experts, model_dim,
        torch.bfloat16, block_m,
    )
    out = torch.zeros(tokens, topk, inter_dim, device=DEVICE, dtype=torch.float16)
    nblk_n_w1 = (2 * inter_dim) // blk_n; nblk_k_w1 = model_dim // blk_k
    w1_sc = data["w1_bscale_flat"].view(experts, nblk_n_w1, nblk_k_w1).contiguous()
    def _run():
        ck_moe_stage1_fwd(
            hidden_states=data["a1_bq"], w1=data["w1_bq_shuf"], w2=data["w2_bq_shuf"],
            sorted_token_ids=sids, sorted_expert_ids=se, num_valid_ids=nv,
            out=out, topk=topk, kernelName="", w1_scale=w1_sc,
            a1_scale=data["a1_bscale"], block_m=block_m, sorted_weights=None,
            quant_type=QuantType.per_1x128, activation=ActivationType.Silu,
            dst_type=torch.float16,
        )
    return _run


def _time(run_fn):
    run_fn(); torch.cuda.synchronize()
    _, us = run_perftest(run_fn, num_iters=20, num_warmup=5)
    return us


def _kcount(run_fn, n=10):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as p:
        for _ in range(n): run_fn()
        torch.cuda.synchronize()
    ks = [(e.self_device_time_total/n, e.key) for e in p.key_averages()
          if e.self_device_time_total > 0]
    ks.sort(key=lambda r: -r[0])
    return ks


def main():
    print(f"# stage1 before-fusion vs after-fusion vs CK, DSR1 TP=8")
    print(f"# shape: model_dim={MODEL_DIM}, inter_dim={INTER_DIM}, "
          f"experts={EXPERTS}, topk={TOPK}")
    print()
    print(f"# pre   = fp8 kb=2 legacy chain (bf16 GEMM + silu_and_mul_fq + fill)")
    print(f"# bf16k1 = bf16 kb=1 Phase A (silu+mul fused, no quant) — GEMM cost baseline")
    print(f"# post  = fp8 kb=1 Phase B fused (silu+mul+fp8+scale all in epilog)")
    print()
    print(f"# {'M':>3s}  {'pre_us':>7s} {'pre_k':>3s}  "
          f"{'bf16k1':>7s}  {'post_us':>7s} {'post_k':>4s}  "
          f"{'ck_us':>7s}  {'B_cost':>7s}  {'fuse_save':>9s}")

    for M in M_VALUES:
        data = _prepare_data(M, MODEL_DIM, INTER_DIM, EXPERTS, TOPK)
        pre_run    = _fly_run_fn(data, TILE, k_batch=2)
        bf16k1_run = _fly_bf16_kb1_run_fn(data, TILE)
        post_run   = _fly_run_fn(data, TILE, k_batch=1)
        ck_run     = _ck_run_fn(data)

        pre_us    = _time(pre_run)
        bf16k1_us = _time(bf16k1_run)
        post_us   = _time(post_run)
        ck_us     = _time(ck_run)

        pre_k  = len(_kcount(pre_run))
        post_k = len(_kcount(post_run))

        b_cost    = post_us - bf16k1_us           # Phase-B quant epilog overhead
        fuse_save = pre_us - post_us              # net benefit of fusion vs legacy
        print(f"  {M:>3d}  {pre_us:7.2f} {pre_k:>3d}  "
              f"{bf16k1_us:7.2f}  {post_us:7.2f} {post_k:>4d}  "
              f"{ck_us:7.2f}  {b_cost:+7.2f}  {fuse_save:+9.2f}", flush=True)

        if M == 1:
            print(f"  # pre (kb=2) kernels at M=1:")
            for ts, name in _kcount(pre_run):
                print(f"  #   {ts:6.2f}us  {name[:80]}")
            print(f"  # bf16k1 kernels at M=1:")
            for ts, name in _kcount(bf16k1_run):
                print(f"  #   {ts:6.2f}us  {name[:80]}")
            print(f"  # post (kb=1 fused fp8) kernels at M=1:")
            for ts, name in _kcount(post_run):
                print(f"  #   {ts:6.2f}us  {name[:80]}")

        del data, pre_run, bf16k1_run, post_run, ck_run
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

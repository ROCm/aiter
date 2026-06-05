"""Isolated FlyDSL gemm1/gemm2 config sweep for the mxfp4-sort hybrid (M=256, BM=32)."""
import torch
import aiter
from aiter import dtypes
from aiter.test_common import run_perftest
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
from aiter.utility.fp4_utils import moe_mxfp4_sort
from bench_up_moe_v1 import KIMI, build_weights, build_inputs

device = torch.device("cuda")
shape = KIMI
fly_w, mx_w = build_weights(shape, device)
M = 256
BM = 32
hidden, topk_ids, topk_weight = build_inputs(shape, M, device)
topk = topk_ids.shape[1]
NE, D_HIDDEN, D_INTER = shape.NE, shape.H, shape.INTER

# ---- mxfp4 sort + quant prologue (once) ----
active = min(NE, M * topk)
max_sorted = ((M * topk + active * (BM - 1) + BM - 1) // BM) * BM
def z(n, dt=dtypes.i32):
    return torch.empty((n,), device=device, dtype=dt)
sorted_token_ids = z(max_sorted); sorted_expert_ids = z(max_sorted // BM)
cumsum_tensor = z(1); reverse_sorted = z(M * topk)
sorted_weights = z(max_sorted, dtypes.fp32); masked_m = z(NE); m_indices = z(max_sorted)
a_quant = torch.empty((M, D_HIDDEN // 2), device=device, dtype=torch.uint8)
a_scale = torch.empty((M, D_HIDDEN // 32), device=device, dtype=torch.uint8)
e = torch.empty((0,), dtype=dtypes.bf16, device=device)
aiter.mxfp4_moe_sort(topk_ids=topk_ids, topk_weight=topk_weight,
    sorted_token_ids=sorted_token_ids, sorted_expert_ids=sorted_expert_ids,
    cumsum_tensor=cumsum_tensor, reverse_sorted=reverse_sorted, sorted_weights=sorted_weights,
    masked_m=masked_m, m_indices=m_indices, bf16_zero_out=e, bf16_zero_workspace=e,
    M_logical=M, NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, D_INTER=D_INTER, MB=BM, prologue=1)
aiter.mxfp4_moe_quant(a_input=hidden, a_quant=a_quant, a_scale=a_scale,
    bf16_zero_out=e, NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, MB=BM)
num_valid_ids = cumsum_tensor.repeat(2)
a1 = a_quant.view(dtypes.fp4x2)
a1_scale = moe_mxfp4_sort(a_scale.view(dtypes.fp8_e8m0).view(M, 1, -1),
    sorted_ids=sorted_token_ids, num_valid_ids=num_valid_ids, token_num=M, block_size=BM)
w1 = fly_w["w1"].view(dtypes.fp4x2); w1s = fly_w["w1_scale"].view(dtypes.fp8_e8m0)
w2 = fly_w["w2"].view(dtypes.fp4x2); w2s = fly_w["w2_scale"].view(dtypes.fp8_e8m0)


def g1(tile_n, wpe, bnt, kb):
    def fn():
        return flydsl_moe_stage1(a=a1, w1=w1, sorted_token_ids=sorted_token_ids,
            sorted_expert_ids=sorted_expert_ids, num_valid_ids=num_valid_ids, topk=topk,
            tile_m=BM, tile_n=tile_n, tile_k=256, a_dtype="fp4", b_dtype="fp4",
            out_dtype="fp4", w1_scale=w1s, a1_scale=a1_scale, sorted_weights=None,
            use_async_copy=True, k_batch=kb, waves_per_eu=wpe, b_nt=bnt,
            gate_mode="separated")
    return fn


print("== gemm1 (tile_m=32) sweep ==  [mxfp4 BM32 ref = 184us]")
best = None
for tile_n in (64, 128):
    for wpe in (2, 3, 4):
        for bnt in (0, 2):
            for kb in (1,):
                try:
                    out, us = run_perftest(g1(tile_n, wpe, bnt, kb), num_warmup=8, num_iters=40)
                    print(f"  t32x{tile_n} wpe{wpe} bnt{bnt} kb{kb}: {us:.1f} us")
                    if best is None or us < best[0]:
                        best = (us, tile_n, wpe, bnt, kb, out)
                except Exception as ex:
                    print(f"  t32x{tile_n} wpe{wpe} bnt{bnt} kb{kb}: FAIL {type(ex).__name__}")
print(f"  -> best gemm1: {best[0]:.1f} us @ t32x{best[1]} wpe{best[2]} bnt{best[3]} kb{best[4]}")

inter = best[5]
inter_q, inter_scale = (inter[0], inter[1]) if isinstance(inter, tuple) else (inter, None)


def g2(tile_n, bnt):
    out_buf = torch.zeros((M, D_HIDDEN), dtype=dtypes.bf16, device=device)
    def fn():
        flydsl_moe_stage2(inter_states=inter_q, w2=w2, sorted_token_ids=sorted_token_ids,
            sorted_expert_ids=sorted_expert_ids, num_valid_ids=num_valid_ids, topk=topk,
            out=out_buf, tile_m=BM, tile_n=tile_n, tile_k=256, a_dtype="fp4", b_dtype="fp4",
            out_dtype="bf16", mode="atomic", w2_scale=w2s, a2_scale=inter_scale,
            sorted_weights=sorted_weights, b_nt=bnt)
        return out_buf
    return fn


print("== gemm2 (tile_m=32) sweep ==  [mxfp4 BM32 ref = 102us]")
for tile_n in (128, 256):
    for bnt in (0, 2):
        try:
            _, us = run_perftest(g2(tile_n, bnt), num_warmup=8, num_iters=40)
            print(f"  t32x{tile_n} bnt{bnt}: {us:.1f} us")
        except Exception as ex:
            print(f"  t32x{tile_n} bnt{bnt}: FAIL {type(ex).__name__}")

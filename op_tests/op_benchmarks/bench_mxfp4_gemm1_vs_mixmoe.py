"""Perf: a8w4 gemm1 -- mxfp4 PORT vs the mixed_moe_gemm_2stage PIPELINE.

Both do the same stage1 work (fp8 act x fp4 weight gate/up GEMM + silu_mul + fp4
requant) at the kimi shape. The port consumes the threestage-sorted layout; the
reference flydsl_moe_stage1 consumes the CK moe_sorting layout. Interleave gate
mode, out_dtype=fp4. Run:
  python op_tests/op_benchmarks/bench_mxfp4_gemm1_vs_mixmoe.py
"""
import argparse
import csv
import importlib.util
import os

import torch
import aiter
from aiter import QuantType, dtypes
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.ops.quant import per_1x32_f8_scale_f8_quant
from aiter.utility.fp4_utils import moe_mxfp4_sort
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1

_TEST = os.path.join(os.path.dirname(__file__), "..", "test_mxfp4_flydsl_gemm1.py")
_spec = importlib.util.spec_from_file_location("_g1test", _TEST)
_t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t)


def _routing(NE, H, TOPK, M, device, seed=2):
    torch.manual_seed(seed + 1)
    hidden = torch.randn((M, H), dtype=dtypes.bf16, device=device) / 10
    g = torch.Generator(device=device).manual_seed(seed + 1)
    bias = torch.randn(NE - 1, generator=g, device=device) * 0.5
    scores = torch.randn(M, NE - 1, generator=g, device=device) + bias
    rw, rid = torch.topk(scores.softmax(-1), TOPK - 1, dim=-1)
    sid = torch.full((M, 1), NE - 1, device=device, dtype=rid.dtype)
    sw = torch.ones((M, 1), device=device, dtype=rw.dtype)
    topk_ids = torch.cat([sid, rid], 1).to(torch.int32)
    topk_weight = torch.cat([sw, rw], 1).to(torch.float32)
    return hidden, topk_ids, topk_weight


def _build_port(NE, H, INTER, TOPK, M, device, BM, use_nt, iq, seed=2):
    tq = aiter.get_torch_quant(QuantType.per_1x32)
    torch.manual_seed(seed)
    w1 = torch.randn((NE, 2 * INTER, H), dtype=dtypes.bf16, device=device) / 10
    w1q, w1s = tq(w1, quant_dtype=dtypes.fp4x2)
    w1u8 = shuffle_weight_a16w4(w1q, 16, True).view(torch.uint8)
    w1_scale = shuffle_scale_a16w4(w1s, NE, True).view(torch.uint8)
    hidden, topk_ids, topk_weight = _routing(NE, H, TOPK, M, device, seed)
    active = min(NE, M * TOPK)
    max_sorted = ((M * TOPK + active * (BM - 1) + BM - 1) // BM) * BM
    sti, sei, cumsum, mind = _t._torch_threestage_sort(
        topk_ids, topk_weight, M, NE, TOPK, BM, max_sorted
    )
    aq, asc = per_1x32_f8_scale_f8_quant(
        hidden, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    aq = aq.view(torch.uint8).view(M, H).contiguous()
    asc = asc.view(torch.uint8).view(M, H // 32).contiguous()
    assh = _t._torch_a_scale_sorted_shuffled(asc, sti, cumsum, max_sorted, H, BM=BM)
    isq = torch.zeros((max_sorted, INTER // 2), device=device, dtype=torch.uint8)
    isc = INTER // 32
    isr = (((max_sorted * (2 * INTER // 64) * 4) + isc - 1) // isc + 31) // 32 * 32
    iss = torch.zeros((isr, isc), device=device, dtype=torch.uint8)

    def call():
        flydsl_mxfp4_gemm1(
            a_quant=aq, a_scale_sorted_shuffled=assh, w1_u8=w1u8,
            w1_scale_u8=w1_scale, sorted_expert_ids=sei, cumsum_tensor=cumsum,
            m_indices=mind, inter_sorted_quant=isq, inter_sorted_shuffled_scale=iss,
            hidden_states=hidden, n_tokens=M, BM=BM, use_nt=use_nt,
            inline_quant=iq, NE=NE, D_HIDDEN=H, D_INTER=INTER, topk=TOPK,
            interleave=True, a_dtype="fp8",
        )

    return call


def _build_ref(NE, H, INTER, TOPK, M, device, BM, tile_n, tile_k, seed=2):
    BM = max(int(BM), 32)  # mixed_moe stage1 min tile_m is 32 (port BM16 -> 32 here)
    tq = aiter.get_torch_quant(QuantType.per_1x32)
    torch.manual_seed(seed)
    w1 = torch.randn((NE, 2 * INTER, H), dtype=dtypes.bf16, device=device) / 10
    w1q, w1s = tq(w1, quant_dtype=dtypes.fp4x2)
    w1u8 = shuffle_weight_a16w4(w1q, 16, True)
    w1_scale = shuffle_scale_a16w4(w1s, NE, True)
    hidden, topk_ids, topk_weight = _routing(NE, H, TOPK, M, device, seed)
    a1q, a1s = per_1x32_f8_scale_f8_quant(
        hidden, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    sorted_ids, sorted_w, sorted_eids, num_valid, _ = moe_sorting(
        topk_ids, topk_weight, NE, H, dtypes.bf16, BM
    )
    a1s_sort = moe_mxfp4_sort(
        a1s[:M, :].view(M, 1, -1), sorted_ids=sorted_ids,
        num_valid_ids=num_valid, token_num=M, block_size=BM,
    )

    def call():
        flydsl_moe_stage1(
            a=a1q, w1=w1u8, sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_eids, num_valid_ids=num_valid, topk=TOPK,
            tile_m=BM, tile_n=tile_n, tile_k=tile_k,
            a_dtype="fp8", b_dtype="fp4", out_dtype="fp4", act="silu",
            w1_scale=w1_scale, a1_scale=a1s_sort, gate_mode="interleave",
        )

    return call


def _time(call, iters, warmup, repeats=3):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            call()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters * 1e3)
    return best


def _variant_from_kname(kname):
    u = kname.upper()
    if "BM16" in u and "INLINEQUANT" in u:
        return 16, True, True
    if "BM32" in u and "NT" in u:
        return 32, True, False
    if "BM32" in u and "CACHED" in u:
        return 32, False, False
    if "BM128" in u:
        return 128, False, False
    raise ValueError(f"cannot map variant from kernelName: {kname!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument(
        "--csv",
        default="/root/aiter/aiter/configs/model_configs/kimik2_5_mxfp4_tuned_fmoe.csv",
    )
    ap.add_argument("--max-token", type=int, default=8192)
    ap.add_argument("--tile-n", type=int, default=256)
    ap.add_argument("--tile-k", type=int, default=256)
    args = ap.parse_args()
    device = torch.device("cuda")

    NE, INTER, TOPK = 385, 512, 9
    rows = list(csv.DictReader(open(args.csv)))
    print(f"# a8w4 gemm1: PORT vs mixed_moe stage1 (interleave, out=fp4) NE={NE} "
          f"INTER={INTER} TOPK={TOPK}")
    print(f"{'M':>6} {'H':>5} {'variant':<14} {'port_us':>9} {'ref_us':>9} "
          f"{'port/ref':>9}")
    print("-" * 60)
    for r in rows:
        M = int(r["token"])
        H = int(r["model_dim"])
        if M > args.max_token:
            continue
        BM, use_nt, iq = _variant_from_kname(r["kernelName1"])
        vtag = f"bm{BM}_{'iq' if iq else ('nt' if use_nt else 'cached')}"
        port_us = ref_us = float("nan")
        try:
            port_us = _time(_build_port(NE, H, INTER, TOPK, M, device, BM, use_nt, iq),
                            args.iters, args.warmup)
        except Exception as e:
            print(f"{M:>6} {H:>5} {vtag:<14} PORT FAIL {str(e)[:40]}")
        try:
            ref_us = _time(_build_ref(NE, H, INTER, TOPK, M, device, BM,
                                      args.tile_n, args.tile_k),
                           args.iters, args.warmup)
        except Exception as e:
            print(f"{M:>6} {H:>5} {vtag:<14} REF FAIL {str(e)[:40]}")
        ratio = port_us / ref_us if ref_us == ref_us and ref_us else float("nan")
        print(f"{M:>6} {H:>5} {vtag:<14} {port_us:>9.2f} {ref_us:>9.2f} {ratio:>9.3f}")


if __name__ == "__main__":
    main()

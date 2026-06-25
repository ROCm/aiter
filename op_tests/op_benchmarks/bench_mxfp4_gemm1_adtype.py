"""Perf comparison: MXFP4 MoE gemm1 a4w4 (fp4 act) vs a8w4 (fp8 act).

Both use fp4 weights + the same per_1x32 e8m0 scales; only the activation dtype
(and its A-payload load/LDS/MFMA path) differs. Run:
  python op_tests/op_benchmarks/bench_mxfp4_gemm1_adtype.py
"""
import argparse
import csv
import importlib.util
import os

import torch
import aiter
from aiter import QuantType, dtypes
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.ops.quant import per_1x32_f8_scale_f8_quant
from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1

_TEST = os.path.join(os.path.dirname(__file__), "..", "test_mxfp4_flydsl_gemm1.py")
_spec = importlib.util.spec_from_file_location("_g1test", _TEST)
_t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t)


def _build(NE, H, INTER, TOPK, M, a_dtype, device, seed=2,
           BM=32, use_nt=True, inline_quant=False, interleave=True):
    topk = TOPK
    tq = aiter.get_torch_quant(QuantType.per_1x32)
    torch.manual_seed(seed)
    w1 = torch.randn((NE, 2 * INTER, H), dtype=dtypes.bf16, device=device) / 10
    w1q, w1s = tq(w1, quant_dtype=dtypes.fp4x2)
    w1u8 = shuffle_weight_a16w4(w1q, 16, interleave).view(torch.uint8)
    w1_scale = shuffle_scale_a16w4(w1s, NE, interleave).view(torch.uint8)

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

    active = min(NE, M * topk)
    max_sorted = ((M * topk + active * (BM - 1) + BM - 1) // BM) * BM
    sti, sei, cumsum, mind = _t._torch_threestage_sort(
        topk_ids, topk_weight, M, NE, topk, BM, max_sorted
    )
    if a_dtype == "fp8":
        aq, asc = per_1x32_f8_scale_f8_quant(
            hidden, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
        aq = aq.view(torch.uint8).view(M, H).contiguous()
    else:
        aq, asc = tq(hidden, quant_dtype=dtypes.fp4x2)
        aq = aq.view(torch.uint8).view(M, H // 2).contiguous()
    asc = asc.view(torch.uint8).view(M, H // 32).contiguous()
    assh = _t._torch_a_scale_sorted_shuffled(asc, sti, cumsum, max_sorted, H, BM=BM)

    isq = torch.zeros((max_sorted, INTER // 2), device=device, dtype=torch.uint8)
    isc = INTER // 32
    N_OUT = 2 * INTER
    isr = (((max_sorted * (N_OUT // 64) * 4) + isc - 1) // isc + 31) // 32 * 32
    iss = torch.zeros((isr, isc), device=device, dtype=torch.uint8)

    def call():
        flydsl_mxfp4_gemm1(
            a_quant=aq, a_scale_sorted_shuffled=assh, w1_u8=w1u8,
            w1_scale_u8=w1_scale, sorted_expert_ids=sei, cumsum_tensor=cumsum,
            m_indices=mind, inter_sorted_quant=isq, inter_sorted_shuffled_scale=iss,
            hidden_states=hidden, n_tokens=M, BM=BM, use_nt=use_nt,
            inline_quant=inline_quant, NE=NE, D_HIDDEN=H, D_INTER=INTER,
            topk=topk, interleave=interleave, a_dtype=a_dtype,
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
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument(
        "--csv",
        default="/root/aiter/aiter/configs/model_configs/kimik2_5_mxfp4_tuned_fmoe.csv",
    )
    ap.add_argument("--max-token", type=int, default=8192)
    args = ap.parse_args()
    device = torch.device("cuda")

    NE, INTER, TOPK = 385, 512, 9  # kimi
    rows = list(csv.DictReader(open(args.csv)))
    print(f"# kimi shape NE={NE} INTER={INTER} TOPK={TOPK}, tuned CSV={args.csv}")
    print(
        f"{'M':>6} {'H':>5} {'variant':<16} "
        f"{'a4w4_us':>9} {'a8w4_us':>9} {'a8/a4':>7}"
    )
    print("-" * 64)
    for r in rows:
        M = int(r["token"])
        H = int(r["model_dim"])
        if M > args.max_token:
            continue
        BM, use_nt, iq = _variant_from_kname(r["kernelName1"])
        vtag = f"bm{BM}_{'iq' if iq else ('nt' if use_nt else 'cached')}"
        a4 = _build(NE, H, INTER, TOPK, M, "fp4", device,
                    BM=BM, use_nt=use_nt, inline_quant=iq)
        a4_us = _time(a4, args.iters, args.warmup)
        a8 = _build(NE, H, INTER, TOPK, M, "fp8", device,
                    BM=BM, use_nt=use_nt, inline_quant=iq)
        a8_us = _time(a8, args.iters, args.warmup)
        ratio = a8_us / a4_us if a4_us else float("nan")
        print(f"{M:>6} {H:>5} {vtag:<16} {a4_us:>9.2f} {a8_us:>9.2f} {ratio:>7.3f}")


if __name__ == "__main__":
    main()

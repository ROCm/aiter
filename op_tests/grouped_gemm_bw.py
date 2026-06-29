#!/usr/bin/env python3
"""Grouped-GEMM (MoE) memory-traffic / bandwidth calculator.

Pass the measured per-stage kernel times with ``--us1`` (gemm1) and ``--us2``
(gemm2); this prints the input/output byte volumes per stage and each stage's
achieved bandwidth. Defaults mirror test6.sh (T=64, topk=6, E=256, K=4096,
I=2048) which runs with balanced routing (AITER_MOE_EXPERT_BALANCE=true), so by
default every expert is active.

Example:
    python op_tests/grouped_gemm_bw.py --us1 120 --us2 80
    python op_tests/grouped_gemm_bw.py --us1 120 --us2 80 --data-format a4w4
    python op_tests/grouped_gemm_bw.py --us1 120 --us2 80 --active-experts 128
"""
import argparse


def fmt_bytes(b: float) -> str:
    return f"{b/1e6:.3f} MB"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--us1", type=float, required=True,
                   help="measured gemm1 (stage1) time in microseconds")
    p.add_argument("--us2", type=float, required=True,
                   help="measured gemm2 (stage2) time in microseconds")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--model-dim", type=int, default=4096, help="K")
    p.add_argument("--inter-dim", type=int, default=2048, help="I")
    p.add_argument("--data-format", choices=("a4w4", "a8w4"), default="a8w4")
    p.add_argument("--active-experts", type=int, default=-1,
                   help="experts that received >=1 token. -1 => balanced "
                        "routing assumption min(E, M) (test6.sh sets "
                        "AITER_MOE_EXPERT_BALANCE=true); pass the kernel's real "
                        "count for an exact number.")
    p.add_argument("--scale-block", type=int, default=32,
                   help="elements per e8m0 scale byte")
    args = p.parse_args()

    T, topk, E = args.tokens, args.topk, args.experts
    K, I = args.model_dim, args.inter_dim
    M = T * topk                      # routed token-expert pairs
    sb = args.scale_block

    # dtype byte sizes
    a_bytes = 1.0 if args.data_format == "a8w4" else 0.5   # fp8 vs fp4
    w_bytes = 0.5                                           # mxfp4 weight
    out_bytes = 2.0                                         # bf16
    scale_bytes = 1.0                                       # e8m0, per `sb` elems

    # activated experts
    if args.active_experts >= 0:
        active = min(args.active_experts, E)
        active_note = "given"
    else:
        # test6.sh uses balanced routing: M assignments spread round-robin,
        # so distinct active experts = min(E, M).
        active = min(E, M)
        active_note = "balanced routing min(E, M)"

    # ---- weights (loaded once per activated expert) ----
    w1_total = active * ((2 * I * K) * w_bytes + (2 * I * K / sb) * scale_bytes)
    w2_total = active * ((K * I) * w_bytes + (K * I / sb) * scale_bytes)

    # ---- gemm1 (stage1): read A + W1, write the (M, I) intermediate ----
    s1_a = M * K * a_bytes + (M * K / sb) * scale_bytes            # stage1 A in
    s1_out = M * I * out_bytes                                     # stage1 out bf16
    g1_total = w1_total + s1_a + s1_out
    bw1 = g1_total / (args.us1 * 1e-6) / 1e12  # TB/s

    # ---- gemm2 (stage2): read A2 + W2, write the reduced (T, K) output ----
    s2_a = M * I * a_bytes + (M * I / sb) * scale_bytes           # stage2 A2 in
    s2_out = T * K * out_bytes                                     # final out bf16
    g2_total = w2_total + s2_a + s2_out
    bw2 = g2_total / (args.us2 * 1e-6) / 1e12  # TB/s

    total = g1_total + g2_total
    bw = total / ((args.us1 + args.us2) * 1e-6) / 1e12

    print(f"config: T={T} topk={topk} M={M} E={E} K={K} I={I} "
          f"fmt={args.data_format}")
    print(f"active experts: {active:.1f} ({active_note})\n")

    print(f"--- gemm1 (stage1) ---  us={args.us1:.2f}")
    print(f"  W1 weights ({active:.1f} experts): {fmt_bytes(w1_total)}")
    print(f"  A  in  (M,K)            : {fmt_bytes(s1_a)}")
    print(f"  out    (M,I) bf16       : {fmt_bytes(s1_out)}")
    print(f"  gemm1 total             : {fmt_bytes(g1_total)}")
    print(f"  gemm1 bandwidth         : {bw1:.2f} TB/s\n")

    print(f"--- gemm2 (stage2) ---  us={args.us2:.2f}")
    print(f"  W2 weights ({active:.1f} experts): {fmt_bytes(w2_total)}")
    print(f"  A2 in  (M,I)            : {fmt_bytes(s2_a)}")
    print(f"  out    (T,K) bf16       : {fmt_bytes(s2_out)}")
    print(f"  gemm2 total             : {fmt_bytes(g2_total)}")
    print(f"  gemm2 bandwidth         : {bw2:.2f} TB/s\n")

    print(f"combined traffic : {fmt_bytes(total)}")
    print(f"combined time    : {args.us1 + args.us2:.2f} us")
    print(f"combined bandwidth: {bw:.2f} TB/s")


if __name__ == "__main__":
    main()

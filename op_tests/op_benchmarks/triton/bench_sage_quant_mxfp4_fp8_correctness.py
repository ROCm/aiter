# SPDX-License-Identifier: MIT
# Correctness + performance comparison for sage_quant_mxfp4 variants:
#
#   ref    : sage_quant_mxfp4(bf16 q/k)         — ground truth
#   upcast : sage_quant_mxfp4(fp8→bf16 q/k)     — fp8 via host upcast
#   fused  : sage_quant_mxfp4_fp8_input(fp8 q/k) — fp8 via on-chip widening
#
# Correctness is measured on dequantized outputs (packed fp4 → bf16 via
# upcast_from_mxfp) because comparing raw uint8 bytes is too strict.
#
# Metrics per tensor (Q, K, V):
#   cosine_sim  — mean per-row cosine similarity (key for attention logits)
#   sign_agree  — fraction of elements with matching sign
#   max_err     — max absolute elementwise error
#   scale_match — fraction of e8m0 scale bytes that are identical
#
# Usage:
#   python bench_sage_quant_mxfp4_fp8_correctness.py
#   python bench_sage_quant_mxfp4_fp8_correctness.py --layout bhsd

import argparse
import torch
import triton
import pandas as pd

import aiter
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    sage_quant_mxfp4,
    create_hadamard_matrix,
)
from aiter.ops.triton.quant.sage_attention_quant_fp8_input_wrapper import (
    sage_quant_mxfp4_fp8_input,
)
from aiter.ops.triton.moe.quant_moe import upcast_from_mxfp

FP8_TYPE = aiter.dtypes.fp8
FP8_MAX = torch.finfo(FP8_TYPE).max
DEVICE = "cuda"

SHAPES = [
    # (batch, seqlen, heads, head_dim)
    (1, 1024, 16, 128),
    (1, 4096, 16, 128),
    (2, 2048, 16, 128),
    (4, 1024, 16, 128),
]

BENCH_WARMUP = 25
BENCH_REP = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tensors(B, S, H, D, layout):
    shape = (B, S, H, D) if layout == "bshd" else (B, H, S, D)
    q_bf16 = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    k_bf16 = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    v_bf16 = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    q_fp8 = q_bf16.to(FP8_TYPE)
    k_fp8 = k_bf16.to(FP8_TYPE)
    return q_bf16, k_bf16, v_bf16, q_fp8, k_fp8


def quant_kwargs(layout, R):
    return dict(FP8_TYPE=FP8_TYPE, FP8_MAX=FP8_MAX, BLKQ=128, BLKK=64,
                layout=layout, R=R, BLOCK_R=128)


def dequant(packed, scale, layout):
    """Dequantize packed fp4x2 + e8m0 scale → bf16, axis = last dim of logical tensor."""
    # packed shape: (*batch_heads_seq, D//2)  scale: (*batch_heads_seq, D//32)
    # quantized axis is the last one (-1), which is D
    return upcast_from_mxfp(packed, scale, torch.bfloat16, axis=-1)


def cosine_sim(a, b):
    """Mean per-row cosine similarity. Both tensors flattened to 2D."""
    a = a.reshape(-1, a.shape[-1]).float()
    b = b.reshape(-1, b.shape[-1]).float()
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a_n * b_n).sum(dim=-1).mean().item()


def sign_agree(a, b):
    """Fraction of elements with matching sign (ignoring zeros)."""
    a = a.float().flatten()
    b = b.float().flatten()
    nz = (a != 0) & (b != 0)
    if nz.sum() == 0:
        return float("nan")
    return ((torch.sign(a[nz]) == torch.sign(b[nz])).float().mean()).item()


def max_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def scale_match(sa, sb):
    """Fraction of e8m0 scale bytes that are identical."""
    return (sa == sb).float().mean().item()


def metrics(q_fp4, q_sc, k_fp4, k_sc, v_fp8, v_sc,
            ref_q_fp4, ref_q_sc, ref_k_fp4, ref_k_sc, ref_v_fp8, ref_v_sc,
            layout):
    q_dq = dequant(q_fp4, q_sc, layout)
    k_dq = dequant(k_fp4, k_sc, layout)
    rq_dq = dequant(ref_q_fp4, ref_q_sc, layout)
    rk_dq = dequant(ref_k_fp4, ref_k_sc, layout)

    v_dq = v_fp8.to(torch.float32) * v_sc.unsqueeze(1 if layout == "bshd" else 2)
    rv_dq = ref_v_fp8.to(torch.float32) * ref_v_sc.unsqueeze(1 if layout == "bshd" else 2)

    return {
        "Q cos": round(cosine_sim(q_dq, rq_dq), 6),
        "Q sign%": round(sign_agree(q_dq, rq_dq), 6),
        "Q max_err": round(max_err(q_dq, rq_dq), 6),
        "Q sc_match": round(scale_match(q_sc, ref_q_sc), 6),
        "K cos": round(cosine_sim(k_dq, rk_dq), 6),
        "K sign%": round(sign_agree(k_dq, rk_dq), 6),
        "K max_err": round(max_err(k_dq, rk_dq), 6),
        "K sc_match": round(scale_match(k_sc, ref_k_sc), 6),
        "V cos": round(cosine_sim(v_dq, rv_dq), 6),
        "V sign%": round(sign_agree(v_dq, rv_dq), 6),
        "V max_err": round(max_err(v_dq, rv_dq), 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=["bshd", "bhsd"], default="bshd")
    args = parser.parse_args()

    corr_rows = []
    perf_rows = []

    for B, S, H, D in SHAPES:
        print(f"\n=== B={B} S={S} H={H} D={D} ===")
        q_bf16, k_bf16, v_bf16, q_fp8, k_fp8 = make_tensors(B, S, H, D, args.layout)
        R = create_hadamard_matrix(128, device=DEVICE, dtype=torch.bfloat16) / (128 ** 0.5)
        kw = quant_kwargs(args.layout, R)

        # Run all three variants
        ref = sage_quant_mxfp4(q_bf16, k_bf16, v_bf16, **kw)
        ref_q_fp4, ref_q_sc, ref_k_fp4, ref_k_sc, ref_v_fp8, ref_v_sc, _ = ref

        up = sage_quant_mxfp4(q_fp8.to(torch.bfloat16), k_fp8.to(torch.bfloat16), v_bf16, **kw)
        up_q_fp4, up_q_sc, up_k_fp4, up_k_sc, up_v_fp8, up_v_sc, _ = up

        fused = sage_quant_mxfp4_fp8_input(q_fp8, k_fp8, v_bf16, **kw)
        fu_q_fp4, fu_q_sc, fu_k_fp4, fu_k_sc, fu_v_fp8, fu_v_sc, _ = fused

        # Correctness: upcast vs ref (fp8 precision loss)
        m_up_vs_ref = metrics(
            up_q_fp4, up_q_sc, up_k_fp4, up_k_sc, up_v_fp8, up_v_sc,
            ref_q_fp4, ref_q_sc, ref_k_fp4, ref_k_sc, ref_v_fp8, ref_v_sc,
            args.layout,
        )
        # Correctness: fused vs ref (fp8 precision loss + any kernel difference)
        m_fu_vs_ref = metrics(
            fu_q_fp4, fu_q_sc, fu_k_fp4, fu_k_sc, fu_v_fp8, fu_v_sc,
            ref_q_fp4, ref_q_sc, ref_k_fp4, ref_k_sc, ref_v_fp8, ref_v_sc,
            args.layout,
        )
        # Correctness: fused vs upcast (same fp8 inputs — isolates kernel difference)
        m_fu_vs_up = metrics(
            fu_q_fp4, fu_q_sc, fu_k_fp4, fu_k_sc, fu_v_fp8, fu_v_sc,
            up_q_fp4, up_q_sc, up_k_fp4, up_k_sc, up_v_fp8, up_v_sc,
            args.layout,
        )

        for label, m in [("upcast vs ref", m_up_vs_ref),
                         ("fused  vs ref", m_fu_vs_ref),
                         ("fused  vs upcast", m_fu_vs_up)]:
            print(f"  {label:20s}  Q_cos={m['Q cos']:.4f}  K_cos={m['K cos']:.4f}"
                  f"  Q_sign={m['Q sign%']:.4f}  K_sign={m['K sign%']:.4f}"
                  f"  Q_sc={m['Q sc_match']:.4f}  K_sc={m['K sc_match']:.4f}")
            corr_rows.append({"B": B, "S": S, "H": H, "D": D, "comparison": label, **m})

        # Performance
        def run_ref():
            sage_quant_mxfp4(q_bf16, k_bf16, v_bf16, **kw)

        def run_upcast():
            sage_quant_mxfp4(q_fp8.to(torch.bfloat16), k_fp8.to(torch.bfloat16), v_bf16, **kw)

        def run_fused():
            sage_quant_mxfp4_fp8_input(q_fp8, k_fp8, v_bf16, **kw)

        ms_ref    = triton.testing.do_bench(run_ref,    warmup=BENCH_WARMUP, rep=BENCH_REP)
        ms_upcast = triton.testing.do_bench(run_upcast, warmup=BENCH_WARMUP, rep=BENCH_REP)
        ms_fused  = triton.testing.do_bench(run_fused,  warmup=BENCH_WARMUP, rep=BENCH_REP)

        print(f"  perf: bf16={ms_ref:.3f}ms  fp8+upcast={ms_upcast:.3f}ms ({ms_ref/ms_upcast:.3f}x)"
              f"  fp8+fused={ms_fused:.3f}ms ({ms_ref/ms_fused:.3f}x)")

        perf_rows.append({
            "B": B, "S": S, "H": H, "D": D,
            "bf16 (ms)": round(ms_ref, 3),
            "fp8+upcast (ms)": round(ms_upcast, 3),
            "fp8+fused (ms)": round(ms_fused, 3),
            "upcast speedup": round(ms_ref / ms_upcast, 3),
            "fused speedup": round(ms_ref / ms_fused, 3),
        })

    print("\n\n=== CORRECTNESS SUMMARY ===")
    df_corr = pd.DataFrame(corr_rows)
    print(df_corr.to_markdown(index=False))

    print("\n\n=== PERFORMANCE SUMMARY ===")
    df_perf = pd.DataFrame(perf_rows)
    print(df_perf.to_markdown(index=False))


if __name__ == "__main__":
    main()

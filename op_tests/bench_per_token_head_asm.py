"""Companion bench: FP8 per-tensor scaling via the gfx942/gfx950 ASM kernel.

This calls `aiter.fmha_v3_fwd` directly (the ASM forward kernel) to give an
upper-bound reference for FP8 attention performance on the same shape grid as
`bench_per_token_head.py`.

Caveats vs bench_per_token_head.py:
  - PER-TENSOR descale (shape (1,)) — NOT per-token. Optional per-(batch,head)
    via env BENCH_ASM_DESCALE=per_bh.
  - DENSE KV layout (no paging, no block_table, no `page_size` knob).
  - No varlen (single fixed seqlen per batch row).

Run:
    python op_tests/bench_per_token_head_asm.py
    BENCH_SEQS=1024,16384 python op_tests/bench_per_token_head_asm.py
    BENCH_ASM_DESCALE=per_bh python op_tests/bench_per_token_head_asm.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiter
from aiter import dtypes


# ----- shape grid (mirrors bench_per_token_head.py) -----
_DEFAULT_SEQS = (1024, 16384, 32768, 65536, 131072)
_seqs_env = os.environ.get("BENCH_SEQS")
SEQS = (
    tuple(int(x) for x in _seqs_env.split(",") if x.strip())
    if _seqs_env
    else _DEFAULT_SEQS
)
BATCHES = tuple(
    int(x) for x in os.environ.get("BENCH_BATCHES", "2,4,6,8").split(",") if x.strip()
)
NHQ, NHK, HD = 8, 1, 128
CAUSAL = True
DESCALE_MODE = os.environ.get("BENCH_ASM_DESCALE", "per_tensor")  # or "per_bh"

WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))
ITERS = int(os.environ.get("BENCH_ITERS", "10"))


# ----- H20 reference (same numbers as bench_per_token_head.py) -----
H20_REF = {
    (2,   1024): 146.2, (4,   1024): 169.4, (6,   1024): 186.7, (8,   1024): 188.7,
    (2,  16384): 266.5, (4,  16384): 269.8, (6,  16384): 270.1, (8,  16384): 268.0,
    (2,  32768): 269.6, (4,  32768): 272.3, (6,  32768): 272.5, (8,  32768): 270.3,
    (2,  65536): 270.8, (4,  65536): 273.6, (6,  65536): 273.6, (8,  65536): 271.5,
    (2, 131072): 272.8, (4, 131072): 274.2, (6, 131072): 272.1, (8, 131072): 272.1,
}
H20_LAT_MS = {
    (2,   1024):   0.029, (4,   1024):   0.051, (6,   1024):   0.069, (8,   1024):   0.091,
    (2,  16384):   4.13,  (4,  16384):   8.15,  (6,  16384):  12.21,  (8,  16384):  16.41,
    (2,  32768):  16.31,  (4,  32768):  32.30,  (6,  32768):  48.43,  (8,  32768):  65.08,
    (2,  65536):  64.97,  (4,  65536): 128.61,  (6,  65536): 192.89,  (8,  65536): 259.22,
    (2, 131072): 257.92,  (4, 131072): 513.27,  (6, 131072): 775.93,  (8, 131072): 1034.50,
}


def make_inputs(b, s, nhq, nhk, hd, descale_mode):
    """Dense [B,S,H,D] fp8 tensors + descales matching ASM kernel constraints."""
    dev = "cuda"
    q_bf = torch.randn(b, s, nhq, hd, device=dev, dtype=torch.bfloat16)
    k_bf = torch.randn(b, s, nhk, hd, device=dev, dtype=torch.bfloat16)
    v_bf = torch.randn(b, s, nhk, hd, device=dev, dtype=torch.bfloat16)
    # quant via amax → fp8 e4m3
    def q_to_fp8(t):
        amax = t.abs().amax().clamp(min=1e-6)
        scale = (amax / 240.0).to(torch.float32)         # fp8 e4m3 max ≈ 240
        return (t / scale).to(dtypes.fp8), scale
    q, qs_t = q_to_fp8(q_bf)
    k, ks_t = q_to_fp8(k_bf)
    v, vs_t = q_to_fp8(v_bf)

    if descale_mode == "per_tensor":
        qs = qs_t.reshape(1)
        ks = ks_t.reshape(1)
        vs = vs_t.reshape(1)
    elif descale_mode == "per_bh":
        # broadcast same scalar across (B, nhead_k)
        qs = qs_t.expand(b, nhk).contiguous()
        ks = ks_t.expand(b, nhk).contiguous()
        vs = vs_t.expand(b, nhk).contiguous()
    else:
        raise ValueError(f"unknown BENCH_ASM_DESCALE={descale_mode}")
    return q, k, v, qs, ks, vs


def run_asm(b, s, nhq, nhk, hd, causal, descale_mode):
    q, k, v, qs, ks, vs = make_inputs(b, s, nhq, nhk, hd, descale_mode)
    softmax_scale = hd ** -0.5
    win_l, win_r = (-1, 0) if causal else (-1, -1)

    def call():
        return aiter.fmha_v3_fwd(
            q, k, v,
            0.0,                      # dropout_p
            softmax_scale,
            causal,                   # is_causal
            win_l, win_r,
            False,                    # return_softmax_lse
            False,                    # return_dropout_randval
            1,                        # how_v3_bf16_cvt
            None,                     # out
            None,                     # bias
            None,                     # alibi_slopes
            qs, ks, vs,
            None,                     # gen
        )

    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        call()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / ITERS
    return ms * 1000.0  # us


def attn_flops(b, s_q, s_k, nhq, hd, causal):
    # 2x (S_q*S_k) for QK + 2x (S_q*S_k) for softmax*V == 4 * S_q * S_k * hd * nhq * b
    # halve when causal
    f = 4.0 * s_q * s_k * hd * nhq * b
    if causal:
        f *= 0.5
    return f


def main():
    print(f"[asm bench] descale={DESCALE_MODE} warmup={WARMUP} iters={ITERS}")
    print(
        f"{'batch':>5} {'seq':>6} {'nhq':>4} {'nhk':>4} {'hd':>4} | "
        f"{'ASM FP8 (per-tensor)':>26} | "
        f"{'H20 TFLOPS':>10} | {'H20 lat(ms)':>11} | {'MI/H20 TF':>9} | {'H20/MI lat':>11}"
    )
    print("-" * 140)
    for s in SEQS:
        for b in BATCHES:
            try:
                us = run_asm(b, s, NHQ, NHK, HD, CAUSAL, DESCALE_MODE)
                tflops = attn_flops(b, s, s, NHQ, HD, CAUSAL) / us / 1e6
                cell = f"{us:8.2f} us  {tflops:7.2f} TFLOPS"
            except Exception as e:
                msg = str(e).splitlines()[-1][:60]
                cell = f"FAIL: {msg:>20}"
                tflops = None
                us = None
            h20_tf = H20_REF.get((b, s))
            h20_ms = H20_LAT_MS.get((b, s))
            tf_ratio = (
                f"{tflops / h20_tf:>8.2f}x" if (tflops and h20_tf) else f"{'-':>9}"
            )
            lat_ratio = (
                f"{(h20_ms * 1000.0) / us:>10.2f}x" if (us and h20_ms) else f"{'-':>11}"
            )
            h20_tf_str = f"{h20_tf:>10.1f}" if h20_tf else f"{'-':>10}"
            h20_ms_str = f"{h20_ms:>11.3f}" if h20_ms else f"{'-':>11}"
            print(
                f"{b:>5} {s:>6} {NHQ:>4} {NHK:>4} {HD:>4} | {cell:>26} | "
                f"{h20_tf_str} | {h20_ms_str} | {tf_ratio} | {lat_ratio}"
            )


if __name__ == "__main__":
    main()

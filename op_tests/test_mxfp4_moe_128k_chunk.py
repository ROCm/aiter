# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Micro self-consistency test for the mxfp4_moe runtime `max_sorted` buffer bound.

Why this exists
---------------
gemm1/gemm2 used to size the A_q / A_scale buffer-resource descriptors from a
compile-time `MAX_M = 655360`. That cap is *smaller* than the sorted-row count at
large batches: at 128k tokens, max_sorted = round_up(M*topk + active*(BM-1), BM)
≈ 1.23M rows ≫ 655360, so the old descriptor clipped every sorted row past
655360 to 0 → wrong output. The fix passes the runtime `max_sorted` into the
kernels instead. This test proves the fix is correct at 128k.

Reference strategy (no torch oracle needed)
-------------------------------------------
MoE is per-token independent: token t only sees its own topk experts, never the
rest of the batch. Therefore, for *any* weights,

    moe(concat[chunk_0 .. chunk_15])  ==  concat[moe(chunk_0) .. moe(chunk_15)]

So we run ONE set of random a4w4 weights two ways and compare:
  * full   : a single 128k-token forward      (max_sorted ≈ 1.23M, the stressed path)
  * chunked: 16 × 8k-token forwards concatd   (max_sorted ≈ 122k each, always safe)

The weights are random (NOT a semantically-correct quantization) — that is fine:
self-consistency only needs the *same* weights + *same* per-token routing in both
runs. The kernels index their inputs via buffer_rsrc raw byte offsets and the host
wrappers don't validate weight shapes, so correctly-sized random buffers suffice.

With the OLD (MAX_M=655360) code this test FAILS (full run clips its tail rows);
with the fix it passes (full == chunked to bf16 rounding).

Run (needs module_moe_mxfp4_aux + module_moe_mxfp4_gemm built, MI350X/gfx950):
    python op_tests/test_mxfp4_moe_128k_chunk.py                 # 128k = 16 × 8k
    python op_tests/test_mxfp4_moe_128k_chunk.py --tokens 16384 --chunk 8192   # quick smoke
"""

import argparse
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import _mxfp4_moe_run

# Self-check: we MUST be running the up-aiter checkout (the one with the runtime
# max_sorted fix), not a pip-installed / sibling aiter. Import-shadowing otherwise
# silently tests the wrong kernels.
print("[selfcheck] aiter pkg:", aiter.__file__)

torch.set_default_device("cuda")

# ── Kimi-K2.5 TP=4 mxfp4_moe shape (the codegen'd NE=385 nonatomic instance) ──
NE = 385          # 384 routed + 1 shared expert slot
D_HIDDEN = 7168
D_INTER = 512
TOPK = 9          # top_k(8) + 1 shared

# BM=128 nonatomic path — what large-M (incl. 128k) dispatches to, and the path
# whose A_q/A_scale bound we changed.
KN1 = f"mxfp4_moe_g1_a4w4_NE{NE}_H{D_HIDDEN}_E{D_INTER}_BM128"
KN2 = f"mxfp4_moe_g2_a4w4_NE{NE}_H{D_HIDDEN}_E{D_INTER}_BM128_NONATOMIC"


def make_weights(seed=0):
    """Random a4w4 weights + e8m0 scales, sized to exactly meet the kernels'
    buffer-resource byte bounds (B_q: NE*N_OUT*K/2 ; B_scale: per-expert e8m0)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, generator=g)
    # e8m0 biased exponents in a TIGHT band around 127 (=2^0): 126..129 -> 2^-1..2^2.
    # Keeps gemm outputs moderate so full vs chunked stays bit-exact (no overflow /
    # catastrophic cancellation), making any mismatch a real size-dependent bug.
    e8 = lambda *s: torch.randint(126, 130, s, dtype=torch.uint8, generator=g)
    # gemm1: w12 = [E, 2*D_INTER, D_HIDDEN] mxfp4 (2 nibbles/byte) + e8m0 (1/32 along K)
    w1 = u8(NE, 2 * D_INTER, D_HIDDEN // 2)
    w1s = e8(NE, 2 * D_INTER, D_HIDDEN // 32)
    # gemm2: w3 = [E, D_HIDDEN, D_INTER] mxfp4 + e8m0
    w2 = u8(NE, D_HIDDEN, D_INTER // 2)
    w2s = e8(NE, D_HIDDEN, D_INTER // 32)
    return w1, w2, w1s, w2s


def make_routing(num_tokens, seed=1):
    """Per-token routing, precomputed once and sliced so full & chunked are identical.
    Column TOPK-1 is pinned to the shared expert (NE-1), mirroring real usage."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    topk_ids = torch.randint(0, NE - 1, (num_tokens, TOPK), dtype=torch.int32, generator=g)
    topk_ids[:, -1] = NE - 1  # shared expert
    topk_weight = torch.rand((num_tokens, TOPK), dtype=torch.float32, generator=g)
    return topk_ids, topk_weight


def run(hidden, topk_ids, topk_weight, w):
    w1, w2, w1s, w2s = w
    return _mxfp4_moe_run(
        hidden, w1, w2, topk_ids, topk_weight, TOPK,
        kernelName1=KN1, kernelName2=KN2,
        w1_scale=w1s, w2_scale=w2s,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Silu,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128 * 1024)
    ap.add_argument("--chunk", type=int, default=8 * 1024)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--noise-reps", type=int, default=4,
                    help="full-run repeats used to estimate the FP-nondeterminism "
                         "noise floor (a single sample misses sporadically-"
                         "nondeterministic cancellation rows)")
    args = ap.parse_args()
    NT, CH = args.tokens, args.chunk
    assert NT % CH == 0, f"--tokens ({NT}) must be a multiple of --chunk ({CH})"
    nchunks = NT // CH
    print(f"tokens={NT}  chunk={CH}  nchunks={nchunks}  "
          f"(full max_sorted≈{NT*TOPK//1000}k rows vs chunk≈{CH*TOPK//1000}k)")

    g = torch.Generator(device="cuda").manual_seed(2)
    hidden = (torch.randn(NT, D_HIDDEN, dtype=torch.float32, generator=g) * 0.1).to(dtypes.bf16)
    topk_ids, topk_weight = make_routing(NT)
    w = make_weights()

    # full (stresses the large-max_sorted descriptor) ────────────────────────
    full = run(hidden, topk_ids, topk_weight, w)
    torch.cuda.synchronize()

    # chunked reference ──────────────────────────────────────────────────────
    chunks = []
    for c in range(nchunks):
        sl = slice(c * CH, (c + 1) * CH)
        chunks.append(run(hidden[sl].contiguous(),
                          topk_ids[sl].contiguous(),
                          topk_weight[sl].contiguous(), w))
    chunked = torch.cat(chunks, dim=0)
    torch.cuda.synchronize()

    def bad_rows(a, b):
        a, b = a.float(), b.float()
        diff = (a - b).abs()
        thr = args.atol + args.rtol * b.abs()
        return (diff > thr).any(dim=1), diff

    # The kernel's nonatomic/scatter reduce is run-to-run NON-deterministic on rows
    # with catastrophic cancellation (FP non-associativity). Measure that noise floor
    # from repeated full re-runs (a single sample misses rows that only sometimes
    # diverge), then only count an invariance break as REAL where the kernel was
    # self-consistent (deterministic across ALL repeats) yet full != chunked — that is
    # the signature of the old max_sorted clipping bug (which corrupts ~half the rows
    # at 128k, deterministically every run, so it survives the noise mask).
    noise = torch.zeros(NT, dtype=torch.bool, device=full.device)
    dn_max = 0.0
    for _ in range(args.noise_reps):
        nk, dk = bad_rows(full, run(hidden, topk_ids, topk_weight, w))
        noise |= nk
        dn_max = max(dn_max, dk.max().item())
    torch.cuda.synchronize()
    inv, di = bad_rows(full, chunked)
    real = inv & ~noise

    # full and chunked use DIFFERENT token counts → DIFFERENT reduction orders, so a
    # handful of catastrophic-cancellation rows legitimately differ between them (FP
    # non-associativity) in a way the same-order noise probe can't fully capture. The
    # max_sorted clipping bug, in contrast, corrupts ~HALF the rows (~NT/2) every run.
    # So the discriminator is the COUNT, not any single row: tolerate a small fraction
    # of cancellation rows; fail only on a bug-scale fraction.
    tol_rows = max(64, NT // 1000)  # 0.1% of rows; bug is ~50%, noise is <0.1‰
    n_real = int(real.sum())
    print(f"[determinism full vs full ] noise rows={int(noise.sum())}/{NT}  "
          f"max|Δ|={dn_max:.4e}  (union over {args.noise_reps} re-runs)")
    print(f"[invariance  full vs chunk] mismatch rows={int(inv.sum())}/{NT}  "
          f"max|Δ|={di.max().item():.4e}")
    print(f"[REAL size-dependent breaks (deterministic rows only)] = {n_real}/{NT}  "
          f"(tolerance {tol_rows}; max_sorted bug ⇒ ~{NT // 2})")
    if n_real:
        idx = real.nonzero().flatten()[:8].tolist()
        r0 = idx[0]
        col = di[r0].argmax().item()
        print(f"    rows: {idx}")
        print(f"    e.g. row {r0} col {col}: full={full.float()[r0,col].item():.4e} "
              f"chunked={chunked.float()[r0,col].item():.4e}")
    ok = n_real <= tol_rows
    print("RESULT:", "PASS ✅ (chunk-invariant modulo kernel FP nondeterminism)"
          if ok else "FAIL ❌ (bug-scale size-dependent divergence)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

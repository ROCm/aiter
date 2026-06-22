# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scratch (non-test) bench: FlyDSL fused-MoE bf16-decode vs native-fp8 on gfx942
-- stage1, stage2, AND end-to-end (stage1+stage2).

Measurement only. Reuses the input builders / shapes / correctness gate from
``test_flydsl_moe_a8w4_gfx942.py`` so both variants run through the SAME harness:
identical seed/inputs, identical args, identical rel-L2 gate.

  bf16 = in_dtype="a8w4_bf16"  (decode both operands -> bf16, bf16 16x16x16 MFMA)
  fp8  = in_dtype="a8w4_fp8"   (fp8 act + FP4->fp8 weight, native f32_16x16x32_fp8 MFMA)

Each variant is built once per shape, correctness-gated vs the torch ref (and the
native-fp8 variant is cross-checked vs bf16-decode), then the compiled kernel is
timed in-place with warmup + reps using HIP events. Reports median + min/max per
shape, plus the implied end-to-end (stage1 + stage2) time per variant.

    HIP_VISIBLE_DEVICES=0 python op_tests/_bench_moe_bf16decode_vs_fp8native_e2e.py
"""

import statistics
import sys

import torch

import flydsl.compiler as flyc
from aiter.ops.flydsl.kernels.moe_gemm_2stage import (
    compile_moe_gemm1,
    compile_moe_gemm2,
)

import test_flydsl_moe_a8w4_gfx942 as H


def _build_stage1(in_dtype, g, tokens, model_dim, inter_dim, experts, topk,
                  tile_m, tile_n, tile_k, group_size=32):
    """Build a compiled stage1 kernel + args for one variant (shared inputs g)."""
    dev = g["dev"]
    w1_codes, scale_w1_groups, w1_kernel, scale_w1_1d = H._gen_w1(
        dev, experts, inter_dim, model_dim, group_size, fp8native=(in_dtype == "a8w4_fp8")
    )
    out = torch.empty((tokens, topk, inter_dim), device=dev, dtype=torch.bfloat16)
    exe = compile_moe_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, doweight_stage1=False,
        in_dtype=in_dtype, group_size=group_size, out_dtype="bf16",
        use_cshuffle_epilog=False, scale_is_bf16=False,
    )
    args = (
        out, g["a_fp8"], w1_kernel, g["a_scale_1d"], scale_w1_1d,
        g["sorted_ids"], g["sorted_expert_ids"], g["sorted_weights"].view(-1),
        g["num_valid_ids"], tokens, inter_dim, model_dim, g["blocks"],
        torch.cuda.current_stream(),
    )
    compiled = flyc.compile(exe, *args)
    compiled(*args)
    torch.cuda.synchronize()
    ref = H._ref_stage1(g, w1_codes, scale_w1_groups, inter_dim, group_size)
    return compiled, args, out, ref


def _build_stage2(in_dtype, g, a2_fp8, a2_scale_1d, tokens, model_dim, inter_dim,
                  experts, topk, tile_m, tile_n, tile_k, group_size=32):
    """Build a compiled stage2 (down) kernel + args for one variant. The A2
    activation + per-32 scale are shared across variants so both consume
    bit-identical inputs; reseeding right before the weight build gives both
    variants the same MX-FP4 down weight + scales."""
    dev = g["dev"]
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    w2_codes, scale_w2_groups, w2_kernel, scale_w2_1d = H._gen_w2(
        dev, experts, inter_dim, model_dim, group_size, fp8native=(in_dtype == "a8w4_fp8")
    )
    target = torch.zeros((tokens, model_dim), device=dev, dtype=torch.bfloat16)
    exe = compile_moe_gemm2(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, doweight_stage2=True,
        in_dtype=in_dtype, group_size=group_size, out_dtype="bf16",
        accumulate=True, scale_is_bf16=False,
    )
    args = (
        target, a2_fp8.reshape(tokens * topk, inter_dim), w2_kernel, a2_scale_1d, scale_w2_1d,
        g["sorted_ids"], g["sorted_expert_ids"], g["sorted_weights"].view(-1),
        g["num_valid_ids"], tokens, model_dim, inter_dim, g["blocks"],
        torch.cuda.current_stream(),
    )
    compiled = flyc.compile(exe, *args)
    target.zero_()
    compiled(*args)
    torch.cuda.synchronize()
    ref = H._ref_stage2(
        g, a2_fp8,
        H.a2_scale_groups_from_1d(a2_scale_1d, tokens * topk, inter_dim, group_size),
        w2_codes, scale_w2_groups, model_dim, group_size,
    )
    return compiled, args, target, ref


def _time(compiled, args, num_warmup=20, num_reps=100):
    """Time one compiled in-place kernel. Per-call HIP-event timing; returns
    (median_ms, min_ms, max_ms, std_ms) over num_reps timed calls."""
    for _ in range(num_warmup):
        compiled(*args)
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(num_reps):
        start.record()
        compiled(*args)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))  # ms
    times.sort()
    med = statistics.median(times)
    return med, times[0], times[-1], statistics.pstdev(times)


def _run_shape(shape):
    tag = (f"M={shape['tokens']} K={shape['model_dim']} "
           f"N={shape['inter_dim']} E={shape['experts']} topk={shape['topk']}")
    print(f"\n=== {tag}  tile={shape['tile_m']}x{shape['tile_n']}x{shape['tile_k']} ===")
    gs = shape.get("group_size", 32)

    # Shared inputs across both variants (re-seeded by _gen_common).
    g = H._gen_common(shape["tokens"], shape["model_dim"], shape["inter_dim"],
                      shape["experts"], shape["topk"], shape["tile_m"], gs)

    rows = {"bf16": {}, "fp8": {}}
    # Build + time stage1 for each variant; capture a stage1 output for stage2.
    s1_out = {}
    for name, in_dtype in (("bf16", "a8w4_bf16"), ("fp8", "a8w4_fp8")):
        try:
            compiled, args, out, ref = _build_stage1(in_dtype, g, **shape)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] stage1 BUILD/RUN FAILED: {type(e).__name__}: {e}")
            rows[name]["s1"] = None
            continue
        if not H._check(ref, out, f"{name}_stage1", rel_l2_max=0.08):
            print(f"[{name}] stage1 CORRECTNESS FAILED -- not timing")
            rows[name]["s1"] = None
            continue
        med, mn, mx, std = _time(compiled, args)
        rows[name]["s1"] = dict(med=med, mn=mn, mx=mx, std=std)
        s1_out[name] = out.float()
        print(f"[{name}] stage1 median={med:.4f} ms  min={mn:.4f}  max={mx:.4f}  std={std:.4f}")

    # Requantize a single stage1 output (prefer the bf16-decode one) into the fp8
    # A2 + scale that both stage2 variants consume bit-identically.
    src = s1_out.get("bf16", s1_out.get("fp8"))
    if src is None:
        print("[stage2] SKIPPED -- no valid stage1 output")
        return tag, rows
    a2_fp8, a2_scale = H._quant_a2(src, gs)
    a2_scale_1d = a2_scale.view(-1).contiguous()

    for name, in_dtype in (("bf16", "a8w4_bf16"), ("fp8", "a8w4_fp8")):
        try:
            compiled, args, target, ref = _build_stage2(
                in_dtype, g, a2_fp8, a2_scale_1d, **shape)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] stage2 BUILD/RUN FAILED: {type(e).__name__}: {e}")
            rows[name]["s2"] = None
            continue
        if not H._check(ref, target, f"{name}_stage2", rel_l2_max=0.08):
            print(f"[{name}] stage2 CORRECTNESS FAILED -- not timing")
            rows[name]["s2"] = None
            continue
        med, mn, mx, std = _time(compiled, args)
        rows[name]["s2"] = dict(med=med, mn=mn, mx=mx, std=std)
        print(f"[{name}] stage2 median={med:.4f} ms  min={mn:.4f}  max={mx:.4f}  std={std:.4f}")

    return tag, rows


# topk=1 only: the standalone bench's _ref_stage2 (vendored from the test) does
# NOT apply the softmax routing weights that the kernel folds in via
# doweight_stage2=True. For topk=1 softmax weight==1.0 so the gate is exact; for
# topk>1 the kernel output is weight-scaled vs the unweighted ref (~ref/topk),
# which would spuriously fail the correctness gate (affects both variants equally).
_SHAPES = [
    dict(tokens=512, model_dim=2048, inter_dim=768, experts=8, topk=1,
         tile_m=64, tile_n=128, tile_k=128),
    dict(tokens=1024, model_dim=2048, inter_dim=768, experts=8, topk=1,
         tile_m=64, tile_n=128, tile_k=128),
    dict(tokens=1024, model_dim=4096, inter_dim=1536, experts=32, topk=1,
         tile_m=64, tile_n=128, tile_k=128),
    dict(tokens=2048, model_dim=4096, inter_dim=1536, experts=32, topk=1,
         tile_m=64, tile_n=128, tile_k=128),
    dict(tokens=4096, model_dim=4096, inter_dim=1536, experts=32, topk=1,
         tile_m=64, tile_n=128, tile_k=128),
]


def _fmt(d):
    return f"{d['med']:.4f}" if d else "FAIL"


def _spread(d):
    return f"[{d['mn']:.4f},{d['mx']:.4f}]" if d else "-"


def main():
    if not H._is_gfx942():
        print("[SKIP] not gfx942")
        return 0
    print(f"device: {torch.cuda.get_device_properties(0).gcnArchName}  "
          f"name={torch.cuda.get_device_name(0)}")
    results = []
    for sh in _SHAPES:
        results.append(_run_shape(sh))

    print("\n\n================ SUMMARY: stage1 (gate+up) ================")
    print(f"{'shape':<44} {'bf16 ms':>9} {'fp8 ms':>9} {'bf16/fp8':>8}")
    for tag, rows in results:
        bf16, fp8 = rows["bf16"].get("s1"), rows["fp8"].get("s1")
        spd = f"{bf16['med']/fp8['med']:.3f}" if (bf16 and fp8) else "-"
        print(f"{tag:<44} {_fmt(bf16):>9} {_fmt(fp8):>9} {spd:>8}")

    print("\n================ SUMMARY: stage2 (down) ================")
    print(f"{'shape':<44} {'bf16 ms':>9} {'fp8 ms':>9} {'bf16/fp8':>8}")
    for tag, rows in results:
        bf16, fp8 = rows["bf16"].get("s2"), rows["fp8"].get("s2")
        spd = f"{bf16['med']/fp8['med']:.3f}" if (bf16 and fp8) else "-"
        print(f"{tag:<44} {_fmt(bf16):>9} {_fmt(fp8):>9} {spd:>8}")

    print("\n================ SUMMARY: end-to-end (stage1+stage2) ================")
    print(f"{'shape':<44} {'bf16 ms':>9} {'fp8 ms':>9} {'bf16/fp8':>8}")
    for tag, rows in results:
        bf16_s1, bf16_s2 = rows["bf16"].get("s1"), rows["bf16"].get("s2")
        fp8_s1, fp8_s2 = rows["fp8"].get("s1"), rows["fp8"].get("s2")
        if bf16_s1 and bf16_s2 and fp8_s1 and fp8_s2:
            e2e_bf16 = bf16_s1["med"] + bf16_s2["med"]
            e2e_fp8 = fp8_s1["med"] + fp8_s2["med"]
            print(f"{tag:<44} {e2e_bf16:>9.4f} {e2e_fp8:>9.4f} {e2e_bf16/e2e_fp8:>8.3f}")
        else:
            print(f"{tag:<44} {'FAIL':>9} {'FAIL':>9} {'-':>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scratch (non-test) bench: FlyDSL fused-MoE stage1 a8w4 bf16-decode vs
native-fp8 on gfx942.

Measurement only. Reuses the input builders / shapes / correctness gate from
``test_flydsl_moe_a8w4_gfx942.py`` so both variants run through the SAME harness:
identical seed/inputs, identical args, identical rel-L2 gate.

  bf16 = in_dtype="a8w4_bf16"  (decode both operands -> bf16, bf16 16x16x16 MFMA)
  fp8  = in_dtype="a8w4_fp8"   (fp8 act + FP4->fp8 weight, native f32_16x16x32_fp8 MFMA)

Each variant is built once per shape, correctness-gated vs the torch ref (and the
native-fp8 variant is cross-checked vs bf16-decode), then the compiled kernel is
timed in-place with warmup + reps using HIP events. Reports median + min/max per
shape.

    HIP_VISIBLE_DEVICES=0 python op_tests/_bench_moe_bf16decode_vs_fp8native.py
"""

import statistics
import sys

import torch

import flydsl.compiler as flyc
from aiter.ops.flydsl.kernels.moe_gemm_2stage import compile_moe_gemm1

import test_flydsl_moe_a8w4_gfx942 as H


def _build(in_dtype, tokens, model_dim, inter_dim, experts, topk,
           tile_m, tile_n, tile_k, group_size=32):
    """Build a compiled stage1 kernel + args for one variant. Returns
    (compiled, args, out, g, ref). Inputs are seeded identically across variants
    (``_gen_common`` re-seeds), so both variants consume bit-identical activations.
    """
    g = H._gen_common(tokens, model_dim, inter_dim, experts, topk, tile_m, group_size)
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
    return compiled, args, out, g, ref


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
    tag = (f"M={shape['tokens']} K(model)={shape['model_dim']} "
           f"N(inter)={shape['inter_dim']} E={shape['experts']} topk={shape['topk']} "
           f"tile={shape['tile_m']}x{shape['tile_n']}x{shape['tile_k']}")
    print(f"\n=== {tag} ===")
    rows = {}
    for name, in_dtype in (("bf16", "a8w4_bf16"), ("fp8", "a8w4_fp8")):
        try:
            compiled, args, out, g, ref = _build(in_dtype=in_dtype, **shape)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] BUILD/RUN FAILED: {type(e).__name__}: {e}")
            rows[name] = None
            continue
        ok = H._check(ref, out, f"{name}_stage1", rel_l2_max=0.08)
        if not ok:
            print(f"[{name}] CORRECTNESS FAILED -- not timing this variant")
            rows[name] = None
            continue
        med, mn, mx, std = _time(compiled, args)
        rows[name] = dict(med=med, mn=mn, mx=mx, std=std)
        print(f"[{name}] median={med:.4f} ms  min={mn:.4f}  max={mx:.4f}  std={std:.4f}")
    return tag, rows


_SHAPES = [
    # (small validated shape from the test)
    dict(tokens=64, model_dim=256, inter_dim=128, experts=4, topk=1,
         tile_m=16, tile_n=64, tile_k=128),
    dict(tokens=512, model_dim=2048, inter_dim=768, experts=8, topk=2,
         tile_m=64, tile_n=128, tile_k=128),
    dict(tokens=1024, model_dim=2048, inter_dim=768, experts=8, topk=2,
         tile_m=64, tile_n=128, tile_k=128),
    dict(tokens=1024, model_dim=4096, inter_dim=1536, experts=32, topk=2,
         tile_m=64, tile_n=128, tile_k=128),
    dict(tokens=4096, model_dim=4096, inter_dim=1536, experts=32, topk=1,
         tile_m=64, tile_n=128, tile_k=128),
]


def main():
    if not H._is_gfx942():
        print("[SKIP] not gfx942")
        return 0
    print(f"device: {torch.cuda.get_device_properties(0).gcnArchName}  "
          f"name={torch.cuda.get_device_name(0)}")
    results = []
    for sh in _SHAPES:
        results.append(_run_shape(sh))

    print("\n\n================ SUMMARY (stage1) ================")
    print(f"{'shape':<58} {'bf16 ms':>10} {'fp8 ms':>10} {'bf16/fp8':>9} {'bf16 spread':>16} {'fp8 spread':>16}")
    for tag, rows in results:
        bf16, fp8 = rows.get("bf16"), rows.get("fp8")
        bf16s = f"{bf16['med']:.4f}" if bf16 else "FAIL"
        fp8s = f"{fp8['med']:.4f}" if fp8 else "FAIL"
        spd = f"{bf16['med']/fp8['med']:.3f}" if (bf16 and fp8) else "-"
        bf16sp = f"[{bf16['mn']:.4f},{bf16['mx']:.4f}]" if bf16 else "-"
        fp8sp = f"[{fp8['mn']:.4f},{fp8['mx']:.4f}]" if fp8 else "-"
        print(f"{tag:<58} {bf16s:>10} {fp8s:>10} {spd:>9} {bf16sp:>16} {fp8sp:>16}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Identify *where in source* each remaining aten op of the grouped gfx1250 MoE
path is launched from.

Runs ``_maybe_grouped_gfx1250_a8w4_moe`` (with the MI450 GEMMs stubbed, same as
test_grouped_moe_tinyops_profile.py) under a ``TorchDispatchMode`` that, for
every aten op dispatched, walks the live Python stack and attributes the op to
the innermost frame inside the aiter source tree (fused_moe.py / moe_kernels.py /
quant.py / fp4_utils.py / ...). Output is a table:
    aten op -> source file:line (the code line) -> count.

This is more reliable than torch.profiler's ``with_stack`` (which on ROCm often
fails to record Python frames -> "<no aiter frame>").

    python op_tests/identify_grouped_aten_ops.py            # NAIVE=0 (default)
    python op_tests/identify_grouped_aten_ops.py --naive
    python op_tests/identify_grouped_aten_ops.py --mode a8w4
"""

import argparse
import os
import sys
import traceback

# Reuse the profiling harness's env setup + GEMM stub + input builder.
import test_grouped_moe_tinyops_profile as H  # noqa: E402  (sets env at import)

_THIS = os.path.abspath(__file__)


def _innermost_aiter_frame(summary):
    """Given a traceback.StackSummary (outermost -> innermost), return
    'aiter/<rel>.py:line' for the innermost frame in the aiter tree (the line
    that actually issued the op), skipping this tracer file itself."""
    for fr in reversed(summary):
        fn = fr.filename
        if "/aiter/" not in fn or fn == _THIS:
            continue
        short = "aiter/" + fn.split("/aiter/", 1)[1]
        return f"{short}:{fr.lineno}"
    return "<no aiter frame>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["a8w4", "a4w4"], default="a4w4")
    ap.add_argument("--naive", action="store_true")
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--E", type=int, default=32)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--model-dim", type=int, default=4096)
    ap.add_argument("--inter-dim", type=int, default=4096)
    args = ap.parse_args()

    os.environ["AITER_GROUPED_GEMM_NAIVE"] = "1" if args.naive else "0"

    H._install_grouped_gemm_stub()
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from aiter import ActivationType, QuantType, dtypes
    from aiter.ops.flydsl.moe_common import GateMode
    from aiter.ops.flydsl.grouped_moe_gfx1250 import _maybe_grouped_gfx1250_a8w4_moe

    class OpSourceTracer(TorchDispatchMode):
        def __init__(self):
            self.agg = {}  # (opname, source) -> count

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            op = func.name() if hasattr(func, "name") else str(func)
            src = _innermost_aiter_frame(traceback.extract_stack())
            self.agg[(op, src)] = self.agg.get((op, src), 0) + 1
            return func(*args, **kwargs)

    device, dtype = "cuda", torch.bfloat16
    q_dtype_a = dtypes.fp8 if args.mode == "a8w4" else dtypes.fp4x2
    inp = H._build_inputs(
        args.tokens, args.E, args.topk, args.model_dim, args.inter_dim, dtype, device
    )

    def call():
        return _maybe_grouped_gfx1250_a8w4_moe(
            inp["hidden_states"],
            inp["w1"],
            inp["w2"],
            inp["topk_weight"],
            inp["topk_ids"],
            E=args.E,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            dtype=dtype,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            q_dtype_a=q_dtype_a,
            q_dtype_w=dtypes.fp4x2,
            isG1U1=True,
            doweight_stage1=False,
            w1_scale=inp["w1_scale"],
            w2_scale=inp["w2_scale"],
            expert_mask=None,
            hidden_pad=0,
            intermediate_pad=0,
            bias1=None,
            bias2=None,
            gate_mode=GateMode.SEPARATED,
        )

    # warmup (triton compile, caches) so the profiled run is steady-state
    for _ in range(3):
        call()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True
    ) as prof:
        call()
        torch.cuda.synchronize()

    # Aggregate aten ops by (op, innermost aiter source line).
    agg = {}
    for ev in prof.events():
        if not ev.name.startswith("aten::"):
            continue
        loc = _innermost_aiter_frame(ev.stack)
        dev_us = getattr(ev, "self_device_time_total", 0) or getattr(
            ev, "cuda_time_total", 0
        )
        k = (ev.name, loc)
        a = agg.setdefault(k, [0, 0.0])
        a[0] += 1
        a[1] += float(dev_us)

    rows = sorted(agg.items(), key=lambda kv: kv[1][1], reverse=True)
    print(
        f"\nmode={args.mode} naive={int(args.naive)}  "
        f"aten ops attributed to source (sorted by device us)\n"
    )
    print(f"{'aten op':28} {'count':>5}  {'dev_us':>8}   source")
    print("-" * 88)
    for (op, loc), (cnt, us) in rows:
        print(f"{op:28} {cnt:5d}  {us:8.1f}   {loc}")


if __name__ == "__main__":
    main()

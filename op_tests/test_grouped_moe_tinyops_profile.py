# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Profile the host-side "tiny operators" in the grouped gfx1250 MoE path.

``aiter/ops/flydsl/grouped_moe_gfx1250.py::_maybe_grouped_gfx1250_a8w4_moe`` is the gfx1250/MI450
grouped a8w4/a4w4 path. The two heavy grouped GEMMs (stage1/stage2) are
MI450-only and cannot compile/run on MI308 (gfx942). This test STUBS those two
GEMMs out (without modifying fused_moe.py) so the *rest* of the function -- the
many small host-side ops around the GEMMs -- runs on MI308, and profiles them so
you can see where the per-call overhead comes from.

What is stubbed: the four ``compile_moe_grouped_gemm{1,2}_{a8w4,mxfp4}_masked``
factories are replaced by no-op stages (the output buffers are pre-zeroed by
fused_moe, so a no-op stage leaves valid zeros). Everything else runs for real:
per-token quant, the route-gather scatter-copy kernel, e8m0 scale preshuffle,
a2 quant, doweight, and the gather-reduce epilogue.

Run on the MI308 (gfx942) box:
    python op_tests/test_grouped_moe_tinyops_profile.py
    python op_tests/test_grouped_moe_tinyops_profile.py --tokens 1 --E 32 \
        --model-dim 2880 --inter-dim 2880 --topk 4 --iters 50
    python op_tests/test_grouped_moe_tinyops_profile.py --naive   # naive route/epilogue
    python op_tests/test_grouped_moe_tinyops_profile.py --doweight
    python op_tests/test_grouped_moe_tinyops_profile.py --mode a4w4  # fp4 act + fp4 weight
    python op_tests/test_grouped_moe_tinyops_profile.py --real-gemm  # run real GEMMs (gfx1250 HW)

Note: this exercises AITER_GROUPED_GEMM_NAIVE=0 (the optimized kernels) by
default; pass --naive to compare the per-expert Python loops.

--mode selects the quant recipe (both share the same packed-fp4 weight layout,
so only ``q_dtype_a`` differs and it flips fused_moe's data_format):
    a8w4 (default): fp8 activation + fp4 weight  (data_format="a8w4")
    a4w4          : fp4 activation + fp4 weight  (data_format="fp4")
"""

import argparse
import os
import sys
import types

# ---------------------------------------------------------------------------
# Env: opt into the grouped path and force the gfx1250 arch gate to pass on
# MI308. These are read by fused_moe at call time, so setting them here is fine.
# AITER_LOG_MORE=1 makes test_common.get_trace_perf log the per-kernel table
# (name, cnt, host/device time) -- that table is the tiny-op breakdown.
# ---------------------------------------------------------------------------
os.environ.setdefault("AITER_USE_GROUPED_GEMM", "1")
os.environ.setdefault("AITER_FORCE_GFX1250", "1")  # passes the arch gate (line ~656)
os.environ.setdefault("AITER_LOG_MORE", "1")
# Allow `import aiter` on boxes with triton<3.6 (the gluon kernels' hard
# requirement); we don't use gluon here. Without this, importing aiter raises.
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")
os.environ.pop("AITER_DISABLE_GROUPED_A8W4", None)


def _install_grouped_gemm_stub():
    """Replace the MI450-only grouped GEMM module with no-op stages.

    fused_moe imports the four compile_* factories *inside* the function via
    ``from aiter.ops.flydsl.kernels.moe_grouped_gemm_mxscale_gfx1250 import ...``.
    Injecting a fake module under that name makes the from-import resolve to our
    stubs, so the real gfx1250 kernels are never imported or compiled.
    """
    name = "aiter.ops.flydsl.kernels.moe_grouped_gemm_mxscale_gfx1250"

    def _stub_compile(**kwargs):
        # Return a no-op "stage" callable. The output buffer (grouped_a2 /
        # grouped_out) is pre-zeroed by fused_moe, so doing nothing is safe.
        def _stage(*args, **kw):
            return None

        return _stage

    fake = types.ModuleType(name)
    fake.compile_moe_grouped_gemm1_a8w4_masked = _stub_compile
    fake.compile_moe_grouped_gemm2_a8w4_masked = _stub_compile
    fake.compile_moe_grouped_gemm1_mxfp4_masked = _stub_compile
    fake.compile_moe_grouped_gemm2_mxfp4_masked = _stub_compile
    sys.modules[name] = fake


def _build_inputs(token_num, E, topk, model_dim, inter_dim, dtype, device, seed=0):
    """Synthetic a8w4 grouped-MoE inputs (shapes that pass the eligibility gates
    and satisfy the host-side scale reshapes; GEMM math is stubbed)."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    hidden_states = torch.randn(
        (token_num, model_dim), generator=g, device=device, dtype=dtype
    )
    topk_ids = torch.stack(
        [torch.randperm(E, generator=g, device=device)[:topk] for _ in range(token_num)]
    ).to(torch.int32)
    topk_weight = torch.rand(
        (token_num, topk), generator=g, device=device, dtype=torch.float32
    )
    topk_weight = (topk_weight / topk_weight.sum(-1, keepdim=True)).to(dtype)

    # a8w4: a = fp8 (1 byte/elem), w = packed fp4 (2/byte). Weights/scales are
    # only viewed/reshaped on the host path (GEMM is stubbed), so content is
    # irrelevant -- but element counts must match the reshapes in fused_moe.
    w1 = torch.randint(
        0, 256, (E, 2 * inter_dim, model_dim // 2), dtype=torch.uint8, device=device
    )
    w2 = torch.randint(
        0, 256, (E, model_dim, inter_dim // 2), dtype=torch.uint8, device=device
    )
    w1_scale = torch.randint(
        0, 256, (E, 2 * inter_dim, model_dim // 32), dtype=torch.uint8, device=device
    )
    w2_scale = torch.randint(
        0, 256, (E, model_dim, inter_dim // 32), dtype=torch.uint8, device=device
    )
    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_weight": topk_weight,
        "topk_ids": topk_ids,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1, help="decode=1, prefill=N")
    ap.add_argument("--E", type=int, default=32)
    ap.add_argument("--topk", type=int, default=8)
    # NOTE: model_dim and inter_dim must be multiples of 128 -- the e8m0 scale
    # preshuffle requires (dim//32) % (tile_k//32 == 4) == 0.
    ap.add_argument("--model-dim", type=int, default=4096)
    ap.add_argument("--inter-dim", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--naive",
        action="store_true",
        help="use the per-expert Python loops (AITER_GROUPED_GEMM_NAIVE=1)",
    )
    ap.add_argument(
        "--mode",
        choices=["a8w4", "a4w4"],
        default="a8w4",
        help="quant recipe: a8w4=fp8 act+fp4 weight (default), "
        "a4w4=fp4 act+fp4 weight",
    )
    ap.add_argument("--doweight", action="store_true", help="doweight_stage1=True")
    ap.add_argument(
        "--rotate",
        type=int,
        default=1,
        help="num_rotate_args for run_perftest (1 = no input rotation)",
    )
    ap.add_argument(
        "--real-gemm",
        action="store_true",
        help="do NOT stub the grouped GEMMs -- run the real MI450/"
        "gfx1250 stage1/stage2 kernels (requires gfx1250 HW)",
    )
    args = ap.parse_args()

    if args.model_dim % 128 != 0 or args.inter_dim % 128 != 0:
        raise SystemExit(
            f"model_dim ({args.model_dim}) and inter_dim ({args.inter_dim}) must be "
            "multiples of 128 (the e8m0 scale preshuffle needs (dim//32) % 4 == 0)."
        )

    os.environ["AITER_GROUPED_GEMM_NAIVE"] = "1" if args.naive else "0"

    # Decide whether to run the real grouped GEMMs or the no-op stub.
    # On real gfx1250/MI450 HW the genuine stage1/stage2 kernels can compile and
    # run, so call the real grouped_gemm (the whole point on this box). The stub
    # exists only for non-gfx1250 boxes (e.g. MI308/gfx942) where the MI450 GEMM
    # cannot run; there we keep stubbing so the host-side tiny-ops still profile.
    from aiter.jit.utils.chip_info import get_gfx

    on_gfx1250 = get_gfx() == "gfx1250"
    real_gemm = args.real_gemm or on_gfx1250
    if real_gemm and not args.real_gemm:
        print(
            "[info] detected gfx1250 HW -> running real grouped GEMMs "
            "(pass --real-gemm explicitly to silence this)"
        )

    # Stub the MI450 GEMM (unless running real GEMMs), then import the function
    # under test. With real GEMMs the genuine gfx1250 stage1/stage2 kernels run,
    # so the grouped GEMMs show up in the per-kernel breakdown (gfx1250 HW only).
    if not real_gemm:
        _install_grouped_gemm_stub()
    import torch
    from aiter import ActivationType, QuantType, dtypes
    from aiter.ops.flydsl.moe_common import GateMode
    from aiter.ops.flydsl.grouped_moe_gfx1250 import _maybe_grouped_gfx1250_a8w4_moe
    from aiter.test_common import run_perftest

    device = "cuda"
    dtype = torch.bfloat16
    # a8w4: fp8 activation; a4w4: fp4 activation. The weight is packed fp4 in both
    # (q_dtype_w=fp4x2), so only q_dtype_a changes -- fused_moe derives
    # data_format ("a8w4" vs "fp4") from this pair.
    q_dtype_a = dtypes.fp8 if args.mode == "a8w4" else dtypes.fp4x2
    inp = _build_inputs(
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
            doweight_stage1=args.doweight,
            w1_scale=inp["w1_scale"],
            w2_scale=inp["w2_scale"],
            expert_mask=None,
            hidden_pad=0,
            intermediate_pad=0,
            bias1=None,
            bias2=None,
            gate_mode=GateMode.SEPARATED,
        )

    out = call()
    if out is None:
        raise SystemExit(
            "function returned None -- an eligibility gate failed. Check that "
            "AITER_USE_GROUPED_GEMM=1 and the gfx1250 arch gate passed "
            "(AITER_FORCE_GFX1250=1), and that q_dtype_a/q_dtype_w are fp8/fp4x2."
        )
    print(
        f"config: mode={args.mode} tokens={args.tokens} E={args.E} topk={args.topk} "
        f"model_dim={args.model_dim} inter_dim={args.inter_dim} "
        f"naive={int(args.naive)} doweight={int(args.doweight)} -> out {tuple(out.shape)}"
    )

    # ---- Profile via the project-standard run_perftest ----
    # Returns (output, avg_device_us). With AITER_LOG_MORE=1 (set above),
    # get_trace_perf logs the per-kernel dataframe: each row is one op with
    # `cnt` (calls across all iters) and host/device time -- divide cnt by
    # num_iters to get launches-per-call. That table is the tiny-op breakdown
    # showing why the path issues so many small kernels (memset/copy/cast/
    # quant/permute/scatter-copy/gather-reduce/...).
    out, avg_us = run_perftest(
        call,
        num_iters=args.iters,
        num_warmup=args.warmup,
        num_rotate_args=args.rotate,
    )
    print(
        f"\n[run_perftest] avg device time: {avg_us:.1f} us/iter "
        f"(per-kernel breakdown logged above via AITER_LOG_MORE=1)"
    )


if __name__ == "__main__":
    main()

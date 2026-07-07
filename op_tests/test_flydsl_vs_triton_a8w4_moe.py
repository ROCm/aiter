#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single UT that compares a8w4 MoE performance: FlyDSL grouped GEMM vs the
Triton ``moe_gemm_a8w4`` kernel, on gfx1250.

Fairness / alignment guarantees
-------------------------------
* **Same tokens**: both paths consume the identical ``hidden`` activation.
* **Same routing decision**: both derive top-k from a single shared gating
  ``score``. ``fused_topk`` (FlyDSL) applies softmax then top-k while
  ``routing`` (Triton) does top-k on the logits directly; softmax is monotonic,
  so the *selected experts* (and hence the per-expert token counts / load
  balance) are identical. Only the routing *weights* differ, which does not
  affect kernel timing.
* **Same shapes**: ``model_dim`` -> Triton ``dim1``, ``2*inter_dim`` -> Triton
  ``dim2``; experts/topk match.
* **Same quantization scheme**: MXFP8 (per-1x32) activations x MXFP4 weights on
  both sides (Triton ``mx8`` activation path).
* **Same timing harness**: both core MoE callables are timed through
  ``run_perftest`` with CUDA graph enabled (identical warmup/iters). Weight
  init/shuffle and FlyDSL's top-k input construction stay outside the timed
  region; Triton routing and dynamic per-call MoE work stay inside (activation
  quantization, route scatter/remap, stage2 input quantization, GEMMs, and final
  combine/gather).

Weight *values* differ between the two paths (FlyDSL consumes pre-shuffled
packed mxfp4 bytes; Triton downcasts bf16 to mxfp). That is intentional and
does not change the GEMM timing -- these kernels are data-independent in cost.
This test therefore compares *performance only*, not numerical parity.

Run directly for a sweep:

    python op_tests/test_flydsl_vs_triton_a8w4_moe.py \
        --experts 256 --topk 6 --model-dim 4096 --inter-dim 2048 \
        --tokens 64 128 256 --active-experts 32
"""

from __future__ import annotations

import argparse
import os
import sys

import pytest
import torch

# a8w4 needs AITER_FORCE_A8W4=1 so fused_moe routes the a8w4 grouped path.
os.environ.setdefault("AITER_FORCE_A8W4", "1")

from aiter import ActivationType, QuantType  # noqa: E402
from aiter.fused_moe import fused_moe, fused_topk  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
from aiter.ops.shuffle import moe_shuffle_scale, shuffle_weight  # noqa: E402
from aiter.test_common import run_perftest  # noqa: E402
from aiter.utility import dtypes  # noqa: E402

import aiter.ops.triton.moe.moe_routing.routing as routing_mod  # noqa: E402
from aiter.ops.triton.moe.moe_routing.routing import routing  # noqa: E402
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import (  # noqa: E402
    moe_gemm_a8w4,
    swizzle_scales_gfx1250,
)
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

SCALE_BLOCK = 32
DEFAULT_SCALE_BYTE = 127  # e8m0 byte for 2^0 = 1.0


def _is_gfx1250() -> bool:
    try:
        from flydsl.runtime.device import get_rocm_arch

        return "gfx1250" in get_rocm_arch().lower()
    except Exception:
        return False


def _require_gfx1250() -> None:
    if not _is_gfx1250():
        pytest.skip("requires gfx1250 hardware")


# ---------------------------------------------------------------------------
# Shared routing: one gating score feeds both paths (identical expert selection)
# ---------------------------------------------------------------------------
def _triton_herd_enabled(tokens: int) -> bool:
    return bool(
        getattr(routing_mod, "_USE_HERD", False)
        and getattr(routing_mod, "_HERD_MIN_M", 16)
        <= tokens
        <= getattr(routing_mod, "_HERD_MAX_M", 128)
    )


def _build_shared_routing(
    hidden: torch.Tensor, experts: int, topk: int, active_experts: int = 0
):
    """Return ``(score, topk_id, topk_w, routing_mode)``.

    ``score`` is the single source of truth. FlyDSL builds ``topk_id/topk_w``
    from the same stock/HERD routing mode that Triton ``routing`` will use.
    """
    tokens = hidden.shape[0]
    if active_experts < 0:
        raise ValueError(f"active_experts must be >= 0, got {active_experts}")
    if active_experts and (active_experts < topk or active_experts > experts):
        raise ValueError(
            f"active_experts={active_experts} must be in [topk={topk}, experts={experts}]"
        )
    score = torch.randn((tokens, experts), dtype=torch.float32, device=hidden.device)
    if active_experts:
        selected = torch.randperm(experts, device=hidden.device)[:active_experts]
        restricted = torch.full_like(score, float("-inf"))
        restricted[:, selected] = score[:, selected]
        score = restricted
    routing_mode = "herd" if _triton_herd_enabled(tokens) else "stock"
    if routing_mode == "herd":
        if active_experts and active_experts < topk + 1:
            raise ValueError(
                "HERD needs at least topk+1 active experts for the extra "
                f"candidate: active_experts={active_experts}, topk={topk}"
            )
    topk_w, topk_id = fused_topk(hidden, score, topk, True)
    topk_id = topk_id.to(torch.int32)
    topk_w = topk_w.to(torch.bfloat16)
    return score, topk_id, topk_w, routing_mode


# ---------------------------------------------------------------------------
# FlyDSL a8w4 grouped MoE inputs (pre-shuffled packed mxfp4 weights + e8m0 scale)
# ---------------------------------------------------------------------------
def _build_flydsl_inputs(experts, model_dim, inter_dim, seed, device):
    torch.manual_seed(seed)
    K = model_dim
    inter = inter_dim
    # FlyDSL side intentionally tests GGUU: all gate rows first, then all up rows.
    w1_logical = torch.randint(
        0, 256, (experts, 2 * inter, K // 2), dtype=torch.uint8, device=device
    )
    w2_logical = torch.randint(
        0, 256, (experts, K, inter // 2), dtype=torch.uint8, device=device
    )

    def _scale(rows, blocks):
        r = torch.randint(0, 3, (experts, rows, blocks), dtype=torch.int16, device=device)
        return (r + (DEFAULT_SCALE_BYTE - 1)).to(torch.uint8)

    w1_scale_raw = _scale(2 * inter, K // SCALE_BLOCK)
    w2_scale_raw = _scale(K, inter // SCALE_BLOCK)

    w1_grouped = shuffle_weight(w1_logical, layout=(16, 16))
    w2_grouped = shuffle_weight(w2_logical, layout=(16, 16))
    w1_scale = moe_shuffle_scale(w1_scale_raw.contiguous(), experts_cnt=experts)
    w2_scale = moe_shuffle_scale(w2_scale_raw.contiguous(), experts_cnt=experts)

    return {
        "w1": w1_grouped,
        "w2": w2_grouped,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "bias1": None,
        "bias2": None,
    }


# ---------------------------------------------------------------------------
# Triton mx8 x mx4 MoE inputs (downcast bf16 -> mxfp, gfx1250 scale swizzle)
# ---------------------------------------------------------------------------
def _swizzle(scale, N, K):
    if N % 32 == 0 and K % (32 * 8) == 0:
        return swizzle_scales_gfx1250(scale), "GFX1250_SCALE"
    return scale, None


def _mx8(x):
    # MXFP8 (per-1x32) activation quant along the last dim.
    return downcast_to_mxfp(x, torch.float8_e4m3fn, axis=1)


def _build_triton_inputs(experts, model_dim, inter_dim, seed, device):
    torch.manual_seed(seed + 1)
    dim1 = model_dim  # K of gemm1 / N of gemm2
    dim2 = 2 * inter_dim  # N of gemm1 (gate+up)
    w1 = torch.randn((experts, dim1, dim2), device=device, dtype=torch.bfloat16)
    w2 = torch.randn((experts, dim2 // 2, dim1), device=device, dtype=torch.bfloat16)
    w1_q, w1_scale = downcast_to_mxfp(w1, torch.uint8, axis=1)
    w2_q, w2_scale = downcast_to_mxfp(w2, torch.uint8, axis=1)
    w1_scale, sw1 = _swizzle(w1_scale, dim2, dim1)
    w2_scale, sw2 = _swizzle(w2_scale, dim1, dim2 // 2)
    return {
        "w1_q": w1_q,
        "w2_q": w2_q,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "sw1": sw1,
        "sw2": sw2,
        "b1": None,
        "b2": None,
    }


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------
def _time_callable(fn, *, warmup: int, iters: int) -> float:
    torch.cuda.synchronize()
    _, us = run_perftest(
        fn,
        num_warmup=warmup,
        num_iters=iters,
        testGraph=True,
    )
    return us


def compare_a8w4_moe(
    *,
    experts: int,
    tokens: int,
    topk: int,
    model_dim: int,
    inter_dim: int,
    active_experts: int = 0,
    activation: ActivationType = ActivationType.Silu,
    swiglu_limit: float = 7.0,
    warmup: int = 5,
    iters: int = 2,
    seed: int = 0,
    device: str = "cuda",
) -> dict:
    torch.manual_seed(seed)
    hidden = (torch.randn((tokens, model_dim), device=device) * 0.5).to(torch.bfloat16)

    score, topk_id, topk_w, routing_mode = _build_shared_routing(
        hidden, experts, topk, active_experts
    )

    fly = _build_flydsl_inputs(experts, model_dim, inter_dim, seed, device)
    tri = _build_triton_inputs(experts, model_dim, inter_dim, seed, device)
    flydsl_layout = "gguu"

    # ---- FlyDSL forward (grouped a8w4 path auto-selected on gfx1250) ----
    def _flydsl_fwd():
        return fused_moe(
            hidden,
            fly["w1"],
            fly["w2"],
            topk_w,
            topk_id,
            activation=activation,
            quant_type=QuantType.per_1x32,
            w1_scale=fly["w1_scale"],
            w2_scale=fly["w2_scale"],
            bias1=fly["bias1"],
            bias2=fly["bias2"],
            gate_mode=GateMode.SEPARATED.value,  # GGUU
            dtype=dtypes.bf16,
            swiglu_limit=swiglu_limit,
        )

    # ---- Triton core forward (mx8 activation x mx4 weight) ----
    # Triton's a8w4 MoE kernel exposes the gate/up fused epilogue through
    # apply_swiglu. Even when FlyDSL runs ActivationType.Silu to match the tuned
    # CSV, Triton still needs this enabled so stage1 writes inter_dim columns
    # instead of 2*inter_dim; otherwise stage2 sees a K mismatch and can OOB.
    triton_apply_gate_up = True

    def _triton_fwd():
        # Triton consumes logits and builds its routing metadata per call, so
        # include routing in the timed core.
        rdata, gather_indx, scatter_indx = routing(score, topk)
        # These two quantizations are necessary per-call work for the Triton mx8
        # path, so keep them inside the timed core as well.
        xq, xs = _mx8(hidden)
        x1 = moe_gemm_a8w4(
            xq,
            tri["w1_q"],
            xs,
            tri["w1_scale"],
            None,
            None,
            tri["b1"],
            rdata,
            gather_indx=gather_indx,
            swizzle_mx_scale=tri["sw1"],
            apply_swiglu=triton_apply_gate_up,
            limit=swiglu_limit,
        )
        assert x1.shape[-1] == inter_dim, (
            f"Triton stage1 output K mismatch: got {x1.shape[-1]}, "
            f"expected {inter_dim}"
        )
        x1q, x1s = _mx8(x1)
        return moe_gemm_a8w4(
            x1q,
            tri["w2_q"],
            x1s,
            tri["w2_scale"],
            None,
            None,
            tri["b2"],
            rdata,
            scatter_indx=scatter_indx,
            swizzle_mx_scale=tri["sw2"],
        )

    # Compile/allocate once before the measured loop. run_perftest also has its
    # own warmup, but this keeps first-call setup out of both sides symmetrically.
    _flydsl_fwd()
    _triton_fwd()
    fly_us = _time_callable(_flydsl_fwd, warmup=warmup, iters=iters)
    tri_us = _time_callable(_triton_fwd, warmup=warmup, iters=iters)

    speedup = tri_us / fly_us if fly_us else float("nan")
    print(
        f"[a8w4 moe] tokens={tokens:>6} E={experts} topk={topk} "
        f"active={active_experts or experts} flydsl_layout={flydsl_layout} "
        f"routing={routing_mode} model_dim={model_dim} inter_dim={inter_dim} | "
        f"FlyDSL_core={fly_us:8.2f}us  Triton_core={tri_us:8.2f}us  "
        f"speedup(triton/flydsl)={speedup:5.2f}x",
        flush=True,
    )
    return {
        "tokens": tokens,
        "experts": experts,
        "topk": topk,
        "active_experts": active_experts or experts,
        "flydsl_layout": flydsl_layout,
        "routing": routing_mode,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "flydsl_us": fly_us,
        "triton_us": tri_us,
        "speedup": speedup,
    }


# ---------------------------------------------------------------------------
# Pytest entry: small shape, just make sure both paths run and are timed.
# ---------------------------------------------------------------------------
def test_a8w4_moe_flydsl_vs_triton():
    _require_gfx1250()
    torch.set_default_device("cuda")
    m = compare_a8w4_moe(
        experts=32,
        tokens=128,
        topk=4,
        model_dim=2048,
        inter_dim=1024,
        activation=ActivationType.Silu,
    )
    assert m["flydsl_us"] > 0 and m["triton_us"] > 0


# ---------------------------------------------------------------------------
# CLI sweep
# ---------------------------------------------------------------------------
def _summarize(rows: list):
    if not rows:
        return
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        try:
            table = df.to_markdown(index=False)
        except ImportError:
            table = df.to_string(index=False)
        print("\n[a8w4 moe perf summary]\n" + table, flush=True)
    except ImportError:
        print("\n[a8w4 moe perf summary]", flush=True)
        for r in rows:
            print(f"  {r}", flush=True)


def main() -> None:
    if not _is_gfx1250():
        print("skipping: requires gfx1250")
        sys.exit(0)
    torch.set_default_device("cuda")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument(
        "--active-experts",
        type=int,
        default=0,
        help="number of experts allowed to receive tokens. 0 means all experts.",
    )
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--model-dim", type=int, default=4096)
    parser.add_argument("--inter-dim", type=int, default=2048)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[64, 128, 256], metavar="N"
    )
    parser.add_argument("--act", choices=("silu", "swiglu"), default="silu")
    parser.add_argument("--swiglu-limit", type=float, default=7.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=51)
    args = parser.parse_args()

    activation = (
        ActivationType.Swiglu if args.act == "swiglu" else ActivationType.Silu
    )
    rows = []
    for tok in args.tokens:
        rows.append(
            compare_a8w4_moe(
                experts=args.experts,
                active_experts=args.active_experts,
                tokens=tok,
                topk=args.topk,
                model_dim=args.model_dim,
                inter_dim=args.inter_dim,
                activation=activation,
                swiglu_limit=args.swiglu_limit,
                warmup=args.warmup,
                iters=args.iters,
            )
        )
    _summarize(rows)


if __name__ == "__main__":
    main()

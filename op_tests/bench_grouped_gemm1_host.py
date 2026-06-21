"""Host-timed microbenchmark for the grouped MoE **gemm1** kernel (gfx1250).

Isolates stage1 (the grouped gemm1: gate+up projection, swiglu, gguu layout)
and times it with a host clock:

    torch.cuda.synchronize()
    t0 = perf_counter()
    for _ in range(ROUNDS): stage1(...)
    torch.cuda.synchronize()
    t1 = perf_counter()
    us/round = (t1 - t0) / ROUNDS

Inputs (grouped activations, weights, preshuffled scales, masked_m, m-tile
maps) are built by the real fused_moe path via ``run_moe``; we monkeypatch the
stage1 compiler to capture the compiled launch + its already-built arguments on
the single setup call, then replay only that launch under host timing.

Config: 128 experts, topk=4, 64 tokens, gguu (GateMode.SEPARATED), mxfp4 swiglu.
Sizes:  model_dim x inter_dim = 3072x3072 and 8192x8192.

Run:  python op_tests/bench_grouped_gemm1_host.py
"""

import os

# Match the env the grouped path expects (see test3.sh).
os.environ.setdefault("AITER_USE_GROUPED_GEMM", "1")
os.environ.setdefault("AITER_FORCE_GFX1250", "1")
os.environ.setdefault("WEIGHT_SCALE_OP_SEL", "1")
os.environ.setdefault("AITER_GROUPED_GEMM_NAIVE", "0")
os.environ.setdefault("AITER_MOE_EXPERT_BALANCE", "true")
os.environ.setdefault("AITER_GROUPED_DEBUG", "0")
os.environ.setdefault("FLYDSL_DUMP_IR", "0")

import sys
import time

import torch

from aiter import ActivationType
import aiter.ops.flydsl.kernels.moe_grouped_gemm_mxscale_gfx1250 as _mod

# --- capture the compiled stage1 launch + its runtime args ------------------
# grouped_moe imports this symbol lazily inside its dispatch fn, so patching the
# source-module attribute is enough.
_orig_compile = _mod.compile_moe_grouped_gemm1_mxfp4_masked
_cap = {}


def _capturing_compile(**kw):
    launch = _orig_compile(**kw)

    def _recording_launch(*a, **k):
        _cap["launch"] = launch
        _cap["args"] = a
        _cap["kwargs"] = k
        return launch(*a, **k)

    return _recording_launch


_mod.compile_moe_grouped_gemm1_mxfp4_masked = _capturing_compile

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import test_flydsl_grouped_gemm_gfx1250 as T  # noqa: E402

ROUNDS = 10
WARMUP = 3
EXPERTS = 128
TOKENS = 64
TOPK = 4


def bench(model_dim: int, inter_dim: int) -> None:
    _cap.clear()
    # One setup call: builds all grouped inputs + runs fused_moe once, during
    # which our wrapper captures stage1's launch and its concrete arguments.
    T.run_moe(
        "a4w4",
        experts=EXPERTS,
        tokens=TOKENS,
        topk=TOPK,
        model_dim=model_dim,
        inter_dim=inter_dim,
        layout="gguu",
        activation=ActivationType.Swiglu,
        bench=False,
        raise_on_fail=False,
    )
    if "launch" not in _cap:
        raise RuntimeError("stage1 launch was not captured (grouped path not taken?)")

    launch = _cap["launch"]
    args = _cap["args"]
    kwargs = dict(_cap["kwargs"])
    kwargs["stream"] = torch.cuda.current_stream()

    for _ in range(WARMUP):
        launch(*args, **kwargs)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ROUNDS):
        launch(*args, **kwargs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    us = (t1 - t0) / ROUNDS * 1e6
    print(
        f"[gemm1 gguu] model_dim={model_dim} inter_dim={inter_dim} "
        f"experts={EXPERTS} topk={TOPK} tokens={TOKENS}: "
        f"{us:.2f} us/round  (host timing, {ROUNDS} rounds)",
        flush=True,
    )


def main() -> None:
    print(f"grouped MoE gemm1 host-timed bench (rounds={ROUNDS}, warmup={WARMUP})\n")
    for md, idim in ((3072, 3072), (8192, 8192)):
        bench(md, idim)


if __name__ == "__main__":
    main()

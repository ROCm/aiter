# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""HIP-graph microbench of fused 1-stage AR start_sync wait.

AITER-only replica of a decode-shaped TP8 graph: GDN, gemma silu, local
out-proj, fused 1-stage gemma AR, local router, topk, fused 1-stage gemma
AR. No vLLM. Local GEMMs are `F.linear`, not a split-K GEMM. At tokens=1
that is a GEMV, so CUDA-event AR tails are smaller than a CU-wide split-K
producer that contends with RankData mappings during `start_sync`. CUDA
events time GDN, the silu-to-GEMM gap, and both fused ARs.

Do not insert torch.cuda._sleep. That kernel is at::cuda::spin_kernel and
is not in a serving graph.

Run:

    torchrun --standalone --nproc_per_node=8 \\
      op_tests/multigpu_tests/bench_fused_ar_graph_pause.py \\
      --unroll 32 --repeats 12 --ones-check
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import statistics
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F


DEFAULT_TOKENS = 1
DEFAULT_HIDDEN = 8192
DEFAULT_EPS = 1e-6
DEFAULT_NUM_K_HEADS = 16
DEFAULT_NUM_V_HEADS = 128
DEFAULT_HEAD_K = 128
DEFAULT_HEAD_V = 128
DEFAULT_CONV_WIDTH = 4
DEFAULT_CACHE_LINES = 8
DEFAULT_EXPERTS = 512
DEFAULT_TOPK = 10


def dump_silu(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # Gemma-style RMS upcasts to fp32, then add/mean/mul/pow/rsqrt/silu.
    y = (x + residual).to(torch.float32)
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = y * weight.to(torch.float32)
    return F.silu(y).to(dtype=x.dtype)


def import_gdn():
    conv = None
    conv_err = []
    for path in (
        "aiter.ops.triton.causal_conv1d_update_single_token",
        "aiter.ops.triton.conv.causal_conv1d_update_single_token",
    ):
        try:
            mod = importlib.import_module(path)
            conv = getattr(mod, "fused_reshape_causal_conv1d_update_single_token")
            break
        except Exception as exc:
            conv_err.append(f"{path}: {type(exc).__name__}: {exc}")
    rearrange = None
    rearrange_err = []
    for path, name in (
        (
            "aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr",
            "fused_rearrange_sigmoid_gated_delta_rule",
        ),
        (
            "aiter.ops.triton.gated_delta_net",
            "fused_rearrange_sigmoid_gated_delta_rule",
        ),
    ):
        try:
            mod = importlib.import_module(path)
            rearrange = getattr(mod, name)
            break
        except Exception as exc:
            rearrange_err.append(f"{path}.{name}: {type(exc).__name__}: {exc}")
    if conv is None or rearrange is None:
        raise RuntimeError(
            "GDN import failed. conv="
            + repr(conv_err)
            + " rearrange="
            + repr(rearrange_err)
        )
    return conv, rearrange


def import_topk():
    try:
        from aiter.ops.moe_op import topk_softmax

        return topk_softmax
    except Exception:
        from aiter.ops.triton.moe_op import topk_softmax  # type: ignore

        return topk_softmax


def filter_kwargs(fn, kwargs: dict) -> dict:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def summarize(name: str, samples: list[float]) -> str:
    xs = sorted(samples)
    n = len(xs)
    p50 = xs[n // 2]
    p90 = xs[int(n * 0.90)]
    p99 = xs[min(n - 1, int(n * 0.99))]
    return (
        f"{name}: n={n} min={xs[0]:.1f} us  p50={p50:.1f} us  "
        f"p90={p90:.1f} us  p99={p99:.1f} us  max={xs[-1]:.1f} us  "
        f"mean={statistics.fmean(xs):.1f} us"
    )


def init_dist() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    return local_rank, world, device


def make_aiter_ar(device: torch.device):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    group = dist.group.WORLD
    try:
        group = dist.new_group(backend="gloo")
    except Exception as exc:
        print(f"gloo group failed ({exc}). Using WORLD.", flush=True)
    ar = CustomAllreduce(group, device)
    if getattr(ar, "disabled", False):
        raise RuntimeError(
            f"AITER CustomAllreduce is disabled on rank {dist.get_rank()}"
        )
    return ar


@dataclass
class LayerTensors:
    qkvz: torch.Tensor
    ba: torch.Tensor
    z: torch.Tensor
    core: torch.Tensor
    conv_state: torch.Tensor
    conv_w: torch.Tensor
    conv_b: torch.Tensor
    conv_idx: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    ssm: torch.Tensor
    cu_seqlens: torch.Tensor
    residual: torch.Tensor
    silu_w: torch.Tensor
    out_w: torch.Tensor
    router_w: torch.Tensor
    ar_residual: torch.Tensor
    ar_w: torch.Tensor
    moe_residual: torch.Tensor
    moe_w: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    token_expert_indices: torch.Tensor


def make_tensors(
    *,
    tokens: int,
    hidden: int,
    nk: int,
    nv: int,
    dk: int,
    dv: int,
    conv_width: int,
    cache_lines: int,
    experts: int,
    topk: int,
    device: torch.device,
) -> LayerTensors:
    dtype = torch.bfloat16
    key_dim = nk * dk
    value_dim = nv * dv
    qkv_dim = key_dim * 2 + value_dim
    z_dim = value_dim
    conv_dim = qkv_dim
    return LayerTensors(
        qkvz=torch.randn(tokens, qkv_dim + z_dim, device=device, dtype=dtype) * 0.05,
        ba=torch.randn(tokens, nv * 2, device=device, dtype=dtype) * 0.05,
        z=torch.empty(tokens, nv, dv, device=device, dtype=dtype),
        core=torch.empty(tokens, nv, dv, device=device, dtype=dtype),
        conv_state=torch.zeros(
            cache_lines, conv_dim, conv_width - 1, device=device, dtype=dtype
        ),
        conv_w=torch.randn(conv_dim, conv_width, device=device, dtype=dtype) * 0.02,
        conv_b=torch.zeros(conv_dim, device=device, dtype=dtype),
        conv_idx=torch.arange(tokens, device=device, dtype=torch.int32),
        A_log=torch.randn(nv, device=device, dtype=torch.float32).clamp(-2.0, 0.5)
        * 0.02,
        dt_bias=torch.randn(nv, device=device, dtype=dtype) * 0.005,
        ssm=torch.zeros(cache_lines, nv, dv, dk, device=device, dtype=dtype),
        cu_seqlens=torch.tensor([0, tokens], device=device, dtype=torch.int32),
        residual=torch.randn(tokens, value_dim, device=device, dtype=dtype),
        silu_w=torch.ones(value_dim, device=device, dtype=dtype),
        out_w=torch.randn(hidden, value_dim, device=device, dtype=dtype) * 0.02,
        router_w=torch.randn(experts, hidden, device=device, dtype=dtype) * 0.02,
        ar_residual=torch.randn(tokens, hidden, device=device, dtype=dtype),
        ar_w=torch.ones(hidden, device=device, dtype=dtype),
        moe_residual=torch.randn(tokens, hidden, device=device, dtype=dtype),
        moe_w=torch.ones(hidden, device=device, dtype=dtype),
        topk_weights=torch.empty(tokens, topk, dtype=torch.float32, device=device),
        topk_ids=torch.empty(tokens, topk, dtype=torch.int32, device=device),
        token_expert_indices=torch.empty(
            tokens, topk, dtype=torch.int32, device=device
        ),
    )


class Layer:
    """One unroll. GDN, silu, out-proj, fused AR, router, topk, fused AR."""

    def __init__(
        self,
        tensors: LayerTensors,
        rms_silu,
        ar,
        conv_fn,
        rearrange_fn,
        topk_fn,
        *,
        eps: float,
        nk: int,
        nv: int,
        dk: int,
        dv: int,
        registered: bool,
        use_1stage: bool,
        gemma_norm: bool,
        qkvz_layout: str | None,
        ar_mode: str,
        moe_tail: bool,
        use_gdn: bool,
    ) -> None:
        self.t = tensors
        self.rms_silu = rms_silu
        self.ar = ar
        self.conv_fn = conv_fn
        self.rearrange_fn = rearrange_fn
        self.topk_fn = topk_fn
        self.eps = eps
        self.nk = nk
        self.nv = nv
        self.dk = dk
        self.dv = dv
        self.registered = registered
        self.use_1stage = use_1stage
        self.gemma_norm = gemma_norm
        self.qkvz_layout = qkvz_layout
        self.ar_mode = ar_mode
        self.moe_tail = moe_tail
        self.use_gdn = use_gdn
        self.ev_gdn_start = torch.cuda.Event(enable_timing=True)
        self.ev_silu_done = torch.cuda.Event(enable_timing=True)
        self.ev_gemm_start = torch.cuda.Event(enable_timing=True)
        self.ev_gemm_done = torch.cuda.Event(enable_timing=True)
        self.ev_ar1_done = torch.cuda.Event(enable_timing=True)
        self.ev_topk_done = torch.cuda.Event(enable_timing=True)
        self.ev_ar2_done = torch.cuda.Event(enable_timing=True)

    def run_gdn(self) -> torch.Tensor:
        tokens = self.t.qkvz.size(0)
        conv_kwargs = {
            "x": self.t.qkvz,
            "num_actual_tokens": tokens,
            "num_k_heads": self.nk,
            "num_v_heads": self.nv,
            "head_k_dim": self.dk,
            "head_v_dim": self.dv,
            "ba": self.t.ba,
            "z_out": self.t.z,
            "core_attn_out": self.t.core,
            "conv_state": self.t.conv_state,
            "weight": self.t.conv_w,
            "bias": self.t.conv_b,
            "activation": "silu",
            "conv_state_indices": self.t.conv_idx,
            "validate_data": False,
        }
        if self.qkvz_layout:
            conv_kwargs["qkvz_layout"] = self.qkvz_layout
        mixed_qkv, b, a = self.conv_fn(**filter_kwargs(self.conv_fn, conv_kwargs))
        rear_kwargs = {
            "A_log": self.t.A_log,
            "a": a,
            "b": b,
            "dt_bias": self.t.dt_bias,
            "qkv": mixed_qkv,
            "key_dim": self.nk * self.dk,
            "value_dim": self.nv * self.dv,
            "head_k_dim": self.dk,
            "head_v_dim": self.dv,
            "K": self.dk,
            "V": self.dv,
            "beta": 1.0,
            "threshold": 20.0,
            "scale": self.dk**-0.5,
            "initial_state": self.t.ssm,
            "inplace_final_state": True,
            "cu_seqlens": self.t.cu_seqlens,
            "ssm_state_indices": self.t.conv_idx,
            "use_qk_l2norm_in_kernel": True,
            "is_kda": False,
            "core_attn_out": self.t.core.reshape(-1),
        }
        self.rearrange_fn(**filter_kwargs(self.rearrange_fn, rear_kwargs))
        return self.t.core.reshape(tokens, -1)

    def fused_ar(self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor):
        if self.ar_mode == "none":
            return inp
        if self.ar_mode == "unfused":
            reduced = self.ar.custom_all_reduce(inp)
            if reduced is None:
                raise RuntimeError("custom_all_reduce returned None")
            y = reduced.to(torch.float32) + residual.to(torch.float32)
            y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            y = y * (1.0 + weight.to(torch.float32))
            return y.to(dtype=inp.dtype)
        result = self.ar.fused_ar_rms(
            inp,
            residual,
            w=weight,
            eps=self.eps,
            registered=self.registered,
            use_1stage=self.use_1stage,
            gemma_norm=self.gemma_norm,
        )
        if result is None:
            raise RuntimeError("fused_ar_rms returned None")
        if isinstance(result, (tuple, list)):
            return result[0]
        return result

    def run_topk(self, gating: torch.Tensor) -> None:
        self.topk_fn(
            self.t.topk_weights,
            self.t.topk_ids,
            self.t.token_expert_indices,
            gating.float(),
            True,
        )

    def run(self) -> None:
        self.ev_gdn_start.record()
        if self.use_gdn:
            x = self.run_gdn()
            act = self.rms_silu(x, self.t.residual, self.t.silu_w, self.eps)
        else:
            act = self.t.residual
        self.ev_silu_done.record()
        self.ev_gemm_start.record()
        gemm = F.linear(act, self.t.out_w)
        self.ev_gemm_done.record()
        hidden = self.fused_ar(gemm, self.t.ar_residual, self.t.ar_w)
        self.ev_ar1_done.record()
        if not self.moe_tail:
            return
        gating = F.linear(hidden, self.t.router_w)
        self.run_topk(gating)
        self.ev_topk_done.record()
        self.fused_ar(hidden, self.t.moe_residual, self.t.moe_w)
        self.ev_ar2_done.record()

    def gdn_us(self) -> float:
        return self.ev_gdn_start.elapsed_time(self.ev_silu_done) * 1000.0

    def gap_us(self) -> float:
        return self.ev_silu_done.elapsed_time(self.ev_gemm_start) * 1000.0

    def gemm_us(self) -> float:
        return self.ev_gemm_start.elapsed_time(self.ev_gemm_done) * 1000.0

    def ar1_us(self) -> float:
        return self.ev_gemm_done.elapsed_time(self.ev_ar1_done) * 1000.0

    def ar2_us(self) -> float:
        if not self.moe_tail:
            return 0.0
        return self.ev_topk_done.elapsed_time(self.ev_ar2_done) * 1000.0


def ones_check(ar, device: torch.device, hidden: int, world: int) -> None:
    inp = torch.ones(1, hidden, device=device, dtype=torch.bfloat16)
    residual = torch.zeros_like(inp)
    weight = torch.ones(hidden, device=device, dtype=torch.bfloat16)
    result = ar.fused_ar_rms(
        inp,
        residual,
        w=weight,
        eps=DEFAULT_EPS,
        registered=False,
        use_1stage=True,
        gemma_norm=True,
    )
    res_out = result[1] if isinstance(result, (tuple, list)) else result
    torch.cuda.synchronize()
    got = res_out.float()
    want = float(world)
    err = (got - want).abs().max().item()
    print(
        f"rank={dist.get_rank()} ones residual_out mean={got.mean().item():.6f} "
        f"max_abs_err={err:.6e} want={want}",
        flush=True,
    )
    bad = torch.tensor([1.0 if err > 0.05 else 0.0], device=device)
    dist.all_reduce(bad)
    if bad.item() > 0:
        raise RuntimeError("ones-check failed: residual_out is not world_size")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--unroll", type=int, default=32)
    p.add_argument("--repeats", type=int, default=12)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--eager-warmup", type=int, default=8)
    p.add_argument("--registered", type=int, choices=(0, 1), default=1)
    p.add_argument("--use-1stage", type=int, choices=(0, 1), default=1)
    p.add_argument("--gemma-norm", type=int, choices=(0, 1), default=1)
    p.add_argument("--moe-tail", type=int, choices=(0, 1), default=1)
    p.add_argument("--gdn", type=int, choices=(0, 1), default=1)
    p.add_argument("--qkvz-layout", default="flat")
    p.add_argument("--ar", choices=("fused", "unfused", "none"), default="fused")
    p.add_argument("--ones-check", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rank, world, device = init_dist()
    if world < 2:
        print("need WORLD_SIZE>=2", file=sys.stderr)
        return 2
    if DEFAULT_NUM_K_HEADS % world != 0 or DEFAULT_NUM_V_HEADS % world != 0:
        print(f"heads not divisible by TP={world}", file=sys.stderr)
        return 2
    nk = DEFAULT_NUM_K_HEADS // world
    nv = DEFAULT_NUM_V_HEADS // world
    dk = DEFAULT_HEAD_K
    dv = DEFAULT_HEAD_V
    layout = args.qkvz_layout.strip() or None
    print(
        f"rank={rank}/{world} device={device} nk={nk} nv={nv} dk={dk} dv={dv} "
        f"unroll={args.unroll} ar={args.ar} gdn={args.gdn} "
        f"moe_tail={args.moe_tail} registered={args.registered} "
        f"use_1stage={args.use_1stage} gemma_norm={args.gemma_norm} "
        f"layout={layout!r}",
        flush=True,
    )
    conv_fn, rearrange_fn = import_gdn() if args.gdn else (None, None)
    topk_fn = import_topk() if args.moe_tail else None
    rms_silu = torch.compile(dump_silu, fullgraph=True, dynamic=False)
    ar = make_aiter_ar(device)
    if args.ones_check and args.ar == "fused":
        ones_check(ar, device, args.hidden, world)

    def make_layer(registered: bool) -> Layer:
        return Layer(
            make_tensors(
                tokens=args.tokens,
                hidden=args.hidden,
                nk=nk,
                nv=nv,
                dk=dk,
                dv=dv,
                conv_width=DEFAULT_CONV_WIDTH,
                cache_lines=DEFAULT_CACHE_LINES,
                experts=DEFAULT_EXPERTS,
                topk=DEFAULT_TOPK,
                device=device,
            ),
            rms_silu,
            ar,
            conv_fn,
            rearrange_fn,
            topk_fn,
            eps=DEFAULT_EPS,
            nk=nk,
            nv=nv,
            dk=dk,
            dv=dv,
            registered=registered,
            use_1stage=bool(args.use_1stage),
            gemma_norm=bool(args.gemma_norm),
            qkvz_layout=layout,
            ar_mode=args.ar,
            moe_tail=bool(args.moe_tail),
            use_gdn=bool(args.gdn),
        )

    probe = make_layer(False)
    for _ in range(args.eager_warmup):
        probe.run()
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print("eager warmup finished", flush=True)

    # One buffer set per unroll, matching vLLM graph address registration.
    layers = [make_layer(bool(args.registered)) for _ in range(args.unroll)]
    capture_ctx = ar.capture() if hasattr(ar, "capture") else None
    if capture_ctx is not None:
        capture_ctx.__enter__()
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            saved = [layer.registered for layer in layers]
            for layer in layers:
                layer.registered = False
                layer.run()
            for layer, flag in zip(layers, saved):
                layer.registered = flag
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        dist.barrier()
        try:
            g = torch.cuda.CUDAGraph(keep_graph=True)
        except TypeError:
            g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for layer in layers:
                layer.run()
    finally:
        if capture_ctx is not None:
            capture_ctx.__exit__(None, None, None)

    print(f"rank={rank} captured unroll={args.unroll}", flush=True)
    dist.barrier()

    gdn: list[float] = []
    gaps: list[float] = []
    gemms: list[float] = []
    ar1: list[float] = []
    ar2: list[float] = []
    for i in range(args.warmup + args.repeats):
        g.replay()
        torch.cuda.synchronize()
        if i < args.warmup:
            continue
        for layer in layers:
            gdn.append(layer.gdn_us())
            gaps.append(layer.gap_us())
            gemms.append(layer.gemm_us())
            ar1.append(layer.ar1_us())
            if args.moe_tail:
                ar2.append(layer.ar2_us())

    payload = torch.tensor(
        [
            min(gdn),
            statistics.median(gdn),
            max(gdn),
            min(gaps),
            statistics.median(gaps),
            max(gaps),
            min(ar1),
            statistics.median(ar1),
            max(ar1),
            min(ar2) if ar2 else 0.0,
            statistics.median(ar2) if ar2 else 0.0,
            max(ar2) if ar2 else 0.0,
        ],
        device=device,
        dtype=torch.float64,
    )
    gathered = [torch.zeros_like(payload) for _ in range(world)]
    dist.all_gather(gathered, payload)

    print(summarize(f"rank{rank} gdn_to_silu", gdn), flush=True)
    print(summarize(f"rank{rank} silu_to_gemm_gap", gaps), flush=True)
    print(summarize(f"rank{rank} gemm", gemms), flush=True)
    print(summarize(f"rank{rank} fused_ar1", ar1), flush=True)
    if ar2:
        print(summarize(f"rank{rank} fused_ar2", ar2), flush=True)

    if rank == 0:
        print(
            "per-rank CUDA events (min / p50 / max us) gdn | silu_to_gemm | ar1 | ar2:",
            flush=True,
        )
        ar_ms = False
        for i, t in enumerate(gathered):
            vals = t.tolist()
            print(
                f"  rank{i} gdn {vals[0]:.1f}/{vals[1]:.1f}/{vals[2]:.1f}  "
                f"gap {vals[3]:.1f}/{vals[4]:.1f}/{vals[5]:.1f}  "
                f"ar1 {vals[6]:.1f}/{vals[7]:.1f}/{vals[8]:.1f}  "
                f"ar2 {vals[9]:.1f}/{vals[10]:.1f}/{vals[11]:.1f}",
                flush=True,
            )
            if vals[8] >= 1000.0 or vals[11] >= 1000.0:
                ar_ms = True
        if args.ar == "none":
            print("AR skipped. CUDA events are local work plus graph overhead.", flush=True)
        elif ar_ms:
            print(
                "fused AR is >=1 ms on at least one rank. That is start_sync wait.",
                flush=True,
            )
        else:
            print("No millisecond fused AR on any rank.", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

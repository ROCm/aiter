#!/usr/bin/env python3
"""SonicMoE-parity backward benchmark for the gfx950 Opus/Triton paths.

The default shape and FLOP convention match sonic-moe/benchmarks/moe-cute.py:
T,H,I,E,topk = 32768,2048,1024,64,8 and 12*T*H*I*topk backward FLOPs.

Three end-to-end numbers are deliberately kept separate:

* expert_direct: backward through the expert MoE ending at d(router_logits).
* full_direct: expert backward plus router_w backward and the router dx branch.
* full_derived: (fresh forward + backward) - forward, the Sonic reporting style.

All reported timings use batched HIP events.  Metadata construction and module
compilation happen before the timed region.  Defaults are 5 warmups/500 repeats.
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import torch

from aiter.ops.opus import moe_bwd as opus
from aiter.ops.triton import moe_bwd_ref as triton_ref


BF16 = torch.bfloat16
ACT = opus.SONIC_SWIGLU


def cuda_time_ms(fn: Callable[[], object], warmups: int, repeats: int) -> float:
    """Return mean GPU time using one event interval around all repeats."""
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def make_topk_ids(T: int, E: int, topk: int, device: torch.device) -> torch.Tensor:
    token = torch.arange(T, device=device, dtype=torch.int64)[:, None]
    rank = torch.arange(topk, device=device, dtype=torch.int64)[None, :]
    return ((token * topk + rank) % E).to(torch.int64)


def make_inputs(args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    T, H, I, E, topk = args.tokens, args.hidden, args.inter, args.experts, args.topk

    x = (0.2 * torch.randn(T, H, device=device, dtype=BF16)).contiguous()
    w1 = (0.02 * torch.randn(E, 2 * I, H, device=device, dtype=BF16)).contiguous()
    w2 = (0.02 * torch.randn(E, H, I, device=device, dtype=BF16)).contiguous()
    dout = (0.2 * torch.randn(T, H, device=device, dtype=BF16)).contiguous()

    if args.routing == "balanced":
        # Make F.linear(x, router_w) select exactly topk routes/token with equal
        # expert counts.  Only the first E input columns are reserved for the
        # synthetic logits; expert GEMM shapes and memory traffic are unchanged.
        if H < E:
            raise ValueError("balanced full routing requires hidden >= experts")
        topk_ids = make_topk_ids(T, E, topk, device)
        x[:, :E] = -4.0
        values = torch.linspace(4.0, 3.0, topk, device=device, dtype=BF16)
        x[:, :E].scatter_(1, topk_ids, values[None, :].expand(T, -1))
        router_w = torch.zeros(E, H, device=device, dtype=BF16)
        router_w[:, :E] = torch.eye(E, device=device, dtype=BF16)
        router_logits = x[:, :E].clone()
    else:
        router_w = (0.02 * torch.randn(E, H, device=device, dtype=BF16)).contiguous()
        router_logits = torch.nn.functional.linear(x, router_w).detach()

    return {
        "x": x,
        "w1": w1,
        "w2": w2,
        "router_w": router_w.contiguous(),
        "router_logits": router_logits.contiguous(),
        "dout": dout,
    }


def leaf_clones(tensors: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    return tuple(t.detach().clone().requires_grad_(True) for t in tensors)


def direct_backward_ms(
    fn: Callable[..., torch.Tensor],
    inputs: Tuple[torch.Tensor, ...],
    dout: torch.Tensor,
    topk: int,
    warmups: int,
    repeats: int,
) -> float:
    leaves = leaf_clones(inputs)
    out = fn(*leaves, topk, ACT)
    # Compile/autotune and populate allocator caches before the timed warmups.
    torch.autograd.grad(out, leaves, dout, retain_graph=True)

    def backward() -> object:
        return torch.autograd.grad(out, leaves, dout, retain_graph=True)

    return cuda_time_ms(backward, warmups, repeats)


def forward_and_derived_ms(
    fn: Callable[..., torch.Tensor],
    inputs: Tuple[torch.Tensor, ...],
    dout: torch.Tensor,
    topk: int,
    warmups: int,
    repeats: int,
) -> Tuple[float, float, float]:
    leaves = leaf_clones(inputs)

    def forward() -> object:
        return fn(*leaves, topk, ACT)

    def forward_backward() -> object:
        out = fn(*leaves, topk, ACT)
        return torch.autograd.grad(out, leaves, dout)

    # Exercise both paths once before their independent batched measurements.
    forward()
    forward_backward()
    fwd_ms = cuda_time_ms(forward, warmups, repeats)
    fwd_bwd_ms = cuda_time_ms(forward_backward, warmups, repeats)
    return fwd_ms, fwd_bwd_ms, fwd_bwd_ms - fwd_ms


def component_benchmark(
    backend: str,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, float]]:
    """Kernel-oriented expert breakdown with balanced, prebuilt route metadata."""
    T, H, I, E, topk = args.tokens, args.hidden, args.inter, args.experts, args.topk
    M = T * topk
    if M % E:
        raise ValueError("component benchmark requires T*topk divisible by E")
    device = torch.device("cuda")
    per_expert = M // E
    offs = (torch.arange(E + 1, device=device) * per_expert).to(torch.int32)
    lens = (offs[1:] - offs[:-1]).contiguous()

    dy = torch.randn(M, H, device=device, dtype=BF16)
    h = torch.randn(M, I, device=device, dtype=BF16)
    xg = torch.randn(M, H, device=device, dtype=BF16)
    dact = torch.randn(M, 2 * I, device=device, dtype=BF16)
    actin = torch.randn(M, 2 * I, device=device, dtype=BF16)
    w1 = (0.02 * torch.randn(E, 2 * I, H, device=device, dtype=BF16)).contiguous()
    w2 = (0.02 * torch.randn(E, H, I, device=device, dtype=BF16)).contiguous()

    if backend == "opus":
        seid, bms, bme = opus.build_dgrad_block_meta(offs, 128)
        w1t = w1.transpose(1, 2).contiguous()
        w2t = w2.transpose(1, 2).contiguous()
        dh = torch.empty(M, I, device=device, dtype=BF16)
        dx_route = torch.empty(M, H, device=device, dtype=BF16)
        dw1 = torch.empty(E, 2 * I, H, device=device, dtype=BF16)
        dw2 = torch.empty(E, H, I, device=device, dtype=BF16)

        calls = {
            "stage2_dgrad": lambda: opus.opus_moe_dgrad_uniform_prepared(
                dy, w2t, per_expert, dh
            ),
            "dW2": lambda: opus._opus_moe_wgrad_tn_bf16_raw(dy, h, offs, dw2),
            "activation": lambda: opus.opus_moe_act_bwd_bf16(dh, actin, ACT),
            "stage1_dgrad": lambda: opus.opus_moe_dgrad_uniform_prepared(
                dact, w1t, per_expert, dx_route
            ),
            "dW1": lambda: opus._opus_moe_wgrad_tn_bf16_raw(dact, xg, offs, dw1),
        }
    else:
        calls = {
            "stage2_dgrad": lambda: triton_ref._dgrad(dy, w2, lens),
            "dW2": lambda: triton_ref._wgrad(dy, h, lens),
            "activation": lambda: triton_ref.act_bwd_triton(
                torch.empty(M, I, device=device, dtype=BF16), actin, ACT
            ),
            "stage1_dgrad": lambda: triton_ref._dgrad(dact, w1, lens),
            "dW1": lambda: triton_ref._wgrad(dact, xg, lens),
        }

    gemm_flops = {
        "stage2_dgrad": 2 * M * H * I,
        "dW2": 2 * M * H * I,
        "stage1_dgrad": 4 * M * H * I,
        "dW1": 4 * M * H * I,
    }
    results: Dict[str, Dict[str, float]] = {}
    for name, call in calls.items():
        ms = cuda_time_ms(call, args.warmups, args.repeats)
        item = {"ms": ms}
        if name in gemm_flops:
            item["tflops"] = gemm_flops[name] / (ms * 1.0e9)
        results[name] = item
    return results


def benchmark_backend(
    backend: str,
    data: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Dict[str, object]:
    if backend == "opus":
        expert_fn, full_fn = opus.opus_moe_ref, opus.opus_moe
    else:
        expert_fn, full_fn = triton_ref.triton_moe_ref, triton_ref.triton_moe

    expert_inputs = (data["x"], data["w1"], data["w2"], data["router_logits"])
    full_inputs = (data["x"], data["w1"], data["w2"], data["router_w"])
    result: Dict[str, object] = {}
    if "components" in args.sections:
        result["components"] = component_benchmark(backend, args)
    if "expert" in args.sections:
        result["expert_direct_ms"] = direct_backward_ms(
            expert_fn, expert_inputs, data["dout"], args.topk, args.warmups, args.repeats
        )
    if "full" in args.sections:
        result["full_direct_ms"] = direct_backward_ms(
            full_fn, full_inputs, data["dout"], args.topk, args.warmups, args.repeats
        )
    if "derived" in args.sections:
        fwd, both, derived = forward_and_derived_ms(
            full_fn, full_inputs, data["dout"], args.topk, args.warmups, args.repeats
        )
        result.update(
            full_forward_ms=fwd,
            full_forward_backward_ms=both,
            full_derived_backward_ms=derived,
        )
    return result


def print_backend(name: str, result: Dict[str, object], backward_flops: int) -> None:
    print(f"\n{name}")
    components = result.get("components")
    if isinstance(components, dict):
        for component, item in components.items():
            suffix = f", {item['tflops']:.1f} TFLOP/s" if "tflops" in item else ""
            print(f"  {component:20s} {item['ms']:9.4f} ms{suffix}")
    for key, label in (
        ("expert_direct_ms", "expert-only direct bwd"),
        ("full_direct_ms", "complete MoE direct bwd"),
        ("full_derived_backward_ms", "complete MoE derived bwd"),
    ):
        if key in result:
            ms = float(result[key])
            print(f"  {label:27s} {ms:9.4f} ms, {backward_flops/(ms*1e9):.1f} TFLOP/s")
    if "full_forward_ms" in result:
        print(f"  {'complete MoE forward':27s} {result['full_forward_ms']:9.4f} ms")
        print(f"  {'complete MoE fwd+bwd':27s} {result['full_forward_backward_ms']:9.4f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--inter", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--routing", choices=("balanced", "natural"), default="balanced")
    parser.add_argument("--backends", nargs="+", choices=("opus", "triton"), default=["opus", "triton"])
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("components", "expert", "full", "derived"),
        default=["components", "expert", "full", "derived"],
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(torch.cuda.current_device())
    data = make_inputs(args)
    backward_flops = 12 * args.tokens * args.hidden * args.inter * args.topk
    print(
        f"shape=({args.tokens},{args.hidden},{args.inter},{args.experts},{args.topk}) "
        f"dtype=BF16 activation=standard-SwiGLU routing={args.routing}"
    )
    print(
        f"backward_flops={backward_flops/1e12:.9f} TFLOP, "
        f"warmups={args.warmups}, repeats={args.repeats}"
    )

    report: Dict[str, object] = {
        "config": {
            "tokens": args.tokens,
            "hidden": args.hidden,
            "inter": args.inter,
            "experts": args.experts,
            "topk": args.topk,
            "dtype": "bfloat16",
            "activation": "standard_swiglu",
            "routing": args.routing,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "backward_flops": backward_flops,
            "device": torch.cuda.get_device_name(),
        },
        "backends": {},
    }
    for backend in args.backends:
        result = benchmark_backend(backend, data, args)
        report["backends"][backend] = result
        print_backend(backend, result, backward_flops)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nJSON: {args.json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Correctness tests for SonicMoE grouped GEMM MoE layer.

Tests moe_TC_softmax_topk_layer forward and backward against a PyTorch reference
for all supported activation types.

Usage:
    python op_tests/test_sonicmoe.py
    python op_tests/test_sonicmoe.py --activation swiglu
    python op_tests/test_sonicmoe.py --benchmark --T 4096 --H 4096 --I 2048 --E 64 --K 8
"""

import argparse

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from aiter.ops.triton.sonicmoe import (
    SonicMoEActivationType,
    moe_TC_softmax_topk_layer,
    sonicmoe_is_glu,
)

_ACT_MAP = {
    "swiglu": SonicMoEActivationType.SWIGLU,
    "geglu": SonicMoEActivationType.GEGLU,
    "reglu": SonicMoEActivationType.REGLU,
    "gelu": SonicMoEActivationType.GELU,
    "relu": SonicMoEActivationType.RELU,
    "silu": SonicMoEActivationType.SILU,
    "relu_sq": SonicMoEActivationType.RELU_SQ,
}


@torch.autocast(device_type="cuda", dtype=torch.float32)
def _ref_activation(h, activation_type):
    if sonicmoe_is_glu(activation_type):
        g = h[:, ::2]
        u = h[:, 1::2]
        if activation_type == SonicMoEActivationType.SWIGLU:
            return F.silu(g) * u
        elif activation_type == SonicMoEActivationType.GEGLU:
            return (F.gelu(g.float(), approximate="tanh") * u).to(g.dtype)
        elif activation_type == SonicMoEActivationType.REGLU:
            return (F.relu(g) * u).to(g.dtype)
    else:
        if activation_type == SonicMoEActivationType.GELU:
            return F.gelu(h.float(), approximate="tanh").to(h.dtype)
        elif activation_type == SonicMoEActivationType.RELU:
            return F.relu(h)
        elif activation_type == SonicMoEActivationType.SILU:
            return F.silu(h)
        elif activation_type == SonicMoEActivationType.RELU_SQ:
            return (F.relu(h) ** 2).to(h.dtype)
    raise ValueError(f"Unknown activation: {activation_type}")


def ref_moe_topk(x, router_w, w1_orig, b1, w2_orig, b2, K, act_type):
    T, H = x.shape
    E = router_w.shape[0]

    router_logits = F.linear(x, router_w)
    top_logits, topk_indices = router_logits.topk(K, dim=1)
    topk_scores = F.softmax(top_logits, dim=-1, dtype=torch.float32)

    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    expert_freq = torch.zeros(E, dtype=torch.int32, device=x.device)

    for e in range(E):
        mask = topk_indices == e
        if not mask.any():
            continue
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        expert_freq[e] = token_idx.shape[0]
        scores = topk_scores[token_idx, slot_idx]

        h = F.linear(x[token_idx], w1_orig[e], bias=(b1[e] if b1 is not None else None))
        a = _ref_activation(h, act_type)
        y = F.linear(a, w2_orig[e], bias=(b2[e] if b2 is not None else None))
        out[token_idx] += y * scores.unsqueeze(-1)

    return out, router_logits, expert_freq


def run_correctness(T, H, I, E, K, act_type, dtype):
    x = 0.2 * torch.randn(T, H, device="cuda", dtype=dtype, requires_grad=True)
    router_w = torch.randn(E, H, device="cuda", dtype=dtype)
    I_full = 2 * I if sonicmoe_is_glu(act_type) else I
    w1_orig = torch.randn(E, I_full, H, device="cuda", dtype=dtype)
    w2_orig = torch.randn(E, H, I, device="cuda", dtype=dtype)
    b1 = None
    b2 = None

    torch.nn.init.normal_(w1_orig, 0, 0.02)
    torch.nn.init.normal_(w2_orig, 0, 0.02)
    torch.nn.init.normal_(router_w, 0, 0.02)

    w1_orig.requires_grad_(True)
    w2_orig.requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    dout = 0.2 * torch.randn_like(x)

    w1 = w1_orig.permute(1, 2, 0).contiguous().requires_grad_(True)
    w2 = w2_orig.permute(1, 2, 0).contiguous().requires_grad_(True)

    stream_id = torch.cuda.current_stream().cuda_stream

    o, _router_logits, _expert_freq = moe_TC_softmax_topk_layer(
        x,
        router_w,
        w1,
        b1,
        w2,
        b2,
        K,
        stream_id,
        act_type,
        False,
    )

    ref_o, _ref_logits, _ref_freq = ref_moe_topk(
        x_ref,
        router_w,
        w1_orig,
        b1,
        w2_orig,
        b2,
        K,
        act_type,
    )

    def rel_err(a, b):
        return ((a.float() - b.float()).abs() / (b.float().abs() + 1e-6)).mean().item()

    o_err = rel_err(o, ref_o)
    print(f"  output rel err: {o_err:.4f}")

    dx, dw1, dw2 = torch.autograd.grad(o, [x, w1, w2], grad_outputs=dout)
    ref_dx, ref_dw1, ref_dw2 = torch.autograd.grad(
        ref_o,
        [x_ref, w1_orig, w2_orig],
        grad_outputs=dout,
    )

    dx_err = rel_err(dx, ref_dx)
    dw1_err = rel_err(dw1, ref_dw1.permute(1, 2, 0))
    dw2_err = rel_err(dw2, ref_dw2.permute(1, 2, 0))

    print(f"  dx rel err:     {dx_err:.4f}")
    print(f"  dw1 rel err:    {dw1_err:.4f}")
    print(f"  dw2 rel err:    {dw2_err:.4f}")

    status = "PASS" if max(o_err, dx_err, dw1_err, dw2_err) < 0.05 else "FAIL"
    print(f"  Status: {status}")
    return o_err, dx_err, dw1_err, dw2_err, status


def run_benchmark(T, H, I, E, K, act_type, dtype, rep=100, trials=5):
    I_full = 2 * I if sonicmoe_is_glu(act_type) else I

    w1_orig = torch.empty(E, I_full, H, device="cuda", dtype=dtype)
    w2_orig = torch.empty(E, H, I, device="cuda", dtype=dtype)
    router_w = torch.randn(E, H, device="cuda", dtype=dtype)
    torch.nn.init.normal_(w1_orig, 0, 0.02)
    torch.nn.init.normal_(w2_orig, 0, 0.02)
    torch.nn.init.normal_(router_w, 0, 0.02)
    w1_orig.requires_grad_(True)
    w2_orig.requires_grad_(True)
    w1 = w1_orig.permute(1, 2, 0)
    w2 = w2_orig.permute(1, 2, 0)

    stream_id = torch.cuda.current_stream().cuda_stream
    flops_mul = 4 if sonicmoe_is_glu(act_type) else 2
    TK = T * K

    x = 0.2 * torch.randn(T, H, device="cuda", dtype=dtype, requires_grad=True)
    dout = 0.2 * torch.randn_like(x)

    # warmup
    o, _, _ = moe_TC_softmax_topk_layer(
        x,
        router_w,
        w1,
        None,
        w2,
        None,
        K,
        stream_id,
        act_type,
        False,
    )
    torch.autograd.grad(o, [x, w1_orig, w2_orig], dout)
    x.grad = w1_orig.grad = w2_orig.grad = None

    fwd_ms = do_bench(
        lambda: moe_TC_softmax_topk_layer(
            x,
            router_w,
            w1,
            None,
            w2,
            None,
            K,
            stream_id,
            act_type,
            False,
        ),
        warmup=10,
        rep=rep,
    )
    fwd_flops = flops_mul * 2 * TK * I * H
    fwd_tflops = fwd_flops / (fwd_ms / 1e3) / 1e12

    def e2e_fn():
        o, _, _ = moe_TC_softmax_topk_layer(
            x,
            router_w,
            w1,
            None,
            w2,
            None,
            K,
            stream_id,
            act_type,
            False,
        )
        torch.autograd.grad(o, [x, w1_orig, w2_orig], dout, retain_graph=False)
        x.grad = w1_orig.grad = w2_orig.grad = None

    e2e_ms = do_bench(e2e_fn, warmup=10, rep=rep, grad_to_none=[x, w1_orig, w2_orig])
    e2e_flops = flops_mul * 6 * TK * I * H
    e2e_tflops = e2e_flops / (e2e_ms / 1e3) / 1e12

    bwd_ms = e2e_ms - fwd_ms
    bwd_flops = flops_mul * 4 * TK * I * H
    bwd_tflops = bwd_flops / (bwd_ms / 1e3) / 1e12

    print(f"Config: T={T}, H={H}, I={I}, E={E}, K={K}, act={act_type.value}")
    print(f"  [Fwd] {fwd_ms:.2f} ms, {fwd_tflops:.1f} TFLOPS")
    print(f"  [Bwd] {bwd_ms:.2f} ms, {bwd_tflops:.1f} TFLOPS")
    print(f"  [E2E] {e2e_ms:.2f} ms, {e2e_tflops:.1f} TFLOPS")


def main():
    parser = argparse.ArgumentParser(description="Test SonicMoE integration")
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--I", type=int, default=64)
    parser.add_argument("--E", type=int, default=4)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument(
        "--activation", type=str, default=None, choices=list(_ACT_MAP.keys())
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark instead of correctness test",
    )
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    if args.benchmark:
        act_type = _ACT_MAP[args.activation or "swiglu"]
        run_benchmark(args.T, args.H, args.I, args.E, args.K, act_type, torch_dtype)
        return

    print("=== SonicMoE Correctness Test ===")
    from aiter.ops.triton.utils.sonicmoe_config_utils import (
        get_grouped_gemm_dw_config,
        get_grouped_gemm_fwd_config,
        get_token_gather_config,
        load_sonicmoe_configs,
    )

    sonic_cfgs = load_sonicmoe_configs()
    if sonic_cfgs is None:
        print("SONICMOE JSON: not found for this arch (will autotune)")
    else:
        large_fwd = get_grouped_gemm_fwd_config(4096, 4096, 64)
        large_dw = get_grouped_gemm_dw_config(4096, 4096, 64)
        large_g = get_token_gather_config(4096)
        small_fwd = get_grouped_gemm_fwd_config(128, 128, 4)
        assert large_fwd is not None and large_fwd["BLOCK_M"] == 128
        assert large_fwd["BLOCK_N"] == 128 and large_fwd["BLOCK_K"] == 64
        assert large_fwd["num_warps"] == 4
        assert large_dw is not None and large_dw["BLOCK_K"] == 128
        assert large_g is not None and large_g["BLOCK_H"] == 4096
        assert small_fwd is not None and small_fwd["BLOCK_M"] == 64
        print("SONICMOE JSON: gfx bucket selection OK")

    activations = [args.activation] if args.activation else list(_ACT_MAP.keys())
    all_pass = True
    for act_name in activations:
        at = _ACT_MAP[act_name]
        print(f"\n{act_name}:")
        _, _, _, _, status = run_correctness(
            args.T, args.H, args.I, args.E, args.K, at, torch_dtype
        )
        if status != "PASS":
            all_pass = False

    print(f"\n{'All tests PASSED' if all_pass else 'Some tests FAILED'}.")


if __name__ == "__main__":
    main()

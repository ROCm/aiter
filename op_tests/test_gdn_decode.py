import argparse
import os
import statistics
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from aiter.ops.hip.gated_delta_net import hip_fused_sigmoid_gating_delta_rule_update


@dataclass
class GDNDecodeArgs:
    dtype: torch.dtype
    batch_size: int
    seq_len: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    use_qk_l2norm: bool = True


DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DEFAULT_WARMUP = 20
DEFAULT_ITERS = 100
ONLINE_POOL_SIZE = 256
ONLINE_HIDDEN_DIM = 640
ONLINE_CONV_KERNEL_SIZE = 4
KERNEL_NAME_SUBSTR = "gdn_decode_kernel"


def create_inputs(args: GDNDecodeArgs, pool_size: int | None = None):
    pool_size = max(pool_size or args.batch_size, args.batch_size)
    q_shape = (args.batch_size, args.seq_len, args.num_k_heads, args.head_k_dim)
    v_shape = (args.batch_size, args.seq_len, args.num_v_heads, args.head_v_dim)

    query = torch.randn(q_shape, dtype=args.dtype, device="cuda")
    key = torch.randn(q_shape, dtype=args.dtype, device="cuda")
    value = torch.randn(v_shape, dtype=args.dtype, device="cuda")
    a = torch.randn(
        (args.batch_size, args.seq_len, args.num_v_heads),
        dtype=args.dtype,
        device="cuda",
    )
    b = torch.randn_like(a)
    dt_bias = torch.randn((args.num_v_heads,), dtype=args.dtype, device="cuda")
    dt_bias.uniform_(1, 2)
    a_log = torch.randn((args.num_v_heads,), dtype=torch.float32, device="cuda")
    a_log.uniform_(0, 16)
    indices = torch.randperm(pool_size, device="cuda")[: args.batch_size].to(
        torch.int32
    )
    state = torch.randn(
        (pool_size, args.num_v_heads, args.head_k_dim, args.head_v_dim),
        dtype=torch.float32,
        device="cuda",
    )
    out = torch.empty(v_shape, dtype=args.dtype, device="cuda")
    return query, key, value, a, b, dt_bias, a_log, indices, state, out


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_gdn_decode_reference(
    args: GDNDecodeArgs,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    indices: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
) -> None:
    beta = b.sigmoid()
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias, beta=1.0, threshold=20.0)

    if args.num_v_heads // args.num_k_heads > 1:
        repeat = args.num_v_heads // args.num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    if args.use_qk_l2norm:
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

    query = query.transpose(1, 2).contiguous().to(torch.float32)
    key = key.transpose(1, 2).contiguous().to(torch.float32)
    value = value.transpose(1, 2).contiguous().to(torch.float32)
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)
    g = g.transpose(1, 2).contiguous().to(torch.float32)

    query = query * (args.head_k_dim**-0.5)
    recurrent_state = state[indices.to(torch.long)]

    for token_idx in range(args.seq_len):
        q_t = query[:, :, token_idx]
        k_t = key[:, :, token_idx]
        v_t = value[:, :, token_idx]
        g_t = g[:, :, token_idx].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, token_idx].unsqueeze(-1)

        recurrent_state = recurrent_state * g_t
        kv_mem = (recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        recurrent_state = recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, token_idx] = (recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    state[indices.to(torch.long)] = recurrent_state


def hip_gdn_decode(
    args: GDNDecodeArgs,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    indices: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
) -> None:
    state_vk = state.permute(0, 1, 3, 2).contiguous()
    result = hip_fused_sigmoid_gating_delta_rule_update(
        a_log,
        a,
        dt_bias,
        1.0,
        20.0,
        query,
        key,
        value,
        b,
        state_vk,
        indices,
        scale=float(args.head_k_dim**-0.5),
        use_qk_l2norm_in_kernel=args.use_qk_l2norm,
    )
    out.copy_(result)
    state.copy_(state_vk.permute(0, 1, 3, 2).contiguous())


def clone_case(inputs):
    query, key, value, a, b, dt_bias, a_log, indices, state, out = inputs
    return (
        query.clone(),
        key.clone(),
        value.clone(),
        a.clone(),
        b.clone(),
        dt_bias.clone(),
        a_log.clone(),
        indices.clone(),
        state.clone(),
        torch.empty_like(out),
    )


def validate_case(args: GDNDecodeArgs, mode: str) -> tuple[float, float]:
    pool_size = ONLINE_POOL_SIZE if mode == "online" else args.batch_size
    base_inputs = create_inputs(args, pool_size=pool_size)
    hip_inputs = clone_case(base_inputs)
    ref_inputs = clone_case(base_inputs)

    hip_gdn_decode(args, *hip_inputs)
    torch_gdn_decode_reference(args, *ref_inputs)
    torch.cuda.synchronize()

    max_out_diff = (hip_inputs[-1] - ref_inputs[-1]).abs().max().item()
    max_state_diff = (hip_inputs[-2] - ref_inputs[-2]).abs().max().item()
    return max_out_diff, max_state_diff


def make_online_companion(args: GDNDecodeArgs):
    inproj_out = (
        args.num_k_heads * args.head_k_dim * 2 + args.num_v_heads * args.head_v_dim
    )
    hidden = torch.randn(
        (args.batch_size, ONLINE_HIDDEN_DIM), dtype=args.dtype, device="cuda"
    )
    qkv_weight = torch.randn(
        (ONLINE_HIDDEN_DIM, inproj_out), dtype=args.dtype, device="cuda"
    )
    out_weight = torch.randn(
        (args.num_v_heads * args.head_v_dim, ONLINE_HIDDEN_DIM),
        dtype=args.dtype,
        device="cuda",
    )
    conv_state = torch.randn(
        (args.batch_size, inproj_out, ONLINE_CONV_KERNEL_SIZE),
        dtype=args.dtype,
        device="cuda",
    )
    conv_weight = torch.randn(
        (ONLINE_CONV_KERNEL_SIZE,), dtype=args.dtype, device="cuda"
    )
    return hidden, qkv_weight, out_weight, conv_state, conv_weight


def run_online_step(args: GDNDecodeArgs, inputs, companion) -> None:
    hidden, qkv_weight, out_weight, conv_state, conv_weight = companion
    _ = torch.matmul(hidden, qkv_weight)
    _ = (conv_state * conv_weight).sum(-1)
    hip_gdn_decode(args, *inputs)
    out = inputs[-1]
    _ = out.float() * torch.rsqrt(out.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    _ = torch.matmul(out.reshape(args.batch_size, -1), out_weight)


def benchmark_case(
    args: GDNDecodeArgs, mode: str, warmup: int, iterations: int
) -> tuple[float, float, float]:
    pool_size = ONLINE_POOL_SIZE if mode == "online" else args.batch_size
    inputs = list(create_inputs(args, pool_size=pool_size))
    companion = make_online_companion(args) if mode == "online" else None

    def step() -> None:
        if companion is None:
            hip_gdn_decode(args, *inputs)
        else:
            run_online_step(args, inputs, companion)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            step()
    torch.cuda.synchronize()

    samples = []
    for event in prof.events():
        event_name = getattr(event, "name", "") or ""
        device_time = getattr(event, "device_time_total", 0)
        if KERNEL_NAME_SUBSTR in event_name and device_time > 0:
            samples.append(device_time)

    if not samples:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)

    samples.sort()
    return min(samples), statistics.median(samples), max(samples)


def parse_args():
    parser = argparse.ArgumentParser(description="HIP inline-ASM GDN decode test")
    parser.add_argument(
        "--mode",
        choices=["default", "online"],
        default="default",
        help="default tests the operator; online adds serving-like neighbor ops",
    )
    parser.add_argument("--bs", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    gpu = os.environ.get(
        "HIP_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    )
    print(f"=== HIP inline-ASM GDN decode | GPU {gpu} | mode={cli_args.mode} ===")

    all_passed = True
    for batch_size in cli_args.bs:
        args = GDNDecodeArgs(
            dtype=torch.bfloat16,
            batch_size=batch_size,
            seq_len=1,
            num_k_heads=2,
            num_v_heads=8,
            head_k_dim=128,
            head_v_dim=128,
        )
        out_diff, state_diff = validate_case(args, cli_args.mode)
        passed = out_diff <= cli_args.atol and state_diff <= cli_args.rtol
        all_passed = all_passed and passed
        min_us, med_us, max_us = benchmark_case(
            args, cli_args.mode, cli_args.warmup, cli_args.iters
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"BS={batch_size:>4d} {status} "
            f"max_out={out_diff:.6f} max_state={state_diff:.6f} "
            f"hip_us={med_us:.2f} [{min_us:.2f}~{max_us:.2f}]"
        )

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

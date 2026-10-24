# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import json
import sys

import torch
import triton

from aiter.ops.triton.attention.chunk_kda import (
    CHUNK_SIZE,
    chunk_kda,
    chunk_kda_prepare,
    chunk_kda_walk,
    prepare_chunk_kda_metadata,
)
from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn
from aiter.ops.triton.utils._triton import arch_info
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
)

K3_HEAD_DIM = 128
K3_NUM_HEADS = [24, 12]  # 96 heads sharded tp4 / tp8
K3_LOWER_BOUND = -5.0


def make_inputs(B, T, H, device):
    """B sequences of T tokens in vLLM's layout: dense q/k/v/g, beta a column of in_proj."""
    D, total = K3_HEAD_DIM, B * T
    width = 4 * H * D + D + H
    width += -width % 16
    in_proj = torch.randn(1, total, width, dtype=torch.bfloat16, device=device)

    def dense():
        return torch.randn(1, total, H, D, dtype=torch.bfloat16, device=device)

    return {
        "q": dense(),
        "k": dense(),
        "v": dense(),
        "g": dense(),
        "beta": in_proj[..., 4 * H * D + D : 4 * H * D + D + H],
        "A_log": torch.log(
            torch.empty(H, dtype=torch.float32, device=device).uniform_(1, 16)
        ),
        "dt_bias": torch.randn(H * D, dtype=torch.float32, device=device),
        "cu_seqlens": torch.arange(0, total + 1, T, dtype=torch.int32, device=device),
    }


def traffic_bytes(B, T, H, stage, device, state_dtype=torch.float32):
    D = K3_HEAD_DIM
    tok = B * T * H * D * 2  # one [T, H, 128] bf16 tensor
    nt = B * triton.cdiv(T, CHUNK_SIZE) * H
    ws = (
        3 * tok + tok // 2 + nt * D * (CHUNK_SIZE * 2 + 4)
    )  # qg, w, u, aqk, kg_t, decay
    state = (
        B
        * H
        * D
        * D
        * torch.tensor([], dtype=state_dtype, device=device).element_size()
    )
    prepare = 4 * tok + B * T * H * 2 + ws
    walk = ws + tok + 2 * state
    return {"prepare": prepare, "walk": walk, "total": prepare + walk}[stage]


def _generate_kda_pool_inputs(B, T, H, device):
    """Build one dense q/k/v/g/beta/state/out set.

    Pure setup, not measured. Each call is one rotating-buffer pool entry:
    independent random projections and state, so replaying through the pool
    during timing defeats cache residency instead of timing one fixed buffer
    repeatedly. cu_seqlens/chunk_indices/chunk_offsets stay fixed outside the
    pool -- they are deterministic given (B, T), and prepare_chunk_kda_metadata
    syncs the host, which a graph capture rejects.
    """
    inp = make_inputs(B, T, H, device)
    state = torch.randn(B, H, K3_HEAD_DIM, K3_HEAD_DIM, device=device)
    out = torch.empty_like(inp["v"])
    return (
        inp["q"],
        inp["k"],
        inp["v"],
        inp["g"],
        inp["beta"],
        inp["A_log"],
        inp["dt_bias"],
        state,
        out,
    )


_KDA_POOL_FIELDS = ["q", "k", "v", "g", "beta", "A_log", "dt_bias", "state", "out"]


def _generate_kda_walk_pool_inputs(B, T, H, device, cu_seqlens, chunk_indices):
    """Build one walk workspace via an untimed chunk_kda_prepare call."""
    inp = make_inputs(B, T, H, device)
    ws = chunk_kda_prepare(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        lower_bound=K3_LOWER_BOUND,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    state = torch.randn(B, H, K3_HEAD_DIM, K3_HEAD_DIM, device=device)
    out = torch.empty_like(ws["u"])
    return ws["qg"], ws["w"], ws["u"], ws["kg_t"], ws["aqk"], ws["decay"], state, out


_KDA_WALK_POOL_FIELDS = ["qg", "w", "u", "kg_t", "aqk", "decay", "state", "out"]


def _make_gluon_kernel_fn(
    cu_seqlens, chunk_indices, chunk_offsets, state_indices, has_initial_state, config
):
    def kernel_fn(q, k, v, g, beta, A_log, dt_bias, state, out):
        chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=K3_LOWER_BOUND,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            out=out,
            config=config,
            state_cache=state,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
        )

    return kernel_fn


def _make_prepare_kernel_fn(cu_seqlens, chunk_indices):
    def kernel_fn(q, k, v, g, beta, A_log, dt_bias, **_):
        chunk_kda_prepare(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=K3_LOWER_BOUND,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    return kernel_fn


def _make_walk_kernel_fn(
    cu_seqlens, chunk_offsets, state_indices, has_initial_state, config
):
    def kernel_fn(qg, w, u, kg_t, aqk, decay, state, out):
        chunk_kda_walk(
            qg=qg,
            w=w,
            u=u,
            kg_t=kg_t,
            aqk=aqk,
            decay=decay,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            out=out,
            config=config,
            state_cache=state,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
        )

    return kernel_fn


def _make_triton_kernel_fn(cu_seqlens, state_indices, has_initial_state):
    def kernel_fn(q, k, v, g, beta, A_log, dt_bias, state, out):
        chunk_kimi_delta_attn(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=K3_LOWER_BOUND,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
            out=out,
            state_cache=state,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
        )

    return kernel_fn


def benchmark(args):
    state_dtype = torch.bfloat16 if args.state_dtype == "bf16" else torch.float32
    lines = ["gluon", "gluon_prepare"] + [f"gluon_walk:{c}" for c in args.walk_configs]
    if "triton" in args.backends:
        lines.append("triton")
    configs = [
        triton.testing.Benchmark(
            x_names=["H", "B", "T"],
            x_vals=[
                (h, b, t)
                for h in args.num_heads
                for b in args.batch_sizes
                for t in args.seq_lens
            ],
            line_arg="provider",
            line_vals=lines,
            line_names=lines,
            plot_name=get_caller_name_no_ext(),
            styles=[],
            ylabel="us" if args.metric == "time" else "TB/s",
            args={},
        )
    ]

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    @triton.testing.perf_report(configs)
    def bench_chunk_kda(H, B, T, provider):
        torch.manual_seed(0)
        device = args.device
        total = B * T
        # Deterministic given (B, T), so computed once and shared by every
        # pool entry rather than rotated.
        cu_seqlens = torch.arange(0, total + 1, T, dtype=torch.int32, device=device)
        state_indices = torch.arange(B, dtype=torch.int32, device=device)
        has_initial_state = torch.ones(B, dtype=torch.bool, device=device)
        # built once: the metadata syncs the host, which a graph capture rejects
        ci, co = prepare_chunk_kda_metadata(cu_seqlens)
        stage = "total"

        if provider == "gluon":
            pool_gen = functools.partial(_generate_kda_pool_inputs, B, T, H, device)
            rot_bufs = generate_rotating_buffers_pool(
                [pool_gen], [_KDA_POOL_FIELDS], **kwargs_rot_bufs
            )
            kernel_fn = _make_gluon_kernel_fn(
                cu_seqlens, ci, co, state_indices, has_initial_state, args.config
            )
        elif provider == "gluon_prepare":
            stage = "prepare"
            pool_gen = functools.partial(_generate_kda_pool_inputs, B, T, H, device)
            rot_bufs = generate_rotating_buffers_pool(
                [pool_gen], [_KDA_POOL_FIELDS], **kwargs_rot_bufs
            )
            kernel_fn = _make_prepare_kernel_fn(cu_seqlens, ci)
        elif provider.startswith("gluon_walk:"):
            stage = "walk"
            bv, nw = (int(x) for x in provider.split(":")[1].split(","))
            walk_config = {**args.config, "BV": bv, "num_warps": nw}
            pool_gen = functools.partial(
                _generate_kda_walk_pool_inputs, B, T, H, device, cu_seqlens, ci
            )
            rot_bufs = generate_rotating_buffers_pool(
                [pool_gen], [_KDA_WALK_POOL_FIELDS], **kwargs_rot_bufs
            )
            kernel_fn = _make_walk_kernel_fn(
                cu_seqlens, co, state_indices, has_initial_state, walk_config
            )
        else:
            pool_gen = functools.partial(_generate_kda_pool_inputs, B, T, H, device)
            rot_bufs = generate_rotating_buffers_pool(
                [pool_gen], [_KDA_POOL_FIELDS], **kwargs_rot_bufs
            )
            kernel_fn = _make_triton_kernel_fn(
                cu_seqlens, state_indices, has_initial_state
            )

        ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)
        if args.metric == "time":
            return ms * 1e3
        return traffic_bytes(B, T, H, stage, args.device, state_dtype) / ms * 1e-9

    bench_chunk_kda.run(
        save_path="." if args.o else None, print_data=True, show_plots=False
    )


def parse_args():
    parser = get_parser(kernel_name="chunked KDA prefill")
    override_parser_action(
        parser, action="--metric", choices=["time", "throughput"], default="time"
    )
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 3, 8])
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--num_heads", type=int, nargs="+", default=K3_NUM_HEADS)
    parser.add_argument("--state_dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument(
        "--walk_configs",
        nargs="+",
        default=None,
        help="walk BV,num_warps points to time on their own",
    )
    parser.add_argument("--backends", nargs="+", default=["gluon", "triton"])
    parser.add_argument(
        "--config",
        type=json.loads,
        default={},
        help="json overrides of the tuned walk config",
    )
    parser.add_argument(
        "--timing",
        choices=["cudagraph", "do_bench"],
        default="do_bench",
        help="Unused: timing is always do_bench_aiter_triton's rotating-buffer "
        "CUDA graph replay now. Kept for CLI compatibility.",
    )
    parser.add_argument(
        "--graph_ms",
        type=float,
        default=20.0,
        help="Unused, see --timing. Kept for CLI compatibility.",
    )
    parser.add_argument(
        "--n_replays",
        type=int,
        default=20,
        help="Unused, see --timing. Kept for CLI compatibility.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Unused, see --timing. Kept for CLI compatibility.",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="Unused, see --timing. Kept for CLI compatibility.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write CSV results"
    )
    return parser.parse_args()


def main():
    if arch_info.get_arch() not in ("gfx1250", "gfx950"):
        sys.exit(f"chunk KDA gluon needs gfx1250 or gfx950, got {arch_info.get_arch()}")
    args = parse_args()
    if args.walk_configs is None:
        # BV 128 with KS 2 needs more LDS than gfx950 has.
        args.walk_configs = ["32,2", "64,4"]
        if arch_info.get_arch() == "gfx1250":
            args.walk_configs.append("128,4")
    benchmark(args)


if __name__ == "__main__":
    main()

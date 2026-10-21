# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Performance benchmark for fused KDA decode kernel vs 3-kernel baseline.

Uses triton.testing.do_bench for measurement (same as other bench files).
Uses real Triton rmsnorm kernel (inlined from ATOM) for fair comparison.

Usage examples
--------------
python bench_fused_kda_decode.py
python bench_fused_kda_decode.py --batches 1 16 64 128
python bench_fused_kda_decode.py --Hloc 8
"""

import functools

import torch
import triton
import triton.language as tl
from einops import rearrange

from aiter.ops.triton._triton_kernels.gated_delta_net.decode.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from aiter.ops.triton.gated_delta_net.causal_conv1d_decode import (
    causal_conv1d_update_split_qkv,
)
from aiter.ops.triton.gated_delta_net.fused_kda_decode import fused_kda_decode
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
)

DEVICE = "cuda"
D = 128
W = 4
DTYPE = torch.bfloat16


# -- Inlined from ATOM's atom/model_ops/kimi_k3/activations.py --


@triton.jit
def _rmsnorm_gated_kernel(
    x_ptr,
    w_ptr,
    g_ptr,
    y_ptr,
    H,
    eps,
    stride_xm,
    stride_ym,
    stride_g_outer,
    stride_g_head,
    HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    g_off = (row // HEADS) * stride_g_outer + (row % HEADS) * stride_g_head + cols
    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(g_ptr + g_off, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd * w) * tl.sigmoid(gate)
    tl.store(y_ptr + row * stride_ym + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def rmsnorm_gated_bf16(x, weight, gate, eps):
    """Gated RMSNorm -> bf16 (real Triton kernel)."""
    h = x.shape[-1]
    x2 = x.reshape(-1, h).contiguous()
    m = x2.shape[0]
    heads = x.shape[1] if x.ndim == 3 else 1
    y = torch.empty_like(x2)
    if gate.ndim == 3:
        stride_g_outer, stride_g_head = gate.stride(0), gate.stride(1)
    else:
        stride_g_outer, stride_g_head = gate.stride(0), 0
    BLOCK = triton.next_power_of_2(h)
    _rmsnorm_gated_kernel[(m,)](
        x2,
        weight,
        gate,
        y,
        h,
        float(eps),
        x2.stride(0),
        y.stride(0),
        stride_g_outer,
        stride_g_head,
        HEADS=heads,
        BLOCK=BLOCK,
    )
    return y.reshape_as(x)


# -- End inlined kernel --


def _make_inputs(batch, Hloc):
    lp = Hloc * D
    # ssm_state_indices below covers exactly 0..batch-1; no NULL-block slot
    # here (unlike _make_spec_inputs), so no padding beyond that.
    num_slots = batch
    torch.manual_seed(0)
    return {
        "mixed_qkv": torch.randn(batch, 3 * lp, dtype=DTYPE, device=DEVICE),
        "conv_weight": torch.randn(3 * lp, W, dtype=DTYPE, device=DEVICE) * 0.1,
        "conv_state": torch.randn(num_slots, 3 * lp, W - 1, dtype=DTYPE, device=DEVICE)
        * 0.1,
        "gate": torch.randn(1, batch, Hloc, D, dtype=DTYPE, device=DEVICE) * 0.5,
        "beta": torch.randn(1, batch, Hloc, dtype=DTYPE, device=DEVICE),
        "out_gate": torch.randn(batch, lp, dtype=DTYPE, device=DEVICE),
        "A_log": torch.randn(Hloc, dtype=DTYPE, device=DEVICE) * 0.1,
        "dt_bias": torch.randn(lp, dtype=DTYPE, device=DEVICE) * 0.1,
        "ssm_state": torch.randn(
            num_slots, Hloc, D, D, dtype=torch.float32, device=DEVICE
        )
        * 0.01,
        "norm_weight": torch.ones(D, dtype=DTYPE, device=DEVICE),
        "ssm_state_indices": torch.arange(batch, dtype=torch.int32, device=DEVICE),
        "cu_seqlens": torch.arange(batch + 1, dtype=torch.int64, device=DEVICE),
    }


def _make_spec_inputs(batch, Hloc, num_spec):
    """Speculative decode inputs: num_spec + 1 tokens per sequence, 2-D state
    indices (slot 0 is vLLM's NULL block, so real slots start at 1), and a
    convolution cache that holds history plus previous draft candidates."""
    lp = Hloc * D
    S = num_spec + 1
    T = batch * S
    num_slots = batch * S + 2
    torch.manual_seed(0)
    return {
        "mixed_qkv": torch.randn(T, 3 * lp, dtype=DTYPE, device=DEVICE),
        "conv_weight": torch.randn(3 * lp, W, dtype=DTYPE, device=DEVICE) * 0.1,
        "conv_state": torch.randn(
            num_slots, 3 * lp, W - 1 + num_spec, dtype=DTYPE, device=DEVICE
        )
        * 0.1,
        "gate": torch.randn(1, T, Hloc, D, dtype=DTYPE, device=DEVICE) * 0.5,
        "beta": torch.randn(1, T, Hloc, dtype=DTYPE, device=DEVICE),
        "out_gate": torch.randn(T, lp, dtype=DTYPE, device=DEVICE),
        "A_log": torch.randn(Hloc, dtype=DTYPE, device=DEVICE) * 0.1,
        "dt_bias": torch.randn(lp, dtype=DTYPE, device=DEVICE) * 0.1,
        "ssm_state": torch.randn(
            num_slots, Hloc, D, D, dtype=torch.float32, device=DEVICE
        )
        * 0.01,
        "norm_weight": torch.ones(D, dtype=DTYPE, device=DEVICE),
        "ssm_state_indices": torch.arange(
            1, T + 1, dtype=torch.int32, device=DEVICE
        ).reshape(batch, S),
        "num_accepted_tokens": torch.ones(batch, dtype=torch.int32, device=DEVICE),
        "conv_state_indices": torch.arange(
            1, batch + 1, dtype=torch.int32, device=DEVICE
        ),
        "cu_seqlens": torch.arange(0, T + 1, S, dtype=torch.int64, device=DEVICE),
    }


def _generate_spec_pool_inputs(batch, Hloc, num_spec):
    """Build one speculative-decode case, with its own state clones.

    Pure setup, not measured. Each call is one rotating-buffer pool entry:
    independent random projections and state, so replaying through the pool
    during timing defeats cache residency instead of timing one fixed
    buffer repeatedly. State drifts across replays same as it did across
    do_bench reps in the single-buffer version; that does not change the
    launch shape or instruction path.
    """
    inp = _make_spec_inputs(batch, Hloc, num_spec)
    conv_state = inp["conv_state"].clone()
    ssm_state = inp["ssm_state"].clone()
    out = torch.empty(inp["mixed_qkv"].shape[0], Hloc * D, dtype=DTYPE, device=DEVICE)
    return (
        inp["mixed_qkv"],
        conv_state,
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        ssm_state,
        inp["ssm_state_indices"],
        inp["cu_seqlens"],
        inp["norm_weight"],
        inp["num_accepted_tokens"],
        inp["conv_state_indices"],
        out,
    )


_SPEC_POOL_FIELDS = [
    "mixed_qkv",
    "conv_state",
    "conv_weight",
    "gate",
    "beta",
    "out_gate",
    "A_log",
    "dt_bias",
    "ssm_state",
    "ssm_state_indices",
    "cu_seqlens",
    "norm_weight",
    "num_accepted_tokens",
    "conv_state_indices",
    "out",
]


def _make_spec_kernel_fn(Hloc):
    def kernel_fn(
        mixed_qkv,
        conv_state,
        conv_weight,
        gate,
        beta,
        out_gate,
        A_log,
        dt_bias,
        ssm_state,
        ssm_state_indices,
        cu_seqlens,
        norm_weight,
        num_accepted_tokens,
        conv_state_indices,
        out,
    ):
        fused_kda_decode(
            mixed_qkv,
            conv_state,
            conv_weight,
            gate,
            beta,
            out_gate,
            A_log,
            dt_bias,
            ssm_state,
            ssm_state_indices,
            cu_seqlens,
            norm_weight,
            1e-6,
            D,
            Hloc,
            -5.0,
            num_accepted_tokens=num_accepted_tokens,
            conv_state_indices=conv_state_indices,
            out=out,
        )

    return kernel_fn


def _generate_decode_pool_inputs(batch, Hloc):
    """Build one decode case, with its own state clone for one arm.

    Pure setup, not measured. See _generate_spec_pool_inputs for why state
    drift across replays is fine here. Each arm (3-kernel vs fused) gets its
    own pool built from this, sized to the buffers that arm actually
    touches -- a pool shared between both arms would carry the other arm's
    state clone too, inflating generate_rotating_buffers_pool's per-entry
    size estimate past what either arm's replay touches.
    """
    inp = _make_inputs(batch, Hloc)
    return (
        inp["mixed_qkv"],
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        inp["ssm_state_indices"],
        inp["cu_seqlens"],
        inp["norm_weight"],
        inp["conv_state"].clone(),
        inp["ssm_state"].clone(),
    )


_DECODE_3K_POOL_FIELDS = [
    "mixed_qkv",
    "conv_weight",
    "gate",
    "beta",
    "out_gate",
    "A_log",
    "dt_bias",
    "ssm_state_indices",
    "cu_seqlens",
    "norm_weight",
    "cs_3k",
    "ss_3k",
]

_DECODE_FUSED_POOL_FIELDS = [
    "mixed_qkv",
    "conv_weight",
    "gate",
    "beta",
    "out_gate",
    "A_log",
    "dt_bias",
    "ssm_state_indices",
    "cu_seqlens",
    "norm_weight",
    "cs_f",
    "ss_f",
]


def _make_3k_kernel_fn(Hloc):
    def kernel_fn(
        mixed_qkv,
        conv_weight,
        gate,
        beta,
        out_gate,
        A_log,
        dt_bias,
        ssm_state_indices,
        cu_seqlens,
        norm_weight,
        cs_3k,
        ss_3k,
    ):
        T = mixed_qkv.shape[0]
        lp = Hloc * D
        q, k, v = causal_conv1d_update_split_qkv(
            mixed_qkv,
            cs_3k,
            conv_weight,
            lp,
            lp,
            bias=None,
            activation="silu",
            conv_state_indices=ssm_state_indices,
            use_gluon=False,
        )
        out = torch.empty(T, Hloc, D, dtype=q.dtype, device=DEVICE)
        fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=gate,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=rearrange(q, "t (h d) -> 1 t h d", d=D),
            k=rearrange(k, "t (h d) -> 1 t h d", d=D),
            v=rearrange(v, "t (h d) -> 1 t h d", d=D),
            b=beta,
            initial_state_source=ss_3k,
            initial_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        og3d = rearrange(out_gate[:T], "t (h d) -> t h d", d=D)
        rmsnorm_gated_bf16(out, norm_weight, og3d, 1e-6)

    return kernel_fn


def _make_fused_kernel_fn(Hloc):
    def kernel_fn(
        mixed_qkv,
        conv_weight,
        gate,
        beta,
        out_gate,
        A_log,
        dt_bias,
        ssm_state_indices,
        cu_seqlens,
        norm_weight,
        cs_f,
        ss_f,
    ):
        fused_kda_decode(
            mixed_qkv,
            cs_f,
            conv_weight,
            gate,
            beta,
            out_gate,
            A_log,
            dt_bias,
            ss_f,
            ssm_state_indices,
            cu_seqlens,
            norm_weight,
            1e-6,
            D,
            Hloc,
            -5.0,
        )

    return kernel_fn


def _count_parallel_launches(fkd_module, fn):
    """How many times one call launches the parallel-V kernel.

    Observing the launch keeps this benchmark honest about which path the
    dispatch gate chose, without restating the gate's conditions here.
    """
    real = fkd_module.fused_kda_spec_parallel_v_kernel
    calls = []

    class _Spy:
        def __getitem__(self, grid):
            inner = real[grid]

            def launch(*a, **k):
                calls.append(1)
                return inner(*a, **k)

            return launch

    fkd_module.fused_kda_spec_parallel_v_kernel = _Spy()
    try:
        fn()
    finally:
        fkd_module.fused_kda_spec_parallel_v_kernel = real
    return len(calls)


def run_spec_benchmark(args):
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)
    Hloc, num_spec = args.Hloc, args.num_spec
    print(f"\nSpeculative decode: Hloc={Hloc}, D={D}, W={W}, tokens/seq={num_spec + 1}")
    header = (
        f"{'Batch':>6}  {'Generic(us)':>12}  {'Selected(us)':>13}  "
        f"{'Speedup':>8}  {'Kernel':>10}"
    )
    print(header)
    print("-" * len(header))
    for batch in args.batches:
        pool_gen = functools.partial(_generate_spec_pool_inputs, batch, Hloc, num_spec)
        rot_bufs = generate_rotating_buffers_pool(
            [pool_gen], [_SPEC_POOL_FIELDS], **kwargs_rot_bufs
        )
        kernel_fn = _make_spec_kernel_fn(Hloc)

        # Time the generic kernel on the same inputs by failing the dispatch
        # gate's arch test, so the speedup is reproducible rather than taken on
        # trust. Which kernel the gate picks is observed, not recomputed: the
        # gate has several other conditions (tile config, K, V, conv width) and
        # a copy of it here would drift out of date silently.
        import aiter.ops.triton.gated_delta_net.fused_kda_decode as _fkd

        real_arch = _fkd.get_arch
        try:
            # Fail only the gfx950 specialized-path gate. Do not return gfx942:
            # that would also change the generic launch from gfx950's four
            # warps to gfx942's two warps.
            _fkd.get_arch = lambda: "gfx950-generic"
            t_generic = (
                do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)
                * 1000
            )
        finally:
            _fkd.get_arch = real_arch

        launched = _count_parallel_launches(
            _fkd,
            lambda kernel_fn=kernel_fn, rot_bufs=rot_bufs: kernel_fn(**rot_bufs[0]),
        )
        t_fused = (
            do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph) * 1000
        )

        picked = "parallel" if launched else "generic"
        # A "generic" row timed the same kernel twice, so its ratio would be
        # 1.00x by construction. Print "-" so it cannot be read as a measured
        # tie between the two paths.
        ratio = f"{t_generic / t_fused:>7.2f}x" if launched else f"{'-':>8}"
        print(
            f"{batch:>6}  {t_generic:>12.1f}  {t_fused:>13.1f}  {ratio}  {picked:>10}"
        )


def run_benchmark(args):
    if args.num_spec > 0:
        run_spec_benchmark(args)
        return
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)
    Hloc = args.Hloc
    batches = args.batches

    header = f"{'Batch':>6}  {'3-kernel(us)':>13}  {'Fused(us)':>10}  {'Speedup':>8}"
    print(f"\nHloc={Hloc}, D={D}, W={W}")
    print(header)
    print("-" * len(header))

    for batch in batches:
        # Each arm gets its own pool and its own state clone per entry, so
        # the two arms never share a state buffer -- charging both would
        # drag the ratio toward 1.0 otherwise -- and neither pool's size
        # estimate carries bytes the other arm's replay never touches.
        pool_gen = functools.partial(_generate_decode_pool_inputs, batch, Hloc)
        rot_bufs_3k = generate_rotating_buffers_pool(
            [pool_gen], [_DECODE_3K_POOL_FIELDS], **kwargs_rot_bufs
        )
        rot_bufs_fused = generate_rotating_buffers_pool(
            [pool_gen], [_DECODE_FUSED_POOL_FIELDS], **kwargs_rot_bufs
        )
        kernel_fn_3k = _make_3k_kernel_fn(Hloc)
        kernel_fn_fused = _make_fused_kernel_fn(Hloc)

        t_3k = (
            do_bench_aiter_triton(kernel_fn_3k, rot_bufs_3k, **kwargs_do_bench_graph)
            * 1000
        )
        t_fused = (
            do_bench_aiter_triton(
                kernel_fn_fused, rot_bufs_fused, **kwargs_do_bench_graph
            )
            * 1000
        )
        speedup = t_3k / t_fused

        print(f"{batch:>6}  {t_3k:>13.1f}  {t_fused:>10.1f}  {speedup:>7.2f}x")


def parse_args(args=None):
    parser = get_parser(kernel_name="Fused KDA Decode")
    parser.add_argument(
        "--Hloc",
        type=int,
        default=12,
        help="Number of local heads (Kimi K3 = 12)",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64, 128, 256],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--num-spec",
        type=int,
        default=0,
        help="Draft tokens per sequence; > 0 benchmarks the speculative path "
        "(Kimi-K3 DSpark uses 7)",
    )
    return parser.parse_args(args=args)


def main(args=None):
    run_benchmark(parse_args(args=args))


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the fused MLA qkv_a projection + q/kv RMSNorm op.

Times the fused op under CUDA graph replay with rotating input buffers, and with
``--baseline`` also the AITER chain it replaces: ``per_1x128`` FP8 quant ->
``gemm_a8w8_blockscale_bpreshuffle`` -> ``fused_qk_rmsnorm``. Reports device time per
call and the achieved bandwidth over the compulsory traffic (weight + scales + activation
+ outputs). Usage:
    python3 bench_fused_mla_qkv_a_proj_rmsnorm.py
    python3 bench_fused_mla_qkv_a_proj_rmsnorm.py --batches 1,8,32 --baseline
"""

import functools
import sys

import torch

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.fusions.fused_mla_qkv_a_proj_rmsnorm import (
    fused_mla_qkv_a_proj_rmsnorm,
    fused_mla_qkv_a_proj_rmsnorm_supported,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

_PEAK_BW = {"gfx950": 8.0e12}
# GLM-5.2: q_lora_rank 2048, kv_lora_rank 512, qk_rope_head_dim 64, hidden 6144.
_Q, _KV, _ROPE, _K = 2048, 512, 64, 6144
_N = _Q + _KV + _ROPE
_EPS = 1e-5
_DEFAULT_BATCHES = "1,2,4,8,16,32"
_KEYS = ["hidden", "weight", "weight_scale", "q_norm_weight", "kv_norm_weight"]


def make_inputs(m, device="cuda"):
    weight = (0.25 * torch.randn(_N, _K, dtype=torch.float32, device=device)).to(
        torch.float8_e4m3fn
    )
    weight_scale = torch.rand(
        -(-_N // 128), _K // 128, dtype=torch.float32, device=device
    )
    weight_scale = (0.5 + weight_scale) * 2.0**-6
    return (
        torch.randn(m, _K, dtype=torch.bfloat16, device=device),
        shuffle_weight(weight, layout=(16, 16)),
        weight_scale.contiguous(),
        torch.randn(_Q, dtype=torch.bfloat16, device=device),
        torch.randn(_KV, dtype=torch.bfloat16, device=device),
    )


def bytes_moved(m):
    total = _N * _K  # fp8 weight
    total += -(-_N // 128) * (_K // 128) * 4  # weight scales
    total += m * _K * 2  # bf16 activation
    total += (_Q + _KV) * 2  # norm weights
    total += m * _N * 2  # bf16 outputs
    return total


def fused_op(hidden, weight, weight_scale, q_norm_weight, kv_norm_weight):
    return fused_mla_qkv_a_proj_rmsnorm(
        hidden,
        weight,
        weight_scale,
        q_norm_weight,
        kv_norm_weight,
        rope_dim=_ROPE,
        eps=_EPS,
    )


def aiter_chain(hidden, weight, weight_scale, q_norm_weight, kv_norm_weight):
    from aiter import dtypes
    from aiter.ops.enum import QuantType
    from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
    from aiter.ops.quant import get_hip_quant

    m = hidden.shape[0]
    # transpose_scale writes the column-major scale the bpreshuffle GEMM reads (M >= 2).
    x_fp8, x_scale = get_hip_quant(QuantType.per_1x128)(
        hidden, quant_dtype=dtypes.fp8, transpose_scale=m >= 2
    )
    if m >= 2:
        x_scale = x_scale.as_strided((m, _K // 128), (1, m))
    projected = gemm_a8w8_blockscale_bpreshuffle(
        x_fp8, weight, x_scale, weight_scale, dtypes.bf16
    )
    q = torch.empty((m, _Q), dtype=torch.bfloat16, device=hidden.device)
    kv = torch.empty((m, _KV), dtype=torch.bfloat16, device=hidden.device)
    fused_qk_rmsnorm(
        q_out_quantized=q,
        q=projected[:, :_Q],
        q_weight=q_norm_weight,
        q_epsilon=_EPS,
        k_out=kv,
        k=projected[:, _Q : _Q + _KV],
        k_weight=kv_norm_weight,
        k_epsilon=_EPS,
        quant_type=QuantType.No,
    )
    return q, kv, projected[:, _Q + _KV :]


def bench_one(bencher, m, fn):
    bencher.build_pool([functools.partial(make_inputs, m)], [_KEYS])
    return bencher.bench(fn) * 1000.0  # ms -> us


def main():
    ap = get_parser(kernel_name="Fused MLA qkv_a projection + RMSNorm")
    ap.add_argument("--batches", default=_DEFAULT_BATCHES, help="comma list of M")
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="also time the AITER quant -> bpreshuffle GEMM -> fused_qk_rmsnorm chain",
    )
    args = ap.parse_args()

    arch = get_arch()
    inputs = make_inputs(1)
    ok, reason = fused_mla_qkv_a_proj_rmsnorm_supported(*inputs, rope_dim=_ROPE)
    if not ok:
        print(f"fused_mla_qkv_a_proj_rmsnorm is not supported here: {reason}")
        return 0
    peak = _PEAK_BW.get(arch)
    print(
        f"fused_mla_qkv_a_proj_rmsnorm on {torch.cuda.get_device_name(0)} ({arch}), "
        f"N={_N} K={_K} (q {_Q}, kv {_KV}, rope {_ROPE})"
    )
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    header = f"{'M':>6}{'fused us':>10}{'TB/s':>8}{'%pk':>6}"
    if args.baseline:
        header += f"{'chain us':>10}{'speedup':>9}"
    print(header)
    for m in (int(b) for b in args.batches.split(",")):
        ok, reason = fused_mla_qkv_a_proj_rmsnorm_supported(
            *make_inputs(m), rope_dim=_ROPE
        )
        if not ok:
            print(f"{m:>6}{'(uncovered: ' + reason + ')':>10}")
            continue
        us = bench_one(bencher, m, fused_op)
        tbs = bytes_moved(m) / (us * 1e-6) / 1e12
        pct = 100.0 * tbs * 1e12 / peak if peak else float("nan")
        line = f"{m:>6}{us:>10.2f}{tbs:>8.2f}{pct:>5.0f}%"
        if args.baseline:
            chain_us = bench_one(bencher, m, aiter_chain)
            line += f"{chain_us:>10.2f}{chain_us / us:>9.2f}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

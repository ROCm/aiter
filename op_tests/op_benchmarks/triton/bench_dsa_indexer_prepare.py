# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the fused DSA indexer prepare op against the unfused aiter chain.

The fused op is two launches. The baseline is what SGLang runs on gfx950 with
block-FP8 indexer weights: two ``per_1x128`` activation quants, two
``gemm_a8w8_blockscale_bpreshuffle`` GEMMs, one bf16 ``tgemm`` and
``indexer_qk_rope_quant_and_cache`` (six launches; the writer omits the
Hadamard). Both are timed under CUDA graph replay over a rotating input pool,
so weights and activations are not cache resident between calls.

Usage:
    python3 bench_dsa_indexer_prepare.py
    python3 bench_dsa_indexer_prepare.py --tokens 1,2,4,8,16,32 --key-only
"""

import functools
import sys

import torch

import aiter
from aiter import dtypes, get_hip_quant
from aiter.ops.cache import indexer_k_quant_and_cache, indexer_qk_rope_quant_and_cache
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.attention.dsa_indexer_prepare import (
    dsa_indexer_prepare,
    dsa_indexer_prepare_supported,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.tuned_gemm import tgemm
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

H, D, HID, QK, PAGE, STRIDE = 32, 128, 6144, 2048, 64, 132
EPS = 1e-6
WEIGHTS_SCALE = D**-0.5 * H**-0.5
_ROT_KEYS = [
    "x",
    "q_lora",
    "wq_b",
    "wq_b_scale",
    "wk",
    "wk_scale",
    "weights_proj",
    "norm_weight",
    "norm_bias",
    "positions",
    "cos_cache",
    "sin_cache",
    "slot_mapping",
]
_FIXED_KEYS = ["kv_cache"]


def make_inputs(tokens, hidden, q_rank, heads, max_pos, device="cuda"):
    bf16 = {"dtype": torch.bfloat16, "device": device}
    f32 = {"dtype": torch.float32, "device": device}
    fp8 = torch.float8_e4m3fn
    angles = torch.randn(max_pos, 32, **f32) * 3.0
    inp = {
        "x": torch.randn(tokens, hidden, **bf16),
        "q_lora": torch.randn(tokens, q_rank, **bf16),
        "wq_b": shuffle_weight(
            torch.randn(heads * D, q_rank, **f32).to(fp8), layout=(16, 16)
        ),
        "wq_b_scale": torch.rand(heads, q_rank // 128, **f32) * 2e-2,
        "wk": shuffle_weight(torch.randn(D, hidden, **f32).to(fp8), layout=(16, 16)),
        "wk_scale": torch.rand(1, hidden // 128, **f32) * 2e-2,
        "weights_proj": torch.randn(heads, hidden, **bf16) * hidden**-0.5,
        "norm_weight": torch.ones(D, **bf16),
        "norm_bias": torch.zeros(D, **bf16),
        "positions": torch.randint(0, max_pos, (tokens,), device=device),
        "cos_cache": angles.cos().to(torch.bfloat16),
        "sin_cache": angles.sin().to(torch.bfloat16),
        "slot_mapping": torch.randperm(4096 * PAGE, device=device)[
            :tokens
        ].contiguous(),
    }
    return tuple(inp[k] for k in _ROT_KEYS)


def make_cache(pages, device="cuda"):
    return (torch.zeros((pages, PAGE, STRIDE), dtype=torch.uint8, device=device),)


def fused(key_only, **kw):
    return dsa_indexer_prepare(
        kw["x"],
        None if key_only else kw["q_lora"],
        None if key_only else kw["wq_b"],
        None if key_only else kw["wq_b_scale"],
        kw["wk"],
        kw["wk_scale"],
        None if key_only else kw["weights_proj"],
        kw["norm_weight"],
        kw["norm_bias"],
        kw["positions"],
        kw["cos_cache"],
        kw["sin_cache"],
        kw["slot_mapping"],
        kw["kv_cache"],
        epsilon=EPS,
        weights_scale=WEIGHTS_SCALE,
        key_only=key_only,
    )


_quant = get_hip_quant(aiter.QuantType.per_1x128)


def _fp8_linear(x, w, w_scale):
    q, s = _quant(x, quant_dtype=dtypes.fp8)
    return gemm_a8w8_blockscale_bpreshuffle(q, w, s, w_scale, dtype=torch.bfloat16)


def unfused(key_only, **kw):
    """SGLang's aiter chain: Indexer._aiter_fused_fp8_prepare_and_store / _forward_cuda_k_only."""
    key = _fp8_linear(kw["x"], kw["wk"], kw["wk_scale"])
    kv_cache = kw["kv_cache"].view(torch.float8_e4m3fn)
    gamma, beta = kw["norm_weight"].float(), kw["norm_bias"].float()
    if key_only:
        # SGLang also runs the aiter RoPE and a Hadamard here; both are left out,
        # so this baseline is conservative.
        key = torch.nn.functional.layer_norm(key.float(), (D,), gamma, beta, EPS).to(
            torch.bfloat16
        )
        indexer_k_quant_and_cache(
            key, kv_cache, kw["slot_mapping"], 128, "ue8m0", preshuffle=True
        )
        return None, None
    query = _fp8_linear(kw["q_lora"], kw["wq_b"], kw["wq_b_scale"]).view(-1, H, D)
    gate = tgemm.mm(kw["x"], kw["weights_proj"], None, None, None)
    q_out = torch.empty_like(query, dtype=torch.float8_e4m3fn)
    weights = torch.empty_like(gate, dtype=torch.float32)
    indexer_qk_rope_quant_and_cache(
        query,
        q_out,
        gate,
        weights,
        key,
        kv_cache,
        kw["slot_mapping"],
        gamma,
        beta,
        kw["positions"],
        kw["cos_cache"],
        kw["sin_cache"],
        EPS,
        128,
        "ue8m0",
        WEIGHTS_SCALE,
        preshuffle=True,
        is_neox=False,
        compute_all_q_rope=True,
    )
    return q_out, weights


def bench_one(bencher, tokens, pages, max_pos, key_only, fn):
    gen = functools.partial(make_inputs, tokens, HID, QK, H, max_pos)
    bencher.build_pool(
        [gen], [_ROT_KEYS], [functools.partial(make_cache, pages)], [_FIXED_KEYS]
    )
    return bencher.bench(functools.partial(fn, key_only)) * 1000.0  # ms -> us


def eager_us(tokens, pages, max_pos, key_only, fn, iters=200):
    """Host-side wall time per eager call (no graph), after warmup."""
    kw = dict(
        zip(_ROT_KEYS, make_inputs(tokens, HID, QK, H, max_pos)),
        kv_cache=make_cache(pages)[0],
    )
    for _ in range(20):
        fn(key_only, **kw)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(key_only, **kw)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def main():
    ap = get_parser(kernel_name="DSA indexer prepare")
    ap.add_argument("--tokens", default="1,2,4,8,16,32")
    ap.add_argument(
        "--pages", type=int, default=4096, help="index-K cache pages of 64 tokens"
    )
    ap.add_argument("--max-pos", type=int, default=8192)
    ap.add_argument(
        "--key-only", action="store_true", help="time the key-only (skip-logits) path"
    )
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument(
        "--eager", action="store_true", help="also report eager wall time per call"
    )
    args = ap.parse_args()

    arch = get_arch()
    if arch != "gfx950":
        print(f"dsa_indexer_prepare is gfx950-only; this is {arch}")
        return 0
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    print(
        f"dsa_indexer_prepare on {torch.cuda.get_device_name(0)} ({arch}), hidden={HID} q_rank={QK} "
        f"heads={H} pages={args.pages} key_only={args.key_only}"
    )
    header = f"{'tokens':>7}{'fused us':>11}{'unfused us':>12}{'speedup':>9}"
    if args.eager:
        header += f"{'fused eager':>13}{'unfused eager':>15}"
    print(header)
    for t in [int(v) for v in args.tokens.split(",")]:
        probe = make_inputs(t, HID, QK, H, args.max_pos)
        ok, reason = dsa_indexer_prepare_supported(
            *probe[:6], *probe[6:13], make_cache(args.pages)[0], args.key_only
        )
        if not ok:
            print(f"{t:>7}  unsupported: {reason}")
            continue
        us = bench_one(bencher, t, args.pages, args.max_pos, args.key_only, fused)
        if args.no_baseline:
            print(f"{t:>7}{us:>11.2f}")
            continue
        base = bench_one(bencher, t, args.pages, args.max_pos, args.key_only, unfused)
        line = f"{t:>7}{us:>11.2f}{base:>12.2f}{base / us:>9.2f}"
        if args.eager:
            line += (
                f"{eager_us(t, args.pages, args.max_pos, args.key_only, fused):>13.1f}"
            )
            line += f"{eager_us(t, args.pages, args.max_pos, args.key_only, unfused):>15.1f}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

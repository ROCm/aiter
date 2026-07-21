# SPDX-License-Identifier: MIT

import argparse

import torch
import aiter
from aiter.utility import dtypes


def build_inputs(m: int, seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # User-provided microbenchmark shapes.
    qkv = torch.randn(m, 9216, dtype=torch.bfloat16, device="cuda")
    q_norm_w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    k_norm_w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    rope_emb = torch.randn(1048576, 128, dtype=torch.bfloat16, device="cuda")
    mrope_pos = torch.randint(0, 4096, (3, m), dtype=torch.int64, device="cuda")
    q_out = torch.empty(m, 8192, dtype=torch.bfloat16, device="cuda")

    # Use the repository's HIP-aware fp8 alias so MI355x picks the right runtime type.
    kv_dtype = dtypes.fp8
    kv_k = torch.zeros(22988, 64, 4, 128, dtype=kv_dtype, device="cuda")
    kv_v = torch.zeros(22988, 64, 4, 128, dtype=kv_dtype, device="cuda")
    slot_map = torch.randint(0, 22988 * 64, (m,), dtype=torch.int64, device="cuda")
    per_tensor_k_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    per_tensor_v_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    # fp8 cache => element_size=1, x=16.
    x = 16 // torch.empty(0, dtype=kv_dtype, device="cuda").element_size()

    return {
        "qkv": qkv,
        "q_norm_w": q_norm_w,
        "k_norm_w": k_norm_w,
        "rope_emb": rope_emb,
        "mrope_pos": mrope_pos,
        "q_out": q_out,
        "kv_k": kv_k,
        "kv_v": kv_v,
        "slot_map": slot_map,
        "kv_dtype": kv_dtype,
        "per_tensor_k_scale": per_tensor_k_scale,
        "per_tensor_v_scale": per_tensor_v_scale,
        "x": x,
    }


def allocate_outputs(inputs, return_kv: bool):
    m = inputs["qkv"].shape[0]
    kv_dtype = inputs["kv_dtype"]

    return {
        "k_out": torch.empty(m, 4, 128, dtype=kv_dtype, device="cuda") if return_kv else None,
        "v_out": torch.empty(m, 4, 128, dtype=kv_dtype, device="cuda") if return_kv else None,
    }


def run_once(inputs, outputs, return_kv: bool):
    m = inputs["qkv"].shape[0]

    aiter.fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
        inputs["qkv"],
        inputs["q_norm_w"],
        inputs["k_norm_w"],
        inputs["rope_emb"],
        inputs["mrope_pos"],
        m,
        64,
        4,
        4,
        128,
        True,
        [24, 20, 20],
        True,
        1e-6,
        inputs["q_out"],
        inputs["kv_k"],
        inputs["kv_v"],
        inputs["slot_map"],
        inputs["per_tensor_k_scale"],
        inputs["per_tensor_v_scale"],
        outputs["k_out"],
        outputs["v_out"],
        return_kv,
        True,
        64,
        inputs["x"],
        128,
        False,
    )


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required")

    inputs = build_inputs(args.m, args.seed)
    outputs = allocate_outputs(inputs, args.return_kv)

    for _ in range(args.warmup):
        run_once(inputs, outputs, args.return_kv)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(args.iters):
        run_once(inputs, outputs, args.return_kv)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_us = (total_ms * 1000.0) / args.iters

    slots = inputs["kv_k"].shape[0] * inputs["kv_k"].shape[1]
    print(
        "[microbench] fused_qk_norm_mrope_3d_cache_pts_quant_shuffle "
        f"M={args.m} slots={slots} qkv={tuple(inputs['qkv'].shape)} "
        f"kvcache={tuple(inputs['kv_k'].shape)} kv_dtype={inputs['kv_dtype']} "
        f"x={inputs['x']} warmup={args.warmup} iters={args.iters} avg={avg_us:.2f} us"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal rocprof-compute benchmark for fused_qk_norm_mrope_3d_cache_pts_quant_shuffle"
    )
    parser.add_argument("--m", type=int, default=32768, help="Number of tokens")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--return-kv", action="store_true", help="Enable return_kv path")
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())

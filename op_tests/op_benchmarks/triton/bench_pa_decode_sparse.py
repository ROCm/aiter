# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for ``pa_decode_sparse`` — DeepSeek-V4 sparse MLA paged decode.

Providers
---------
``bf16``        a16w16: bf16 Q ``[T, H, 512]`` against a bf16 pool ``[P, 512]``.
``fp8``         a8w8: the DSv4 "2buff" packed-fp8 pool (448 fp8 NoPE | 14 dup
                E8M0 group scales | 50 pad, plus a bf16 ``[P, 64]`` RoPE plane)
                with the matching packed fp8 Q + bf16 Q-RoPE plane. Byte-for-byte
                the layout the MLA-v4 asm decode kernel reads.
``fp8_stage``   the same a8w8 inputs as ``fp8``, but forced onto the
                dequantize-to-bf16-and-stage kernel instead of the MX dots --
                the A/B for "native MX" vs "dequant + bf16 WMMA".
``fp8_qbf16``   a8w16: the same fp8 pool, but Q handed over as plain bf16.
``asm``         the reference point for ``fp8``: ``aiter.mla.mla_decode_fwd_v4_nm``
                -> ``_ZN5aiter35mla_a8w8_qh64_1tg_16mx4_64nx1_sparseE``. Reads the
                exact same tensors as ``fp8`` (gfx1250 only, gqa in {16, 64, 128}).

Usage
-----
  # default DSv4 sweep (H=128, D=512, the CSA (384) and HCA (136) kv lengths)
  python op_tests/op_benchmarks/triton/bench_pa_decode_sparse.py

  # one shape, cudagraph timing, all providers
  python op_tests/op_benchmarks/triton/bench_pa_decode_sparse.py \
      --shape 512 128 512 384 --cudagraph

  # bandwidth instead of latency, fp8 vs asm only
  python op_tests/op_benchmarks/triton/bench_pa_decode_sparse.py \
      --metric bandwidth --providers fp8 asm
"""

import argparse
import sys

import torch
import triton

from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse
from aiter.ops.triton.utils._triton import arch_info

# DSv4 MLA head geometry: one 512-wide latent per token, of which the last 64
# dims are the RoPE half. V is the whole row, so the output is 512 wide too.
NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM  # 512
FP8_DTYPE = torch.float8_e4m3fn  # OCP e4m3 — what the asm .co consumes
NUM_TILES = NOPE_DIM // 64  # 7 E8M0 quant groups

# The asm decode kernel is dispatched per (gqa, qSeqLen); at qSeqLen=1 only
# these head counts resolve to a shipped .co (hsa/gfx1250/mla_v4/mla_v4_asm.csv
# + the gqa remap in csrc/py_itfs_cu/asm_mla_v4.cu).
_ASM_SHIPPED_GQA = (16, 64, 128)

ALL_PROVIDERS = ("bf16", "fp8", "fp8_stage", "fp8_qbf16", "mx", "asm")
METRICS = ("time", "bandwidth", "throughput")

# DSv4-Pro: 128 Q heads per rank under dp-attention, kv_len 384 on the CSA
# layers and 136 on the HCA layers.
DEFAULT_SHAPES = [
    (T, 128, HEAD_DIM, kv_len) for kv_len in (136, 384) for T in (1, 32, 128, 512, 1024)
]


# ---------------------------------------------------------------------------
# Packing (mirrors op_tests/triton_tests/attention/test_pa_decode_sparse.py and
# ATOM atom/model_ops/v4_kernels/v4_quant.py)
# ---------------------------------------------------------------------------
def v4_pack_2buff(x_bf16):
    """``[..., 512]`` bf16 NoPE||RoPE -> ``(packed [..., 512] fp8, rope [..., 64] bf16)``."""
    lead = x_bf16.shape[:-1]
    nope = x_bf16[..., :NOPE_DIM].float()
    rope = x_bf16[..., NOPE_DIM:].contiguous()

    tiled = nope.reshape(*lead, NUM_TILES, 64)
    fp8_max = float(torch.finfo(FP8_DTYPE).max)
    scale = torch.pow(
        2.0, torch.clamp_min(tiled.abs().amax(dim=-1) / fp8_max, 1e-4).log2().ceil()
    )
    nope_fp8 = (tiled / scale.unsqueeze(-1)).to(FP8_DTYPE).reshape(*lead, NOPE_DIM)
    e8m0 = (scale.log2().round().to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)

    packed = torch.zeros((*lead, HEAD_DIM), dtype=torch.uint8, device=x_bf16.device)
    packed[..., :NOPE_DIM] = nope_fp8.view(torch.uint8)
    packed[..., NOPE_DIM : NOPE_DIM + 2 * NUM_TILES] = e8m0.repeat_interleave(2, dim=-1)
    return packed.view(FP8_DTYPE), rope


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def v4_pack_mx(x_bf16):
    """EXPERIMENTAL layout: ``[..., 512]`` bf16 -> (e4m3 [..., 512],
    E8M0 [..., 16] uint8). The whole head, RoPE included, in 8 quant groups of
    64; each group's scale byte stored twice so the 16 bytes are the MX scale
    operand as-is."""
    lead = x_bf16.shape[:-1]
    d = x_bf16.shape[-1]
    tiled = x_bf16.float().reshape(*lead, d // 64, 64)
    fp8_max = float(torch.finfo(FP8_DTYPE).max)
    scale = torch.pow(
        2.0, torch.clamp_min(tiled.abs().amax(dim=-1) / fp8_max, 1e-4).log2().ceil()
    )
    vals = (tiled / scale.unsqueeze(-1)).to(FP8_DTYPE).reshape(*lead, d)
    e8m0 = (scale.log2().round().to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)
    return vals, e8m0.repeat_interleave(2, dim=-1)


_INPUT_CACHE = {}


def build_inputs(T, H, D, kv_len, var_len=False, seed=0, device="cuda"):
    """Inputs for one shape, memoized.

    perf_report calls the bench fn once per (shape, provider), so without the
    cache a 5-provider sweep rebuilds and re-quantizes the whole KV pool 5
    times -- at T=512, kv_len=384 that is ~100M elements quantized three
    different ways per provider, which dominated the sweep's wall clock.
    """
    key = (T, H, D, kv_len, bool(var_len), seed, str(device))
    if key in _INPUT_CACHE:
        return _INPUT_CACHE[key]
    torch.manual_seed(seed)
    pages = T * kv_len

    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=device) * 0.5
    kv = torch.randn(pages, D, dtype=torch.bfloat16, device=device) * 0.5
    sink = torch.randn(H, dtype=torch.float32, device=device) * 0.1

    if var_len:
        lens = torch.randint(1, kv_len + 1, (T,), device=device, dtype=torch.int64)
    else:
        lens = torch.full((T,), kv_len, device=device, dtype=torch.int64)
    indptr = torch.zeros(T + 1, device=device, dtype=torch.int64)
    indptr[1:] = lens.cumsum(0)
    total_indices = int(indptr[-1].item())
    indices = torch.randint(
        0, pages, (total_indices,), device=device, dtype=torch.int32
    )

    kv_packed, kv_rope = v4_pack_2buff(kv)
    q_packed, q_rope = v4_pack_2buff(q)
    kv_mx, kv_mx_s = v4_pack_mx(kv)
    q_mx, q_mx_s = v4_pack_mx(q)
    _INPUT_CACHE[key] = {
        "kv_mx": kv_mx,
        "kv_mx_s": kv_mx_s,
        "q_mx": q_mx,
        "q_mx_s": q_mx_s,
        "q": q,
        "kv": kv,
        "q_packed": q_packed,
        "q_rope": q_rope,
        "kv_packed": kv_packed,
        "kv_rope": kv_rope,
        "indices": indices,
        "indptr": indptr.to(torch.int32),
        "sink": sink,
        "total_indices": total_indices,
        "softmax_scale": float(D) ** -0.5,
    }
    return _INPUT_CACHE[key]


def _make_fn(provider, inp, T, H, D):
    """Return ``(callable, bytes_moved)`` for one provider, or None if unsupported."""
    ind, iptr, sink = inp["indices"], inp["indptr"], inp["sink"]
    scale = inp["softmax_scale"]
    n_idx = inp["total_indices"]
    out_bytes = T * H * D * 2

    if provider == "bf16":
        q, kv = inp["q"], inp["kv"]
        fn = lambda: pa_decode_sparse(q, kv, ind, iptr, sink, scale, has_invalid=False)
        # gathered KV + Q read + output written
        return fn, n_idx * D * 2 + T * H * D * 2 + out_bytes

    if provider in ("fp8", "fp8_stage", "fp8_qbf16"):
        kvp, kvr = inp["kv_packed"], inp["kv_rope"]
        packed_q = provider in ("fp8", "fp8_stage")
        q = inp["q_packed"] if packed_q else inp["q"]
        qr = inp["q_rope"] if packed_q else None
        fn = lambda: pa_decode_sparse(
            q,
            kvp,
            ind,
            iptr,
            sink,
            scale,
            has_invalid=False,
            unified_kv_rope=kvr,
            q_rope=qr,
            use_mx=None if provider == "fp8" else False,
        )
        kv_row = HEAD_DIM * 1 + ROPE_DIM * 2  # 512 B fp8 + 128 B bf16 RoPE
        q_row = kv_row if packed_q else D * 2
        return fn, n_idx * kv_row + T * H * q_row + out_bytes

    if provider == "mx":
        fn = lambda: pa_decode_sparse(
            inp["q_mx"],
            inp["kv_mx"],
            ind,
            iptr,
            sink,
            scale,
            has_invalid=False,
            kv_mx_scales=inp["kv_mx_s"],
            q_mx_scales=inp["q_mx_s"],
        )
        row = HEAD_DIM * 1 + (HEAD_DIM // 32)  # 512 e4m3 + 16 E8M0
        return fn, n_idx * row + T * H * row + out_bytes

    if provider == "asm":
        if arch_info.get_arch() != "gfx1250" or H not in _ASM_SHIPPED_GQA:
            return None
        try:
            import aiter.mla
        except ImportError:
            return None
        # Dispatched from a prebuilt .co + a csv row; a tree without those
        # assets has no asm decode to time.
        mla_decode_fwd_v4_nm = getattr(aiter.mla, "mla_decode_fwd_v4_nm", None)
        if mla_decode_fwd_v4_nm is None:
            return None
        qp, qr = inp["q_packed"], inp["q_rope"]
        kvp = inp["kv_packed"].view(-1, 1, 1, HEAD_DIM)
        kvr = inp["kv_rope"].view(-1, 1, 1, ROPE_DIM)
        qo_indptr = torch.arange(0, T + 1, dtype=torch.int32, device=qp.device)
        out = torch.empty((T, H, HEAD_DIM), dtype=torch.bfloat16, device=qp.device)
        fn = lambda: mla_decode_fwd_v4_nm(
            qp,
            qr,
            kvp,
            kvr,
            out,
            qo_indptr,
            iptr,
            ind,
            1,  # max_seqlen_q — page_size=1, one query row per sequence
            sink=sink,
            sm_scale=scale,
        )
        kv_row = HEAD_DIM * 1 + ROPE_DIM * 2
        return fn, n_idx * kv_row + T * H * kv_row + out_bytes

    raise ValueError(f"unknown provider {provider}")


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------
def bench_fn(T, H, D, kv_len, provider, metric, var_len, cudagraph, rep):
    inp = build_inputs(T, H, D, kv_len, var_len=var_len)
    made = _make_fn(provider, inp, T, H, D)
    if made is None:
        return float("nan")
    fn, nbytes = made

    try:
        fn()  # compile / dispatch outside the timed region
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 — one bad shape must not kill the sweep
        print(f"  [{provider}] T={T} H={H} kv_len={kv_len}: {e}", file=sys.stderr)
        return float("nan")

    if cudagraph:
        ms = triton.testing.do_bench_cudagraph(fn, rep=rep)
    else:
        ms = triton.testing.do_bench(fn, warmup=max(1, rep // 4), rep=rep)

    if metric == "time":
        return ms * 1e3  # us
    if metric == "bandwidth":
        return nbytes / (ms * 1e-3) * 1e-12  # TB/s
    if metric == "throughput":
        # per token: H heads x kv_len keys x (QK over D + PV over D), 2 flop each
        flops = 2.0 * H * inp["total_indices"] * 2 * D
        return flops / (ms * 1e-3) * 1e-12  # TFLOP/s
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    providers = list(args.providers)
    unit = {"time": "us", "bandwidth": "TB/s", "throughput": "TFLOP/s"}[args.metric]

    if args.shape:
        shapes = [tuple(args.shape)]
    else:
        shapes = DEFAULT_SHAPES

    benchmark = triton.testing.Benchmark(
        x_names=["T", "H", "D", "kv_len"],
        x_vals=shapes,
        line_arg="provider",
        line_vals=providers,
        line_names=list(providers),  # perf_report appends the ylabel
        # one style per provider -- cycled, so adding a provider cannot
        # IndexError in triton's plotting path
        styles=[
            [
                ("green", "-"),
                ("blue", "-"),
                ("cyan", "--"),
                ("magenta", "--"),
                ("red", "-"),
            ][i % 5]
            for i in range(len(providers))
        ],
        ylabel=unit,
        plot_name=(
            f"pa-decode-sparse-{args.metric}"
            f"{'-cudagraph' if args.cudagraph else ''}"
            f"{'-varlen' if args.var_len else ''}"
        ),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def _bench(T, H, D, kv_len, provider):
        return bench_fn(
            T,
            H,
            D,
            kv_len,
            provider,
            args.metric,
            args.var_len,
            args.cudagraph,
            args.rep,
        )

    _bench.run(save_path="." if args.o else None, print_data=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Benchmark pa_decode_sparse (DSv4 sparse MLA decode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--shape",
        type=int,
        nargs=4,
        metavar=("T", "H", "D", "KV_LEN"),
        default=None,
        help="Single shape to benchmark. Default sweeps the DSv4 decode shapes.",
    )
    p.add_argument(
        "--providers",
        nargs="+",
        choices=ALL_PROVIDERS,
        default=list(ALL_PROVIDERS),
        help="Which implementations to time.",
    )
    p.add_argument("--metric", choices=METRICS, default="time")
    p.add_argument(
        "--var_len",
        action="store_true",
        help="Random per-token kv_len in [1, KV_LEN] instead of a fixed length.",
    )
    p.add_argument(
        "--cudagraph",
        action="store_true",
        help="Use do_bench_cudagraph instead of do_bench — removes the launch "
        "overhead that dominates these small, bandwidth-bound decode kernels.",
    )
    p.add_argument(
        "--rep",
        type=int,
        default=100,
        help="Milliseconds of timed repetitions per measurement. Lower it to "
        "keep a sweep short; the DSv4 decode kernels are tens of us, so even "
        "a few ms of reps is many iterations.",
    )
    p.add_argument("-o", action="store_true", help="Write the results to ./")
    return p.parse_args(argv)


def main(argv=None):
    if not torch.cuda.is_available():
        raise SystemExit("pa_decode_sparse benchmark requires a CUDA/HIP device")
    run_benchmark(parse_args(argv))


if __name__ == "__main__":
    main()

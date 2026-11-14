# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark ``pa_prefill_sparse`` -- sparse paged-prefill attention over two
KV sources (prefix + extend) with per-head sink.

Widths matter here. DSv4-Pro alternates ``compress_ratio`` 128 and 4, which
makes the gathered per-token history about **134** and **352** entries
respectively -- not the 1152 worst case. Benching a uniform 1152 overstates
the win (12.9x claimed vs 8.3x on realistic lengths, 2026-10-07), so the
default shapes here are the realistic ones and they are variable-length.

  # default: the two DSv4 layer classes, variable length
  python op_tests/op_benchmarks/triton/bench_pa_prefill_sparse.py

  # one shape, fixed length, bandwidth instead of time
  python op_tests/op_benchmarks/triton/bench_pa_prefill_sparse.py \\
      --shape 4096 64 512 134 1 --no-variable-len --metric bandwidth
"""

import argparse
import sys

import torch
import triton

from aiter.ops.triton.attention.pa_prefill_sparse import pa_prefill_sparse
from aiter.ops.triton.utils._triton import arch_info

_V4_DIM_NOPE, _V4_DIM_ROPE = 448, 64
_V4_DIM_QK = _V4_DIM_NOPE + _V4_DIM_ROPE
_V4_TILE = 64
_V4_NUM_TILES = _V4_DIM_NOPE // _V4_TILE
_V4_FP8 = torch.float8_e4m3fn


def v4_pack_2buff(x_bf16):
    """``[..., 512]`` bf16 -> ``(packed fp8 [..., 512], rope [..., 64])``.

    Must match op_tests/.../test_pa_prefill_sparse.py::_v4_pack_2buff.
    """
    lead = x_bf16.shape[:-1]
    nope = x_bf16[..., :_V4_DIM_NOPE].float()
    rope = x_bf16[..., _V4_DIM_NOPE:].contiguous()
    tiled = nope.reshape(*lead, _V4_NUM_TILES, _V4_TILE)
    fp8_max = float(torch.finfo(_V4_FP8).max)
    scale = torch.pow(
        2.0, torch.clamp_min(tiled.abs().amax(dim=-1) / fp8_max, 1e-4).log2().ceil()
    )
    nope_fp8 = (tiled / scale.unsqueeze(-1)).to(_V4_FP8).reshape(*lead, _V4_DIM_NOPE)
    e8m0 = (scale.log2().round().to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)
    packed = torch.zeros((*lead, _V4_DIM_QK), dtype=torch.uint8, device=x_bf16.device)
    packed[..., :_V4_DIM_NOPE] = nope_fp8.view(torch.uint8)
    packed[..., _V4_DIM_NOPE : _V4_DIM_NOPE + 2 * _V4_NUM_TILES] = (
        e8m0.repeat_interleave(2, dim=-1)
    )
    return packed.view(_V4_FP8), rope


DEVICE_ARCH = arch_info.get_arch()

METRICS = ("time", "throughput", "bandwidth")

# (T, H, D, prefix_len, extend_len, label)
# prefix_len is the per-token gathered width; 134/352 are the measured DSv4
# cr=128 / cr=4 widths. T is the prefill chunk's token count.
DEFAULT_SHAPES = [
    (1024, 64, 512, 134, 1, "cr128-1k"),
    (1024, 64, 512, 352, 1, "cr4-1k"),
    (4096, 64, 512, 134, 1, "cr128-4k"),
    (4096, 64, 512, 352, 1, "cr4-4k"),
    (16128, 64, 512, 134, 1, "cr128-16k"),
    (16128, 64, 512, 352, 1, "cr4-16k"),
]


def build_inputs(
    T,
    H,
    D,
    prefix_len,
    extend_len,
    variable_len,
    seed=0,
    dtype=torch.bfloat16,
    extend_pool=None,
    pool=None,
):
    """Same construction as op_tests/.../test_pa_prefill_sparse.py::_make_inputs.

    Kept in step with the test deliberately: the bench must measure the shape
    the test proves correct.
    """
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    # Pool sizes: enough distinct rows that the gather is not an L2 hit.
    # Pool size sets the reuse factor and therefore whether this bench
    # measures a cache-resident or a DRAM-bound regime. Default is small
    # (96-256x reuse); sweep it up to find the production regime.
    total_pages = pool if pool is not None else max(T * 4, prefix_len * 64)
    # aiter's test_pa_sparse_prefill.py uses --total_tokens, i.e. a pool of
    # exactly that many rows. At 1 row every token's extend index reads row
    # 0 (L2-resident); our default 16384-row pool makes it a cold gather.
    total_extend_tokens = (
        extend_pool if extend_pool is not None else max(T, extend_len * 64)
    )

    q = torch.randn(T, H, D, dtype=dtype, device=dev) * 0.5
    unified_kv = torch.randn(total_pages, D, dtype=dtype, device=dev) * 0.5
    kv = torch.randn(total_extend_tokens, D, dtype=dtype, device=dev) * 0.5
    attn_sink = torch.randn(H, dtype=torch.float32, device=dev) * 0.1

    def ragged(per_token, pool):
        if variable_len and per_token > 1:
            lens = torch.randint(1, per_token + 1, (T,), device=dev, dtype=torch.int64)
        else:
            lens = torch.full((T,), per_token, device=dev, dtype=torch.int64)
        indptr = torch.zeros(T + 1, device=dev, dtype=torch.int64)
        indptr[1:] = lens.cumsum(0)
        total = int(indptr[-1].item())
        idx = torch.randint(0, pool, (total,), device=dev, dtype=torch.int32)
        return idx, indptr.to(torch.int32), total

    p_idx, p_indptr, total_p = ragged(prefix_len, total_pages)
    e_idx, e_indptr, total_e = ragged(extend_len, total_extend_tokens)

    return {
        "q": q,
        "unified_kv": unified_kv,
        "kv_indices_prefix": p_idx,
        "kv_indptr_prefix": p_indptr,
        "kv": kv,
        "kv_indices_extend": e_idx,
        "kv_indptr_extend": e_indptr,
        "attn_sink": attn_sink,
        "softmax_scale": float(D) ** -0.5,
        # No -1 sentinels are generated anywhere in this bench. Left as None
        # the launcher infers True from the average prefix length and the
        # kernel pays a sentinel check on every slot -- worth ~1.35x, and it
        # is why this bench read slower than aiter's for the SAME kernel.
        "has_invalid": False,
        "_total_p": total_p,
        "_total_e": total_e,
        "_uniq_p": int(torch.unique(p_idx).numel()),
        "_uniq_e": int(torch.unique(e_idx).numel()),
    }


# Bytes an actual KV row occupies, which is NOT D * elem_size for fp8:
#   bf16      : D * 2
#   fp8 2buff : 512 B packed record (448 NoPE + inline scales + pad)
#               + 64 bf16 RoPE lanes = 640 B
def _kv_row_bytes(D, fp8):
    return 512 + 64 * 2 if fp8 else D * 2


def flops_bytes(T, H, D, n_slots, n_uniq_rows, fp8):
    """FLOPs over every gathered slot; DRAM traffic over DISTINCT rows.

    The two are not the same and conflating them is how this bench previously
    reported 10-11 TB/s for the bf16 kernel -- above any plausible HBM rate.
    Every index slot does a dot, so FLOPs scale with ``n_slots``. But a row
    selected by many tokens is fetched from DRAM roughly once and served from
    cache thereafter, so traffic scales with ``n_uniq_rows``.

    Distinct-row traffic is a LOWER bound (a row re-touched after it has been
    evicted is fetched twice) and slot-count traffic is an UPPER bound; the
    truth is between, and the ``reuse`` column says how far apart they are.
    Treat TB/s as a floor, not a measurement.
    """
    flops = 4.0 * H * D * n_slots  # QK (2) + PV (2)
    kv_bytes = float(n_uniq_rows) * _kv_row_bytes(D, fp8)
    # q in, o out. Q is the packed record on the fp8 path; O is bf16 either way.
    q_bytes = float(T) * H * (640 if fp8 else D * 2)
    o_bytes = float(T) * H * D * 2
    return flops, kv_bytes + q_bytes + o_bytes


def bench_one(
    shape, variable_len, rep, cudagraph, fp8=False, extend_pool=None, pool=None
):
    T, H, D, p_len, e_len, label = shape
    inp = build_inputs(
        T, H, D, p_len, e_len, variable_len, extend_pool=extend_pool, pool=pool
    )
    total_p, total_e = inp.pop("_total_p"), inp.pop("_total_e")
    uniq_p, uniq_e = inp.pop("_uniq_p"), inp.pop("_uniq_e")

    if fp8 and e_len > 1:
        # TWO POOLS. e_len > 1 is the trigger: a 1-wide extend pool is the
        # degenerate stand-in used to keep index counts comparable, not a real
        # second source. Mirrors production, where CSA and SWA are separate.
        q_packed, q_rope = v4_pack_2buff(inp["q"])
        kv_packed, kv_rope_p = v4_pack_2buff(inp["unified_kv"])
        ext_packed, ext_rope = v4_pack_2buff(inp["kv"])
        args = {
            "q": q_packed,
            "unified_kv": kv_packed,
            "kv_indices_prefix": inp["kv_indices_prefix"],
            "kv_indptr_prefix": inp["kv_indptr_prefix"],
            "kv": ext_packed,
            "kv_indices_extend": inp["kv_indices_extend"],
            "kv_indptr_extend": inp["kv_indptr_extend"],
            "attn_sink": inp["attn_sink"],
            "softmax_scale": inp["softmax_scale"],
            "q_rope": q_rope,
            "unified_kv_rope": kv_rope_p,
            "kv_rope": ext_rope,
            "has_invalid": False,
        }

        def fn():
            return pa_prefill_sparse(**args)

    elif fp8:
        # Prefix-only.
        q_packed, q_rope = v4_pack_2buff(inp["q"])
        kv_packed, kv_rope = v4_pack_2buff(inp["unified_kv"])
        args = {
            "q": q_packed,
            "unified_kv": kv_packed,
            "kv_indices_prefix": inp["kv_indices_prefix"],
            "kv_indptr_prefix": inp["kv_indptr_prefix"],
            "kv": None,
            "kv_indices_extend": None,
            "kv_indptr_extend": None,
            "attn_sink": inp["attn_sink"],
            "softmax_scale": inp["softmax_scale"],
            "q_rope": q_rope,
            "unified_kv_rope": kv_rope,
            "has_invalid": False,
        }
        total_e = 0
        uniq_e = 0  # the fp8 path drops the extend pool entirely

        def fn():
            return pa_prefill_sparse(**args)

    else:

        def fn():
            return pa_prefill_sparse(**inp)

    try:
        fn()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - report and keep going
        print(f"  {label:<12} FAIL {type(e).__name__}: {str(e)[:70]}", file=sys.stderr)
        return None

    if cudagraph:
        ms = triton.testing.do_bench_cudagraph(fn, rep=rep)
    else:
        ms = triton.testing.do_bench(fn, warmup=max(1, rep // 4), rep=rep)

    flops, byts = flops_bytes(T, H, D, total_p + total_e, uniq_p + uniq_e, fp8)
    return {
        "label": label,
        "T": T,
        "H": H,
        "D": D,
        "prefix": p_len,
        "extend": e_len,
        "indices": total_p + total_e,
        "uniq": uniq_p + uniq_e,
        "reuse": (total_p + total_e) / max(1, uniq_p + uniq_e),
        "us": ms * 1e3,
        "tflops": flops / (ms * 1e-3) / 1e12,
        "tbps": byts / (ms * 1e-3) / 1e12,
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--shape",
        type=int,
        nargs=5,
        action="append",
        metavar=("T", "H", "D", "PREFIX_LEN", "EXTEND_LEN"),
        help="Shape to benchmark; repeat for several. PREFIX_LEN is the "
        "per-token gathered width (DSv4: 134 for cr=128, 352 for cr=4). "
        "Default: the six DSv4 shapes.",
    )
    p.add_argument(
        "--variable-len",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw per-token lengths in [1, PREFIX_LEN] instead of "
        "a uniform width (default: on -- uniform overstates).",
    )
    p.add_argument(
        "--fp8",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the DSv4 packed-fp8 2buff path (prefix-only) " "instead of bf16.",
    )
    p.add_argument(
        "--pool",
        type=int,
        default=None,
        help="rows in the prefix KV pool. Default is small enough that the "
        "working set is cache-resident; raise it until 'reuse' approaches 1.",
    )
    p.add_argument(
        "--extend-pool",
        type=int,
        default=None,
        help="rows in the extend KV pool (aiter's --total_tokens). "
        "Default sizes it from T; pass 1 to match aiter's harness.",
    )
    p.add_argument("--metric", choices=METRICS, default="time")
    p.add_argument(
        "--rep", type=int, default=50, help="do_bench target in ms (default 50)."
    )
    p.add_argument("--cudagraph", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args(argv)

    if DEVICE_ARCH not in ("gfx1250", "gfx950"):
        print(
            f"pa_prefill_sparse has no tuned path on {DEVICE_ARCH}; "
            "running the portable Triton fallback.",
            file=sys.stderr,
        )

    shapes = (
        [tuple(s) + (f"T{s[0]}-p{s[3]}",) for s in a.shape]
        if a.shape
        else DEFAULT_SHAPES
    )

    print(
        f"arch={DEVICE_ARCH}  fp8={a.fp8}  variable_len={a.variable_len}  "
        f"cudagraph={a.cudagraph}  rep={a.rep}"
    )
    print(
        f"{'shape':<12}{'T':>7}{'H':>5}{'D':>5}{'prefix':>8}{'extend':>8}"
        f"{'indices':>10}{'uniq':>9}{'reuse':>7}{'us':>10}{'TF/s':>9}{'TB/s':>8}"
    )
    print("-" * 98)
    rows = []
    for s in shapes:
        r = bench_one(
            s, a.variable_len, a.rep, a.cudagraph, a.fp8, a.extend_pool, a.pool
        )
        if r is None:
            continue
        rows.append(r)
        print(
            f"{r['label']:<12}{r['T']:>7}{r['H']:>5}{r['D']:>5}"
            f"{r['prefix']:>8}{r['extend']:>8}{r['indices']:>10}"
            f"{r['uniq']:>9}{r['reuse']:>7.1f}"
            f"{r['us']:>10.1f}{r['tflops']:>9.1f}{r['tbps']:>8.2f}"
        )
    if not rows:
        sys.exit("no shape completed")
    return rows


if __name__ == "__main__":
    main()

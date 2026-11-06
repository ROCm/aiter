# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for FlyDSL A16W16 HGEMM (gfx950).

The model path (``tuned_gemm.flydsl_gemm``) is BF16 x BF16:

    a    : [M, K]  contiguous
    w    : [N, K]  contiguous
    out  : [M, N]  preallocated
    y    = flydsl_hgemm(a, w, out=out, ...)   # kernel sees w.t() as NT

Run:
    python op_tests/test_flydsl_hgemm.py
    python op_tests/test_flydsl_hgemm.py -s 128,4096,4096
    python op_tests/test_flydsl_hgemm.py -s 32,384,7168
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_hgemm
from aiter.ops.flydsl.gemm_a16w16_band import band_stream_config
from aiter.ops.flydsl.gemm_kernels import (
    flydsl_hgemm_kernel_name,
    get_flydsl_hgemm_kernel_params,
)
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

torch.set_default_device("cuda")

# Public wrapper currently supports gfx950 only. Positive allow-list: an unknown
# new card must not silently run an unbuilt kernel.
SUPPORTED_GFX = ["gfx950"]
SEED = 0

# (m, n, k, block_m, block_n, block_k, stages, split_k, m_waves, n_waves, k_waves,
#  group_m, hti)
_TUNED = [
    (8, 4096, 4096, 16, 16, 256, 4, 1, 1, 1, 2, 4, False),
    (16, 4096, 4096, 16, 16, 256, 4, 1, 1, 1, 2, 4, False),
    (32, 4096, 4096, 16, 32, 256, 4, 1, 1, 2, 2, 4, False),
    (64, 4096, 4096, 32, 32, 128, 6, 1, 2, 2, 1, 4, False),
    (128, 4096, 4096, 32, 64, 128, 4, 1, 1, 4, 1, 0, False),
    (256, 4096, 4096, 64, 64, 128, 4, 1, 4, 2, 1, 0, False),
    (512, 4096, 4096, 64, 128, 64, 5, 1, 2, 4, 1, 4, False),
    (1024, 4096, 4096, 128, 128, 64, 4, 1, 2, 4, 1, 4, False),
    (2048, 4096, 4096, 256, 128, 64, 3, 1, 4, 2, 2, 0, False),
    (1024, 1024, 1024, 64, 64, 64, 4, 1, 1, 4, 1, 4, False),
    (2048, 2048, 2048, 128, 128, 64, 4, 1, 2, 4, 1, 0, False),
    (4096, 4096, 4096, 256, 256, 64, 2, 1, 2, 4, 1, 4, True),
    (4096, 4096, 8192, 256, 256, 64, 2, 1, 2, 4, 1, 4, True),
    (8192, 8192, 8192, 256, 256, 64, 2, 1, 2, 4, 1, 4, True),
    (8, 7168, 2048, 16, 16, 64, 8, 1, 1, 1, 1, 4, False),
    (32, 384, 7168, 16, 16, 256, 4, 1, 1, 1, 2, 0, False),
    (32, 14336, 4096, 32, 64, 128, 5, 1, 2, 2, 1, 4, False),
    (16, 28672, 4096, 16, 64, 256, 2, 1, 1, 2, 2, 4, False),
    (4096, 256, 4096, 64, 64, 128, 4, 1, 4, 2, 1, 4, False),
    (1, 5120, 2880, 16, 64, 64, 7, 3, 1, 2, 1, 0, False),
    (2, 5120, 2880, 16, 64, 64, 7, 3, 1, 2, 1, 0, False),
    (4, 5120, 2880, 16, 64, 64, 7, 3, 1, 2, 1, 0, False),
    (8, 5120, 2880, 16, 64, 64, 8, 3, 1, 2, 1, 0, False),
    (16, 5120, 2880, 16, 64, 64, 8, 3, 1, 2, 1, 0, False),
    (32, 5120, 2880, 16, 32, 64, 8, 1, 1, 2, 1, 0, False),
    (48, 5120, 2880, 16, 64, 64, 8, 1, 1, 2, 1, 0, False),
    (32, 384, 7168, 32, 32, 64, 8, 8, 2, 1, 1, 0, False),
    (32, 384, 16384, 32, 32, 64, 8, 8, 1, 2, 1, 0, False),
    (800, 384, 7168, 64, 96, 64, 4, 4, 2, 2, 1, 0, False),
    (32, 7168, 2048, 32, 32, 64, 8, 1, 2, 1, 1, 0, False),
    (32, 2880, 2048, 32, 32, 64, 8, 2, 1, 2, 1, 0, False),
]
_CONCURRENT_COPY_MNK = [(8192, 8192, 128)]

# Skinny ("sk") policy: one workgroup per block_n columns with the full K, which
# k_waves waves split; no split_k, M/N waves or group_m.
# (m, n, k, block_m, block_n, block_k, stages, k_waves)
_SKINNY = [
    # GLM-5.2 o_proj decode (N=6144, K=2048): each block_m tier the tuned rows use,
    # at its top and with m well below it (rows >= m are masked, not computed)
    (1, 6144, 2048, 8, 24, 64, 4, 4),
    (8, 6144, 2048, 8, 24, 64, 4, 4),
    (9, 6144, 2048, 16, 24, 64, 4, 4),
    (16, 6144, 2048, 16, 24, 64, 4, 4),
    (18, 6144, 2048, 24, 24, 64, 3, 4),
    (24, 6144, 2048, 24, 24, 64, 3, 4),
    (17, 6144, 2048, 32, 24, 64, 3, 4),
    (32, 6144, 2048, 32, 24, 64, 3, 4),
    # GLM-5.2 qkv_a_proj decode (N=2624, K=6144): the tuned rows, including the
    # M that reach a row through padding (3 -> 4, 7 -> 8)
    (1, 2624, 6144, 4, 16, 128, 3, 4),
    (3, 2624, 6144, 4, 16, 128, 4, 4),
    (6, 2624, 6144, 8, 16, 128, 3, 8),
    (7, 2624, 6144, 8, 16, 128, 2, 4),
    # 171 slabs: not a multiple of the 8 XCDs, so the linear slab mapping runs
    (5, 4104, 2048, 8, 24, 64, 4, 4),
    # other slab widths, deeper K, two waves
    (16, 4096, 7168, 16, 32, 64, 4, 4),
    (32, 7168, 4096, 32, 16, 64, 3, 4),
    (3, 2048, 8192, 8, 32, 128, 2, 2),
]


def _policy(hti):
    return "hti" if hti else "ft"


def _unpack(row):
    (
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        stages,
        split_k,
        m_waves,
        n_waves,
        k_waves,
        group_m,
        hti,
    ) = row
    tile = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "stages": stages,
        "split_k": split_k,
        "m_waves": m_waves,
        "n_waves": n_waves,
        "k_waves": k_waves,
        "group_m": group_m,
        "use_half_tile_interleaved": hti,
    }
    return m, n, k, _policy(hti), split_k, tile


def _unpack_skinny(row):
    m, n, k, block_m, block_n, block_k, stages, k_waves = row
    tile = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "stages": stages,
        "split_k": 1,
        "m_waves": 1,
        "n_waves": 1,
        "k_waves": k_waves,
        "group_m": 0,
        "use_half_tile_interleaved": False,
    }
    return m, n, k, "sk", 1, tile


def _all_rows():
    return [_unpack(row) for row in _TUNED] + [_unpack_skinny(row) for row in _SKINNY]


def _default_mnk():
    seen = []
    got = set()
    for m, n, k, *_rest in _all_rows():
        key = (m, n, k)
        if key not in got:
            got.add(key)
            seen.append(key)
    return seen


def _tile_for(m, n, k, policy, split_k):
    for rm, rn, rk, rpolicy, rsplit_k, tile in _all_rows():
        if (rm, rn, rk, rpolicy, rsplit_k) == (m, n, k, policy, split_k):
            return tile
    return None


def run_torch(a, w, dtype):
    # Reference only: fp32 math, cast back. Not timed, not in the table.
    return torch.mm(a.to(dtypes.fp32), w.t().to(dtypes.fp32)).to(dtype)


def _atol_rtol(k, split_k, k_waves, dtype):
    k_scale = (k / 8192) ** 0.5 * split_k * k_waves
    if dtype is dtypes.bf16:
        return 2e-1 * k_scale, 2e-1
    return 5e-2 * k_scale, 5e-2


@benchmark()
def test_hgemm(m, n, k, dtype, policy, split_k):
    tile = _tile_for(m, n, k, policy, split_k)
    assert tile is not None, f"no tuned tile for {(m, n, k, policy, split_k)}"

    torch.manual_seed(SEED)
    a = torch.empty((m, k), dtype=dtype, device="cuda").uniform_(-1, 1)
    w = torch.empty((n, k), dtype=dtype, device="cuda").uniform_(-1, 1)
    # Dirty preallocated C, as the model passes out= a buffer it owns.
    out = torch.randn((m, n), dtype=dtype, device="cuda")
    ref = run_torch(a, w, dtype)

    kwargs = {
        "block_m": tile["block_m"],
        "block_n": tile["block_n"],
        "block_k": tile["block_k"],
        "stages": tile["stages"],
        "split_k": tile["split_k"],
        "m_waves": tile["m_waves"],
        "n_waves": tile["n_waves"],
        "k_waves": tile["k_waves"],
        "group_m": tile["group_m"],
        "policy": policy,
    }

    def _run_flydsl():
        return flydsl_hgemm(a, w, out=out, **kwargs)

    out_mm = torch.empty_like(out)

    def _run_torch_mm():
        torch.mm(a, w.t(), out=out_mm)
        return out_mm

    candidates = {"flydsl": _run_flydsl, "torch_mm": _run_torch_mm}
    if policy == "sk":
        # Decode GEMMs run inside a CUDA graph: capture one call and replay it.
        # The eager call first compiles the kernel outside the capture.
        out_graph = torch.randn((m, n), dtype=dtype, device="cuda")
        flydsl_hgemm(a, w, out=out_graph, **kwargs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flydsl_hgemm(a, w, out=out_graph, **kwargs)
        out_graph.normal_()

        def _run_flydsl_graph():
            graph.replay()
            return out_graph

        candidates["flydsl_graph"] = _run_flydsl_graph

    flops = 2 * m * n * k
    nbytes = (
        m * k * a.element_size() + n * k * w.element_size() + m * n * out.element_size()
    )
    if policy == "sk":
        # partials are summed in fp32 inside the workgroup: one bf16 rounding
        atol, rtol = _atol_rtol(k, 1, 1, dtype)
    else:
        atol, rtol = _atol_rtol(k, split_k, tile["k_waves"], dtype)
    ret = {
        "gfx": get_gfx(),
        "tile": (
            f"{tile['block_m']}x{tile['block_n']}x{tile['block_k']}"
            f"s{tile['stages']}"
        ),
        "waves": f"{tile['m_waves']}x{tile['n_waves']}x{tile['k_waves']}",
    }
    for name, fn in candidates.items():
        # Not `out`: `_run_flydsl` closes over the preallocated buffer.
        y, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            y.to(dtypes.fp32),
            rtol=rtol,
            atol=atol,
            msg=f"{name}: flydsl hgemm",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_hgemm_concurrent_copy(m, n, k, dtype):
    # The HTI tail LDS race shows only with exactly two K tiles (no main loop) and
    # memory traffic from another stream. Without split-K the output is
    # deterministic, so every launch must match the idle-GPU output bitwise.
    tile = {
        "block_m": 256,
        "block_n": 256,
        "block_k": k // 2,
        "stages": 2,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 4,
        "k_waves": 1,
        "group_m": 0,
    }
    torch.manual_seed(SEED)
    a = torch.empty((m, k), dtype=dtype, device="cuda").uniform_(-1, 1)
    w = torch.empty((n, k), dtype=dtype, device="cuda").uniform_(-1, 1)
    golden = flydsl_hgemm(a, w, **tile, policy="hti")
    atol, rtol = _atol_rtol(k, 1, 1, dtype)
    err = checkAllclose(
        run_torch(a, w, dtype).to(dtypes.fp32),
        golden.to(dtypes.fp32),
        rtol=rtol,
        atol=atol,
        msg="flydsl: idle hgemm",
    )

    side_stream = torch.cuda.Stream()
    src = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    out = torch.empty_like(golden)
    launches = 1000
    bad_launches = torch.zeros((), dtype=torch.int64, device="cuda")
    for _ in range(launches):
        with torch.cuda.stream(side_stream):
            dst.copy_(src)
        out.fill_(float("nan"))
        flydsl_hgemm(a, w, out=out, **tile, policy="hti")
        bad_launches += (out != golden).any()
    torch.cuda.synchronize()
    bad = bad_launches.item()
    assert bad == 0, f"{bad} of {launches} launches differ from the idle-GPU output"
    return {
        "gfx": get_gfx(),
        "tile": (
            f"{tile['block_m']}x{tile['block_n']}x{tile['block_k']}"
            f"s{tile['stages']}"
        ),
        "flydsl err": err,
        "launches": launches,
        "bad launches": bad,
    }


def check_skinny_rejects():
    """Shapes the skinny policy cannot run must raise, not compute a wrong C."""
    sk = {
        "block_m": 8,
        "block_n": 24,
        "block_k": 64,
        "stages": 4,
        "split_k": 1,
        "m_waves": 1,
        "n_waves": 1,
        "k_waves": 4,
        "group_m": 0,
        "policy": "sk",
    }
    cases = {
        "N not a multiple of block_n": {"m": 4, "n": 6148, "k": 2048},
        "K not a multiple of k_waves*block_k": {"m": 4, "n": 6144, "k": 2000},
        "M above block_m": {"m": 9, "n": 6144, "k": 2048},
        "bias": {"m": 4, "n": 6144, "k": 2048, "bias": True},
        "fp16 input": {"m": 4, "n": 6144, "k": 2048, "dtype": dtypes.fp16},
    }
    for what, c in cases.items():
        dtype = c.get("dtype", dtypes.bf16)
        a = torch.randn((c["m"], c["k"]), dtype=dtype)
        w = torch.randn((c["n"], c["k"]), dtype=dtype)
        bias = torch.randn((c["n"],), dtype=dtype) if c.get("bias") else None
        try:
            flydsl_hgemm(a, w, bias=bias, **sk)
        except ValueError:
            continue
        raise AssertionError(f"skinny policy accepted an unsupported input: {what}")

    # the launch hands the kernel raw pointers, so a host tensor that gets past
    # here is a GPU memory fault rather than an exception
    host = {
        "b on cpu": lambda: flydsl_hgemm(
            torch.randn((4, 2048), dtype=dtypes.bf16),
            torch.randn((6144, 2048), dtype=dtypes.bf16, device="cpu"),
            **sk,
        ),
        "out on cpu": lambda: flydsl_hgemm(
            torch.randn((4, 2048), dtype=dtypes.bf16),
            torch.randn((6144, 2048), dtype=dtypes.bf16),
            out=torch.empty((4, 6144), dtype=dtypes.bf16, device="cpu"),
            **sk,
        ),
    }
    for what, call in host.items():
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"skinny policy accepted a host tensor: {what}")

    aiter.logger.info(
        "skinny policy rejects all %d unsupported inputs", len(cases) + len(host)
    )


def check_skinny_large_weight():
    """A weight past 2 GiB must not wrap the DMA's 32-bit offset.

    B's byte offset is cast to a signed i32, so a descriptor based at the tensor
    goes negative once the workgroup's slab passes 2 GiB and the load silently
    reads elsewhere. N here is the smallest multiple of block_n that crosses it.
    """
    n, k = 131088, 8192
    need = n * k * 2 + (1 << 28)
    free, _ = torch.cuda.mem_get_info()
    if free < need:
        aiter.logger.info(
            "skipping the >2 GiB skinny weight check: %.1f GiB free, %.1f GiB needed",
            free / 2**30,
            need / 2**30,
        )
        return
    sk = {
        "block_m": 8,
        "block_n": 24,
        "block_k": 64,
        "stages": 4,
        "split_k": 1,
        "m_waves": 1,
        "n_waves": 1,
        "k_waves": 4,
        "group_m": 0,
        "policy": "sk",
    }
    torch.manual_seed(SEED)
    a = torch.empty((1, k), dtype=dtypes.bf16).uniform_(-1, 1)
    w = torch.empty((n, k), dtype=dtypes.bf16).uniform_(-1, 1)
    y = flydsl_hgemm(a, w, **sk)
    atol, rtol = _atol_rtol(k, 1, 1, dtypes.bf16)
    checkAllclose(
        run_torch(a, w, dtypes.bf16).to(dtypes.fp32),
        y.to(dtypes.fp32),
        rtol=rtol,
        atol=atol,
        msg=f"flydsl: skinny N={n} K={k} ({n * k * 2 / 2**30:.2f} GiB weight)",
    )
    del a, w, y
    torch.cuda.empty_cache()


# K-stagger/streaming regression geometries. The first two are Kimi-K3
# row-major BF16 projection shapes; the rest exercise K wraps,
# short final split partitions, slice-K and HTI pipeline boundaries.
# Same row schema as _TUNED.
_MEMORY_TILES = [
    (16, 18432, 1536, 16, 96, 64, 6, 1, 1, 2, 1, 0, False),
    (32, 18432, 1536, 32, 96, 64, 4, 1, 2, 2, 1, 0, False),
    (19, 120, 128, 32, 64, 64, 3, 1, 2, 2, 1, 0, False),
    (19, 120, 1088, 32, 64, 64, 2, 9, 2, 2, 1, 0, False),
    (33, 136, 448, 32, 64, 64, 3, 1, 2, 2, 1, 0, False),
    (35, 136, 1344, 32, 64, 64, 3, 3, 2, 2, 1, 0, False),
    (32, 128, 640, 32, 64, 128, 3, 1, 2, 2, 2, 0, False),
    (65, 264, 768, 32, 64, 64, 5, 1, 2, 2, 1, 4, False),
    (513, 520, 128, 128, 128, 64, 2, 1, 2, 4, 1, 4, True),
    (257, 264, 768, 128, 128, 64, 2, 3, 2, 4, 1, 4, True),
    (512, 512, 640, 128, 128, 64, 2, 1, 2, 4, 1, 4, True),
]


@benchmark()
def test_hgemm_memory(
    m, n, k, dtype, out_dtype, policy, split_k, k_stagger, b_non_temporal
):
    tile = next(
        tile
        for rm, rn, rk, rp, rs, tile in map(_unpack, _MEMORY_TILES)
        if (rm, rn, rk, rp, rs) == (m, n, k, policy, split_k)
    )
    kwargs = {
        key: value for key, value in tile.items() if key != "use_half_tile_interleaved"
    }
    kwargs.update(policy=policy, out_dtype=out_dtype)
    torch.manual_seed(SEED)
    a = torch.empty((m, k), dtype=dtype, device="cuda").uniform_(-1, 1)
    w = torch.empty((n, k), dtype=dtype, device="cuda").uniform_(-1, 1)
    bias = torch.empty((n,), dtype=dtype, device="cuda").uniform_(-1, 1)
    ref = (a.float() @ w.float().t() + bias.float()).to(out_dtype)
    # Contiguous model-owned output, with guards before and after the view.
    guard = torch.full((m * n + 32,), 8192, dtype=out_dtype, device="cuda")
    out = guard[16:-16].view(m, n)
    base_out = torch.empty_like(out)
    candidates = {
        "flydsl_base": lambda: flydsl_hgemm(a, w, out=base_out, bias=bias, **kwargs),
        "flydsl_memory": lambda: flydsl_hgemm(
            a,
            w,
            out=out,
            bias=bias,
            k_stagger=k_stagger,
            b_non_temporal=b_non_temporal,
            **kwargs,
        ),
    }
    flops = 2 * m * n * k
    nbytes = (m * k + n * k + n) * a.element_size() + m * n * out.element_size()
    atol, rtol = _atol_rtol(k, split_k, tile["k_waves"], dtype)
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        y, us = run_perftest(fn)
        err = checkAllclose(
            ref.float(), y.float(), rtol=rtol, atol=atol, msg=f"{name}: hgemm memory"
        )
        assert err == 0, f"{name}: {err} elements outside the reference tolerance"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    assert torch.all(guard[:16] == 8192) and torch.all(
        guard[-16:] == 8192
    ), "output guard overwritten"

    # Exact binary fractions expose missing/repeated K tiles without depending
    # on a relaxed BF16 split-K tolerance. Check repeated dirty-output launches.
    a.copy_(torch.randint(-2, 3, a.shape, device=a.device).to(dtype) / 8)
    w.copy_(torch.randint(-2, 3, w.shape, device=w.device).to(dtype) / 8)
    bias.copy_(torch.randint(-2, 3, bias.shape, device=bias.device).to(dtype) / 8)
    exact = (a.float() @ w.float().t() + bias.float()).to(out_dtype)
    for _ in range(3):
        out.fill_(8192)
        y = candidates["flydsl_memory"]()
        err = checkAllclose(
            exact.float(),
            y.float(),
            rtol=0,
            atol=0,
            msg="K tile coverage",
            tol_err_ratio=0,
        )
        assert err == 0, f"K tile coverage mismatch: {err}"

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidates["flydsl_memory"]()
    a.neg_()
    out.fill_(8192)
    graph.replay()
    exact = (a.float() @ w.float().t() + bias.float()).to(out_dtype)
    err = checkAllclose(
        exact.float(),
        out.float(),
        rtol=0,
        atol=0,
        msg="changed-input graph",
        tol_err_ratio=0,
    )
    assert err == 0, f"graph replay mismatch: {err}"
    return ret


# Kimi-K3 tuned rows and M values that reach them through padding.
_TUNED_MEMORY_MNK = [
    (m, n, k)
    for m, (n, k) in itertools.product(
        (12, 16, 24, 32, 48, 64, 112, 120, 128),
        ((18432, 1536), (20480, 7168)),
    )
]


@benchmark()
def test_hgemm_tuned_memory(m, n, k, dtype):
    from aiter.tuned_gemm import gemm_a16w16_out

    torch.manual_seed(SEED)
    a = torch.empty((m, k), dtype=dtype, device="cuda").uniform_(-1, 1)
    w = torch.empty((n, k), dtype=dtype, device="cuda").uniform_(-1, 1)
    out = torch.empty((m, n), dtype=dtype, device="cuda")
    ref = run_torch(a, w, dtype)

    # Follow the actual tuned lookup, including M padding, rather than
    # launching a hand-selected tile that the model might never use.
    def launch():
        gemm_a16w16_out(a, w, out, otype=dtype)
        return out

    candidates = {"aiter_tuned": launch}
    ret = {"gfx": get_gfx()}
    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n) * a.element_size()
    for name, fn in candidates.items():
        y, us = run_perftest(fn)
        err = checkAllclose(ref.float(), y.float(), rtol=1e-2, atol=1e-2, msg=name)
        assert err == 0, f"{name}: tuned memory row mismatch {err}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


_BAND_MNK = [
    (m, n, k)
    for m, (n, k) in itertools.product(
        (
            1,
            7,
            8,
            12,
            16,
            21,
            24,
            32,
            35,
            40,
            48,
            49,
            56,
            64,
            80,
            96,
            98,
            112,
            120,
            128,
        ),
        ((18432, 1536), (20480, 7168)),
    )
]


@benchmark()
def test_hgemm_band(m, n, k, dtype):
    config = band_stream_config(m, n, k)
    config.pop("use_half_tile_interleaved")
    torch.manual_seed(SEED)
    a = torch.empty((m, k), dtype=dtype, device="cuda").uniform_(-1, 1)
    w = torch.empty((n, k), dtype=dtype, device="cuda").uniform_(-1, 1)
    ref = run_torch(a, w, dtype)
    guard = torch.full((m * n + 32,), 8192, dtype=dtype, device="cuda")
    out = guard[16:-16].view(m, n)
    candidates = {"flydsl_band": lambda: flydsl_hgemm(a, w, out=out, **config)}
    ret = {"gfx": get_gfx()}
    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n) * a.element_size()
    for name, fn in candidates.items():
        y, us = run_perftest(fn)
        err = checkAllclose(ref.float(), y.float(), rtol=1e-2, atol=1e-2, msg=name)
        assert err == 0, f"{name}: mismatch {err}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    golden = out.clone()
    for _ in range(10):
        out.fill_(float("nan"))
        candidates["flydsl_band"]()
        assert torch.equal(out, golden), "band-stream is not deterministic"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidates["flydsl_band"]()
    a.neg_()
    ref = run_torch(a, w, dtype)
    for _ in range(10):
        out.fill_(float("nan"))
        graph.replay()
        err = checkAllclose(
            ref.float(),
            out.float(),
            rtol=1e-2,
            atol=1e-2,
            msg="band graph",
            printLog=False,
        )
        assert err == 0, f"band graph mismatch {err}"
    assert torch.all(guard[:16] == 8192) and torch.all(
        guard[-16:] == 8192
    ), "band output overrun"
    return ret


@benchmark()
def test_hgemm_tuned_strides(m, n, k, dtype, layout):
    from aiter.tuned_gemm import gemm_a16w16_out

    torch.manual_seed(SEED)
    if layout[0] == "t":
        a = torch.empty((k, m), dtype=dtype, device="cuda").uniform_(-1, 1).t()
    else:
        a = torch.empty((m, k * 2), dtype=dtype, device="cuda").uniform_(-1, 1)[:, ::2]
    if layout[1] == "t":
        w = torch.empty((k, n), dtype=dtype, device="cuda").uniform_(-1, 1).t()
    else:
        w = torch.empty((n, k * 2), dtype=dtype, device="cuda").uniform_(-1, 1)[:, ::2]
    out = torch.empty((m, n), dtype=dtype, device="cuda")
    ref = run_torch(a, w, dtype)

    def launch():
        gemm_a16w16_out(a, w, out, otype=dtype)
        return out

    ret = {"gfx": get_gfx()}
    candidates = {"tuned_strided": launch}
    flops = 2 * m * n * k
    # Logical bytes; normalization adds physical traffic not counted here.
    nbytes = (m * k + n * k + m * n) * a.element_size()
    for name, fn in candidates.items():
        y, us = run_perftest(fn)
        err = checkAllclose(ref.float(), y.float(), rtol=1e-2, atol=1e-2, msg=name)
        assert err == 0, f"{name}: {err} mismatches"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        a.neg_()
        out.fill_(float("nan"))
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    ref = run_torch(a, w, dtype)
    err = checkAllclose(
        ref.float(), out.float(), rtol=1e-2, atol=1e-2, msg="strided graph"
    )
    assert err == 0, f"strided graph mismatch {err}"
    return ret


def check_band_rejects():
    config = band_stream_config(16, 18432, 1536)
    config.pop("use_half_tile_interleaved")
    a = torch.zeros((16, 1536), dtype=dtypes.bf16, device="cuda")
    w = torch.zeros((18432, 1536), dtype=dtypes.bf16, device="cuda")
    for override in (
        {"split_k": 2},
        {"k_waves": 2},
        {"group_m": 4},
        {"b_non_temporal": False},
        {"out_dtype": dtypes.fp32},
        {"bias": torch.zeros(18432, device="cuda", dtype=dtypes.bf16)},
    ):
        try:
            flydsl_hgemm(a, w, **(config | override))
        except ValueError:
            continue
        raise AssertionError(f"band-stream accepted {override}")
    for x, weight in (
        (a.half(), w.half()),
        (a[:, ::2], w[:, ::2]),
        (a, w[:1024]),
        (a.cpu(), w.cpu()),
    ):
        try:
            flydsl_hgemm(x, weight, **config)
        except (ValueError, RuntimeError):
            continue
        raise AssertionError("band-stream accepted unsupported tensors")
    aiter.logger.info("band-stream rejects unsupported input/configurations")


def check_hgemm_memory_options():
    """Legacy names, tuning options and unsupported architectures stay explicit."""
    config = _unpack(_MEMORY_TILES[0])[-1]

    def name(cfg, gfx="gfx950"):
        return flydsl_hgemm_kernel_name(
            dtype=dtypes.bf16,
            out_dtype=dtypes.bf16,
            config=cfg,
            has_bias=False,
            target_gfx=gfx,
        )

    legacy = name(config)
    for stagger, nt in itertools.product((0, 1, 3, 127), (False, True)):
        cfg = dict(config, k_stagger=stagger, b_non_temporal=nt)
        encoded = name(cfg)
        parsed = get_flydsl_hgemm_kernel_params(encoded)
        assert {key: parsed[key] for key in cfg} == cfg, encoded
        if stagger == 0 and not nt:
            assert encoded == legacy
    for cfg, gfx in [
        (dict(config, k_stagger=-1), "gfx950"),
        (dict(config, k_stagger=1.5), "gfx950"),
        (dict(config, k_stagger=True), "gfx950"),
        (dict(config, b_non_temporal=1), "gfx950"),
        (dict(config, k_stagger=1), "gfx1250"),
        (dict(config, b_non_temporal=True), "gfx1250"),
        (dict(config, k_stagger=1, policy="sk"), "gfx950"),
        (dict(config, b_non_temporal=True, policy="sk"), "gfx950"),
    ]:
        try:
            name(cfg, gfx)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"accepted unsupported memory config: {cfg}, {gfx}")
    aiter.logger.info("hgemm memory-option names and rejection checks passed")


def summarize(title, rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    aiter.logger.info("%s summary (markdown):\n%s", title, df.to_markdown(index=False))


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("flydsl hgemm unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16}",
        help="""Data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=_default_mnk()
        + _CONCURRENT_COPY_MNK
        + [row[:3] for row in _MEMORY_TILES]
        + _TUNED_MEMORY_MNK
        + _BAND_MNK,
        help="""Shape of mnk.
        e.g.:   -s 128,4096,4096
                --mnk 32,384,7168""",
    )
    parser.add_argument("--k-stagger", type=int, nargs="*", default=[1, 3, 127])
    parser.add_argument(
        "--b-non-temporal", type=dtypes.str2bool, nargs="*", default=[False, True]
    )
    parser.add_argument(
        "--out-dtype", type=dtypes.str2Dtype, nargs="*", default="bf16,fp32,"
    )
    args = parser.parse_args()
    wanted = set(args.mnk)

    for dtype in args.dtype:
        rows = []
        for m, n, k, policy, split_k, _tile in _all_rows():
            if (m, n, k) not in wanted:
                continue
            rows.append(test_hgemm(m, n, k, dtype, policy, split_k))
        summarize(f"flydsl_hgemm {dtype}", rows)
        rows = [
            test_hgemm_concurrent_copy(m, n, k, dtype)
            for m, n, k in _CONCURRENT_COPY_MNK
            if (m, n, k) in wanted
        ]
        summarize(f"flydsl_hgemm {dtype} with a concurrent copy", rows)
    for dtype, out_dtype in itertools.product(args.dtype, args.out_dtype):
        if out_dtype not in (dtype, dtypes.fp32):
            continue
        rows = [
            test_hgemm_memory(m, n, k, dtype, out_dtype, policy, split_k, stagger, nt)
            for (m, n, k, policy, split_k, _), stagger, nt in itertools.product(
                map(_unpack, _MEMORY_TILES), args.k_stagger, args.b_non_temporal
            )
            if (m, n, k) in wanted
        ]
        summarize(f"flydsl_hgemm memory {dtype} -> {out_dtype}", rows)
    for dtype in args.dtype:
        if dtype == dtypes.bf16:
            summarize(
                f"tuned hgemm memory {dtype}",
                [
                    test_hgemm_tuned_memory(m, n, k, dtype)
                    for m, n, k in _TUNED_MEMORY_MNK
                    if (m, n, k) in wanted
                ],
            )
    for dtype in args.dtype:
        if dtype == dtypes.bf16:
            summarize(
                "band-stream HGEMM",
                [
                    test_hgemm_band(m, n, k, dtype)
                    for m, n, k in _BAND_MNK
                    if (m, n, k) in wanted
                ],
            )
    summarize(
        "tuned HGEMM strided inputs",
        [
            test_hgemm_tuned_strides(m, n, k, dtypes.bf16, layout)
            for (m, n, k), layout in itertools.product(
                ((16, 18432, 1536), (112, 20480, 7168)), ("ss", "st", "ts", "tt")
            )
        ],
    )
    check_band_rejects()
    check_hgemm_memory_options()
    check_skinny_rejects()
    check_skinny_large_weight()


if __name__ == "__main__":
    main()

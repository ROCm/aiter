# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for arch-dispatched FlyDSL A16W16 HGEMM.

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
from unittest.mock import patch

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import flydsl_hgemm
from aiter.ops.flydsl.gemm_a16w16_band import band_stream_config
from aiter.ops.flydsl.gemm_kernels import (
    flydsl_hgemm_kernel_name,
    get_flydsl_hgemm_kernel_params,
)
from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
    GEMM_A16W16_DTYPE_BF16,
    split_k_workspace_supported,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

# The public wrapper selects the architecture; unsupported cards skip in main.
SUPPORTED_GFX = ["gfx950", "gfx1250"]
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
    (1798, 6288, 7168, 256, 256, 64, 2, 1, 2, 4, 1, 0, True),
    (670, 3584, 7168, 256, 256, 64, 2, 4, 2, 4, 1, 0, True),
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
    (576, 20480, 7168, 192, 256, 64, 2, 1, 4, 4, 1, 0, False),
    (4032, 2880, 7168, 192, 256, 64, 2, 1, 4, 4, 1, 0, False),
    (2078, 6288, 7168, 208, 256, 64, 2, 1, 2, 4, 1, 0, False),
    (2079, 6288, 7168, 224, 256, 64, 2, 1, 2, 4, 1, 0, False),
    (279, 7168, 35840, 320, 192, 64, 2, 5, 4, 2, 1, 0, False),
    (312, 6288, 7168, 320, 192, 64, 2, 7, 4, 4, 1, 0, False),
    (320, 6288, 7168, 320, 192, 64, 2, 7, 4, 4, 1, 0, False),
    (322, 6288, 7168, 384, 192, 64, 2, 7, 8, 1, 1, 0, False),
    (376, 6288, 7168, 384, 192, 64, 2, 7, 8, 1, 1, 0, False),
    (384, 6288, 7168, 384, 192, 64, 2, 7, 8, 1, 1, 0, False),
    (264, 8448, 7168, 320, 192, 64, 2, 4, 4, 4, 1, 0, False),
    (280, 8448, 7168, 320, 192, 64, 2, 4, 4, 4, 1, 0, False),
    (248, 8448, 7168, 256, 256, 64, 2, 7, 4, 2, 1, 0, False),
    (255, 8448, 7168, 256, 256, 64, 2, 7, 2, 4, 1, 0, False),
    (256, 8448, 7168, 256, 256, 64, 2, 7, 4, 2, 1, 0, False),
    (288, 8448, 7168, 320, 192, 64, 2, 4, 4, 4, 1, 0, False),
    (292, 8448, 7168, 320, 192, 64, 2, 4, 4, 4, 1, 0, False),
    (706, 2880, 7168, 256, 256, 64, 2, 7, 4, 2, 1, 0, False),
    (1152, 2112, 7168, 320, 192, 64, 2, 4, 4, 4, 1, 0, False),
    (386, 6288, 7168, 400, 128, 64, 2, 4, 5, 2, 1, 0, False),
    (522, 8448, 7168, 320, 192, 64, 2, 2, 4, 4, 1, 0, False),
    (543, 6288, 7168, 128, 128, 64, 3, 1, 2, 2, 2, 1, False),
    (635, 6288, 7168, 128, 128, 64, 3, 1, 4, 4, 1, 1, False),
    (639, 6288, 7168, 128, 128, 64, 3, 1, 2, 2, 2, 1, False),
    (640, 6288, 7168, 128, 128, 64, 3, 1, 4, 4, 1, 1, False),
    (4033, 2880, 7168, 256, 192, 64, 2, 1, 4, 4, 1, 0, False),
    (5382, 2112, 7168, 256, 192, 64, 2, 1, 4, 4, 1, 0, False),
    (3079, 8448, 7168, 288, 192, 64, 2, 1, 2, 4, 1, 0, False),
    (4609, 3584, 7168, 320, 224, 64, 2, 1, 4, 2, 1, 2, False),
    (5758, 2880, 7168, 320, 224, 64, 2, 1, 4, 2, 1, 16, False),
    (5762, 2880, 7168, 224, 320, 64, 2, 1, 2, 4, 1, 8, False),
    (6136, 2880, 7168, 224, 320, 64, 2, 1, 2, 4, 1, 8, False),
    (1025, 8448, 7168, 224, 192, 64, 3, 1, 2, 6, 1, 0, False),
    (2686, 2880, 7168, 160, 192, 64, 3, 1, 2, 4, 1, 16, False),
    (2689, 3072, 512, 192, 192, 64, 3, 1, 3, 4, 1, 0, False),
    (5381, 3072, 512, 320, 224, 64, 2, 1, 4, 2, 1, 4, False),
]
_CONCURRENT_COPY_MNK = [(8192, 8192, 128)]
# GLM-5.3 / K3.1 Pro indexer wq_b: N = 32 heads * 128, K = q_lora_rank.
_PRESHUFFLE_MNK = [(m, 4096, k) for m in (6, 32, 512) for k in (1536, 2048)] + [
    (33, 80, 1536),
    (513, 4112, 2048),
]

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


@benchmark()
def test_hgemm_preshuffle(m, n, k, dtype):
    """One AITER weight layout for a native BF16 GEMM and a Mono indexer."""
    torch.manual_seed(SEED)
    a = torch.randn(m, k, dtype=dtype, device="cuda") * 0.1
    raw = torch.randn(n, k, dtype=dtype, device=a.device) * 0.1
    weight = shuffle_weight(raw)
    out = torch.full((m, n), float("nan"), dtype=dtype, device=a.device)
    config = {
        "block_m": (8 if m <= 8 else 32) if m <= 32 else 128,
        "block_n": 16 if m <= 32 else 128,
        "block_k": 64,
        "stages": 3 if m <= 32 else 2,
        "m_waves": 1 if m <= 32 else 2,
        "n_waves": 1 if m <= 32 else 2,
        "k_waves": 4 if m <= 32 else 1,
        "policy": "sk" if m <= 32 else "ft",
    }

    def eager():
        return flydsl_hgemm(a, weight, out=out, bpreshuffle=True, **config)

    eager()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        eager()
    a.mul_(0.75)
    out.fill_(float("nan"))
    ref = run_torch(a, raw, dtype)

    def replay():
        graph.replay()
        return out

    ret = {"gfx": get_gfx_runtime()}
    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n) * 2
    for name, fn in {"preshuffle_graph": replay, "preshuffle": eager}.items():
        actual, us = run_perftest(fn)
        assert actual.data_ptr() == out.data_ptr()
        err = checkAllclose(ref.float(), actual.float(), atol=0.003, rtol=0.02)
        assert err == 0
        ret.update(
            {
                f"{name} us": us,
                f"{name} TFLOPS": flops / us / 1e6,
                f"{name} TB/s": nbytes / us / 1e6,
                f"{name} err": err,
            }
        )
    return ret


def check_preshuffle_compiled_dispatch():
    """A strided 3D input must not send shuffled bytes through F.linear."""
    from aiter import tuned_gemm as tg

    a = torch.randn(2, 3, 1536, dtype=dtypes.bf16, device="cuda").transpose(0, 1)
    raw = torch.randn(4096, 1536, dtype=a.dtype, device=a.device) * 0.02
    weight = torch.nn.Parameter(shuffle_weight(raw), requires_grad=False)
    weight.is_shuffled = True
    out = torch.empty(3, 2, 4096, dtype=a.dtype, device=a.device)
    config = {
        "block_m": 8,
        "block_n": 16,
        "block_k": 64,
        "stages": 3,
        "split_k": 1,
        "m_waves": 1,
        "n_waves": 1,
        "k_waves": 4,
        "group_m": 0,
        "policy": "sk",
    }
    name = flydsl_hgemm_kernel_name(
        dtype=a.dtype,
        out_dtype=a.dtype,
        config=config,
        has_bias=False,
        target_gfx="gfx950",
    )

    def lookup(**kwargs):
        assert kwargs["bpreshuffle"]
        return {"libtype": "flydsl", "solidx": 0, "kernelName": name}

    compiled = torch.compile(lambda x, w, y: tg.tgemm.mm(x, w, out=y), fullgraph=True)
    with patch.object(tg, "get_GEMM_A16W16_config", lookup):
        compiled(a, weight, out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled(a, weight, out)
        a.neg_()
        out.fill_(float("nan"))
        graph.replay()
        ref = torch.nn.functional.linear(a.float(), raw.float())
        assert checkAllclose(ref, out.float(), rtol=0.02, atol=0.003) == 0
    for override in ({"policy": "ht"}, {"split_k": 2}, {"k_waves": 2}):
        try:
            flydsl_hgemm(
                a.reshape(6, 1536),
                weight,
                bpreshuffle=True,
                **({"policy": "ft"} | override),
            )
        except ValueError:
            continue
        raise AssertionError(
            f"preshuffle accepted unsupported configuration: {override}"
        )


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

    candidates = {"flydsl": _run_flydsl}
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
    if split_k_workspace_supported(
        m, n, split_k, GEMM_A16W16_DTYPE_BF16 if dtype is dtypes.bf16 else 0, False
    ):
        candidates["flydsl_skws"] = lambda: flydsl_hgemm(
            a, w, out=out, split_k_workspace=True, **kwargs
        )

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
        "gfx": get_gfx_runtime(),
        "tile": (
            f"{tile['block_m']}x{tile['block_n']}x{tile['block_k']}s{tile['stages']}"
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
        assert err == 0, f"{name}: {err:.2%} of elements outside tolerance"
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

    def check_wave_group_race(name, shape, config, repeats):
        m_race, n_race, k_race = shape
        gen = torch.Generator(device="cuda")
        gen.manual_seed(SEED)
        a_race = torch.randint(
            -1, 2, (m_race, k_race), generator=gen, device="cuda"
        ).to(dtype)
        w_race = torch.randint(
            -1, 2, (n_race, k_race), generator=gen, device="cuda"
        ).to(dtype)
        ref = run_torch(a_race, w_race, dtype)
        out_race = torch.empty_like(ref)
        wrong = torch.zeros((), dtype=torch.int64, device="cuda")
        for _ in range(repeats):
            out_race.fill_(float("nan"))
            flydsl_hgemm(
                a_race,
                w_race,
                out=out_race,
                block_k=64,
                stages=2,
                split_k=1,
                m_waves=2,
                k_waves=1,
                group_m=0,
                policy="hti",
                **config,
            )
            wrong += (out_race != ref).any()
        torch.cuda.synchronize()
        bad_race = int(wrong)
        assert bad_race == 0, f"{name}: {bad_race} of {repeats} launches are wrong"

    check_wave_group_race(
        "HTI main loop",
        (2048, 4096, 1024),
        {"block_m": 64, "block_n": 64, "n_waves": 2},
        512,
    )
    for block_m in (128, 256):
        check_wave_group_race(
            f"HTI {block_m}x256 epilogue",
            (1024, 1024, 1024),
            {"block_m": block_m, "block_n": 256, "n_waves": 2},
            3,
        )

    return {
        "gfx": get_gfx_runtime(),
        "tile": (
            f"{tile['block_m']}x{tile['block_n']}x{tile['block_k']}s{tile['stages']}"
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
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        y, us = run_perftest(fn)
        err = checkAllclose(
            ref.float(), y.float(), rtol=rtol, atol=atol, msg=f"{name}: hgemm memory"
        )
        assert err == 0, f"{name}: {err} elements outside the reference tolerance"
        assert err == 0, f"{name}: {err:.2%} of elements outside tolerance"
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

    captured_exact = exact.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidates["flydsl_memory"]()
    a.neg_()
    out.fill_(8192)
    graph.replay()
    exact = (a.float() @ w.float().t() + bias.float()).to(out_dtype)
    if out_dtype == dtypes.bf16 and split_k > 1:
        # Packed BF16 atomic split-K may add the same partials in a different
        # order during graph replay. Accept only the two adjacent BF16 values;
        # the exact launches above still catch missing or repeated K tiles.
        lower = torch.nextafter(exact, torch.full_like(exact, -float("inf")))
        upper = torch.nextafter(exact, torch.full_like(exact, float("inf")))
        within_one_ulp = (out == exact) | (out == lower) | (out == upper)
        assert torch.isfinite(out).all(), "graph replay produced a non-finite value"
        assert within_one_ulp.all(), (
            "graph replay exceeded one BF16 ULP: "
            f"{(~within_one_ulp).sum().item()} mismatched elements"
        )
    else:
        err = checkAllclose(
            exact.float(),
            out.float(),
            rtol=0,
            atol=0,
            msg="changed-input graph",
            tol_err_ratio=0,
        )
        assert err == 0, f"graph replay mismatch: {err}"
    assert torch.any(out != captured_exact), "graph replay reused the captured input"
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
    ret = {"gfx": get_gfx_runtime()}
    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n) * a.element_size()
    for name, fn in candidates.items():
        y, us = run_perftest(fn)
        err = checkAllclose(ref.float(), y.float(), rtol=1e-2, atol=1e-2, msg=name)
        assert err == 0, f"{name}: tuned memory row mismatch {err}"
        assert err == 0, f"{name}: {err:.2%} of elements outside tolerance"
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
    ret = {"gfx": get_gfx_runtime()}
    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n) * a.element_size()
    for name, fn in candidates.items():
        y, us = run_perftest(fn)
        err = checkAllclose(ref.float(), y.float(), rtol=1e-2, atol=1e-2, msg=name)
        assert err == 0, f"{name}: mismatch {err}"
        assert err == 0, f"{name}: {err:.2%} of elements outside tolerance"
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

    ret = {"gfx": get_gfx_runtime()}
    candidates = {"tuned_strided": launch}
    flops = 2 * m * n * k
    # Logical bytes; normalization adds physical traffic not counted here.
    nbytes = (m * k + n * k + m * n) * a.element_size()
    for name, fn in candidates.items():
        y, us = run_perftest(fn)
        err = checkAllclose(ref.float(), y.float(), rtol=1e-2, atol=1e-2, msg=name)
        assert err == 0, f"{name}: {err} mismatches"
        assert err == 0, f"{name}: {err:.2%} of elements outside tolerance"
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


# Existing #7851 CSV keys (M is decode tokens, N=128, K=4096) and the
# supplied short-K reference; all are ordinary --mnk sweep entries.
_TDM_FP32_MNK = [
    (1, 128, 4096),
    (2, 128, 4096),
    (4, 128, 4096),
    (8, 128, 4096),
    (16, 128, 4096),
    # DSV4.1's first 12 existing gfx1250 CSV rows, in file order:
    # decode M tokens, weight[N=32,K=5120], no preshuffle.
    *[(m, 32, 5120) for m in (6, 12, 18, 24, 30, 36, 42, 48, 72, 96, 128, 144)],
]

_TDM_MNK = _TDM_FP32_MNK + [
    (1, 128, 128),
    (137, 144, 768),
    (131, 139, 512),
    (257, 264, 2304),
    (8, 4096, 4096),
    (8192, 8192, 8192),
]


def _tdm_config(m, n, k, split_k, stages):
    block_k = 128 if k % (128 * split_k) == 0 else 64
    if k % (block_k * split_k) or k // split_k < (stages - 1) * block_k:
        return None
    tile = 256 if min(m, n, k) >= 4096 else 128
    return {
        "block_m": tile,
        "block_n": tile,
        "block_k": block_k,
        "stages": stages,
        "m_waves": 2,
        "n_waves": 2,
        "k_waves": 1,
        "group_m": 0,
        "split_k": split_k,
    }


def _guarded_matrix(rows, cols, dtype):
    pitch = (cols + 7) // 8 * 8 + 16
    storage = torch.full((rows + 2, pitch), 17, device="cuda", dtype=dtype)
    return storage[1 : rows + 1, 8 : cols + 8], storage


def _assert_guards(storage, cols):
    assert torch.all(storage[0] == 17) and torch.all(storage[-1] == 17)
    assert torch.all(storage[:, :8] == 17) and torch.all(storage[:, cols + 8 :] == 17)


def run_torch_tdm(a, weight, bias, dtype, split_k, with_rounding_budget=False):
    # Each split retains FP32 partials; round only after fixed-order reduction.
    # Legacy tdma/tdmia names execute this same path.
    if split_k == 1:
        product = a.float() @ weight.float().t()
        torch.cuda.current_stream(a.device).synchronize()
        partial_magnitude = product.abs()
    else:
        width = a.shape[1] // split_k
        product = torch.zeros(
            a.shape[0], weight.shape[0], device=a.device, dtype=torch.float32
        )
        partial_magnitude = torch.zeros_like(product)
        for part in range(split_k):
            start = part * width
            partial = (
                a[:, start : start + width].float()
                @ weight[:, start : start + width].float().t()
            )
            torch.cuda.current_stream(a.device).synchronize()
            partial_magnitude += partial.abs()
            product += partial
    result = product.to(dtype)
    if bias is not None:
        result = (result.float() + bias.float()).to(dtype)
    if with_rounding_budget:
        # Bound independent FP32 dot-product order error separately from final
        # output rounding. No input-dtype rounding budget per split.
        magnitude = partial_magnitude + product.abs()
        if bias is not None:
            magnitude = magnitude + bias.float().abs()
        # Retain the existing componentwise budget, without the former
        # split_k multiplier for BF16 atomic-update rounding.
        return result, torch.finfo(dtype).eps * magnitude
    return result


@benchmark()
def test_hgemm_tdm(m, n, k, dtype, has_bias, layout, split_k, stages, interleave):
    config = _tdm_config(m, n, k, split_k, stages)
    assert config is not None, "K must contain complete TDM tiles in every split"
    torch.manual_seed(SEED)
    a, a_storage = _guarded_matrix(m, k, dtype)
    a.uniform_(-0.25, 0.25)
    if layout == "nt":
        weight, w_storage = _guarded_matrix(n, k, dtype)
    else:
        transposed, w_storage = _guarded_matrix(k, n, dtype)
        weight = transposed.t()
    weight.uniform_(-0.25, 0.25)
    bias = (
        torch.empty(n, device="cuda", dtype=dtype).uniform_(-0.25, 0.25)
        if has_bias
        else None
    )
    ref, rounding_budget = run_torch_tdm(
        a, weight, bias, dtype, split_k, with_rounding_budget=True
    )
    out, storage = _guarded_matrix(m, n, dtype)

    def launch():
        if layout == "nn":
            from aiter.ops.flydsl import flydsl_gemm_a16w16_tdm

            kwargs = {key: value for key, value in config.items() if key != "k_waves"}
            return flydsl_gemm_a16w16_tdm(
                a, weight.t(), out, bias, layout="nn", interleave=interleave, **kwargs
            )
        return flydsl_hgemm(
            a,
            weight,
            out=out,
            bias=bias,
            policy="tdmi" if interleave else "tdm",
            **config,
        )

    candidates = {"tdm": launch}
    outputs = {"tdm": (out, storage)}
    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n + (n if has_bias else 0)) * a.element_size()
    ret = {
        "gfx": get_gfx_runtime(),
        "tile": f"{config['block_m']}x{config['block_n']}x{config['block_k']}",
    }
    for name, fn in candidates.items():
        out, storage = outputs[name]
        out.fill_(float("nan"))
        y = fn()
        assert y.data_ptr() == out.data_ptr()
        y, us = run_perftest(fn)
        atol = 0.012 if dtype == dtypes.bf16 else 0.002
        rtol = 0.02 if dtype == dtypes.bf16 else 0.003
        if split_k > 1:
            # All policy names use the same FP32 partial reduction.
            budget = atol + rounding_budget
            normalized_error = (ref.float() - y.float()).abs() / budget
            err = checkAllclose(
                torch.zeros_like(normalized_error),
                normalized_error,
                atol=1,
                rtol=0,
                msg=f"{name} componentwise rounding budget",
                tol_err_ratio=0,
            )
        else:
            err = checkAllclose(
                ref.float(),
                y.float(),
                atol=atol,
                rtol=rtol,
                msg=name,
                tol_err_ratio=0,
            )
        assert torch.isfinite(y).all() and err == 0, f"{name}: reference mismatch {err}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        _assert_guards(storage, n)
    _assert_guards(a_storage, k)
    _assert_guards(w_storage, k if layout == "nt" else n)

    # Each split contributes a distinct exact product. Repeated poisoned-output
    # replays detect missing/reread splits, stale bias and output accumulation.
    a.zero_()
    weight.zero_()
    width = k // split_k
    row_factor = ((torch.arange(m, device="cuda") % 3) - 1).to(dtype)
    col_factor = ((torch.arange(n, device="cuda") % 3) - 1).to(dtype)
    for part in range(split_k):
        a[:, part * width] = row_factor * (part + 1) / 16
        weight[:, part * width] = col_factor / 16
    if bias is not None:
        bias.copy_(col_factor / 8)
    exact = run_torch_tdm(a, weight, bias, dtype, split_k)
    for name, fn in candidates.items():
        out, storage = outputs[name]
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                fn()
            for _ in range(3):
                out.fill_(float("nan"))
                graph.replay()
                err = checkAllclose(
                    exact.float(),
                    out.float(),
                    atol=0,
                    rtol=0,
                    tol_err_ratio=0,
                    msg=f"{name} exact split coverage",
                )
                assert err == 0
            a.neg_()
            if bias is not None:
                bias.neg_()
            changed = run_torch_tdm(a, weight, bias, dtype, split_k)
            out.fill_(float("nan"))
            graph.replay()
            err = checkAllclose(
                changed.float(),
                out.float(),
                atol=0,
                rtol=0,
                tol_err_ratio=0,
                msg=f"{name} changed graph",
            )
            assert err == 0
            a.neg_()
            if bias is not None:
                bias.neg_()
            _assert_guards(storage, n)
        finally:
            graph.reset()
    return ret


@benchmark()
def test_hgemm_tdm_tuned(m, n, k, dtype, has_bias, interleave):
    from unittest import mock

    import aiter.tuned_gemm as tuned
    from aiter.jit.utils.chip_info import get_cu_num

    config = {
        "block_m": 128,
        "block_n": 128,
        "block_k": 128,
        "stages": 2,
        "split_k": 3,
        "m_waves": 2,
        "n_waves": 2,
        "k_waves": 1,
        "group_m": 0,
    }
    a = torch.zeros(m, k, device="cuda", dtype=dtype)
    weight = torch.zeros(n, k, device="cuda", dtype=dtype)
    row = ((torch.arange(m, device="cuda") % 3) - 1).to(dtype)
    col = ((torch.arange(n, device="cuda") % 3) - 1).to(dtype)
    for part in range(3):
        a[:, part * (k // 3)] = row * (part + 1) / 16
        weight[:, part * (k // 3)] = col / 16
    bias = col / 8 if has_bias else None
    reference = run_torch_tdm(a, weight, bias, dtype, 3)
    policies = ("tdmi" if interleave else "tdm",)
    outputs = {
        policy: torch.full((m, n), float("nan"), device="cuda", dtype=dtype)
        for policy in policies
    }
    ret = {"gfx": get_gfx_runtime()}
    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n + (n if has_bias else 0)) * a.element_size()
    key = (
        get_gfx_runtime(),
        get_cu_num(),
        m,
        n,
        k,
        has_bias,
        str(dtype),
        str(dtype),
        False,
        False,
    )
    for policy in policies:
        name = flydsl_hgemm_kernel_name(
            dtype=dtype,
            out_dtype=dtype,
            has_bias=has_bias,
            target_gfx=get_gfx_runtime(),
            config={**config, "policy": policy},
        )
        table = {
            key: {"libtype": "flydsl", "solidx": 0, "splitK": 3, "kernelName": name}
        }
        out = outputs[policy]

        def launch(out=out):
            tuned.gemm_a16w16_out(a, weight, out, bias=bias, otype=dtype)
            return out

        candidates = {policy: launch}
        tuned.get_GEMM_A16W16_config.cache_clear()
        try:
            with mock.patch.object(
                tuned, "get_GEMM_A16W16_config_", return_value=table
            ):
                selected = tuned.get_GEMM_A16W16_tuned_config(
                    m,
                    n,
                    k,
                    has_bias,
                    str(dtype),
                    str(dtype),
                    gfx=get_gfx_runtime(),
                    cu_num=get_cu_num(),
                )
                assert selected["kernelName"] == name
                for candidate, fn in candidates.items():
                    y, us = run_perftest(fn)
                    err = checkAllclose(
                        reference.float(),
                        y.float(),
                        atol=0,
                        rtol=0,
                        tol_err_ratio=0,
                        msg=f"{candidate} tuned dispatch",
                    )
                    assert y.data_ptr() == out.data_ptr() and err == 0
                    ret[f"{candidate} us"] = us
                    ret[f"{candidate} TFLOPS"] = flops / us / 1e6
                    ret[f"{candidate} TB/s"] = nbytes / us / 1e6
                    ret[f"{candidate} err"] = err
        finally:
            tuned.get_GEMM_A16W16_config.cache_clear()
    return ret


def run_torch_tdm_fp32(a, weight, bias, dtype, policy="tdm"):
    product = a.to(dtypes.fp32) @ weight.to(dtypes.fp32).t()
    # Preserve each backend's public bias convention. The original FT/HT
    # path fuses bias in FP32; TDM rounds the product before adding bias.
    if policy in ("ft", "ht"):
        return (product if bias is None else product + bias.float()).to(dtype)
    product = product.to(dtype)
    return product if bias is None else (product.float() + bias.float()).to(dtype)


@benchmark()
def test_hgemm_tdm_fp32(m, n, k, dtype, has_bias):
    """Model decode outputs and the split5 short-K reference; no CSV dependency."""
    # #7851: M=1/2/4/8/16, N=128, K=4096. The user's separate short-K
    # reference is M=6,N=32,K=5120, with four K tiles per split.
    split_k = 5 if k == 5120 else 8
    block_k = 256 if k == 5120 else 128
    cfg = {
        "block_m": 16,
        "block_n": 16,
        "block_k": block_k,
        "stages": 5,
        "m_waves": 1,
        "n_waves": 1,
        "k_waves": 1,
        "split_k": split_k,
        "policy": "tdm",
    }
    a, a_store = _guarded_matrix(m, k, dtype)
    weight, w_store = _guarded_matrix(n, k, dtype)
    out, o_store = _guarded_matrix(m, n, dtype)
    a.zero_()
    weight.zero_()
    row_sign = (1 - 2 * (torch.arange(m, device="cuda") % 2)).to(dtype)
    col_sign = (1 - 2 * (torch.arange(n, device="cuda") % 2)).to(dtype)
    factor_a = 1 / 16 if dtype == dtypes.bf16 else 1 / 32
    factor_b = 1 / 16 if dtype == dtypes.bf16 else 1 / 64
    for part in range(3):
        a[:, part * (k // split_k)] = row_sign
        weight[:, part * (k // split_k)] = col_sign
        a[:, part * (k // split_k) + 1] = row_sign * factor_a
        weight[:, part * (k // split_k) + 1] = col_sign * factor_b
    bias = col_sign / 2 if has_bias else None
    ref = run_torch_tdm_fp32(a, weight, bias, dtype)
    # Mutation guard: same-dtype partial rounding MUST disagree with the
    # independent reference for these deliberately constructed inputs.
    # Three nonzero partitions avoid the power-of-two split8 midpoint trap.
    old_rounding = (row_sign[:, None] * col_sign[None, :] * 3).to(dtype)
    if bias is not None:
        old_rounding = (old_rounding.float() + bias.float()).to(dtype)
    assert torch.any(old_rounding != ref), "precision regression input lost sensitivity"

    # Exercise the original FT/HT backend alongside cluster TDM, using the
    # same valid tile/partition and independent reference. These names are
    # different backends, not additional TDM split-K reduction policies.
    candidates = {
        name: (
            lambda name=name: flydsl_hgemm(
                a, weight, out=out, bias=bias, **{**cfg, "policy": name}
            )
        )
        for name in ("tdm", "ft", "ht")
    }
    ret = {"gfx": get_gfx_runtime(), "split_k": split_k, "tile": f"16x16x{block_k}"}
    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n + (n if has_bias else 0)) * a.element_size()
    for name, fn in candidates.items():
        candidate_ref = run_torch_tdm_fp32(a, weight, bias, dtype, policy=name)
        out.fill_(float("nan"))
        y, us = run_perftest(fn)
        err = checkAllclose(
            candidate_ref.to(dtypes.fp32),
            y.to(dtypes.fp32),
            atol=0,
            rtol=0,
            tol_err_ratio=0,
            msg=f"{name} FP32 partials",
        )
        assert err == 0 and y.data_ptr() == out.data_ptr() and torch.isfinite(y).all()
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        for _ in range(6):
            a.neg_()
            if bias is not None:
                bias.neg_()
            expected = run_torch_tdm_fp32(a, weight, bias, dtype, policy=name)
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert (
                checkAllclose(
                    expected.float(),
                    out.float(),
                    atol=0,
                    rtol=0,
                    tol_err_ratio=0,
                    msg=f"{name} modified-input graph",
                )
                == 0
            )
        graph.reset()

    # Two captures originating on the same stream must own independent
    # scratch when replayed concurrently on two streams.
    capture_stream = torch.cuda.Stream()
    graphs, slots = [], []
    for idx in range(2):
        aa = a.clone()
        aa.mul_(-1 if idx else 1)
        bb = weight.clone()
        yy = torch.empty_like(out)
        bias_i = None if bias is None else bias.clone()
        expected = run_torch_tdm_fp32(aa, bb, bias_i, dtype)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=capture_stream):
            junk = torch.empty(4096, device="cuda")
            junk.fill_(999)
            del junk
            flydsl_hgemm(aa, bb, out=yy, bias=bias_i, stream=capture_stream, **cfg)
        graphs.append(gr)
        slots.append((aa, bb, yy, bias_i, expected))
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for _ in range(20):
        for st, gr in zip(streams, graphs):
            with torch.cuda.stream(st):
                gr.replay()
    torch.cuda.synchronize()
    for st, slot in zip(streams, slots):
        aa, bb, yy, bias_i, expected = slot
        assert (
            checkAllclose(
                expected.float(),
                yy.float(),
                atol=0,
                rtol=0,
                tol_err_ratio=0,
                msg="independent graph scratch",
            )
            == 0
        )
        flydsl_hgemm(aa, bb, out=yy, bias=bias_i, stream=st, **cfg)
    torch.cuda.synchronize()
    for _, _, yy, _, expected in slots:
        assert (
            checkAllclose(
                expected.float(),
                yy.float(),
                atol=0,
                rtol=0,
                tol_err_ratio=0,
                msg="explicit stream scratch",
            )
            == 0
        )
    for gr in graphs:
        gr.reset()
    for backing, cols in ((a_store, k), (w_store, k), (o_store, n)):
        _assert_guards(backing, cols)
    return ret


def check_tdm_contracts(dtype):
    from aiter.ops.flydsl import flydsl_gemm_a16w16_tdm as gemm

    a = torch.zeros(128, 768, device="cuda", dtype=dtype)
    w = torch.zeros(128, 768, device="cuda", dtype=dtype)
    out = torch.empty(128, 128, device="cuda", dtype=dtype)
    cfg = {
        "block_m": 128,
        "block_n": 128,
        "block_k": 128,
        "stages": 2,
        "m_waves": 2,
        "n_waves": 2,
        "split_k": 3,
    }
    calls = {
        "host input": lambda: gemm(a.cpu(), w.t(), out, **cfg),
        "overlap": lambda: gemm(a, w.t(), a[:, :128], **cfg),
        "bad bias": lambda: gemm(
            a, w.t(), out, torch.zeros(127, device="cuda", dtype=dtype), **cfg
        ),
        "invalid K": lambda: gemm(a[:, :767], w[:, :767].t(), out, **cfg),
        "invalid interleave type": lambda: gemm(a, w.t(), out, interleave="yes", **cfg),
    }
    for reason, call in calls.items():
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError(f"TDM accepted {reason}")
    # Fractional contributions survive FP32 partial storage. Old same-dtype
    # partials rounded each term down, changing the final answer.
    factor = 1 / 16 if dtype == dtypes.bf16 else 1 / 64
    for part in range(3):
        a[:, part * 256] = w[:, part * 256] = 1
        a[:, part * 256 + 1] = w[:, part * 256 + 1] = factor
    y = gemm(a, w.t(), out, **cfg)
    assert (
        checkAllclose(
            torch.full_like(y, 3 * (1 + factor * factor)).float(),
            y.float(),
            atol=0,
            rtol=0,
            tol_err_ratio=0,
            msg="FP32 workspace preserves partials",
        )
        == 0
    )
    a.zero_()
    w.zero_()
    a[:, 0] = w[:, 0] = 1
    a[:, 1] = w[:, 1] = factor
    bias = torch.full((128,), -1.0, device="cuda", dtype=dtype)
    unsplit = {**cfg, "split_k": 1}
    for interleave in (False, True):
        y = gemm(a, w.t(), out, bias, interleave=interleave, **unsplit)
        assert (
            checkAllclose(
                torch.zeros_like(y).float(),
                y.float(),
                atol=0,
                rtol=0,
                tol_err_ratio=0,
                msg="round before bias",
            )
            == 0
        )
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        normal = gemm(a, w.t(), **cfg)
        again = gemm(a, w.t(), **cfg)
        torch.testing.assert_close(again, normal, atol=0, rtol=0)
    finally:
        torch.use_deterministic_algorithms(deterministic)
    a.zero_()
    w.zero_()
    a[:, 0] = 1
    w[:, 0] = 1
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = [torch.empty_like(out), torch.empty_like(out)]
    for stream, target in zip(streams, outputs):
        stream.wait_stream(torch.cuda.current_stream())
        gemm(a, w.t(), target, stream=stream, **cfg)
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for target in outputs:
        assert (
            checkAllclose(
                torch.ones_like(target).float(),
                target.float(),
                atol=0,
                rtol=0,
                tol_err_ratio=0,
                msg="TDM two streams",
            )
            == 0
        )


def summarize(title, rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    aiter.logger.info("%s summary (markdown):\n%s", title, df.to_markdown(index=False))


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl hgemm unsupported on %s; skipping", get_gfx_runtime()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.bf16, dtypes.fp16],
        nargs="*",
        default="bf16,fp16," if get_gfx_runtime() == "gfx1250" else "bf16,",
        metavar="{bf16,fp16}",
        help="""Data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=(
            _TDM_MNK
            if get_gfx_runtime() == "gfx1250"
            else (
                _default_mnk()
                + _CONCURRENT_COPY_MNK
                + [row[:3] for row in _MEMORY_TILES]
                + _TUNED_MEMORY_MNK
                + _BAND_MNK
                + _PRESHUFFLE_MNK
            )
        ),
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
    parser.add_argument(
        "--layout", nargs="*", choices=["nt", "nn"], default=["nt", "nn"]
    )
    parser.add_argument(
        "--bias", nargs="*", type=dtypes.str2bool, default=[False, True]
    )
    parser.add_argument(
        "--interleave", nargs="*", type=dtypes.str2bool, default=[False, True]
    )
    parser.add_argument(
        "--split-k", nargs="*", type=int, choices=range(1, 10), default=[1, 3, 8]
    )
    parser.add_argument(
        "--stages", nargs="*", type=int, choices=range(2, 10), default=[2]
    )
    args = parser.parse_args()
    if get_gfx_runtime() == "gfx1250":
        for dtype in args.dtype:
            rows = []
            for (
                (m, n, k),
                bias,
                layout,
                split_k,
                stages,
                interleave,
            ) in itertools.product(
                args.mnk,
                args.bias,
                args.layout,
                args.split_k,
                args.stages,
                args.interleave,
            ):
                # NN has no interleaved schedule; exclude it before launch.
                if layout == "nn" and interleave:
                    aiter.logger.warning(
                        "Skipping NN interleave: TDM supports interleave only for NT"
                    )
                    continue
                if _tdm_config(m, n, k, split_k, stages) is None:
                    aiter.logger.warning(
                        "Skipping unsupported TDM K partition: %s",
                        (m, n, k, split_k, stages),
                    )
                    continue
                rows.append(
                    test_hgemm_tdm(
                        m, n, k, dtype, bias, layout, split_k, stages, interleave
                    )
                )
            assert rows, "no valid TDM cases selected"
            summarize(f"TDM HGEMM {dtype}", rows)
            dispatch_rows = [
                test_hgemm_tdm_tuned(m, n, k, dtype, bias, interleave)
                for (m, n, k), bias, interleave in itertools.product(
                    args.mnk, args.bias, args.interleave
                )
                if k % 384 == 0 and m <= 257 and n % 8 == 0 and n <= 512
            ]
            summarize(f"TDM tuned dispatch {dtype}", dispatch_rows)
            precision_rows = [
                test_hgemm_tdm_fp32(m, n, k, dtype, bias)
                for (m, n, k), bias in itertools.product(args.mnk, args.bias)
                if (m, n, k) in _TDM_FP32_MNK
            ]
            summarize(f"TDM FP32 partials {dtype}", precision_rows)
            check_tdm_contracts(dtype)
        return
    torch.set_default_device("cuda")
    wanted = set(args.mnk)
    if get_gfx_runtime() == "gfx950" and dtypes.bf16 in args.dtype:
        check_preshuffle_compiled_dispatch()
        summarize(
            "AITER BF16 preshuffle HGEMM",
            [
                test_hgemm_preshuffle(m, n, k, dtypes.bf16)
                for m, n, k in _PRESHUFFLE_MNK
                if (m, n, k) in wanted
            ],
        )

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

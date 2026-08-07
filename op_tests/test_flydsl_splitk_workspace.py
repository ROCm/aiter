# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL split-K HGEMM: workspace + reduce combine.

Each split writes an fp32 partial into its own slot of a `[slots, m, n]`
workspace and a separate reduce kernel sums them, so the combine carries no
cross-block coordination state.

Two independent things live here:

* **Correctness** (pytest): the result matches an fp32 torch reference, it is
  bit-reproducible run to run, and a CUDA-graph replay matches the eager
  result. `CASES` covers every structurally distinct split-K epilogue in the
  two kernel families -- generic hgemm (plain and slice-K) and small_m (plain,
  B_TO_LDS, wide-N repeat, persistent-N).
* **Performance** (`main()`): a `@benchmark` sweep over shape x split_k x
  tiling that reports us / TFLOPS / TB per s per config and prints a markdown
  summary table. That bandwidth figure deliberately counts the fp32 workspace
  round trip, which is this design's main cost (see `test_flydsl_splitk_gemm`).

Usage:
    python op_tests/test_flydsl_splitk_workspace.py     # perf sweep + table
    pytest -q op_tests/test_flydsl_splitk_workspace.py  # correctness only
"""

from __future__ import annotations

import argparse
import itertools
from typing import Any, NamedTuple

import pandas as pd
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.test_common import benchmark, checkAllclose, run_perftest

if not is_flydsl_available():
    pytest.skip("flydsl is not installed.", allow_module_level=True)

# Reuse the library's own tiling validators rather than restating their algebra
# here: a tiling that they reject would abort the launch, and the sweep must
# filter those out before launching (see `_supported`).
from aiter.ops.flydsl.gemm_kernels import (
    _get_split_k_workspace,
    _validate_hgemm_tiling,
    flydsl_hgemm,
    flydsl_splitk_prewarm_capture_workspace,
)
from aiter.ops.flydsl.kernels.small_m_hgemm import _validate_small_m_registry_config

# Archs the FlyDSL split-K HGEMM family is built and validated for.
SUPPORTED_GFX = ("gfx942", "gfx950")
# `compile_small_m_hgemm_kernel` rejects gfx942 (it targets the async-copy bf16
# path only), so the small_m cases are gfx950-only.
SMALL_M_GFX = ("gfx950",)

# Whole-module arch gate for pytest. `main()` gates separately with the same
# allow-list (the skill's rule: gate in main(), not by returning from inside the
# @benchmark fn, which would still emit an args-only row).
pytestmark = pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX,
    reason=f"FlyDSL split-K HGEMM unsupported on {get_gfx()}",
)

DEFAULT_SEED = 20260401
# Workspace state each replay starts from: a plain replay first, then values
# that would corrupt the result if the combine read anything it had not
# written itself in the same launch.
REPLAY_POISONS = (None, 1e30, float("nan"), 1e30)
# Fraction of elements allowed to fail isclose(atol=rtol=1e-2) vs the fp32 ref.
MAX_MISMATCH_RATIO = 0.001


class Case(NamedTuple):
    """One shape + tiling. `kernel` is passed straight to `flydsl_hgemm`."""

    name: str
    mnk: tuple[int, int, int]
    tiles: tuple[int, int, int]  # tile_m, tile_n, tile_k
    split_k: int
    overrides: dict[str, Any] = {}  # noqa: RUF012 - read-only

    @property
    def kernel(self) -> dict[str, Any]:
        # Warp defaults match the runner in test_flydsl_splitk_hgemm.py.
        tile_m, tile_n, tile_k = self.tiles
        return {
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "split_k": self.split_k,
            "block_m_warps": 1,
            "block_n_warps": 4,
            **self.overrides,
        }

    @property
    def is_small_m(self) -> bool:
        return self.overrides.get("kernel_family") == "small_m"


SMALL_M = {"kernel_family": "small_m"}

CASES = [
    Case("hgemm_m104_n384_k7168_spk8", (104, 384, 7168), (32, 64, 128), 8),
    Case("hgemm_m1_n7168_k512_spk2", (1, 7168, 512), (16, 128, 128), 2),
    # Slice-K: BLOCK_K_WARPS=2 gives every k-slice warp group its own workspace
    # slot, so the slice combine is fp32 as well.
    Case(
        "hgemm_m104_n384_k7168_spk8_slicek2",
        (104, 384, 7168),
        (32, 64, 128),
        8,
        {"block_n_warps": 2, "block_k_warps": 2},
    ),
    Case("small_m_m1_n7168_k512_spk2", (1, 7168, 512), (16, 128, 128), 2, SMALL_M),
    # small_m B_TO_LDS path (`run_b_to_lds_tile`), higher split_k.
    Case(
        "small_m_m1_n7168_k512_spk8_blds",
        (1, 7168, 512),
        (16, 128, 64),
        8,
        SMALL_M | {"b_to_lds": True},
    ),
    # small_m wide-N repeat path (N_TILE_REPEAT > 1, non-B_TO_LDS).
    Case(
        "small_m_m1_n7168_k512_spk4_nr2",
        (1, 7168, 512),
        (16, 64, 128),
        4,
        SMALL_M | {"block_n_warps": 1, "n_tile_repeat": 2},
    ),
    # small_m persistent-N path (PERSISTENT_N_TILES > 1, B_TO_LDS).
    Case(
        "small_m_m1_n7168_k512_spk4_pn4",
        (1, 7168, 512),
        (16, 128, 128),
        4,
        SMALL_M | {"block_n_warps": 2, "b_to_lds": True, "persistent_n_tiles": 4},
    ),
]


def skip_if_unsupported(case: Case) -> None:
    if case.is_small_m and get_gfx() not in SMALL_M_GFX:
        pytest.skip(f"small_m kernel unsupported on {get_gfx()}")


def make_inputs(case: Case, seed: int = DEFAULT_SEED):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    m, n, k = case.mnk
    kw = {"generator": gen, "device": "cuda", "dtype": torch.bfloat16}
    return torch.rand((m, k), **kw), torch.rand((n, k), **kw)


def run_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp32 reference. Compared against, never timed."""
    return a.float() @ b.float().t()


def run_case(case: Case, a: torch.Tensor, b: torch.Tensor, out=None) -> torch.Tensor:
    m, n, _ = case.mnk
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    flydsl_hgemm(a, b, out, **case.kernel)
    return out


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_workspace_matches_reference(case: Case):
    skip_if_unsupported(case)
    a, b = make_inputs(case)
    out = run_case(case, a, b)
    torch.cuda.synchronize()
    close = torch.isclose(out.float(), run_torch(a, b), atol=1e-2, rtol=1e-2)
    mismatch = int((~close).sum().item())
    assert (
        mismatch / close.numel() < MAX_MISMATCH_RATIO
    ), f"{case.name}: {mismatch}/{close.numel()} elements outside tolerance"


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_workspace_is_deterministic(case: Case):
    skip_if_unsupported(case)
    a, b = make_inputs(case)
    first = run_case(case, a, b).clone()
    torch.cuda.synchronize()
    second = run_case(case, a, b).clone()
    torch.cuda.synchronize()
    assert torch.equal(first, second), f"{case.name}: not bit-reproducible"


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_graph_replay_matches_eager(case: Case):
    """A captured replay must reproduce the eager result exactly, even after
    the workspace has been deliberately corrupted.

    The corruption is the point. A clean capture-and-replay passes on the old
    cross-block-counter combine too, so on its own it proves nothing; the
    counter only wedged the kernel once it was left dirty. Here the equivalent
    adversarial state is the workspace itself, so it is filled with a huge
    value and with NaN between replays. Both survive because the combine reads
    only slots it wrote in the same launch -- there is no carried state.
    """
    skip_if_unsupported(case)
    a, b = make_inputs(case)
    m, n, _ = case.mnk
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)

    # Eager pass: compiles the kernels and sizes the workspace.
    run_case(case, a, b, out)
    torch.cuda.synchronize()
    eager = out.clone()

    # The workspace cannot grow during capture, so size it on the stream
    # torch.cuda.graph() will capture on first.
    flydsl_splitk_prewarm_capture_workspace(
        m,
        n,
        split_k=case.split_k,
        block_k_warps=case.overrides.get("block_k_warps", 1),
        device=a.device,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_case(case, a, b, out)

    slots = case.split_k * case.overrides.get("block_k_warps", 1)
    workspace = _get_split_k_workspace(a.device, slots * m * n)
    for replay, poison in enumerate(REPLAY_POISONS):
        if poison is not None:
            workspace.fill_(poison)
        out.zero_()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), (
            f"{case.name}: replay {replay} diverged from eager "
            f"(workspace pre-filled with {poison})"
        )


# ---------------------------------------------------------------------------
# Performance sweep
# ---------------------------------------------------------------------------

# Tilings the sweep tries per (shape, split_k). Anything the library's
# validators reject for a given shape is filtered out in `main()` before launch.
SWEEP_TILINGS = [
    # tile_m, tile_n, tile_k, block_n_warps, block_k_warps
    (16, 128, 128, 4, 1),
    (32, 64, 128, 4, 1),
    (32, 64, 128, 2, 2),  # slice-K: one extra workspace slot per k-slice group
    (32, 128, 64, 4, 1),
    (64, 128, 128, 4, 1),
]

# STAGES is fixed at 2 by the kernel, and the pipeline needs at least that many
# K iterations per split (`assert BLOCK_K_LOOPS >= STAGES` in splitk_hgemm.py).
FIXED_STAGES = 2


def _supported(
    m, n, k, split_k, tile_m, tile_n, tile_k, block_n_warps, block_k_warps, family
):
    """True when this (shape, split_k, tiling) is a launchable config.

    Filters before launch instead of letting the kernel raise, so an
    unsupported combination leaves no row in the table rather than a failed one.
    """
    if family == "small_m":
        # small_m hard-wires tile_m=16 / block_m_warps=1 and rejects gfx942.
        if tile_m != 16 or block_k_warps != 1 or get_gfx() not in SMALL_M_GFX:
            return False
        try:
            _validate_small_m_registry_config(
                m,
                n,
                k,
                tile_n=tile_n,
                tile_k=tile_k,
                split_k=split_k,
                block_n_warps=block_n_warps,
                n_tile_repeat=1,
                persistent_n_tiles=1,
                waves_per_eu=0,
                b_to_lds_unroll=0,
                b_to_lds=False,
            )
        except ValueError:
            return False
        return (k // split_k) // tile_k >= FIXED_STAGES
    try:
        _validate_hgemm_tiling(
            m,
            n,
            k,
            dtype="bf16",
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            pack_n=1,
            split_k=split_k,
            stages=FIXED_STAGES,
            block_m_warps=1,
            block_n_warps=block_n_warps,
            block_k_warps=block_k_warps,
            b_to_lds=True,
        )
    except ValueError:
        return False
    return (k // split_k) // tile_k >= FIXED_STAGES


@benchmark()
def test_flydsl_splitk_gemm(
    m, n, k, split_k, family, tile_m, tile_n, tile_k, block_n_warps, block_k_warps
):
    kernel = {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "split_k": split_k,
        "block_m_warps": 1,
        "block_n_warps": block_n_warps,
        "block_k_warps": block_k_warps,
    }
    if family == "small_m":
        kernel["kernel_family"] = "small_m"

    gen = torch.Generator(device="cuda")
    gen.manual_seed(DEFAULT_SEED)
    a = torch.rand((m, k), generator=gen, device="cuda", dtype=dtypes.bf16)
    b = torch.rand((n, k), generator=gen, device="cuda", dtype=dtypes.bf16)
    out = torch.empty((m, n), dtype=dtypes.bf16, device="cuda")
    ref = run_torch(a, b)

    # A[m,k] @ B[n,k]^T -> C[m,n]:
    #   FLOPs = 2 * m * n * k (multiply-add).
    # Bytes must include the fp32 split-K workspace, because trading the
    # in-place atomic combine for a workspace is exactly what this design does:
    # every split writes its partial once and the reduce kernel reads it once,
    # so the workspace contributes 2x its size. Size it the way the kernel
    # actually allocates it: SLOTS = split_k * block_k_warps (each k-slice warp
    # group owns a slot) over the real m rows -- the layout is unpadded and the
    # stores are masked to row < m.
    slots = split_k * (block_k_warps if family != "small_m" else 1)
    ws_bytes = 2 * slots * m * n * 4 if split_k > 1 else 0
    flops = 2 * m * n * k
    nbytes = (m * k + n * k) * a.element_size() + m * n * out.element_size() + ws_bytes

    ret = {"gfx": get_gfx()}
    res, us = run_perftest(lambda: flydsl_hgemm(a, b, out, **kernel))
    err = checkAllclose(
        ref.to(dtypes.fp32),
        res.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg=f"{family} split-K hgemm m={m} n={n} k={k} split_k={split_k}",
    )
    ret["flydsl us"] = us
    ret["flydsl TFLOPS"] = flops / us / 1e6
    ret["flydsl TB/s"] = nbytes / us / 1e6
    ret["flydsl err"] = err
    return ret


# The skill names the swept function `test_*`, but it takes shape arguments and
# is driven by `main()`, not by pytest. Opt it out of collection so `pytest -q`
# on this file runs only the correctness cases above.
test_flydsl_splitk_gemm.__test__ = False


def main():
    # Positive allow-list: an unknown new card must not run an unbuilt kernel.
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "FlyDSL split-K HGEMM unsupported on %s; skipping", get_gfx()
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
        choices=[dtypes.d_dtypes["bf16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16}",
        help="""Data type. The FlyDSL split-K path is bf16-only.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            # The shapes the split-K path is actually selected for: skinny-M
            # decode GEMMs where K is long enough to be worth splitting.
            (1, 7168, 512),
            (1, 7168, 7168),
            (1, 3072, 1536),
            (16, 7168, 512),
            (104, 384, 7168),
            (128, 2112, 7168),
        ],
        help="""Shape (m, n, k).
        e.g.: -s 104,384,7168""",
    )
    parser.add_argument(
        "--split-k",
        type=int,
        nargs="*",
        default=[2, 4, 8],
        help="""Split-K factors to sweep.
        e.g.: --split-k 4 8""",
    )
    parser.add_argument(
        "--family",
        type=str,
        nargs="*",
        choices=["hgemm", "small_m"],
        default=["hgemm", "small_m"],
        help="""Kernel family.
        e.g.: --family hgemm""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        del dtype  # bf16-only; kept so the axis stays a swept list
        df = []
        for (m, n, k), split_k, family, tiling in itertools.product(
            args.mnk, args.split_k, args.family, SWEEP_TILINGS
        ):
            tile_m, tile_n, tile_k, block_n_warps, block_k_warps = tiling
            if not _supported(
                m,
                n,
                k,
                split_k,
                tile_m,
                tile_n,
                tile_k,
                block_n_warps,
                block_k_warps,
                family,
            ):
                continue
            df.append(
                test_flydsl_splitk_gemm(
                    m,
                    n,
                    k,
                    split_k,
                    family,
                    tile_m,
                    tile_n,
                    tile_k,
                    block_n_warps,
                    block_k_warps,
                )
            )
        if not df:
            aiter.logger.warning("no supported split-K configs for this sweep")
            continue
        df = pd.DataFrame(df)
        try:
            table = df.to_markdown(index=False)
        except ImportError:
            # to_markdown needs the optional `tabulate` package; plain fallback,
            # mirroring op_tests/test_flydsl_grouped_gemm_gfx1250.py.
            table = df.to_string(index=False)
        aiter.logger.info("flydsl split-K hgemm summary (markdown):\n%s", table)


if __name__ == "__main__":
    main()

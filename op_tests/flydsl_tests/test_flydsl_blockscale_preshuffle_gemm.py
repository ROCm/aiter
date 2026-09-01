# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the FlyDSL a8w8 blockscale bpreshuffle GEMM.

Covers the three things that can break independently: the kernel's numerics, the
tuned-CSV kernelName round trip, and the gemm_a8w8_blockscale_bpreshuffle
dispatch that joins them.

Usage:
    python op_tests/flydsl_tests/test_flydsl_blockscale_preshuffle_gemm.py
    pytest -q op_tests/flydsl_tests/test_flydsl_blockscale_preshuffle_gemm.py
"""

from __future__ import annotations

import math

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.shuffle import shuffle_weight

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)

# CDNA only: gfx95x the kernel emits v_mfma_f32_16x16x32_fp8_fp8
_GFX = get_gfx()
if not _GFX.startswith(("gfx942", "gfx95")):
    pytest.skip(
        f"blockscale bpreshuffle GEMM needs gfx942/gfx95x, got {_GFX}",
        allow_module_level=True,
    )
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL block-scale GEMM tests.",
        allow_module_level=True,
    )

try:
    from aiter import dtypes
    from aiter.ops.flydsl.gemm_kernels import (
        _compile_flydsl_blockscale,
        flydsl_gemm_a8w8_blockscale_bpreshuffle,
        select_blockscale_tile_config,
    )
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_bpreshuffle_common import (
        FLAG_CANDIDATES,
        TILE_CANDIDATES,
        WAVE_CANDIDATES,
        kernelInstance,
        kernels_list,
        parse_kernel_name,
        tile_is_valid,
    )
    from aiter.ops.flydsl.kernels.gemm_blockscale_preshuffle import (
        compile_blockscale_preshuffle_gemm,
    )
except ImportError as exc:
    pytest.skip(
        f"Unable to import FlyDSL block-scale GEMM kernels: {exc}",
        allow_module_level=True,
    )

torch.set_default_device("cuda")

# fp8 quantization on both operands dominates the error budget; measured rel-norm
# against the fp32 reference sits at ~1.5e-3 across every shape below. The gate is
# an order of magnitude looser so it fails on a broken kernel, not on rounding.
DEFAULT_REL_TOL = 2e-2
# Per-fragment gate; clean runs sit at ~2e-3. Tight on purpose: an indexing regression
# permutes rather than destroys, so it barely moves any norm; a swapped n-fragment
# scores 1.7e-2, and a looser gate would miss it.
DEFAULT_TILE_REL_TOL = 1e-2
DEFAULT_INPUT_SEED = 20260401

SCALE_BLOCK_N = 128
SCALE_BLOCK_K = 128

# M is deliberately ragged. The kernel has three separate M-dependent paths (the
# masked tail (m % tile_m), the runtime-M x_scale stride, and the XCD swizzle, whose
# un-rotation is only a permutation when the ragged-group correction is right), and
# an all-powers-of-two table exercises none of them.
PRECISION_CASES = [
    {"name": "m1_n512_k512_auto", "m": 1, "n": 512, "k": 512},
    {"name": "m33_n512_k512_auto", "m": 33, "n": 512, "k": 512},
    {"name": "m100_n512_k512_auto", "m": 100, "n": 512, "k": 512},
    {"name": "m512_n512_k512_auto", "m": 512, "n": 512, "k": 512},
    {
        "name": "m17_n512_k512_t16x64x256",
        "m": 17,
        "n": 512,
        "k": 512,
        "tile": (16, 64, 256),
    },
    {
        "name": "m100_n512_k512_t32x64x128",
        "m": 100,
        "n": 512,
        "k": 512,
        "tile": (32, 64, 128),
    },
    {
        "name": "m1024_n512_k512_t64x256x128",
        "m": 1024,
        "n": 512,
        "k": 512,
        "tile": (64, 256, 128),
    },
    {
        "name": "m64_n512_k512_fp16",
        "m": 64,
        "n": 512,
        "k": 512,
        "out_dtype": torch.float16,
    },
    # Multi-tile grids. The cases above all land on num_wg % 8 == 0 with an even k-tile
    # count, leaving the XCD un-rotation's remainder term and the k-loop's odd branch
    # untested. m % tile_m != 0 also exercises the M tail.
    {"name": "m513_n768_k1152_ragged", "m": 513, "n": 768, "k": 1152},
    {"name": "m1025_n1280_k640_wide_n", "m": 1025, "n": 1280, "k": 640},
    {"name": "m2048_n1024_k1152_aligned", "m": 2048, "n": 1024, "k": 1152},
]


def make_inputs(m: int, n: int, k: int, *, seed: int = DEFAULT_INPUT_SEED):
    """Quantized operands plus the layouts the preshuffle contract requires.

    Returns (x, w, x_scale, w_scale, w_shuffled, x_scale_k_major). The last two
    are what the kernel is called with; the first four are what the reference uses.
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    scale_n = (n + SCALE_BLOCK_N - 1) // SCALE_BLOCK_N
    scale_k = (k + SCALE_BLOCK_K - 1) // SCALE_BLOCK_K
    # /10 keeps the products inside fp8e4m3's range so the reference is comparing
    # against the same values the kernel saw, not against saturated ones.
    x = (torch.rand((m, k), generator=gen, device="cuda", dtype=torch.float32) / 10).to(
        dtypes.fp8
    )
    w = (torch.rand((n, k), generator=gen, device="cuda", dtype=torch.float32) / 10).to(
        dtypes.fp8
    )
    x_scale = torch.rand(
        (m, scale_k), generator=gen, device="cuda", dtype=torch.float32
    )
    w_scale = torch.rand(
        (scale_n, scale_k), generator=gen, device="cuda", dtype=torch.float32
    )
    w_shuffled = shuffle_weight(w, layout=(16, 16))
    x_scale_km = x_scale.transpose(0, 1).contiguous().view(m, scale_k)
    return x, w, x_scale, w_scale, w_shuffled, x_scale_km


def run_torch(x, w, x_scale, w_scale) -> torch.Tensor:
    """fp32 reference: dequantize both operands, then a plain matmul."""
    m, k = x.shape
    n = w.shape[0]
    scale_k = x_scale.shape[1]
    xd = (
        x.float().view(m, scale_k, SCALE_BLOCK_K) * x_scale.float().unsqueeze(-1)
    ).view(m, k)
    scale_n = w_scale.shape[0]
    wd = (
        w.float().view(scale_n, SCALE_BLOCK_N, scale_k, SCALE_BLOCK_K)
        * w_scale.float()[:, None, :, None]
    ).view(n, k)
    return xd @ wd.T


def rel_norm(ref: torch.Tensor, out: torch.Tensor) -> float:
    """Relative Frobenius error. Elementwise tolerances are the wrong instrument
    for a K=5120 reduction: a handful of near-cancelling rows fail any fixed atol
    while the result is fine, and the norm ratio is what actually degrades when the
    kernel is wrong."""
    ref_f = ref.float()
    return ((out.float() - ref_f).norm() / ref_f.norm().clamp_min(1e-30)).item()


def max_tile_rel(ref: torch.Tensor, out: torch.Tensor, tile: int = 16) -> float:
    """Worst 16x16 output fragment, by relative Frobenius norm.

    16x16 is the MFMA output tile, the granularity at which the swizzle and preshuffle
    coordinate math can go wrong. Such a fault is localized, so the whole-matrix ratio
    dilutes it away at large N. The per-tile floor keeps a near-cancelling tile from
    inflating the metric the way a fixed atol would.
    """
    r, o = ref.float(), out.float()
    m, n = r.shape
    pad = (0, (-n) % tile, 0, (-m) % tile)
    r = torch.nn.functional.pad(r, pad)
    o = torch.nn.functional.pad(o, pad)
    mm, nn = r.shape

    def to_tiles(t):
        return (
            t.view(mm // tile, tile, nn // tile, tile)
            .permute(0, 2, 1, 3)
            .reshape(-1, tile * tile)
        )

    rt, ot = to_tiles(r), to_tiles(o)
    rn = rt.norm(dim=1)
    real = rn > 0  # padding tiles are all-zero in both and carry no signal
    if not bool(real.any()):
        return 0.0
    floor = rn[real].mean() * 0.05
    denom = rn[real].clamp_min(floor).clamp_min(1e-30)
    return ((ot - rt).norm(dim=1)[real] / denom).max().item()


def run_precision_case(case: dict, *, rel_tol: float = DEFAULT_REL_TOL):
    m, n, k = case["m"], case["n"], case["k"]
    tile = case.get("tile") or (0, 0, 0)
    out_dtype = case.get("out_dtype", torch.bfloat16)
    print("=" * 80)
    print(
        f"[flydsl] blockscale bpreshuffle case={case['name']} "
        f"shape=({m}, {n}, {k}) tile={tile if any(tile) else 'auto'} "
        f"out={out_dtype}"
    )

    x, w, x_scale, w_scale, w_shuf, x_scale_km = make_inputs(m, n, k)
    ref = run_torch(x, w, x_scale, w_scale)

    out = torch.zeros((m, n), dtype=out_dtype, device="cuda")
    flydsl_gemm_a8w8_blockscale_bpreshuffle(
        x, w_shuf, x_scale_km, w_scale, out, *tile, SCALE_BLOCK_K
    )
    torch.cuda.synchronize()

    rel = rel_norm(ref, out)
    tile_rel = max_tile_rel(ref, out)
    passed = rel <= rel_tol and tile_rel <= DEFAULT_TILE_REL_TOL
    print(
        f"  rel={rel:.3e} (tol={rel_tol:.1e})  "
        f"max_tile_rel={tile_rel:.3e} (tol={DEFAULT_TILE_REL_TOL:.1e})"
        f"  --> {'PASS' if passed else 'FAIL'}"
    )
    return passed, rel, tile_rel


@pytest.mark.parametrize(
    "case", [pytest.param(c, id=c["name"]) for c in PRECISION_CASES]
)
def test_flydsl_blockscale_precision(case: dict):
    passed, rel, tile_rel = run_precision_case(case)
    assert passed, (
        f"{case['name']}: rel={rel:.3e} (tol {DEFAULT_REL_TOL:.1e}), "
        f"max_tile_rel={tile_rel:.3e} (tol {DEFAULT_TILE_REL_TOL:.1e})"
    )


def test_one_compile_serves_every_m():
    """One compile must serve every M.

    M changes from call to call at serving time. If it were baked into codegen, each
    new M would JIT on the critical path, so this asserts the compile cache takes
    exactly one miss across four different M values.
    """
    n, k, tile = 512, 512, (64, 256, 128)
    _compile_flydsl_blockscale.cache_clear()
    for m in (64, 100, 1024, 2049):
        x, _, _, w_scale, w_shuf, x_scale_km = make_inputs(m, n, k)
        out = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
        flydsl_gemm_a8w8_blockscale_bpreshuffle(
            x, w_shuf, x_scale_km, w_scale, out, *tile, SCALE_BLOCK_K
        )
    torch.cuda.synchronize()
    info = _compile_flydsl_blockscale.cache_info()
    print(f"  compile cache: {info}")
    assert info.misses == 1, f"expected a single compile across 4 M values, got {info}"


def test_invalid_tile_raises_runtime_error():
    """A tile the kernel cannot serve must raise RuntimeError, not abort.

    The RuntimeError propagates: the libtype=="flydsl" dispatch does not catch it, so
    a tuned CSV row naming an over-large or misaligned tile takes the GEMM call down.
    That is the intended contract. What this guards against is the alternative: the
    backend aborting the whole process on "local memory (N) exceeds limit", which no
    caller can catch or report.
    """
    m, n, k = 64, 512, 512
    x, _, _, w_scale, w_shuf, x_scale_km = make_inputs(m, n, k)
    out = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError):
        # n % tile_n != 0
        flydsl_gemm_a8w8_blockscale_bpreshuffle(
            x, w_shuf, x_scale_km, w_scale, out, 64, 384, 128, SCALE_BLOCK_K
        )


def test_unsupported_out_dtype_raises_runtime_error():
    m, n, k = 64, 512, 512
    x, _, _, w_scale, w_shuf, x_scale_km = make_inputs(m, n, k)
    out = torch.zeros((m, n), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        flydsl_gemm_a8w8_blockscale_bpreshuffle(
            x, w_shuf, x_scale_km, w_scale, out, 0, 0, 0, SCALE_BLOCK_K
        )


@pytest.mark.parametrize("kernel_id", sorted(kernels_list))
def test_kernel_name_round_trip(kernel_id: int):
    """The tuner writes kernelName; the dispatch parses it back. A mismatch
    silently degrades every tuned row to the heuristic tile, which looks like a
    performance regression with no error anywhere, so pin the round trip."""
    ki = kernels_list[kernel_id]
    parsed = parse_kernel_name(ki.name)
    assert parsed == ki, f"{ki.name!r} parsed back as {parsed}"


def test_kernel_name_rejects_foreign_names():
    """Names from the CK and rowwise-FlyDSL families must not parse as this one."""
    for name in (
        "",
        "a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_intrawave_v1",
        "flydsl_a8w8_bpreshuflle_64x128x128_F8_F8_B16_ls2_ce0_ac1_wpe0_default",
        "flydsl_blockscale_bpreshuffle_64x256x128_F8_F8_B16_default",  # no sbk field
    ):
        assert parse_kernel_name(name) is None, f"{name!r} should not parse"


def test_tile_candidates_are_append_only():
    """kernelId in a tuned CSV row indexes into TILE_CANDIDATES. Reordering or
    removing an entry re-points every already-written row at a different tile, so
    the first entries are pinned here as a tripwire."""
    assert TILE_CANDIDATES[:3] == ((16, 64, 256), (16, 128, 256), (32, 64, 128))
    assert TILE_CANDIDATES[10] == (64, 256, 128)
    assert len(kernels_list) == len(TILE_CANDIDATES) * len(WAVE_CANDIDATES) * len(
        FLAG_CANDIDATES
    )
    # The wave and flag axes are appended, never interleaved, and the all-default
    # blocks sort first: ids [0, len(TILE_CANDIDATES)) stay the 4-wave no-flag tiles
    # in TILE_CANDIDATES order, so a row written by any earlier sweep still resolves
    # to the kernel it measured.
    assert WAVE_CANDIDATES[0] == 4
    assert FLAG_CANDIDATES[0] == (False, False)
    for f_idx, (ac, cs) in enumerate(FLAG_CANDIDATES):
        for w_idx, num_waves in enumerate(WAVE_CANDIDATES):
            for i, tile in enumerate(TILE_CANDIDATES):
                idx = (f_idx * len(WAVE_CANDIDATES) + w_idx) * len(TILE_CANDIDATES) + i
                ki = kernels_list[idx]
                assert (ki.tile_m, ki.tile_n, ki.tile_k) == tile
                assert ki.num_waves == num_waves
                assert ki.use_async_copy is ac
                assert ki.use_cshuffle_epilog is cs


def test_kernel_name_wave_field_is_backward_compatible():
    """A name written before the wave axis existed has no _w field. It must still
    parse, as the 4 waves it was measured at; otherwise every already-tuned row
    silently re-points at a different kernel."""
    legacy = "flydsl_blockscale_bpreshuffle_64x256x128_F8_F8_B16_sbk128_default"
    ki = parse_kernel_name(legacy)
    assert ki is not None and ki.num_waves == 4
    assert (ki.tile_m, ki.tile_n, ki.tile_k) == (64, 256, 128)
    assert ki.use_async_copy is False and ki.use_cshuffle_epilog is False
    # a name from the intermediate sweep, carrying only some of the fields
    partial = parse_kernel_name(legacy + "_w8")
    assert partial.num_waves == 8
    assert partial.use_async_copy is False and partial.use_cshuffle_epilog is False
    # and every generated name round-trips, both widths
    for k in kernels_list.values():
        assert parse_kernel_name(k.name) == k


def test_heuristic_tile_is_always_valid():
    """Whatever the heuristic picks must be something the kernel accepts; a pick the
    validator then rejects raises RuntimeError out of the dispatch, uncaught.

    Swept rather than spot-checked: this is CPU-only, so a grid costs nothing.
    """
    for n in (512, 768, 1024, 1280, 2048, 4352):
        for k in (512, 640, 1152, 2176, 4096, 5120):
            for m in (1, 33, 512, 8192, 32768):
                tile = select_blockscale_tile_config(m, n, k, SCALE_BLOCK_K)
                assert tile_is_valid(
                    *tile, n, k, SCALE_BLOCK_K
                ), f"heuristic picked invalid tile {tile} for M={m} N={n} K={k}"


def test_dispatch_matches_direct_call():
    """The same numbers must come out of gemm_a8w8_blockscale_bpreshuffle as out of
    the op called directly, i.e. the tuned-row plumbing (libtype, kernelName, the
    scale/weight layouts the wrapper assumes) does not reshape anything on the way."""
    import aiter

    m, n, k, tile = 512, 512, 512, (64, 256, 128)
    x, _, _, w_scale, w_shuf, x_scale_km = make_inputs(m, n, k)

    direct = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
    flydsl_gemm_a8w8_blockscale_bpreshuffle(
        x, w_shuf, x_scale_km, w_scale, direct, *tile, SCALE_BLOCK_K
    )

    ki = kernelInstance(*tile, scale_block_k=SCALE_BLOCK_K)
    config = {"libtype": "flydsl", "kernelName": ki.name, "splitK": 0}
    via_dispatch = aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale_flydsl(
        x,
        w_shuf,
        x_scale_km,
        w_scale,
        torch.zeros((m, n), dtype=torch.bfloat16, device="cuda"),
        config,
    )
    torch.cuda.synchronize()
    assert torch.equal(direct, via_dispatch), "dispatch and direct call disagree"


@pytest.mark.parametrize("strided_x_scale", [False, True])
def test_real_dispatch_branch(monkeypatch, strided_x_scale: bool):
    """Drive aiter.gemm_a8w8_blockscale_bpreshuffle itself, not the wrapper below it.

    test_dispatch_matches_direct_call stops one frame short: it calls
    gemm_a8w8_blockscale_flydsl, so the tuned-row lookup and the
    kernelName.startswith("flydsl_blockscale_bpreshuffle_") branch that selects it
    never run. Injecting the config exercises both.

    Both x_scale layouts the op accepts are covered. The strided one is the shape a
    bare .contiguous() re-materialises M-major, which the kernel then reads
    transposed, which is silently wrong rather than an error.
    """
    import aiter
    import aiter.ops.gemm_op_a8w8 as gemm_op

    m, n, k, tile = 512, 512, 512, (64, 256, 128)
    x, _, x_scale, w_scale, w_shuf, x_scale_km = make_inputs(m, n, k)

    ki = kernelInstance(*tile, scale_block_k=SCALE_BLOCK_K)
    cfg = {"libtype": "flydsl", "kernelName": ki.name, "splitK": 0}
    monkeypatch.setattr(gemm_op, "get_CKGEMM_config", lambda *a, **kw: cfg)

    scale_k = k // SCALE_BLOCK_K
    if strided_x_scale:
        xs = x_scale.transpose(0, 1).contiguous().transpose(0, 1)
        assert xs.stride(0) == 1 and not xs.is_contiguous()
    else:
        xs = x_scale_km
        assert xs.is_contiguous()
    assert xs.shape == (m, scale_k)

    got = aiter.gemm_a8w8_blockscale_bpreshuffle(x, w_shuf, xs, w_scale)

    expect = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
    flydsl_gemm_a8w8_blockscale_bpreshuffle(
        x, w_shuf, x_scale_km, w_scale, expect, *tile, SCALE_BLOCK_K
    )
    torch.cuda.synchronize()
    assert torch.equal(got, expect), (
        f"real dispatch disagrees with the direct call "
        f"(strided_x_scale={strided_x_scale})"
    )


@pytest.mark.parametrize(
    "knob",
    [
        pytest.param({}, id="defaults"),
        pytest.param({"num_waves": 8}, id="num_waves8"),
        pytest.param({"use_cshuffle_epilog": True}, id="cshuffle"),
        pytest.param({"use_async_copy": True}, id="async_copy"),
        # The staged A-scale path; nothing else in this file compiles it.
        pytest.param(
            {"use_async_copy": True, "stage_a_scales": True}, id="async_copy_staged"
        ),
    ],
)
def test_optional_kernel_knobs(knob: dict):
    """num_waves, use_cshuffle_epilog and use_async_copy have no production caller yet.

    They are compiled and checked here anyway: an untested knob that ships is one that
    silently rots until the first caller finds it broken.
    """
    import flydsl.expr as fx

    # Ragged M and an odd k-tile count: the cshuffle epilogue and the async-copy path
    # rewrite the output and A-staging paths, so exercise the tail there too.
    m, n, k = 513, 1024, 1152
    x, w, x_scale, w_scale, w_shuf, x_scale_km = make_inputs(m, n, k)
    ref = run_torch(x, w, x_scale, w_scale)

    exe = compile_blockscale_preshuffle_gemm(
        N=n, K=k, tile_m=64, tile_n=256, tile_k=128, **knob
    )
    out = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
    exe(
        out,
        x,
        w_shuf,
        x_scale_km.reshape(-1),
        w_scale.reshape(-1),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    torch.cuda.synchronize()
    rel, tile_rel = rel_norm(ref, out), max_tile_rel(ref, out)
    assert (
        rel <= DEFAULT_REL_TOL and tile_rel <= DEFAULT_TILE_REL_TOL
    ), f"{knob} -> rel={rel:.3e}, max_tile_rel={tile_rel:.3e}"


def _run_with_monkeypatch(fn, *args):
    """Run a monkeypatch-taking test outside pytest, for the __main__ runner."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        return fn(mp, *args)
    finally:
        mp.undo()


def main() -> int:
    results: list[tuple[str, str, float]] = []
    for case in PRECISION_CASES:
        try:
            passed, rel, tile_rel = run_precision_case(case)
            results.append(
                (case["name"], "PASS" if passed else "FAIL", max(rel, tile_rel))
            )
        except Exception:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            results.append((case["name"], "ERROR", float("nan")))

    checks = [
        ("one_compile_serves_every_m", test_one_compile_serves_every_m),
        ("invalid_tile_raises", test_invalid_tile_raises_runtime_error),
        (
            "unsupported_out_dtype_raises",
            test_unsupported_out_dtype_raises_runtime_error,
        ),
        (
            "kernel_name_round_trip",
            lambda: [test_kernel_name_round_trip(i) for i in kernels_list],
        ),
        ("kernel_name_rejects_foreign", test_kernel_name_rejects_foreign_names),
        ("tile_candidates_append_only", test_tile_candidates_are_append_only),
        ("heuristic_tile_valid", test_heuristic_tile_is_always_valid),
        ("dispatch_matches_direct", test_dispatch_matches_direct_call),
        # test_real_dispatch_branch is the only check that drives
        # aiter.gemm_a8w8_blockscale_bpreshuffle through the new dispatch branch, so it
        # must not be pytest-only; _MonkeyPatch is what pytest hands its fixture.
        (
            "real_dispatch_branch",
            lambda: [
                _run_with_monkeypatch(test_real_dispatch_branch, strided)
                for strided in (False, True)
            ],
        ),
        (
            "optional_kernel_knobs",
            lambda: [
                test_optional_kernel_knobs(knob)
                for knob in (
                    {},
                    {"num_waves": 8},
                    {"use_cshuffle_epilog": True},
                    {"use_async_copy": True},
                    {"use_async_copy": True, "stage_a_scales": True},
                )
            ],
        ),
    ]
    for name, fn in checks:
        try:
            fn()
            results.append((name, "PASS", float("nan")))
        except Exception:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            results.append((name, "FAIL", float("nan")))

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for name, status, rel in results:
        rel_s = "" if math.isnan(rel) else f"  rel={rel:.3e}"
        print(f"  {status:>5s}  {name:<38s}{rel_s}")
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n  {n_pass}/{len(results)} passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

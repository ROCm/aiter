# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""4-wave MXFP8 backend (aiter.ops.flydsl.gemm_a8w8_mxfp8_4wave): kernelName round trip,
launch plan, scale shuffle, and the dispatcher entry ``run_gemm_a8w8_mxfp8_gfx950`` against
the 8-wave backend (bit-identical: same MFMA, same per-accumulator K order)."""

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.gemm_a8w8_mxfp8_4wave import (
    launch_plan,
    make_mxfp8_4wave_kernel_name,
    parse_mxfp8_4wave_kernel_name,
)

# The running GPU, not the build target (get_gfx() reads GPU_ARCHS): ask only with a GPU present.
_GFX950 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="needs a gfx950 GPU",
)
_8W = "flydsl_mxfp8_8w_256x256x128_F8_F8_B16_2x0"


@pytest.mark.parametrize("xcd", [True, False])
@pytest.mark.parametrize("nt", [True, False])
def test_kernel_name_round_trip(xcd, nt):
    name = make_mxfp8_4wave_kernel_name(xcd, nt)
    cfg = parse_mxfp8_4wave_kernel_name(name)
    assert cfg["xcd_remap"] is xcd and cfg["store_nt"] is nt
    assert (cfg["block_m"], cfg["block_n"], cfg["block_k"]) == (256, 256, 128)
    assert parse_mxfp8_4wave_kernel_name(name + "_x") is None
    assert (
        parse_mxfp8_4wave_kernel_name("flydsl_mxfp8_8w_256x256x128_F8_F8_B16_1x8")
        is None
    )


def test_launch_plan():
    # no remainder round / a full round: one launch over the parent grid
    assert launch_plan(2048, 6144, 256, True) == [((4, 4), 0, 192)]
    assert launch_plan(5120, 6144, 256, True) == [
        ((4, 4), 0, 480)
    ]  # 480 % 256 = 224 > 128
    # 288 parents: 32 left over -> 128x128 sub-tiles (4 per parent, parents dealt 8 at a time)
    assert launch_plan(3072, 6144, 256, True) == [((4, 4), 0, 256), ((2, 2), 256, 128)]
    # 384 parents: 128 left over -> 128x256 sub-tiles (2 per parent)
    assert launch_plan(4096, 6144, 256, True) == [((4, 4), 0, 256), ((2, 4), 256, 256)]
    assert launch_plan(3072, 6144, 256, False) == [((4, 4), 0, 256), ((2, 2), 256, 128)]


def _scale_case(m, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(
        100, 140, (m, k // 32), dtype=torch.uint8, device="cuda", generator=g
    )


@_GFX950
@pytest.mark.parametrize("m", [1, 31, 32, 1000, 4100, 8192])
@pytest.mark.parametrize("k", [512, 8192])
def test_scale_shuffle_matches_a16w4(m, k):
    from aiter.ops.flydsl.gemm_a8w8_mxfp8_4wave import _shuffled_scale
    from aiter.ops.shuffle import shuffle_scale_a16w4

    s = _scale_case(m, k)
    pad = -(-m // 32) * 32
    p = torch.full((pad, k // 32), 127, dtype=torch.uint8, device="cuda")
    p[:m] = s
    ref = (
        shuffle_scale_a16w4(p.view(dtypes.fp8_e8m0), 1, False)
        .view(torch.uint8)
        .view(-1)
    )
    assert torch.equal(_shuffled_scale(s.view(dtypes.fp8_e8m0)), ref)


def _nan_out(m, n):
    """A bf16 output filled with NaN: an element the kernel never stores fails the comparison."""
    return torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")


def _operands(m, n, k, seed=0):
    from aiter.ops.quant import per_group_quant_hip
    from aiter.ops.shuffle import shuffle_weight

    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.05
    kw = {"quant_dtype": dtypes.fp8, "group_size": 32, "scale_type": dtypes.fp8_e8m0}
    xq, xs = per_group_quant_hip(x, **kw)
    wq, ws = per_group_quant_hip(w, **kw)
    return xq, xs, shuffle_weight(wq, layout=(16, 16)), ws


@_GFX950
@pytest.mark.parametrize(
    "m,n,k",
    [
        (300, 512, 1024),
        (300, 512, 512),  # K == 512
        (2048, 6144, 768),  # K % 512 == 256
        (3072, 6144, 8192),  # 288 parents: a 128x128 sub-tile remainder round
        (4096, 6144, 8192),  # 384 parents: a 128x256 sub-tile remainder round
        (4100, 6144, 8192),
        (5120, 2560, 6144),
    ],
)
@pytest.mark.parametrize("xcd", [True, False])
@pytest.mark.parametrize("nt", [False, True])
def test_dispatch_bit_identical_to_8wave(m, n, k, xcd, nt):
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import run_gemm_a8w8_mxfp8_gfx950

    xq, xs, wsh, ws = _operands(m, n, k)
    ref = _nan_out(m, n)
    run_gemm_a8w8_mxfp8_gfx950(xq, wsh, xs, ws, ref, _8W)
    got = _nan_out(m, n)
    run_gemm_a8w8_mxfp8_gfx950(
        xq, wsh, xs, ws, got, make_mxfp8_4wave_kernel_name(xcd, nt)
    )
    assert torch.equal(got, ref)


@_GFX950
def test_public_op_runs_an_injected_4wave_row(monkeypatch):
    """``gemm_a8w8_blockscale_bpreshuffle`` runs the 4-wave kernel a (monkeypatched) tuned row names, and
    its output equals the 8-wave's. The row is injected: the test does not depend on the shipped CSVs.
    """
    import aiter.ops.flydsl.gemm_a8w8_mxfp8_4wave as backend
    import aiter.ops.gemm_op_a8w8 as op

    m, n, k = 3072, 6144, 8192
    xq, xs, wsh, ws = _operands(m, n, k)
    name = make_mxfp8_4wave_kernel_name(True)
    row = {"libtype": "flydsl", "kernelId": "mxfp8", "kernelName": name}
    monkeypatch.setattr(op, "get_mxscale_bpreshuffle_config", lambda *a: dict(row))
    calls = []
    orig = backend.run_gemm_a8w8_mxfp8_4wave
    monkeypatch.setattr(
        backend,
        "run_gemm_a8w8_mxfp8_4wave",
        lambda *a, **kw: calls.append(a[5]) or orig(*a, **kw),
    )
    got = op.gemm_a8w8_blockscale_bpreshuffle(xq, wsh, xs, ws, dtypes.bf16)
    assert calls == [name], "the op did not run the injected 4-wave row"
    row["kernelName"] = _8W  # the lookup now names the 8-wave row
    ref = op.gemm_a8w8_blockscale_bpreshuffle(xq, wsh, xs, ws, dtypes.bf16)
    assert torch.equal(got, ref)


@_GFX950
def test_a_row_the_kernel_does_not_take_runs_the_heuristic(monkeypatch):
    """A padded-M lookup can hand the 4-wave row an M it does not take (``fits_shape`` false, e.g. 32-bit
    buffer sizes): the call runs the generic heuristic tile instead of raising."""
    import aiter.ops.flydsl.gemm_a8w8_mxfp8_4wave as backend
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import run_gemm_a8w8_mxfp8_gfx950
    from aiter.test_common import checkAllclose

    m, n, k = 300, 512, 1024
    xq, xs, wsh, ws = _operands(m, n, k)
    ref = _nan_out(m, n)
    run_gemm_a8w8_mxfp8_gfx950(xq, wsh, xs, ws, ref, _8W)
    seen = []
    monkeypatch.setattr(backend, "fits_shape", lambda *a: False)
    monkeypatch.setattr(
        backend, "run_gemm_a8w8_mxfp8_4wave", lambda *a, **kw: seen.append(1)
    )
    got = _nan_out(m, n)
    run_gemm_a8w8_mxfp8_gfx950(xq, wsh, xs, ws, got, make_mxfp8_4wave_kernel_name(True))
    assert not seen
    assert checkAllclose(ref.float(), got.float(), rtol=1e-2, atol=5e-2) == 0


@_GFX950
def test_launch_rejects_malformed_operands():
    """The producer-facing entry validates what the raw-scale wrapper does, and the scale lengths."""
    from aiter.ops.flydsl.gemm_a8w8_mxfp8_4wave import (
        _shuffled_scale,
        launch_mxfp8_4wave_shuffled,
    )

    m, n, k = 256, 512, 1024
    xq, xs, wsh, ws = _operands(m, n, k)
    xs_s, ws_s = _shuffled_scale(xs), _shuffled_scale(ws)
    out = _nan_out(m, n)
    launch_mxfp8_4wave_shuffled(xq, wsh, xs_s, ws_s, out)
    assert not out.isnan().any()
    with pytest.raises(ValueError):  # N = 384 is no whole number of 256-column tiles
        launch_mxfp8_4wave_shuffled(
            xq, wsh[:384], xs_s, ws_s[: 384 * (k // 32)], _nan_out(m, 384)
        )
    with pytest.raises(ValueError):  # a shuffled x_scale shorter than the padded rows
        launch_mxfp8_4wave_shuffled(xq, wsh, xs_s[:-1], ws_s, out)
    with pytest.raises(ValueError):  # not bf16
        launch_mxfp8_4wave_shuffled(xq, wsh, xs_s, ws_s, out.float())


@_GFX950
def test_empty_batch_returns_out():
    from aiter.ops.flydsl.gemm_a8w8_mxfp8_4wave import flydsl_mxfp8_4wave_gemm

    _, _, wsh, ws = _operands(256, 512, 1024)
    xq = torch.empty(0, 1024, dtype=dtypes.fp8, device="cuda")
    xs = torch.empty(0, 32, dtype=torch.uint8, device="cuda").view(dtypes.fp8_e8m0)
    out = torch.empty(0, 512, dtype=torch.bfloat16, device="cuda")
    assert flydsl_mxfp8_4wave_gemm(xq, wsh, xs, ws, out) is out


@_GFX950
def test_w_scale_cache_follows_in_place_update():
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import run_gemm_a8w8_mxfp8_gfx950

    m, n, k = 512, 512, 1024
    xq, xs, wsh, ws = _operands(m, n, k)
    name = make_mxfp8_4wave_kernel_name(True)
    a = _nan_out(m, n)
    run_gemm_a8w8_mxfp8_gfx950(xq, wsh, xs, ws, a, name)
    ws.view(torch.uint8).add_(1)  # a new weight scale in the same storage
    b = _nan_out(m, n)
    run_gemm_a8w8_mxfp8_gfx950(xq, wsh, xs, ws, b, name)
    ref = _nan_out(m, n)
    run_gemm_a8w8_mxfp8_gfx950(xq, wsh, xs, ws, ref, _8W)
    assert torch.equal(b, ref) and not torch.equal(a, b)


@_GFX950
def test_w_scale_cache_accepts_inference_tensor():
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import run_gemm_a8w8_mxfp8_gfx950

    m, n, k = 512, 512, 1024
    xq, xs, wsh, ws = _operands(m, n, k)
    with torch.inference_mode():
        ws_inf = ws.clone()  # an inference tensor: no version counter
    assert ws_inf.is_inference()
    name = make_mxfp8_4wave_kernel_name(True)
    a, b, ref = _nan_out(m, n), _nan_out(m, n), _nan_out(m, n)
    run_gemm_a8w8_mxfp8_gfx950(xq, wsh, xs, ws_inf, a, name)
    run_gemm_a8w8_mxfp8_gfx950(xq, wsh, xs, ws_inf, b, name)  # cache hit
    run_gemm_a8w8_mxfp8_gfx950(xq, wsh, xs, ws, ref, name)
    assert torch.equal(a, ref) and torch.equal(b, ref)


@_GFX950
def test_w_scale_cache_miss_in_capture_is_not_published():
    """A first call for a w_scale inside a CUDA graph capture only records the shuffle: the cache must not
    publish that buffer to eager calls or other graphs."""
    from aiter.ops.flydsl import gemm_a8w8_mxfp8_4wave as backend
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import run_gemm_a8w8_mxfp8_gfx950

    m, n, k = 512, 512, 1024
    xq, xs, wsh, ws = _operands(m, n, k)
    name = make_mxfp8_4wave_kernel_name(True)
    warm = _nan_out(m, n)
    run_gemm_a8w8_mxfp8_gfx950(
        xq, wsh, xs, ws, warm, name
    )  # compile outside the capture
    ws2 = ws.clone()  # a scale the cache has not seen
    n_entries = len(backend._W_SCALE_CACHE)
    out = _nan_out(m, n)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_gemm_a8w8_mxfp8_gfx950(xq, wsh, xs, ws2, out, name)
    assert len(backend._W_SCALE_CACHE) == n_entries
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, warm)
    eager = _nan_out(m, n)
    run_gemm_a8w8_mxfp8_gfx950(
        xq, wsh, xs, ws2, eager, name
    )  # now cached from an eager call
    assert len(backend._W_SCALE_CACHE) == n_entries + 1
    assert torch.equal(eager, warm)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-x"]))

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops import gemm_op_a4w4
from aiter.ops.flydsl.decode_gemm_mxfp4 import (
    decode_gemm_mxfp4_lds_bytes,
    flydsl_decode_gemm_mxfp4,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils


def raw_scales(shuffled, rows, k):
    sm, sn = shuffled.shape
    return (
        shuffled.view(torch.uint8)
        .view(sm // 32, sn // 8, 4, 16, 2, 2)
        .permute(0, 5, 3, 1, 4, 2)
        .contiguous()
        .view(sm, sn)[:rows, : k // 32]
        .contiguous()
    )


def dequant(q, scales, rows, k):
    # scales are raw [rows, k/32] e8m0; e8m0_to_f32 differs from a plain 2**(s-127)
    # only for 0xFF (NaN) and 0x00 (denormal 2**-127, which exp2 also represents).
    scale_f32 = fp4_utils.e8m0_to_f32(scales).repeat_interleave(32, dim=-1)
    return fp4_utils.mxfp4_to_f32(q.view(torch.uint8))[:, :k] * scale_f32


def metrics(actual, reference):
    delta = (actual.float() - reference.float()).abs()
    return {
        "max_abs": delta.max().item(),
        "mismatches": (actual != reference).sum().item(),
    }


@pytest.mark.parametrize(
    "M,N,K,tile_n,k_waves,pattern",
    [
        (8, 5120, 8704, 32, 2, "random"),
        (8, 8192, 5120, 32, 2, "random"),
        (1, 5120, 8704, 32, 2, "random"),
        (16, 5120, 8704, 32, 2, "random"),
        (8, 64, 1024, 64, 1, "structured"),
        (8, 32, 1280, 32, 2, "structured"),
        (1, 32, 2880, 32, 2, "random"),
        (8, 32, 2880, 32, 2, "structured"),
        (1, 64, 2880, 64, 1, "random"),
        (8, 128, 2880, 64, 1, "structured"),
        (8, 96, 7232, 32, 2, "structured"),
        (8, 64, 192, 64, 1, "structured"),
    ],
)
def test_decode_gemm_mxfp4(
    M,
    N,
    K,
    tile_n,
    k_waves,
    pattern,
    split_k=1,
    capture=False,
    supplied_workspace=False,
):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        pytest.skip(f"requires gfx950, got {arch}")
    torch.manual_seed(1219)
    quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    if pattern == "random":
        x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    else:
        x = (((torch.arange(M * K, device="cuda").reshape(M, K) % 19) - 9) / 4).to(
            torch.bfloat16
        )
        w = (((torch.arange(N * K, device="cuda").reshape(N, K) % 23) - 11) / 4).to(
            torch.bfloat16
        )
    xq, xs = quant(x, shuffle=True)
    wq, ws = quant(w, shuffle=True)
    b = shuffle_weight(wq, layout=(16, 16))
    reference = (
        dequant(xq, raw_scales(xs, M, K), M, K)
        @ dequant(wq, raw_scales(ws, N, K), N, K).T
    ).to(torch.bfloat16)
    if K % 256:
        # Keep only the consumer's scale rows and poison padded K groups.
        # Zero data must not encounter an E8M0 NaN scale in the last MFMA.
        padded_k = (K + 255) // 256 * 256
        poisoned = []
        for scales, rows in ((xs, 32), (ws, N)):
            raw = raw_scales(scales, rows, padded_k)
            raw[:, K // 32 :] = 0xFF
            poisoned.append(
                raw.view(rows // 32, 2, 16, padded_k // 256, 2, 4)
                .permute(0, 3, 5, 2, 4, 1)
                .contiguous()
                .view(rows, padded_k // 32)
                .view(scales.dtype)
            )
        xs, ws = poisoned
    out = torch.full((M, N), torch.nan, device="cuda", dtype=torch.bfloat16)
    workspace = (
        torch.full((split_k, M, N), torch.nan, device="cuda", dtype=torch.float32)
        if supplied_workspace
        else None
    )

    def run():
        return flydsl_decode_gemm_mxfp4(
            xq,
            b,
            xs,
            ws,
            out,
            tile_n=tile_n,
            k_waves=k_waves,
            split_k=split_k,
            workspace=workspace,
        )

    assert run() is out
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(2):
            out.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)
    torch.cuda.synchronize()
    print(
        f"M={M} N={N} K={K} tile_n={tile_n} k_waves={k_waves} split_k={split_k} "
        f"reference={metrics(out, reference)}",
        flush=True,
    )
    torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)


# K=7168 is 28 K256 tiles: split 4 partitions evenly (7 each), split 3 unevenly
# (10/9/9). K=2880 is 11.25 tiles, so its last tile is a K tail. Both (tile_n,k_waves)
# configs are covered; supplied_workspace exercises the caller-owned buffer.
@pytest.mark.parametrize(
    "M,N,K,tile_n,k_waves,split_k,supplied_workspace,pattern",
    [
        (8, 576, 7168, 32, 2, 4, False, "random"),  # even partition
        (8, 576, 7168, 64, 1, 4, True, "random"),  # even, caller workspace
        (8, 576, 7168, 32, 2, 3, False, "random"),  # uneven partition
        (1, 1536, 7168, 64, 1, 3, True, "random"),  # uneven, caller workspace
        (8, 576, 2880, 32, 2, 4, False, "random"),  # K tail
        (8, 576, 2880, 32, 2, 4, False, "structured"),  # K tail, structured data
        (1, 2880, 2880, 64, 1, 2, True, "random"),  # K tail, caller workspace
    ],
)
def test_decode_gemm_mxfp4_split(
    M, N, K, tile_n, k_waves, split_k, supplied_workspace, pattern
):
    test_decode_gemm_mxfp4(
        M,
        N,
        K,
        tile_n,
        k_waves,
        pattern,
        split_k=split_k,
        supplied_workspace=supplied_workspace,
    )


def test_decode_gemm_mxfp4_split_graph():
    test_decode_gemm_mxfp4(8, 576, 7168, 64, 1, "random", split_k=4, capture=True)


def test_decode_gemm_mxfp4_lds_guard():
    M, N, K = 16, 64, 28672
    expected = 260096
    assert decode_gemm_mxfp4_lds_bytes(M, K, 32, 2) == expected
    assert decode_gemm_mxfp4_lds_bytes(M, K, 32, 2, split_k=2) == 131072
    assert decode_gemm_mxfp4_lds_bytes(8, 1024, 64, 1) == 5120
    assert decode_gemm_mxfp4_lds_bytes(8, 2880, 32, 2, split_k=4) == 5888
    # The feasibility guard runs before any device query or compilation.
    a = torch.empty((M, K // 2), dtype=torch.uint8)
    b = torch.empty((N, K // 2), dtype=torch.uint8)
    scales = torch.empty(1, dtype=torch.uint8)
    out = torch.empty((M, N), dtype=torch.bfloat16)
    with pytest.raises(
        ValueError, match=f"A residency needs {expected} B LDS > 163840"
    ):
        flydsl_decode_gemm_mxfp4(a, b, scales, scales, out)
    for split_k in (0, 113, 1.5):
        with pytest.raises(ValueError, match="integer split_k"):
            decode_gemm_mxfp4_lds_bytes(M, K, 32, 2, split_k)


def test_parse_flydsl_decode_name():
    parse = gemm_op_a4w4._parse_flydsl_decode_name
    assert parse("flydsl_decode_t16x32x512_kw2_nb4_sk1") == {
        "tile_n": 32,
        "k_waves": 2,
        "num_buffers": 4,
        "split_k": 1,
    }
    assert parse("flydsl_decode_t16x64x256_kw1_nb4_sk14")["split_k"] == 14
    for bad in (
        "",
        None,
        "flydsl_decode_bn32_kg2_bs4_sk1",
        "flydsl_decode_t16x32x256_kw2_nb4_sk1",
        "flydsl_decode_t8x32x512_kw2_nb4_sk1",
        "flydsl_decode_t16x32x512_kw2_nb4_sk1_x",
        "xflydsl_decode_t16x32x512_kw2_nb4_sk1",
        "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E",
    ):
        assert parse(bad) is None


_A4W4_HEADER = "gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n"


def _use_tuned_rows(monkeypatch, tmp_path, rows, tag):
    """Point gemm_a4w4 at a temporary tuned CSV and drop every cache of the old one."""
    path = tmp_path / f"a4w4_{tag}.csv"
    path.write_text(_A4W4_HEADER + "".join(rows))
    monkeypatch.setenv("AITER_CONFIG_GEMM_A4W4", str(path))
    # get_config_file is lru_cached on the env *name*, not its value.
    AITER_CONFIGS.get_config_file.cache_clear()
    gemm_op_a4w4.get_GEMM_config.cache_clear()
    if hasattr(gemm_op_a4w4.get_GEMM_config, "gemm_dict"):
        del gemm_op_a4w4.get_GEMM_config.gemm_dict


@pytest.fixture
def clean_a4w4_config(monkeypatch):
    yield
    # monkeypatch restores the env after this teardown; clear caches again so
    # later tests re-read the default config.
    monkeypatch.delenv("AITER_CONFIG_GEMM_A4W4", raising=False)
    AITER_CONFIGS.get_config_file.cache_clear()
    gemm_op_a4w4.get_GEMM_config.cache_clear()
    if hasattr(gemm_op_a4w4.get_GEMM_config, "gemm_dict"):
        del gemm_op_a4w4.get_GEMM_config.gemm_dict


@pytest.mark.parametrize(
    "M,N,K,name,expect_flydsl",
    [
        (8, 5120, 8704, "flydsl_decode_t16x64x256_kw1_nb4_sk1", True),
        (8, 8192, 5120, "flydsl_decode_t16x32x512_kw2_nb4_sk1", True),
        (16, 5120, 8704, "flydsl_decode_t16x32x512_kw2_nb4_sk4", True),
        # Infeasible for FlyDSL (A residency exceeds LDS): must fall back to asm.
        (16, 16384, 53248, "flydsl_decode_t16x32x512_kw2_nb4_sk1", False),
    ],
)
def test_gemm_a4w4_dispatch_flydsl(
    M, N, K, name, expect_flydsl, monkeypatch, tmp_path, clean_a4w4_config
):
    _require_gfx950()
    from aiter.ops.flydsl import decode_gemm_mxfp4 as decode_gemm_mxfp4_mod

    args, reference = _operands(M, N, K, 1219)
    row = (
        f"{get_gfx_runtime()},{get_cu_num()},{M},{N},{K},-1,0,1.0,{name},0.0,0.0,0.0\n"
    )
    _use_tuned_rows(monkeypatch, tmp_path, [row], "flydsl")
    assert gemm_op_a4w4.get_GEMM_config(M, N, K)["kernelName"] == name
    # The spy proves the FlyDSL entry point ran (or, for the infeasible shape,
    # did not), so a silent fallback cannot satisfy a FlyDSL case.
    calls = []
    real = decode_gemm_mxfp4_mod.flydsl_decode_gemm_mxfp4

    def spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(decode_gemm_mxfp4_mod, "flydsl_decode_gemm_mxfp4", spy)
    out = aiter.gemm_a4w4(*args, bpreshuffle=True)
    torch.cuda.synchronize()
    assert len(calls) == (1 if expect_flydsl else 0)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16
    print(f"M={M} N={N} K={K} {name} reference={metrics(out, reference)}", flush=True)
    torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)


def _require_gfx950():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        pytest.skip(f"requires gfx950, got {arch}")


def _operands(M, N, K, seed):
    """Shuffled operands for gemm_a4w4 plus an independent dequant reference."""
    torch.manual_seed(seed)
    quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    xq, xs = quant(
        torch.randn((M, K), device="cuda", dtype=torch.bfloat16), shuffle=True
    )
    wq, ws = quant(
        torch.randn((N, K), device="cuda", dtype=torch.bfloat16), shuffle=True
    )
    reference = (
        dequant(xq, raw_scales(xs, M, K), M, K)
        @ dequant(wq, raw_scales(ws, N, K), N, K).T
    ).to(torch.bfloat16)
    b = shuffle_weight(wq, layout=(16, 16))
    return (xq, b, xs, ws), reference


_SPLITK_SHAPE = (16, 5120, 8704, "flydsl_decode_t16x32x512_kw2_nb4_sk4")


def _use_splitk_row(monkeypatch, tmp_path):
    M, N, K, name = _SPLITK_SHAPE
    row = (
        f"{get_gfx_runtime()},{get_cu_num()},{M},{N},{K},-1,0,1.0,{name},0.0,0.0,0.0\n"
    )
    _use_tuned_rows(monkeypatch, tmp_path, [row], "splitk")
    assert gemm_op_a4w4.get_GEMM_config(M, N, K)["kernelName"] == name


def test_gemm_a4w4_dispatch_flydsl_concurrent_streams(
    monkeypatch, tmp_path, clean_a4w4_config
):
    _require_gfx950()
    M, N, K, _ = _SPLITK_SHAPE
    ops = [_operands(M, N, K, seed) for seed in (1219, 4242)]
    _use_splitk_row(monkeypatch, tmp_path)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    torch.cuda.synchronize()
    results = [[], []]
    # Interleave launches without host syncs so the split-K partials of the two
    # streams are in flight together; a shared workspace would mix them.
    for _ in range(8):
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                results[i].append(aiter.gemm_a4w4(*ops[i][0], bpreshuffle=True))
    torch.cuda.synchronize()
    assert not torch.equal(ops[0][1], ops[1][1])
    for i in range(2):
        for out in results[i]:
            torch.testing.assert_close(out, ops[i][1], rtol=1e-2, atol=1e-2)


def test_gemm_a4w4_dispatch_flydsl_graph_and_eager(
    monkeypatch, tmp_path, clean_a4w4_config
):
    _require_gfx950()
    M, N, K, _ = _SPLITK_SHAPE
    (args_a, ref_a), (args_b, ref_b) = (_operands(M, N, K, s) for s in (1219, 4242))
    _use_splitk_row(monkeypatch, tmp_path)
    # The eager warmup and the graph capture must each get their own split-K
    # scratch: a workspace cached by the eager call must not be reused by the
    # capture.
    torch.testing.assert_close(
        aiter.gemm_a4w4(*args_a, bpreshuffle=True), ref_a, rtol=1e-2, atol=1e-2
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = aiter.gemm_a4w4(*args_a, bpreshuffle=True)
    for _ in range(2):
        graph_out.fill_(torch.nan)
        graph.replay()
        # An eager call with different operands between replays must not share
        # the graph's split-K scratch.
        eager = aiter.gemm_a4w4(*args_b, bpreshuffle=True)
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_out, ref_a, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(eager, ref_b, rtol=1e-2, atol=1e-2)


def test_gemm_a4w4_dispatch_real_config_uses_flydsl(monkeypatch, clean_a4w4_config):
    _require_gfx950()
    from aiter.ops.flydsl import decode_gemm_mxfp4 as decode_gemm_mxfp4_mod

    M, N, K = 8, 8192, 5120
    # No tuned-CSV override: the merged shipped config must route this shape.
    row = gemm_op_a4w4.get_GEMM_config(M, N, K)
    assert row is not None and row["kernelName"].startswith("flydsl_decode_")
    calls = []
    real = decode_gemm_mxfp4_mod.flydsl_decode_gemm_mxfp4

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(decode_gemm_mxfp4_mod, "flydsl_decode_gemm_mxfp4", spy)
    args, reference = _operands(M, N, K, 1219)
    out = aiter.gemm_a4w4(*args, bpreshuffle=True)
    torch.cuda.synchronize()
    assert len(calls) == 1
    assert out.shape == (M, N)
    torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-rs", "-s"]))

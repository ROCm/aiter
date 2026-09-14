# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernel, dispatch, graph and AOT regressions for FlyDSL MXFP8.

The standard correctness/perf sweep lives in op_tests/test_flydsl_mxfp8.py.
Run this integration suite with pytest; it does not time candidate comparisons.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_mxfp8 import (
    CONFIG_KEYS,
    DEFAULT_CONFIG,
    flydsl_mxfp8_gemm,
    flydsl_mxfp8_kernel_name,
    get_flydsl_mxfp8_configs,
    get_flydsl_mxfp8_kernel_params,
)
from aiter.ops.gemm_op_mxfp8 import MXFP8_KEYS
from aiter.ops.shuffle import shuffle_weight
from op_tests.test_flydsl_mxfp8 import run_torch

ROOT = Path(__file__).resolve().parents[2]
GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="requires gfx950",
)
CONFIGS = {
    "ft": dict(DEFAULT_CONFIG),
    "split": dict(DEFAULT_CONFIG, split_k=2),
    "slice": dict(DEFAULT_CONFIG, block_k=256, k_waves=2),
    "split_slice": dict(DEFAULT_CONFIG, block_k=256, k_waves=2, split_k=2),
    "hti": dict(
        DEFAULT_CONFIG,
        block_m=128,
        block_n=128,
        m_waves=2,
        use_half_tile_interleaved=True,
    ),
    "hti_split": dict(
        DEFAULT_CONFIG,
        block_m=128,
        block_n=128,
        m_waves=2,
        use_half_tile_interleaved=True,
        split_k=2,
    ),
    "mma32": dict(
        DEFAULT_CONFIG,
        block_m=32,
        block_n=32,
        block_k=64,
        n_waves=1,
        mma_m=32,
        mma_n=32,
        mma_k=64,
    ),
}


def inputs(m, n, k, block=32, dtype=torch.bfloat16, bias=False, bp=False):
    torch.manual_seed(42)
    a = torch.empty(m, k, device="cuda").uniform_(-1, 1).to(torch.float8_e4m3fn)
    w = torch.empty(n, k, device="cuda").uniform_(-1, 1).to(a.dtype)
    sa = torch.randint(123, 130, (m, k // block), device="cuda", dtype=torch.uint8)
    sb = torch.randint(
        123,
        130,
        (n if block == 32 else (n + 127) // 128, k // block),
        device="cuda",
        dtype=torch.uint8,
    )
    b = torch.randn(n, device="cuda", dtype=dtype) if bias else None
    ref = run_torch(a, w, sa, sb, block, torch.float32)
    if b is not None:
        ref += b.float()
    return a, shuffle_weight(w) if bp else w, sa, sb, b, ref.to(dtype)


def assert_result(y, ref, *, bf16_atol=0.2):
    if y.dtype == torch.float32:
        # Dense FP8 includes tiny products; scaled MFMA differs from a
        # dequantized FP32 GEMM by ~1e-3 on these inputs. A separate exact-value
        # test below checks that cshuffle does not truncate FP32 to BF16.
        torch.testing.assert_close(y, ref, atol=2e-3, rtol=2e-5)
    else:
        torch.testing.assert_close(y, ref, atol=bf16_atol, rtol=0.03)


@pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS)
@pytest.mark.parametrize("block,bp", [(32, False), (32, True), (128, True)])
@pytest.mark.parametrize("dtype,bias", [(torch.bfloat16, False), (torch.float32, True)])
@GPU
def test_kernel_paths(config, block, bp, dtype, bias):
    a, w, sa, sb, b, ref = inputs(33, 256, 1024, block, dtype, bias, bp)
    # Both scale dtypes must mean identical bytes.
    sa, sb = sa.view(torch.float8_e8m0fnu), sb.view(torch.float8_e8m0fnu)
    sat = block == 128
    if sat:
        sa = sa.t().contiguous().t()
    out = torch.full_like(ref, 17)
    for _ in range(3):
        out.fill_(-13)
        y = flydsl_mxfp8_gemm(
            a,
            w,
            sa,
            sb,
            out=out,
            bias=b,
            config=config,
            bpreshuffle=bp,
            scale_block=block,
            scale_a_transposed=sat,
        )
        assert y.data_ptr() == out.data_ptr()
        assert_result(y, ref)


@pytest.mark.parametrize(
    "m,n,k", [(1, 48, 128), (17, 80, 384), (65, 144, 768), (32, 384, 7168)]
)
@pytest.mark.parametrize("bp", [False, True])
@GPU
def test_tails_and_dsv4_shape(m, n, k, bp):
    a, w, sa, sb, _bias, ref = inputs(m, n, k, bp=bp)
    # Padded/offset but correctly aligned row strides and strided scale views.
    ap = torch.empty(m, k + 16, device="cuda", dtype=a.dtype)
    ap[:, 16:] = a
    sp = torch.empty(m, k // 32 * 2, device="cuda", dtype=sa.dtype)
    sp[:, ::2] = sa
    y = flydsl_mxfp8_gemm(ap[:, 16:], w, sp[:, ::2], sb, bpreshuffle=bp)
    assert_result(y, ref)


@pytest.mark.parametrize("bp", [False, True])
@GPU
def test_stream_and_graph(bp):
    a, w, sa, sb, b, ref = inputs(17, 128, 512, bias=True, bp=bp)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    out = torch.empty_like(ref)
    config = CONFIGS["hti_split"]
    # First use on this stream also creates and initializes split-K state.
    y = flydsl_mxfp8_gemm(
        a, w, sa, sb, out=out, bias=b, config=config, bpreshuffle=bp, stream=stream
    )
    stream.synchronize()
    assert_result(y, ref)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        flydsl_mxfp8_gemm(
            a, w, sa, sb, out=out, bias=b, config=config, bpreshuffle=bp, stream=stream
        )
    for _ in range(3):
        out.fill_(99)
        graph.replay()
        torch.cuda.synchronize()
        assert_result(out, ref)


@pytest.mark.parametrize("backend", ["eager", "inductor"])
@GPU
def test_tuned_gemm_and_compile(backend):
    from aiter.tuned_gemm import gemm_a16w16, tgemm

    a, w, sa, sb, b, ref = inputs(6, 64, 256, bias=True, bp=True)
    # Non-viewable batched input must not fall back to unscaled F.linear.
    x = a.reshape(2, 3, 256).transpose(0, 1)
    sx = sa.reshape(2, 3, 8).transpose(0, 1)
    expected = ref.reshape(2, 3, 64).transpose(0, 1)
    y = tgemm.mm(x, w, b, scale_a=sx, scale_b=sb)
    assert_result(y, expected)
    # Explicit layout survives the torch.compile boundary; Python tensor
    # attributes alone are not part of the operator schema.
    a, w, sa, sb, b, ref = inputs(6, 64, 256)
    sa, sb = sa.view(torch.float8_e8m0fnu), sb.view(torch.float8_e8m0fnu)
    compiled = torch.compile(gemm_a16w16, backend=backend, fullgraph=True)
    assert_result(compiled(a, w, scale_a=sa, scale_b=sb), ref)
    assert_result(
        compiled(a, shuffle_weight(w), scale_a=sa, scale_b=sb, bpreshuffle=True), ref
    )


@pytest.mark.parametrize("m", [1, 33, 128])
@GPU
def test_blockscale_model_entry(m):
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle

    a, w, sa, sb, _b, ref = inputs(m, 384, 512, 128, bp=True)
    # This is the exact transpose_scale=True byte layout of per_group_quant.
    packed = sa.t().contiguous().reshape(m, -1).view(torch.float8_e8m0fnu)
    out = torch.empty_like(ref)
    y = gemm_a8w8_blockscale_bpreshuffle(
        a,
        w,
        packed,
        sb.view(torch.float8_e8m0fnu),
        out=out,
    )
    assert y.data_ptr() == out.data_ptr()
    assert_result(y, ref)


@pytest.mark.parametrize("use_out", [False, True])
@GPU
def test_blockscale_inductor_and_graph(use_out):
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle

    a, w, sa, sb, _b, ref = inputs(17, 256, 512, 128, bp=True)
    sa = sa.t().contiguous().reshape(17, -1).view(torch.float8_e8m0fnu)
    sb = sb.view(torch.float8_e8m0fnu)
    out = torch.empty_like(ref)

    def fn(a, w, sa, sb, out):
        return gemm_a8w8_blockscale_bpreshuffle(
            a, w, sa, sb, out=out if use_out else None
        )

    compiled = torch.compile(fn, fullgraph=True)
    y = compiled(a, w, sa, sb, out)
    assert_result(y, ref)
    if use_out:
        assert y.data_ptr() == out.data_ptr()
        assert_result(out, ref)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(a, w, sa, sb, out)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y = fn(a, w, sa, sb, out)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert_result(y, ref)


@GPU
def test_invalid_inputs():
    a, w, sa, sb, _b, _ref = inputs(8, 64, 256)
    with pytest.raises(ValueError, match="scales"):
        flydsl_mxfp8_gemm(a, w, sa.float(), sb)
    with pytest.raises(ValueError, match="shape"):
        flydsl_mxfp8_gemm(a, w, sa[:, :-1], sb)
    with pytest.raises(ValueError, match="positive"):
        flydsl_mxfp8_gemm(a[:0], w, sa[:0], sb)
    with pytest.raises(ValueError, match="row-major"):
        flydsl_mxfp8_gemm(a, w, sa, sb, scale_a_transposed=True)
    with pytest.raises(ValueError, match="out"):
        flydsl_mxfp8_gemm(a, w, sa, sb, out=torch.empty(64, 8, device="cuda").t())
    with pytest.raises((ValueError, AssertionError), match="output|bf16|fp32"):
        flydsl_mxfp8_gemm(a, w, sa, sb, out_dtype=torch.float16)


@pytest.mark.parametrize("m,k", [(1, 128), (1, 7168), (17, 128)])
@GPU
def test_singleton_scale_strides(m, k):
    a, w, sa, sb, _b, ref = inputs(m, 128, k, 128, bp=True)
    # Both views are contiguous to Torch, but may not have the unit leading
    # stride that the FlyDSL ABI needs. Mirrors tuner data generation.
    sa = sa.t().contiguous().t()
    y = flydsl_mxfp8_gemm(
        a, w, sa, sb, scale_block=128, bpreshuffle=True, scale_a_transposed=True
    )
    assert_result(y, ref)


def test_names_and_catalog():
    for config in CONFIGS.values():
        for block, bp, sat in [
            (32, False, False),
            (32, True, False),
            (128, True, True),
        ]:
            name = flydsl_mxfp8_kernel_name(
                config, scale_block=block, bpreshuffle=bp, scale_a_transposed=sat
            )
            parsed = get_flydsl_mxfp8_kernel_params(name, config["split_k"])
            assert {key: parsed[key] for key in CONFIG_KEYS} == config
            assert parsed["scale_block"] == block
            assert parsed["bpreshuffle"] == bp
            assert parsed["scale_a_transposed"] == sat
            assert (
                get_flydsl_mxfp8_kernel_params(name.replace("gfx950", "gfx1250"))
                is None
            )
            if config["split_k"] == 1:
                assert (
                    get_flydsl_mxfp8_kernel_params(name.replace("_ks1_", "_ks0_"))
                    is None
                )
    assert get_flydsl_mxfp8_kernel_params("flydsl_hgemm_bad") is None
    if torch.cuda.is_available() and get_gfx() == "gfx950":
        assert not get_flydsl_mxfp8_configs(0, 64, 256)
        assert not get_flydsl_mxfp8_configs(8, 63, 256, bpreshuffle=True)
        assert get_flydsl_mxfp8_configs(8, 64, 256, bpreshuffle=True)


def _run(code_or_args, env):
    args = [sys.executable]
    args += ["-c", code_or_args] if isinstance(code_or_args, str) else code_or_args
    ret = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    assert ret.returncode == 0, ret.stdout + "\n" + ret.stderr
    return ret.stdout


@pytest.mark.parametrize("config_name", ["ft", "split_slice", "hti_split", "mma32"])
@GPU
def test_cpu_aot_fresh_run_only(tmp_path, config_name):
    config = CONFIGS[config_name]
    csv_path = tmp_path / "tuned.csv"
    rows = []
    for dtype, bias in [(torch.bfloat16, False), (torch.float32, True)]:
        for block, bp, sat in [
            (32, False, False),
            (32, True, False),
            (128, True, True),
        ]:
            row = dict(
                zip(
                    MXFP8_KEYS,
                    ["gfx950", 256, 33, 256, 1024, str(dtype), bias, block, bp, sat],
                )
            )
            row.update(
                libtype="flydsl",
                kernelId=0,
                splitK=config["split_k"],
                us=1,
                kernelName=flydsl_mxfp8_kernel_name(
                    config,
                    out_dtype=dtype,
                    has_bias=bias,
                    scale_block=block,
                    bpreshuffle=bp,
                    scale_a_transposed=sat,
                ),
            )
            rows.append(row)
    with csv_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    env = dict(
        os.environ,
        FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path / "cache"),
        AITER_CONFIG_GEMM_MXFP8=str(csv_path),
    )
    aot_env = dict(
        env,
        AITER_AOT_IMPORT="1",
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
        GPU_ARCHS="gfx950",
        FLYDSL_GPU_ARCH="gfx950",
        AITER_FLYDSL_AOT_WORKERS="2",
    )
    aot_env.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    _run(["-m", "aiter.aot.flydsl.gemm", "--csv", str(csv_path)], aot_env)
    code = f"""
import csv, torch
from op_tests.flydsl_tests.test_mxfp8_integration import inputs, assert_result
from aiter.ops.gemm_op_mxfp8 import gemm_mxfp8
from aiter.tuned_gemm import tgemm
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
for row in csv.DictReader(open({str(csv_path)!r})):
    block = int(row['scale_block'])
    bp, bias, sat = [row[k] == 'True' for k in ('bpreshuffle','bias','scale_a_transposed')]
    dtype = getattr(torch, row['outdtype'].split('.')[1])
    a,w,sa,sb,b,ref = inputs(33,256,1024,block,dtype,bias,bp)
    if sat:
        sa = sa.t().contiguous().t()
    out = torch.full_like(ref, 33)
    y = gemm_mxfp8(a,w,sa,sb,out=out,bias=b,dtype=dtype,
                  scale_block=block,bpreshuffle=bp,scale_a_transposed=sat)
    assert y.data_ptr() == out.data_ptr()
    assert_result(y,ref)
    if block == 32:
        assert_result(tgemm.mm(a,w,b,otype=dtype,scale_a=sa,scale_b=sb),ref)
    elif not bias:
        packed = sa.t().contiguous().reshape(33,-1).view(torch.float8_e8m0fnu)
        y = gemm_a8w8_blockscale_bpreshuffle(a,w,packed,sb,out=out)
        assert_result(y,ref)
print('run-only PASS')
"""
    env["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
    assert "run-only PASS" in _run(code, env)


@GPU
def test_tuner_roundtrip(tmp_path):
    shapes = tmp_path / "shapes.csv"
    shapes.write_text(
        "M,N,K,outdtype,bias,scale_block,bpreshuffle,scale_a_transposed\n"
        "16,64,128,torch.bfloat16,False,32,False,False\n"
        "16,64,128,torch.bfloat16,False,32,True,False\n"
        "16,128,128,torch.bfloat16,False,128,True,True\n"
    )
    tuned = tmp_path / "tuned.csv"
    env = dict(os.environ)
    env.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    script = "csrc/gemm_mxfp8/gemm_mxfp8_tune.py"
    _run(
        [
            script,
            "-i",
            str(shapes),
            "-o",
            str(tuned),
            "--warmup",
            "1",
            "--iters",
            "3",
            "--screen-topk",
            "2",
        ],
        env,
    )
    rows = list(csv.DictReader(tuned.open()))
    assert len(rows) == 3
    assert all(
        float(row["errRatio"]) == 0 and row["libtype"] == "flydsl" for row in rows
    )
    assert len({tuple(row[k] for k in MXFP8_KEYS) for row in rows}) == 3
    env["AITER_CONFIG_GEMM_MXFP8"] = str(tuned)
    _run([script, "--run_config", str(tuned), "--warmup", "1", "--iters", "3"], env)


@GPU
def test_real_quantizer_to_model():
    from aiter import dtypes
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
    from aiter.ops.quant import per_group_quant_hip

    m, n, k = 33, 256, 512
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    a, sa = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, transpose_scale=True
    )
    _, row_scale = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, transpose_scale=False
    )
    logical_scale = sa.view(torch.uint8).reshape(k // 128, m).t()
    assert torch.equal(logical_scale, row_scale.view(torch.uint8))
    w = torch.randn(n, k, device="cuda").to(dtypes.fp8)
    sw = torch.randint(124, 129, (n // 128, k // 128), device="cuda", dtype=torch.uint8)
    ref = (a.float() * row_scale.float().repeat_interleave(128, 1)) @ (
        w.float()
        * torch.exp2(sw.float() - 127)
        .repeat_interleave(128, 0)
        .repeat_interleave(128, 1)
    ).t()
    out = gemm_a8w8_blockscale_bpreshuffle(
        a, shuffle_weight(w), sa, sw.view(dtypes.fp8_e8m0)
    )
    assert_result(out, ref.to(torch.bfloat16))


@pytest.mark.parametrize(
    "config,k",
    [
        (dict(DEFAULT_CONFIG, stages=3, group_m=4), 768),
        (dict(DEFAULT_CONFIG, stages=4, split_k=2), 1536),
        (dict(CONFIGS["hti"], block_n=256, n_waves=4, group_m=4), 1536),
        (dict(CONFIGS["hti"], block_n=256, n_waves=4), 2048),
    ],
)
@pytest.mark.parametrize("block", [32, 128])
@GPU
def test_pipeline_wrap_and_scale_chunk(config, k, block):
    a, w, sa, sb, _b, ref = inputs(257, 384, k, block, bp=True)
    y = flydsl_mxfp8_gemm(
        a, w, sa, sb, config=config, scale_block=block, bpreshuffle=True
    )
    # BF16 split-K rounds each partial before the atomic sum. Like HGEMM,
    # scale the cancellation allowance with K and the number of partitions.
    # Strict FP32 precision is tested separately for every pipeline below.
    assert_result(y, ref, bf16_atol=0.2 * (k / 1024) ** 0.5 * config["split_k"])


@pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS)
@GPU
def test_fp32_keeps_accumulator_precision(config):
    a, w, sa, sb, b, _ = inputs(33, 128, 1024, 32, torch.float32, True)
    # Binary-exact FP8 operands, no tiny products. Double reference isolates
    # cshuffle/split/slice accuracy from the scaled-MFMA tiny-value behavior.
    a = torch.randint(-16, 17, a.shape, device="cuda").div(16).to(a.dtype)
    w = torch.randint(-16, 17, w.shape, device="cuda").div(16).to(w.dtype)
    ref = (a.double() * torch.exp2(sa.double() - 127).repeat_interleave(32, 1)) @ (
        w.double() * torch.exp2(sb.double() - 127).repeat_interleave(32, 1)
    ).t() + b.double()
    y = flydsl_mxfp8_gemm(
        a,
        shuffle_weight(w),
        sa,
        sb,
        bias=b,
        config=config,
        out_dtype=torch.float32,
        bpreshuffle=True,
    )
    torch.testing.assert_close(y, ref.float(), atol=2e-5, rtol=1e-6)


@GPU
def test_empty_aot_cache_fails_loudly(tmp_path):
    code = """
import torch
from op_tests.flydsl_tests.test_mxfp8_integration import inputs
from aiter.ops.flydsl.gemm_mxfp8 import flydsl_mxfp8_gemm
a,w,sa,sb,b,ref = inputs(16,64,128)
try:
    flydsl_mxfp8_gemm(a,w,sa,sb)
except RuntimeError as exc:
    assert 'no usable AOT cache' in str(exc), str(exc)
else:
    raise AssertionError('run-only mode silently compiled a missing kernel')
"""
    _run(
        code,
        dict(
            os.environ,
            FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path),
            FLYDSL_RUNTIME_RUN_ONLY="1",
        ),
    )


def test_config_keys_and_invalid_row(tmp_path, monkeypatch):
    from aiter.ops import gemm_op_mxfp8 as op
    from aiter.ops.flydsl.gemm_mxfp8 import DEFAULT_CONFIG

    path = tmp_path / "tuned.csv"
    rows = []
    for sb, bp, sat in [(32, False, False), (32, True, False), (128, True, True)]:
        row = dict(
            zip(
                MXFP8_KEYS,
                ["gfx950", 256, 32, 128, 512, "torch.bfloat16", False, sb, bp, sat],
            )
        )
        row.update(
            libtype="flydsl",
            splitK=2,
            kernelName=flydsl_mxfp8_kernel_name(
                CONFIGS["split"], scale_block=sb, bpreshuffle=bp, scale_a_transposed=sat
            ),
        )
        rows.append(row)
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    loaded = op._load_mxfp8_configs(str(path))
    assert len(loaded) == 3
    # Legacy scalar-FP8 table is never queried for the MXFP8 path.
    from types import SimpleNamespace

    monkeypatch.setattr(
        op, "AITER_CONFIGS", SimpleNamespace(AITER_CONFIG_GEMM_MXFP8_FILE=str(path))
    )
    op.get_mxfp8_config.cache_clear()
    if torch.cuda.is_available() and get_gfx() == "gfx950":
        for sb, bp, sat in [(32, False, False), (32, True, False), (128, True, True)]:
            assert (
                op.get_mxfp8_config(32, 128, 512, torch.bfloat16, False, sb, bp, sat)[
                    "split_k"
                ]
                == 2
            )
        # A name/row mode mismatch cannot dispatch a differently laid-out kernel.
        next(iter(loaded.values()))["kernelName"] = flydsl_mxfp8_kernel_name(
            CONFIGS["split"], bpreshuffle=True
        )
        op.get_mxfp8_config.cache_clear()
        assert op.get_mxfp8_config(32, 128, 512, torch.bfloat16) == DEFAULT_CONFIG
    op.get_mxfp8_config.cache_clear()
    op._load_mxfp8_configs.cache_clear()


@GPU
def test_tuner_rejects_unstable_splitk(monkeypatch):
    from csrc.gemm_mxfp8 import gemm_mxfp8_tune as tuner

    config = CONFIGS["split"]
    keys = ("gfx950", 256, 1, 64, 256, "torch.bfloat16", False, 32, False, False)
    name = flydsl_mxfp8_kernel_name(config)
    info = (keys, 0, 2, name)
    calls = []

    def unstable(a, b, sa, sb, out, bias, config, block, bp, sat):
        calls.append(None)
        out.copy_(tuner.reference(a, b, sa, sb, bias, out.dtype, block))
        if len(calls) == 2:
            out.fill_(float("nan"))
        return out

    monkeypatch.setattr(tuner, "run_kernel", unstable)
    result = tuner.check_splitk_stability([(info, 5.0, 0.0)])
    assert result == [(info, 5.0, 1.0)]
    assert len(calls) == 2


def test_dynamic_split_name():
    name = flydsl_mxfp8_kernel_name(dict(DEFAULT_CONFIG, split_k=2))
    assert "_ksd_" in name
    for split_k in (2, 4, 7, 8):
        assert flydsl_mxfp8_kernel_name(dict(DEFAULT_CONFIG, split_k=split_k)) == name
        params = get_flydsl_mxfp8_kernel_params(name, split_k)
        assert params["split_k"] == split_k
    assert get_flydsl_mxfp8_kernel_params(name) is None
    assert get_flydsl_mxfp8_kernel_params(name, 0) is None
    assert get_flydsl_mxfp8_kernel_params(name, 2.5) is None
    assert get_flydsl_mxfp8_kernel_params(name.replace("_ksd_", "_ks1_"), 2) is None


@GPU
def test_hgemm_search_space():
    configs = get_flydsl_mxfp8_configs(
        32, 768, 7168, torch.bfloat16, False, 128, True, True
    )
    assert {c["split_k"] for c in configs} == {1, 2, 4, 7, 8}
    assert {c["k_waves"] for c in configs} == {1, 2, 4}
    assert {c["block_k"] for c in configs} == {128, 256, 512}
    assert {c["stages"] for c in configs} == set(range(2, 10))
    assert any(c["block_m"] == 32 and c["block_n"] == 96 for c in configs)
    assert any(c["split_k"] == 7 and c["k_waves"] > 1 for c in configs)


@pytest.mark.parametrize("split_k,slice_k", [(3, 1), (7, 1), (7, 2), (7, 4)])
@GPU
def test_expanded_split_slice_space(split_k, slice_k):
    k = 384 if split_k == 3 else 7168
    config = dict(
        DEFAULT_CONFIG, block_k=128 * slice_k, split_k=split_k, k_waves=slice_k
    )
    a, w, sa, sb, _b, _ref = inputs(17, 128, k, 128)
    # Exact-valued inputs isolate slice/split correctness from tiny MFMA products.
    a = torch.randint(-16, 17, a.shape, device="cuda").div(16).to(a.dtype)
    w = torch.randint(-16, 17, w.shape, device="cuda").div(16).to(w.dtype)
    ref = run_torch(a, w, sa, sb, 128, torch.float32)
    y = flydsl_mxfp8_gemm(
        a,
        shuffle_weight(w),
        sa,
        sb,
        config=config,
        out_dtype=torch.float32,
        scale_block=128,
        bpreshuffle=True,
        scale_a_transposed=True,
    )
    torch.testing.assert_close(y, ref, atol=1e-4, rtol=1e-5)


@GPU
def test_ksd_aot_reuses_dynamic_split_count(tmp_path):
    config = dict(DEFAULT_CONFIG, split_k=2)
    name = flydsl_mxfp8_kernel_name(
        config,
        out_dtype=torch.float32,
        scale_block=128,
        bpreshuffle=True,
        scale_a_transposed=True,
    )
    path = tmp_path / "tuned.csv"
    row = dict(
        zip(
            MXFP8_KEYS,
            ["gfx950", 256, 1, 128, 7168, "torch.float32", False, 128, True, True],
        )
    )
    row.update(libtype="flydsl", kernelName=name, splitK=2)
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    env = dict(os.environ, FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path / "cache"))
    aot_env = dict(
        env,
        AITER_AOT_IMPORT="1",
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
        GPU_ARCHS="gfx950",
    )
    aot_env.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    _run(["-m", "aiter.aot.flydsl.gemm", "--csv", str(path)], aot_env)
    code = """
import torch
from aiter.ops.flydsl.gemm_mxfp8 import DEFAULT_CONFIG, flydsl_mxfp8_gemm
from aiter.ops.flydsl.kernels.scaled_gemm_gfx950 import scaled_gemm_gfx950
a = torch.ones(1,7168,device='cuda').to(torch.float8_e4m3fn)
w = torch.ones(128,7168,device='cuda').to(a.dtype)
sa = torch.full((1,56),127,device='cuda',dtype=torch.uint8)
sb = torch.full((1,56),127,device='cuda',dtype=torch.uint8)
for split in (2,4,7,8):
    y = flydsl_mxfp8_gemm(a,w,sa,sb,config=dict(DEFAULT_CONFIG,split_k=split),
                          out_dtype=torch.float32,scale_block=128,
                          bpreshuffle=True,scale_a_transposed=True)
    torch.testing.assert_close(y,torch.full_like(y,7168),atol=0,rtol=0)
assert len(scaled_gemm_gfx950._compiled_cache) == 1
"""
    _run(code, dict(env, FLYDSL_RUNTIME_RUN_ONLY="1"))


def test_shared_hgemm_space():
    from aiter.ops.flydsl.gemm_a16w16_policy import gemm_config_space

    bf16 = gemm_config_space(7168)
    mx = gemm_config_space(7168, block_k=(128, 256, 512), k_waves=(1, 2, 4))
    assert set(bf16["split_k"]) == {1, 2, 4, 7, 8}
    for key in bf16:
        if key not in ("block_k", "k_waves"):
            assert bf16[key] == mx[key]
    assert set(mx["block_k"]) == {128, 256, 512}
    assert set(mx["k_waves"]) == {1, 2, 4}


def test_aot_rejects_mismatched_split_column(tmp_path):
    from aiter.aot.flydsl.gemm import parse_csv

    name = flydsl_mxfp8_kernel_name(dict(DEFAULT_CONFIG, split_k=2))
    path = tmp_path / "invalid.csv"
    with path.open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gfx",
                "cu_num",
                "M",
                "N",
                "K",
                "libtype",
                "kernelName",
                "splitK",
            ],
        )
        writer.writeheader()
        writer.writerow(
            dict(
                gfx="gfx950",
                cu_num=256,
                M=1,
                N=64,
                K=256,
                libtype="flydsl",
                kernelName=name,
                splitK=1,
            )
        )
    with pytest.raises(ValueError, match="name/splitK"):
        parse_csv(path)

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Standard 1x32 MXFP8 integration regressions; perf sweep is test_flydsl_mxfp8.py."""

import csv
import itertools
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
    get_flydsl_mxfp8_kernel_params,
)
from aiter.ops.gemm_op_mxfp8 import MXFP8_KEYS
from aiter.ops.shuffle import shuffle_weight
from op_tests.test_flydsl_mxfp8 import run_torch

ROOT = Path(__file__).resolve().parents[2]
GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950", reason="requires gfx950"
)
CONFIGS = {
    "ft": dict(DEFAULT_CONFIG),
    "split": dict(DEFAULT_CONFIG, split_k=7),
    "slice": dict(DEFAULT_CONFIG, block_k=512, k_waves=4),
    "split_slice": dict(DEFAULT_CONFIG, block_k=256, k_waves=2, split_k=14),
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
        split_k=7,
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


def inputs(m=17, n=128, k=7168, dtype=torch.bfloat16, bias=False, bp=False):
    torch.manual_seed(0)
    a = torch.empty(m, k, device="cuda").uniform_(-1, 1).to(torch.float8_e4m3fn)
    w = torch.empty(n, k, device="cuda").uniform_(-1, 1).to(a.dtype)
    sa = torch.randint(124, 129, (m, k // 32), device="cuda", dtype=torch.uint8)
    sb = torch.randint(124, 129, (n, k // 32), device="cuda", dtype=torch.uint8)
    b = torch.randn(n, device="cuda", dtype=dtype) if bias else None
    ref = run_torch(a, w, sa, sb, torch.float32)
    if b is not None:
        ref += b.float()
    return a, shuffle_weight(w) if bp else w, sa, sb, b, ref.to(dtype)


@pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS)
@pytest.mark.parametrize("bp", [False, True])
@pytest.mark.parametrize("dtype,bias", [(torch.bfloat16, False), (torch.float32, True)])
@GPU
def test_kernel_paths(config, bp, dtype, bias):
    a, w, sa, sb, b, ref = inputs(dtype=dtype, bias=bias, bp=bp)
    out = torch.empty_like(ref)
    for _ in range(3):
        out.fill_(float("nan"))
        y = flydsl_mxfp8_gemm(
            a,
            w,
            sa.view(torch.float8_e8m0fnu),
            sb.view(torch.float8_e8m0fnu),
            out=out,
            bias=b,
            config=config,
            bpreshuffle=bp,
        )
        assert y.data_ptr() == out.data_ptr()
        torch.testing.assert_close(y, ref, atol=0.1, rtol=0.03)


@pytest.mark.parametrize(
    "m,n,k", [(1, 48, 128), (33, 80, 384), (65, 144, 768), (1, 32, 64)]
)
@GPU
def test_strides_and_tails(m, n, k):
    a, w, sa, sb, _b, ref = inputs(m, n, k, bp=True)
    ap = torch.empty((m, k + 16), device="cuda", dtype=a.dtype)
    ap[:, 16:] = a
    sp = torch.empty((m, sa.shape[1] * 2), device="cuda", dtype=sa.dtype)
    sp[:, ::2] = sa
    from aiter.ops.gemm_op_mxfp8 import get_mxfp8_config

    cfg = get_mxfp8_config(m, n, k, ref.dtype, False, True)
    y = flydsl_mxfp8_gemm(ap[:, 16:], w, sp[:, ::2], sb, config=cfg, bpreshuffle=True)
    torch.testing.assert_close(y, ref, atol=0.1, rtol=0.03)


@GPU
def test_stream_graph_and_tgemm_compile():
    from aiter.tuned_gemm import tgemm

    a, w, sa, sb, b, ref = inputs(bias=True, bp=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    out = torch.empty_like(ref)

    def run():
        return flydsl_mxfp8_gemm(
            a,
            w,
            sa,
            sb,
            out=out,
            bias=b,
            config=CONFIGS["hti_split"],
            bpreshuffle=True,
            stream=stream,
        )

    run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, atol=0.1, rtol=0.03)

    def model(a, w, sa, sb, b):
        return tgemm.mm(a, w, b, scale_a=sa, scale_b=sb, bpreshuffle=True)

    compiled = torch.compile(model, fullgraph=True)
    y = compiled(a, w, sa.view(torch.float8_e8m0fnu), sb.view(torch.float8_e8m0fnu), b)
    torch.testing.assert_close(y, ref, atol=0.1, rtol=0.03)


@GPU
def test_reject_non_mxfp8_scales():
    a, w, sa, sb, _b, _ref = inputs()
    for sx, sw in [
        (sa.float(), sb.float()),
        (sa[:, : sa.shape[1] // 4], sb[:, : sb.shape[1] // 4]),
    ]:
        with pytest.raises(ValueError, match="scale"):
            flydsl_mxfp8_gemm(a, w, sx, sw)
    with pytest.raises(TypeError):
        flydsl_mxfp8_gemm(a, w, sa, sb, scale_block=128)
    old = "flydsl_mxfp8_bf16_sb128_bp1_sat1_t32x64x128x2_ks1_w1x2x1_mma16x16x128_bias0_gm0_pft_gfx950"
    assert get_flydsl_mxfp8_kernel_params(old) is None


def test_names():
    for cfg in CONFIGS.values():
        name = flydsl_mxfp8_kernel_name(cfg, bpreshuffle=True)
        p = get_flydsl_mxfp8_kernel_params(name, cfg["split_k"])
        assert {k: p[k] for k in CONFIG_KEYS} == cfg
        assert "_sb" not in name and "_sat" not in name
    name = flydsl_mxfp8_kernel_name(dict(DEFAULT_CONFIG, split_k=2))
    for sk in [2, 4, 7, 14, 28]:
        assert name == flydsl_mxfp8_kernel_name(dict(DEFAULT_CONFIG, split_k=sk))
        assert get_flydsl_mxfp8_kernel_params(name, sk)["split_k"] == sk
    assert get_flydsl_mxfp8_kernel_params(name, 1) is None


def run_process(args, env):
    p = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert p.returncode == 0, p.stdout + "\n" + p.stderr
    return p.stdout


@GPU
def test_tuner_csv_aot_runtime_roundtrip(tmp_path):
    shapes = tmp_path / "shapes.csv"
    shapes.write_text(
        "M,N,K,outdtype,bias,bpreshuffle\n"
        "16,64,128,torch.bfloat16,False,False\n"
        "16,64,128,torch.bfloat16,False,True\n"
    )
    tuned = tmp_path / "tuned.csv"
    env = dict(os.environ)
    env.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    script = "csrc/gemm_mxfp8/gemm_mxfp8_tune.py"
    run_process(
        [
            script,
            "-i",
            str(shapes),
            "-o",
            str(tuned),
            "--screen-topk",
            "2",
            "--warmup",
            "1",
            "--iters",
            "3",
        ],
        env,
    )
    rows = list(csv.DictReader(tuned.open()))
    assert len(rows) == 2 and all(float(r["errRatio"]) == 0 for r in rows)
    assert set(MXFP8_KEYS).issubset(rows[0])
    env.update(
        FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path / "cache"),
        AITER_CONFIG_GEMM_MXFP8=str(tuned),
    )
    run_process(
        ["-m", "aiter.aot.flydsl.gemm", "--csv", str(tuned)],
        dict(
            env,
            AITER_AOT_IMPORT="1",
            HIP_VISIBLE_DEVICES="",
            ROCR_VISIBLE_DEVICES="",
            GPU_ARCHS="gfx950",
            AITER_FLYDSL_AOT_WORKERS="2",
        ),
    )
    run_process(
        [script, "--run_config", str(tuned), "--warmup", "1", "--iters", "3"],
        dict(env, FLYDSL_RUNTIME_RUN_ONLY="1"),
    )
    code = """
import os,csv,torch
from aiter.tuned_gemm import tgemm
from aiter import gemm_a8w8_mxfp8
from op_tests.flydsl_tests.test_mxfp8_integration import inputs
rows=list(csv.DictReader(open(os.environ['AITER_CONFIG_GEMM_MXFP8'])))
for row in rows:
    bp = row['bpreshuffle'] == 'True'
    a,w,sa,sb,b,ref=inputs(16,64,128,bp=bp)
    y=tgemm.mm(a,w,scale_a=sa,scale_b=sb,bpreshuffle=bp)
    torch.testing.assert_close(y,ref,atol=.1,rtol=.03)
    y=gemm_a8w8_mxfp8(a,w,sa,sb,bpreshuffle=bp)
    torch.testing.assert_close(y,ref,atol=.1,rtol=.03)
"""
    run_process(["-c", code], dict(env, FLYDSL_RUNTIME_RUN_ONLY="1"))
    # Exercise the standard op-test under run-only as well.
    run_process(
        [
            "op_tests/test_flydsl_mxfp8.py",
            "-s",
            "16,64,128",
            "-l",
            "plain",
            "preshuffle",
        ],
        dict(env, FLYDSL_RUNTIME_RUN_ONLY="1"),
    )


@pytest.mark.parametrize("direct_b,hti", [(False, False), (True, False), (False, True)])
@GPU
def test_dynamic_split_aot_and_bias(tmp_path, direct_b, hti):
    rows = []
    for dtype, bias, bp in itertools.product(
        ["bf16", "fp32"], [False, True], ([True] if direct_b else [False, True])
    ):
        dt = getattr(torch, "bfloat16" if dtype == "bf16" else "float32")
        cfg = dict(DEFAULT_CONFIG, split_k=2, direct_b=direct_b)
        if hti:
            cfg.update(
                block_m=128, block_n=128, m_waves=2, use_half_tile_interleaved=True
            )
        row = dict(zip(MXFP8_KEYS, ["gfx950", 256, 17, 128, 7168, str(dt), bias, bp]))
        row.update(
            libtype="flydsl",
            splitK=2,
            kernelName=flydsl_mxfp8_kernel_name(
                cfg, out_dtype=dt, has_bias=bias, bpreshuffle=bp
            ),
        )
        rows.append(row)
    path = tmp_path / "tuned.csv"
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    env = dict(os.environ, FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    run_process(
        ["-m", "aiter.aot.flydsl.gemm", "--csv", str(path)],
        dict(
            env,
            AITER_AOT_IMPORT="1",
            HIP_VISIBLE_DEVICES="",
            ROCR_VISIBLE_DEVICES="",
            GPU_ARCHS="gfx950",
            AITER_FLYDSL_AOT_WORKERS="2",
        ),
    )
    code = f"direct_b = {direct_b!r}\nhti = {hti!r}\n" + """
import itertools,torch
from op_tests.flydsl_tests.test_mxfp8_integration import inputs
from aiter.ops.flydsl.gemm_mxfp8 import DEFAULT_CONFIG,flydsl_mxfp8_gemm
for dtype,bias,bp in itertools.product([torch.bfloat16,torch.float32],[False,True],([True] if direct_b else [False,True])):
    a,w,sa,sb,b,ref=inputs(dtype=dtype,bias=bias,bp=bp)
    cfg=dict(DEFAULT_CONFIG,direct_b=direct_b)
    if hti:
        cfg.update(block_m=128,block_n=128,m_waves=2,use_half_tile_interleaved=True)
    for split_k in [2,4,7,14,28]:
        y=flydsl_mxfp8_gemm(a,w,sa,sb,bias=b,out_dtype=dtype,
                            bpreshuffle=bp,config=dict(cfg,split_k=split_k))
        torch.testing.assert_close(y,ref,atol=.1,rtol=.03)
"""
    run_process(["-c", code], dict(env, FLYDSL_RUNTIME_RUN_ONLY="1"))


@GPU
def test_real_standard_quantizer():
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip
    from aiter.tuned_gemm import tgemm

    x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, 256, device="cuda", dtype=x.dtype)
    a, sa = per_1x32_mx_quant_hip(x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)
    b, sb = per_1x32_mx_quant_hip(w, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)
    assert sa.shape == (17, 8) and sb.shape == (128, 8)
    ref = run_torch(a, b, sa, sb, torch.bfloat16)
    y = tgemm.mm(a, shuffle_weight(b), scale_a=sa, scale_b=sb, bpreshuffle=True)
    torch.testing.assert_close(y, ref, atol=0.1, rtol=0.03)


@GPU
def test_run_only_rejects_empty_cache(tmp_path):
    code = """
from op_tests.flydsl_tests.test_mxfp8_integration import inputs
from aiter.ops.flydsl.gemm_mxfp8 import flydsl_mxfp8_gemm
a,w,sa,sb,b,ref=inputs(16,64,128)
try:
    flydsl_mxfp8_gemm(a,w,sa,sb)
except RuntimeError as exc:
    assert "no usable AOT cache" in str(exc)
else:
    raise AssertionError("run-only silently compiled a missing kernel")
"""
    run_process(
        ["-c", code],
        dict(
            os.environ,
            FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path),
            FLYDSL_RUNTIME_RUN_ONLY="1",
        ),
    )


@pytest.mark.parametrize("bp", [False, True])
@GPU
def test_each_32_elements_has_an_independent_scale(bp):
    # All four groups within K=128 deliberately have different A/B scales.
    # Broadcasting one block128 scale cannot reproduce this exact result.
    from aiter.tuned_gemm import tgemm

    a = torch.ones(2, 128, device="cuda").to(torch.float8_e4m3fn)
    w = torch.ones(32, 128, device="cuda").to(a.dtype)
    sa = torch.tensor(
        [[127, 128, 129, 130], [130, 129, 128, 127]], device="cuda", dtype=torch.uint8
    )
    sb = torch.tensor([127, 129, 128, 130], device="cuda", dtype=torch.uint8)
    sb = sb.expand(32, -1).contiguous()
    # 32*(1*1 + 2*4 + 4*2 + 8*8), and reversed A group scales.
    ref = torch.tensor([2592, 1152], device="cuda", dtype=torch.bfloat16)
    ref = ref[:, None].expand(2, 32)
    y = tgemm.mm(
        a,
        shuffle_weight(w) if bp else w,
        scale_a=sa.view(torch.float8_e8m0fnu),
        scale_b=sb.view(torch.float8_e8m0fnu),
        bpreshuffle=bp,
    )
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


@pytest.mark.parametrize("a_preshuffle", [None, False, True])
def test_public_gfx1250_dispatch_preserves_asm_abi(monkeypatch, a_preshuffle):
    from aiter.ops import gemm_op_a8w8 as op
    from aiter.ops import gemm_op_mxfp8 as mx

    monkeypatch.setattr(op, "_mxfp8_arch", lambda: "gfx1250")
    calls = []
    monkeypatch.setattr(op, "_mxfp8_mxfp8_gemm_asm", lambda *args: calls.append(args))

    def no_gfx950(*args, **kwargs):
        raise AssertionError("gfx1250 must not use the gfx950 tuned table")

    monkeypatch.setattr(mx, "get_mxfp8_config", no_gfx950)
    a = torch.empty((2, 128), device="cpu", dtype=torch.float8_e4m3fn)
    b = torch.empty((16, 128), device="cpu", dtype=a.dtype)
    sa = torch.empty((2, 4), device="cpu", dtype=torch.uint8)
    sb = torch.empty((16, 4), device="cpu", dtype=torch.uint8)
    kwargs = {} if a_preshuffle is None else {"a_preshuffle": a_preshuffle}
    y = op.gemm_a8w8_mxfp8(a, b, sa, sb, kernelName="asm_kernel", **kwargs)
    assert y.shape == (2, 16) and y.dtype == torch.bfloat16
    assert len(calls) == 1
    assert all(got is original for got, original in zip(calls[0][:4], [a, b, sa, sb]))
    assert calls[0][4] is y
    assert calls[0][5:] == ("asm_kernel", int(a_preshuffle is not False))
    with pytest.raises(ValueError, match="gfx950"):
        op.gemm_a8w8_mxfp8(a, b, sa, sb, kernelName="flydsl_mxfp8_invalid")
    with pytest.raises(NotImplementedError, match="gfx1250"):
        op.gemm_a8w8_mxfp8(a, b, sa, sb, bpreshuffle=False)


def test_public_rejects_unsupported_arch(monkeypatch):
    from aiter.ops import gemm_op_a8w8 as op

    monkeypatch.setattr(op, "_mxfp8_arch", lambda: "gfx942")
    x = torch.empty((1, 128), device="cpu")
    with pytest.raises(NotImplementedError, match="gfx942"):
        op.gemm_a8w8_mxfp8(x, x, x, x)


@pytest.mark.parametrize("bp,direct_b", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("use_out", [False, True])
@GPU
def test_public_gfx950_eager_compile_from_config(
    bp, direct_b, use_out, tmp_path, monkeypatch, request
):
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_mxfp8

    a, w, sa, sb, b, ref = inputs(m=17, n=128, k=7168, bias=True, bp=bp)
    out = torch.empty_like(ref)
    cfg = dict(DEFAULT_CONFIG, split_k=7, direct_b=direct_b)
    name = flydsl_mxfp8_kernel_name(cfg, has_bias=True, bpreshuffle=bp)
    from types import SimpleNamespace

    from aiter.ops import gemm_op_mxfp8 as mx

    # No launch knobs on the public call: the chosen split comes from CSV.
    row = dict(
        zip(MXFP8_KEYS, ["gfx950", 256, 17, 128, 7168, "torch.bfloat16", True, bp])
    )
    row.update(libtype="flydsl", kernelName=name, splitK=7)
    path = tmp_path / "tuned.csv"
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    monkeypatch.setattr(
        mx, "AITER_CONFIGS", SimpleNamespace(AITER_CONFIG_GEMM_MXFP8_FILE=str(path))
    )
    mx.get_mxfp8_config.cache_clear()
    request.addfinalizer(mx.get_mxfp8_config.cache_clear)
    request.addfinalizer(mx._load_mxfp8_configs.cache_clear)
    assert mx.get_mxfp8_config(17, 128, 7168, torch.bfloat16, True, bp) == cfg

    def run(a, w, sa, sb, b, out):
        return gemm_a8w8_mxfp8(
            a,
            w,
            sa,
            sb,
            bias=b,
            bpreshuffle=bp,
            out=out if use_out else None,
        )

    for fn in (run, torch.compile(run, fullgraph=True)):
        y = fn(
            a, w, sa.view(torch.float8_e8m0fnu), sb.view(torch.float8_e8m0fnu), b, out
        )
        torch.testing.assert_close(y, ref, atol=0.1, rtol=0.03)
        if use_out:
            assert y.data_ptr() == out.data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(a, w, sa, sb, b, out)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y = run(a, w, sa, sb, b, out)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(y, ref, atol=0.1, rtol=0.03)
    with pytest.raises(ValueError, match="unshuffled A"):
        gemm_a8w8_mxfp8(a, w, sa, sb, a_preshuffle=True)
    with pytest.raises(ValueError, match="kernelName"):
        gemm_a8w8_mxfp8(a, w, sa, sb, kernelName="gfx1250_asm")
    with pytest.raises(TypeError, match="splitK"):
        gemm_a8w8_mxfp8(a, w, sa, sb, splitK=7)


def test_minimax_shapes_only():
    source = (
        ROOT / "aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_minimax_m3.csv"
    )
    target = ROOT / "aiter/configs/model_configs/mxfp8_untuned_gemm_minimax_m3.csv"

    def shapes(path):
        with path.open() as f:
            return {
                tuple(int(r[k]) for k in ("M", "N", "K")) for r in csv.DictReader(f)
            }

    assert len(shapes(target)) == 100
    assert shapes(source) == shapes(target)
    assert not list((ROOT / "aiter/configs/model_configs").glob("dsv4_mxfp8*"))


def test_public_config_lookup_has_no_split_override():
    import inspect

    from aiter.ops import gemm_op_mxfp8 as mx
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_mxfp8

    assert list(inspect.signature(mx.get_mxfp8_config).parameters) == [
        "m",
        "n",
        "k",
        "out_dtype",
        "has_bias",
        "bpreshuffle",
    ]
    for fn in (mx.gemm_mxfp8, gemm_a8w8_mxfp8):
        assert "split_k" not in inspect.signature(fn).parameters
        assert "splitK" not in inspect.signature(fn).parameters


def test_default_config_uses_minimax_without_generic_table(monkeypatch):
    import pandas as pd

    from aiter.jit.core import AITER_CONFIGS

    monkeypatch.delenv("AITER_CONFIG_GEMM_MXFP8", raising=False)
    AITER_CONFIGS.get_config_file.cache_clear()
    try:
        path = Path(AITER_CONFIGS.AITER_CONFIG_GEMM_MXFP8_FILE)
        assert (
            path == ROOT / "aiter/configs/model_configs/mxfp8_tuned_gemm_minimax_m3.csv"
        )
        assert not (ROOT / "aiter/configs/mxfp8_tuned_gemm.csv").exists()
        assert not (ROOT / "aiter/configs/mxfp8_untuned_gemm.csv").exists()
        df = pd.read_csv(path)
        assert len(df) == 100 and not df.duplicated(MXFP8_KEYS).any()
    finally:
        AITER_CONFIGS.get_config_file.cache_clear()


def test_mxfp8_merge_uses_model_shape_keys(tmp_path, monkeypatch):
    import pandas as pd

    from aiter.jit import core

    models = tmp_path / "aiter/configs/model_configs"
    models.mkdir(parents=True)
    schema = models / "mxfp8_untuned_gemm_minimax_m3.csv"
    schema.write_text("M,N,K,outdtype,bias,bpreshuffle\n")
    monkeypatch.setattr(core, "AITER_ROOT_DIR", str(tmp_path))
    row = dict(
        zip(MXFP8_KEYS, ["gfx950", 256, 16, 64, 128, "torch.bfloat16", False, True])
    )
    a, b = models / "a.csv", models / "b.csv"
    pd.DataFrame([dict(row, us=2.0, kernelName="first")]).to_csv(a, index=False)
    pd.DataFrame([dict(row, us=3.0, kernelName="second")]).to_csv(b, index=False)
    # Different timings/names must not conceal a duplicate shape.
    with pytest.raises(RuntimeError, match="duplicate shape"):
        core.AITER_CONFIGS.update_config_files(
            os.pathsep.join([str(a), str(b)]), "mxfp8_tuned_gemm"
        )
    assert len(pd.read_csv(a)) == 1
    assert pd.read_csv(b).empty


@pytest.mark.parametrize("policy", ["ft", "split", "slice", "split_slice"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@GPU
def test_direct_b_and_lds_b_same_config(policy, dtype):
    a, w, sa, sb, bias, ref = inputs(dtype=dtype, bias=True, bp=True)
    names, signatures = set(), set()
    from aiter.ops.flydsl.gemm_mxfp8 import mxfp8_kernel_config
    from aiter.ops.flydsl.kernels.scaled_gemm_gfx950 import (
        make_scaled_gemm_param_and_validate,
    )

    for direct_b in (False, True):
        cfg = dict(CONFIGS[policy], direct_b=direct_b)
        name = flydsl_mxfp8_kernel_name(
            cfg, out_dtype=dtype, has_bias=True, bpreshuffle=True
        )
        names.add(name)
        parsed = get_flydsl_mxfp8_kernel_params(name, cfg["split_k"])
        assert {k: parsed[k] for k in CONFIG_KEYS} == cfg
        param = make_scaled_gemm_param_and_validate(
            17, 128, 7168, mxfp8_kernel_config(cfg, dtype, True, True)
        )
        assert param is not None and param.direct_b == direct_b
        signatures.add(param.__cache_signature__())
        out = torch.full_like(ref, float("nan"))
        y = flydsl_mxfp8_gemm(
            a, w, sa, sb, out=out, bias=bias, config=cfg, bpreshuffle=True
        )
        assert y.data_ptr() == out.data_ptr()
        torch.testing.assert_close(y, ref, atol=0.1, rtol=0.03)
    assert len(names) == len(signatures) == 2


@GPU
def test_b_path_space_and_pruning():
    from aiter.ops.flydsl.gemm_mxfp8 import (
        get_flydsl_mxfp8_configs,
        mxfp8_kernel_config,
    )
    from aiter.ops.flydsl.kernels.scaled_gemm_gfx950 import (
        make_scaled_gemm_param_and_validate,
    )
    from csrc.gemm_mxfp8.gemm_mxfp8_tune import make_tasks, screen_tasks

    configs = get_flydsl_mxfp8_configs(32, 2304, 6144, bpreshuffle=True)
    paired = {}
    for c in configs:
        key = tuple((k, v) for k, v in c.items() if k != "direct_b")
        paired.setdefault(key, set()).add(c["direct_b"])
        if c["use_half_tile_interleaved"]:
            assert not c["direct_b"]
        # If a retained direct tile also fits LDS, pruner must retain BOTH.
        if c["direct_b"]:
            other = dict(c, direct_b=False)
            if (
                make_scaled_gemm_param_and_validate(
                    32,
                    2304,
                    6144,
                    mxfp8_kernel_config(other, torch.bfloat16, False, True),
                )
                is not None
            ):
                assert other in configs
    assert any(v == {False, True} for v in paired.values())
    assert not any(c["direct_b"] for c in get_flydsl_mxfp8_configs(16, 64, 128))
    row = dict(
        zip(MXFP8_KEYS, ["gfx950", 256, 16, 64, 128, "torch.bfloat16", False, True])
    )
    tasks = make_tasks(row, {})
    finalists = screen_tasks(tasks, 1)
    assert {t[4][1]["direct_b"] for t in finalists} == {False, True}


def test_direct_b_rejects_unsupported_combinations():
    from aiter.ops.flydsl.kernels.scaled_gemm_gfx950 import (
        make_scaled_gemm_gfx950_param,
    )

    for kwargs in (
        {"bpreshuffle": False},
        {"bpreshuffle": True, "use_half_tile_interleaved": True},
        {"bpreshuffle": True, "mma_m": 32, "mma_n": 32, "mma_k": 64},
    ):
        with pytest.raises(ValueError, match="direct_b requires"):
            make_scaled_gemm_gfx950_param(direct_b=True, **kwargs)


@pytest.mark.parametrize("direct_b", [False, True])
@pytest.mark.parametrize(
    "m,n,k,config",
    [
        (1, 80, 1152, dict(DEFAULT_CONFIG, stages=4, split_k=3)),
        (65, 144, 1536, dict(DEFAULT_CONFIG, stages=3, split_k=3, group_m=4)),
        (33, 128, 1280, dict(DEFAULT_CONFIG, block_k=256, k_waves=2)),
    ],
)
@GPU
def test_pipeline_boundaries(direct_b, m, n, k, config):
    a, w, sa, sb, _b, ref = inputs(m, n, k, bp=True)
    out = torch.full_like(ref, float("nan"))
    y = flydsl_mxfp8_gemm(
        a, w, sa, sb, out=out, config=dict(config, direct_b=direct_b), bpreshuffle=True
    )
    torch.testing.assert_close(y, ref, atol=0.1, rtol=0.03)


@pytest.mark.parametrize("k", [256, 768, 1536, 2560])
@GPU
def test_hti_scale_chunk_wrap(k):
    a, w, sa, sb, _b, ref = inputs(129, 256, k, bp=True)
    y = flydsl_mxfp8_gemm(a, w, sa, sb, config=CONFIGS["hti"], bpreshuffle=True)
    torch.testing.assert_close(y, ref, atol=0.1, rtol=0.03)


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@GPU
def test_low_level_data_layouts(layout):
    from aiter.ops.flydsl.kernels.scaled_gemm_gfx950 import scaled_gemm

    a, w, sa, sb, b, ref = inputs(32, 64, 1024, dtype=torch.float32, bias=True)
    a = a.t().contiguous().t() if layout[0] == "t" else a
    bt = w.t() if layout[1] == "t" else w.t().contiguous()
    y = scaled_gemm(
        a,
        bt,
        sa,
        sb,
        bias=b,
        out_dtype=torch.float32,
        layout=layout,
        user_kwargs=dict(DEFAULT_CONFIG, block_k=256, k_waves=2, split_k=2),
    )
    torch.testing.assert_close(y, ref, atol=0.1, rtol=0.03)


@GPU
def test_batched_noncontiguous_tgemm():
    from aiter.tuned_gemm import tgemm

    a, w, sa, sb, b, ref = inputs(6, 64, 256, bias=True, bp=True)
    a = a.reshape(2, 3, 256).transpose(0, 1)
    sa = sa.reshape(2, 3, 8).transpose(0, 1)
    expected = ref.reshape(2, 3, 64).transpose(0, 1)
    y = tgemm.mm(a, w, b, scale_a=sa, scale_b=sb, bpreshuffle=True)
    torch.testing.assert_close(y, expected, atol=0.1, rtol=0.03)


@pytest.mark.parametrize(
    "case",
    [
        "rank",
        "k",
        "empty",
        "dtype",
        "out_shape",
        "out_dtype",
        "out_stride",
        "bias_shape",
        "cpu_scale",
    ],
)
@GPU
def test_invalid_public_operands(case):
    from aiter import gemm_a8w8_mxfp8

    a, w, sa, sb, _b, ref = inputs(16, 64, 256, bp=True)
    kw = {}
    if case == "rank":
        a = a[0]
    elif case == "k":
        w = w[:, :128]
    elif case == "empty":
        a, sa = a[:0], sa[:0]
    elif case == "dtype":
        a = a.float()
    elif case == "out_shape":
        kw["out"] = torch.empty(64, 16, device="cuda", dtype=ref.dtype)
    elif case == "out_dtype":
        kw["out"] = ref.float()
    elif case == "out_stride":
        kw["out"] = ref.t().contiguous().t()
    elif case == "bias_shape":
        kw["bias"] = torch.empty(64, 2, device="cuda", dtype=ref.dtype)
    elif case == "cpu_scale":
        sa = sa.cpu()
    with pytest.raises(ValueError):
        gemm_a8w8_mxfp8(a, w, sa, sb, **kw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("splitK", 1),
        ("outdtype", "torch.float32"),
        ("bias", True),
        ("bpreshuffle", False),
        ("gfx", "gfx1250"),
    ],
)
def test_aot_rejects_inconsistent_config(tmp_path, field, value):
    from aiter.aot.flydsl.gemm import parse_csv

    row = dict(
        zip(MXFP8_KEYS, ["gfx950", 256, 16, 64, 256, "torch.bfloat16", False, True])
    )
    row.update(
        libtype="flydsl",
        splitK=2,
        kernelName=flydsl_mxfp8_kernel_name(
            dict(DEFAULT_CONFIG, split_k=2), bpreshuffle=True
        ),
    )
    row[field] = value
    path = tmp_path / "bad.csv"
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="MXFP8"):
        parse_csv(path)


@GPU
def test_runtime_invalid_config_falls_back(tmp_path, monkeypatch, request):
    from types import SimpleNamespace

    from aiter.ops import gemm_op_mxfp8 as mx

    row = dict(
        zip(MXFP8_KEYS, ["gfx950", 256, 16, 64, 256, "torch.bfloat16", False, True])
    )
    row.update(
        libtype="flydsl",
        splitK=1,
        kernelName=flydsl_mxfp8_kernel_name(
            dict(DEFAULT_CONFIG, split_k=2, direct_b=True), bpreshuffle=True
        ),
    )
    path = tmp_path / "bad.csv"
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    monkeypatch.setattr(
        mx, "AITER_CONFIGS", SimpleNamespace(AITER_CONFIG_GEMM_MXFP8_FILE=str(path))
    )
    mx.get_mxfp8_config.cache_clear()
    request.addfinalizer(mx.get_mxfp8_config.cache_clear)
    request.addfinalizer(mx._load_mxfp8_configs.cache_clear)
    assert (
        mx.get_mxfp8_config(16, 64, 256, torch.bfloat16, False, True) == DEFAULT_CONFIG
    )
    a, w, sa, sb, _b, ref = inputs(16, 64, 256, bp=True)
    torch.testing.assert_close(
        mx.gemm_mxfp8(a, w, sa, sb, bpreshuffle=True), ref, atol=0.1, rtol=0.03
    )


@pytest.mark.parametrize("split_k", [1, 7])
@pytest.mark.parametrize("direct_b", [False, True])
@GPU
def test_fp32_and_bias_exact(split_k, direct_b):
    a = torch.ones((1, 896), device="cuda").to(torch.float8_e4m3fn)
    w = torch.ones((64, 896), device="cuda").to(a.dtype)
    sa = torch.full((1, 28), 127, device="cuda", dtype=torch.uint8)
    sb = torch.full((64, 28), 127, device="cuda", dtype=torch.uint8)
    bias = torch.arange(64, device="cuda", dtype=torch.float32) / 32
    y = flydsl_mxfp8_gemm(
        a,
        shuffle_weight(w),
        sa,
        sb,
        bias=bias,
        out_dtype=torch.float32,
        bpreshuffle=True,
        config=dict(DEFAULT_CONFIG, split_k=split_k, direct_b=direct_b),
    )
    torch.testing.assert_close(y, 896 + bias[None, :], atol=0, rtol=0)


@pytest.mark.parametrize(
    "config",
    [
        dict(DEFAULT_CONFIG, split_k=0),
        dict(DEFAULT_CONFIG, stages=1),
        dict(DEFAULT_CONFIG, block_k=0),
        dict(DEFAULT_CONFIG, split_k=3, stages=4),
    ],
)
@GPU
def test_invalid_policy_is_rejected_before_launch(config):
    a, w, sa, sb, _b, _ref = inputs(16, 64, 768, bp=True)
    with pytest.raises(ValueError, match="shape/config"):
        flydsl_mxfp8_gemm(a, w, sa, sb, config=config, bpreshuffle=True)


@pytest.mark.parametrize("direct_b", [False, True])
@GPU
def test_offset_buffers_preserve_guards(direct_b):
    a, w, sa, sb, bias, ref = inputs(17, 80, 1024, bias=True, bp=True)
    # Contiguous scale views with deliberately unaligned byte offsets must
    # be materialized before dword-to-LDS loads. Bias is non-contiguous.
    sa_storage = torch.zeros(sa.numel() + 1, device="cuda", dtype=torch.uint8)
    sa_storage[1:].copy_(sa.flatten())
    sx = sa_storage[1:].view_as(sa)
    bias_storage = torch.zeros(160, device="cuda", dtype=ref.dtype)
    bias_storage[::2].copy_(bias)
    guard = 32
    storage = torch.full((ref.numel() + 2 * guard,), 17, device="cuda", dtype=ref.dtype)
    out = storage[guard:-guard].view_as(ref)
    cfg = dict(DEFAULT_CONFIG, split_k=2, direct_b=direct_b)
    flydsl_mxfp8_gemm(
        a, w, sx, sb, out=out, bias=bias_storage[::2], config=cfg, bpreshuffle=True
    )
    torch.testing.assert_close(out, ref, atol=0.1, rtol=0.03)
    assert torch.equal(storage[:guard], torch.full_like(storage[:guard], 17))
    assert torch.equal(storage[-guard:], torch.full_like(storage[-guard:], 17))

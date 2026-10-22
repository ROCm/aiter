# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import functools
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import (
    get_cdna4_num_xcds,
    get_cu_num,
    get_gfx_runtime,
)
from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_4wave import (
    default_4wave_kernel_name,
    make_4wave_kernel_name,
    parse_4wave_kernel_name,
)
from aiter.ops.gemm_op_a8w8 import get_GEMM_config_with_quant_type
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest


def _is_gfx950() -> bool:
    try:
        return torch.cuda.is_available() and get_gfx_runtime() == "gfx950"
    except (KeyError, RuntimeError):
        return False


requires_gfx950 = pytest.mark.skipif(not _is_gfx950(), reason="gfx950 FlyDSL kernel")

TUNED_SHAPES = (
    (9450, 5120, 5120),
    (9450, 13824, 5120),
    (9450, 5120, 13824),
)


@functools.lru_cache(maxsize=4)
def _make_case(m, n, k, seed=20261022):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randint(
        -31, 32, (m, k), dtype=torch.int8, device="cuda", generator=generator
    )
    weight = torch.randint(
        -31, 32, (n, k), dtype=torch.int8, device="cuda", generator=generator
    )
    x_scale = torch.rand(
        (m, 1), dtype=torch.float32, device="cuda", generator=generator
    )
    w_scale = torch.rand(
        (1, n), dtype=torch.float32, device="cuda", generator=generator
    )
    return {
        "x": x,
        "weight": weight,
        "weight_shuffled": shuffle_weight(weight, layout=(16, 16)).contiguous(),
        "x_scale": x_scale,
        "w_scale": w_scale,
    }


def _public(case, *, bias=None):
    return aiter.gemm_a8w8_bpreshuffle(
        case["x"],
        case["weight_shuffled"],
        case["x_scale"],
        case["w_scale"],
        bias=bias,
    )


def _gluon(case):
    from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8 as gluon_gemm_a8w8

    m = case["x"].shape[0]
    n = case["weight"].shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    return gluon_gemm_a8w8(
        case["x"],
        case["weight"],
        case["x_scale"],
        case["w_scale"],
        None,
        torch.bfloat16,
        out,
        backend="gluon",
    )


@benchmark()
def test_gemm_a8w8_int8_4wave_benchmark(dtype, batch, M, N, K):
    if dtype != dtypes.bf16:
        raise ValueError(f"4-wave INT8 GEMM supports only BF16 output, got {dtype}")
    if batch != 1:
        raise ValueError(f"4-wave INT8 GEMM is unbatched, got batch={batch}")

    case = _make_case(M, N, K, seed=M + N + K)
    reference = _gluon(case)
    candidates = {
        "public_4wave": lambda: _public(case),
        "gluon": lambda: _gluon(case),
    }
    flops = 2 * M * N * K
    nbytes = (
        case["x"].nbytes
        + case["weight"].nbytes
        + case["x_scale"].nbytes
        + case["w_scale"].nbytes
        + M * N * dtype.itemsize
    )

    result = {"gfx": get_gfx_runtime(), "cu_num": get_cu_num()}
    for name, candidate in candidates.items():
        actual, us = run_perftest(candidate)
        err = checkAllclose(
            reference,
            actual,
            rtol=0,
            atol=0,
            msg=f"{name}: 4-wave INT8",
        )
        result[f"{name} us"] = us
        result[f"{name} TFLOPS"] = flops / us / 1e6
        result[f"{name} TB/s"] = nbytes / us / 1e6
        result[f"{name} err"] = err
    return result


test_gemm_a8w8_int8_4wave_benchmark.__test__ = False


def _run_4wave(x, weight_shuffled, x_scale, w_scale, out, num_cu, num_xcds=None):
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.gemm_a8w8_int8_4wave import compile_int8_gemm_4w
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    m, k = x.shape
    n = weight_shuffled.shape[0]
    exe = compile_int8_gemm_4w(
        K=k,
        xcd_swizzle=8,
        num_cu=num_cu,
        num_xcds=num_xcds,
    )
    _run_compiled(
        exe,
        x.view(-1),
        weight_shuffled.view(-1),
        out.view(-1),
        x_scale.view(-1),
        w_scale.view(-1),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )


@pytest.mark.parametrize("cu_num,num_xcds", [(32, 1), (64, 2), (128, 4), (256, 8)])
def test_cdna4_cu_to_xcd_mapping(cu_num, num_xcds):
    assert get_cdna4_num_xcds(cu_num) == num_xcds


@pytest.mark.parametrize("cu_num", [0, 96, 160, 288])
def test_cdna4_cu_to_xcd_mapping_rejects_invalid_counts(cu_num):
    with pytest.raises(ValueError, match="CDNA4"):
        get_cdna4_num_xcds(cu_num)


@pytest.mark.parametrize(
    "config",
    [
        {"K": 5120, "xcd_swizzle": 8, "num_cu": 256, "num_xcds": 8},
        {"K": 5120, "xcd_swizzle": 8, "num_cu": 128, "num_xcds": 4},
        {"K": 13824, "xcd_swizzle": 8, "num_cu": 0, "num_xcds": 8},
        {"K": 13824, "xcd_swizzle": 0, "num_cu": 0, "num_xcds": 4},
    ],
)
def test_4wave_kernel_name_roundtrip(config):
    name = make_4wave_kernel_name(**config)
    parsed = parse_4wave_kernel_name(name)
    assert parsed is not None
    assert parsed["k"] == config["K"]
    for key in ("xcd_swizzle", "num_cu", "num_xcds"):
        assert parsed[key] == config[key]
    assert (parsed["q_dtype_a"], parsed["q_dtype_w"], parsed["dtype"]) == (
        "I8",
        "I8",
        "B16",
    )


def test_4wave_default_persistent_policy():
    persistent = default_4wave_kernel_name(9450, 5120, 5120, cu_num=256, num_xcds=8)
    assert "_p256" in persistent
    assert "_xcd" not in persistent

    mi350p = default_4wave_kernel_name(9450, 5120, 5120, cu_num=128, num_xcds=4)
    assert "_p128_xcd4" in mi350p

    small_grid = default_4wave_kernel_name(512, 5120, 5120, cu_num=128, num_xcds=4)
    assert "_p" not in small_grid
    assert small_grid.endswith("_xcd4")

    long_k = default_4wave_kernel_name(9450, 5120, 13824, cu_num=256, num_xcds=8)
    assert "_p" not in long_k
    for m in (9472, 18900, 37800, 75600):
        assert "_p256" in default_4wave_kernel_name(
            m, 5120, 5120, cu_num=256, num_xcds=8
        )


@requires_gfx950
@pytest.mark.parametrize(
    "m,n,k",
    [
        # 2..9 K tiles: no loop trip, the loop with both tail parities.
        (256, 256, 256),
        (256, 256, 384),
        (256, 256, 512),
        (256, 256, 640),
        (256, 256, 1024),
        (256, 256, 1152),
        # M/N tails of the 256x256 workgroup tile.
        (1, 16, 256),
        (131, 272, 384),
        (513, 784, 1152),
        (1000, 1008, 5120),
        # More blocks than CUs.
        (4352, 4608, 1024),
    ],
)
# 0: one block per workgroup; 8: persistent workgroups take several blocks
# each on the small shapes; None: persistent on all CUs.
@pytest.mark.parametrize("num_cu", [0, 8, None])
@torch.no_grad()
def test_flydsl_int8_4wave_bit_exact(m, n, k, num_cu):
    if num_cu is None:
        num_cu = get_cu_num()
    generator = torch.Generator(device="cuda").manual_seed(m * 7 + n * 3 + k)
    x = torch.randint(
        -31, 32, (m, k), dtype=torch.int8, device="cuda", generator=generator
    )
    weight = torch.randint(
        -31, 32, (n, k), dtype=torch.int8, device="cuda", generator=generator
    )
    x_scale = torch.rand(
        (m, 1), dtype=torch.float32, device="cuda", generator=generator
    )
    w_scale = torch.rand(
        (1, n), dtype=torch.float32, device="cuda", generator=generator
    )
    weight_shuffled = shuffle_weight(weight, layout=(16, 16)).contiguous()
    actual = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")

    _run_4wave(x, weight_shuffled, x_scale, w_scale, actual, num_cu)

    # Small values keep every FP32 product and sum exact, so the kernel's
    # single post-reduction scale product must match bit for bit.
    accum = torch.nn.functional.linear(x.float(), weight.float())
    expected = (accum * (x_scale * w_scale)).to(torch.bfloat16)
    assert torch.equal(actual, expected)


@requires_gfx950
@torch.no_grad()
def test_4wave_explicit_mi350p_mapping_is_bit_exact():
    m, n, k = 513, 784, 512
    generator = torch.Generator(device="cuda").manual_seed(350)
    x = torch.randint(
        -31, 32, (m, k), dtype=torch.int8, device="cuda", generator=generator
    )
    weight = torch.randint(
        -31, 32, (n, k), dtype=torch.int8, device="cuda", generator=generator
    )
    x_scale = torch.rand(
        (m, 1), dtype=torch.float32, device="cuda", generator=generator
    )
    w_scale = torch.rand(
        (1, n), dtype=torch.float32, device="cuda", generator=generator
    )
    actual = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    _run_4wave(
        x,
        shuffle_weight(weight, layout=(16, 16)).contiguous(),
        x_scale,
        w_scale,
        actual,
        num_cu=128,
        num_xcds=4,
    )
    expected = (
        torch.nn.functional.linear(x.float(), weight.float()) * (x_scale * w_scale)
    ).to(torch.bfloat16)
    assert torch.equal(actual, expected)


def test_4wave_aot_csv_parse(tmp_path):
    from aiter.aot.flydsl.gemm import parse_csv

    kernel_name = make_4wave_kernel_name(K=5120, xcd_swizzle=8, num_cu=128, num_xcds=4)
    csv_path = tmp_path / "int8.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,q_dtype_w,libtype,kernelId,splitK,us,"
        "kernelName,tflops,bw,errRatio\n"
        f"gfx950,128,9450,5120,5120,torch.int8,flydsl,1000040,0,0,"
        f"{kernel_name},0,0,0\n"
    )
    jobs = parse_csv(str(csv_path))
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "4wave"
    assert jobs[0]["kernel_name"] == kernel_name
    assert jobs[0]["num_cu"] == 128
    assert jobs[0]["num_xcds"] == 4


@requires_gfx950
@torch.no_grad()
def test_public_int8_fallback_and_post_gemm_bias():
    m, n, k = 257, 272, 384
    generator = torch.Generator(device="cuda").manual_seed(20261022)
    x = torch.randint(
        -31, 32, (m, k), dtype=torch.int8, device="cuda", generator=generator
    )
    weight = torch.randint(
        -31, 32, (n, k), dtype=torch.int8, device="cuda", generator=generator
    )
    x_scale = torch.rand(
        (m, 1), dtype=torch.float32, device="cuda", generator=generator
    )
    w_scale = torch.rand(
        (1, n), dtype=torch.float32, device="cuda", generator=generator
    )
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda", generator=generator)
    weight_shuffled = shuffle_weight(weight, layout=(16, 16)).contiguous()

    actual = aiter.gemm_a8w8_bpreshuffle(
        x, weight_shuffled, x_scale, w_scale, bias=bias
    )
    accum = torch.nn.functional.linear(x.float(), weight.float())
    expected = (accum * (x_scale * w_scale)).to(torch.bfloat16) + bias
    assert torch.equal(actual, expected)


@pytest.fixture
def spy_4wave(monkeypatch):
    import aiter.ops.flydsl.gemm_a8w8_bpreshuffle_4wave as module

    calls = []
    original = module.run_gemm_a8w8_bpreshuffle_4wave

    def spy(XQ, WQ, x_scale, w_scale, Out, kernel_name):
        calls.append(kernel_name)
        return original(XQ, WQ, x_scale, w_scale, Out, kernel_name)

    monkeypatch.setattr(module, "run_gemm_a8w8_bpreshuffle_4wave", spy)
    return calls


@requires_gfx950
@pytest.mark.parametrize("tuned_m,tuned_n,tuned_k", TUNED_SHAPES)
@torch.no_grad()
def test_public_tuned_rows_select_4wave_and_match_gluon(
    spy_4wave, tuned_m, tuned_n, tuned_k
):
    config = get_GEMM_config_with_quant_type(
        tuned_m,
        tuned_n,
        tuned_k,
        torch.int8,
        AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE,
    )
    assert config is not None
    expected_kernel = default_4wave_kernel_name(tuned_m, tuned_n, tuned_k)
    assert config["libtype"] == "flydsl"
    assert config["kernelName"] == expected_kernel

    case = _make_case(tuned_m, tuned_n, tuned_k)
    actual = _public(case)
    assert spy_4wave == [expected_kernel]
    assert torch.equal(actual, _gluon(case))
    assert torch.equal(_public(case), actual)


@requires_gfx950
@pytest.mark.parametrize("m", [512, 9472])
@torch.no_grad()
def test_public_untuned_shapes_use_correct_fallback(spy_4wave, m):
    n = k = 5120
    config = get_GEMM_config_with_quant_type(
        m,
        n,
        k,
        torch.int8,
        AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE,
    )
    assert config is None
    expected_kernel = default_4wave_kernel_name(m, n, k)
    case = _make_case(m, n, k, seed=m)
    actual = _public(case)
    assert spy_4wave == [expected_kernel]
    assert torch.equal(actual, _gluon(case))


@requires_gfx950
@torch.no_grad()
def test_public_torch_compile_fullgraph_matches_eager():
    torch._dynamo.reset()
    case = _make_case(257, 272, 384, seed=41)
    eager = _public(case)

    def run(x, weight, x_scale, w_scale):
        return aiter.gemm_a8w8_bpreshuffle(x, weight, x_scale, w_scale)

    compiled = torch.compile(run, fullgraph=True)
    args = (
        case["x"],
        case["weight_shuffled"],
        case["x_scale"],
        case["w_scale"],
    )
    actual = compiled(*args)
    torch.cuda.synchronize()
    assert torch.equal(actual, eager)
    torch._dynamo.reset()


@requires_gfx950
@torch.no_grad()
def test_public_cuda_graph_capture_replay_and_stable_output():
    case = _make_case(257, 272, 384, seed=42)
    static_x = case["x"].clone()
    static_scale = case["x_scale"].clone()

    def run():
        return aiter.gemm_a8w8_bpreshuffle(
            static_x,
            case["weight_shuffled"],
            static_scale,
            case["w_scale"],
        )

    warm = run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    output_ptr = actual.data_ptr()

    graph.replay()
    torch.cuda.synchronize()
    assert actual.data_ptr() == output_ptr
    assert torch.equal(actual, warm)

    other = _make_case(257, 272, 384, seed=43)
    static_x.copy_(other["x"])
    static_scale.copy_(other["x_scale"])
    graph.replay()
    torch.cuda.synchronize()
    reference = aiter.gemm_a8w8_bpreshuffle(
        other["x"],
        case["weight_shuffled"],
        other["x_scale"],
        case["w_scale"],
    )
    assert actual.data_ptr() == output_ptr
    assert torch.equal(actual, reference)


@requires_gfx950
@torch.no_grad()
def test_public_runs_on_nondefault_stream():
    case = _make_case(257, 272, 384, seed=44)
    expected = _public(case)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        actual = _public(case)
        consumed = actual.float() * 2
    side.synchronize()
    assert torch.equal(actual, expected)
    assert torch.equal(consumed, expected.float() * 2)


@requires_gfx950
@torch.no_grad()
def test_public_handles_zero_rows():
    n, k = 256, 256
    x = torch.empty((0, k), dtype=torch.int8, device="cuda")
    weight = torch.zeros((n, k), dtype=torch.int8, device="cuda")
    actual = aiter.gemm_a8w8_bpreshuffle(
        x,
        shuffle_weight(weight, layout=(16, 16)).contiguous(),
        torch.empty((0, 1), dtype=torch.float32, device="cuda"),
        torch.ones((1, n), dtype=torch.float32, device="cuda"),
    )
    assert actual.shape == (0, n)


@requires_gfx950
@torch.no_grad()
def test_public_signed_extrema_match_reference():
    m, n, k = 17, 32, 256
    x = torch.full((m, k), -128, dtype=torch.int8, device="cuda")
    weight = torch.full((n, k), 127, dtype=torch.int8, device="cuda")
    x_scale = torch.full((m, 1), 1e-3, dtype=torch.float32, device="cuda")
    w_scale = torch.full((1, n), 2e-3, dtype=torch.float32, device="cuda")
    actual = aiter.gemm_a8w8_bpreshuffle(
        x,
        shuffle_weight(weight, layout=(16, 16)).contiguous(),
        x_scale,
        w_scale,
    )
    expected = (
        torch.nn.functional.linear(x.float(), weight.float()) * (x_scale * w_scale)
    ).to(torch.bfloat16)
    assert torch.equal(actual, expected)


@requires_gfx950
@torch.no_grad()
def test_public_rejects_invalid_int8_contracts():
    case = _make_case(256, 256, 256, seed=45)
    with pytest.raises(ValueError, match="BF16 output"):
        aiter.gemm_a8w8_bpreshuffle(
            case["x"],
            case["weight_shuffled"],
            case["x_scale"],
            case["w_scale"],
            dtype=torch.float16,
        )
    with pytest.raises(ValueError, match="FP32"):
        aiter.gemm_a8w8_bpreshuffle(
            case["x"],
            case["weight_shuffled"],
            case["x_scale"].half(),
            case["w_scale"],
        )
    with pytest.raises(ValueError, match="bias"):
        _public(case, bias=torch.zeros(256, dtype=torch.float32, device="cuda"))

    short_x = torch.zeros((256, 128), dtype=torch.int8, device="cuda")
    short_weight = torch.zeros((256, 128), dtype=torch.int8, device="cuda")
    with pytest.raises(ValueError, match="K >= 256"):
        aiter.gemm_a8w8_bpreshuffle(
            short_x,
            shuffle_weight(short_weight, layout=(16, 16)).contiguous(),
            torch.ones((256, 1), dtype=torch.float32, device="cuda"),
            torch.ones((1, 256), dtype=torch.float32, device="cuda"),
        )


@pytest.mark.skipif(
    not _is_gfx950() or torch.cuda.device_count() < 2,
    reason="requires two gfx950 devices",
)
@torch.no_grad()
def test_public_rejects_mixed_devices():
    case = _make_case(256, 256, 256, seed=46)
    with pytest.raises(ValueError, match="share one device"):
        aiter.gemm_a8w8_bpreshuffle(
            case["x"],
            case["weight_shuffled"],
            case["x_scale"].to("cuda:1"),
            case["w_scale"],
        )


@requires_gfx950
@torch.no_grad()
def test_public_rejects_int8_on_non_gfx950(monkeypatch):
    import aiter.ops.gemm_op_a8w8 as operation

    monkeypatch.setattr(operation, "get_gfx", lambda: "gfx942")
    case = _make_case(256, 256, 256, seed=47)
    with pytest.raises(NotImplementedError, match="gfx950"):
        _public(case)


@requires_gfx950
@torch.no_grad()
def test_public_fp8_dispatch_is_unchanged(monkeypatch):
    import aiter.ops.gemm_op_a8w8 as operation

    m, n, k = 17, 32, 256
    x = torch.zeros((m, k), dtype=torch.float8_e4m3fn, device="cuda")
    weight = torch.zeros((n, k), dtype=torch.float8_e4m3fn, device="cuda")
    weight = shuffle_weight(weight, layout=(16, 16)).contiguous()
    x_scale = torch.ones((m, 1), dtype=torch.float32, device="cuda")
    w_scale = torch.ones((1, n), dtype=torch.float32, device="cuda")
    seen = {}

    monkeypatch.setattr(
        operation,
        "get_GEMM_config_with_quant_type",
        lambda *_args, **_kwargs: {"libtype": "ck", "splitK": 0},
    )

    def fake_ck(XQ, WQ, xs, ws, output, split_k):
        seen["dtypes"] = (XQ.dtype, WQ.dtype)
        seen["split_k"] = split_k
        output.zero_()
        return output

    monkeypatch.setattr(operation, "gemm_a8w8_bpreshuffle_ck", fake_ck)
    actual = aiter.gemm_a8w8_bpreshuffle(x, weight, x_scale, w_scale)
    assert seen == {
        "dtypes": (torch.float8_e4m3fn, torch.float8_e4m3fn),
        "split_k": 0,
    }
    assert torch.count_nonzero(actual) == 0


def main() -> int:
    if not _is_gfx950():
        aiter.logger.warning("4-wave INT8 benchmark requires gfx950")
        return 0

    parser = argparse.ArgumentParser(description="Benchmark gfx950 4-wave INT8 GEMM")
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        choices=[dtypes.bf16],
        default=[dtypes.bf16],
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1])
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=list(TUNED_SHAPES),
        help="M,N,K shapes, for example -s 9450,5120,5120",
    )
    args = parser.parse_args()

    rows = [
        test_gemm_a8w8_int8_4wave_benchmark(dtype, batch, M, N, K)
        for dtype, batch, (M, N, K) in itertools.product(
            args.dtype, args.batch, args.mnk
        )
    ]
    aiter.logger.info(
        "4-wave INT8 summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib

import pytest
import torch
import triton

from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
)
from aiter.ops.triton.utils.types import get_fp8_dtypes, str_to_torch_dtype

e5m2_type, e4m3_type = get_fp8_dtypes()

# Twelve representative modes for the N256/K512 and N512/K192 config buckets. Other API modes
# retain their native contract; this is bounded coverage rather than a cartesian
# sweep. Cases use B8, resident E4M3 weights and the separately selected M shape.
NATIVE_VALUE_MODES = [
    {
        "name": "batch_first",
        "transpose_bm_in": False,
        "transpose_bm": False,
        "output": True,
    },
    {"name": "token_first", "transpose_bm_in": True, "transpose_bm": True},
    {"name": "transposed_output", "transpose_bm_in": False, "transpose_bm": True},
    {"name": "transposed_input", "transpose_bm_in": True, "transpose_bm": False},
    {"name": "fp16_output", "dtype": torch.float16},
    {"name": "allocated_output", "output": False},
    {
        "name": "token_first_allocated",
        "transpose_bm_in": True,
        "transpose_bm": True,
        "output": False,
    },
    {"name": "bias", "has_bias": True},
    {"name": "fp16_bias", "dtype": torch.float16, "has_bias": True},
    {"name": "group64", "group_size": 64},
    {"name": "group256", "group_size": 256},
    {
        "name": "group256_token_first_bias",
        "group_size": 256,
        "transpose_bm_in": True,
        "transpose_bm": True,
        "has_bias": True,
    },
]


def run_group_quantized_fp32(
    x, weight, w_scale, group_size=128, bias=None, transpose_bm=True
):
    """Independent FP8/group-quantized oracle for K192/K512 native GEMMs.

    Reproduce the native bias epilogue's intermediate rounding, while retaining
    FP32 group accumulation, scaling and the pre-output-rounding reference.
    """
    assert x.shape[-1] == weight.shape[-1] and x.shape[-1] in (192, 512)
    out = torch.zeros(
        (x.shape[0], x.shape[1], weight.shape[1]), device=x.device, dtype=torch.float32
    )
    dtype_max = torch.finfo(weight.dtype).max
    for first in range(0, x.shape[-1], group_size):
        # The kernel masks the final group with zeroes. Reducing and multiplying
        # only valid values is equivalent, without including storage beyond K.
        last = min(first + group_size, x.shape[-1])
        a = x[..., first:last].float()
        scale = a.abs().amax(-1, keepdim=True).clamp_min(1e-10) * (1.0 / dtype_max)
        quantized = (
            (a * scale.reciprocal()).clamp(-dtype_max, dtype_max).to(weight.dtype)
        )
        out += (
            torch.bmm(
                quantized.float(), weight[..., first:last].float().transpose(1, 2)
            )
            * scale
        )
    out *= w_scale
    if bias is not None:
        out = (out.to(bias.dtype) + bias).float()
    return out.transpose(0, 1) if transpose_bm else out


def check_group_quantized_fp32(
    actual, x, weight, w_scale, group_size=128, bias=None, transpose_bm=True
):
    reference = run_group_quantized_fp32(
        x, weight, w_scale, group_size, bias, transpose_bm
    )
    torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
    nrmse = torch.sqrt(torch.mean((actual.float() - reference).square())) / torch.sqrt(
        torch.mean(reference.square())
    ).clamp_min(1e-20)
    assert nrmse.item() <= 0.01
    return {
        "quantized_nrmse": nrmse.item(),
        "quantized_max_absolute_error": (actual.float() - reference).abs().max().item(),
    }


def generate_batched_gemm_a16w8_inputs(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype | str,
    has_bias: bool,
    output: bool,
    layout: str = "TN",
    transpose_bm: bool = False,
):
    """
    Returns:
        - x: shape (B, M, K)
        - weight: shape (B, N, K)
        - x_scale: shape (B, M, 1)
        - w_scale: shape (B, 1, N)
    """
    torch.manual_seed(0)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]
    if layout[0] == "T":
        x = (torch.rand((B, M, K), dtype=torch.float16, device="cuda") / 10).to(
            torch.bfloat16
        )
    else:
        x = (
            (torch.rand((B, K, M), dtype=torch.float16, device="cuda") / 10)
            .to(torch.bfloat16)
            .permute(0, 2, 1)
        )

    if layout[1] == "N":
        weight = (torch.rand((B, N, K), dtype=torch.float16, device="cuda") / 10).to(
            e4m3_type
        )
    else:
        weight = (
            (torch.rand((B, N, K), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .permute(0, 2, 1)
        )

    w_scale = torch.rand([1], dtype=torch.float32, device="cuda")[0]
    if has_bias:
        bias = torch.rand([B, 1, N], dtype=dtype).cuda() * 10
    else:
        bias = None

    y = None
    if output:
        if transpose_bm:
            y = torch.empty((M, B, N), dtype=dtype, device=x.device)
        else:
            y = torch.empty((B, M, N), dtype=dtype, device=x.device)

    return x, weight, w_scale, bias, y


def prepare_batched_gemm_input(x, transpose_bm_in=False, input_pad=0, pad_value=0):
    """Prepare the actual input layout, optionally retaining padded row strides."""
    assert input_pad >= 0
    if transpose_bm_in:
        x = x.transpose(0, 1).contiguous()
    if input_pad:
        k = x.shape[-1]
        storage = x.new_full((*x.shape[:-1], k + input_pad), pad_value)
        storage[..., :k].copy_(x)
        x = storage[..., :k]
    return x


def generate_native_value_mode_inputs(mode):
    """Shared signed/zero/extreme-group fixture for K192/K512 native modes."""
    assert mode.get("K", 512) in (192, 512)
    x, weight, scale, bias, output = generate_batched_gemm_a16w8_inputs(
        mode.get("B", 8),
        mode.get("M", 48),
        mode.get("N", 256),
        mode.get("K", 512),
        mode.get("dtype", torch.bfloat16),
        mode.get("has_bias", False),
        mode.get("output", True),
        layout=mode.get("layout", "TN"),
        transpose_bm=mode.get("transpose_bm", False),
    )
    x = (x - 0.05).contiguous()
    x[:, 0, :128] = 0
    x[..., 128:256] *= 16
    x[..., 256:384] *= 0.0625
    weight = (weight.float() - 0.05).to(weight.dtype)
    scale.fill_(0.125)
    inputs = prepare_batched_gemm_input(
        x,
        mode.get("transpose_bm_in", False),
        mode.get("input_pad", 0),
        # Mode correctness fixtures poison the unused tail to expose unmasked
        # loads during activation scale computation. Timed fixtures retain zero.
        pad_value=1024,
    )
    return inputs, weight, scale, bias, output


def run_torch(x, weight, w_scale, bias=None, dtype=torch.bfloat16, transpose_bm=True):
    B = x.size(0)
    M = x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=torch.bfloat16, device="cuda")
    w_bf16 = weight.to(torch.bfloat16) * w_scale.to(torch.bfloat16)
    out = torch.bmm(x, w_bf16.transpose(1, 2))
    if bias is not None:
        out = out + bias
    if transpose_bm:
        out = out.transpose(0, 1)
    return out.to(dtype)


def run_triton(
    x,
    weight,
    w_scale,
    group_size=128,
    bias=None,
    dtype=torch.bfloat16,
    y=None,
    transpose_bm=False,
):
    return batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x,
        weight,
        w_scale,
        group_size=group_size,
        bias=bias,
        dtype=dtype,
        YQ=y,
        transpose_bm=transpose_bm,
    )


@pytest.fixture
def gfx950_value_lookup(monkeypatch):
    """Exercise the real loader on CPU, without querying a GPU architecture."""
    from aiter.ops.triton.utils import config_utils, gemm_config_utils
    from aiter.ops.triton.utils._triton import arch_info

    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx950")
    config_utils.load_config_json.cache_clear()
    gemm_config_utils._get_gemm_config_cached.cache_clear()
    yield
    config_utils.load_config_json.cache_clear()
    gemm_config_utils._get_gemm_config_cached.cache_clear()


@pytest.mark.parametrize(
    "m, n, k",
    [(m, 256, 512) for m in range(33, 129)] + [(m, 512, 192) for m in range(33, 65)],
)
def test_native_value_config_bucket(m, n, k, gfx950_value_lookup):
    from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        _get_config,
    )

    config, tuned = _get_config(m, n, k)
    expected = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg",
    }
    assert tuned and config == expected
    config["BLOCK_SIZE_M"] = -1
    fresh, tuned = _get_config(m, n, k)
    assert tuned and fresh == expected and fresh is not config


@pytest.mark.parametrize(
    "m, n, k, bm, bn, warps, waves",
    [
        (32, 256, 512, 32, 128, 8, 2),
        (129, 256, 512, 64, 256, 8, 1),
        (32, 512, 192, 32, 128, 8, 2),
        (65, 512, 192, 64, 256, 8, 1),
    ],
)
def test_native_value_config_neighbors(
    m, n, k, bm, bn, warps, waves, gfx950_value_lookup
):
    from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        _get_config,
    )

    config, tuned = _get_config(m, n, k)
    assert tuned
    assert (
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["num_warps"],
        config["waves_per_eu"],
    ) == (bm, bn, warps, waves)


@pytest.mark.parametrize(
    "m, n, k",
    [(128, 255, 512), (128, 257, 512), (128, 256, 511), (128, 256, 513)]
    + [(64, 511, 192), (64, 513, 192), (64, 512, 191), (64, 512, 193)],
)
def test_native_value_config_exact_nk(m, n, k, gfx950_value_lookup):
    from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

    family = "BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT"
    actual, tuned = get_gemm_config(family, m, n, k)
    default, _ = get_gemm_config(family, m)
    assert not tuned
    assert actual == default


@pytest.mark.parametrize("m, n, k", [(64, 256, 512), (128, 256, 512), (64, 512, 192)])
def test_native_value_config_u02_isolation(m, n, k, gfx950_value_lookup):
    from aiter.ops.triton._triton_kernels.fusions.fused_bmm_rope_kv_cache import (
        _get_fp8_config,
    )
    from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        _get_config,
    )

    fused, tuned = _get_fp8_config(m, n, k)
    native, _ = _get_config(m, n, k)
    assert tuned
    assert (fused["BLOCK_SIZE_M"], fused["BLOCK_SIZE_N"], fused["num_warps"]) == (
        64,
        256,
        8,
    )
    assert fused != native


def test_native_value_explicit_config(monkeypatch, gfx950_value_lookup):
    """The real public wrapper must bypass lookup when an explicit config is supplied."""
    op = importlib.import_module(
        "aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant"
    )
    from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

    # An unrelated shape's real config stands in for a caller-chosen override.
    config, _ = get_gemm_config(
        "BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT",
        48,
        backend="triton",
    )
    observed = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: observed.append(kwargs)

    def fail_lookup(*args):
        raise AssertionError("Explicit config must bypass the loader")

    monkeypatch.setattr(op, "_get_config", fail_lookup)
    monkeypatch.setattr(
        op,
        "_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel",
        Kernel(),
    )
    x = torch.zeros((1, 48, 512), dtype=torch.bfloat16)
    weight = torch.zeros((1, 256, 512), dtype=torch.float8_e4m3fn)
    scale = torch.ones(())
    output = torch.empty((1, 48, 256), dtype=torch.bfloat16)
    assert (
        op.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
            x, weight, scale, YQ=output, config=config
        )
        is output
    )
    assert observed == [{**config, "DTYPE_MAX": 448.0, "DTYPE_MIN": -448.0}]


@pytest.mark.parametrize("group_size", [64, 128, 256])
def test_native_value_oracle_cpu(group_size):
    x = torch.zeros((1, 16, 512), dtype=torch.bfloat16)
    for group, magnitude in enumerate((1, 2, 4, 8)):
        x[..., group * 128 : (group + 1) * 128] = magnitude
    weight = torch.ones((1, 32, 512)).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.25)
    expected = torch.full((1, 16, 32), 480.0)
    torch.testing.assert_close(
        run_group_quantized_fp32(x, weight, scale, group_size, transpose_bm=False),
        expected,
        atol=1e-4,
        rtol=1e-6,
    )
    x.zero_()
    assert (
        torch.count_nonzero(run_group_quantized_fp32(x, weight, scale, group_size)) == 0
    )


@pytest.mark.parametrize("group_size", [64, 128, 256])
def test_native_query_oracle_masked_tail_cpu(group_size):
    # Poison storage beyond valid K: the reference must ignore the masked tail.
    storage = torch.full((1, 16, 256), 1024.0, dtype=torch.bfloat16)
    x = storage[..., :192]
    x[..., :128] = 1
    x[..., 128:] = 2
    weight = torch.ones((1, 32, 192)).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.25)
    expected = torch.full((1, 16, 32), 64.0)
    torch.testing.assert_close(
        run_group_quantized_fp32(x, weight, scale, group_size, transpose_bm=False),
        expected,
        atol=1e-4,
        rtol=1e-6,
    )
    x.zero_()
    assert (
        torch.count_nonzero(run_group_quantized_fp32(x, weight, scale, group_size)) == 0
    )
    assert torch.all(storage[..., 192:] == 1024)


def test_native_query_benchmark_layout_cpu(monkeypatch):
    benchmark = importlib.import_module(
        "op_tests.op_benchmarks.triton.bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant"
    )
    x = torch.ones((8, 64, 192), dtype=torch.bfloat16)
    weight = torch.ones((8, 512, 192)).to(torch.float8_e4m3fn)

    def generate(*args, **kwargs):
        assert kwargs["output"] is False
        return x, weight, torch.ones(()), None, None

    def operation(inputs, *args, **kwargs):
        assert inputs.shape == (64, 8, 192)
        assert inputs.stride() == (8 * 256, 256, 1)
        assert not inputs.is_contiguous()
        assert kwargs["YQ"] is None
        assert kwargs["transpose_bm_in"] and not kwargs["transpose_bm"]
        assert "backend" not in kwargs and "config" not in kwargs
        torch.testing.assert_close(inputs.transpose(0, 1), x)
        return torch.empty((8, 64, 512), dtype=torch.bfloat16)

    def timer(fn, **kwargs):
        assert kwargs == {"warmup": 25, "rep": 100, "return_mode": "median"}
        fn()
        return 0.0123

    monkeypatch.setattr(
        benchmark, "generate_batched_gemm_a8w8_per_token_group_inputs", generate
    )
    monkeypatch.setattr(
        benchmark,
        "batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant",
        operation,
    )
    monkeypatch.setattr(triton.testing, "do_bench", timer)
    assert (
        benchmark.bench_gemm_fn(
            8,
            64,
            512,
            192,
            "time",
            "TN",
            128,
            False,
            False,
            True,
            supplied_output=False,
            input_pad=64,
        )
        == 0.0123
    )


def test_native_value_benchmark_median(monkeypatch):
    """Exercise native layout preparation and timer options without a GPU timer."""
    benchmark = importlib.import_module(
        "op_tests.op_benchmarks.triton.bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant"
    )
    x = torch.zeros((8, 48, 512), dtype=torch.bfloat16)
    weight = torch.zeros((8, 256, 512), dtype=torch.float8_e4m3fn)
    output = torch.empty((48, 8, 256), dtype=torch.bfloat16)
    monkeypatch.setattr(
        benchmark,
        "generate_batched_gemm_a8w8_per_token_group_inputs",
        lambda *args, **kwargs: (x, weight, torch.ones(()), None, output),
    )
    observed = []

    def operation(inputs, *args, **kwargs):
        assert inputs.shape == (48, 8, 512)
        assert inputs.is_contiguous()
        assert kwargs["transpose_bm"] and kwargs["transpose_bm_in"]
        assert "backend" not in kwargs and "config" not in kwargs
        observed.append(1)
        return output

    def timer(fn, **kwargs):
        assert kwargs == {"warmup": 25, "rep": 100, "return_mode": "median"}
        fn()
        return 0.0123

    monkeypatch.setattr(
        benchmark,
        "batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant",
        operation,
    )
    monkeypatch.setattr(triton.testing, "do_bench", timer)
    assert (
        benchmark.bench_gemm_fn(8, 48, 256, 512, "time", "TN", 128, False, True, True)
        == 0.0123
    )
    assert observed == [1]


@pytest.mark.parametrize(
    "m, n, k",
    [
        (48, 256, 512),
        (64, 512, 192),
        (128, 512, 192),
        (256, 512, 192),
        (128, 256, 512),
        (256, 256, 512),
    ],
)
@pytest.mark.parametrize("mode", NATIVE_VALUE_MODES, ids=lambda mode: mode["name"])
def test_native_batched_modes_replay(mode, m, n, k):
    options = {
        "M": m,
        "N": n,
        "K": k,
        "dtype": torch.bfloat16,
        "output": k == 512,
        "has_bias": False,
        "group_size": 128,
        "transpose_bm": k == 512,
        "transpose_bm_in": True,
        "input_pad": 64 if k == 192 else 0,
    }
    options.update({key: value for key, value in mode.items() if key != "name"})
    inputs, weight, scale, bias, output = generate_native_value_mode_inputs(options)
    operands = (inputs, weight, scale) + ((bias,) if bias is not None else ())
    pristine = [tensor.clone() for tensor in operands]

    def check_state(changed_input=None):
        for index, (actual, original) in enumerate(zip(operands, pristine)):
            expected = (
                changed_input if index == 0 and changed_input is not None else original
            )
            assert torch.equal(
                actual.reshape(-1).view(torch.uint8),
                expected.reshape(-1).view(torch.uint8),
            )

    def operation():
        return batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
            inputs,
            weight,
            scale,
            group_size=options["group_size"],
            bias=bias,
            dtype=options["dtype"],
            YQ=output,
            transpose_bm=options["transpose_bm"],
            transpose_bm_in=options["transpose_bm_in"],
        )

    def check(result):
        canonical = inputs.transpose(0, 1) if options["transpose_bm_in"] else inputs
        check_group_quantized_fp32(
            result,
            canonical,
            weight,
            scale,
            options["group_size"],
            bias,
            options["transpose_bm"],
        )

    check(operation())
    check_state()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output = operation()
    check_state()
    inputs.mul_(0.5)
    changed_input = inputs.clone()
    repeated = None
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        check(replay_output)
        check_state(changed_input)
        if repeated is not None:
            assert torch.equal(replay_output, repeated)
        repeated = replay_output.clone()
    inputs.copy_(pristine[0])
    graph.replay()
    torch.cuda.synchronize()
    check(replay_output)
    check_state()


def get_x_vals():

    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
        (1, 8192, 1024),
        (32, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (192, 8192, 1024),
        (256, 8192, 1024),
        (320, 8192, 1024),
        (512, 8192, 1024),
        (1024, 8192, 1024),
        (2048, 8192, 1024),
        (4096, 8192, 1024),
        (8192, 8192, 1024),
        (16384, 8192, 1024),
    ]
    x_vals += [(v**2, 128, 512) for v in range(7)]
    x_vals += [(v**2, 512, 128) for v in range(7)]
    x_vals += [(m, 256, 512) for m in (32, 33, 48, 63, 64, 65)]
    x_vals += [(m, 512, 192) for m in (32, 33, 64, 65, 128, 129, 256, 257)]
    x_vals += [(m, 256, 512) for m in (128, 129, 256, 257)]
    x_vals += [(1, 128, 1)]  # minimal case
    return x_vals


@pytest.mark.parametrize(
    "dtype, b, m, n, k, group_size, has_bias, output, transpose_bm",
    [
        (dtype, b, *shape, group_size, has_bias, output, transpose_bm)
        for output in [True, False]
        for dtype in ["bf16"]
        for b in [16]
        for shape in get_x_vals()
        for group_size in [128]
        for has_bias in [True, False]
        for transpose_bm in [True, False]
    ],
)
def test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
    dtype, b, m, n, k, group_size, has_bias, output, transpose_bm
):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    dtype = str_to_torch_dtype[dtype]
    x, weight, w_scale, bias, y = generate_batched_gemm_a16w8_inputs(
        b, m, n, k, dtype, has_bias, output, transpose_bm=transpose_bm
    )
    a = run_torch(x, weight, w_scale, bias, dtype, transpose_bm)
    b = run_triton(
        x,
        weight,
        w_scale,
        group_size=group_size,
        bias=bias,
        dtype=dtype,
        y=y,
        transpose_bm=transpose_bm,
    )

    triton.testing.assert_close(a, b, atol=0.1, rtol=0.1)

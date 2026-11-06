# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run with: pytest -q op_tests/flydsl_tests/test_hgemm_k_stagger.py."""

import pytest
import torch

pytest.importorskip("flydsl")

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_kernels import (
    flydsl_hgemm,
    flydsl_hgemm_kernel_name,
    get_flydsl_hgemm_kernel_params,
)
from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
    gemm_a16w16,
    make_gemm_a16w16_gfx950_kernel_name,
    make_gemm_a16w16_gfx950_param,
)

FT = dict(
    block_m=32,
    block_n=64,
    block_k=64,
    stages=3,
    split_k=1,
    m_waves=2,
    n_waves=2,
    k_waves=1,
    group_m=0,
    use_half_tile_interleaved=False,
)
HTI = dict(
    block_m=256,
    block_n=256,
    block_k=64,
    stages=2,
    split_k=1,
    m_waves=2,
    n_waves=4,
    k_waves=1,
    group_m=4,
    use_half_tile_interleaved=True,
)
GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950", reason="requires gfx950"
)


def name(config, gfx="gfx950"):
    return flydsl_hgemm_kernel_name(
        dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        config=config,
        has_bias=False,
        target_gfx=gfx,
    )


@pytest.mark.parametrize("stagger", [0, 1, 3, 127])
@pytest.mark.parametrize("nt", [False, True])
def test_kernel_names(stagger, nt):
    config = dict(FT, k_stagger=stagger, b_non_temporal=nt)
    encoded = name(config)
    parsed = get_flydsl_hgemm_kernel_params(encoded)
    assert {key: parsed[key] for key in config} == config
    assert ("_stg" in encoded) == (stagger != 0)
    if stagger == 0:
        assert encoded == name(dict(FT, b_non_temporal=nt))
        assert (
            get_flydsl_hgemm_kernel_params(encoded.replace("_gm0", "_gm0_stg0"))[
                "k_stagger"
            ]
            == 0
        )


@pytest.mark.parametrize("value", [-1, 1.5, True, "3"])
def test_invalid_stagger(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        name(dict(FT, k_stagger=value))
    with pytest.raises(ValueError, match="non-negative integer"):
        make_gemm_a16w16_gfx950_param(2, 2, **dict(FT, k_stagger=value))


@pytest.mark.parametrize(
    "gfx,policy", [("gfx1250", "ft"), ("gfx1250-strict", "ht"), ("gfx950", "sk")]
)
def test_unsupported_name(gfx, policy):
    with pytest.raises(ValueError, match="gfx950"):
        name(dict(FT, policy=policy, k_stagger=1), gfx)
    encoded = (
        name(dict(FT, k_stagger=1))
        .replace("_pft_", f"_p{policy}_")
        .replace("gfx950", gfx)
    )
    assert get_flydsl_hgemm_kernel_params(encoded) is None


@GPU
def test_device_cache_names():
    off = make_gemm_a16w16_gfx950_param(2, 2, **FT)
    one = make_gemm_a16w16_gfx950_param(2, 2, **dict(FT, k_stagger=1))
    three = make_gemm_a16w16_gfx950_param(2, 2, **dict(FT, k_stagger=3))
    names = [make_gemm_a16w16_gfx950_kernel_name(p) for p in (off, one, three)]
    assert len(set(names)) == 3
    assert "_stg" not in names[0]


# Minimum pipeline, odd/non-power-of-two tile counts, full and ragged M/N,
# split-K, slice-K, HTI's two-tile pipeline, and group_m swizzle on/off.
CASES = [
    (19, 120, 128, FT),
    (19, 120, 1088, dict(FT, stages=2, split_k=9)),
    (33, 136, 448, FT),
    (35, 136, 1344, dict(FT, split_k=3)),
    (32, 128, 640, dict(FT, block_k=128, k_waves=2)),
    (65, 264, 768, dict(FT, stages=5, group_m=4)),
    (513, 520, 128, HTI),
    (257, 264, 768, dict(HTI, split_k=3)),
    (512, 512, 640, HTI),
]


@GPU
@pytest.mark.parametrize("m,n,k,config", CASES)
@pytest.mark.parametrize(
    "dtype,out_fp32",
    [(torch.bfloat16, False), (torch.float16, False), (torch.bfloat16, True)],
)
@pytest.mark.parametrize("stagger", [0, 1, 3, 127])
@pytest.mark.parametrize("nt", [False, True])
def test_correctness(m, n, k, config, dtype, out_fp32, stagger, nt):
    if out_fp32 and config["use_half_tile_interleaved"]:
        # 256x256 FP32 C-shuffle exceeds gfx950's 160 KiB LDS.
        config = dict(config, block_m=128, block_n=128)
    torch.manual_seed(123)
    # Binary fractions keep the reference exact, including split/slice
    # partials. A duplicated or omitted tile cannot hide in a loose tolerance.
    a = torch.randint(-2, 3, (m, k), device="cuda").to(dtype) / 8
    b = torch.randint(-2, 3, (n, k), device="cuda").to(dtype) / 8
    bias = torch.randint(-2, 3, (n,), device="cuda").to(dtype) / 8
    otype = torch.float32 if out_fp32 else dtype
    # Guard both sides of a dirty, preallocated contiguous output.
    guard = torch.full((m * n + 32,), 8192, device="cuda", dtype=otype)
    out = guard[16:-16].view(m, n)
    kwargs = {
        key: val for key, val in config.items() if key != "use_half_tile_interleaved"
    }
    kwargs.update(
        policy="hti" if config["use_half_tile_interleaved"] else "ft",
        k_stagger=stagger,
        b_non_temporal=nt,
    )
    ref = (a.float() @ b.float().T + bias.float()).to(otype)
    for _ in range(3):
        out.fill_(8192)
        result = flydsl_hgemm(a, b, out=out, bias=bias, **kwargs)
        torch.testing.assert_close(result, ref, rtol=0, atol=0)
    assert torch.all(guard[:16] == 8192) and torch.all(guard[-16:] == 8192)


@GPU
@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("nt", [False, True])
def test_low_level_layouts(layout, nt):
    torch.manual_seed(17)
    a = torch.randn(32, 448, device="cuda", dtype=torch.float16)
    b = torch.randn(448, 128, device="cuda", dtype=torch.float16)
    if layout[0] == "t":
        a = a.T.contiguous().T
    if layout[1] == "t":
        b = b.T.contiguous().T
    out = gemm_a16w16(
        a, b, layout=layout, user_kwargs=dict(FT, k_stagger=3, b_non_temporal=nt)
    )
    torch.testing.assert_close(
        out, (a.float() @ b.float()).half(), rtol=1e-2, atol=1e-2
    )


@GPU
def test_graph_and_tuned_dispatch():
    from aiter.tuned_gemm import flydsl_gemm

    torch.manual_seed(3)
    a = torch.randn(32, 448, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 448, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(32, 128, device="cuda", dtype=torch.bfloat16)
    config = {"kernelName": name(dict(FT, k_stagger=3, b_non_temporal=True))}

    def launch():
        return flydsl_gemm(a, b, 0, otype=a.dtype, config=config, out=out)

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(5):
        a.normal_()
        out.fill_(8192)
        graph.replay()
        torch.testing.assert_close(
            out, (a.float() @ b.float().T).bfloat16(), rtol=1e-2, atol=1e-2
        )


@GPU
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("policy", ["ft", "hti"])
def test_random_fp32_output(dtype, policy):
    torch.manual_seed(91)
    a = torch.randn(129, 896, device="cuda", dtype=dtype)
    b = torch.randn(264, 896, device="cuda", dtype=dtype)
    bias = torch.randn(264, device="cuda", dtype=dtype)
    config = dict(FT if policy == "ft" else HTI)
    if policy == "hti":
        config.update(block_m=128, block_n=128)
    config.pop("use_half_tile_interleaved")
    ref = a.double() @ b.double().T + bias.double()
    out = flydsl_hgemm(
        a,
        b,
        bias=bias,
        out_dtype=torch.float32,
        policy=policy,
        k_stagger=3,
        **config,
    )
    torch.testing.assert_close(out.double(), ref, rtol=2e-5, atol=2e-4)


@GPU
def test_k_tail_still_rejected():
    a = torch.zeros(32, 136, device="cuda", dtype=torch.bfloat16)
    b = torch.zeros(128, 136, device="cuda", dtype=a.dtype)
    for stagger in (0, 3):
        with pytest.raises(AssertionError, match="K-tail"):
            flydsl_hgemm(a, b, block_k=64, k_stagger=stagger)


def test_tuner_preserves_original_candidates():
    from unittest.mock import patch
    import aiter.ops.flydsl.gemm_a16w16_policy as policy

    # Limit the expensive Cartesian product: exercise the actual augmentation
    # and name round-trip, independently of device occupancy pruning.
    with (
        patch.object(policy, "get_gfx", return_value="gfx950"),
        patch.object(policy.itertools, "product", return_value=[tuple(FT.values())]),
        patch.object(
            policy.GemmConfigPruner, "prune", side_effect=lambda configs: configs
        ),
        patch.object(policy.torch.cuda, "current_device", return_value=0),
        patch.object(policy.torch.cuda, "get_device_properties", return_value=None),
        patch.object(
            policy, "make_gemm_a16w16_param_and_validate", return_value=object()
        ),
    ):
        configs = policy.get_flydsl_a16w16_configs(
            32, 18432, 1536, torch.bfloat16, torch.bfloat16, False
        )
    assert configs == [FT] + [
        dict(FT, k_stagger=stride, b_non_temporal=nt)
        for stride in (0, 1, 3)
        for nt in (False, True)
        if stride or nt
    ]
    for config in configs:
        parsed = get_flydsl_hgemm_kernel_params(name(config))
        assert {key: parsed[key] for key in config} == config


@pytest.mark.parametrize("value", [-1, 1, "true", None])
def test_invalid_b_non_temporal(value):
    with pytest.raises(ValueError, match="must be a bool"):
        name(dict(FT, b_non_temporal=value))
    with pytest.raises(ValueError, match="must be a bool"):
        make_gemm_a16w16_gfx950_param(2, 2, **dict(FT, b_non_temporal=value))


@pytest.mark.parametrize("gfx,policy", [("gfx1250", "ft"), ("gfx950", "sk")])
def test_unsupported_non_temporal(gfx, policy):
    with pytest.raises(ValueError, match="gfx950"):
        name(dict(FT, b_non_temporal=True, policy=policy), gfx)
    encoded = (
        name(dict(FT, b_non_temporal=True))
        .replace("_pft_", f"_p{policy}_")
        .replace("gfx950", gfx)
    )
    assert get_flydsl_hgemm_kernel_params(encoded) is None

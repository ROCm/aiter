# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU checks for config routing, scale contracts and blockscale tuner wiring."""

import importlib
import inspect
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from aiter.ops import gemm_op_a8w8 as ops


@pytest.fixture
def backend():
    pytest.importorskip("flydsl")
    from aiter.ops.flydsl import gemm_a8w8_blockscale

    return gemm_a8w8_blockscale


@pytest.fixture
def catalog(backend):
    from aiter.ops.flydsl.gemm_tune import flydsl_gemm_a8w8_blockscale_common

    return flydsl_gemm_a8w8_blockscale_common


@pytest.fixture
def tuner_module():
    root = Path(__file__).resolve().parents[2]
    with patch.object(
        sys, "path", [str(root / "csrc/ck_gemm_a8w8_blockscale"), *sys.path]
    ):
        yield importlib.import_module("gemm_a8w8_blockscale_tune")


def _inputs(dtype=torch.bfloat16):
    return (
        torch.empty((33, 512), dtype=torch.float8_e4m3fn),
        torch.empty((384, 512), dtype=torch.float8_e4m3fn),
        torch.empty((33, 4)),
        torch.empty((3, 4)),
        torch.empty((33, 384), dtype=dtype),
    )


@pytest.mark.parametrize("preshuffle", [False, True])
def test_tuned_dispatch_uses_flydsl(backend, catalog, preshuffle):
    x, w, sa, sb, out = _inputs()
    ki = next(
        ki for ki in catalog.kernels_list.values() if ki.preshuffle_b == preshuffle
    )
    config = {"libtype": "flydsl", "kernelName": ki.name, "splitK": 0}
    with (
        patch.object(ops, "get_gfx", return_value="gfx950"),
        patch.object(ops, "get_CKGEMM_config", return_value=config),
        patch.object(backend, "is_supported", return_value=True),
        patch.object(backend, "run_gemm_a8w8_blockscale", return_value=out) as run,
    ):
        if preshuffle:
            result = ops.gemm_a8w8_blockscale_bpreshuffle(x, w, sa, sb, out=out)
            assert run.call_args.args[4] is out
        else:
            result = ops.gemm_a8w8_blockscale(x, w, sa, sb)
        assert result is out
        assert run.call_args.args[5:] == (ki.name, preshuffle)


@pytest.mark.parametrize("preshuffle", [False, True])
@pytest.mark.parametrize("libtype", [None, "ck", "cktile"])
def test_existing_routes_unchanged(preshuffle, libtype):
    x, w, sa, sb, out = _inputs()
    config = (
        None
        if libtype is None
        else {"libtype": libtype, "kernelName": "existing", "splitK": 0}
    )
    suffix = "bpreshuffle_" if preshuffle else ""
    target = f"gemm_a8w8_blockscale_{suffix}{libtype or 'ck'}"
    with (
        patch.object(ops, "get_gfx", return_value="gfx950"),
        patch.object(ops, "get_CKGEMM_config", return_value=config),
        patch.object(ops, "gemm_a8w8_blockscale_flydsl") as flydsl,
        patch.object(ops, target, return_value=out) as existing,
    ):
        if preshuffle:
            result = ops.gemm_a8w8_blockscale_bpreshuffle(x, w, sa, sb, out=out)
        else:
            result = ops.gemm_a8w8_blockscale(x, w, sa, sb)
        assert result is out
        existing.assert_called_once()
        flydsl.assert_not_called()


@pytest.mark.parametrize("preshuffle", [False, True])
def test_fp16_tuned_config_asserts_without_fallback(preshuffle):
    x, w, sa, sb, out = _inputs(torch.float16)
    config = {
        "libtype": "flydsl",
        "kernelName": "flydsl_blockscale_8w_test",
        "splitK": 0,
    }
    target = (
        "gemm_a8w8_blockscale_bpreshuffle_ck"
        if preshuffle
        else "gemm_a8w8_blockscale_ck"
    )
    with (
        patch.object(ops, "get_gfx", return_value="gfx950"),
        patch.object(ops, "get_CKGEMM_config", return_value=config),
        patch.object(ops, target, return_value=out) as fallback,
    ):
        with pytest.raises(AssertionError, match="BF16 output"):
            if preshuffle:
                ops.gemm_a8w8_blockscale_bpreshuffle(
                    x, w, sa, sb, dtype=torch.float16, out=out
                )
            else:
                ops.gemm_a8w8_blockscale(x, w, sa, sb, dtype=torch.float16)
        fallback.assert_not_called()


def test_gfx1250_mxfp8_route_unchanged():
    x, w, sa, sb, out = _inputs()
    sa, sb = sa.to(ops.dtypes.fp8_e8m0), sb.to(ops.dtypes.fp8_e8m0)
    config = {"libtype": "flydsl", "kernelName": "existing_mxfp8_128"}
    with (
        patch.object(ops, "get_gfx", return_value="gfx1250"),
        patch.object(ops, "get_CKGEMM_config", return_value=config),
        patch.object(
            ops, "gemm_a8w8_mxfp8_128_bpreshuffle_flydsl", return_value=out
        ) as existing,
        patch.object(ops, "gemm_a8w8_blockscale_flydsl") as new,
    ):
        assert ops.gemm_a8w8_blockscale_bpreshuffle(x, w, sa, sb, out=out) is out
        existing.assert_called_once()
        new.assert_not_called()


@pytest.mark.parametrize("m", [1, 4, 33])
def test_scale_layouts(backend, m):
    n, k = 384, 512
    sa = torch.arange(m * 4, dtype=torch.float32).reshape(m, 4)
    sb = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expected = sa.T.contiguous().flatten()
    plain, _ = backend._prepare_scales(sa, sb, m, n, k, False)
    torch.testing.assert_close(plain, expected, rtol=0, atol=0)
    for layout in (
        sa.T.contiguous().view_as(sa),
        sa.T.contiguous().T,
        sa.T.contiguous(),
    ):
        actual, actual_b = backend._prepare_scales(layout, sb, m, n, k, True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_b, sb.flatten(), rtol=0, atol=0)
        assert actual.data_ptr() == layout.data_ptr()


def test_invalid_scales_are_rejected(backend):
    _, _, sa, sb, _ = _inputs()
    with pytest.raises(ValueError, match="FP32"):
        backend._prepare_scales(sa.half(), sb, 33, 384, 512, False)
    with pytest.raises(ValueError, match="w_scale"):
        backend._prepare_scales(sa, sb[:1], 33, 384, 512, False)
    with pytest.raises(ValueError, match="x_scale"):
        backend._prepare_scales(sa[:1], sb, 33, 384, 512, False)


def test_kernel_names_and_shape_gates(catalog):
    assert len(catalog.kernels_by_name) == len(catalog.kernels_list) == 2
    assert set(catalog.kernels_list) == {2, 6}
    assert catalog.kernels_list[2].name.endswith("_ps0_sm1_tdma0")
    assert catalog.kernels_list[6].name.endswith("_ps1_sm1_tdma0")
    for ki in catalog.kernels_list.values():
        assert catalog.kernels_by_name[ki.name] is ki
        assert catalog.kernel_fits_shape(ki, 33, 384, 512, "gfx950")
        assert catalog.kernel_fits_shape(ki, 257, 272, 768, "gfx950")
        assert not catalog.kernel_fits_shape(ki, 33, 384, 512, "gfx942")
        assert not catalog.kernel_fits_shape(ki, 33, 384, 384, "gfx950")
        assert not catalog.kernel_fits_shape(ki, 0, 384, 512, "gfx950")
        assert not catalog.kernel_fits_shape(ki, 33, 383, 512, "gfx950")
        assert catalog.kernel_fits_shape(ki, 33, 384, 524288, "gfx950")
        assert not catalog.kernel_fits_shape(ki, 33, 384, 2**23, "gfx950")
        assert not catalog.kernel_fits_shape(ki, 2**27, 384, 512, "gfx950")


@pytest.mark.parametrize("k", [256, 6144, 16384, 131072, 393216, 524288])
def test_lds_bytes_excludes_scalar_b_scales(catalog, k):
    assert catalog.lds_bytes(k) == 136 * 1024


def test_i32_bounds_are_local_but_scales_remain_global(catalog):
    for ki in catalog.kernels_list.values():
        # A/B/C matrices may exceed 4 GiB; bases are rebased before i32 offsets.
        for shape in (
            (262401, 272, 16384),
            (257, 262416, 16384),
            (32769, 65552, 512),
            (2**24, 384, 512),
        ):
            assert catalog.kernel_fits_shape(ki, *shape, "gfx950")
        assert catalog.kernel_fits_shape(ki, 1, 2**22 - 16, 512, "gfx950")
        assert catalog.kernel_fits_shape(ki, 1, 2**22 - 256, 512, "gfx950")
        assert not catalog.kernel_fits_shape(ki, 1, 2**22, 512, "gfx950")
        # ScaleA's final speculative offset, not M*K, bounds large M.
        last_m = ((2**31 - 1) // (6 * 4)) // 256 * 256
        assert catalog.kernel_fits_shape(ki, last_m, 256, 512, "gfx950")
        assert not catalog.kernel_fits_shape(ki, last_m + 256, 256, 512, "gfx950")


def test_removed_modes_are_not_catalog_aliases(catalog):
    prefix = f"{catalog.KERNEL_PREFIX}256x256x128_F8_F8_B16_"
    for ps in (0, 1):
        for suffix in ("sm0_tdma0", "sm0_tdma1", "sm1_tdma1"):
            assert f"{prefix}ps{ps}_{suffix}" not in catalog.kernels_by_name


def test_compiler_signature_has_no_removed_flags(backend):
    from aiter.ops.flydsl.kernels.gemm_a8w8_blockscale_8wave import (
        compile_gemm_fp8_8wave,
    )

    signature = inspect.signature(compile_gemm_fp8_8wave)
    assert list(signature.parameters) == [
        "TILE_M",
        "TILE_N",
        "TILE_K",
        "N",
        "K",
        "pid_swizzle",
        "permlane_epilogue",
        "preshuffle_b",
    ]
    for keyword in ("with_scale", "useTileDMA", "split_m"):
        for value in (False, True):
            with pytest.raises(TypeError, match=keyword):
                compile_gemm_fp8_8wave(256, 256, 128, 256, 512, **{keyword: value})


@pytest.mark.parametrize("preshuffle", [False, True])
def test_compile_adapter_uses_fixed_raw_signature(backend, catalog, preshuffle):
    from aiter.ops.flydsl.kernels import gemm_a8w8_blockscale_8wave as kernel

    ki = catalog.kernels_list[6 if preshuffle else 2]
    launch = SimpleNamespace(compile_hints={})
    backend._compile_gemm.cache_clear()
    try:
        with patch.object(
            kernel, "compile_gemm_fp8_8wave", return_value=launch
        ) as compile:
            assert backend._compile_gemm(384, 512, ki.name, 0) is launch
            compile.assert_called_once_with(
                256, 256, 128, 384, 512, preshuffle_b=preshuffle
            )
            assert launch.compile_hints == {"opt_level": 2}
    finally:
        backend._compile_gemm.cache_clear()


@pytest.mark.parametrize("preshuffle", [False, True])
def test_adapter_preserves_matrix_rank_for_large_address_abi(
    backend, catalog, preshuffle
):
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels import tensor_shim

    x, w, sa, sb, out = _inputs()
    ki = catalog.kernels_list[6 if preshuffle else 2]
    stream = object()
    with (
        patch.object(backend, "is_supported", return_value=True),
        patch.object(backend, "_compile_gemm", return_value=object()),
        patch.object(torch.cuda, "device", return_value=nullcontext()),
        patch.object(torch.cuda, "current_stream", return_value=None),
        patch.object(fx, "Stream", return_value=stream),
        patch.object(tensor_shim, "_run_compiled") as run,
    ):
        assert (
            backend.run_gemm_a8w8_blockscale(x, w, sa, sb, out, ki.name, preshuffle)
            is out
        )
    _, actual_a, actual_b, actual_c, actual_sa, actual_sb, m, actual_stream = (
        run.call_args.args
    )
    assert actual_a.shape == x.shape and actual_a.dtype == torch.int8
    assert actual_b.shape == w.shape and actual_b.dtype == torch.int8
    assert actual_a.data_ptr() == x.data_ptr() and actual_b.data_ptr() == w.data_ptr()
    assert actual_c is out and actual_sa.ndim == actual_sb.ndim == 1
    assert m == 33 and actual_stream is stream


@pytest.mark.parametrize("preshuffle", [False, True])
def test_tuner_tasks_and_result_roundtrip(tuner_module, catalog, preshuffle):
    cls = tuner_module.GemmA8W8BlockScaleTuner
    tuner = cls.__new__(cls)
    tuner.keys = ["gfx", "cu_num", "M", "N", "K"]
    tuner.columns = tuner.keys + [
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]
    tuner.topk = 1
    shape = ("gfx950", 256, 33, 384, 512)
    tasks = tuner.get_gemm_a8w8_blockscale_flydsl_tune_task(shape, 0, preshuffle, {})
    assert len(tasks) == 1
    assert tasks[0][0][1] == (6 if preshuffle else 2)
    for task in tasks:
        info = task[0]
        _, kernel_id, split_k, name, libtype, ps = info
        assert libtype == "flydsl" and split_k == 0 and ps == preshuffle
        assert catalog.kernels_list[kernel_id].name == name
        assert tuner.getKernelName(kernel_id, libtype, preshuffle) == name
        assert task[3] is tuner_module.run_gemm_a8w8_blockscale_flydsl
        assert task[4][0][1:3] == (
            ["weight_shuffle", "x_scale_t"] if preshuffle else ["weight", "x_scale"]
        )
        assert task[-1] == ("out",)
    df = tuner.result_to_df([(tasks[0][0], 12.0, 0.0)])
    assert df.iloc[0]["kernelName"] == tasks[0][0][3]
    assert df.iloc[0]["libtype"] == "flydsl"
    assert (
        tuner.get_gemm_a8w8_blockscale_flydsl_tune_task(
            ("gfx942", *shape[1:]), 0, preshuffle, {}
        )
        == []
    )


@pytest.mark.parametrize(
    "libtype,expect_flydsl", [("all", True), ("flydsl", True), ("both", False)]
)
def test_tuner_includes_backend_without_changing_both(
    tuner_module, libtype, expect_flydsl
):
    tuner = tuner_module.GemmA8W8BlockScaleTuner.__new__(
        tuner_module.GemmA8W8BlockScaleTuner
    )
    args = SimpleNamespace(
        splitK=False,
        mp=1,
        preshuffle=False,
        shape_grouped=True,
        errRatio=0.05,
        blockPerCu=[1],
        warmup=1,
        iters=3,
        libtype=libtype,
        timeout=30,
        verbose=False,
    )
    shapes = pd.DataFrame([{"M": 256, "N": 256, "K": 512}])
    with (
        patch.object(tuner, "get_cu_num", return_value=256),
        patch.object(tuner, "get_gfx", return_value="gfx950"),
        patch.object(tuner, "get_gemm_a8w8_blockscale_tune_task", return_value=[]),
        patch.object(
            tuner, "get_gemm_a8w8_blockscale_cktile_tune_task", return_value=[]
        ),
        patch.object(tuner, "get_gemm_a8w8_blockscale_asm_tune_task", return_value=[]),
        patch.object(tuner, "get_gemm_a8w8_blockscale_opus_tune_task", return_value=[]),
        patch.object(
            tuner, "get_gemm_a8w8_blockscale_flydsl_tune_task", return_value=[]
        ) as flydsl,
    ):
        assert tuner.tune(shapes, pd.DataFrame(), args) == []
        assert flydsl.called == expect_flydsl


@pytest.mark.parametrize("preshuffle", [False, True])
def test_missing_flydsl_asserts_without_fallback(preshuffle):
    x, w, sa, sb, out = _inputs()
    target = (
        "gemm_a8w8_blockscale_bpreshuffle_ck"
        if preshuffle
        else "gemm_a8w8_blockscale_ck"
    )
    with (
        patch.object(ops, "get_gfx", return_value="gfx950"),
        patch.dict(sys.modules, {"aiter.ops.flydsl.gemm_a8w8_blockscale": None}),
        patch.object(ops, target, return_value=out) as fallback,
    ):
        with pytest.raises(AssertionError, match="import failed") as error:
            ops.gemm_a8w8_blockscale_flydsl(
                x,
                w,
                sa,
                sb,
                out,
                {"kernelName": "flydsl_blockscale_8w_test"},
                preshuffle,
            )
        assert isinstance(error.value.__cause__, ImportError)
        fallback.assert_not_called()


def test_kernel_name_layout_mismatch_is_rejected(backend, catalog):
    x, w, sa, sb, out = _inputs()
    plain = catalog.kernels_list[2].name
    with pytest.raises(ValueError, match="B layout"):
        backend.run_gemm_a8w8_blockscale(x, w, sa, sb, out, plain, True)
    with pytest.raises(ValueError, match="Unknown"):
        backend.run_gemm_a8w8_blockscale(x, w, sa, sb, out, "not_a_kernel")


def test_codegen_ignores_flydsl_rows(catalog):
    from aiter.jit.utils import chip_info

    rows = pd.DataFrame(
        [
            {
                "gfx": "gfx950",
                "cu_num": 256,
                "M": 256,
                "N": 256,
                "K": 512,
                "libtype": "flydsl",
                "kernelId": 2,
                "kernelName": catalog.kernels_list[2].name,
            }
        ]
    )
    fallback = object()
    with patch.object(chip_info, "get_build_targets", return_value=[("gfx950", 256)]):
        for libtype in ("ck", "cktile"):
            result = chip_info.build_tune_dict(
                rows, {-1: fallback}, {}, libtype=libtype, kernels_by_name={}
            )
            assert result == {-1: fallback}


def test_missing_flydsl_tuner_skips_candidates(tuner_module):
    tuner = tuner_module.GemmA8W8BlockScaleTuner.__new__(
        tuner_module.GemmA8W8BlockScaleTuner
    )
    with patch.dict(
        sys.modules,
        {"aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_common": None},
    ):
        assert (
            tuner.get_gemm_a8w8_blockscale_flydsl_tune_task(
                ("gfx950", 256, 256, 256, 512), 0, False, {}
            )
            == []
        )


@pytest.mark.parametrize("gfx", ["gfx942", "gfx950"])
def test_benchmark_import_and_no_gpu_skip(gfx):
    from op_tests import test_gemm_a8w8_blockscale_flydsl as bench

    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(bench, "get_gfx", return_value=gfx),
    ):
        bench.main()

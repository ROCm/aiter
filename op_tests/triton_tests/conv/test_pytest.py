# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pytest wrappers for the conv2d test suite."""

from __future__ import annotations
import os
import pytest
import torch

from .suite import TestSuite
from ._registry import ORDERED_METHODS, ALL_METHODS
from .test_edge import (
    test_edge_cases as _run_edge_cases,
    test_activations as _run_activations,
    test_no_bias as _run_no_bias,
)
from .test_fuzz import test_random_fuzzing as _run_random_fuzzing
from .test_models import test_models as _run_models


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _require_model_path(env_var: str, default: str) -> str:
    """Resolve a diffusion model checkpoint path or skip the test.

    Skip explicitly when missing — passing `model_path=None` to `_run_models`
    leaves `suite.results` empty, which `_assert_suite` then reports as pass.
    """
    path = os.environ.get(env_var, default)
    if not os.path.isdir(path):
        pytest.skip(f"{env_var} not set and default path {default!r} not found")
    return path


@pytest.fixture(
    params=[
        (torch.float16, "nchw"),
        (torch.bfloat16, "nchw"),
        (torch.float16, "nhwc"),
        (torch.bfloat16, "nhwc"),
    ],
    ids=["fp16_nchw", "bf16_nchw", "fp16_nhwc", "bf16_nhwc"],
)
def suite(request):
    _require_cuda()
    dtype, layout = request.param
    return TestSuite(
        device="cuda",
        dtype=dtype,
        verbose=True,
        bench_enabled=False,
        layout_mode=layout,
    )


def _assert_suite(suite: TestSuite):
    failed = [r for r in suite.results if not r.passed]
    assert not failed, f"{len(failed)} tests failed: {[r.name for r in failed]}"


# -- Edge case tests ---------------------------------------------------------


@pytest.mark.parametrize("method", ALL_METHODS)
def test_edge(suite, method):
    _run_edge_cases(suite, method=method)
    _assert_suite(suite)


# -- Fuzz tests ---------------------------------------------------------------


@pytest.mark.parametrize("method", ALL_METHODS)
def test_fuzz(suite, method):
    _run_random_fuzzing(suite, num_tests=50, method=method)
    _assert_suite(suite)


# -- Model tests ---------------------------------------------------------------


@pytest.mark.parametrize("method", ALL_METHODS)
def test_models_resnet50(suite, method):
    _run_models(suite, models="resnet50", num_layers=5, method=method)
    _assert_suite(suite)


def test_models_sd_unet(suite):
    model_path = _require_model_path(
        "SD_UNET_MODEL_PATH", "<path to model>/stable-diffusion-v1-5"
    )
    _run_models(
        suite, models="sd_unet", num_layers=5, method="default", model_path=model_path
    )
    _assert_suite(suite)


def test_models_sd35(suite):
    model_path = _require_model_path(
        "SD35_MODEL_PATH", "<path to model>/stable-diffusion-3.5-medium"
    )
    _run_models(
        suite, models="sd35_vae", num_layers=5, method="default", model_path=model_path
    )
    _assert_suite(suite)


# -- Bench-path smoke ---------------------------------------------------------


def test_models_method_all_bench_smoke():
    """Regression guard: `--method all --benchmark` on a non-3x3 layer used
    to KeyError on `METHOD_REGISTRY["all"]`. The parametrized `suite` fixture
    has `bench_enabled=False`, so the bench path is otherwise untested here.
    """
    _require_cuda()
    suite = TestSuite(
        device="cuda",
        dtype=torch.float16,
        verbose=False,
        bench_enabled=True,
        layout_mode="nchw",
    )
    _run_models(suite, models="resnet50", num_layers=3, method="all")
    _assert_suite(suite)


# -- Activation tests ---------------------------------------------------------


@pytest.mark.parametrize("activation", ["relu", "relu6", "gelu"])
@pytest.mark.parametrize("method", ORDERED_METHODS)
def test_activations(suite, activation, method):
    """Test fused activations for each kernel method."""
    _run_activations(suite, method=method, activation=activation)
    _assert_suite(suite)


# -- Bias=None test -----------------------------------------------------------


def test_no_bias(suite):
    """Test all kernel paths with bias=None."""
    _run_no_bias(suite)
    _assert_suite(suite)

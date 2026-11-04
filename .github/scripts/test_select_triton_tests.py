"""CPU regression tests for gfx942 sparse MLA CI routing."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def selector():
    spec = importlib.util.spec_from_file_location(
        "select_triton_tests", Path(__file__).with_name("select_triton_tests.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "path",
    [
        "aiter/ops/flydsl/sparse_mla_qblock_kernels.py",
        "aiter/ops/flydsl/kernels/sparse_mla_qblock.py",
        "op_tests/test_sparse_mla_qblock.py",
        "op_tests/triton_tests/attention/test_sparse_mla_qblock.py",
    ],
)
def test_sparse_mla_selection(selector, path):
    selected, _ = selector.select([path])
    assert selected == [
        "op_tests/triton_tests/attention/test_sparse_mla_qblock.py",
        "op_tests/triton_tests/gemm/basic/test_gemm_a16w4.py",
        "op_tests/triton_tests/kimi_delta_attn/test_fast_launch.py",
        "op_tests/triton_tests/kimi_delta_attn/test_flash_kda.py",
    ]
    assert selector.legacy_trigger([path])


@pytest.mark.parametrize(
    "path,expected",
    [
        ("aiter/ops/flydsl/kernels/sparse_mla_qblock.py", "true"),
        ("aiter/ops/flydsl/sparse_mla_qblock_kernels.py", "true"),
        ("aiter/ops/triton/attention/pa_decode_sparse.py", "true"),
        ("op_tests/triton_tests/attention/test_sparse_mla_qblock.py", "true"),
        ("op_tests/test_sparse_mla_qblock.py", "true"),
        ("aiter/ops/flydsl/kernels/unrelated.py", "false"),
    ],
)
def test_sparse_mla_hardware_trigger(selector, path, expected, monkeypatch, tmp_path):
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(selector, "changed_files", lambda _: [path])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_triton_tests.py",
            "--merge-ref",
            "test-ref",
            "--dry-run",
            "--output",
            str(tmp_path / "tests.list"),
        ],
    )
    selector.main()
    assert f"needs_mi300x={expected}\n" in output.read_text()
    tests = (tmp_path / "tests.list").read_text().splitlines()
    if expected == "true":
        assert selector.TEST_SPARSE_MLA in tests
    else:
        assert tests == []

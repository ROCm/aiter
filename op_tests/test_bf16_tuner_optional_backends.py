# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from csrc.gemm_a16w16 import gemm_a16w16_tune as tune


KEYS = [
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
]
RESULTS = [
    "libtype",
    "solidx",
    "splitK",
    "us",
    "kernelName",
    "err_ratio",
    "tflops",
    "bw",
]


def _tuner():
    return tune.GemmA16W16Tuner("test", KEYS, RESULTS)


def test_vllm_flag_is_exact_and_backend_is_opt_in():
    parser = _tuner().parser
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--with-vllm-wvsplitk" in options
    assert "--with-vllm-vwsplitk" not in options
    assert "vllm_wvsplitk" in tune.ALL_LIBTYPES


def test_flag_absent_does_not_resolve_or_leak_vllm(monkeypatch):
    tuner = _tuner()
    monkeypatch.setattr(
        tune,
        "resolve_vllm_wvsplitk",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not resolve")),
    )
    monkeypatch.setattr(tuner, "get_gfx", lambda: "gfx942")
    monkeypatch.setattr(tuner, "get_cu_num", lambda: 304)
    monkeypatch.setattr(tune, "_VLLM_WVSPLITK_OP", object())
    monkeypatch.setattr(tune, "_VLLM_WVSPLITK_METADATA", {"stale": True})
    args = SimpleNamespace(
        libtype=["all"],
        with_hipblaslt=False,
        with_vllm_wvsplitk=False,
        mp=1,
        shape_grouped=True,
        errRatio=0.05,
        timeout=1,
        verbose=False,
    )
    assert tuner.tune(pd.DataFrame(), pd.DataFrame(), args) == []
    assert tune._VLLM_WVSPLITK_OP is None
    assert tune._VLLM_WVSPLITK_METADATA is None


def test_vllm_wrapper_resolution_preserves_argument_order(monkeypatch):
    calls = []

    def wrapper(weight, activation, cu_count, bias=None):
        calls.append((weight, activation, cu_count, bias))
        return activation

    module = SimpleNamespace(wvSplitK=wrapper, __file__=None)
    monkeypatch.setattr(tune.importlib, "import_module", lambda _: module)
    op, metadata = tune.resolve_vllm_wvsplitk()
    assert metadata["source"] == "vllm._custom_ops.wvSplitK"
    weight, activation, bias = object(), object(), object()
    assert op(weight, activation, bias, 304) is activation
    assert calls == [(weight, activation, 304, bias)]


def test_vllm_unavailable_and_invalid_library_are_nonfatal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tune.importlib,
        "import_module",
        lambda _: (_ for _ in ()).throw(ImportError("not installed")),
    )
    monkeypatch.setattr(
        tune,
        "_registered_wvsplitk_schema",
        lambda _: (_ for _ in ()).throw(ValueError("bad schema")),
    )
    op, metadata = tune.resolve_vllm_wvsplitk()
    assert op is None
    assert metadata["status"] == "unavailable"
    op, metadata = tune.resolve_vllm_wvsplitk(tmp_path / "missing.so")
    assert op is None
    assert "does not exist" in metadata["reason"]


def test_registered_vllm_op_resolution_and_schema(monkeypatch):
    calls = []

    class FakeOp:
        default = SimpleNamespace(
            _schema=(
                "_rocm_C::wvSplitK(Tensor in_a, Tensor in_b, "
                "Tensor? in_bias, int CuCount) -> Tensor"
            )
        )

        def __call__(self, weight, activation, bias, cu_count):
            calls.append((weight, activation, bias, cu_count))
            return activation

    fake = FakeOp()
    monkeypatch.setattr(
        tune.importlib,
        "import_module",
        lambda _: (_ for _ in ()).throw(ImportError("no wrapper")),
    )
    monkeypatch.setattr(tune.torch.ops._rocm_C, "wvSplitK", fake, raising=False)
    op, metadata = tune.resolve_vllm_wvsplitk()
    assert metadata["source"] == "torch.ops._rocm_C.wvSplitK"
    weight, activation = object(), object()
    assert op(weight, activation, None, 304) is activation
    assert calls == [(weight, activation, None, 304)]


def test_supplied_vllm_library_rejects_bad_registered_schema(tmp_path, monkeypatch):
    library = tmp_path / "extension.so"
    library.write_bytes(b"test")
    bad_op = SimpleNamespace(default=SimpleNamespace(_schema="bad schema"))
    monkeypatch.setattr(tune.torch.ops, "load_library", lambda _: None)
    monkeypatch.setattr(tune.torch.ops._rocm_C, "wvSplitK", bad_op, raising=False)
    op, metadata = tune.resolve_vllm_wvsplitk(library)
    assert op is None
    assert "unexpected" in metadata["reason"]


def test_vllm_eligibility_covers_m5_and_rejects_m6():
    eligible, reason = tune.vllm_wvsplitk_eligibility(
        5, 128, 128, torch.bfloat16, torch.bfloat16
    )
    assert eligible and reason == "eligible"
    eligible, reason = tune.vllm_wvsplitk_eligibility(
        6, 128, 128, torch.bfloat16, torch.bfloat16
    )
    assert not eligible and "M=1..5" in reason


def test_vllm_returned_output_policy_and_finite_check(monkeypatch):
    activation = torch.randn(5, 8, dtype=torch.bfloat16)
    weight = torch.randn(7, 8, dtype=torch.bfloat16)
    expected = (activation.float() @ weight.float().T).bfloat16()
    monkeypatch.setattr(tune, "get_cu_num", lambda: 304)
    monkeypatch.setattr(
        tune,
        "_VLLM_WVSPLITK_OP",
        lambda actual_weight, actual_activation, bias, cu_count: expected,
    )
    out = tune.run_vllm_wvsplitk_bf16(activation, weight)
    assert out is expected
    monkeypatch.setattr(
        tune,
        "_VLLM_WVSPLITK_OP",
        lambda *_: torch.full_like(expected, torch.nan),
    )
    tune._VLLM_WVSPLITK_FINITE_CHECKED.clear()
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        tune.run_vllm_wvsplitk_bf16(activation, weight)


def test_small_m_task_generation_uses_distinct_libtype(monkeypatch):
    config = {
        "kernel_family": "small_m",
        "stage": 2,
        "tile_m": 16,
        "tile_n": 128,
        "tile_k": 64,
        "split_k": 1,
        "block_m_warps": 1,
        "block_n_warps": 2,
        "n_tile_repeat": 1,
        "persistent_n_tiles": 1,
        "waves_per_eu": 0,
        "b_to_lds_unroll": 0,
        "async_copy": True,
        "b_to_lds": False,
        "lds_staging": "direct",
        "c_to_lds": False,
        "dtype": "bf16",
        "out_dtype": "bf16",
        "target_gfx": "gfx942",
    }
    monkeypatch.setattr(tune, "get_gfx_runtime", lambda: "gfx942")
    monkeypatch.setattr(
        tune, "iter_small_m_registry_configs", lambda *args, **kwargs: iter([config])
    )
    tasks = _tuner()._get_flydsl_small_m_tasks(
        (
            "gfx942",
            304,
            5,
            128,
            128,
            False,
            str(torch.bfloat16),
            str(torch.bfloat16),
            False,
            False,
        ),
        False,
        torch.bfloat16,
        torch.bfloat16,
        False,
        False,
        {},
    )
    assert len(tasks) == 1
    assert tasks[0][0][4] == "flydsl_small_m"
    assert "_m5_n128_k128_" in tasks[0][0][3]


def test_vllm_results_are_profile_only(tmp_path):
    tuner = _tuner()
    info_keys = (
        "gfx942",
        304,
        5,
        128,
        128,
        False,
        str(torch.bfloat16),
        str(torch.bfloat16),
        False,
        False,
    )
    result = (
        (
            info_keys,
            0,
            0,
            "vllm_wvsplitk_compare_v1",
            "vllm_wvsplitk",
            False,
        ),
        2.0,
        0.0,
    )
    profile = tmp_path / "profile.csv"
    args = SimpleNamespace(profile_file=str(profile))
    assert tuner.post_process([result], args, topk=-1) == []
    text = profile.read_text()
    assert "vllm_wvsplitk" in text
    assert args.profile_file == str(profile)

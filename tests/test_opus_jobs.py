"""Legacy OPUS worker arguments remain ceilings at the executor boundary."""

import importlib.util
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(
    params=["csrc/opus_gemm/gen_co/build_co.py", "op_tests/opus/device/setup.py"]
)
def builder(request):
    path = ROOT / request.param
    spec = importlib.util.spec_from_file_location("opus_jobs_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PoolReached(Exception):
    pass


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], 6),
        (["-j2"], 2),
        (["-j", "2"], 2),
        (["--jobs", "2"], 2),
        (["--jobs=2"], 2),
        (["-j999"], 6),
        (["-j0"], 1),
        (["--jobs=-3"], 1),
    ],
)
def test_cli_reaches_bounded_compilation(
    builder, flags, expected, monkeypatch, tmp_path
):
    monkeypatch.setattr(sys, "argv", [builder.__file__, *flags])
    monkeypatch.setattr(
        builder, "get_worker_count_for", lambda count: min(6, max(1, count))
    )
    observed = []

    def pool(*args, **kwargs):
        observed.append(kwargs["max_workers"])
        raise PoolReached

    if hasattr(builder, "build_one"):
        tag = next(iter(builder._PIPELINE_HEADERS))
        monkeypatch.setattr(
            builder,
            "gfx1250_4wave_co_kernels_declared",
            {i: SimpleNamespace(kernel_tag=tag) for i in range(10)},
        )
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--out-dir", str(tmp_path)])

        def serial(*args):
            observed.append(1)
            raise PoolReached

        monkeypatch.setattr(builder, "build_one", serial)
    else:
        monkeypatch.setattr(builder, "_find_hipcc", lambda: "unused")
        monkeypatch.setattr(builder, "_detect_arch", lambda: "gfx1250")
    with patch("concurrent.futures.ThreadPoolExecutor", side_effect=pool), patch(
        "concurrent.futures.ProcessPoolExecutor", side_effect=pool
    ), pytest.raises(PoolReached):
        builder.main()
    assert observed == [expected]


def test_invalid_jobs_rejected_before_build(builder, monkeypatch):
    monkeypatch.setattr(sys, "argv", [builder.__file__, "--jobs=invalid"])
    with pytest.raises(SystemExit) as error:
        builder.main()
    assert error.value.code == 2


def test_device_python_api_and_work_count_ceiling(monkeypatch):
    path = ROOT / "op_tests/opus/device/setup.py"
    spec = importlib.util.spec_from_file_location("opus_device_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_find_hipcc", lambda: "unused")
    monkeypatch.setattr(module, "_detect_arch", lambda: "gfx1250")
    monkeypatch.setattr(module, "_CU_SOURCES", ["test_mfma_f16.cu"])
    monkeypatch.setattr(module, "get_worker_count_for", lambda count: min(8, count))
    with patch(
        "concurrent.futures.ProcessPoolExecutor", side_effect=PoolReached
    ) as pool, pytest.raises(PoolReached):
        module.build(jobs=99)
    assert pool.call_args.kwargs["max_workers"] == 1

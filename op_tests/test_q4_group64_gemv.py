# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

from aiter import q4_group64_gemv
from aiter.ops.q4_group64_gemv import (
    _AUTO_MAPPING,
    _MAPPING_IDS,
    _gfx_arch_for_index,
    _q4_group64_gemv,
    _q4_group64_gemv_out,
    _require_experimental_enabled,
    _selected_mapping,
)
from op_tests.op_benchmarks.hip import bench_q4_group64_gemv as q4_benchmark
from op_tests.q4_group64_reference import pack_group64 as _pack_group64

RTOL = 5.0e-4
ATOL = 5.0e-3


def _is_gfx1201() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
        return arch.lower().split(":", maxsplit=1)[0] == "gfx1201"
    except Exception:  # noqa: BLE001
        return False


requires_gfx1201 = pytest.mark.skipif(
    not _is_gfx1201(), reason="q4_group64_gemv requires gfx1201"
)


@pytest.fixture(autouse=True)
def _enable_experimental_operator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AITER_ENABLE_EXPERIMENTAL", "1")


def _make_case(
    n: int,
    k: int,
    *,
    seed: int,
    extremes: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if extremes:
        indices = torch.arange(n * k, dtype=torch.int64).reshape(n, k)
        q = torch.where(indices.remainder(2) == 0, -8, 7).to(torch.int8)
    else:
        q = torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8)
    scales = torch.empty((n, k // 64), dtype=torch.float32).uniform_(
        0.005, 0.05, generator=generator
    )
    x = torch.empty(k, dtype=torch.float32).uniform_(-0.25, 0.25, generator=generator)
    packed = _pack_group64(q, scales)
    return x, packed, q, scales.to(torch.float16).float()


def _reference(x: torch.Tensor, q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    result = torch.zeros(n, dtype=torch.float32, device=x.device)
    q_device = q.to(device=x.device, dtype=torch.float32)
    scales_device = scales.to(x.device)
    for group in range(k // 64):
        start = group * 64
        result += (q_device[:, start : start + 64] * x[start : start + 64]).sum(
            dim=1
        ) * scales_device[:, group]
    return result


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_import_does_not_require_gfx1201() -> None:
    assert callable(q4_group64_gemv)


def test_python_experimental_gate_rejects_unset_and_disabled(monkeypatch) -> None:
    x = torch.zeros(64, dtype=torch.float32)
    packed = torch.zeros((1, 1, 1088), dtype=torch.uint8)
    for value in (None, "0", "not-an-integer"):
        if value is None:
            monkeypatch.delenv("AITER_ENABLE_EXPERIMENTAL", raising=False)
        else:
            monkeypatch.setenv("AITER_ENABLE_EXPERIMENTAL", value)
        with pytest.raises(RuntimeError, match="q4_group64_gemv is experimental"):
            q4_group64_gemv(x, packed)

    monkeypatch.setenv("AITER_ENABLE_EXPERIMENTAL", "1")
    _require_experimental_enabled()


def test_arch_cache_is_keyed_by_device_and_does_not_cache_errors(monkeypatch) -> None:
    calls: list[int] = []

    class Properties:
        def __init__(self, arch: str):
            self.gcnArchName = arch

    def get_device_properties(index: int) -> Properties:
        calls.append(index)
        if index == 7:
            raise RuntimeError("invalid device ordinal")
        return Properties("gfx1201:sramecc+:xnack-" if index == 0 else "gfx1100")

    _gfx_arch_for_index.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    try:
        assert _gfx_arch_for_index(0) == "gfx1201"
        assert _gfx_arch_for_index(1) == "gfx1100"
        assert _gfx_arch_for_index(0) == "gfx1201"
        with pytest.raises(RuntimeError, match="invalid device ordinal"):
            _gfx_arch_for_index(7)
        with pytest.raises(RuntimeError, match="invalid device ordinal"):
            _gfx_arch_for_index(7)
    finally:
        _gfx_arch_for_index.cache_clear()

    assert calls == [0, 1, 7, 7]


def test_auto_dispatch_has_all_measured_plain_shapes() -> None:
    assert _AUTO_MAPPING == {
        (512, 3584): "small32x32",
        (1024, 3072): "small32x32",
        (1024, 4096): "small32x32",
        (3072, 3072): "split8",
        (3072, 8192): "split8",
        (3584, 3584): "split8",
        (3584, 18944): "split8",
        (4096, 4096): "split8",
        (4096, 12288): "split8",
        (4096, 14336): "split8",
        (8192, 3072): "split4",
        (12288, 4096): "split8",
        (14336, 4096): "split8",
        (18944, 3584): "split8",
    }


def test_python_and_cpp_auto_dispatch_tables_do_not_drift() -> None:
    source = (Path(__file__).parents[1] / "csrc/kernels/q4_group64_gemv.cu").read_text(
        encoding="utf-8"
    )
    table = source.split("constexpr DispatchEntry kMeasuredDispatch[] = {", maxsplit=1)[
        1
    ].split("};", maxsplit=1)[0]
    entries = re.findall(r"\{(\d+),\s*(\d+),\s*Mapping::([A-Za-z0-9]+)\}", table)
    cpp_mapping = {(int(n), int(k)): mapping.lower() for n, k, mapping in entries}

    assert cpp_mapping == _AUTO_MAPPING
    assert "return Mapping::Old;" in source


def test_cpp_auto_dispatch_has_exact_rx_9070_xt_guard() -> None:
    repository = Path(__file__).parents[1]
    header = (repository / "csrc/include/q4_group64_gemv.h").read_text(encoding="utf-8")
    source = (repository / "csrc/kernels/q4_group64_gemv.cu").read_text(
        encoding="utf-8"
    )
    expected_static_cases = (
        'static_assert(q4_group64_is_tuned_rx_9070_xt(0x7550, 32, "AMD Radeon RX 9070 XT"))',
        'static_assert(q4_group64_is_tuned_rx_9070_xt(0x7550, 32, ""))',
        'static_assert(!q4_group64_is_tuned_rx_9070_xt(0x7550, 28, "AMD Radeon RX 9070"))',
        'static_assert(!q4_group64_is_tuned_rx_9070_xt(0x7550, 24, "AMD Radeon RX 9070 GRE"))',
        'static_assert(!q4_group64_is_tuned_rx_9070_xt(0x7551, 32, "AMD Radeon AI PRO R9700"))',
        'static_assert(!q4_group64_is_tuned_rx_9070_xt(0x7550, 32, "AMD Radeon AI PRO R9700"))',
        'static_assert(!q4_group64_is_tuned_rx_9070_xt(0x7550, 32, "AMD Radeon RX 9070 XT Future"))',
        'static_assert(!q4_group64_is_tuned_rx_9070_xt(0, 0, "unknown"))',
    )
    assert all(case in header for case in expected_static_cases)
    assert "hipDeviceAttributePciChipId" in source
    assert source.index("HipDeviceGuard device_guard") < source.index(
        "cached_is_tuned_rx_9070_xt(x.device_id)"
    )
    assert (
        "cached_is_tuned_rx_9070_xt(x.device_id) ? select_mapping(rows, columns)"
        in source
    )


def test_cpu_call_is_rejected_before_jit() -> None:
    x = torch.zeros(64, dtype=torch.float32)
    packed = torch.zeros((1, 1, 1088), dtype=torch.uint8)
    with pytest.raises(ValueError, match="CUDA/HIP tensors"):
        q4_group64_gemv(x, packed)


def test_test_packer_rejects_non_tile_rows_and_non_group_columns() -> None:
    with pytest.raises(ValueError, match="N must be positive"):
        _pack_group64(torch.zeros((0, 64), dtype=torch.int8), torch.ones((0, 1)))
    with pytest.raises(ValueError, match="K must be positive"):
        _pack_group64(torch.zeros((32, 0), dtype=torch.int8), torch.ones((32, 0)))
    with pytest.raises(ValueError, match="N must be divisible by 32"):
        _pack_group64(torch.zeros((33, 64), dtype=torch.int8), torch.ones((33, 1)))
    with pytest.raises(ValueError, match="K must be divisible by 64"):
        _pack_group64(torch.zeros((32, 65), dtype=torch.int8), torch.ones((32, 1)))


def test_benchmark_cli_rejects_when_no_requested_mapping_is_legal(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_q4_group64_gemv.py",
            "--shape",
            "32",
            "64",
            "--mappings",
            "split2",
        ],
    )
    with pytest.raises(ValueError, match=r"no legal mapping requests.*\(32, 64\)"):
        q4_benchmark.main()


def test_benchmark_auto_uses_public_call_and_controls_allocate_equally(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str | None]] = []
    result = object()

    def public_call(x, packed):
        calls.append(("public", None))
        return result

    def private_call(x, packed, *, mapping, out=None):
        assert out is None
        calls.append(("private", mapping))
        return result

    monkeypatch.setattr(q4_benchmark, "q4_group64_gemv", public_call)
    monkeypatch.setattr(q4_benchmark, "_q4_group64_gemv", private_call)
    assert q4_benchmark._invoke_request(object(), object(), "auto", "auto") is result
    assert (
        q4_benchmark._invoke_request(object(), object(), "selected", "split8") is result
    )
    assert calls == [("public", None), ("private", "split8")]
    assert q4_benchmark._call_path("auto") == "public:q4_group64_gemv"
    assert "out=None" in q4_benchmark._call_path("selected")


def test_benchmark_pci_chip_id_uses_valid_aiter_helper_value(monkeypatch) -> None:
    monkeypatch.setattr(q4_benchmark, "_get_pci_chip_id", lambda _: 0x7550)

    def unexpected_cdll(_):
        raise AssertionError("fallback must not run for a valid helper value")

    monkeypatch.setattr(q4_benchmark.ctypes, "CDLL", unexpected_cdll)
    identity = q4_benchmark._benchmark_pci_chip_id(0)
    assert identity["effective_value"] == 0x7550
    assert identity["fallback_used"] is False


def test_benchmark_pci_chip_id_falls_back_from_rocm72_helper_value(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    class Libhip:
        @staticmethod
        def hipDeviceGetAttribute(pointer, attribute: int, device_id: int) -> int:
            calls.append((attribute, device_id))
            pointer._obj.value = 0x7550
            return 0

    monkeypatch.setattr(q4_benchmark, "_get_pci_chip_id", lambda _: 0x100)
    monkeypatch.setattr(q4_benchmark.ctypes, "CDLL", lambda _: Libhip())
    identity = q4_benchmark._benchmark_pci_chip_id(3)
    assert identity == {
        "aiter_helper_raw": 0x100,
        "aiter_helper_raw_hex": "0x100",
        "fallback_attribute_id": 10020,
        "fallback_value": 0x7550,
        "fallback_value_hex": "0x7550",
        "fallback_used": True,
        "effective_value": 0x7550,
    }
    assert calls == [(10020, 3)]


def test_benchmark_pci_chip_id_fallback_fails_closed(monkeypatch) -> None:
    class Libhip:
        error = 0
        value = 0

        @classmethod
        def hipDeviceGetAttribute(cls, pointer, attribute: int, device_id: int) -> int:
            pointer._obj.value = cls.value
            return cls.error

    monkeypatch.setattr(q4_benchmark, "_get_pci_chip_id", lambda _: 0x100)
    monkeypatch.setattr(q4_benchmark.ctypes, "CDLL", lambda _: Libhip())

    Libhip.error = 17
    with pytest.raises(RuntimeError, match="fallback failed"):
        q4_benchmark._benchmark_pci_chip_id(0)
    Libhip.error = 0
    Libhip.value = 0x7551
    with pytest.raises(RuntimeError, match="expected 0x7550"):
        q4_benchmark._benchmark_pci_chip_id(0)


def test_benchmark_requires_exact_rx_9070_xt_identity(monkeypatch) -> None:
    class Properties:
        def __init__(self, arch: str, name: str, multiprocessors: int):
            self.gcnArchName = arch
            self.name = name
            self.multi_processor_count = multiprocessors

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    identity = {
        "properties": Properties("gfx1201", "", 32),
        "name": "",
        "chip": 0x7550,
    }
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: identity["properties"]
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: identity["name"])
    monkeypatch.setattr(q4_benchmark, "_get_pci_chip_id", lambda _: identity["chip"])

    blank = q4_benchmark._check_arch()
    assert blank["blank_name_compatibility_used"] is True
    identity["name"] = "AMD Radeon RX 9070 XT"
    exact = q4_benchmark._check_arch()
    assert exact["blank_name_compatibility_used"] is False

    for properties, name, chip in (
        (Properties("gfx1201", "", 28), "", 0x7550),
        (Properties("gfx1201", "", 24), "", 0x7550),
        (Properties("gfx1201", "", 32), "AMD Radeon AI PRO R9700", 0x7550),
        (Properties("gfx1201", "", 32), "AMD Radeon RX 9070 XT Future", 0x7550),
        (Properties("gfx1201", "", 32), "", 0x7551),
        (Properties("gfx1100", "", 32), "", 0x7550),
    ):
        identity.update(properties=properties, name=name, chip=chip)
        with pytest.raises(RuntimeError, match="benchmark requires|14-shape"):
            q4_benchmark._check_arch()


def test_packed_tile_round_trip_extremes_scales_and_byte_order() -> None:
    row = torch.arange(32, dtype=torch.int64).unsqueeze(1)
    column = torch.arange(64, dtype=torch.int64).unsqueeze(0)
    q = torch.where((row + column).remainder(2) == 0, -8, 7).to(torch.int8)
    scales = torch.linspace(0.0, 0.031, 32, dtype=torch.float32).unsqueeze(1)
    packed = _pack_group64(q, scales)

    decoded_scales = packed[0, 0, :64].contiguous().view(torch.float16).float()
    torch.testing.assert_close(decoded_scales, scales[:, 0].half().float())

    values = packed[0, 0, 64:].reshape(32, 32)
    low = torch.bitwise_and(values, 0xF).to(torch.int16)
    high = torch.bitwise_right_shift(values, 4).to(torch.int16)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    decoded = torch.empty((32, 64), dtype=torch.int8)
    decoded[:, 0::2] = low.T.to(torch.int8)
    decoded[:, 1::2] = high.T.to(torch.int8)
    assert torch.equal(decoded, q)


@requires_gfx1201
@pytest.mark.parametrize("mapping", [name for name in _MAPPING_IDS if name != "auto"])
def test_all_explicit_mappings(mapping: str) -> None:
    x_cpu, packed_cpu, q, scales = _make_case(256, 320, seed=2026082701)
    x = x_cpu.cuda()
    packed = packed_cpu.cuda()
    expected = _reference(x, q, scales)

    actual = _q4_group64_gemv(x, packed, mapping=mapping)
    torch.cuda.synchronize()
    _assert_close(actual, expected)


@requires_gfx1201
def test_public_auto_known_shape_matches_selected_mapping() -> None:
    n, k = 512, 3584
    assert _selected_mapping(n, k) == "small32x32"
    x_cpu, packed_cpu, q, scales = _make_case(n, k, seed=2026082702)
    x = x_cpu.cuda()
    packed = packed_cpu.cuda()
    expected = _reference(x, q, scales)

    public = q4_group64_gemv(x, packed)
    selected = _q4_group64_gemv(x, packed, mapping=_selected_mapping(n, k))
    torch.cuda.synchronize()
    _assert_close(public, expected)
    _assert_close(public, selected)


@requires_gfx1201
def test_unseen_auto_falls_back_to_boundary_safe_old() -> None:
    n, k = 32, 128
    assert _selected_mapping(n, k) == "old"
    x_cpu, packed_cpu, q, scales = _make_case(n, k, seed=2026082703)
    x = x_cpu.cuda()
    packed = packed_cpu.cuda()
    expected = _reference(x, q, scales)

    auto = q4_group64_gemv(x, packed)
    old = _q4_group64_gemv(x, packed, mapping="old")
    torch.cuda.synchronize()
    _assert_close(auto, expected)
    _assert_close(auto, old)


@requires_gfx1201
def test_extreme_int4_and_zero_scales() -> None:
    x_cpu, packed_cpu, q, scales = _make_case(32, 64, seed=2026082704, extremes=True)
    scales[::2] = 0.0
    packed_cpu = _pack_group64(q, scales)
    x = x_cpu.cuda()
    packed = packed_cpu.cuda()
    expected = _reference(x, q, scales)

    actual = q4_group64_gemv(x, packed)
    torch.cuda.synchronize()
    _assert_close(actual, expected)
    assert torch.count_nonzero(actual[::2]).item() == 0


@requires_gfx1201
def test_public_non_default_stream_and_preallocated_private_output() -> None:
    x_cpu, packed_cpu, q, scales = _make_case(256, 256, seed=2026082705)
    x = x_cpu.cuda()
    packed = packed_cpu.cuda()
    expected = _reference(x, q, scales)
    out = torch.empty(256, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        public = q4_group64_gemv(x, packed)
        actual = _q4_group64_gemv(x, packed, mapping="split8", out=out)
    stream.synchronize()

    assert actual.data_ptr() == out.data_ptr()
    _assert_close(public, expected)
    _assert_close(actual, expected)


@requires_gfx1201
def test_runtime_gate_disables_loaded_public_and_direct_cpp_entries(
    monkeypatch,
) -> None:
    x_cpu, packed_cpu, _, _ = _make_case(32, 64, seed=2026082707)
    x = x_cpu.cuda()
    packed = packed_cpu.cuda()
    out = torch.empty(32, device=x.device, dtype=torch.float32)

    monkeypatch.setenv("AITER_ENABLE_EXPERIMENTAL", "1")
    q4_group64_gemv(x, packed)
    _q4_group64_gemv_out(x, packed, out, _MAPPING_IDS["old"])
    torch.cuda.synchronize()

    for disabled_value in (None, "0", "1junk"):
        if disabled_value is None:
            monkeypatch.delenv("AITER_ENABLE_EXPERIMENTAL", raising=False)
        else:
            monkeypatch.setenv("AITER_ENABLE_EXPERIMENTAL", disabled_value)
        with pytest.raises(RuntimeError, match="q4_group64_gemv is experimental"):
            q4_group64_gemv(x, packed)
        with pytest.raises(RuntimeError, match="q4_group64_gemv is experimental"):
            _q4_group64_gemv_out(x, packed, out, _MAPPING_IDS["old"])

    monkeypatch.setenv("AITER_ENABLE_EXPERIMENTAL", "1")
    q4_group64_gemv(x, packed)
    _q4_group64_gemv_out(x, packed, out, _MAPPING_IDS["old"])
    torch.cuda.synchronize()


@requires_gfx1201
def test_invalid_shapes_dtypes_layout_and_mapping() -> None:
    x = torch.zeros(64, device="cuda", dtype=torch.float32)
    packed = torch.zeros((1, 1, 1088), device="cuda", dtype=torch.uint8)

    with pytest.raises(ValueError, match="K must be positive and divisible by 64"):
        q4_group64_gemv(torch.zeros(65, device="cuda"), packed)
    with pytest.raises(ValueError, match="dtype uint8"):
        q4_group64_gemv(x, packed.float())
    with pytest.raises(ValueError, match="contiguous FP32"):
        q4_group64_gemv(x.half(), packed)
    with pytest.raises(ValueError, match="shape"):
        q4_group64_gemv(x, torch.zeros((1, 1, 1087), device="cuda", dtype=torch.uint8))
    with pytest.raises(ValueError, match="does not match"):
        q4_group64_gemv(x, torch.zeros((1, 2, 1088), device="cuda", dtype=torch.uint8))

    noncontiguous_x = torch.zeros(128, device="cuda", dtype=torch.float32)[::2]
    with pytest.raises(ValueError, match="contiguous FP32"):
        q4_group64_gemv(noncontiguous_x, packed)
    noncontiguous_packed = torch.zeros((1, 1, 2176), device="cuda", dtype=torch.uint8)[
        ..., ::2
    ]
    with pytest.raises(ValueError, match="must be contiguous"):
        q4_group64_gemv(x, noncontiguous_packed)
    with pytest.raises(ValueError, match="unknown mapping"):
        _q4_group64_gemv(x, packed, mapping="unsupported")

    packed_storage = torch.zeros(1089, device="cuda", dtype=torch.uint8)
    misaligned_packed = packed_storage[1:].reshape(1, 1, 1088)
    with pytest.raises(ValueError, match="2-byte aligned"):
        q4_group64_gemv(x, misaligned_packed)


@requires_gfx1201
def test_invalid_explicit_split_shape_is_rejected() -> None:
    x_cpu, packed_cpu, _, _ = _make_case(32, 64, seed=2026082706)
    with pytest.raises(ValueError, match="split2 requires N divisible by 128"):
        _q4_group64_gemv(x_cpu.cuda(), packed_cpu.cuda(), mapping="split2")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

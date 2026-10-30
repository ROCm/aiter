"""Routing and tuner coverage for the gfx950 4-wave INT8 bpreshuffle GEMM.

Launches are replaced by fakes, so these tests check dispatch decisions only.
Numerical cases run from ``op_tests/test_gemm_a8w8.py``.
"""

import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

import aiter.ops.flydsl.gemm_a8w8_bpreshuffle_4wave as four_wave
from aiter.jit.utils.chip_info import get_cdna4_num_xcds
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops import gemm_op_a8w8 as gemm_mod
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_common import (
    PIPELINES,
    ensure_int8_4wave_candidates,
    kernel_name_for_quant_dtype,
)
from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import _xcd_contiguous_index

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU device"
)
requires_gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="4-wave tuner candidates exist only on gfx950",
)


def _int8_operands(k, weight_k):
    return (
        torch.zeros((2, k), device="cuda", dtype=torch.int8),
        torch.zeros((16, weight_k), device="cuda", dtype=torch.int8),
        torch.ones((2, 1), device="cuda", dtype=torch.float32),
        torch.ones((1, 16), device="cuda", dtype=torch.float32),
    )


def _capture_flydsl(monkeypatch, seen, cu_num=256, num_xcds=8):
    def fake_flydsl(XQ, WQ, xs, ws, output, config):
        seen["x_shape"] = tuple(XQ.shape)
        seen["kernel_name"] = config["kernelName"]
        return output

    monkeypatch.setattr(gemm_mod, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_flydsl", fake_flydsl)
    monkeypatch.setattr(four_wave, "get_cu_num", lambda: cu_num)
    monkeypatch.setattr(four_wave, "get_cdna4_num_xcds", lambda _cu=None: num_xcds)


@requires_gpu
def test_padded_k_dispatch_uses_effective_k(monkeypatch):
    seen = {"config_shapes": []}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shapes"].append((m, n, k))
        if k == 256:
            return {
                "libtype": "flydsl",
                "splitK": 0,
                "kernelName": four_wave.make_4wave_kernel_name(K=256),
            }
        return None

    _capture_flydsl(monkeypatch, seen)
    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    gemm_mod.gemm_a8w8_bpreshuffle(*_int8_operands(256, 384))

    assert seen["config_shapes"] == [(2, 16, 384)]
    assert seen["x_shape"] == (2, 384)
    assert "_k384" in seen["kernel_name"]


@requires_gpu
def test_128cu_tuning_miss_uses_xcd4_runtime_variant(monkeypatch):
    seen = {}
    _capture_flydsl(monkeypatch, seen, cu_num=128, num_xcds=4)
    monkeypatch.setattr(
        gemm_mod, "get_GEMM_config_with_quant_type", lambda *_args, **_kwargs: None
    )
    gemm_mod.gemm_a8w8_bpreshuffle(*_int8_operands(256, 256))

    assert seen["kernel_name"].endswith("_k256_xcd4")


@requires_gpu
@pytest.mark.parametrize(
    "libtype,expected",
    [
        ("flydsl", "row_kernel"),
        ("asm", four_wave.make_4wave_kernel_name(K=256)),
    ],
)
def test_int8_dispatch_uses_only_flydsl_rows(monkeypatch, libtype, expected):
    seen = {}
    _capture_flydsl(monkeypatch, seen)
    monkeypatch.setattr(
        gemm_mod,
        "get_GEMM_config_with_quant_type",
        lambda *_args, **_kwargs: {
            "libtype": libtype,
            "splitK": 0,
            "kernelName": "row_kernel",
        },
    )
    gemm_mod.gemm_a8w8_bpreshuffle(*_int8_operands(256, 256))

    assert seen["kernel_name"] == expected


@requires_gpu
@pytest.mark.parametrize(
    "row,expected_name,expected_split",
    [
        ({"libtype": "flydsl", "splitK": 0, "kernelName": "flydsl_row"}, "", 1),
        ({"libtype": "asm", "splitK": 2, "kernelName": "asm_row"}, "asm_row", 2),
    ],
)
def test_asm_wrapper_ignores_non_asm_rows(
    monkeypatch, row, expected_name, expected_split
):
    seen = {}

    def fake_asm(XQ, WQ, xs, ws, output, kernel_name, bias, **kwargs):
        seen["kernel_name"] = kernel_name
        seen["split_k"] = kwargs["splitK"]
        return output

    monkeypatch.setattr(
        gemm_mod, "get_GEMM_config_with_quant_type", lambda *_args, **_kwargs: row
    )
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_asm", fake_asm)
    x, weight, x_scale, w_scale = _int8_operands(256, 256)
    bias = torch.zeros((1, 16), device="cuda", dtype=torch.float32)
    gemm_mod.gemm_a8w8_ASM(x, weight, x_scale, w_scale, bias)

    assert seen == {"kernel_name": expected_name, "split_k": expected_split}


def test_cdna4_xcd_count_accepts_whole_xcd_topologies():
    assert get_cdna4_num_xcds(96) == 3
    assert get_cdna4_num_xcds(128) == 4
    assert get_cdna4_num_xcds(160) == 5
    assert get_cdna4_num_xcds(256) == 8


def test_non_power_of_two_xcd_remap_is_a_permutation():
    for num_xcds in (3, 5):
        for num_wg in (1, 7, 31, 64):
            remapped = [
                _xcd_contiguous_index(wgid, num_wg, num_xcds) for wgid in range(num_wg)
            ]
            assert sorted(remapped) == list(range(num_wg))


def test_persistent_mode_rejects_odd_k128_tile_count():
    with pytest.raises(ValueError, match="even number of K128 tiles"):
        four_wave._validate_compile_config(K=384, xcd_swizzle=8, num_cu=256, num_xcds=8)


def test_common_tuner_registers_int8_4wave_after_existing_pipelines():
    assert [pipeline.name for pipeline in PIPELINES] == [
        "preshuffle",
        "8wave",
        "int8_4wave",
    ]
    id_sets = [set(pipeline.kernels_list) for pipeline in PIPELINES]
    assert not (id_sets[0] & id_sets[1])
    assert not (id_sets[0] & id_sets[2])
    assert not (id_sets[1] & id_sets[2])


def test_common_tuner_names_generic_int8_candidates_for_replay():
    generic = next(iter(PIPELINES[0].kernels_list.values()))
    assert "_I8_I8_" in kernel_name_for_quant_dtype(generic, "int8")


def _four_wave_candidates(cu_num, m, n, k):
    ensure_int8_4wave_candidates(cu_num, k)
    pipeline = next(pipe for pipe in PIPELINES if pipe.name == "int8_4wave")
    return [
        (kernel_id, kernel)
        for kernel_id, kernel in pipeline.kernels_list.items()
        if kernel.cu_num == cu_num and pipeline.fits(kernel, m, n, k)
    ]


@requires_gfx950
def test_int8_4wave_candidates_encode_shape_topology_and_persistence():
    square = _four_wave_candidates(256, 9450, 5120, 5120)
    assert {kernel_id for kernel_id, _ in square} == {1_000_044, 1_000_045}
    assert {kernel.persistent for _, kernel in square} == {False, True}
    assert all(kernel.num_xcds == 8 for _, kernel in square)
    assert all("_k5120" in kernel.name for _, kernel in square)

    long_k = _four_wave_candidates(256, 9450, 5120, 13824)
    assert {kernel_id for kernel_id, _ in long_k} == {1_000_046, 1_000_047}
    assert {kernel.persistent for _, kernel in long_k} == {False, True}

    small_grid = _four_wave_candidates(256, 512, 5120, 5120)
    assert [kernel.persistent for _, kernel in small_grid] == [False]

    xcd4 = _four_wave_candidates(128, 9450, 5120, 5120)
    assert len(xcd4) == 2
    assert all(kernel.name.endswith("_xcd4") for _, kernel in xcd4)

    for cu_num, num_xcds in ((96, 3), (160, 5)):
        candidates = _four_wave_candidates(cu_num, 9450, 5120, 5120)
        assert len(candidates) == 2
        assert all(kernel.num_xcds == num_xcds for _, kernel in candidates)


@requires_gfx950
def test_int8_4wave_candidates_cover_any_supported_k():
    for k in (1024, 8192):
        candidates = _four_wave_candidates(256, 9450, 5120, k)
        assert len(candidates) == 2
        assert all(kernel.k == k for _, kernel in candidates)
    assert _four_wave_candidates(256, 9450, 5119, 5120) == []
    assert _four_wave_candidates(256, 9450, 5120, 128) == []

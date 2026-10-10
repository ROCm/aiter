# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Screening execution checks with fake GPU setup and measurement callables.

Normal ROCm/FlyDSL imports are required; all GPU execution and worker launch
entry points are fail-closed and GPU visibility is empty.
"""

import weakref
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from op_tests.tuning_tests.test_mxfp4_tuner_lifecycle import (
    KEYS,
    RESULTS,
    devices,
    moe,
    prohibit_real_shape_workers_and_gpu_data,  # noqa: F401 - imported autouse fixture
    shape,
)


def test_g1_screening_times_only_direct_kernel_after_untimed_setup(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    events = []

    def prepare(*args: "object") -> "tuple[dict[str, object], tuple[object, ...]]":
        events.append("case")
        return {}, ()

    def prepare_launch(
        *args: "object", **kwargs: "object"
    ) -> "tuple[Callable[[], None], object, object]":
        events.append("wrapper")
        return lambda: events.append("kernel"), object(), object()

    def measure(
        launch: "Callable[[], object]", *, num_warmup: "int", num_iters: "int"
    ) -> "tuple[None, float]":
        assert (num_warmup, num_iters) == (3, 5)
        assert events == ["case", "wrapper"]
        events.append("timing")
        launch()
        return None, 1.25

    monkeypatch.setattr(tuner, "_prepare_screening_case", prepare, raising=False)
    monkeypatch.setattr(
        tuner, "_prepare_gemm1_screening", prepare_launch, raising=False
    )
    monkeypatch.setattr("aiter.test_common.run_perftest", measure)
    score = tuner._screen_gemm1(
        shape(128),
        "flydsl_mxmoe_g1_a4w4_32x128x256_swiglu",
        SimpleNamespace(warmup=3, iters=5),
    )
    assert score == 1.25
    assert events == ["case", "wrapper", "timing", "kernel"]


class FakeTensor:
    def __init__(
        self,
        shape: "tuple[int, ...]",
        dtype: "torch.dtype | None" = None,
        value: "object" = 3,
    ) -> "None":
        self.shape = shape
        self.dtype = dtype or moe.dtypes.bf16
        self.device = "cpu"
        self.value = value

    def view(self, dtype: "torch.dtype | None") -> "FakeTensor":
        return self

    def numel(self) -> "int":
        product = 1
        for dimension in self.shape:
            product *= dimension
        return product

    def zero_(self) -> "FakeTensor":
        self.value = 0
        return self

    def __getitem__(self, index: "int") -> "SimpleNamespace":
        return SimpleNamespace(item=lambda: 128)


def fake_case() -> "tuple[dict[str, object], tuple[object, ...]]":
    data = {
        "input": FakeTensor((4, 3072)),
        "w1_a16": FakeTensor((128, 1024, 1536), torch.uint8),
        "w2_a16": FakeTensor((128, 3072, 256), torch.uint8),
        "w1s_a16": FakeTensor((128, 32), torch.uint8),
        "w2s_a16": FakeTensor((128, 32), torch.uint8),
        "topk_ids": FakeTensor((4, 4), torch.int32),
        "topk_weights": FakeTensor((4, 4), torch.float32),
    }
    for name in ("a1_qt", "w1_qt", "w2_qt", "a1_scale", "w1_scale", "w2_scale"):
        data[name] = object()
    sorting = (
        FakeTensor((256,), torch.int32),
        FakeTensor((256,), torch.float32),
        FakeTensor((16,), torch.int32),
        FakeTensor((2,), torch.int32),
        FakeTensor((4, 3072)),
        FakeTensor((256,), torch.int32),
        FakeTensor((4,), torch.int32),
    )
    return data, sorting


@pytest.mark.parametrize("bm", [16, 32, 64, 128])
@pytest.mark.parametrize(
    "act,configured_limit,expected_limit",
    [
        ("silu", None, float("inf")),
        ("swiglu", None, 7.0),
        ("situv2", None, float("inf")),
        ("silu", 5, 5.0),
        ("swiglu", 5, 5.0),
    ],
)
def test_g1_screening_matches_bm_input_quantization_and_effective_default_parameters(
    monkeypatch: "pytest.MonkeyPatch",
    bm: "int",
    act: "str",
    configured_limit: "float | None",
    expected_limit: "float",
) -> "None":
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    row = shape(
        act_type=f"ActivationType.{ {'silu': 'Silu', 'swiglu': 'Swiglu', 'situv2': 'Situv2'}[act] }"
    )
    name = f"flydsl_mxmoe_g1_a4w4_{bm}x128x256"
    if bm == 16:
        name += "_f16in_nt"
    if act != "silu":
        name += f"_{act}"
    data, sorting = fake_case()
    packed = FakeTensor((4, 1536), moe.dtypes.fp4x2)
    scales = FakeTensor((256, 96), torch.uint8)
    intermediate = FakeTensor((256, 256), torch.uint8), FakeTensor(
        (256, 16), torch.uint8
    )
    timed = False
    events = []
    wrapper_parameters = {}

    def prepare_case(*args: "object") -> "dict[str, object]":
        assert not timed
        events.append("case")
        return data

    def sort(*args: "object", **kwargs: "object") -> "tuple[object, ...]":
        assert not timed
        assert kwargs == {"block_size": bm, "accumulate": False, "output_aux": "opus"}
        events.append("sort")
        return sorting

    def prequant(**kwargs: "object") -> "tuple[object, object]":
        assert not timed
        assert kwargs["input"] is data["input"]
        assert kwargs["num_experts_upper_bound"] == 128
        assert kwargs["block_size"] == bm and kwargs["topk"] == 4
        events.append("prequant")
        return packed, scales

    def prepare_output(
        input: "object", *args: "object", **kwargs: "object"
    ) -> "object":
        assert not timed
        assert input is (data["input"] if bm == 16 else packed)
        assert kwargs["a1_scale"] is (None if bm == 16 else scales)
        assert kwargs["swiglu_limit"] == expected_limit
        wrapper_parameters.update(kwargs)
        events.append("wrapper")
        return intermediate

    def direct(**kwargs: "object") -> "None":
        assert timed
        assert kwargs["hidden_states"] is (data["input"] if bm == 16 else packed)
        assert kwargs["inline_quant"] == kwargs["native_scale_layout"] == (bm == 16)
        assert kwargs["interleave"] is False
        assert kwargs["a_dtype"] == kwargs["out_dtype"] == "fp4"
        assert kwargs["swiglu_limit"] == wrapper_parameters["swiglu_limit"]
        assert (
            kwargs["situ_beta"]
            == wrapper_parameters["situ_beta"]
            == (moe.DEFAULT_SITUV2_BETA if act == "situv2" else 1.0)
        )
        assert (
            kwargs["situ_linear_beta"]
            == wrapper_parameters["situ_linear_beta"]
            == (moe.DEFAULT_SITUV2_LINEAR_BETA if act == "situv2" else 1.0)
        )
        assert kwargs["inter_sorted_quant"] is intermediate[0]
        assert kwargs["inter_sorted_shuffled_scale"] is intermediate[1]
        events.append("kernel")

    def measure(
        call: "Callable[[], object]", *, num_warmup: "int", num_iters: "int"
    ) -> "tuple[None, float]":
        nonlocal timed
        assert (num_warmup, num_iters) == (3, 5)
        timed = True
        try:
            call()
        finally:
            timed = False
        return None, 1.5

    monkeypatch.delenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", raising=False)
    if configured_limit is not None:
        monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", str(configured_limit))
    monkeypatch.setattr(tuner, "_prepare_case", prepare_case)
    monkeypatch.setattr(moe, "moe_sorting", sort)
    monkeypatch.setattr(moe.aiter, "fused_dynamic_mxfp4_quant_moe_sort", prequant)
    monkeypatch.setattr(moe, "_mxfp4_a4w4_stage1_fw", prepare_output)
    monkeypatch.setattr(
        torch, "empty", lambda *args, **kwargs: FakeTensor((0,), torch.uint8)
    )
    monkeypatch.setattr(
        "aiter.ops.flydsl.mxfp4_gemm1_kernels.flydsl_mxfp4_gemm1", direct
    )
    monkeypatch.setattr("aiter.test_common.run_perftest", measure)
    assert tuner._screen_gemm1(row, name, SimpleNamespace(warmup=3, iters=5)) == 1.5
    assert events == (
        ["case", "sort", "wrapper", "kernel"]
        if bm == 16
        else ["case", "sort", "prequant", "wrapper", "kernel"]
    )


def test_g2_screening_owns_fresh_intermediate_and_times_complete_output_path(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    data, _ = fake_case()
    names = [
        f"flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_{epilog}_sbm128"
        for epilog in ("atomic", "reduce", "scatter")
    ]
    producer_name = "flydsl_mxmoe_g1_a4w4_128x128x256_swiglu"
    timed = False
    intermediates = []
    measured = []
    output_refs = []
    events = []

    def prepare_case(*args: "object") -> "dict[str, object]":
        assert not timed
        return data

    def sort(*args: "object", **kwargs: "object") -> "tuple[object, ...]":
        assert not timed
        _, sorting = fake_case()
        output_refs.append(weakref.ref(sorting[4]))
        return sorting

    def producer(*args: "object", **kwargs: "object") -> "tuple[object, object]":
        assert not timed
        assert all(reference() is None for reference in intermediates)
        assert kwargs["kernelName1"] == producer_name and kwargs["swiglu_limit"] == 7.0
        payload, scale = FakeTensor((256, 256), torch.uint8), FakeTensor(
            (256, 16), torch.uint8
        )
        intermediates.extend((weakref.ref(payload), weakref.ref(scale)))
        events.append("producer")
        return payload, scale

    def consume(
        payload: "FakeTensor",
        w1: "object",
        w2: "object",
        sti: "object",
        sei: "object",
        nvi: "object",
        output: "FakeTensor",
        topk: "int",
        **kwargs: "object",
    ) -> "object":
        assert timed
        assert (
            payload is intermediates[-2]() and kwargs["a2_scale"] is intermediates[-1]()
        )
        assert output is output_refs[-1]() and output.dtype == moe.dtypes.bf16
        assert (
            kwargs["sorted_weights"] is not None
            and kwargs["reverse_sorted"] is not None
        )
        assert topk == 4
        name = kwargs["kernelName2"]
        if "atomic" in name:
            assert output.value == 0
        else:
            assert output.value != 0
        output.value = 7
        measured.append(name)
        events.append("complete-output")
        return output

    def measure(
        call: "Callable[[], object]", *, num_warmup: "int", num_iters: "int"
    ) -> "tuple[None, float]":
        nonlocal timed
        assert (num_warmup, num_iters) == (3, 5)
        timed = True
        try:
            for _ in range(2):
                assert call().value == 7
        finally:
            timed = False
        return None, 2.5

    monkeypatch.delenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", raising=False)
    monkeypatch.setattr(tuner, "_prepare_case", prepare_case)
    monkeypatch.setattr(moe, "moe_sorting", sort)
    monkeypatch.setattr(
        moe.aiter,
        "fused_dynamic_mxfp4_quant_moe_sort",
        lambda **kwargs: (
            FakeTensor((4, 1536), moe.dtypes.fp4x2),
            FakeTensor((256, 96), torch.uint8),
        ),
    )
    monkeypatch.setattr(moe, "_mxfp4_a4w4_stage1_fw", producer)
    monkeypatch.setattr(moe, "_mxfp4_a4w4_stage2_fw", consume)
    monkeypatch.setattr(
        torch, "empty", lambda *args, **kwargs: FakeTensor((0,), torch.uint8)
    )
    monkeypatch.setattr("aiter.test_common.run_perftest", measure)
    for name in names:
        assert (
            tuner._screen_gemm2(
                shape(2048), producer_name, name, SimpleNamespace(warmup=3, iters=5)
            )
            == 2.5
        )
        assert all(reference() is None for reference in intermediates)
    assert measured == [name for name in names for _ in range(2)]
    assert events == ["producer", "complete-output", "complete-output"] * 3


@pytest.mark.parametrize("limit", [None, 5.0])
def test_reference_consumer_uses_torch_layout_before_untimed_pair_bootstrap(
    monkeypatch: "pytest.MonkeyPatch",
    limit: "float | None",
) -> "None":
    monkeypatch.delenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", raising=False)
    if limit is not None:
        monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", str(limit))
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    data, sorting = fake_case()
    torch_intermediate = FakeTensor((256, 256), torch.uint8), FakeTensor(
        (256, 16), torch.uint8
    )
    output, ref1, ref2 = object(), object(), object()
    events = []
    kn1 = "flydsl_mxmoe_g1_a4w4_128x128x256_swiglu"
    kn2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_atomic_sbm128"

    def torch_stage1(*args: "object", **kwargs: "object") -> "object":
        assert kwargs["swiglu_limit"] == limit
        assert kwargs["activation"] == moe.ActivationType.Swiglu
        assert kwargs["doweight_stage1"] is False and kwargs["topk"] == 4
        events.append("torch-stage1")
        return ref1

    def populate(
        reference: "object",
        intermediate: "dict[str, object]",
        token: "int",
        topk: "int",
        bm: "int",
    ) -> "None":
        assert reference == {"ref1": ref1, "adtype": "fp4"}
        assert (token, topk, bm) == (2048, 4, 128)
        assert (
            intermediate["sti"] is sorting[0] and intermediate["cumsum"] is sorting[3]
        )
        events.append("layout-population")
        intermediate.update(isq=torch_intermediate[0], iss=torch_intermediate[1])

    def consume(
        payload: "FakeTensor",
        *args: "object",
        **kwargs: "object",
    ) -> "object":
        assert (
            payload is torch_intermediate[0]
            and kwargs["a2_scale"] is torch_intermediate[1]
        )
        events.append("torch-consumer")
        return output

    def torch_stage2(*args: "object", **kwargs: "object") -> "object":
        assert args[0] is ref1
        events.append("torch-stage2")
        return ref2

    def compare(reference: "object", actual: "object", **kwargs: "object") -> "float":
        assert reference is ref2 and actual is output
        events.append("compare")
        return 0.01

    def pair(*args: "object", **kwargs: "object") -> "object":
        assert kwargs["swiglu_limit"] == limit
        assert args[1:3] == (kn1, kn2)
        events.append("normal-pair")
        return output

    def torch_ref(*args: "object", **kwargs: "object") -> "object":
        assert kwargs["swiglu_limit"] == limit
        events.append("normal-reference")
        return ref2

    monkeypatch.setattr(tuner, "_prepare_screening_case", lambda *args: (data, sorting))
    monkeypatch.setattr(tuner, "_prepare_case", lambda *args: data)
    monkeypatch.setattr(moe.FmoeTuner, "run_torch_moe_stage1", torch_stage1)
    monkeypatch.setattr(moe.FmoeTuner, "run_torch_moe_stage2", torch_stage2)
    monkeypatch.setattr(moe, "_v2_populate_stage2", populate)
    monkeypatch.setattr(moe, "_mxfp4_a4w4_stage2_fw", consume)
    monkeypatch.setattr(moe, "cosine_diff_compare", compare)
    monkeypatch.setattr(tuner, "_port_e2e", pair)
    monkeypatch.setattr(tuner, "_torch_ref", torch_ref)
    args = SimpleNamespace(errRatio=0.1)
    assert tuner._check_reference_consumer(shape(2048), kn2, args)
    assert tuner._check_reference_pair(shape(2048), kn1, kn2, args)
    assert events == [
        "torch-stage1",
        "layout-population",
        "torch-consumer",
        "torch-stage2",
        "compare",
        "normal-pair",
        "normal-reference",
        "compare",
    ]

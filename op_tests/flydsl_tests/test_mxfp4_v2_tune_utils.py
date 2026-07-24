import torch
import pytest
from aiter.ops.flydsl.utils import is_flydsl_available

pytestmark = pytest.mark.skipif(not is_flydsl_available(), reason="flydsl not available")


def test_build_v2_inputs_smoke():
    # gen/build_v2_inputs create tensors on the default device (bench sets this
    # at module import via torch.set_default_device("cuda")); mirror that here.
    torch.set_default_device("cuda")
    from aiter.ops.flydsl.mxfp4_v2_tune_utils import gen, build_v2_inputs
    token, md, idim, E, topk, bm = 64, 6144, 512, 257, 9, 32
    d = gen(token, md, idim, E, topk, bm, adtype="fp4")
    v = build_v2_inputs(d, token, md, idim, E, topk, bm)
    assert v["isq"].dtype == torch.uint8
    assert v["isq"].shape[0] == v["max_sorted"]
    assert v["sti"].numel() == v["max_sorted"]
    assert d["ref1"].shape == (token, topk, idim)
    assert d["ref2"].shape == (token, md)


def test_v2_stage1_sorted_ref_and_compare():
    import torch
    torch.set_default_device("cuda")
    from aiter.ops.flydsl.mxfp4_v2_tune_utils import (
        gen, build_v2_inputs, populate_baseline_v2_intermediate,
        v2_stage1_sorted_ref, v2_stage1_dequant_cosine_err,
    )
    token, md, idim, E, topk, bm = 64, 6144, 512, 257, 9, 32
    d = gen(token, md, idim, E, topk, bm, adtype="fp4")
    v = build_v2_inputs(d, token, md, idim, E, topk, bm)
    params = {"tile_m": bm, "tile_n": 64, "tile_k": 256, "k_batch": 1,
              "waves_per_eu": 3, "b_nt": 2, "k_wave": 1}
    populate_baseline_v2_intermediate(d, v, token, topk, params, bm)
    ref = v2_stage1_sorted_ref(
        d["ref1"], d["topk_ids"], v["sti"], v["sei"], v["n"],
        token=token, inter_dim=idim, bm_s1=bm, max_sorted=v["max_sorted"],
    )
    err = v2_stage1_dequant_cosine_err(
        ref, v["isq"], msg="t", printLog=False, inter_dim=idim, adtype="fp4",
    )
    assert err < 0.1  # cosine 通过 (err = 1 - cos)


@pytest.mark.parametrize(
    ("q_dtype_a", "expected_key"),
    [
        (torch.float8_e4m3fn, "a1_qt_fp8_cast"),
        (torch.float4_e2m1fn_x2, "a1_qt"),
    ],
)
def test_v2_stage1_task_selects_activation_input_by_dtype(q_dtype_a, expected_key):
    from aiter import ActivationType, QuantType, dtypes
    from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import FmoeTuner

    info = (
        "gfx950",
        304,
        4,
        7168,
        768,
        384,
        6,
        ActivationType.Silu,
        dtypes.bf16,
        q_dtype_a,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        False,
    )
    tasks = FmoeTuner.gen_flydsl_v2_2stages_task(None, info, [32])
    stage1_task = next(task for task in tasks if task[0][1] == "stage1")

    assert stage1_task[4][0][0] == expected_key


def test_v2_tasks_propagate_swiglu():
    from aiter import ActivationType, QuantType, dtypes
    from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import FmoeTuner

    activation = ActivationType.Swiglu
    info = (
        "gfx950",
        304,
        4,
        7168,
        768,
        384,
        6,
        activation,
        dtypes.bf16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        False,
    )
    tasks = FmoeTuner.gen_flydsl_v2_2stages_task(None, info, [32])
    stage1_task = next(task for task in tasks if task[0][1] == "stage1")
    stage2_task = next(task for task in tasks if task[0][1] == "stage2")

    assert stage1_task[2][-1] == activation
    assert stage1_task[4][-2] == activation
    assert stage2_task[2][-1] == activation


@pytest.mark.parametrize(
    ("activation_name", "expected"),
    [("Silu", "silu"), ("Swiglu", "swiglu")],
)
def test_v2_stage1_launch_uses_activation(monkeypatch, activation_name, expected):
    from aiter import ActivationType
    from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

    activation = getattr(ActivationType, activation_name)
    called = {}
    dummy = torch.zeros(1, dtype=torch.uint8, device="cpu")

    def fake_flydsl_moe_stage1(**kwargs):
        called.update(kwargs)
        return kwargs["out"], dummy

    monkeypatch.setattr(gemm_moe_tune, "flydsl_moe_stage1", fake_flydsl_moe_stage1)
    params = {
        "tile_m": 32,
        "tile_n": 64,
        "tile_k": 256,
        "a_dtype": "fp4",
        "b_dtype": "fp4",
        "out_dtype": "fp4",
    }
    gemm_moe_tune.FmoeTuner.run_flydsl_v2_stage1_out(
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        1,
        256,
        256,
        1,
        1,
        32,
        "fp4",
        activation,
        params,
    )

    assert called["act"] == expected


@pytest.mark.parametrize(
    "generator_name",
    ["generate_v2_stage1_data", "generate_v2_stage2_data"],
)
def test_v2_data_generator_forwards_activation(monkeypatch, generator_name):
    from aiter import ActivationType
    from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

    class StopGeneration(Exception):
        pass

    captured = {}

    def fake_gen(*args, **kwargs):
        captured.update(kwargs)
        raise StopGeneration

    monkeypatch.setattr(gemm_moe_tune, "_v2_gen", fake_gen)
    generator = getattr(gemm_moe_tune.FmoeTuner, generator_name)

    with pytest.raises(StopGeneration):
        generator(4, 256, 256, 1, 1, 32, "fp4", ActivationType.Swiglu)

    assert captured["activation"] == ActivationType.Swiglu


def test_v2_gen_uses_activation_in_reference(monkeypatch):
    from aiter import ActivationType
    from aiter.ops.flydsl import mxfp4_v2_tune_utils

    torch.set_default_device("cuda")
    captured = {}
    original = mxfp4_v2_tune_utils.torch_moe_stage1

    def capture_activation(*args, **kwargs):
        captured["activation"] = kwargs["activation"]
        return original(*args, **kwargs)

    monkeypatch.setattr(mxfp4_v2_tune_utils, "torch_moe_stage1", capture_activation)
    mxfp4_v2_tune_utils.gen(
        1,
        256,
        256,
        1,
        1,
        32,
        adtype="fp4",
        activation=ActivationType.Swiglu,
    )

    assert captured["activation"] == ActivationType.Swiglu


def test_v2_stage2_producer_uses_activation(monkeypatch):
    from aiter import ActivationType
    from aiter.ops.flydsl import mxfp4_v2_tune_utils

    called = {}
    data = torch.zeros(1, dtype=torch.uint8, device="cpu")
    scale = torch.zeros((1, 1), dtype=torch.uint8, device="cpu")
    indices = torch.zeros(1, dtype=torch.int32, device="cpu")
    d = {
        "a1_qt": data,
        "a1_scale": scale,
        "base": {
            "adtype": "fp4",
            "a2_dtype": "fp4",
            "w1_qt_shuf": data,
            "w1_scale_shuf": data,
        },
    }
    v = {
        "sti": indices,
        "sei": indices,
        "cumsum": indices,
        "isq": data,
    }
    params = {
        "tile_m": 32,
        "tile_n": 64,
        "tile_k": 256,
    }

    def fake_flydsl_moe_stage1(**kwargs):
        called.update(kwargs)
        return kwargs["out"], data

    monkeypatch.setattr(
        mxfp4_v2_tune_utils, "moe_mxfp4_sort", lambda *args, **kwargs: data
    )
    monkeypatch.setattr(
        mxfp4_v2_tune_utils, "flydsl_moe_stage1", fake_flydsl_moe_stage1
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    mxfp4_v2_tune_utils.populate_baseline_v2_intermediate(
        d,
        v,
        1,
        1,
        params,
        32,
        activation=ActivationType.Swiglu,
    )

    assert called["act"] == "swiglu"

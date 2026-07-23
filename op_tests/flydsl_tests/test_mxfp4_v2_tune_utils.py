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

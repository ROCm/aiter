import csv

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_kernels import build_flydslv2_gemm2_name
from aiter.ops.flydsl.mxfp4_kname import parse_flydsl_v2_gemm2_kernel


def test_v2_tuner_filters_stage2_tiles_for_shape():
    from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import FmoeTuner

    tuner = FmoeTuner.__new__(FmoeTuner)
    info = (
        "gfx950",
        256,
        1,
        384,
        256,
        2,
        1,
        ActivationType.Silu,
        dtypes.bf16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        False,
    )
    tasks = tuner.gen_flydsl_v2_2stages_task(info, [32])
    configs = [
        parse_flydsl_v2_gemm2_kernel(task[0][2])
        for task in tasks
        if task[0][1] == "stage2"
    ]

    assert {cfg["tile_n"] for cfg in configs} == {128}
    assert {cfg["tile_k"] for cfg in configs} == {128, 256}


def test_v2_tuner_excludes_bm16_retiling():
    from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import FmoeTuner

    tuner = FmoeTuner.__new__(FmoeTuner)
    info = (
        "gfx950",
        256,
        16,
        6144,
        512,
        257,
        9,
        ActivationType.Silu,
        dtypes.bf16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        False,
    )
    tasks = tuner.gen_flydsl_v2_2stages_task(info, [16, 32, 64, 128])
    configs = [
        parse_flydsl_v2_gemm2_kernel(task[0][2])
        for task in tasks
        if task[0][1] == "stage2"
    ]

    assert configs
    assert all(cfg["tile_m"] != 16 for cfg in configs)


def test_v2_tuner_launch_forwards_tiles(monkeypatch):
    from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune
    from aiter.ops.flydsl.kernels import mxmoe_dispatcher

    called = {}
    monkeypatch.setattr(
        mxmoe_dispatcher,
        "mxfp4_moe_gemm2",
        lambda **kwargs: called.update(kwargs),
    )
    tensor = torch.empty(1, device="cpu")
    ref2 = torch.empty((1, 256), dtype=torch.bfloat16, device="cpu")
    params = {
        "tile_m": 32,
        "tile_n": 128,
        "tile_k": 128,
        "sort_block_m": 32,
        "epilog": "atomic",
        "use_nt": False,
        "a_dtype": "fp4",
        "persist": False,
    }

    gemm_moe_tune.FmoeTuner.run_flydsl_v2_stage2_out(
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        32,
        32,
        ref2,
        256,
        128,
        1,
        1,
        params,
    )

    assert (called["BN"], called["BK"]) == (128, 128)


def test_runtime_wrapper_forwards_tiles_from_name(monkeypatch):
    from aiter import fused_moe
    from aiter.ops.flydsl.kernels import mxmoe_dispatcher

    called = {}
    monkeypatch.setattr(
        mxmoe_dispatcher,
        "mxfp4_moe_gemm2",
        lambda **kwargs: called.update(kwargs),
    )
    name = build_flydslv2_gemm2_name(
        "fp4",
        "fp4",
        "bf16",
        tm=32,
        tn=128,
        tk=128,
        epilog="atomic",
        persist=False,
        use_nt=False,
        sbm=32,
    )
    inter = torch.empty((32, 64), dtype=torch.uint8, device="cpu")
    scale = torch.empty(1, dtype=torch.uint8, device="cpu")
    ids = torch.empty(32, dtype=torch.int32, device="cpu")
    weights = torch.empty(32, dtype=torch.float32, device="cpu")
    out = torch.empty((1, 256), dtype=torch.bfloat16, device="cpu")

    fused_moe._flydsl_v2_stage2_wrapper(
        inter_states=inter,
        w1=None,
        w2=inter,
        sorted_token_ids=ids,
        sorted_expert_ids=ids,
        num_valid_ids=ids,
        out=out,
        topk=1,
        kernelName=name,
        model_dim=256,
        inter_dim=128,
        num_experts=1,
        w2_scale=scale,
        a2_scale=scale,
        sorted_weights=weights,
    )

    assert (called["BN"], called["BK"]) == (128, 128)


def test_v2_aot_job_preserves_tiles(tmp_path):
    from aiter.aot.flydsl.mxfp4_moe import _job_key, parse_csv

    name = build_flydslv2_gemm2_name(
        "fp4",
        "fp4",
        "bf16",
        tm=32,
        tn=128,
        tk=128,
        epilog="atomic",
        persist=False,
        use_nt=False,
        sbm=32,
    )
    csv_path = tmp_path / "v2.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "topk",
                "model_dim",
                "expert",
                "inter_dim",
                "kernelName1",
                "kernelName2",
                "cu_num",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "topk": 1,
                "model_dim": 256,
                "expert": 1,
                "inter_dim": 128,
                "kernelName1": "",
                "kernelName2": name,
                "cu_num": 256,
            }
        )

    jobs = parse_csv(str(csv_path))
    assert len(jobs) == 1
    assert (jobs[0]["BN"], jobs[0]["BK"]) == (128, 128)
    assert jobs[0]["D_INTER"] == 128
    assert jobs[0]["inter_dim_pad"] == 0
    assert not jobs[0]["has_pad"]

    other = {**jobs[0], "BK": 256}
    assert _job_key(jobs[0]) != _job_key(other)


def test_v2_aot_path_b_stage1_uses_stage2_bk_alignment(tmp_path):
    from aiter.aot.flydsl.mxfp4_moe import parse_csv

    stage2_name = build_flydslv2_gemm2_name(
        "fp4",
        "fp4",
        "bf16",
        tm=32,
        tn=128,
        tk=128,
        epilog="atomic",
        persist=False,
        use_nt=False,
        sbm=32,
    )
    csv_path = tmp_path / "path_b.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "topk",
                "model_dim",
                "expert",
                "inter_dim",
                "kernelName1",
                "kernelName2",
                "cu_num",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "topk": 1,
                "model_dim": 256,
                "expert": 1,
                "inter_dim": 384,
                "kernelName1": "flydsl_mxmoe_g1_a4w4_32x256x256",
                "kernelName2": stage2_name,
                "cu_num": 256,
            }
        )

    jobs = parse_csv(str(csv_path))
    stage1_job = next(job for job in jobs if job["stage"] == 1)
    assert stage1_job["D_INTER"] == 384

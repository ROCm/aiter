# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""BMM tuner CSV and dispatch regressions; tensor generation is mocked."""

import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "csrc" / "bmm_a8w8_mxscale"))
import bmm_a8w8_mxscale_bpreshuffle_tune as joint

opus = joint.opus_tune


@pytest.fixture
def tuner(monkeypatch):
    t = joint.BmmA8W8MxscaleBpreshuffleTuner()
    monkeypatch.setattr(t, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(t, "get_cu_num", lambda: 256)
    return t


def shapes():
    return pd.DataFrame(
        [
            ("gfx950", 4, 16, 1024, 4096, "32x32"),
            ("gfx950", 4, 256, 1024, 4096, "128x128"),
            ("gfx942", 4, 32, 1024, 4096, "128x128"),
        ],
        columns=opus.OpusBmmMxscaleTuner.KEYS,
    )


def preprocess(tuner, tmp_path, extra=(), *, saved=False):
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    shapes().to_csv(source, index=False)
    if saved:
        shapes().to_csv(output, index=False)
    args = tuner.parser.parse_args(["-i", str(source), "-o", str(output), *extra])
    tuner.pre_process(args)
    return args


def test_preserves_sparse_shape_scale_keys(tuner, tmp_path):
    args = preprocess(tuner, tmp_path)
    pd.testing.assert_frame_equal(tuner.untunedf, shapes().iloc[:2])
    assert not args.all


def test_filters_scale_without_adding_rows(tuner, tmp_path):
    preprocess(tuner, tmp_path, ["--w_scale_block", "128x128"])
    pd.testing.assert_frame_equal(
        tuner.untunedf, shapes().iloc[[1]].reset_index(drop=True)
    )


@pytest.mark.parametrize("retune, expected", [(False, 0), (True, 2)])
def test_all_controls_existing_shapes(tuner, tmp_path, retune, expected):
    preprocess(tuner, tmp_path, ["--all"] if retune else [], saved=True)
    assert len(tuner.untunedf) == expected
    assert len(tuner.tunedf) == 3


def test_block_list_expands_shapes_without_duplicates(tuner, tmp_path):
    source = tmp_path / "shapes.csv"
    pd.DataFrame({"b": [1], "m": [17], "n": [128], "k": [128]}).to_csv(
        source, index=False
    )
    args = tuner.parser.parse_args(
        [
            "-i",
            str(source),
            "-o",
            str(tmp_path / "out.csv"),
            "--w_scale_block",
            "32x32,128x128,32x32",
        ]
    )
    tuner.pre_process(args)
    assert list(tuner.untunedf.w_scale_block) == ["32x32", "128x128"]


@pytest.mark.parametrize("block", ["32", "64x64", "32x32,"])
def test_invalid_block_option_rejected(tuner, block):
    with pytest.raises(SystemExit):
        tuner.parser.parse_args(["--w_scale_block", block])


def test_only_missing_scale_column_expands(tuner):
    data = pd.DataFrame({"G": [2], "M": [16], "N": [256], "K": [512]})
    normalized = tuner._normalize_rows(data, default_blocks=("32x32", "128x128"))
    assert list(normalized.w_scale_block) == ["32x32", "128x128"]
    assert list(normalized.b) == [2, 2]


@pytest.mark.parametrize(
    "column,value",
    [("b", 0), ("m", 1.5), ("n", 129), ("k", float("nan")), ("w_scale_block", "64x64")],
)
def test_invalid_shapes_rejected(tuner, column, value):
    data = shapes().iloc[:1].copy()
    data[column] = value
    with pytest.raises(ValueError):
        tuner._normalize_rows(data)


def mock_data():
    return tuple(Mock(name=f"tensor_{i}") for i in range(10))


def saved_row(kid=8477, block="128x128"):
    return {
        "gfx": "gfx950",
        "b": 4,
        "m": 16,
        "n": 1024,
        "k": 4096,
        "w_scale_block": block,
        "libtype": "opus",
        "kernelId": kid,
        "splitK": 1,
    }


def test_saved_opus_passes_all_operands(tuner, monkeypatch):
    data = mock_data()
    monkeypatch.setattr(opus, "gen_bmm_mxscale_data", Mock(return_value=data))
    run, operands, ref = tuner._saved_benchmark(saved_row(), 7)
    assert run is opus.run_bmm_mxscale_bench
    assert operands == tuple(data[i] for i in (0, 1, 2, 3, 4, 5, 7, 8, 9)) + (8477, 1)
    assert ref is data[6]
    data[2].fill_.assert_called_once()


@pytest.mark.parametrize(
    "kid,block", [(8477, "32x32"), (8000, "128x128"), (99999, "128x128")]
)
def test_saved_opus_rejects_wrong_contract(tuner, kid, block):
    with pytest.raises(ValueError):
        tuner._saved_benchmark(saved_row(kid, block), 1)


@pytest.mark.parametrize(
    "preshuffle,group", [(False, 32), (False, 128), (True, 32), (True, 128)]
)
def test_default_routes_layout_and_scale(tuner, monkeypatch, preshuffle, group):
    from aiter.ops import batched_gemm_op_a8w8 as ops

    data = mock_data()
    generate = Mock(return_value=data)
    monkeypatch.setattr(opus, "gen_bmm_mxscale_data", generate)
    tuner._bpreshuffle = preshuffle
    run, operands, ref = tuner._default_benchmark(
        saved_row(block=f"{group}x{group}"), 3
    )
    expected = (
        ops.batched_gemm_a8w8_mxscale_bpreshuffle
        if preshuffle
        else ops.batched_gemm_a8w8_mxscale
    )
    assert run is expected
    assert operands == (data[0], data[7 if preshuffle else 1], data[3], data[4])
    assert ref is data[6]
    assert generate.call_args.args[-2:] == (opus.BMM_DATA_KIDS[group], 1)


@pytest.mark.parametrize("group", [32, 128])
def test_saved_flydsl_validation(tuner, monkeypatch, group):
    name = joint.pick_bmm_kernel_name(4, 16, 1024, 4096, group, group, group)
    config = joint.parse_bmm_kernel_name(name)
    row = {
        **saved_row(block=f"{group}x{group}"),
        "libtype": "flydsl",
        "kernelId": -1,
        "kernelName": name,
        "splitK": config["splits"],
    }
    data = mock_data()[:6]
    generate = Mock(return_value=data)
    monkeypatch.setattr(joint, "gen_flydsl_bmm_data", generate)
    run, operands, ref = tuner._saved_benchmark(row, 2)
    assert run is joint.run_flydsl_bmm_bench
    assert operands == data[:5] + (name,)
    assert ref is data[5]
    assert generate.call_args.args[-1] == group
    with pytest.raises(ValueError, match="splitK"):
        tuner._saved_benchmark({**row, "splitK": config["splits"] + 1}, 2)


def test_switching_config_preserves_loaded_modules(tuner, tmp_path, monkeypatch):
    from aiter.jit import core

    tuner._bpreshuffle = True
    env = tuner.get_arg_defaults()["config_env_name"]
    monkeypatch.setenv(env, "/tmp/original_bmm.csv")
    old_modules = dict(getattr(core, "__mds"))
    rebuilds = list(core.rebuilded_list)
    rebuild_flag = core.AITER_REBUILD
    args = tuner.parser.parse_args(["-o", str(tmp_path / "candidate.csv")])
    previous = tuner._set_config_env_for_run_config(args)
    assert core.AITER_REBUILD == rebuild_flag
    assert dict(getattr(core, "__mds")) == old_modules
    assert core.rebuilded_list == rebuilds
    assert os.environ[env] == args.tune_file
    tuner._restore_config_env(env, *previous)
    assert os.environ[env] == "/tmp/original_bmm.csv"
    assert core.AITER_REBUILD == rebuild_flag


def test_run_saved_config_respects_group_filter(tuner, tmp_path, monkeypatch):
    from aiter import test_common

    source = tmp_path / "saved.csv"
    pd.DataFrame([saved_row(), saved_row(9179, "32x32")]).to_csv(source, index=False)
    args = tuner.parser.parse_args(
        ["--run_config", str(source), "--w_scale_block", "32x32", "-o", str(source)]
    )
    prepare = Mock(return_value=(lambda: None, (), object()))
    monkeypatch.setattr(tuner, "_saved_benchmark", prepare)
    monkeypatch.setattr(test_common, "run_perftest", lambda *a, **kw: (object(), 2.0))
    monkeypatch.setattr(test_common, "checkAllclose", lambda *a, **kw: 0.0)
    tuner.run(args)
    assert prepare.call_count == 1
    assert prepare.call_args.args[0]["w_scale_block"] == "32x32"


def test_task_operands_and_outputs(tuner, tmp_path):
    args = preprocess(tuner, tmp_path, ["--libtype", "all"])
    tuner.opus_policy = {8477: [1], 9179: [1]}
    tasks = list(tuner._iter_tuning_tasks(pd.Series(saved_row()), 1, args))
    opus_tasks = [task for task in tasks if task[0][1] == 8477]
    fly_tasks = [task for task in tasks if task[0][1] == -1]
    assert len(opus_tasks) == 1
    assert fly_tasks
    assert opus_tasks[0][4] == ((0, 1, 2, 3, 4, 5, 7, 8, 9), 8477, 1)
    assert opus_tasks[0][7] == ([6],)
    assert opus_tasks[0][-1] == [2]
    assert all(task[7] == ([5],) and task[-1] == [4] for task in fly_tasks)


@pytest.mark.parametrize("preshuffle", [False, True])
def test_opus_default_output_follows_layout(monkeypatch, preshuffle):
    tuner = opus.OpusBmmMxscaleTuner()
    monkeypatch.setattr(tuner, "get_gfx", lambda: "gfx950")
    args = tuner.parser.parse_args(["--bpreshuffle"] if preshuffle else [])
    tuner.pre_process(args)
    assert args.tune_file == (opus.BPRESHUFFLE_CSV if preshuffle else opus.DEFAULT_OUT)


def test_shipped_source_preserves_all_current_arch_keys(tuner, tmp_path):
    args = tuner.parser.parse_args(["-o", str(tmp_path / "new.csv")])
    tuner.pre_process(args)
    expected = pd.read_csv(opus.BPRESHUFFLE_CSV)
    expected = expected[expected.gfx == "gfx950"][tuner.keys].drop_duplicates()
    pd.testing.assert_frame_equal(tuner.untunedf, expected.reset_index(drop=True))


def test_1x32_candidates_do_not_mix_square_blocks(tuner, tmp_path):
    args = preprocess(tuner, tmp_path, ["--libtype", "all"])
    tuner.opus_policy = {9700: [1, 2], 9179: [1], 8477: [1]}
    row = pd.Series(saved_row(9700, "1x32"))
    tasks = list(tuner._iter_tuning_tasks(row, 1, args))
    assert [(task[0][1], task[0][2]) for task in tasks] == [(9700, 1), (9700, 2)]
    with pytest.raises(ValueError, match="Scale block"):
        tuner._saved_benchmark(saved_row(9179, "1x32"), 1)
    with pytest.raises(ValueError, match="Scale block"):
        tuner._saved_benchmark(saved_row(9700, "32x32"), 1)


def test_1x32_shape_only_expansion_and_explicit_filter(tuner, tmp_path):
    path = tmp_path / "shapes.csv"
    pd.DataFrame({"b": [1], "m": [17], "n": [16], "k": [128]}).to_csv(path, index=False)
    args = tuner.parser.parse_args(
        ["-i", str(path), "-o", str(tmp_path / "out.csv"), "--w_scale_block", "1x32"]
    )
    tuner.pre_process(args)
    assert list(tuner.untunedf.w_scale_block) == ["1x32"]
    assert opus._applicable(9710, 1, 17, 16, 128, "preb", split_ks=[1, 2, 4]) == [1]
    assert tuner._normalize_rows(shapes(), blocks=("1x32",)).empty


def test_1x32_default_data_and_policy(tuner, monkeypatch):
    from aiter.ops.opus import policy

    monkeypatch.setattr(policy, "get_gfx", lambda: "gfx950")
    assert policy.mxscale_bmm_kid_block(9700) == "1x32"
    assert policy.mxscale_bmm_kid_block(9179) == "32x32"
    assert (
        policy._heuristic_mxscale_bmm_bpreshuffle_kid(
            1, 17, 16, 128, group_size=32, group_n=1
        )
        == 9710
    )
    with pytest.raises(ValueError, match="preshuffled"):
        policy.resolve_a8w8_mxscale_bmm_plan(1, 17, 16, 128, w_scale_block="1x32")
    generate = Mock(return_value=mock_data())
    monkeypatch.setattr(opus, "gen_bmm_mxscale_data", generate)
    tuner._bpreshuffle = True
    tuner._default_benchmark(saved_row(9700, "1x32"), 4)
    assert generate.call_args.args[-2:] == (9704, 1)


def test_1x32_public_dispatch_propagates_both_groups(monkeypatch):
    from aiter.ops import batched_gemm_op_a8w8 as ops
    from aiter.ops.opus import gemm_op_a8w8 as backend

    monkeypatch.setattr(ops, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(ops, "lookup_mxscale_bmm_config", lambda *a, **kw: None)
    launch = Mock(return_value=object())
    monkeypatch.setattr(backend, "bmm_a8w8_mxscale_opus", launch)
    x, w, xs, ws = (
        Mock(shape=s) for s in [(17, 2, 512), (2, 48, 512), (17, 2, 16), (2, 48, 16)]
    )
    ops._batched_gemm_a8w8_mxscale_bpreshuffle_impl(x, w, xs, ws)
    assert launch.call_args.kwargs["group_size"] == 32
    assert launch.call_args.kwargs["group_n"] == 1
    assert launch.call_args.kwargs["b_preshuffled"] is True


def test_1x32_cpu_quantization_and_reference():
    import torch

    from aiter.ops.quant import per_1x32_f8_scale_f8_quant

    torch.manual_seed(91)
    exponents = torch.arange(12).reshape(2, 3, 2) % 8 - 3
    weights = torch.randn(2, 3, 64) * torch.exp2(exponents.float()).repeat_interleave(
        32, -1
    )
    activations = torch.randn(2, 5, 64)
    wq, ws, wf = opus._quant_block_e8m0(weights, group=32, group_n=1)
    xq, xs, xf = opus._quant_per_token_e8m0(activations, group=32)
    assert ws.shape == (2, 3, 2)
    assert xs.shape == (2, 5, 2)
    assert torch.unique(ws).numel() >= 4
    reference = torch.bmm(
        xq.float() * xf.repeat_interleave(32, -1),
        (wq.float() * wf.repeat_interleave(32, -1)).transpose(1, 2),
    )
    torch.testing.assert_close(
        opus.run_torch(xq, wq, xf, wf, group=32, group_n=1), reference
    )
    # The existing public quantizer emits the same ordinary E8M0 scale shape.
    pq, ps = per_1x32_f8_scale_f8_quant(weights, scale_type=torch.uint8)
    assert pq.shape == weights.shape and ps.shape == (6, 2)
    assert torch.isfinite(pq.float()).all()
    # float8_e8m0fnu.float() is already a scale value, whereas uint8.float()
    # is the biased exponent. Decode the raw bytes consistently for both.
    torch.testing.assert_close(ps.view(torch.uint8).view_as(ws), ws)
    torch.testing.assert_close(pq.float(), wq.float())

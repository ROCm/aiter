# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host dispatch and launcher-cache checks using meta tensors and mocked kernels."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from aiter.ops.flydsl import mhc_seam_kernels as seam


@pytest.fixture
def dispatch(monkeypatch):
    state = SimpleNamespace(
        compiled=[], launches=[], packs=[], norms=[], stream=object(), capturing=False
    )

    def builder(name):
        def build(**kwargs):
            state.compiled.append((name, kwargs))
            return name

        return build

    def pack(fn, capturing):
        state.packs.append((fn, capturing))
        return fn.to(torch.bfloat16)

    def norm(stage, weight, eps):
        out = torch.empty_like(stage)
        state.norms.append((stage, weight, eps, out))
        return out

    monkeypatch.setattr(seam, "_arch", lambda index: "gfx942")
    monkeypatch.setattr(seam, "_cu_num", lambda: 304)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: state.stream)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    monkeypatch.setattr(seam, "ptr_arg", lambda tensor, dtype: tensor)
    monkeypatch.setattr(seam.flyc, "from_c_void_p", lambda dtype, address: None)
    monkeypatch.setattr(seam, "_pack_fn_nonorm", pack)
    monkeypatch.setattr(seam, "rms_norm", norm)
    monkeypatch.setattr(
        seam, "_run_compiled", lambda fn, *args: state.launches.append((fn, args))
    )
    for module, names in (
        (seam.mhc_seam, ("compile_mhc_seam_fused", "compile_mhc_seam_finish")),
        (
            seam.mhc_seam_nonorm,
            ("compile_mhc_seam_fused_nonorm", "compile_mhc_seam_finish_nonorm"),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, builder(name))
    seam._compile_launcher.cache_clear()
    yield state
    seam._compile_launcher.cache_clear()


def _inputs(T=129, H=512):
    def empty(*shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype, device="meta")

    return {
        "residual": empty(T, 4, H, dtype=torch.bfloat16),
        "fn": empty(24, 4 * H),
        "hc_scale": empty(3),
        "hc_base": empty(24),
        "rms_eps": 1e-6,
        "hc_pre_eps": 1e-5,
        "hc_sinkhorn_eps": 1e-6,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": 20,
        "pre_mix": empty(T, 4),
        "sublayer_out": empty(T, H, dtype=torch.bfloat16),
        "post_layer_mix": empty(T, 4, 1),
        "comb_res_mix": empty(T, 4, 4),
        "norm_weight": empty(H, dtype=torch.bfloat16),
        "norm_eps": 1e-6,
    }


def test_config_overrides_preserve_policy_and_cache_semantics(dispatch):
    inputs = _inputs()
    seam.flydsl_mhc_seam(**inputs)
    assert dispatch.compiled[0][0] == "compile_mhc_seam_fused_nonorm"

    seam.flydsl_mhc_seam(**inputs, config={})
    assert dispatch.compiled[2][0] == "compile_mhc_seam_fused"
    assert dispatch.compiled[2][1]["chunk_c"] == 256

    config = {"NORM": "fused", "BLOCK_M": 32, "CHUNK_C": 128}
    seam.flydsl_mhc_seam(**inputs, config=config)
    seam.flydsl_mhc_seam(**inputs, config=dict(reversed(config.items())))
    assert len(dispatch.compiled) == 6

    config["CHUNK_C"] = 64
    seam.flydsl_mhc_seam(**inputs, config=config)
    assert len(dispatch.compiled) == 8
    assert dispatch.compiled[-2][1]["chunk_c"] == 64

    config["CHUNK_C"] = 96
    with pytest.raises(ValueError, match="chunk_c"):
        seam.flydsl_mhc_seam(**inputs, config=config)
    assert len(dispatch.compiled) == 8


@pytest.mark.parametrize("norm", ["fused", "separate"])
def test_cached_runner_uses_each_calls_tensors_scratch_and_stream(dispatch, norm):
    first = _inputs()
    first_out = seam.flydsl_mhc_seam(**first, config={"NORM": norm})
    first_stream = dispatch.stream
    dispatch.stream = object()
    dispatch.capturing = True
    second = _inputs()
    second["residual_out"] = torch.empty_like(second["residual"])
    second_out = seam.flydsl_mhc_seam(**second, config={"NORM": norm})

    assert len(dispatch.compiled) == 2
    assert len(dispatch.launches) == 4
    first_fused, first_finish, second_fused, second_finish = (
        args for _, args in dispatch.launches
    )
    assert first_fused[0] is first["residual"]
    assert second_fused[0] is second["residual"]
    assert first_fused[6] is first_out[0]
    assert second_fused[6] is second_out[0] is second["residual_out"]
    assert first_fused[8] is first_finish[0]
    assert second_fused[8] is second_finish[0]
    assert first_fused[8] is not second_fused[8]
    assert first_fused[-1] is first_finish[-1] is first_stream
    assert second_fused[-1] is second_finish[-1] is dispatch.stream
    for finish, outputs in ((first_finish, first_out), (second_finish, second_out)):
        assert finish[-6] is outputs[1]
        assert finish[-5] is outputs[2]
        assert finish[-4] is outputs[4]

    if norm == "separate":
        assert dispatch.packs[0][0] is first["fn"]
        assert dispatch.packs[0][1] is False
        assert dispatch.packs[1][0] is second["fn"]
        assert dispatch.packs[1][1] is True
        assert first_out[3] is dispatch.norms[0][3]
        assert second_out[3] is dispatch.norms[1][3]
    else:
        assert first_fused[5] is first["fn"]
        assert second_fused[5] is second["fn"]
        assert first_fused[7] is first_out[3]
        assert second_fused[7] is second_out[3]
        assert not dispatch.packs and not dispatch.norms


@pytest.mark.parametrize("norm", ["fused", "separate"])
def test_launcher_cache_does_not_retain_call_tensors(dispatch, norm):
    def call():
        inputs = _inputs()
        outputs = seam.flydsl_mhc_seam(**inputs, config={"NORM": norm})
        tensors = [v for v in inputs.values() if isinstance(v, torch.Tensor)]
        tensors.extend(outputs)
        for _, args in dispatch.launches:
            tensors.extend(v for v in args if isinstance(v, torch.Tensor))
        return [weakref.ref(t) for t in tensors]

    refs = call()
    dispatch.launches.clear()
    dispatch.packs.clear()
    dispatch.norms.clear()
    gc.collect()
    assert seam._compile_launcher.cache_info().currsize == 1
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize(
    "name,value",
    [
        ("rms_eps", 1e-4),
        ("hc_pre_eps", 1e-4),
        ("hc_sinkhorn_eps", 1e-4),
        ("hc_post_mult_value", 3.0),
        ("sinkhorn_repeat", 10),
        ("norm_eps", 1e-4),
    ],
)
def test_scalar_constants_select_a_new_launcher(dispatch, name, value):
    inputs = _inputs()
    seam.flydsl_mhc_seam(**inputs, config={})
    inputs[name] = value
    seam.flydsl_mhc_seam(**inputs, config={})
    assert len(dispatch.compiled) == 4
    assert dispatch.compiled[-1][1][name] == value


@pytest.mark.parametrize("has_post", [False, True])
@pytest.mark.parametrize("identity_pre", [False, True])
def test_empty_batch_skips_configuration_and_launch(dispatch, has_post, identity_pre):
    inputs = _inputs(T=0)
    if not has_post:
        inputs.update(sublayer_out=None, post_layer_mix=None, comb_res_mix=None)
    if identity_pre:
        inputs["pre_mix"] = None
    outputs = seam.flydsl_mhc_seam(**inputs, config={"unused": "ignored for T=0"})
    assert [tuple(t.shape) for t in outputs] == [
        (0, 4, 512),
        (0, 4, 1),
        (0, 4, 4),
        (0, 512),
        (0, 4),
    ]
    if not has_post:
        assert outputs[0] is inputs["residual"]
    assert not dispatch.compiled and not dispatch.launches and not dispatch.packs

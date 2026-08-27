# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-side coverage for gfx1250 pre-built OPUS A16W16 kernels."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aiter.ops.opus.launch_plan import _get_cached_a16w16_launch_plan
from csrc.opus_gemm.opus_gemm_common import (
    BIAS_AWARE_KIDS,
    CO_KERNELS_JSON,
    DEFAULT_COMPILED_KIDS_BY_ARCH,
    GFX1250_4WAVE_CO_KIDS,
    NON_SPLITK_KIDS,
    OPUS_KERNEL_TAGS_BY_ARCH_FAMILY,
    SPLITK_KIDS,
    _load_co_kernels,
    co_image_path,
    get_kernel_instance,
    gfx1250_4wave_co_kernels_list,
    kernel_needs_external_workspace,
)


def _co_plan(kid: int, **overrides):
    arguments = {
        "arch": "gfx1250",
        "M": 128,
        "N": 128,
        "K": 4096,
        "batch": 1,
        "cu_num": 80,
        "has_bias": False,
        "input_dtype": torch.bfloat16,
        "output_dtype": torch.bfloat16,
        "kid": kid,
        "split_k": 0,
    }
    arguments.update(overrides)
    return _get_cached_a16w16_launch_plan(**arguments)


def test_gfx1250_co_registry_and_launch_contract():
    assert len(gfx1250_4wave_co_kernels_list) == 219
    assert GFX1250_4WAVE_CO_KIDS == frozenset(gfx1250_4wave_co_kernels_list)
    assert min(GFX1250_4WAVE_CO_KIDS) == 21016
    assert max(GFX1250_4WAVE_CO_KIDS) == 21315
    assert GFX1250_4WAVE_CO_KIDS <= NON_SPLITK_KIDS
    assert GFX1250_4WAVE_CO_KIDS <= DEFAULT_COMPILED_KIDS_BY_ARCH["gfx1250"]
    assert GFX1250_4WAVE_CO_KIDS.isdisjoint(SPLITK_KIDS)
    assert GFX1250_4WAVE_CO_KIDS.isdisjoint(BIAS_AWARE_KIDS)
    assert {
        "a16w16_4wave_co",
        "a16w16_4wave_wl_co",
    } <= OPUS_KERNEL_TAGS_BY_ARCH_FAMILY["gfx1250"]["a16w16"]

    kid = min(GFX1250_4WAVE_CO_KIDS)
    assert get_kernel_instance(
        "gfx1250", "a16w16", kid, torch.bfloat16
    ) is gfx1250_4wave_co_kernels_list[kid]
    assert get_kernel_instance("gfx1250", "a16w16", kid, torch.float32) is None
    assert not kernel_needs_external_workspace("gfx1250", "a16w16", kid)

    for split_k in (0, 1):
        plan = _co_plan(kid, split_k=split_k)
        assert plan.resolved_kid == kid
        assert plan.abi_split_k == split_k
        assert plan.workspace_spec is None

    with pytest.raises(ValueError, match="does not support split-K"):
        _co_plan(kid, split_k=2)
    with pytest.raises(ValueError, match="does not support output dtype"):
        _co_plan(kid, output_dtype=torch.float32)
    with pytest.raises(ValueError, match="does not support bias"):
        _co_plan(kid, has_bias=True)
    with pytest.raises(ValueError, match="incompatible with shape"):
        _co_plan(kid, batch=2)


def test_gfx1250_co_assets_and_host_only_codegen(tmp_path, monkeypatch):
    symbols_by_kid = {
        kid: instance.name
        for kid, instance in gfx1250_4wave_co_kernels_list.items()
    }
    assert len(set(symbols_by_kid.values())) == 219

    for instance in gfx1250_4wave_co_kernels_list.values():
        image = Path(co_image_path(CO_KERNELS_JSON, instance))
        with image.open("rb") as stream:
            assert stream.read(4) == b"\x7fELF"
        assert instance.splitk_workspace_dtype is None

    image_dir = Path(CO_KERNELS_JSON).parent / "gfx1250"
    build_info = json.loads((image_dir / "build_info.json").read_text())
    assert len(build_info["kernels"]) == 219
    assert {entry["kernarg_segment_size"] for entry in build_info["kernels"]} == {
        64
    }
    assert {
        entry["kid"]: entry["symbol"] for entry in build_info["kernels"]
    } == symbols_by_kid
    assert {image.stem for image in image_dir.glob("*.co")} == set(
        symbols_by_kid.values()
    )

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "csrc" / "opus_gemm")
    )
    from codegen.gen_instances_gfx1250 import (
        KARGS_NAME_MAP,
        TRAITS_HEADER_MAP,
        TRAITS_NAME_MAP,
        gen_4wave_co_instance,
    )

    instance = next(iter(gfx1250_4wave_co_kernels_list.values()))
    codegen = SimpleNamespace(
        impl_path=str(tmp_path),
        _host_instantiations=[],
        _device_instantiations=[],
    )
    gen_4wave_co_instance(
        codegen,
        instance,
        traits_header=TRAITS_HEADER_MAP[instance.kernel_tag],
        traits_name=TRAITS_NAME_MAP[instance.kernel_tag],
        kargs_name=KARGS_NAME_MAP[instance.kernel_tag],
    )
    generated = (tmp_path / f"{instance.name}.cuh").read_text()
    assert "opus_co_launch_gfx1250<Traits>" in generated
    assert "aiter_tensor_t &workspace" not in generated
    assert "splitK == 0 || splitK == 1" in generated
    assert len(codegen._host_instantiations) == 1
    assert codegen._device_instantiations == []


def test_gfx1250_co_loader_handles_missing_assets(tmp_path, capsys):
    assert _load_co_kernels(tmp_path / "missing.json") == {}

    document = json.loads(Path(CO_KERNELS_JSON).read_text())
    document["gfx1250"] = document["gfx1250"][:1]
    manifest = tmp_path / "co_kernels.json"
    manifest.write_text(json.dumps(document))

    assert _load_co_kernels(manifest) == {}
    assert "1 pre-compiled (.co) kid(s) dropped" in capsys.readouterr().err
    assert len(_load_co_kernels(manifest, require_image=False)) == 1


@pytest.mark.parametrize("duplicate", ["kid", "name"])
def test_gfx1250_co_loader_rejects_duplicate_identity(tmp_path, duplicate):
    document = json.loads(Path(CO_KERNELS_JSON).read_text())
    first = document["gfx1250"][0]
    second = json.loads(json.dumps(first))
    if duplicate == "name":
        second["kid"] += 1
    document["gfx1250"] = [first, second]
    manifest = tmp_path / "co_kernels.json"
    manifest.write_text(json.dumps(document))

    expected = "duplicate co kid" if duplicate == "kid" else "same symbol"
    with pytest.raises(AssertionError, match=expected):
        _load_co_kernels(manifest, require_image=False)


def test_gfx1250_tuner_selects_co_without_split_k(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "csrc" / "opus_gemm")
    )
    from opus_gemm_tune import (
        _gfx1250_select_candidates,
        candidate_splitK,
        kid_rejects_shape,
    )

    selected = _gfx1250_select_candidates(64, 128, 4096, 256)
    assert selected & GFX1250_4WAVE_CO_KIDS
    assert selected.isdisjoint(range(27000, 30000))

    instance = gfx1250_4wave_co_kernels_list[min(GFX1250_4WAVE_CO_KIDS)]
    assert candidate_splitK(64, 128, 4096, 1, 256, instance) == [0]
    assert not kid_rejects_shape(instance, 65, 129, 4097)

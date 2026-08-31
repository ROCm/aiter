# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Integration coverage for gfx1250 pre-built OPUS A16W16 kernels."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aiter.ops.opus import opus_gemm
from aiter.ops.opus._arch import GFX1250, SUPPORTED_OPUS_ARCHES
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
        "arch": GFX1250,
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


def _runtime_arch() -> str | None:
    if torch.version.hip is None or not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(properties.gcnArchName).split(":", 1)[0].lower()


@pytest.fixture(scope="module")
def _gfx1250_device() -> torch.device:
    arch = _runtime_arch()
    if arch != GFX1250:
        pytest.skip(f"requires gfx1250 hardware, got {arch!r}")
    return torch.device("cuda", torch.cuda.current_device())


def _run_gfx1250_co_case(kid: int, device: torch.device | None = None) -> None:
    instance = gfx1250_4wave_co_kernels_list[kid]
    if device is None:
        arch = _runtime_arch()
        if arch != GFX1250:
            raise RuntimeError(f"requires gfx1250 hardware, got {arch!r}")
        device = torch.device("cuda", torch.cuda.current_device())

    M = instance.B_M * instance.cluster_wg_m + 1
    N = instance.B_N * instance.cluster_wg_n + 1
    K = instance.B_K * instance.num_slots + 1
    context = (
        f"kid={kid}, symbol={instance.name}, tag={instance.kernel_tag}, "
        f"tile=({instance.B_M},{instance.B_N},{instance.B_K}), "
        f"cluster=({instance.cluster_wg_m},{instance.cluster_wg_n}), "
        f"shape=({M},{N},{K})"
    )

    generator = torch.Generator(device=device).manual_seed(kid)
    A = torch.randn(
        (M, K), device=device, dtype=torch.bfloat16, generator=generator
    )
    B = torch.randn(
        (N, K), device=device, dtype=torch.bfloat16, generator=generator
    )
    Y = torch.full((M, N), float("nan"), device=device, dtype=torch.bfloat16)

    actual = opus_gemm(A, B, Y, kid=kid, split_k=0)
    torch.cuda.synchronize(device)
    assert actual is Y, context
    assert torch.isfinite(actual).all().item(), context

    reference = A.float() @ B.float().T
    try:
        torch.testing.assert_close(actual.float(), reference, rtol=0.03, atol=0.5)
    except AssertionError as error:
        raise AssertionError(f"{context}\n{error}") from error


def _run_gfx1250_co_override_case(available_kid: int, missing_kid: int) -> None:
    override_root = Path(os.environ["OPUS_GEN_CO_DIR"])
    missing_instance = gfx1250_4wave_co_kernels_list[missing_kid]
    missing_image = override_root / GFX1250 / f"{missing_instance.name}.co"
    try:
        _run_gfx1250_co_case(missing_kid)
    except RuntimeError as error:
        if str(missing_image) not in str(error):
            raise AssertionError(
                f"override lookup did not report expected path {missing_image}: {error}"
            ) from error
    else:
        raise AssertionError(f"override lookup unexpectedly found {missing_image}")

    _run_gfx1250_co_case(available_kid)


def test_co_manifest_uses_supported_architecture_keys():
    document = json.loads(Path(CO_KERNELS_JSON).read_text())
    manifest_arches = {key for key in document if not key.startswith("_")}

    assert GFX1250 in manifest_arches
    assert manifest_arches <= SUPPORTED_OPUS_ARCHES


def test_gfx1250_co_provenance_matches_sources_and_artifacts():
    repo_root = Path(__file__).resolve().parents[1]
    builder = repo_root / "csrc" / "opus_gemm" / "gen_co" / "build_co.py"
    completed = subprocess.run(
        [sys.executable, str(builder), "--verify-provenance"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "verified provenance for 219 gfx1250 CO kernels" in completed.stdout


def test_gfx1250_co_registry_and_launch_contract():
    assert len(gfx1250_4wave_co_kernels_list) == 219
    assert GFX1250_4WAVE_CO_KIDS == frozenset(gfx1250_4wave_co_kernels_list)
    assert min(GFX1250_4WAVE_CO_KIDS) == 21016
    assert max(GFX1250_4WAVE_CO_KIDS) == 21315
    assert GFX1250_4WAVE_CO_KIDS <= NON_SPLITK_KIDS
    assert GFX1250_4WAVE_CO_KIDS <= DEFAULT_COMPILED_KIDS_BY_ARCH[GFX1250]
    assert GFX1250_4WAVE_CO_KIDS.isdisjoint(SPLITK_KIDS)
    assert GFX1250_4WAVE_CO_KIDS.isdisjoint(BIAS_AWARE_KIDS)
    assert {
        "a16w16_4wave_co",
        "a16w16_4wave_wl_co",
    } <= OPUS_KERNEL_TAGS_BY_ARCH_FAMILY[GFX1250]["a16w16"]

    kid = min(GFX1250_4WAVE_CO_KIDS)
    assert (
        get_kernel_instance(GFX1250, "a16w16", kid, torch.bfloat16)
        is gfx1250_4wave_co_kernels_list[kid]
    )
    assert get_kernel_instance(GFX1250, "a16w16", kid, torch.float32) is None
    assert not kernel_needs_external_workspace(GFX1250, "a16w16", kid)

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
        kid: instance.name for kid, instance in gfx1250_4wave_co_kernels_list.items()
    }
    assert len(set(symbols_by_kid.values())) == 219

    for instance in gfx1250_4wave_co_kernels_list.values():
        image = Path(co_image_path(CO_KERNELS_JSON, instance))
        with image.open("rb") as stream:
            assert stream.read(4) == b"\x7fELF"
        assert instance.splitk_workspace_dtype is None

    image_dir = Path(CO_KERNELS_JSON).parent / GFX1250
    build_info = json.loads((image_dir / "build_info.json").read_text())
    assert len(build_info["kernels"]) == 219
    assert {entry["kernarg_segment_size"] for entry in build_info["kernels"]} == {64}
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


@pytest.mark.parametrize(
    "kid",
    sorted(gfx1250_4wave_co_kernels_list),
    ids=lambda kid: f"kid-{kid}",
)
def test_gfx1250_co_kernel_matches_torch(kid, _gfx1250_device, monkeypatch):
    monkeypatch.setenv("OPUS_GEN_CO_DIR", str(Path(CO_KERNELS_JSON).parent))
    _run_gfx1250_co_case(kid, _gfx1250_device)


def test_gfx1250_co_dir_override_matches_torch(_gfx1250_device, tmp_path):
    kid = max(gfx1250_4wave_co_kernels_list)
    missing_kid = min(gfx1250_4wave_co_kernels_list)
    instance = gfx1250_4wave_co_kernels_list[kid]
    override_root = tmp_path / "co-root"
    override_arch_dir = override_root / GFX1250
    override_arch_dir.mkdir(parents=True)
    shutil.copy2(
        co_image_path(CO_KERNELS_JSON, instance),
        override_arch_dir / f"{instance.name}.co",
    )

    environment = os.environ.copy()
    environment["OPUS_GEN_CO_DIR"] = str(override_root)
    command = (
        "from op_tests.test_opus_co_integration import "
        "_run_gfx1250_co_override_case; "
        f"_run_gfx1250_co_override_case({kid}, {missing_kid})"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_gfx1250_co_loader_handles_missing_assets(tmp_path, capsys):
    assert _load_co_kernels(tmp_path / "missing.json") == {}

    document = json.loads(Path(CO_KERNELS_JSON).read_text())
    document[GFX1250] = document[GFX1250][:1]
    manifest = tmp_path / "co_kernels.json"
    manifest.write_text(json.dumps(document))

    assert _load_co_kernels(manifest) == {}
    assert "1 pre-compiled (.co) kid(s) dropped" in capsys.readouterr().err
    assert len(_load_co_kernels(manifest, require_image=False)) == 1


@pytest.mark.parametrize("duplicate", ["kid", "name"])
def test_gfx1250_co_loader_rejects_duplicate_identity(tmp_path, duplicate):
    document = json.loads(Path(CO_KERNELS_JSON).read_text())
    first = document[GFX1250][0]
    second = json.loads(json.dumps(first))
    if duplicate == "name":
        second["kid"] += 1
    document[GFX1250] = [first, second]
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

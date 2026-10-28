# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Integration coverage for gfx1250 pre-built OPUS A16W16 kernels."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aiter.ops.opus import (
    gemm_a16w16_opus,
    gemm_op_a16w16,
    opus_bmm,
    opus_gemm,
)
from aiter.ops.opus._arch import GFX1250, SUPPORTED_OPUS_ARCHES
from aiter.ops.opus.launch_plan import _get_cached_a16w16_launch_plan
from csrc.opus_gemm.opus_gemm_common import (
    CO_KERNELS_JSON,
    DEFAULT_COMPILED_KIDS_BY_NAMESPACE,
    GFX1250_4WAVE_CO_KIDS,
    GFX1250_SPLITK_FUSE_KIDS,
    OPUS_KERNEL_TAGS_BY_ARCH_FAMILY,
    _load_co_kernels,
    co_image_path,
    get_kernel_instance,
    gfx1250_4wave_co_kernels_list,
    group_kids,
    kernel_needs_external_workspace,
)


def _co_plan(kid: int, **overrides):
    arguments = {
        "arch": GFX1250,
        "M": 128,
        "N": 128,
        "K": 4096,
        "batch": 1,
        "cu_num": 256,
        "has_bias": False,
        "input_dtype": torch.bfloat16,
        "output_dtype": torch.bfloat16,
        "kid": kid,
        "split_k": 0,
    }
    arguments.update(overrides)
    return _get_cached_a16w16_launch_plan(**arguments)


def _co_test_root() -> Path:
    # CO symbols are cached per process in the host launcher. Run baseline and
    # named candidates in separate pytest processes, never switch mid-process.
    return Path(os.environ.get("OPUS_TEST_CO_ROOT", str(Path(CO_KERNELS_JSON).parent)))


def _co_test_profile_dirs(root: Path) -> list[tuple[Path, str]]:
    """Image directories plus their relative layout for combined/direct roots."""
    if (root / "build_info.json").is_file():
        return [(root, "")]
    return [
        (root / name, name)
        for name in (GFX1250, f"{GFX1250}-a0")
        if (root / name / "build_info.json").is_file()
    ]


def _runtime_arch() -> str | None:
    if torch.version.hip is None or not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(properties.gcnArchName).split(":", 1)[0].lower()


def _representative_kids(kids) -> tuple[int, ...]:
    ordered = sorted(kids)
    representatives = (ordered[0], ordered[len(ordered) // 2], ordered[-1])
    return tuple(dict.fromkeys(representatives))


def _gfx1250_workspace_representatives() -> tuple[int, ...]:
    by_tag = {}
    for kid in sorted(group_kids("SPLITK_KIDS", GFX1250, cu_num=256)):
        instance = get_kernel_instance(GFX1250, "a16w16", kid, cu_num=256)
        if instance is not None:
            by_tag.setdefault(instance.kernel_tag, kid)
    return tuple(by_tag.values())


_CO_ROUTE_KIDS = _representative_kids(GFX1250_4WAVE_CO_KIDS)


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

    # A0 bring-up runtimes may not provide working GPU RNG/reference kernels.
    # This explicit mode keeps the same shapes and strong checks with CPU data
    # generation/golden; the selected CO still executes on the real device.
    cpu_reference = os.environ.get("OPUS_CO_CPU_REFERENCE") == "1"
    random_device = "cpu" if cpu_reference else device
    generator = torch.Generator(device=random_device).manual_seed(kid)
    A_source = torch.randn(
        (M, K), device=random_device, dtype=torch.bfloat16, generator=generator
    )
    B_source = torch.randn(
        (N, K), device=random_device, dtype=torch.bfloat16, generator=generator
    )
    A, B = A_source.to(device), B_source.to(device)
    Y = torch.full((M, N), float("nan"), device=device, dtype=torch.bfloat16)

    actual = opus_gemm(A, B, Y, family="a16w16", kid=kid, split_k=0)
    torch.cuda.synchronize(device)
    assert actual is Y, context
    checked = actual.cpu() if cpu_reference else actual
    assert torch.isfinite(checked).all().item(), context

    reference = A_source.float() @ B_source.float().T
    try:
        torch.testing.assert_close(checked.float(), reference, rtol=0.03, atol=0.5)
    except AssertionError as error:
        raise AssertionError(f"{context}\n{error}") from error


def _run_gfx1250_co_override_case(available_kid: int, missing_kid: int) -> None:
    override_root = Path(os.environ["OPUS_GEN_CO_DIR"])
    missing_instance = gfx1250_4wave_co_kernels_list[missing_kid]
    missing_images = [
        override_root / directory / f"{missing_instance.name}.co"
        for directory in (GFX1250, f"{GFX1250}-a0", "")
    ]
    try:
        _run_gfx1250_co_case(missing_kid)
    except RuntimeError as error:
        if not any(str(path) in str(error) for path in missing_images):
            raise AssertionError(
                f"override lookup did not report a selected missing-image path {missing_images}: {error}"
            ) from error
    else:
        raise AssertionError(
            f"override lookup unexpectedly found a missing image {missing_images}"
        )

    _run_gfx1250_co_case(available_kid)


def test_co_manifest_uses_supported_architecture_keys():
    document = json.loads(Path(CO_KERNELS_JSON).read_text())
    manifest_arches = {key for key in document if not key.startswith("_")}

    assert GFX1250 in manifest_arches
    assert manifest_arches <= SUPPORTED_OPUS_ARCHES


def test_gfx1250_co_registry_and_launch_contract():
    assert len(gfx1250_4wave_co_kernels_list) == 221
    assert GFX1250_4WAVE_CO_KIDS == frozenset(gfx1250_4wave_co_kernels_list)
    assert min(GFX1250_4WAVE_CO_KIDS) == 496
    assert max(GFX1250_4WAVE_CO_KIDS) == 716
    assert GFX1250_4WAVE_CO_KIDS <= group_kids("NON_SPLITK_KIDS", GFX1250, cu_num=256)
    assert GFX1250_4WAVE_CO_KIDS <= DEFAULT_COMPILED_KIDS_BY_NAMESPACE[(GFX1250, 256)]
    assert GFX1250_4WAVE_CO_KIDS.isdisjoint(
        group_kids("SPLITK_KIDS", GFX1250, cu_num=256)
    )
    assert GFX1250_4WAVE_CO_KIDS.isdisjoint(
        group_kids("BIAS_AWARE_KIDS", GFX1250, cu_num=256)
    )
    assert {
        "a16w16_4wave_co",
        "a16w16_4wave_wl_co",
        "a16w16_4wave_wlr_co",
    } <= OPUS_KERNEL_TAGS_BY_ARCH_FAMILY[GFX1250]["a16w16"]

    kid = min(GFX1250_4WAVE_CO_KIDS)
    assert (
        get_kernel_instance(GFX1250, "a16w16", kid, torch.bfloat16, cu_num=256)
        == gfx1250_4wave_co_kernels_list[kid]
    )
    assert (
        get_kernel_instance(GFX1250, "a16w16", kid, torch.float32, cu_num=256) is None
    )
    assert not kernel_needs_external_workspace(GFX1250, "a16w16", kid, cu_num=256)

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


# Registry/artifact checks own exhaustive identity coverage. The route itself is
# kid-independent, so cover both entries/split modes with representative ids.
@pytest.mark.parametrize(
    ("kid", "split_k", "entry"),
    (
        (_CO_ROUTE_KIDS[0], 0, "opus_bmm"),
        (_CO_ROUTE_KIDS[1], 1, "opus_bmm"),
        (_CO_ROUTE_KIDS[2], 0, "gemm_a16w16_opus"),
        (_CO_ROUTE_KIDS[0], 1, "gemm_a16w16_opus"),
    ),
)
def test_gfx1250_co_bmm_reaches_exact_launch(kid, split_k, entry, monkeypatch):
    from aiter.ops.opus import dispatch

    calls = []

    def capture(XQ, WQ, Y, bias, workspace, launched_kid, launch_split_k):
        calls.append((XQ, WQ, Y, bias, workspace, launched_kid, launch_split_k))

    monkeypatch.setattr(dispatch, "_device_arch_and_cu", lambda _: (GFX1250, 256))
    monkeypatch.setattr(gemm_op_a16w16, "_device_arch_and_cu", lambda _: (GFX1250, 256))
    monkeypatch.setattr(gemm_op_a16w16, "_opus_gemm_a16w16_launch_raw", capture)
    A = torch.empty((2, 64, 512), device="meta", dtype=torch.bfloat16)
    B = torch.empty((2, 64, 512), device="meta", dtype=torch.bfloat16)
    Y = torch.empty((2, 64, 64), device="meta", dtype=torch.bfloat16)

    if entry == "opus_bmm":
        actual = opus_bmm(A, B, Y, family="a16w16", kid=kid, split_k=split_k)
    else:
        actual = gemm_a16w16_opus(A, B, kernelId=kid, splitK=split_k, out=Y)

    assert actual is Y
    assert len(calls) == 1
    XQ, WQ, output, bias, workspace, launched_kid, launch_split_k = calls[0]
    assert XQ is A and WQ is B and output is Y
    assert bias is None and workspace is None
    assert (launched_kid, launch_split_k) == (kid, split_k)


@pytest.mark.parametrize(
    "kid",
    _gfx1250_workspace_representatives(),
)
def test_gfx1250_workspace_kids_reject_bmm(kid):
    with pytest.raises(ValueError, match="workspace kids require batch=1"):
        _co_plan(kid, M=1, batch=2)


def test_gfx1250_co_assets_and_host_only_codegen(tmp_path, monkeypatch):
    symbols_by_kid = {
        kid: instance.name for kid, instance in gfx1250_4wave_co_kernels_list.items()
    }
    assert len(set(symbols_by_kid.values())) == 221

    for instance in gfx1250_4wave_co_kernels_list.values():
        image = Path(co_image_path(CO_KERNELS_JSON, instance))
        with image.open("rb") as stream:
            assert stream.read(4) == b"\x7fELF"
        assert instance.splitk_workspace_dtype is None

    image_dir = Path(CO_KERNELS_JSON).parent / GFX1250
    build_info = json.loads((image_dir / "build_info.json").read_text())
    assert len(build_info["kernels"]) == 221
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


def test_gfx1250_named_co_assets_and_placement():
    root = (
        _co_test_root()
        if os.environ.get("OPUS_TEST_CO_ROOT")
        else Path(CO_KERNELS_JSON).parent / "named_reg"
    )
    profiles = _co_test_profile_dirs(root)
    assert profiles, f"no CO build_info found under combined/direct root {root}"
    symbols = {kid: k.name for kid, k in gfx1250_4wave_co_kernels_list.items()}
    for image_dir, _ in profiles:
        build_info = json.loads((image_dir / "build_info.json").read_text())
        assert {
            entry["kid"]: entry["symbol"] for entry in build_info["kernels"]
        } == symbols
        for entry in build_info["kernels"]:
            assert entry["pinning_mode"] == "named-register"
            assert entry["kernarg_segment_size"] == 64
            assert entry["ir_read_register"] > 0 and entry["ir_write_register"] > 0
            assert entry["named_vgpr_tuples"]
            assert entry["wmma_placement_checked"]
            assert entry["wmma_dst_in_named_bank"] == entry["dis_wmma"] > 0
            assert entry["vgpr_spill_count"] == entry["sgpr_spill_count"] == 0
            assert (
                entry["co_sha256"]
                == hashlib.sha256(
                    (image_dir / f"{entry['symbol']}.co").read_bytes()
                ).hexdigest()
            )


@pytest.mark.parametrize(
    "kid",
    sorted(gfx1250_4wave_co_kernels_list),
    ids=lambda kid: f"kid-{kid}",
)
def test_gfx1250_co_kernel_matches_torch(kid, _gfx1250_device, monkeypatch):
    monkeypatch.setenv("OPUS_GEN_CO_DIR", str(_co_test_root()))
    _run_gfx1250_co_case(kid, _gfx1250_device)


@pytest.mark.parametrize("kid", (543, 544, 596, 612, 715, 716))
@pytest.mark.parametrize("case", ("tiny", "one_step", "ring_tail", "steady_tail"))
@pytest.mark.parametrize("split_k", (0, 1))
@pytest.mark.parametrize("pad", (0, 8))
def test_gfx1250_named_co_bmm_and_graph(
    kid, case, split_k, pad, _gfx1250_device, monkeypatch
):
    """Named tuples must survive ring loops and reinitialize on every launch.

    Cover all three CO families, wave layouts, distinct batches, padded source
    rows and both no-split spellings. Poisoning Y before each graph replay catches
    missed stores as well as an accumulator that carries over from a prior run.
    """
    monkeypatch.setenv("OPUS_GEN_CO_DIR", str(_co_test_root()))
    instance = gfx1250_4wave_co_kernels_list[kid]
    if case == "tiny":
        M = N = K = 1
    else:
        M = instance.B_M * instance.cluster_wg_m + 1
        N = instance.B_N * instance.cluster_wg_n + 1
        K = {
            "one_step": instance.B_K - 1,
            "ring_tail": instance.B_K * instance.num_slots + 1,
            "steady_tail": instance.B_K * (2 * instance.num_slots + 1) + 1,
        }[case]
    context = f"kid={kid}, case={case}, split_k={split_k}, pad={pad}, M/N/K={M}/{N}/{K}"
    cpu_reference = os.environ.get("OPUS_CO_CPU_REFERENCE") == "1"
    random_device = "cpu" if cpu_reference else _gfx1250_device
    generator = torch.Generator(device=random_device).manual_seed(kid + K)
    A_source = torch.randn(
        (2, M, K + pad),
        device=random_device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    B_source = torch.randn(
        (2, N, K + pad),
        device=random_device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    # Transfer the complete padded buffers before slicing, preserving source
    # row and batch strides even in CPU-golden mode.
    A = A_source.to(_gfx1250_device)[..., :K]
    B = B_source.to(_gfx1250_device)[..., :K]
    Y = torch.full(
        (2, M, N), float("nan"), device=_gfx1250_device, dtype=torch.bfloat16
    )
    reference = A_source[..., :K].float() @ B_source[..., :K].float().transpose(-1, -2)

    for _ in range(3):
        actual = opus_bmm(A, B, Y, kid=kid, split_k=split_k)
    torch.cuda.synchronize(_gfx1250_device)
    assert actual is Y, context
    checked = Y.cpu() if cpu_reference else Y
    assert torch.isfinite(checked).all().item(), context
    torch.testing.assert_close(
        checked.float(), reference, rtol=0.03, atol=0.5, msg=context
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        opus_bmm(A, B, Y, kid=kid, split_k=split_k)
    for _ in range(3):
        Y.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(_gfx1250_device)
        checked = Y.cpu() if cpu_reference else Y
        assert torch.isfinite(checked).all().item(), context
        torch.testing.assert_close(
            checked.float(), reference, rtol=0.03, atol=0.5, msg=context
        )


def test_gfx1250_co_dir_override_matches_torch(_gfx1250_device, tmp_path):
    kid = max(gfx1250_4wave_co_kernels_list)
    missing_kid = min(gfx1250_4wave_co_kernels_list)
    instance = gfx1250_4wave_co_kernels_list[kid]
    override_root = tmp_path / "co-root"
    profiles = _co_test_profile_dirs(_co_test_root())
    assert profiles, f"no images under {_co_test_root()}"
    for source_dir, relative in profiles:
        override_image_dir = override_root / relative
        override_image_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            source_dir / f"{instance.name}.co",
            override_image_dir / f"{instance.name}.co",
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
    from aiter.ops.opus import _arch

    # Candidate selection needs gfx1250's local table, independent of the host.
    monkeypatch.setattr(_arch, "_device_arch_and_cu", lambda _: (GFX1250, 256))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
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
    assert selected.isdisjoint(GFX1250_SPLITK_FUSE_KIDS)

    instance = gfx1250_4wave_co_kernels_list[min(GFX1250_4WAVE_CO_KIDS)]
    assert candidate_splitK(64, 128, 4096, 1, 256, instance) == [0]
    assert not kid_rejects_shape(instance, 65, 129, 4097)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

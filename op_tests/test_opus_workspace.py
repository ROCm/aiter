# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU and GPU tests for OPUS call-scoped Torch workspaces."""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from csrc.opus_gemm.opus_gemm_common import (
    SPLITK_KIDS,
    GFX1250_SPLITK_FUSE_ENABLED,
    GFX1250_SPLITK_FUSE_KID_OF,
    GFX1250_SPLITK_FUSE_KIDS,
    gfx1250_clusterlaunch_kernels_list,
    gfx1250_kernels_list,
    gfx1250_splitk_fuse_kernels_list,
    kernel_needs_external_workspace,
    kernels_list,
)

from aiter.ops.opus.gemm_op_a16w16 import (
    LaunchConfig,
    _resolve_exact_a16w16_config,
)


def _workspace_config(
    *,
    arch: str = "gfx950",
    kid: int = 200,
    allocation_split_k: int = 2,
    launch_split_k: int = 2,
) -> LaunchConfig:
    return LaunchConfig(
        arch=arch,
        family="a16w16",
        actual_kid=kid,
        allocation_split_k=allocation_split_k,
        launch_split_k=launch_split_k,
    )


def _init_workspace(
    arch: str,
    kid: int,
    *,
    M: int,
    N: int,
    K: int,
    batch: int,
    split_k: int,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor | None:
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    config = _workspace_config(
        arch=arch,
        kid=kid,
        allocation_split_k=split_k,
        launch_split_k=split_k,
    )
    XQ = torch.empty((batch, M, K), dtype=torch.bfloat16)
    Y = torch.empty((batch, M, N), dtype=torch.bfloat16)
    return gemm._init_a16w16_workspace(config, XQ, Y, workspace)


def test_every_workspace_kid_explicitly_declares_storage_dtype():
    assert SPLITK_KIDS
    assert {
        kernels_list[kid].splitk_workspace_dtype for kid in SPLITK_KIDS
    } == {"bf16_t", "fp32_t"}


@pytest.mark.parametrize(
    ("arch", "family", "kid"),
    [
        ("gfx950", "a8w8", 2),
        ("gfx950", "a8w8_blockscale", 1),
        ("gfx942", "a8w8_blockscale_bpreshuffle", 11000),
    ],
)
def test_a8_families_do_not_acquire_workspace_capability(arch, family, kid):
    assert kid not in SPLITK_KIDS
    assert not kernel_needs_external_workspace(arch, family, kid)


def test_gfx1250_two_stage_registry_matches_pr4246_bf16_contract():
    assert len(gfx1250_kernels_list) == 28
    assert len(gfx1250_clusterlaunch_kernels_list) == 468
    assert {
        instance.splitk_workspace_dtype
        for instance in (
            *gfx1250_kernels_list.values(),
            *gfx1250_clusterlaunch_kernels_list.values(),
        )
    } == {"bf16_t"}


def test_gfx1250_fused_family_is_present_but_unregistered():
    assert not GFX1250_SPLITK_FUSE_ENABLED
    assert not gfx1250_splitk_fuse_kernels_list
    assert not GFX1250_SPLITK_FUSE_KIDS
    assert not GFX1250_SPLITK_FUSE_KID_OF
    assert not (set(range(21000, 30000)) & kernels_list.keys())


@pytest.mark.parametrize(
    ("workspace_dtype", "aiter_dtype", "reduce_vec", "workspace_cpp_type"),
    [
        ("bf16_t", "AITER_DTYPE_bf16", 8, "__bf16"),
        ("fp32_t", "AITER_DTYPE_fp32", 8, "float"),
    ],
)
def test_gfx1250_codegen_uses_exact_kid_workspace_dtype(
    tmp_path, monkeypatch, workspace_dtype, aiter_dtype, reduce_vec, workspace_cpp_type
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "csrc" / "opus_gemm")
    )
    from codegen.gen_instances_gfx1250 import (
        KARGS_NAME_MAP,
        KERNEL_FUNC_MAP,
        PIPELINE_HEADER_MAP,
        TRAITS_HEADER_MAP,
        TRAITS_NAME_MAP,
        gen_cluster_tdm_splitk_ws_instance,
    )
    from gen_instances import opus_gemm_codegen

    instance = replace(
        gfx1250_kernels_list[20000],
        splitk_workspace_dtype=workspace_dtype,
    )
    codegen = SimpleNamespace(
        impl_path=str(tmp_path),
        _host_instantiations=[],
        _device_instantiations=[],
    )
    tag = instance.kernel_tag
    gen_cluster_tdm_splitk_ws_instance(
        codegen,
        instance,
        PIPELINE_HEADER_MAP[tag],
        TRAITS_HEADER_MAP[tag],
        KERNEL_FUNC_MAP[tag],
        "bf16_t",
        "bf16_t",
        TRAITS_NAME_MAP[tag],
        KARGS_NAME_MAP[tag],
    )
    generated = (tmp_path / f"{instance.name}.cuh").read_text()
    assert aiter_dtype in generated
    assert f"constexpr int REDUCE_VEC = {reduce_vec};" in generated
    assert f", {workspace_cpp_type}>" in generated

    full_codegen_path = tmp_path / "full"
    full_codegen_path.mkdir()
    opus_gemm_codegen(str(full_codegen_path), False).gen_instances({20000: instance})
    host_tu = (
        full_codegen_path / "instances" / "all_instances_host_gfx1250.cu"
    ).read_text()
    assert "bool HAS_OOB_, int SPLIT_K_, typename D_WS_>" in host_tu
    reduce_tu = (
        full_codegen_path / "instances" / "splitk_reduce_gfx1250.device.cu"
    ).read_text()
    assert "8, 128, __bf16, true, float, true, 0, __bf16>" in reduce_tu
    assert "8, 128, __bf16, true, float, true, 16, float>" in reduce_tu


def test_gfx1250_clusterlaunch_rounds_only_the_physical_grid(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "csrc" / "opus_gemm")
    )
    from codegen.gen_instances_gfx1250 import (
        KARGS_NAME_MAP,
        KERNEL_FUNC_MAP,
        PIPELINE_HEADER_MAP,
        TRAITS_HEADER_MAP,
        TRAITS_NAME_MAP,
        gen_cluster_tdm_splitk_ws_instance,
    )

    instance = gfx1250_clusterlaunch_kernels_list[20100]
    codegen = SimpleNamespace(
        impl_path=str(tmp_path),
        _host_instantiations=[],
        _device_instantiations=[],
    )
    tag = instance.kernel_tag
    gen_cluster_tdm_splitk_ws_instance(
        codegen,
        instance,
        PIPELINE_HEADER_MAP[tag],
        TRAITS_HEADER_MAP[tag],
        KERNEL_FUNC_MAP[tag],
        "bf16_t",
        "bf16_t",
        TRAITS_NAME_MAP[tag],
        KARGS_NAME_MAP[tag],
    )
    generated = (tmp_path / f"{instance.name}.cuh").read_text()
    assert "int grid_tiles_m =" in generated
    assert "int grid_tiles_n =" in generated
    assert "dim3 grid_main(grid_tiles_m, grid_tiles_n, split_k);" in generated
    assert "must both fill the cluster" not in generated
    assert "kargs.stride_ws_batch = static_cast<int>(workspace_slice_numel);" in generated

    pipeline = (
        Path(__file__).resolve().parents[1]
        / "csrc"
        / "opus_gemm"
        / "include"
        / "gfx1250"
        / "opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh"
    ).read_text()
    assert "const bool tile_oob" in pipeline
    assert "if (tile_oob) return;" in pipeline
    assert "defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)" in pipeline


def test_gfx1250_fused_codegen_validates_tile_major_exact_dtype_workspace(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "csrc" / "opus_gemm")
    )
    from codegen.gen_instances_gfx1250 import (
        KARGS_NAME_MAP,
        KERNEL_FUNC_MAP,
        PIPELINE_HEADER_MAP,
        TRAITS_HEADER_MAP,
        TRAITS_NAME_MAP,
        gen_splitk_fuse_instance,
    )
    from gen_instances import opus_gemm_codegen
    from opus_gemm_common import _a16w16_splitk_fuse_gfx1250

    selected = {}
    for synthetic_kid, (workspace_dtype, expected_aiter_dtype, expected_cpp_type) in enumerate((
        ("bf16_t", "AITER_DTYPE_bf16", "__bf16"),
        ("fp32_t", "AITER_DTYPE_fp32", "float"),
    ), start=21000):
        instance = _a16w16_splitk_fuse_gfx1250(
            16, 32, 128, "tileN", 5, 1, workspace_dtype
        )
        kid = synthetic_kid
        selected[kid] = instance
        impl_dir = tmp_path / workspace_dtype
        impl_dir.mkdir()
        codegen = SimpleNamespace(
            impl_path=str(impl_dir),
            _host_instantiations=[],
            _device_instantiations=[],
        )
        tag = instance.kernel_tag
        gen_splitk_fuse_instance(
            codegen,
            instance,
            PIPELINE_HEADER_MAP[tag],
            TRAITS_HEADER_MAP[tag],
            KERNEL_FUNC_MAP[tag],
            "bf16_t",
            "bf16_t",
            TRAITS_NAME_MAP[tag],
            KARGS_NAME_MAP[tag],
        )
        generated = (impl_dir / f"{instance.name}.cuh").read_text()
        assert expected_aiter_dtype in generated
        assert "[num_tiles_m, num_tiles_n, SplitK-1, B_M, B_N]" in generated
        assert "static_cast<size_t>(4)" in generated
        assert f"<Traits, 5, {expected_cpp_type}, 1, __bf16>" in generated
        assert "opus_validate_workspace" in generated
        assert "splitk_reduce_kernel_gfx1250" not in generated

    full_codegen_path = tmp_path / "full_fused"
    full_codegen_path.mkdir()
    opus_gemm_codegen(str(full_codegen_path), False).gen_instances(selected)
    dispatch = (
        full_codegen_path / "opus_gemm_a16w16_kid_dispatch.h"
    ).read_text()
    assert "GENERATE_A16W16_WORKSPACE_KID_DISPATCH_GFX1250_SIZE 2" in dispatch
    assert all(f"{{ {kid}," in dispatch for kid in selected)
    # Fused-only generation must not manufacture a standalone reduce TU.
    assert not (
        full_codegen_path / "instances" / "splitk_reduce_gfx1250.device.cu"
    ).exists()

    pipeline = (
        Path(__file__).resolve().parents[1]
        / "csrc"
        / "opus_gemm"
        / "include"
        / "gfx1250"
        / "opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_fuse_gfx1250.cuh"
    ).read_text()
    assert "opus::tdm<" in pipeline
    assert "opus::make_tdm" in pipeline


def test_gfx1250_bf16_workspace_kid_accepts_fp32_output():
    config = _resolve_exact_a16w16_config(
        arch="gfx1250",
        M=16,
        N=32,
        K=512,
        batch=1,
        cu_num=256,
        has_bias=False,
        input_dtype=torch.bfloat16,
        output_dtype=torch.float32,
        kid=20000,
        split_k=2,
    )
    assert config.actual_kid == 20000
    assert config.allocation_split_k == 2


@pytest.mark.parametrize(
    ("arch", "kid", "M", "N", "K", "batch", "split_k", "shape", "dtype"),
    [
        (
            "gfx950",
            200,
            65,
            33,
            4096,
            2,
            3,
            (3, 2, 128, 64),
            torch.float32,
        ),
        (
            "gfx942",
            10200,
            129,
            129,
            4096,
            2,
            3,
            (3, 2, 256, 256),
            torch.float32,
        ),
        (
            "gfx942",
            10210,
            129,
            384,
            4096,
            2,
            3,
            (3, 2, 256, 384),
            torch.bfloat16,
        ),
        (
            "gfx1250",
            20000,
            17,
            33,
            4096,
            1,
            3,
            (3, 32, 64),
            torch.bfloat16,
        ),
    ],
)
def test_a16w16_workspace_init_uses_actual_kid_tile_and_dtype(
    arch, kid, M, N, K, batch, split_k, shape, dtype
):
    workspace = _init_workspace(
        arch,
        kid,
        M=M,
        N=N,
        K=K,
        batch=batch,
        split_k=split_k,
    )
    assert workspace is not None
    assert workspace.shape == shape
    assert workspace.dtype == dtype
    assert workspace.data_ptr() % 16 == 0


def test_gfx1250_tuner_cannot_select_the_disabled_fused_family(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "csrc" / "opus_gemm")
    )
    from opus_gemm_tune import (
        _gfx1250_fuse_kids_for_tile,
        _gfx1250_select_candidates,
    )

    fused = _gfx1250_fuse_kids_for_tile(
        M=16,
        N=32,
        K=4096,
        cu_num=256,
        bm=16,
        bn=32,
        bk=128,
    )
    with_fused = _gfx1250_select_candidates(64, 128, 4096, 256)
    without_fused = _gfx1250_select_candidates(
        64, 128, 4096, 256, include_fused=False
    )
    assert fused == []
    assert with_fused == without_fused
    assert with_fused
    assert with_fused.isdisjoint(range(21000, 30000))


@pytest.mark.parametrize(
    ("arch", "kid"),
    [
        ("gfx950", 300),
        ("gfx942", 10000),
        # Despite being an atomic split path, this kid has no external partial
        # workspace and must not be classified by its name or numeric band.
        ("gfx942", 10310),
    ],
)
def test_non_workspace_a16w16_kids_initialize_no_workspace(arch, kid):
    assert (
        _init_workspace(arch, kid, M=64, N=64, K=4096, batch=1, split_k=1)
        is None
    )


def test_gfx942_exact_bf16_workspace_kid_is_not_redirected():
    with pytest.raises(ValueError, match="exact kid 10210 requires N"):
        _resolve_exact_a16w16_config(
            arch="gfx942",
            M=128,
            N=768,
            K=4096,
            batch=1,
            cu_num=304,
            has_bias=False,
            input_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            kid=10210,
            split_k=3,
        )

    config = _resolve_exact_a16w16_config(
        arch="gfx942",
        M=128,
        N=384,
        K=4096,
        batch=1,
        cu_num=304,
        has_bias=False,
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        kid=10210,
        split_k=3,
    )
    assert config.actual_kid == 10210

    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    XQ = torch.empty((1, 128, 4096), dtype=torch.bfloat16)
    Y = torch.empty((1, 128, 384), dtype=torch.bfloat16)
    workspace = gemm._init_a16w16_workspace(
        config,
        XQ,
        Y,
    )
    assert workspace is not None
    assert workspace.dtype == torch.bfloat16
    assert workspace.shape == (3, 1, 128, 384)


def test_gfx1250_workspace_init_rejects_batch_greater_than_one():
    with pytest.raises(ValueError, match="require batch=1"):
        _init_workspace(
            "gfx1250", 20000, M=32, N=64, K=4096, batch=2, split_k=3
        )


def test_workspace_init_rejects_split_k_above_per_kid_k_tile_limit():
    with pytest.raises(ValueError, match="exceeds the per-kid K-tile limit 2"):
        _init_workspace("gfx950", 200, M=64, N=64, K=128, batch=1, split_k=3)


def test_explicit_workspace_is_reused_without_allocation():
    workspace = torch.empty(16384, dtype=torch.float32)
    resolved_workspace = _init_workspace(
        "gfx950",
        200,
        M=65,
        N=33,
        K=128,
        batch=1,
        split_k=2,
        workspace=workspace,
    )
    assert resolved_workspace is workspace


def test_prepared_step5_launch_path_allocates_and_passes_workspace():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    config = _workspace_config()
    XQ = torch.empty((1, 65, 128), dtype=torch.bfloat16)
    WQ = torch.empty((1, 33, 128), dtype=torch.bfloat16)
    Y = torch.empty((1, 65, 33), dtype=torch.bfloat16)
    calls = []

    def fake_raw(XQ_, WQ_, Y_, bias_, workspace_, kid_, split_k_):
        calls.append((XQ_, WQ_, Y_, bias_, workspace_, kid_, split_k_))

    result = gemm._launch_a16w16_with_torch_workspace(
        fake_raw, XQ, WQ, Y, None, config
    )
    assert result is Y
    assert len(calls) == 1
    assert calls[0][:4] == (XQ, WQ, Y, None)
    assert calls[0][4].shape == (2, 1, 128, 64)
    assert calls[0][4].dtype == torch.float32
    assert calls[0][5:] == (200, 2)


def test_production_path_allocates_and_passes_call_scoped_workspace(monkeypatch):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    opus = importlib.import_module("aiter.ops.opus")
    monkeypatch.setattr(gemm, "_device_arch_and_cu", lambda _device: ("gfx950", 256))
    gemm._cached_explicit_a16w16_plan.cache_clear()

    calls = []

    def fake_raw(XQ, WQ, Y, bias, workspace, kernelId, splitK):
        calls.append((XQ, WQ, Y, bias, workspace, kernelId, splitK))
        Y.zero_()
        return Y

    monkeypatch.setattr(gemm, "_opus_gemm_a16w16_launch_ctypes_raw", fake_raw)

    XQ = torch.empty((1, 65, 512), dtype=torch.bfloat16)
    WQ = torch.empty((1, 33, 512), dtype=torch.bfloat16)
    Y = torch.empty((1, 65, 33), dtype=torch.bfloat16)
    output = opus.opus_gemm(
        XQ,
        WQ,
        Y,
        kid=200,
    )
    assert output is Y
    assert len(calls) == 1
    XQ, WQ, Y, bias, workspace, kid, split_k = calls[0]
    assert (tuple(XQ.shape), tuple(WQ.shape), tuple(Y.shape)) == (
        (1, 65, 512),
        (1, 33, 512),
        (1, 65, 33),
    )
    assert bias is None
    assert kid == 200
    assert split_k == 0
    assert workspace.shape == (1, 1, 128, 64)
    assert workspace.dtype == torch.float32
    assert workspace.device.type == "cpu"


def test_two_automatic_launches_do_not_share_a_workspace_tensor():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    config = _workspace_config()
    XQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    WQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    workspaces = []

    def fake_raw(_XQ, _WQ, _Y, _bias, workspace, _kid, _split_k):
        workspaces.append(workspace)

    for _ in range(2):
        Y = torch.empty((1, 64, 64), dtype=torch.bfloat16)
        gemm._launch_a16w16_with_torch_workspace(
            fake_raw, XQ, WQ, Y, None, config
        )

    assert len(workspaces) == 2
    assert workspaces[0] is not workspaces[1]
    assert workspaces[0].data_ptr() != workspaces[1].data_ptr()


def test_split_k_limit_fails_before_torch_empty(monkeypatch):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    config = _workspace_config(allocation_split_k=3, launch_split_k=2)
    XQ = torch.empty((1, 64, 128), dtype=torch.bfloat16)
    WQ = torch.empty((1, 64, 128), dtype=torch.bfloat16)
    Y = torch.empty((1, 64, 64), dtype=torch.bfloat16)
    allocations = 0

    def must_not_allocate(*_args, **_kwargs):
        nonlocal allocations
        allocations += 1
        raise AssertionError("invalid split-K reached torch.empty")

    monkeypatch.setattr(gemm.torch, "empty", must_not_allocate)
    with pytest.raises(ValueError, match="exceeds the per-kid K-tile limit 2"):
        gemm._init_a16w16_workspace(config, XQ, Y)
    assert allocations == 0

    raw_calls = []
    monkeypatch.setattr(
        gemm, "_device_arch_and_cu", lambda _device: ("gfx950", 256)
    )
    gemm._cached_explicit_a16w16_plan.cache_clear()
    with pytest.raises(ValueError, match="exceeds the per-kid K-tile limit 2"):
        gemm._explicit_a16w16_launch(
            lambda *_args: raw_calls.append(_args), XQ, WQ, Y, None, 200, 3
        )
    assert raw_calls == []


def test_non_workspace_kid_rejects_explicit_workspace(monkeypatch):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    config = _workspace_config(kid=300, allocation_split_k=1, launch_split_k=0)
    XQ = torch.empty((1, 256, 128), dtype=torch.bfloat16)
    Y = torch.empty((1, 256, 256), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="does not use an external workspace"):
        gemm._init_a16w16_workspace(
            config, XQ, Y, torch.empty(1, dtype=torch.float32)
        )

    WQ = torch.empty((1, 256, 128), dtype=torch.bfloat16)
    raw_calls = []
    monkeypatch.setattr(
        gemm, "_device_arch_and_cu", lambda _device: ("gfx950", 256)
    )
    gemm._cached_explicit_a16w16_plan.cache_clear()
    with pytest.raises(ValueError, match="does not use an external workspace"):
        gemm._explicit_a16w16_launch(
            lambda *_args: raw_calls.append(_args),
            XQ,
            WQ,
            Y,
            None,
            300,
            0,
            workspace=torch.empty(1, dtype=torch.float32),
        )
    assert raw_calls == []


def _runtime_arch(device: int | None = None) -> str | None:
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()


_RAW_CASES = {
    "gfx950": dict(kid=200, M=64, N=64, K=512, split_k=2),
    "gfx942": dict(kid=10200, M=128, N=128, K=512, split_k=2),
    "gfx1250": dict(kid=20000, M=16, N=32, K=512, split_k=2),
}


def _make_raw_case(*, kid: int | None = None):
    arch = _runtime_arch()
    if arch not in _RAW_CASES:
        pytest.skip("requires a gfx942/gfx950/gfx1250 ROCm GPU")
    spec = dict(_RAW_CASES[arch])
    if kid is not None:
        spec["kid"] = kid
    device = torch.device("cuda", torch.cuda.current_device())
    config = _resolve_exact_a16w16_config(
        arch=arch,
        M=spec["M"],
        N=spec["N"],
        K=spec["K"],
        batch=1,
        cu_num=torch.cuda.get_device_properties(device).multi_processor_count,
        has_bias=False,
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        kid=spec["kid"],
        split_k=spec["split_k"],
    )
    XQ = torch.randn(
        (1, spec["M"], spec["K"]), device=device, dtype=torch.bfloat16
    )
    WQ = torch.randn(
        (1, spec["N"], spec["K"]), device=device, dtype=torch.bfloat16
    )
    Y = torch.empty(
        (1, spec["M"], spec["N"]), device=device, dtype=torch.bfloat16
    )
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    workspace = gemm._init_a16w16_workspace(config, XQ, Y)
    assert workspace is not None
    return config, workspace, XQ, WQ, Y


def _raw_launch(case, workspace):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    config, _allocated_workspace, XQ, WQ, Y = case
    return gemm._opus_gemm_a16w16_launch_raw(
        XQ,
        WQ,
        Y,
        None,
        workspace,
        config.actual_kid,
        config.launch_split_k,
    )


@pytest.mark.parametrize(
    ("arch", "kid", "expected_dtype"),
    [
        ("gfx950", 200, torch.float32),
        ("gfx942", 10200, torch.float32),
        ("gfx942", 10210, torch.bfloat16),
        ("gfx1250", 20000, torch.bfloat16),
    ],
)
def test_raw_cpp_accepts_exact_typed_workspace(arch, kid, expected_dtype):
    if _runtime_arch() != arch:
        pytest.skip(f"requires {arch} hardware")
    case = _make_raw_case(kid=kid)
    _config, workspace, _XQ, _WQ, Y = case
    assert workspace.dtype == expected_dtype
    _raw_launch(case, workspace)
    torch.cuda.synchronize(Y.device)
    assert torch.isfinite(Y).all()


def test_raw_cpp_rejects_workspace_one_element_short():
    case = _make_raw_case()
    _config, allocated, _XQ, _WQ, Y = case
    workspace = torch.empty(
        allocated.numel() - 1, device=Y.device, dtype=allocated.dtype
    )
    with pytest.raises(RuntimeError, match="workspace capacity.*elements"):
        _raw_launch(case, workspace)


@pytest.mark.parametrize("failure", ["missing", "dtype", "noncontiguous", "alignment"])
def test_raw_cpp_rejects_invalid_workspace_contract(failure):
    case = _make_raw_case()
    _config, allocated, _XQ, _WQ, Y = case
    if failure == "missing":
        workspace = None
        message = "requires a workspace tensor"
    elif failure == "dtype":
        wrong_dtype = torch.bfloat16 if allocated.dtype == torch.float32 else torch.float32
        workspace = torch.empty(
            allocated.numel(), device=Y.device, dtype=wrong_dtype
        )
        message = "workspace dtype must be"
    elif failure == "noncontiguous":
        workspace = torch.empty(
            (allocated.numel(), 2), device=Y.device, dtype=allocated.dtype
        )[:, 0]
        assert not workspace.is_contiguous()
        message = "workspace must be contiguous"
    else:
        workspace = torch.empty(
            allocated.numel() + 1, device=Y.device, dtype=allocated.dtype
        )[1:]
        assert workspace.is_contiguous()
        assert workspace.data_ptr() % 16 != 0
        message = "workspace address must be aligned"

    with pytest.raises(RuntimeError, match=message):
        _raw_launch(case, workspace)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two ROCm devices for the C++ device-id guard",
)
def test_raw_cpp_rejects_workspace_on_another_device():
    case = _make_raw_case()
    _config, allocated, XQ, _WQ, _Y = case
    input_index = XQ.device.index
    other_index = (input_index + 1) % torch.cuda.device_count()
    workspace = torch.empty(
        allocated.numel(),
        device=torch.device("cuda", other_index),
        dtype=allocated.dtype,
    )
    with pytest.raises(RuntimeError, match="workspace device.*must match input device"):
        _raw_launch(case, workspace)


def test_raw_cpp_non_workspace_kid_requires_none():
    if _runtime_arch() != "gfx950":
        pytest.skip("the checked non-workspace fixture is gfx950-specific")
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    device = torch.device("cuda", torch.cuda.current_device())
    XQ = torch.empty((1, 192, 128), device=device, dtype=torch.bfloat16)
    WQ = torch.empty((1, 64, 128), device=device, dtype=torch.bfloat16)
    Y = torch.empty((1, 192, 64), device=device, dtype=torch.bfloat16)
    workspace = torch.empty(1, device=device, dtype=torch.float32)
    with pytest.raises(
        RuntimeError, match="non-workspace kid 300.*workspace=None"
    ):
        gemm._opus_gemm_a16w16_launch_raw(
            XQ, WQ, Y, None, workspace, 300, 0
        )


def test_gfx1250_raw_cpp_rejects_batch_greater_than_one():
    if _runtime_arch() != "gfx1250":
        pytest.skip("requires gfx1250 hardware")
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    device = torch.device("cuda", torch.cuda.current_device())
    XQ = torch.empty((2, 16, 512), device=device, dtype=torch.bfloat16)
    WQ = torch.empty((2, 32, 512), device=device, dtype=torch.bfloat16)
    Y = torch.empty((2, 16, 32), device=device, dtype=torch.bfloat16)
    workspace = torch.empty(2048, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="supports batch == 1 only"):
        gemm._opus_gemm_a16w16_launch_raw(
            XQ, WQ, Y, None, workspace, 20000, 2
        )

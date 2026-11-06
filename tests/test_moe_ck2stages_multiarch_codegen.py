# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests of the CK 2-stage MoE codegen for a GPU_ARCHS of several archs.

hipcc compiles a module for every GPU_ARCHS arch, so the module must carry each
arch's kernel table, not only the last arch's (#5766).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

CODEGEN = Path(__file__).resolve().parents[1] / "csrc" / "ck_gemm_moe_2stages_codegen"
sys.path.insert(0, str(CODEGEN))
import gemm_moe_ck2stages_common as common
import gen_instances
from gemm_moe_ck2stages_common import get_gemm1_kernels_list, get_gemm2_kernels_list

ARCHS = ("gfx942", "gfx950")
# gen_instances.py arguments, and the get_gemm*_kernels_list arguments they map to.
CONFIGS = {
    "fp8_per_token": (
        "-a f8 -b f8 -c b16 -q per_token -act silu -m 2",
        ("F8", "F8", "B16", 2),
    ),
    "bf16": ("-a b16 -b b16 -c b16 -q no -act silu -m 2", ("B16", "B16", "B16", 0)),
}
HEADS = {
    "ck2stages_moe_stage1_heuristic_dispatch.hpp": gen_instances.gemm1_heuristic_dispatch_head,
    "ck2stages_moe_stage2_heuristic_dispatch.hpp": gen_instances.gemm2_heuristic_dispatch_head,
}


def generate(out, args, gpu_archs):
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(CODEGEN / "gen_instances.py"),
            *args.split(),
            "-w",
            str(out),
        ],
        check=True,
        env={**os.environ, "GPU_ARCHS": gpu_archs},
    )
    return out


def kernel_names(config, arch):
    a, b, c, quant = CONFIGS[config][1]
    _, gemm1 = get_gemm1_kernels_list(a, b, c, False, quant, "silu", False, arch=arch)
    _, gemm2 = get_gemm2_kernels_list(a, b, c, False, quant, True, arch=arch)
    return {kernel.name for kernel in [*gemm1.values(), *gemm2.values()]}


def instances(out):
    return {path.stem: path.read_text() for path in (out / "instances").glob("*.cu")}


def dispatch(out, header):
    text = (out / header).read_text()
    return text[len(HEADS[header]) : -len(gen_instances.heuristic_dispatch_end)]


@pytest.mark.parametrize("config", CONFIGS)
def test_multi_arch_module_carries_each_arch_table(tmp_path, config):
    args = CONFIGS[config][0]
    multi = generate(tmp_path / "multi", args, ";".join(ARCHS))
    tables = {arch: kernel_names(config, arch) for arch in ARCHS}
    assert tables["gfx942"] != tables["gfx950"]

    multi_instances = instances(multi)
    assert set(multi_instances) == set.union(*tables.values())
    for name, text in multi_instances.items():
        used = [arch for arch in ARCHS if name in tables[arch]]
        guard = f"#if !defined(__HIP_DEVICE_COMPILE__) || defined(__{used[0]}__)\n"
        assert text.startswith(guard) == (len(used) == 1), name

    for arch in ARCHS:
        single = generate(tmp_path / arch, args, arch)
        assert set(instances(single)) == tables[arch]
        for header in HEADS:
            # Each arch runs the dispatch a build for it alone has, on it only.
            branch = f'if (gfx == "{arch}")\n    {{{dispatch(single, header)}    }}'
            assert branch in (multi / header).read_text()


def test_table_shared_by_all_archs_is_generated_as_for_one(tmp_path):
    args = "-a f8 -b f8 -c b16 -q per_1x128 -act silu -m 2"
    multi = generate(tmp_path / "multi", args, ";".join(ARCHS))
    single = generate(tmp_path / "single", args, ARCHS[-1])
    files = sorted(p.relative_to(single) for p in single.rglob("*") if p.is_file())
    assert files == sorted(
        p.relative_to(multi) for p in multi.rglob("*") if p.is_file()
    )
    for file in files:
        assert (multi / file).read_text() == (single / file).read_text()


def test_regenerating_in_a_used_directory_rewrites_only_changed_instances(tmp_path):
    # A JIT build regenerates a module's sources where its last build left them.
    args = CONFIGS["bf16"][0]
    out = generate(tmp_path / "out", args, ";".join(ARCHS))
    before = {
        path.stem: (path.read_text(), path.stat().st_mtime_ns)
        for path in (out / "instances").glob("*.cu")
    }
    generate(out, args, "gfx90a")
    fresh = instances(generate(tmp_path / "fresh", args, "gfx90a"))
    assert any(before[name][0] != text for name, text in fresh.items())
    for name, text in fresh.items():
        path = out / "instances" / f"{name}.cu"
        assert path.read_text() == text, name
        if before[name][0] == text:
            assert path.stat().st_mtime_ns == before[name][1], name


def test_native_entry_of_an_arch_list_is_the_build_gpu(monkeypatch):
    # GPU_ARCHS="native;gfx950" builds for this machine's GPU and for gfx950.
    monkeypatch.setattr(common, "get_gfx_list", lambda: ["native", "gfx950"])
    monkeypatch.setattr(common, "get_gfx_runtime", lambda: "gfx942")
    tables = common.get_gemm_kernel_tables("B16", "B16", "B16", False, 0, "silu", 2)
    assert [archs for archs, _, _ in tables.values()] == [["gfx942"], ["gfx950"]]

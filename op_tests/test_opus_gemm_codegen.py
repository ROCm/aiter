# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only coverage for the production OPUS subset-generation command."""

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from csrc.opus_gemm.opus_gemm_common import get_kernel_instance

_FAMILY = "a8w8_blockscale_bpreshuffle"
_GENERATE = """
import shlex
import subprocess
import sys
from pathlib import Path

root, build = map(Path, sys.argv[1:])
# setup.py imports the build configuration this way, without importing aiter.
sys.path.insert(0, str(root / "aiter"))
from jit import core

build.mkdir(parents=True)
table = core.AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE_FILE
(build / "mxscale_table_path.txt").write_text(table)
command = core.get_args_of_build("module_deepgemm_opus")["blob_gen_cmd"]
subprocess.run([sys.executable, *shlex.split(command.format(str(build)))], check=True)
"""


def _dispatch_rows(header, macro):
    definition = re.search(rf"^#define {macro}(?:[ \t].*)?$", header, re.MULTILINE)
    assert definition is not None, f"missing dispatch macro {macro}"
    block = header[definition.start() :].split("\n\n", 1)[0]
    return {
        (int(cu), int(kid), name)
        for cu, kid, name in re.findall(
            r"\{\s*\{\s*(\d+),\s*(\d+)\s*\},\s*&(\w+)<bf16_t>\s*\}",
            block,
        )
    }


def test_gfx950_clean_codegen_covers_tuned_mxscale_kernels(tmp_path):
    # Use the actual default table merge and JIT blob_gen_cmd, in a fresh
    # process/directory. No tuner sidecar, --tune or extra keys can seed it.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AITER_CONFIG_")
        and key not in ("OPUS_GEN_CO_DIR", "CU_NUM")
    }
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    env.update(
        GPU_ARCHS="gfx950",
        AITER_META_DIR=str(_ROOT),
        AITER_JIT_DIR=str(tmp_path / "jit"),
        AITER_SELECT="latency",
        TMPDIR=str(temporary),
    )
    build = tmp_path / "blob"
    result = subprocess.run(
        [sys.executable, "-c", _GENERATE, str(_ROOT), str(build)],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    table = Path((build / "mxscale_table_path.txt").read_text())
    with table.open() as source:
        rows = [
            row
            for row in csv.DictReader(source)
            if row["gfx"] == "gfx950" and row["libtype"] == "opus"
        ]
    assert rows, "default MXFP8 table must exercise the OPUS build path"
    required = {
        (int(row["cu_num"]), int(row["kernelId"]), row["kernelName"]) for row in rows
    }
    compiled = set(json.loads((build / "compiled_kids_opus.json").read_text()))
    missing = sorted(
        f"{_FAMILY}/gfx950/{cu}/{kid}"
        for cu, kid, _ in required
        if f"{_FAMILY}/gfx950/{cu}/{kid}" not in compiled
    )
    assert not missing, f"{len(missing)} tuned kernels missing: {missing[:10]}"

    required_workspace = set()
    for cu, kid, name in required:
        instance = get_kernel_instance("gfx950", _FAMILY, kid, cu_num=cu)
        assert instance is not None, f"unavailable tuned kernel {cu}/{kid}"
        if instance.splitk_workspace_dtype is not None:
            required_workspace.add((cu, kid, name))

    header = (build / "opus_gemm_a8w8_kid_dispatch.h").read_text()
    direct = _dispatch_rows(
        header, "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX950_BF16"
    )
    required_direct = required - required_workspace
    assert required_direct <= direct, sorted(required_direct - direct)
    if required_workspace:
        workspace = _dispatch_rows(
            header,
            "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_WORKSPACE_KID_DISPATCH_GFX950_BF16",
        )
        assert required_workspace <= workspace, sorted(required_workspace - workspace)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

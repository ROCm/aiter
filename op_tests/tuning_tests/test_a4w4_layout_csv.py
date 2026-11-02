# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CSV merging must preserve independent weight/scale layout winners."""

import ast
import logging
import os
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def config_loader(tmp_path, monkeypatch):
    # Execute the real config merge/check methods without importing GPU/JIT code.
    source = ROOT / "aiter/jit/core.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AITER_CONFIG"
    )
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("update_config_files", "_raise_on_duplicate_shapes")
    ]
    ns = {
        "os": os,
        "AITER_ROOT_DIR": str(tmp_path),
        "logger": logging.getLogger(__name__),
        "mp_lock": lambda path, action: action(),
    }
    exec(  # noqa: S102 - only repository-owned AST nodes are executed.
        compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), ns
    )
    configs = tmp_path / "aiter/configs"
    configs.mkdir(parents=True)
    (configs / "a4w4_blockscale_untuned_gemm.csv").write_text("M,N,K\n32,512,7168\n")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    return ns["AITER_CONFIG"]()


def row(**kw):
    return dict(
        gfx="gfx950", cu_num=256, M=32, N=512, K=7168, us=10, kernelName="triton", **kw
    )


def test_legacy_and_raw_tables_merge_without_rewriting_sources(config_loader, tmp_path):
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    pd.DataFrame([row()]).to_csv(old, index=False)
    pd.DataFrame(
        [row(bpreshuffle=0, scale_preshuffle=0), row(bpreshuffle=0, scale_preshuffle=1)]
    ).to_csv(new, index=False)
    before = [p.read_bytes() for p in (old, new)]
    merged = config_loader.update_config_files(
        str(old) + os.pathsep + str(new), "a4w4_blockscale_tuned_gemm"
    )
    frame = pd.read_csv(merged)
    assert set(map(tuple, frame[["bpreshuffle", "scale_preshuffle"]].to_numpy())) == {
        (1, 1),
        (0, 0),
        (0, 1),
    }
    assert [p.read_bytes() for p in (old, new)] == before


def test_one_table_can_hold_all_layouts(config_loader, tmp_path):
    csv = tmp_path / "mixed.csv"
    pd.DataFrame(
        [row(bpreshuffle=b, scale_preshuffle=s) for b in (0, 1) for s in (0, 1)]
    ).to_csv(csv, index=False)
    before = csv.read_bytes()
    assert config_loader.update_config_files(
        str(csv), "a4w4_blockscale_tuned_gemm"
    ) == str(csv)
    assert csv.read_bytes() == before


def test_blank_layout_cells_keep_legacy_meaning(config_loader, tmp_path):
    csv = tmp_path / "blank.csv"
    pd.DataFrame(
        [
            row(bpreshuffle="", scale_preshuffle=""),
            row(bpreshuffle=0, scale_preshuffle=0),
        ]
    ).to_csv(csv, index=False)
    config_loader.update_config_files(str(csv), "a4w4_blockscale_tuned_gemm")
    assert len(pd.read_csv(csv)) == 2


@pytest.mark.parametrize("rows,k", [(31, 7040), (33, 768), (1, 32)])
def test_scale_conversion_preserves_values_and_zeroes_padding(monkeypatch, rows, k):
    torch = pytest.importorskip("torch")
    # Run the production adapter and shuffle helpers with CPU tensors, without
    # importing the GPU runtime. Check padding as well as the logical roundtrip:
    # nonzero padded scale bytes allow ASM's full K tiles to consume stale data.
    helper_path = ROOT / "aiter/ops/triton/utils/shuffle.py"
    helper_names = {
        "_shuffle_scale_tile_gfx950",
        "shuffle_scale_gemm",
        "unshuffle_scale_gemm",
    }
    helper = types.ModuleType("_a4w4_test.triton.utils.shuffle")
    helper.torch = torch
    nodes = [
        n
        for n in ast.parse(helper_path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in helper_names
    ]
    exec(  # noqa: S102 - only repository-owned AST nodes are executed.
        compile(ast.Module(body=nodes, type_ignores=[]), str(helper_path), "exec"),
        helper.__dict__,
    )
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    adapter_path = ROOT / "aiter/ops/gemm_op_a4w4.py"
    adapter = next(
        n
        for n in ast.parse(adapter_path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_a4w4_convert_scale"
    )
    ns = {"torch": torch, "Tensor": torch.Tensor, "__package__": "_a4w4_test"}
    exec(  # noqa: S102 - only repository-owned AST nodes are executed.
        compile(ast.Module(body=[adapter], type_ignores=[]), str(adapter_path), "exec"),
        ns,
    )
    convert = ns["_a4w4_convert_scale"]
    raw = (
        torch.arange(rows * (k // 32))
        .remainder(254)
        .add(1)
        .to(torch.uint8)
        .reshape(rows, k // 32)
    )
    assert convert(raw, rows, k, False, False) is raw
    shuffled = convert(raw, rows, k, False, True)
    restored = helper.unshuffle_scale_gemm(
        shuffled.reshape(shuffled.shape[0] // 32, -1), arch="gfx950"
    )
    assert torch.equal(restored[:rows, : k // 32], raw)
    assert torch.count_nonzero(restored[rows:]) == 0
    assert torch.count_nonzero(restored[:, k // 32 :]) == 0
    assert torch.equal(convert(shuffled, rows, k, True, False), raw)

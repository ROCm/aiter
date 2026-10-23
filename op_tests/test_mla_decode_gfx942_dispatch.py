# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression test for issue #6189: gfx942 BF16 mla_decode_stage1 dispatch."""

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GFX942_MLA_CSV = REPO_ROOT / "hsa" / "gfx942" / "mla" / "mla_asm.csv"
GFX950_MLA_CSV = REPO_ROOT / "hsa" / "gfx950" / "mla" / "mla_asm.csv"
GFX942_KERNEL_CO = (
    REPO_ROOT
    / "hsa"
    / "gfx942"
    / "mla"
    / "MLA_A16W16_1TG_4W_64mx1_16nx1_Coex0_Msk1_QH16.co"
)
EXPECTED_SYMBOL = "_ZN5aiter39mla_a16w16_qh16_m64x1_n16x1_coex0_mask1E"


def _simulate_get_heuristic_kernel_mla(
    csv_path: Path,
    arch_id: str,
    q_type: str,
    kv_type: str,
    gqa_ratio: int,
    max_seqlen_q: int,
    persistent: bool = False,
    causal: int = 1,
    lse: bool = False,
    cprr: bool = False,
):
    """Mirror the dispatch mapping and lookup in csrc/py_itfs_cu/asm_mla.cu."""
    df = pd.read_csv(csv_path)

    ps = 1 if persistent else 0
    prefill = 0
    config_causal = 1 if max_seqlen_q == 1 else causal
    config_max_seqlen_q = max_seqlen_q
    config_gqa_ratio = gqa_ratio

    if gqa_ratio == 128:
        config_max_seqlen_q = 0
        if q_type == "bf16" and kv_type == "bf16" and arch_id == "gfx942":
            ps = 0
    elif gqa_ratio == 16:
        if q_type == "bf16" and kv_type == "bf16":
            if persistent:
                if max_seqlen_q <= 4:
                    config_max_seqlen_q = 4
            elif max_seqlen_q == 1:
                config_max_seqlen_q = 1
            elif max_seqlen_q <= 4:
                config_max_seqlen_q = 4
            else:
                config_max_seqlen_q = 8
    elif gqa_ratio == 32 and q_type == "bf16" and kv_type == "bf16" and not persistent:
        config_max_seqlen_q = 0
    elif gqa_ratio == 64:
        if q_type == "bf16" and kv_type == "bf16":
            if not persistent:
                if max_seqlen_q == 1 and arch_id == "gfx950":
                    config_max_seqlen_q = 1
                else:
                    config_max_seqlen_q = 0
        elif q_type == "fp8" and kv_type == "fp8" and persistent:
            if max_seqlen_q == 1:
                config_max_seqlen_q = 1
            else:
                config_max_seqlen_q = 4
    elif (
        gqa_ratio == 8
        and q_type == "bf16"
        and kv_type == "bf16"
        and not persistent
        and max_seqlen_q == 1
    ):
        config_max_seqlen_q = 1

    lse_flag = 1 if (lse and persistent) else 0
    cprr_flag = 1 if cprr else 0

    # Match rows against the manifest
    for _, row in df.iterrows():
        if row["qType"] != q_type or row["kvType"] != kv_type:
            continue
        if int(row["Gqa"]) != config_gqa_ratio or int(row["ps"]) != ps:
            continue
        if int(row["prefill"]) != prefill:
            continue
        if int(row["causal"]) != config_causal:
            continue
        if int(row["qSeqLen"]) != config_max_seqlen_q:
            continue
        if int(row.get("lse", 0)) != lse_flag:
            continue
        if int(row.get("cprr", 0)) != cprr_flag:
            continue
        return row["knl_name"], row["co_name"]

    return None, None


def test_gfx942_bf16_gqa64_csv_entry():
    """Verify gfx942 bf16 gqa=64 manifest entry has qSeqLen=0 to serve both qlen=1 and qlen>1."""
    assert GFX942_MLA_CSV.exists(), f"Manifest missing: {GFX942_MLA_CSV}"
    df = pd.read_csv(GFX942_MLA_CSV)

    matched = df[
        (df["qType"] == "bf16")
        & (df["kvType"] == "bf16")
        & (df["Gqa"] == 64)
        & (df["ps"] == 0)
        & (df["prefill"] == 0)
        & (df["causal"] == 1)
        & (df["lse"] == 0)
    ]
    assert (
        len(matched) == 1
    ), f"Expected exactly 1 matching row for gfx942 bf16 gqa=64 ps=0, found {len(matched)}"

    row = matched.iloc[0]
    assert (
        int(row["qSeqLen"]) == 0
    ), f"Expected qSeqLen == 0 in manifest (serves both qlen=1 and qlen>1), got {row['qSeqLen']}"
    assert (
        row["knl_name"] == EXPECTED_SYMBOL
    ), f"Unexpected kernel symbol: {row['knl_name']}"
    assert (
        row["co_name"] == GFX942_KERNEL_CO.name
    ), f"Unexpected code object name: {row['co_name']}"


def test_gfx942_bf16_gqa64_code_object_validity():
    """Verify that MLA_A16W16_1TG_4W_64mx1_16nx1_Coex0_Msk1_QH16.co is a valid gfx942 ELF."""
    assert GFX942_KERNEL_CO.exists(), f"Code object missing: {GFX942_KERNEL_CO}"
    data = GFX942_KERNEL_CO.read_bytes()
    assert len(data) > 0, "Code object is empty"

    # ELF64 LE header
    assert data[:4] == b"\x7fELF", "Not an ELF binary"
    ei_class = data[4]
    ei_data = data[5]
    assert ei_class == 2, "Expected 64-bit ELF"
    assert ei_data == 1, "Expected little-endian ELF"

    e_machine = struct.unpack_from("<H", data, 18)[0]
    e_flags = struct.unpack_from("<I", data, 48)[0]
    assert e_machine == 0xE0, f"Expected AMDGPU machine (0xe0), got 0x{e_machine:x}"
    mach = e_flags & 0xFF
    # 0x4c == 76 == EF_AMDGPU_MACH_AMDGCN_GFX942
    assert mach == 0x4C, f"Expected gfx942 target architecture (0x4c), got 0x{mach:x}"

    # Symbol table verification
    e_shoff = struct.unpack_from("<Q", data, 40)[0]
    e_shentsize = struct.unpack_from("<H", data, 58)[0]
    e_shnum = struct.unpack_from("<H", data, 60)[0]
    e_shstrndx = struct.unpack_from("<H", data, 62)[0]
    shstr_offset = struct.unpack_from(
        "<Q", data, e_shoff + e_shstrndx * e_shentsize + 24
    )[0]

    sections = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_name, sh_type, _, _, sh_offset, sh_size, sh_link, _ = struct.unpack_from(
            "<IIQQQQII", data, off
        )
        end = data.find(b"\0", shstr_offset + sh_name)
        sec_name = data[shstr_offset + sh_name : end].decode("ascii", errors="replace")
        sections.append(
            {
                "name": sec_name,
                "type": sh_type,
                "offset": sh_offset,
                "size": sh_size,
                "link": sh_link,
            }
        )

    symbols = set()
    for s in sections:
        if s["type"] in (2, 11):  # SHT_SYMTAB, SHT_DYNSYM
            strtab = sections[s["link"]]
            str_data = data[strtab["offset"] : strtab["offset"] + strtab["size"]]
            sym_count = s["size"] // 24
            for j in range(sym_count):
                sym_off = s["offset"] + j * 24
                st_name = struct.unpack_from("<I", data, sym_off)[0]
                end = str_data.find(b"\0", st_name)
                sym_name = str_data[st_name:end].decode("utf-8", errors="replace")
                if sym_name:
                    symbols.add(sym_name)

    assert (
        EXPECTED_SYMBOL in symbols
    ), f"Symbol {EXPECTED_SYMBOL} not found in code object symbols: {symbols}"


def test_heuristic_kernel_selection_simulation():
    """Verify kernel selection heuristic resolves to the gfx942 kernel when max_seqlen_q=1."""
    knl_name, co_name = _simulate_get_heuristic_kernel_mla(
        csv_path=GFX942_MLA_CSV,
        arch_id="gfx942",
        q_type="bf16",
        kv_type="bf16",
        gqa_ratio=64,
        max_seqlen_q=1,
        persistent=False,
    )
    assert knl_name == EXPECTED_SYMBOL, f"Expected {EXPECTED_SYMBOL}, got {knl_name}"
    assert (
        co_name == GFX942_KERNEL_CO.name
    ), f"Expected {GFX942_KERNEL_CO.name}, got {co_name}"

    # Sequence length 2 on gfx942: routes to 64mx1 (multi-token decode supported via grid tiling)
    knl_name_q2, co_name_q2 = _simulate_get_heuristic_kernel_mla(
        csv_path=GFX942_MLA_CSV,
        arch_id="gfx942",
        q_type="bf16",
        kv_type="bf16",
        gqa_ratio=64,
        max_seqlen_q=2,
        persistent=False,
    )
    assert (
        knl_name_q2 == EXPECTED_SYMBOL
    ), f"Expected {EXPECTED_SYMBOL}, got {knl_name_q2}"
    assert (
        co_name_q2 == GFX942_KERNEL_CO.name
    ), f"Expected {GFX942_KERNEL_CO.name}, got {co_name_q2}"

    # Sequence length 4 on gfx942
    knl_name_q4, co_name_q4 = _simulate_get_heuristic_kernel_mla(
        csv_path=GFX942_MLA_CSV,
        arch_id="gfx942",
        q_type="bf16",
        kv_type="bf16",
        gqa_ratio=64,
        max_seqlen_q=4,
        persistent=False,
    )
    assert (
        knl_name_q4 == EXPECTED_SYMBOL
    ), f"Expected {EXPECTED_SYMBOL}, got {knl_name_q4}"
    assert (
        co_name_q4 == GFX942_KERNEL_CO.name
    ), f"Expected {GFX942_KERNEL_CO.name}, got {co_name_q4}"

    # Nearby case: gfx942 gqa=16, qseqlen=1
    _, co_gqa16 = _simulate_get_heuristic_kernel_mla(
        csv_path=GFX942_MLA_CSV,
        arch_id="gfx942",
        q_type="bf16",
        kv_type="bf16",
        gqa_ratio=16,
        max_seqlen_q=1,
        persistent=False,
    )
    assert co_gqa16 == "mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co"

    # Nearby case: gfx942 gqa=32, qseqlen=1
    _, co_gqa32 = _simulate_get_heuristic_kernel_mla(
        csv_path=GFX942_MLA_CSV,
        arch_id="gfx942",
        q_type="bf16",
        kv_type="bf16",
        gqa_ratio=32,
        max_seqlen_q=1,
        persistent=False,
    )
    assert co_gqa32 == "MLA_A16W16_1TG_4W_32mx1_16nx1_Coex0_Msk1_QH16.co"

    # Nearby case: gfx942 gqa=128, qseqlen=1
    _, co_gqa128 = _simulate_get_heuristic_kernel_mla(
        csv_path=GFX942_MLA_CSV,
        arch_id="gfx942",
        q_type="bf16",
        kv_type="bf16",
        gqa_ratio=128,
        max_seqlen_q=1,
        persistent=False,
    )
    assert co_gqa128 == "mla_dec_stage1_bf16_a16w16_subQ128_mqa128.co"

    # Cross-architecture case: gfx950 gqa=64, qseqlen=1 resolves to dedicated v3 kernel
    _, co_950_q1 = _simulate_get_heuristic_kernel_mla(
        csv_path=GFX950_MLA_CSV,
        arch_id="gfx950",
        q_type="bf16",
        kv_type="bf16",
        gqa_ratio=64,
        max_seqlen_q=1,
        persistent=False,
    )
    assert co_950_q1 == "mla_a16w16_qh64_qseqlen1_gqaratio64_v3.co"

    # Cross-architecture case: gfx950 gqa=64, qseqlen=2 resolves to generic 64mx1 kernel
    _, co_950_q2 = _simulate_get_heuristic_kernel_mla(
        csv_path=GFX950_MLA_CSV,
        arch_id="gfx950",
        q_type="bf16",
        kv_type="bf16",
        gqa_ratio=64,
        max_seqlen_q=2,
        persistent=False,
    )
    assert co_950_q2 == GFX942_KERNEL_CO.name


def test_codegen_produces_expected_entries():
    """Verify that hsa/codegen.py processes CSVs and generates expected configuration headers."""
    codegen_script = REPO_ROOT / "hsa" / "codegen.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        env = dict(os.environ, AITER_GPU_ARCHS="gfx942;gfx950")
        res = subprocess.run(
            [sys.executable, str(codegen_script), "-m", "mla", "-o", tmpdir],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        assert res.returncode == 0

        header_path = Path(tmpdir) / "asm_mla_configs.hpp"
        assert header_path.exists(), "asm_mla_configs.hpp was not generated"
        content = header_path.read_text(encoding="utf-8")

        # The ADD_CFG line for gfx942 64mx1 should have qSeqLen=0
        expected_macro_substring_942 = (
            '"bf16", "bf16",   64,    0,    0,    0,    1,    0,    0, "gfx942", "mla/", '
            f'"{EXPECTED_SYMBOL}", "{GFX942_KERNEL_CO.name}"'
        )
        assert (
            expected_macro_substring_942 in content
        ), f"Expected gfx942 macro entry not found in generated header:\n{expected_macro_substring_942}\n"

        # The ADD_CFG line for gfx950 v3 should have qSeqLen=1
        expected_macro_substring_950 = (
            '"bf16", "bf16",   64,    0,    1,    0,    1,    0,    0, "gfx950", "mla/", '
            '"_ZN5aiter38mla_a16w16_qh64_qseqlen1_gqaratio64_v3E", "mla_a16w16_qh64_qseqlen1_gqaratio64_v3.co"'
        )
        assert (
            expected_macro_substring_950 in content
        ), f"Expected gfx950 v3 macro entry not found in generated header:\n{expected_macro_substring_950}\n"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

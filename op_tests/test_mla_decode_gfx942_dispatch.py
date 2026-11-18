# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression tests for issue #6189: gfx942 BF16 MLA decode stage-1 dispatch and stage-2 mgc selection.

This test file protects against regressions in both production fixes:
1. Stage-1 C++ dispatch (csrc/py_itfs_cu/asm_mla.cu line 953): Ensures gfx942 routes
   non-persistent BF16 GQA64 decode to config_max_seqlen_q=0, matching the manifest.
2. Stage-2 mgc selection (aiter/mla.py line 727): Ensures gfx942 selects mgc=16
   matching the 16-token KV tile size, preventing valid split partials from being dropped.
"""

import contextlib
import importlib.util
import inspect
import os
import struct
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

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


def _is_gfx942() -> bool:
    """Check whether a real gfx942 AMD GPU is present for live kernel execution."""
    if not torch.cuda.is_available():
        return False
    try:
        from aiter.jit.utils.chip_info import get_gfx

        return get_gfx() == "gfx942"
    except Exception:  # noqa: BLE001
        return False


@contextlib.contextmanager
def isolated_mla_environment():
    """Load the real aiter.mla module in an isolated environment without ROCm hardware.

    Provides minimal stubs for JIT build triggers and foreign hardware checks while
    preserving the full, un-simulated implementation of aiter/mla.py and aiter/utility/dtypes.py.
    Guarantees strict restoration of all sys.modules entries and pre-existing module objects
    upon exit, even if an exception is raised.
    """
    keys_to_track = [k for k in sys.modules if k.startswith(("aiter", "triton"))]
    saved_modules = {k: sys.modules[k] for k in keys_to_track}
    modules_before = set(sys.modules.keys())

    _sentinel = object()
    saved_parent_attrs = {}
    if "aiter" in sys.modules:
        parent_aiter = sys.modules["aiter"]
        saved_parent_attrs["aiter"] = {
            "mla": getattr(parent_aiter, "mla", _sentinel),
            "dtypes": getattr(parent_aiter, "dtypes", _sentinel),
        }
    if "triton" in sys.modules:
        parent_triton = sys.modules["triton"]
        saved_parent_attrs["triton"] = {
            "language": getattr(parent_triton, "language", _sentinel),
            "jit": getattr(parent_triton, "jit", _sentinel),
        }

    try:
        sys.modules["aiter"] = types.ModuleType("aiter")
        sys.modules["aiter"].__path__ = [str(REPO_ROOT / "aiter")]

        mock_chip = MagicMock()
        mock_chip.get_gfx = lambda: "gfx942"
        mock_chip.get_gfx_runtime = lambda: "gfx942"
        mock_chip.get_cu_num = lambda: 304
        sys.modules["aiter.jit"] = types.ModuleType("aiter.jit")
        sys.modules["aiter.jit"].__path__ = [str(REPO_ROOT / "aiter" / "jit")]
        sys.modules["aiter.jit.utils"] = types.ModuleType("aiter.jit.utils")
        sys.modules["aiter.jit.utils"].__path__ = [
            str(REPO_ROOT / "aiter" / "jit" / "utils")
        ]
        sys.modules["aiter.jit.utils.chip_info"] = mock_chip

        mock_enum = MagicMock()
        mock_enum.ActivationType = MagicMock()
        mock_enum.QuantType = MagicMock()
        sys.modules["aiter.ops"] = types.ModuleType("aiter.ops")
        sys.modules["aiter.ops"].__path__ = [str(REPO_ROOT / "aiter" / "ops")]
        sys.modules["aiter.ops.enum"] = mock_enum

        sys.modules["aiter.utility"] = types.ModuleType("aiter.utility")
        sys.modules["aiter.utility"].__path__ = [str(REPO_ROOT / "aiter" / "utility")]

        dtypes_spec = importlib.util.spec_from_file_location(
            "aiter.utility.dtypes", REPO_ROOT / "aiter" / "utility" / "dtypes.py"
        )
        dtypes_mod = importlib.util.module_from_spec(dtypes_spec)
        sys.modules["aiter.utility.dtypes"] = dtypes_mod
        dtypes_spec.loader.exec_module(dtypes_mod)
        sys.modules["aiter"].dtypes = dtypes_mod

        mock_triton = MagicMock()
        mock_triton.__version__ = "3.6.0"

        def mock_jit(*args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]

            def decorator(fn):
                return fn

            return decorator

        mock_triton.jit = mock_jit
        mock_triton.next_power_of_2 = lambda x: 1 << (x - 1).bit_length()
        sys.modules["triton"] = mock_triton
        sys.modules["triton.language"] = MagicMock()

        mock_core = MagicMock()
        mock_core.is_experimental_enabled = lambda: False
        sys.modules["aiter.jit.core"] = mock_core

        mock_guard = MagicMock()
        mock_guard.require_gfx1250_asm = lambda op: None
        sys.modules["aiter.jit.utils.asm_guard"] = mock_guard

        mock_asm_utils = MagicMock()
        mock_asm_utils.get_gfx_from_device = lambda dev: "gfx942"
        sys.modules["aiter.ops.asm"] = types.ModuleType("aiter.ops.asm")
        sys.modules["aiter.ops.asm"].__path__ = [
            str(REPO_ROOT / "aiter" / "ops" / "asm")
        ]
        sys.modules["aiter.ops.asm.asm_utils"] = mock_asm_utils

        sys.modules["aiter.ops.asm.mla_decode_v4"] = MagicMock()
        mock_att = MagicMock()
        mock_att.get_mla_decode_fwd_max_splits = lambda *args: 4
        sys.modules["aiter.ops.attention"] = mock_att

        mla_spec = importlib.util.spec_from_file_location(
            "aiter.mla", REPO_ROOT / "aiter" / "mla.py"
        )
        mla = importlib.util.module_from_spec(mla_spec)
        sys.modules["aiter.mla"] = mla
        mla_spec.loader.exec_module(mla)
        yield mla
    finally:
        current_keys = set(sys.modules.keys())
        for k in current_keys - modules_before:
            sys.modules.pop(k, None)
        for k, mod in saved_modules.items():
            sys.modules[k] = mod
        for k in keys_to_track:
            if k not in saved_modules and k in sys.modules:
                sys.modules.pop(k, None)
        for mod_name, attrs in saved_parent_attrs.items():
            if mod_name in sys.modules:
                mod = sys.modules[mod_name]
                for attr_name, orig_val in attrs.items():
                    if orig_val is _sentinel:
                        if hasattr(mod, attr_name):
                            try:
                                delattr(mod, attr_name)
                            except (AttributeError, TypeError):
                                pass
                    else:
                        try:
                            setattr(mod, attr_name, orig_val)
                        except (AttributeError, TypeError):
                            pass


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
            '"bf16", "bf16",   64,    0,    0,    0,    1,    0,    0,    0,    0, "gfx942", "mla/", '
            f'"{EXPECTED_SYMBOL}", "{GFX942_KERNEL_CO.name}"'
        )
        assert (
            expected_macro_substring_942 in content
        ), f"Expected gfx942 macro entry not found in generated header:\n{expected_macro_substring_942}\n"

        # The ADD_CFG line for gfx950 v3 should have qSeqLen=1
        expected_macro_substring_950 = (
            '"bf16", "bf16",   64,    0,    1,    0,    1,    0,    0,    0,    0, "gfx950", "mla/", '
            '"_ZN5aiter38mla_a16w16_qh64_qseqlen1_gqaratio64_v3E", "mla_a16w16_qh64_qseqlen1_gqaratio64_v3.co"'
        )
        assert (
            expected_macro_substring_950 in content
        ), f"Expected gfx950 v3 macro entry not found in generated header:\n{expected_macro_substring_950}\n"


def test_gfx942_manifest_qseqlen_contract():
    """Verify the architecture constraint that dictates C++ dispatch logic in asm_mla.cu.

    In csrc/py_itfs_cu/asm_mla.cu line 953:
        if(max_seqlen_q == 1 && arch_id == "gfx950") config_max_seqlen_q = 1;
        else config_max_seqlen_q = 0;

    This test confirms that gfx942 only possesses a qSeqLen=0 entry for BF16 GQA64,
    and has 0 entries for qSeqLen=1. Therefore, setting config_max_seqlen_q=1 on gfx942
    is mathematically guaranteed to fail get_heuristic_kernel_mla() lookup.
    """
    df_942 = pd.read_csv(GFX942_MLA_CSV)
    df_950 = pd.read_csv(GFX950_MLA_CSV)

    # gfx942 has qSeqLen=0 for BF16 GQA64 non-persistent decode
    match_942_q0 = df_942[
        (df_942["qType"] == "bf16")
        & (df_942["kvType"] == "bf16")
        & (df_942["Gqa"] == 64)
        & (df_942["ps"] == 0)
        & (df_942["qSeqLen"] == 0)
    ]
    assert len(match_942_q0) == 1, "gfx942 must have exactly 1 qSeqLen=0 entry"

    # gfx942 has NO qSeqLen=1 entry (passing qSeqLen=1 aborts at runtime)
    match_942_q1 = df_942[
        (df_942["qType"] == "bf16")
        & (df_942["kvType"] == "bf16")
        & (df_942["Gqa"] == 64)
        & (df_942["ps"] == 0)
        & (df_942["qSeqLen"] == 1)
    ]
    assert (
        len(match_942_q1) == 0
    ), "gfx942 has no qSeqLen=1 entry; dispatching qSeqLen=1 will crash"

    # In contrast, gfx950 DOES have dedicated qSeqLen=1 v3 entry
    match_950_q1 = df_950[
        (df_950["qType"] == "bf16")
        & (df_950["kvType"] == "bf16")
        & (df_950["Gqa"] == 64)
        & (df_950["ps"] == 0)
        & (df_950["qSeqLen"] == 1)
    ]
    assert len(match_950_q1) == 1, "gfx950 must have dedicated qSeqLen=1 v3 entry"


def test_production_mla_decode_fwd_mgc_selection():
    """Exercise the actual production path in aiter/mla.py to verify mgc selection.

    Calls mla_decode_fwd() directly without simulated branch helpers and intercepts
    _fwd_kernel_stage2_asm to assert the exact mgc passed by production code.

    Mutation proof: Reverting aiter/mla.py line 727 causes mgc to evaluate to 32 on gfx942,
    failing this test with AssertionError (assert 32 == 16).
    """
    with isolated_mla_environment() as mla:
        captured = {}
        sys.modules["aiter"].mla_decode_stage1_asm_fwd = lambda *args, **kwargs: None

        class MockLauncher:

            def __getitem__(self, grid):

                def call(*args, **kwargs):
                    captured["mgc"] = kwargs.get("mgc")
                    captured["grid"] = grid

                return call

        mla._fwd_kernel_stage2_asm = MockLauncher()

        # Common tensor inputs
        q_64 = torch.randn(1, 64, 576, dtype=torch.bfloat16)
        kv_64 = torch.randn(1, 1, 1, 576, dtype=torch.bfloat16)
        o_64 = torch.zeros(1, 64, 512, dtype=torch.bfloat16)
        qo_indptr = torch.tensor([0, 1], dtype=torch.int32)
        kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
        kv_indices = torch.tensor([0], dtype=torch.int32)
        kv_last_page_lens = torch.tensor([1], dtype=torch.int32)

        # 1. gfx942 BF16 GQA64 qlen=1 must resolve to mgc=16 (matches 16-token tile size)
        mla.get_gfx = lambda: "gfx942"
        mla.mla_decode_fwd(
            q=q_64,
            kv_buffer=kv_64,
            o=o_64,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_lens=kv_last_page_lens,
            max_seqlen_q=1,
            num_kv_splits=4,
            num_kv_splits_indptr=torch.tensor([0, 4], dtype=torch.int32),
        )
        assert (
            captured.get("mgc") == 16
        ), f"Expected mgc=16 on gfx942, got {captured.get('mgc')}"

        # 2. gfx950 BF16 GQA64 qlen=1 must resolve to mgc=32 (matches 32-token v3 kernel)
        mla.get_gfx = lambda: "gfx950"
        mla.mla_decode_fwd(
            q=q_64,
            kv_buffer=kv_64,
            o=o_64,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_lens=kv_last_page_lens,
            max_seqlen_q=1,
            num_kv_splits=4,
            num_kv_splits_indptr=torch.tensor([0, 4], dtype=torch.int32),
        )
        assert (
            captured.get("mgc") == 32
        ), f"Expected mgc=32 on gfx950, got {captured.get('mgc')}"

        # 3. gfx942 BF16 GQA64 qlen=2 multi-token decode must resolve to mgc=16
        mla.get_gfx = lambda: "gfx942"
        q_q2 = torch.randn(2, 64, 576, dtype=torch.bfloat16)
        o_q2 = torch.zeros(2, 64, 512, dtype=torch.bfloat16)
        qo_indptr_q2 = torch.tensor([0, 2], dtype=torch.int32)
        mla.mla_decode_fwd(
            q=q_q2,
            kv_buffer=kv_64,
            o=o_q2,
            qo_indptr=qo_indptr_q2,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_lens=kv_last_page_lens,
            max_seqlen_q=2,
            num_kv_splits=4,
            num_kv_splits_indptr=torch.tensor([0, 4], dtype=torch.int32),
        )
        assert (
            captured.get("mgc") == 16
        ), f"Expected mgc=16 for qlen=2, got {captured.get('mgc')}"

        # 4. gfx942 BF16 GQA16 single-token decode preserves mgc=64
        mla.get_gfx = lambda: "gfx942"
        q_16 = torch.randn(1, 16, 576, dtype=torch.bfloat16)
        o_16 = torch.zeros(1, 16, 512, dtype=torch.bfloat16)
        mla.mla_decode_fwd(
            q=q_16,
            kv_buffer=kv_64,
            o=o_16,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_lens=kv_last_page_lens,
            max_seqlen_q=1,
            num_kv_splits=4,
            num_kv_splits_indptr=torch.tensor([0, 4], dtype=torch.int32),
        )
        assert (
            captured.get("mgc") == 64
        ), f"Expected mgc=64 for GQA16, got {captured.get('mgc')}"


def test_isolated_mla_environment_restoration():
    """Verify that isolated_mla_environment() strictly restores all module and attribute state.

    Tests:
      1. Normal context exit without pre-existing modules.
      2. Exception unwinding during context execution.
      3. Pre-existing aiter and triton parent packages, verifying that object identities
         and existing submodule attributes are restored 100% on both clean exit and exception.
    """
    # 1. Normal execution path restoration (no pre-existing modules)
    modules_before = set(sys.modules.keys())
    with isolated_mla_environment() as mla:
        assert sys.modules["aiter"] is not None
        assert "aiter.mla" in sys.modules
        assert mla is not None

    modules_after = set(sys.modules.keys())
    leaked_keys = (modules_after - modules_before) & {
        k for k in sys.modules if k.startswith(("aiter", "triton"))
    }
    assert (
        not leaked_keys
    ), f"isolated_mla_environment leaked modules on clean exit: {leaked_keys}"

    # 2. Exception unwinding restoration
    with (
        pytest.raises(RuntimeError, match="simulated failure"),
        isolated_mla_environment(),
    ):
        raise RuntimeError("simulated failure")

    modules_after_exc = set(sys.modules.keys())
    leaked_keys_exc = (modules_after_exc - modules_before) & {
        k for k in sys.modules if k.startswith(("aiter", "triton"))
    }
    assert (
        not leaked_keys_exc
    ), f"isolated_mla_environment leaked modules on exception exit: {leaked_keys_exc}"

    # 3. Pre-existing aiter and triton packages with submodule attributes
    parent_aiter = types.ModuleType("aiter")
    parent_aiter.custom_attr = "canary_val"
    orig_mla = types.ModuleType("aiter.mla")
    orig_mla.canary_mla = "preexisting_mla"
    parent_aiter.mla = orig_mla
    sub_jit = types.ModuleType("aiter.jit")
    sub_jit.jit_attr = "jit_canary"
    parent_aiter.jit = sub_jit

    parent_triton = types.ModuleType("triton")
    parent_triton.triton_attr = "triton_canary"
    sub_tl = types.ModuleType("triton.language")
    sub_tl.tl_attr = "tl_canary"
    parent_triton.language = sub_tl

    sys.modules["aiter"] = parent_aiter
    sys.modules["aiter.mla"] = orig_mla
    sys.modules["aiter.jit"] = sub_jit
    sys.modules["triton"] = parent_triton
    sys.modules["triton.language"] = sub_tl

    id_aiter = id(parent_aiter)
    id_jit = id(sub_jit)
    id_triton = id(parent_triton)
    id_tl = id(sub_tl)

    attrs_aiter = dict(parent_aiter.__dict__)
    attrs_jit = dict(sub_jit.__dict__)
    attrs_triton = dict(parent_triton.__dict__)
    attrs_tl = dict(sub_tl.__dict__)

    # 3a. Clean exit with pre-existing parent packages
    try:
        with isolated_mla_environment():
            # Inside the context, isolated modules are active
            assert sys.modules["aiter"] is not parent_aiter
            assert sys.modules["triton"] is not parent_triton

        # After clean exit, identities and attributes must be perfectly restored
        assert (
            sys.modules["aiter"] is parent_aiter
            and id(sys.modules["aiter"]) == id_aiter
        ), "Pre-existing aiter identity was not restored"
        assert (
            sys.modules["aiter.mla"] is orig_mla
        ), "Pre-existing aiter.mla in sys.modules was not restored"
        assert (
            getattr(sys.modules["aiter"], "mla", None) is orig_mla
        ), "Pre-existing aiter.mla parent attribute was not restored"
        assert (
            sys.modules["aiter.jit"] is sub_jit
            and id(sys.modules["aiter.jit"]) == id_jit
        ), "Pre-existing aiter.jit identity was not restored"
        assert (
            sys.modules["triton"] is parent_triton
            and id(sys.modules["triton"]) == id_triton
        ), "Pre-existing triton identity was not restored"
        assert (
            sys.modules["triton.language"] is sub_tl
            and id(sys.modules["triton.language"]) == id_tl
        ), "Pre-existing triton.language identity was not restored"

        assert (
            sys.modules["aiter"].__dict__ == attrs_aiter
        ), f"aiter attributes altered: {set(sys.modules['aiter'].__dict__) ^ set(attrs_aiter)}"
        assert (
            sys.modules["aiter.jit"].__dict__ == attrs_jit
        ), "aiter.jit attributes altered"
        assert (
            sys.modules["triton"].__dict__ == attrs_triton
        ), "triton attributes altered"
        assert (
            sys.modules["triton.language"].__dict__ == attrs_tl
        ), "triton.language attributes altered"

        # 3b. Exception unwinding with pre-existing parent packages
        with (
            pytest.raises(ValueError, match="simulated crash"),
            isolated_mla_environment(),
        ):
            raise ValueError("simulated crash")

        assert (
            sys.modules["aiter"] is parent_aiter
            and id(sys.modules["aiter"]) == id_aiter
        ), "Pre-existing aiter identity was not restored on exception"
        assert (
            sys.modules["aiter.mla"] is orig_mla
        ), "Pre-existing aiter.mla in sys.modules was not restored on exception"
        assert (
            getattr(sys.modules["aiter"], "mla", None) is orig_mla
        ), "Pre-existing aiter.mla parent attribute was not restored on exception"
        assert (
            sys.modules["aiter.jit"] is sub_jit
            and id(sys.modules["aiter.jit"]) == id_jit
        ), "Pre-existing aiter.jit identity was not restored on exception"
        assert (
            sys.modules["triton"] is parent_triton
            and id(sys.modules["triton"]) == id_triton
        ), "Pre-existing triton identity was not restored on exception"
        assert (
            sys.modules["triton.language"] is sub_tl
            and id(sys.modules["triton.language"]) == id_tl
        ), "Pre-existing triton.language identity was not restored on exception"

        assert (
            sys.modules["aiter"].__dict__ == attrs_aiter
        ), "aiter attributes altered on exception"
        assert (
            sys.modules["aiter.jit"].__dict__ == attrs_jit
        ), "aiter.jit attributes altered on exception"
        assert (
            sys.modules["triton"].__dict__ == attrs_triton
        ), "triton attributes altered on exception"
        assert (
            sys.modules["triton.language"].__dict__ == attrs_tl
        ), "triton.language attributes altered on exception"
    finally:
        sys.modules.pop("aiter", None)
        sys.modules.pop("aiter.mla", None)
        sys.modules.pop("aiter.jit", None)
        sys.modules.pop("triton", None)
        sys.modules.pop("triton.language", None)


def _verify_genuine_stage1_binding(fn):
    """Verify that dispatch_fn is the canonical AITER production wrapper.

    Distinguishes:
      1. Canonical AITER production wrapper (declared in aiter/ops/attention.py and decorated by compile_ops).
      2. Arbitrary Python function spoofing __name__ and __module__.
      3. Compiled C++ execution (which is invoked through the wrapper during live execution).

    Note: The compile_ops wrapper is Python code that resolves and calls the compiled C symbol.
    inspect.unwrap() proves source provenance and wrapper integrity, while actual C++ symbol
    execution is validated when invoked on GPU.
    """
    assert callable(fn), "dispatch_fn is not callable"
    assert (
        getattr(fn, "__name__", "") == "mla_decode_stage1_asm_fwd"
    ), f"Expected function name 'mla_decode_stage1_asm_fwd', got '{getattr(fn, '__name__', '')}'"

    # 1. Must possess compile_ops decorator wrapper chain (__wrapped__)
    assert hasattr(
        fn, "__wrapped__"
    ), "dispatch_fn is missing compile_ops decorator wrapper chain (__wrapped__)"

    # 2. Source provenance verification via unwrapping
    try:
        unwrapped = inspect.unwrap(fn)
    except Exception as e:
        raise AssertionError(f"Could not unwrap dispatch_fn: {e}") from e

    assert (
        getattr(unwrapped, "__name__", "") == "mla_decode_stage1_asm_fwd"
    ), f"Unwrapped function has unexpected name: '{getattr(unwrapped, '__name__', '')}'"

    expected_file = (REPO_ROOT / "aiter" / "ops" / "attention.py").resolve()
    try:
        actual_file = Path(inspect.getsourcefile(unwrapped)).resolve()
    except Exception as e:
        raise AssertionError(
            f"Could not determine source file of dispatch_fn: {e}"
        ) from e

    assert (
        actual_file == expected_file
    ), f"dispatch_fn originates from unexpected source file: {actual_file} (expected {expected_file})"

    # 3. Canonical package export identity
    try:
        import aiter
        import aiter.ops.attention as att

        assert fn is getattr(
            att, "mla_decode_stage1_asm_fwd", None
        ), "dispatch_fn is not identical to aiter.ops.attention.mla_decode_stage1_asm_fwd"
        assert fn is getattr(
            aiter, "mla_decode_stage1_asm_fwd", None
        ), "dispatch_fn is not identical to top-level aiter.mla_decode_stage1_asm_fwd"
    except (ImportError, AttributeError):
        pass  # On offline environments where aiter top-level is not imported


def test_spoofed_binding_rejection():
    """Negative test proving _verify_genuine_stage1_binding rejects spoofed functions.

    Tests that arbitrary Python functions deliberately spoofing __name__ and __module__
    are strictly rejected by wrapper verification.
    """

    # Case 1: Arbitrary function spoofing __name__ and __module__ without wrapper chain
    def spoofed_fn_1(*args, **kwargs):
        return None

    spoofed_fn_1.__name__ = "mla_decode_stage1_asm_fwd"
    spoofed_fn_1.__module__ = "aiter.ops.attention"

    with pytest.raises(
        AssertionError, match="missing compile_ops decorator wrapper chain"
    ):
        _verify_genuine_stage1_binding(spoofed_fn_1)

    # Case 2: Spoofed function with __wrapped__ pointing to foreign source
    def foreign_fn(*args, **kwargs):
        return None

    foreign_fn.__name__ = "mla_decode_stage1_asm_fwd"

    def spoofed_fn_2(*args, **kwargs):
        return None

    spoofed_fn_2.__name__ = "mla_decode_stage1_asm_fwd"
    spoofed_fn_2.__module__ = "aiter.ops.attention"
    spoofed_fn_2.__wrapped__ = foreign_fn

    with pytest.raises(AssertionError, match="originates from unexpected source file"):
        _verify_genuine_stage1_binding(spoofed_fn_2)

    # Case 3: Spoofed function with cyclic __wrapped__
    def spoofed_fn_3(*args, **kwargs):
        return None

    spoofed_fn_3.__name__ = "mla_decode_stage1_asm_fwd"
    spoofed_fn_3.__module__ = "aiter.ops.attention"
    spoofed_fn_3.__wrapped__ = spoofed_fn_3

    with pytest.raises(AssertionError, match="Could not unwrap dispatch_fn"):
        _verify_genuine_stage1_binding(spoofed_fn_3)


@pytest.mark.skipif(
    not _is_gfx942(), reason="Requires live gfx942 hardware for compiled C++ execution"
)
def test_gfx942_bf16_gqa64_stage1_dispatch_live():
    """Live hardware execution test for C++ stage-1 dispatch in asm_mla.cu.

    Invokes the real compiled mla_decode_stage1_asm_fwd on gfx942 with max_seqlen_q=1.
    When asm_mla.cu line 953 is reverted to pre-PR 057e731ad (max_seqlen_q==1 without arch check),
    config_max_seqlen_q is set to 1, causing get_heuristic_kernel_mla() to abort with:
        'AITER_CHECK: cannot get heuristic kernel!'
    Under the PR fix, config_max_seqlen_q is 0 and dispatch succeeds.
    """
    import aiter

    # 1. Verify canonical AITER production wrapper using provenance and export identity
    dispatch_fn = getattr(aiter, "mla_decode_stage1_asm_fwd", None)
    _verify_genuine_stage1_binding(dispatch_fn)

    device = torch.device("cuda")
    total_s = 1
    nhead = 64
    head_size = 576
    v_head_dim = 512
    num_kv_splits = 4
    num_pages = 8
    page_size = 16

    q = torch.randn(total_s, nhead, head_size, dtype=torch.bfloat16, device=device)
    kv = torch.randn(
        num_pages, page_size, 1, head_size, dtype=torch.bfloat16, device=device
    )
    qo_indptr = torch.tensor([0, total_s], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, total_s * page_size], dtype=torch.int32, device=device)
    kv_page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.tensor([page_size], dtype=torch.int32, device=device)
    num_kv_splits_indptr = torch.tensor(
        [0, num_kv_splits], dtype=torch.int32, device=device
    )

    logits = torch.empty(
        (total_s, num_kv_splits, nhead, v_head_dim), dtype=torch.float32, device=device
    )
    attn_lse = torch.empty(
        (total_s, num_kv_splits, nhead, 1), dtype=torch.float32, device=device
    )
    output = torch.empty(
        (total_s, nhead, v_head_dim), dtype=torch.bfloat16, device=device
    )
    valid_split_count = torch.full(
        (1,), num_kv_splits, dtype=torch.int32, device=device
    )

    # Calling compiled stage1 C++ entrypoint live
    aiter.mla_decode_stage1_asm_fwd(
        q,
        kv,
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        num_kv_splits_indptr,
        None,
        None,
        None,
        1,  # max_seqlen_q = 1
        page_size,
        1,  # nhead_kv
        1.0 / (head_size**0.5),
        logits,
        attn_lse,
        output,
        None,  # final_lse
        None,  # q_scale
        None,  # kv_scale
        None,  # g_kv_indptr
        1,  # cp_world_size
        0,  # cp_rank
        valid_split_count,
        1,  # use_valid_split_count_reduce
        True,  # causal
    )
    torch.cuda.synchronize()


def _ref_mla_decode(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    ctx_len: int,
    sm_scale: float,
    v_head_dim: int = 512,
) -> torch.Tensor:
    """Independent PyTorch reference attention for single-token MLA decode.

    Computes mathematically exact scaled dot-product attention over the full
    KV context [0 : ctx_len] to verify that no split partials were dropped:
      1. Slices active tokens from paged buffer: [ctx_len, 1, head_size]
      2. Key: full 576-dim vector (512 NoPE + 64 RoPE)
      3. Value: 512-dim NoPE slice
      4. Computes softmax(Q @ K.T * scale) @ V in float32 and returns BF16.
    """
    # Active tokens: [ctx_len, 1, head_size]
    kvc = kv_buffer.view(-1, 1, q.shape[-1])[:ctx_len]
    k = kvc.float()
    v = kvc[:, :, :v_head_dim].float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum("hqk,khd->qhd", weights, v)
    return ref_out.to(q.dtype)


@pytest.mark.skipif(
    not _is_gfx942(), reason="Requires live gfx942 hardware for numerical validation"
)
def test_gfx942_bf16_gqa64_numerical_regression():
    """Live hardware numerical validation on gfx942 for short-context multi-split shapes.

    Validates that the combination of mgc=16 and config_max_seqlen_q=0 correctly
    computes MLA decode without truncating split partials or failing numerical tolerance.
    Tests the primary shapes that failed prior to PR #6524:
      - (ctx=128, splits=8): splits 4..7 (tokens 64..127) were dropped under old mgc=32
      - (ctx=21,  splits=16): split 1 (tokens 16..20) was dropped under old mgc=32

    Compares production output against independent PyTorch reference attention.
    Under the original mgc=32 bug, dropped splits produce max absolute differences of
    ~0.70-0.77 (ctx=128) and ~1.23-1.37 (ctx=21), causing the assert_close check to fail.
    """
    import aiter

    # Validate that aiter.mla_decode_fwd is genuine production code and not a lambda/stub
    decode_fn = getattr(aiter, "mla_decode_fwd", None)
    assert callable(decode_fn), "aiter.mla_decode_fwd is missing or not callable"
    assert (
        getattr(decode_fn, "__name__", "") != "<lambda>"
    ), "aiter.mla_decode_fwd is a lambda stub; isolation leaked into live test"

    device = torch.device("cuda")
    test_cases = [
        (128, 8),  # 8 splits with 16-token tiles -> all 8 active
        (21, 16),  # 2 active splits -> old mgc=32 only reduced 1 split
    ]

    for ctx_len, splits in test_cases:
        total_s = 1
        nhead = 64
        head_size = 576
        v_head_dim = 512
        page_size = 16
        num_pages = (ctx_len + page_size - 1) // page_size
        sm_scale = 1.0 / (head_size**0.5)

        q = torch.randn(total_s, nhead, head_size, dtype=torch.bfloat16, device=device)
        kv = torch.randn(
            num_pages, page_size, 1, head_size, dtype=torch.bfloat16, device=device
        )
        qo_indptr = torch.tensor([0, total_s], dtype=torch.int32, device=device)
        kv_indptr = torch.tensor([0, ctx_len], dtype=torch.int32, device=device)
        kv_page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        last_page_len = ctx_len % page_size or page_size
        kv_last_page_lens = torch.tensor(
            [last_page_len], dtype=torch.int32, device=device
        )
        o = torch.zeros(total_s, nhead, v_head_dim, dtype=torch.bfloat16, device=device)
        num_kv_splits_indptr = torch.tensor(
            [0, splits], dtype=torch.int32, device=device
        )

        out, _ = aiter.mla_decode_fwd(
            q=q,
            kv_buffer=kv,
            o=o,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_page_indices,
            kv_last_page_lens=kv_last_page_lens,
            max_seqlen_q=1,
            page_size=page_size,
            sm_scale=sm_scale,
            num_kv_splits=splits,
            num_kv_splits_indptr=num_kv_splits_indptr,
        )
        torch.cuda.synchronize()

        # 1. Sanity check for non-finite values
        assert not torch.isnan(
            out
        ).any(), f"NaN detected in output for ctx={ctx_len}, splits={splits}"
        assert not torch.isinf(
            out
        ).any(), f"Inf detected in output for ctx={ctx_len}, splits={splits}"

        # 2. Independent PyTorch reference attention comparison across full context
        ref_out = _ref_mla_decode(
            q=q,
            kv_buffer=kv,
            ctx_len=ctx_len,
            sm_scale=sm_scale,
            v_head_dim=v_head_dim,
        )

        # Standard BF16 attention tolerance (rtol=1e-2, atol=1e-2 matches checkAllclose in test_common.py)
        torch.testing.assert_close(
            out,
            ref_out,
            rtol=1e-2,
            atol=1e-2,
            msg=f"Regression detected: output deviates from reference on ctx={ctx_len}, splits={splits}",
        )


def test_ref_mla_decode_regression_detection():
    """Verify that _ref_mla_decode detects the finite-but-wrong pre-PR mgc=32 error.

    Runs offline on CPU to mathematically prove that comparing against _ref_mla_decode
    strictly catches the original split-dropping defect:
      - ctx=128: mgc=32 dropped tokens 64..127 (splits 4..7) -> max diff ~0.70-0.77 (> 1e-2)
      - ctx=21:  mgc=32 dropped tokens 16..20  (split 1)      -> max diff ~1.23-1.37 (> 1e-2)
    """
    torch.manual_seed(42)
    head_size = 576
    v_head_dim = 512
    sm_scale = 1.0 / (head_size**0.5)

    # Case 1: ctx=128 (8 splits of 16 tokens)
    q_128 = torch.randn(1, 64, head_size, dtype=torch.bfloat16)
    kv_128 = torch.randn(8, 16, 1, head_size, dtype=torch.bfloat16)
    ref_full_128 = _ref_mla_decode(q_128, kv_128, 128, sm_scale, v_head_dim)

    # Full context against itself must pass
    torch.testing.assert_close(ref_full_128, ref_full_128, rtol=1e-2, atol=1e-2)

    # Simulating pre-PR mgc=32 bug: only 4 splits (64 tokens) reduced, tokens 64..127 dropped
    ref_buggy_128 = _ref_mla_decode(q_128, kv_128, 64, sm_scale, v_head_dim)
    diff_128 = (ref_full_128.float() - ref_buggy_128.float()).abs().max().item()
    assert (
        diff_128 > 0.5
    ), f"Expected large difference from dropped splits on ctx=128, got {diff_128}"
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            ref_buggy_128,
            ref_full_128,
            rtol=1e-2,
            atol=1e-2,
            msg="Should fail when splits are dropped",
        )

    # Case 2: ctx=21 (2 splits: 16 tokens + 5 tokens)
    q_21 = torch.randn(1, 64, head_size, dtype=torch.bfloat16)
    kv_21 = torch.randn(2, 16, 1, head_size, dtype=torch.bfloat16)
    ref_full_21 = _ref_mla_decode(q_21, kv_21, 21, sm_scale, v_head_dim)

    torch.testing.assert_close(ref_full_21, ref_full_21, rtol=1e-2, atol=1e-2)

    # Simulating pre-PR mgc=32 bug: only 1 split (16 tokens) reduced, tokens 16..20 dropped
    ref_buggy_21 = _ref_mla_decode(q_21, kv_21, 16, sm_scale, v_head_dim)
    diff_21 = (ref_full_21.float() - ref_buggy_21.float()).abs().max().item()
    assert (
        diff_21 > 0.5
    ), f"Expected large difference from dropped splits on ctx=21, got {diff_21}"
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            ref_buggy_21,
            ref_full_21,
            rtol=1e-2,
            atol=1e-2,
            msg="Should fail when splits are dropped",
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

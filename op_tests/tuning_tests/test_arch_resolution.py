# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 1: Unit tests for GPU SKU resolution and fp8 flavor normalization
(no GPU required).

gfx942 covers three SKUs with different CU counts -- MI300X (304), MI300A (228)
and MI308X (80) -- and both the tuned-config lookup key and the build-time
codegen filter are keyed on (gfx, cu_num). These tests pin the behaviour that
keeps those two sides agreeing, plus the fp8 e4m3 flavor resolution that makes
the shipped shape lists tunable on gfx942.

Covers:
  - chip_info.get_device_name()       SKU mapping by PCI device id
  - chip_info._get_pci_chip_id()      degrades to None instead of raising
  - chip_info.gfx_from_cu_num()       legacy cu_num -> arch backfill
  - chip_info.get_build_targets()     CU resolution on the GPU_ARCHS path
  - dtypes.normalize_fp8_dtype()      e4m3 flavor resolution
  - shipped untuned shape lists are tunable on the running arch
"""

import collections
import glob
import os
import unittest
from unittest import mock

MI300A_CHIP_ID = 0x74A0
MI300X_CHIP_ID = 0x74A1
MI308X_CHIP_ID = 0x74A2


class _EnvGuard:
    """Set/clear env vars for the duration of a block, restoring afterwards."""

    def __init__(self, **kv):
        self._kv = kv
        self._saved = {}

    def __enter__(self):
        for k, v in self._kv.items():
            self._saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, old in self._saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        return False


class TestDeviceName(unittest.TestCase):
    """get_device_name() must distinguish the three gfx942 SKUs."""

    def _name_for(self, gfx, chip_id):
        from aiter.jit.utils import chip_info

        with mock.patch.object(
            chip_info, "get_gfx", return_value=gfx
        ), mock.patch.object(chip_info, "_get_pci_chip_id", return_value=chip_id):
            return chip_info.get_device_name()

    def test_mi300a_chip_id(self):
        self.assertEqual(self._name_for("gfx942", MI300A_CHIP_ID), "MI300A")

    def test_mi308x_chip_ids(self):
        from aiter.jit.utils import chip_info

        for cid in chip_info.MI308_CHIP_IDS:
            self.assertEqual(self._name_for("gfx942", cid), "MI308")

    def test_mi300x_falls_back_to_mi300(self):
        self.assertEqual(self._name_for("gfx942", MI300X_CHIP_ID), "MI300")

    def test_unknown_chip_id_falls_back_to_mi300(self):
        # An unreadable id must not crash dispatch; MI300 is the safe default.
        self.assertEqual(self._name_for("gfx942", None), "MI300")

    def test_other_arches_unaffected(self):
        self.assertEqual(self._name_for("gfx950", MI300A_CHIP_ID), "MI350")
        self.assertEqual(self._name_for("gfx1250", MI300A_CHIP_ID), "MI400")

    def test_mi300a_id_not_in_mi308_set(self):
        from aiter.jit.utils import chip_info

        self.assertNotIn(MI300A_CHIP_ID, chip_info.MI308_CHIP_IDS)
        self.assertIn(MI300A_CHIP_ID, chip_info.MI300A_CHIP_IDS)


class TestPciChipId(unittest.TestCase):
    """The id is resolved without hardcoding a HIP enum ordinal.

    The AMD block of hipDeviceAttribute_t is an unnumbered enum, so ordinals
    shift between ROCm releases; on ROCm 7.2 the previously hardcoded 10019 is
    MaxAvailableVgprsPerThread and returns 512 on every device.
    """

    # _load_hip_runtime() is lru_cached, so patching ctypes.CDLL underneath it
    # does nothing once the real library has been loaded by any earlier call.
    # Patch the loader itself, which is also the contract these tests care about.
    LOADER = "aiter.jit.utils.chip_info._load_hip_runtime"

    def test_returns_none_when_hip_cannot_be_loaded(self):
        from aiter.jit.utils import chip_info

        # OSError on Linux (missing .so); RuntimeError is how _load_hip_runtime
        # reports "no amdhip64*.dll found" on Windows. Both must degrade, not raise.
        for exc in (OSError("no libamdhip64"), RuntimeError("no HIP runtime")):
            with self.subTest(exc=type(exc).__name__):
                with mock.patch(self.LOADER, side_effect=exc):
                    self.assertIsNone(chip_info._get_pci_chip_id())

    def test_returns_none_when_symbol_is_missing(self):
        from aiter.jit.utils import chip_info

        lib = mock.MagicMock()
        del lib.hipDeviceGetPCIBusId  # attribute lookup raises AttributeError
        with mock.patch(self.LOADER, return_value=lib):
            self.assertIsNone(chip_info._get_pci_chip_id())

    def test_returns_none_when_query_fails(self):
        from aiter.jit.utils import chip_info

        lib = mock.MagicMock()
        lib.hipDeviceGetPCIBusId.return_value = 1  # non-zero == failure
        with mock.patch(self.LOADER, return_value=lib):
            self.assertIsNone(chip_info._get_pci_chip_id())

    def test_returns_none_when_sysfs_is_absent(self):
        # Windows has no /sys/bus/pci; the id is simply unknown there.
        from aiter.jit.utils import chip_info

        lib = mock.MagicMock()
        lib.hipDeviceGetPCIBusId.side_effect = self._fill_bus_id
        with mock.patch(self.LOADER, return_value=lib), mock.patch(
            "builtins.open", side_effect=FileNotFoundError
        ):
            self.assertIsNone(chip_info._get_pci_chip_id())

    @staticmethod
    def _fill_bus_id(buf, size, device_id):
        buf.value = b"0000:01:00.0"
        return 0

    def test_reads_the_id_from_sysfs_via_the_bus_id(self):
        from aiter.jit.utils import chip_info

        lib = mock.MagicMock()
        lib.hipDeviceGetPCIBusId.side_effect = self._fill_bus_id
        opener = mock.mock_open(read_data="0x74a0\n")

        with mock.patch(self.LOADER, return_value=lib), mock.patch(
            "builtins.open", opener
        ):
            self.assertEqual(chip_info._get_pci_chip_id(), MI300A_CHIP_ID)

        opener.assert_called_once_with("/sys/bus/pci/devices/0000:01:00.0/device")
        # hipDeviceGetAttribute takes an ordinal from an unnumbered enum, which
        # is exactly what shifted between ROCm releases. It must not be used.
        lib.hipDeviceGetAttribute.assert_not_called()
        lib.hipDeviceGetPCIBusId.assert_called_once()


class TestLegacyCuNumBackfill(unittest.TestCase):
    """Legacy CSVs predating the gfx column are keyed on cu_num alone."""

    def test_known_cu_nums(self):
        from aiter.jit.utils.chip_info import _LEGACY_CU_NUM_TO_GFX

        self.assertEqual(_LEGACY_CU_NUM_TO_GFX[228], "gfx942")  # MI300A
        self.assertEqual(_LEGACY_CU_NUM_TO_GFX[304], "gfx942")  # MI300X
        self.assertEqual(_LEGACY_CU_NUM_TO_GFX[80], "gfx942")  # MI308X
        self.assertEqual(_LEGACY_CU_NUM_TO_GFX[256], "gfx950")

    def test_gfx_from_cu_num_accepts_str_and_int(self):
        from aiter.jit.utils.chip_info import gfx_from_cu_num

        self.assertEqual(gfx_from_cu_num(228), "gfx942")
        self.assertEqual(gfx_from_cu_num("228"), "gfx942")


class TestBuildTargets(unittest.TestCase):
    """GPU_ARCHS must not silently build for the wrong CU count.

    GFX_CU_NUM_MAP holds one canonical cu_num per arch (gfx942 -> 304). Using it
    verbatim on a 228-CU MI300A makes gen_instances filter out every
    (gfx942, 228) tuned row, so the kernels are never compiled and dispatch
    later fails with "not present in the compiled registry".
    """

    def _targets(self, *, gpu_archs, cu_num=None, live=("gfx942", 228)):
        from aiter.jit.utils import chip_info

        patches = []
        if live is None:
            patches.append(
                mock.patch.object(
                    chip_info, "get_gfx_runtime", side_effect=RuntimeError("no GPU")
                )
            )
        else:
            patches.append(
                mock.patch.object(chip_info, "get_gfx_runtime", return_value=live[0])
            )
            patches.append(
                mock.patch.object(chip_info, "get_cu_num", return_value=live[1])
            )
        with _EnvGuard(GPU_ARCHS=gpu_archs, CU_NUM=cu_num):
            for p in patches:
                p.start()
            try:
                return chip_info.get_build_targets()
            finally:
                for p in patches:
                    p.stop()

    def test_live_cu_count_wins_over_table_default(self):
        self.assertEqual(self._targets(gpu_archs="gfx942"), [("gfx942", 228)])

    def test_explicit_cu_num_overrides_live(self):
        self.assertEqual(
            self._targets(gpu_archs="gfx942", cu_num="304"), [("gfx942", 304)]
        )

    def test_cross_compile_keeps_table_default(self):
        # Building for an arch that is not the live GPU must not inherit its CUs.
        self.assertEqual(self._targets(gpu_archs="gfx950"), [("gfx950", 256)])

    def test_multi_arch_refines_only_the_matching_target(self):
        self.assertEqual(
            self._targets(gpu_archs="gfx942;gfx950"),
            [("gfx942", 228), ("gfx950", 256)],
        )

    def test_no_live_gpu_keeps_table_defaults(self):
        self.assertEqual(
            self._targets(gpu_archs="gfx942", live=None), [("gfx942", 304)]
        )

    def test_mi308x_partition_also_refined(self):
        # Same mechanism protects MI308X and non-SPX MI300X partition modes.
        self.assertEqual(
            self._targets(gpu_archs="gfx942", live=("gfx942", 80)), [("gfx942", 80)]
        )


class TestNormalizeFp8Dtype(unittest.TestCase):
    """Only one e4m3 encoding exists per chip; a shape list naming the other
    describes an unrunnable configuration, not a distinct problem."""

    def test_e4m3_resolves_to_arch_native(self):
        import torch

        from aiter import dtypes

        if dtypes.fp8 not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            self.skipTest("arch has no native e4m3 (fp8 falls back to uint8)")
        for flavor in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            self.assertEqual(dtypes.normalize_fp8_dtype(flavor), dtypes.fp8)

    def test_idempotent(self):
        import torch

        from aiter import dtypes

        once = dtypes.normalize_fp8_dtype(torch.float8_e4m3fn)
        self.assertEqual(dtypes.normalize_fp8_dtype(once), once)

    def test_non_e4m3_passes_through(self):
        import torch

        from aiter import dtypes

        for dt in (torch.float8_e5m2, torch.bfloat16, torch.float32, torch.int8):
            self.assertEqual(dtypes.normalize_fp8_dtype(dt), dt)

    def test_unknown_arch_passes_through(self):
        # fp8 falls back to uint8 on an unknown arch; do not reinterpret fp8
        # data as integers.
        import torch

        from aiter import dtypes

        with mock.patch.object(dtypes, "fp8", torch.uint8):
            self.assertEqual(
                dtypes.normalize_fp8_dtype(torch.float8_e4m3fn), torch.float8_e4m3fn
            )


class TestShippedShapeListsAreTunable(unittest.TestCase):
    """The shipped untuned lists hardcode OCP e4m3fn. After ingest
    normalization every fp8 shape must name the arch-native encoding,
    otherwise it aborts with "Unsupported dtype" and cannot be tuned."""

    DTYPE_COLUMNS = ("q_dtype_a", "q_dtype_w", "dtype")
    # Globs rather than fixed names so a newly added model shape list is
    # covered automatically. a8w8_bpreshuffle is included deliberately: its
    # tuner drops any row whose q_dtype_w != dtypes.fp8, so an unresolved
    # e4m3fn there is silently skipped rather than loudly rejected.
    SHAPE_LIST_GLOBS = (
        "aiter/configs/untuned_fmoe.csv",
        "aiter/configs/a8w8_untuned_gemm.csv",
        "aiter/configs/*bpreshuffle_untuned_gemm*.csv",
        "aiter/configs/model_configs/*bpreshuffle_untuned_gemm*.csv",
        "aiter/configs/model_configs/*untuned_fmoe*.csv",
    )

    def _shape_lists(self, root):
        for pattern in self.SHAPE_LIST_GLOBS:
            yield from sorted(glob.glob(os.path.join(root, pattern)))

    def test_normalization_makes_every_fp8_shape_arch_native(self):
        import csv

        import torch

        from aiter import dtypes
        from aiter.jit.core import AITER_ROOT_DIR
        from aiter.utility.base_tuner import TunerCommon

        if dtypes.fp8 not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            self.skipTest("arch has no native e4m3 (fp8 falls back to uint8)")
        native = str(dtypes.fp8)

        checked = collections.Counter()
        for path in self._shape_lists(AITER_ROOT_DIR):
            name = os.path.basename(path)
            with open(path) as f:
                reader = csv.DictReader(f)
                cols = [c for c in self.DTYPE_COLUMNS if c in (reader.fieldnames or [])]
                if not cols:
                    continue
                for row in reader:
                    for col in cols:
                        raw = (row.get(col) or "").strip()
                        if not raw.startswith("torch.float8_e4m3"):
                            continue
                        # Exercise the shared loader's resolver, which is what
                        # every tuner family now runs on ingest.
                        resolved = TunerCommon._resolve_fp8_name(raw)
                        self.assertEqual(
                            resolved,
                            native,
                            f"{name}:{col}={raw} does not resolve to the "
                            f"arch-native encoding {native}",
                        )
                        checked[name] += 1
        self.assertTrue(checked, "no fp8 shapes found to check")

    def test_bpreshuffle_lists_are_covered(self):
        """The bpreshuffle family is the one that fails silently.

        Its tuner returns [] for any row whose q_dtype_w is not dtypes.fp8, so
        on gfx942 an unresolved e4m3fn shape is skipped with a warning rather
        than raising. Guard that at least one such list is actually in scope.
        """
        from aiter.jit.core import AITER_ROOT_DIR

        lists = [p for p in self._shape_lists(AITER_ROOT_DIR) if "bpreshuffle" in p]
        self.assertTrue(lists, "no bpreshuffle shape lists matched the globs")

    def test_resolver_handles_whitespace_and_unknown_names(self):
        import torch

        from aiter import dtypes
        from aiter.utility.base_tuner import TunerCommon

        if dtypes.fp8 not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            self.skipTest("arch has no native e4m3 (fp8 falls back to uint8)")
        native = str(dtypes.fp8)
        # Leading/trailing whitespace must not defeat the match.
        self.assertEqual(
            TunerCommon._resolve_fp8_name("  torch.float8_e4m3fn "), native
        )
        self.assertEqual(TunerCommon._resolve_fp8_name("torch.float8_e4m3fnuz"), native)
        # Anything else is returned untouched, including values that would be
        # dangerous to eval().
        for passthrough in (
            "torch.int8",
            "torch.bfloat16",
            "torch.float8_e5m2",
            "torch.float8_e4m3_not_a_dtype",
            "__import__('os').system('true')",
            "",
            None,
            3,
        ):
            self.assertEqual(TunerCommon._resolve_fp8_name(passthrough), passthrough)


class TestTunedConfigArchConsistency(unittest.TestCase):
    """Every tuned row keyed on cu_num=228 must resolve to gfx942, or the
    build-time codegen filter and the runtime lookup will disagree.

    Not every tuned family carries a gfx column: the older ones (tuned_fmoe.csv
    among them) are keyed on cu_num alone and have their arch backfilled by
    gfx_from_cu_num at load time. Assert on the *resolved* arch, which is what
    dispatch actually keys on, rather than requiring a literal column.
    """

    def test_cu228_rows_resolve_to_gfx942(self):
        import csv
        import glob

        from aiter.jit.core import AITER_ROOT_DIR
        from aiter.jit.utils.chip_info import gfx_from_cu_num

        cfg = os.path.join(AITER_ROOT_DIR, "aiter/configs")
        found = collections.Counter()
        for path in glob.glob(os.path.join(cfg, "**", "*.csv"), recursive=True):
            if "untuned" in os.path.basename(path):
                continue
            try:
                with open(path) as f:
                    # Stream rather than materializing; some tuned CSVs are
                    # thousands of rows and this is a Level 1 test.
                    reader = csv.DictReader(f)
                    if not reader.fieldnames or "cu_num" not in reader.fieldnames:
                        continue
                    has_gfx = "gfx" in reader.fieldnames
                    for row in reader:
                        if str(row.get("cu_num", "")).strip() != "228":
                            continue
                        name = os.path.basename(path)
                        declared = (row.get("gfx") or "").strip() if has_gfx else ""
                        resolved = declared or gfx_from_cu_num(row["cu_num"])
                        self.assertEqual(
                            resolved,
                            "gfx942",
                            f"{name}: cu_num=228 row resolves to {resolved!r}",
                        )
                        found[name] += 1
            except (OSError, UnicodeDecodeError):
                continue
        self.assertTrue(found, "no MI300A (cu_num=228) tuned rows found")

    def test_legacy_cu_only_families_still_resolve(self):
        """A tuned family without a gfx column must still land on gfx942.

        This is the case the previous version of this test got wrong: it
        required a literal gfx column, which tuned_fmoe.csv does not have.
        """
        import csv

        from aiter.jit.core import AITER_ROOT_DIR
        from aiter.jit.utils.chip_info import gfx_from_cu_num

        path = os.path.join(AITER_ROOT_DIR, "aiter/configs/tuned_fmoe.csv")
        if not os.path.exists(path):
            self.skipTest("tuned_fmoe.csv not present")
        with open(path) as f:
            reader = csv.DictReader(f)
            self.assertNotIn(
                "gfx",
                reader.fieldnames or [],
                "tuned_fmoe.csv grew a gfx column; fold it into the main check",
            )
            cu228 = sum(
                1 for row in reader if str(row.get("cu_num", "")).strip() == "228"
            )
        self.assertGreater(cu228, 0, "no cu_num=228 rows in tuned_fmoe.csv")
        self.assertEqual(gfx_from_cu_num(228), "gfx942")


if __name__ == "__main__":
    unittest.main()

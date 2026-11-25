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
  - chip_info._get_pci_chip_id()      version-gated PciChipId attribute
  - chip_info.gfx_from_cu_num()       legacy cu_num -> arch backfill
  - chip_info.get_build_targets()     CU resolution on the GPU_ARCHS path
  - dtypes.normalize_fp8_dtype()      e4m3 flavor resolution
  - shipped untuned shape lists are tunable on the running arch
"""

import collections
import ctypes
import glob
import os
import unittest
from types import SimpleNamespace
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

    def _name_for(self, gfx, chip_id, num_xcds=8):
        from aiter.jit.utils import chip_info

        props = SimpleNamespace(gcnArchName=f"{gfx}:sramecc+:xnack-")
        with mock.patch.object(
            chip_info, "get_current_hip_device", return_value=0
        ), mock.patch.object(
            chip_info, "_get_pci_chip_id", return_value=chip_id
        ), mock.patch.object(
            chip_info, "_num_xcds_for_device", return_value=num_xcds
        ), mock.patch(
            "torch.cuda.get_device_properties", return_value=props
        ):
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
        # An id outside the binned SKU sets is the generic gfx942 part.
        self.assertEqual(self._name_for("gfx942", None), "MI300")

    def test_other_arches_unaffected(self):
        self.assertEqual(self._name_for("gfx950", MI300A_CHIP_ID), "MI350")
        self.assertEqual(self._name_for("gfx1250", MI300A_CHIP_ID), "MI400")

    def test_mi300a_id_not_in_mi308_set(self):
        from aiter.jit.utils import chip_info

        self.assertNotIn(MI300A_CHIP_ID, chip_info.MI308_CHIP_IDS)
        self.assertIn(MI300A_CHIP_ID, chip_info.MI300A_CHIP_IDS)


class TestPciChipId(unittest.TestCase):
    """MI300A is PCI id 0x74A0, read from the version-gated PciChipId attribute.

    ROCm 7.1 inserted MaxAvailableVgprsPerThread ahead of PciChipId, so the
    ordinal is 10019 before 7.1 and 10020 after. A missing runtime version must
    raise rather than query a shifted ordinal.
    """

    def _device_name(self, chip_id, version):
        from aiter.jit.utils import chip_info

        lib = mock.Mock()

        def query(output, attribute, device):
            expected = 10019 if version[:2] < (7, 1) else 10020
            self.assertEqual((attribute, device), (expected, 0))
            ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = chip_id
            return 0

        lib.hipDeviceGetAttribute.side_effect = query
        props = SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-")
        with mock.patch.object(
            chip_info, "get_current_hip_device", return_value=0
        ), mock.patch.object(
            chip_info, "get_hip_runtime_version", return_value=version
        ), mock.patch.object(
            chip_info, "load_hip_runtime", return_value=lib
        ), mock.patch(
            "torch.cuda.get_device_properties", return_value=props
        ):
            return chip_info.get_device_name()

    def test_mi300a_follows_the_runtime_ordinal(self):
        self.assertEqual(self._device_name(MI300A_CHIP_ID, (7, 0, 0)), "MI300A")
        self.assertEqual(self._device_name(MI300A_CHIP_ID, (7, 1, 0)), "MI300A")
        self.assertEqual(self._device_name(MI308X_CHIP_ID, (7, 2, 0)), "MI308")

    def test_unknown_runtime_version_does_not_query_the_attribute(self):
        from aiter.jit.utils import chip_info

        lib = mock.Mock()
        props = SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-")
        with mock.patch.object(
            chip_info, "get_current_hip_device", return_value=0
        ), mock.patch.object(
            chip_info, "get_hip_runtime_version", return_value=None
        ), mock.patch.object(
            chip_info, "load_hip_runtime", return_value=lib
        ), mock.patch(
            "torch.cuda.get_device_properties", return_value=props
        ), self.assertRaisesRegex(
            RuntimeError, "PciChipId attribute"
        ):
            chip_info.get_device_name()
        lib.hipDeviceGetAttribute.assert_not_called()


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
    """GPU_ARCHS must compile the (gfx942, 228) MI300A tuned rows.

    gen_instances keeps only the tuned rows whose (gfx, cu_num) is a build
    target, so a missing 228 target leaves every MI300A row uncompiled and
    dispatch later fails with "not present in the compiled registry". The
    targets must not depend on the build host: a multi-arch wheel built on an
    MI300A still has to carry the 304-CU instances MI300X dispatches to.
    """

    def _targets(self, *, gpu_archs, cu_num=None):
        from aiter.jit.utils import chip_info

        with _EnvGuard(GPU_ARCHS=gpu_archs, CU_NUM=cu_num), mock.patch.object(
            chip_info, "get_cu_num", side_effect=AssertionError("probed live GPU")
        ):
            return chip_info.get_build_targets()

    def test_gfx942_covers_every_sku(self):
        targets = self._targets(gpu_archs="gfx942")
        for cu in (304, 228, 80):  # MI300X, MI300A, MI308X
            self.assertIn(("gfx942", cu), targets)

    def test_multi_arch_wheel_keeps_mi300x_and_mi300a(self):
        targets = self._targets(gpu_archs="gfx942;gfx950")
        self.assertIn(("gfx942", 304), targets)
        self.assertIn(("gfx942", 228), targets)
        self.assertIn(("gfx950", 256), targets)

    def test_explicit_cu_num_selects_mi300a_only(self):
        self.assertEqual(
            self._targets(gpu_archs="gfx942", cu_num="228"), [("gfx942", 228)]
        )

    def test_native_build_uses_live_cu_count(self):
        from aiter.jit.utils import chip_info

        with _EnvGuard(GPU_ARCHS=None, CU_NUM=None), mock.patch.object(
            chip_info, "get_gfx", return_value="gfx942"
        ), mock.patch.object(chip_info, "get_cu_num", return_value=228):
            self.assertEqual(chip_info.get_build_targets(), [("gfx942", 228)])


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

    def test_both_native_encodings_independent_of_host(self):
        # CI runs this suite on a single arch, which only exercises one
        # direction; pin gfx942 (fn -> fnuz) and gfx950 (fnuz -> fn) explicitly.
        import torch

        from aiter import dtypes

        flavors = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        for native in flavors:
            with self.subTest(native=native), mock.patch.object(dtypes, "fp8", native):
                for flavor in flavors:
                    self.assertIs(dtypes.normalize_fp8_dtype(flavor), native)

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
        import torch

        from aiter import dtypes

        if dtypes.fp8 not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            self.skipTest("arch has no native e4m3 (fp8 falls back to uint8)")
        self._assert_shipped_lists_resolve_to(str(dtypes.fp8))

    def test_shipped_lists_resolve_for_each_native_encoding(self):
        import torch

        from aiter import dtypes

        for native in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            with self.subTest(native=native), mock.patch.object(dtypes, "fp8", native):
                self._assert_shipped_lists_resolve_to(str(native))

    def _assert_shipped_lists_resolve_to(self, native):
        import csv

        from aiter.jit.core import AITER_ROOT_DIR
        from aiter.utility.base_tuner import TunerCommon

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

        for native in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            with self.subTest(native=native), mock.patch.object(dtypes, "fp8", native):
                # Leading/trailing whitespace must not defeat the match.
                self.assertEqual(
                    TunerCommon._resolve_fp8_name("  torch.float8_e4m3fn "),
                    str(native),
                )
                self.assertEqual(
                    TunerCommon._resolve_fp8_name("torch.float8_e4m3fnuz"),
                    str(native),
                )
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

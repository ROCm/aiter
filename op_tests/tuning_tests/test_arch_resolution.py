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

    def test_returns_none_when_hip_unavailable(self):
        from aiter.jit.utils import chip_info

        with mock.patch("ctypes.CDLL", side_effect=OSError("no libamdhip64")):
            self.assertIsNone(chip_info._get_pci_chip_id())

    def test_returns_none_when_query_fails(self):
        from aiter.jit.utils import chip_info

        lib = mock.MagicMock()
        lib.hipDeviceGetPCIBusId.return_value = 1  # non-zero == failure
        with mock.patch("ctypes.CDLL", return_value=lib):
            self.assertIsNone(chip_info._get_pci_chip_id())

    def test_reads_the_id_from_sysfs_via_the_bus_id(self):
        from aiter.jit.utils import chip_info

        def _fill_bus_id(buf, size, device_id):
            buf.value = b"0000:01:00.0"
            return 0

        lib = mock.MagicMock()
        lib.hipDeviceGetPCIBusId.side_effect = _fill_bus_id
        opener = mock.mock_open(read_data="0x74a0\n")

        with mock.patch("ctypes.CDLL", return_value=lib), mock.patch(
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
    SHAPE_LISTS = (
        "aiter/configs/untuned_fmoe.csv",
        "aiter/configs/a8w8_untuned_gemm.csv",
    )

    def test_normalization_makes_every_fp8_shape_arch_native(self):
        import csv

        import torch

        from aiter import dtypes
        from aiter.jit.core import AITER_ROOT_DIR

        if dtypes.fp8 not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            self.skipTest("arch has no native e4m3 (fp8 falls back to uint8)")
        foreign = (
            torch.float8_e4m3fn
            if dtypes.fp8 is torch.float8_e4m3fnuz
            else torch.float8_e4m3fnuz
        )

        checked = 0
        for rel in self.SHAPE_LISTS:
            path = os.path.join(AITER_ROOT_DIR, rel)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for row in csv.DictReader(f):
                    for col in self.DTYPE_COLUMNS:
                        raw = (row.get(col) or "").strip()
                        if not raw.startswith("torch.float8_e4m3"):
                            continue
                        resolved = dtypes.normalize_fp8_dtype(eval(raw))
                        self.assertIsNot(
                            resolved,
                            foreign,
                            f"{rel}:{col}={raw} still names the non-native e4m3 "
                            f"encoding after normalization",
                        )
                        self.assertIs(resolved, dtypes.fp8)
                        checked += 1
        self.assertGreater(checked, 0, "no fp8 shapes found to check")


class TestTunedConfigArchConsistency(unittest.TestCase):
    """Every tuned row keyed on cu_num=228 must also be keyed gfx942, or the
    build-time filter and the runtime lookup will disagree."""

    def test_cu228_rows_are_gfx942(self):
        import csv
        import glob

        from aiter.jit.core import AITER_ROOT_DIR

        cfg = os.path.join(AITER_ROOT_DIR, "aiter/configs")
        found = 0
        for path in glob.glob(os.path.join(cfg, "**", "*.csv"), recursive=True):
            if "untuned" in os.path.basename(path):
                continue
            try:
                with open(path) as f:
                    rows = list(csv.DictReader(f))
            except (OSError, UnicodeDecodeError):
                continue
            if not rows or "cu_num" not in rows[0]:
                continue
            for row in rows:
                if str(row.get("cu_num", "")).strip() != "228":
                    continue
                found += 1
                self.assertEqual(
                    (row.get("gfx") or "").strip(),
                    "gfx942",
                    f"{os.path.basename(path)}: cu_num=228 row is not gfx942",
                )
        self.assertGreater(found, 0, "no MI300A (cu_num=228) tuned rows found")


if __name__ == "__main__":
    unittest.main()

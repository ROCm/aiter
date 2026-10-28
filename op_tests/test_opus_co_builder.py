# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-only regression checks for CO pinning evidence (no torch/GPU needed)."""

import importlib.util
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

_BUILDER_PATH = (
    Path(__file__).resolve().parents[1] / "csrc/opus_gemm/gen_co/build_co.py"
)
_spec = importlib.util.spec_from_file_location("opus_co_builder", _BUILDER_PATH)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


class NamedRegisterEvidenceTests(unittest.TestCase):
    IR = """
      %r = tail call <8 x i32> @llvm.read_register.v8i32(metadata !12)
      call void @llvm.write_register.v8i32(metadata !12, <8 x i32> %r)
      declare <8 x i32> @llvm.read_register.v8i32(metadata)
      declare void @llvm.write_register.v8i32(metadata, <8 x i32>)
      !12 = !{!"v[256:263]"}
    """

    def test_elf_profiles_do_not_accept_a_renamed_wrong_target(self):
        def inspect(header, target):
            with mock.patch("builtins.open", return_value=BytesIO(header)):
                return builder._elf_target_metadata("kernel.co", target)

        for target, processor in (("gfx1250", 0x49), ("gfx1250-strict", 0xEB)):
            header = bytearray(64)
            header[:6] = b"\x7fELF\x02\x01"
            header[18:20] = (224).to_bytes(2, "little")
            header[48:52] = (processor | 0x200).to_bytes(4, "little")
            evidence = inspect(header, target)
            self.assertEqual(evidence["elf_processor_id"], processor)
            self.assertEqual(evidence["offload_arch"], target)
            other = "gfx1250" if target.endswith("strict") else "gfx1250-strict"
            with self.assertRaises(SystemExit):
                inspect(header, other)
        with self.assertRaises(SystemExit):
            inspect(b"not an ELF", "gfx1250-strict")

    def test_strict_default_path_is_a_flat_sibling(self):
        a0 = Path(builder._default_co_output_dir("gfx1250", "gfx1250-strict"))
        normal = Path(builder._default_co_output_dir("gfx1250", "gfx1250"))
        self.assertEqual(a0, _BUILDER_PATH.parent / "gfx1250-a0")
        self.assertEqual(normal, _BUILDER_PATH.parent / "named_reg" / "gfx1250")

    def test_strict_layout_guard_does_not_change_the_target(self):
        source = builder._STUB_TEMPLATE
        self.assertIn("defined(__gfx1250_strict__)", source)
        self.assertIn("#define __gfx1250__ 1", source)
        self.assertLess(
            source.index("#define __gfx1250__ 1"), source.index('#include "gfx1250/')
        )

    def test_counts_calls_not_declarations(self):
        self.assertEqual(
            builder._named_register_stats(self.IR),
            {
                "pinning_mode": "named-register",
                "ir_read_register": 1,
                "ir_write_register": 1,
                "named_vgpr_tuples": ["v[256:263]"],
            },
        )

    def test_missing_named_read_or_write_fails(self):
        for ir in (
            "",
            self.IR.replace("@llvm.read_register.", "@hint."),
            self.IR.replace("@llvm.write_register.", "@hint."),
            self.IR.replace('!"v[256:263]"', '!"a[256:263]"'),
        ):
            with self.subTest(ir=ir), self.assertRaises(SystemExit):
                builder._named_register_stats(ir)

    def test_explicit_unpinned_control_does_not_require_intrinsics(self):
        evidence = builder._named_register_stats("", require_pin=False)
        self.assertEqual(evidence["pinning_mode"], "unpinned-control")
        self.assertEqual(evidence["named_vgpr_tuples"], [])

    def test_no_pin_macro_uses_last_definition(self):
        self.assertFalse(builder._no_pin([]))
        self.assertTrue(builder._no_pin(["-DOPUS_NO_PIN"]))
        self.assertTrue(builder._no_pin(["-DOPUS_NO_PIN=1"]))
        self.assertFalse(builder._no_pin(["-DOPUS_NO_PIN=0"]))
        self.assertFalse(builder._no_pin(["-DOPUS_NO_PIN=1", "-DOPUS_NO_PIN=0"]))
        self.assertFalse(builder._no_pin(["-DOPUS_NO_PIN_A=1"]))
        self.assertFalse(builder._no_pin(["-DOPUS_NO_PIN=0x0"]))
        self.assertFalse(builder._no_pin(["-DOPUS_NO_PIN=false"]))
        self.assertFalse(builder._no_pin(["-DOPUS_NO_PIN=1", "-UOPUS_NO_PIN"]))
        self.assertTrue(builder._no_pin(["-DOPUS_NO_PIN=01U"]))
        with self.assertRaises(SystemExit):
            builder._no_pin(["-DOPUS_NO_PIN=(1-1)"])

    def test_every_expected_tuple_has_reads_and_writes(self):
        with self.assertRaises(SystemExit):
            builder._named_register_stats(self.IR, expected_tuples={"v[264:271]"})
        separate = (
            self.IR.replace(
                "@llvm.write_register.v8i32(metadata !12,",
                "@llvm.write_register.v8i32(metadata !13,",
            )
            + '\n!13 = !{!"v[264:271]"}\n'
        )
        with self.assertRaises(SystemExit):
            builder._named_register_stats(separate)

    def test_expected_fragment_banks(self):
        instance = builder.gfx1250_4wave_co_kernels_declared[496]
        tuples = builder._expected_named_tuples(instance, [])
        self.assertEqual(len(tuples), 32)
        self.assertIn("v[256:263]", tuples)
        self.assertIn("v[504:511]", tuples)

    def test_overlapping_experimental_a_c_banks_are_rejected(self):
        instance = builder.gfx1250_4wave_co_kernels_declared[715]
        with self.assertRaises(SystemExit):
            builder._expected_named_tuples(instance, ["-DOPUS_C_BF16F32=1"])

    def test_wmma_producer_must_use_the_named_accumulator_bank(self):
        instance = builder.gfx1250_4wave_co_kernels_declared[496]
        good = "v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[8:15]\n"
        result = builder._check_wmma_placement(good, instance, [])
        self.assertTrue(result["wmma_placement_checked"])
        self.assertEqual(result["wmma_dst_in_named_bank"], 1)
        for bad in (
            "",
            good.replace("/*v[256:263]*/", ""),
            good.replace("/*v[256:263]*/", "/*v[384:391]*/"),
        ):
            with self.subTest(asm=bad), self.assertRaises(SystemExit):
                builder._check_wmma_placement(bad, instance, [])

    def test_wmma_high_tuple_stats_follow_physical_comments(self):
        evidence = builder._asm_stats(
            "v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[8:15]\n"
            "v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[8:15]\n"
            "v_wmma_f32_16x16x32_bf16 v[8:15], v[16:23]\n"
            "s_wait_alu va_vdst(0)\n",
            "dis_",
        )
        self.assertEqual(evidence["dis_wmma"], 3)
        self.assertEqual(evidence["dis_wmma_dst_hi"], 2)
        self.assertEqual(evidence["dis_va_vdst0"], 1)


if __name__ == "__main__":
    unittest.main()

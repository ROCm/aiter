# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU regressions for pinned-register auditing and dependency cache upgrades."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_UTILS = Path(__file__).resolve().parents[1] / "aiter/jit/utils"


def load(name):
    spec = importlib.util.spec_from_file_location(name, _UTILS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load("hk_fp8_moe_codegen")
third_party = load("third_party")


def assembly(instructions, *, metadata_by_variant=None, **metadata):
    fields = {
        "agpr_count": "256",
        "vgpr_count": "512",
        "vgpr_spill_count": "0",
        "private_segment_fixed_size": "0",
        "uses_dynamic_stack": "false",
    } | metadata
    kernels, records = [], []
    for variant in range(4):
        name = f"_Z15moe_gemm_kernel_variant{variant}"
        kernels.append(f"{name}:\n{instructions}\n.Lfunc_end{variant}:\n")
        values = fields | (metadata_by_variant or {}).get(variant, {})
        record = "".join(
            f"    .{key}: {value}\n"
            for key, value in values.items()
            if value is not None
        )
        record += f"    .name: {name}\n"
        records.append("  - " + record[4:])
    return "".join(kernels) + "amdhsa.kernels:\n" + "".join(records)


class TestRegisterAudit(unittest.TestCase):
    def check(self, text):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "kernel.s"
            path.write_text(text)
            with patch("builtins.print"):
                audit.check_assembly(path)

    def test_known_tile_operations(self):
        self.check(assembly(""";;#ASMSTART
v_accvgpr_write_b32 a[0], 0
v_accvgpr_write_b32 a[1], v127
v_accvgpr_read_b32 v127, a[255]
ds_read_b128 v[0x80:0x83], v62 offset:0x800
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[192:199], v[128:135], a[0:3], v83, v0 op_sel:[1,0,0] op_sel_hi:[0,0,0]
v_cvt_pk_bf16_f32 v0, v1, v2
;;#ASMEND
v_writelane_b32 v127, s0, 0"""))

    def test_compiler_and_inline_temporaries(self):
        for instruction in (
            "v_mov_b32 v128, v0",
            "v_cvt_pk_bf16_f32 v200, v0, v1",
            "v_mov_b32 v[0x80], v0",
            "v_accvgpr_read_b32 v200, a0",
            "v_accvgpr_write_b32 a0, v200",
            "ds_read_b128 v[128:131], v200",
            "ds_read_b128 v[127:130], v0",
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[192:199], v[128:135], a[0:3], v128, v0",
        ):
            for inline in (False, True):
                with self.subTest(instruction=instruction, inline=inline):
                    text = (
                        f";;#ASMSTART\n{instruction}\n;;#ASMEND"
                        if inline
                        else instruction
                    )
                    with self.assertRaisesRegex(ValueError, "unsafe register/scratch"):
                        self.check(assembly(text))

    def test_tile_exemptions_require_inline_assembly(self):
        with self.assertRaisesRegex(ValueError, "unsafe register/scratch"):
            self.check(assembly("v_accvgpr_write_b32 a0, 0"))

    def test_scratch_instructions(self):
        for instruction in (
            "scratch_store_dword off, v0, s0",
            "s_mov_b32 flat_scratch_lo, s0",
        ):
            with self.subTest(instruction=instruction), self.assertRaisesRegex(
                ValueError, "unsafe register/scratch"
            ):
                self.check(assembly(instruction))

    def test_missing_or_unsafe_metadata(self):
        for key, value in (
            ("vgpr_spill_count", "1"),
            ("private_segment_fixed_size", "1024"),
            ("uses_dynamic_stack", "true"),
        ):
            for actual in (None, value):
                with self.subTest(key=key, value=actual), self.assertRaisesRegex(
                    ValueError, key
                ):
                    self.check(assembly("v_mov_b32 v0, 0", **{key: actual}))

    def test_missing_or_unsafe_register_allocation(self):
        for key, expected, smaller in (
            ("agpr_count", "256", "128"),
            ("vgpr_count", "512", "256"),
        ):
            for actual in (None, smaller, "invalid", expected + " trailing"):
                for variant in range(4):
                    with (
                        self.subTest(key=key, value=actual, variant=variant),
                        self.assertRaisesRegex(ValueError, key),
                    ):
                        self.check(
                            assembly(
                                "v_mov_b32 v0, 0",
                                metadata_by_variant={variant: {key: actual}},
                            )
                        )

    def test_allocation_fields_must_be_unique_and_kernel_level(self):
        for key, expected in (("agpr_count", "256"), ("vgpr_count", "512")):
            text = assembly("v_mov_b32 v0, 0")
            text = text.replace(
                f".{key}: {expected}\n",
                f".{key}: {expected}\n    .{key}: {expected}\n",
                1,
            )
            with (
                self.subTest(key=key, kind="duplicate"),
                self.assertRaisesRegex(ValueError, key),
            ):
                self.check(text)
            text = assembly("v_mov_b32 v0, 0", **{key: None})
            text = text.replace(
                "    .name:",
                f"    .args:\n      - .{key}: {expected}\n    .name:",
            )
            with (
                self.subTest(key=key, kind="nested"),
                self.assertRaisesRegex(ValueError, key),
            ):
                self.check(text)

    def test_kernel_metadata_name_must_be_unique(self):
        text = assembly("v_mov_b32 v0, 0")
        with self.assertRaisesRegex(ValueError, "duplicate kernel metadata"):
            self.check(text + "  - " + text.split("  - ", 1)[1].split("  - ")[0])

    def test_nested_argument_name_is_ignored(self):
        text = assembly("v_mov_b32 v0, 0")
        self.check(
            text.replace("    .name:", "    .args:\n      - .name: input\n    .name:")
        )

    def test_wrong_kernel_count_and_malformed_markers(self):
        for text in (
            "",
            assembly(";;#ASMSTART"),
            assembly(";;#ASMEND"),
            assembly(";;#ASMSTART\n;;#ASMSTART"),
        ):
            with self.subTest(text=text[:50]), self.assertRaises(ValueError):
                self.check(text)


class TestDependencyCheckout(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote, self.cache = self.root / "remote", self.root / "cache"
        self.git("init", "-q", str(self.remote))
        self.git("-C", str(self.remote), "config", "user.name", "test")
        self.git("-C", str(self.remote), "config", "user.email", "test@example.invalid")
        self.old = self.commit("old")
        self.new = self.commit("new")

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    def commit(self, content):
        (self.remote / "header").write_text(content)
        self.git("-C", str(self.remote), "add", "header")
        self.git("-C", str(self.remote), "commit", "-qm", content)
        return self.git("-C", str(self.remote), "rev-parse", "HEAD")

    def ensure(self, revision, **kwargs):
        third_party.ensure_pinned_checkout(
            self.cache, str(self.remote), revision, **kwargs
        )

    def test_fresh_clone_and_upgrade_shallow_cache(self):
        self.ensure(self.old)
        self.assertEqual((self.cache / "header").read_text(), "old")
        self.ensure(self.new)
        self.assertEqual(self.git("-C", str(self.cache), "rev-parse", "HEAD"), self.new)
        self.assertEqual((self.cache / "header").read_text(), "new")
        # A cache hit must work even with the remote unavailable.
        third_party.ensure_pinned_checkout(self.cache, "/nonexistent", self.new)

    def test_dirty_cache_is_preserved(self):
        self.ensure(self.old)
        (self.cache / "header").write_text("local changes")
        with self.assertRaisesRegex(RuntimeError, "modified dependency"):
            self.ensure(self.new)
        self.assertEqual((self.cache / "header").read_text(), "local changes")
        self.assertEqual(self.git("-C", str(self.cache), "rev-parse", "HEAD"), self.old)

    def test_user_override_is_preserved_without_git(self):
        self.cache.mkdir()
        (self.cache / "header").write_text("custom headers")
        self.ensure(self.new, user_override=True)
        self.assertEqual((self.cache / "header").read_text(), "custom headers")

    def test_source_archive_never_checks_out_parent_repository(self):
        archive = self.remote / "HipKittens"
        archive.mkdir()
        (archive / "header").write_text("bundled headers")
        with self.assertRaisesRegex(RuntimeError, "Not a managed dependency"):
            third_party.ensure_pinned_checkout(archive, str(self.remote), self.old)
        self.assertEqual(
            self.git("-C", str(self.remote), "rev-parse", "HEAD"), self.new
        )
        resolved = third_party.checkout_path(archive, self.root / "build")
        self.assertNotEqual(Path(resolved), archive)
        third_party.ensure_pinned_checkout(resolved, str(self.remote), self.old)
        self.assertEqual((Path(resolved) / "header").read_text(), "old")
        self.assertEqual((archive / "header").read_text(), "bundled headers")
        self.assertEqual(
            third_party.checkout_path(archive, self.root, user_override=True),
            str(archive),
        )

    def test_missing_user_override(self):
        with self.assertRaisesRegex(RuntimeError, "override does not exist"):
            self.ensure(self.new, user_override=True)

    def test_failed_fetch_does_not_publish_cache(self):
        with self.assertRaisesRegex(RuntimeError, "HIP_KITTENS_DIR"):
            self.ensure("0" * 40)
        self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()

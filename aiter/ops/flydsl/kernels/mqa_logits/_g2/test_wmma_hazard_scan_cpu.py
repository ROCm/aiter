# SPDX-License-Identifier: MIT
"""ROUND-W64FIX2 CPU gate: self-test of ``wmma_hazard_scan_cpu.py`` and source guard.

The ISA gate (container_w64fix2_cpu.sh isa) runs the scanner with zero tolerance
on the real no-device ISA of every default kernel; this gate pins the scanner's
semantics (kernel-wiki mi400-wmma-hazards §1.2 B1/A1) on synthetic ISA so a
scanner regression cannot silently pass that gate, and checks that the kernel
source has no inline asm outside the opt-out converter.

    python3 -B test_wmma_hazard_scan_cpu.py -v
"""
import ast
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from wmma_hazard_scan_cpu import main, scan  # noqa: E402

KERNEL = os.path.join(os.path.dirname(_HERE), "qsa_paged_mqa_logits_page4_gfx1250.py")
WMMA = "v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15], v[16:23], v[0:7]"


def _scan(lines):
    with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as fh:
        fh.write("\n".join(["kernel:", *lines, "s_endpgm", ""]))
        path = fh.name
    return scan(path), path


class ScannerSemantics(unittest.TestCase):
    def check(self, lines, kinds, asm=None):
        out, path = _scan(lines)
        os.unlink(path)
        self.assertEqual([d["kind"] for d in out["detail"]], kinds, out["detail"])
        if asm is not None:
            self.assertEqual(out["asm_violations"], asm)
        return out

    def test_war_on_a_and_b(self):
        self.check([WMMA, "v_mov_b32 v8, 0"], ["B1:WAR-vdst-on-AB"])
        self.check([WMMA, "v_mov_b32 v23, 0"], ["B1:WAR-vdst-on-AB"])

    def test_waw_and_raw_on_d(self):
        self.check([WMMA, "v_mov_b32 v3, 0"], ["B1:WAW-vdst-on-D"])
        self.check([WMMA, "v_add_f32 v40, v7, v1"], ["B1:RAW-src-on-D"])

    def test_window_four_valu(self):
        pad = [f"v_mov_b32 v{50 + i}, 0" for i in range(4)]
        self.check([WMMA, *pad[:3], "v_mov_b32 v8, 0"], ["B1:WAR-vdst-on-AB"])
        self.check([WMMA, *pad, "v_mov_b32 v8, 0"], [])
        self.check([WMMA, "v_nop", "v_nop", "v_nop", "v_nop", "v_mov_b32 v8, 0"], [])

    def test_salu_does_not_pad(self):
        self.check([WMMA, "s_nop 7", "s_mov_b32 s0, 0", "v_mov_b32 v8, 0"], ["B1:WAR-vdst-on-AB"])

    def test_independent_is_clean(self):
        self.check([WMMA, "v_mov_b32 v40, v41", "v_perm_b32 v42, v43, v43, 0x1010000"], [])

    def test_inline_asm_is_tagged(self):
        out = self.check([WMMA, ";;#ASMSTART", "v_perm_b32 v14, v30, v30, 0x1010000",
                          ";;#ASMEND"], ["B1:WAR-vdst-on-AB"], asm=1)
        self.assertEqual(out["compiler_violations"], 0)

    def test_wmma_raw_on_previous_d(self):
        nxt = "v_wmma_f32_16x16x32_bf16 v[24:31], v[0:7], v[16:23], v[24:31]"
        self.check([WMMA, nxt], ["A1:wmma-AB-RAW-on-D"])
        self.check([WMMA, *[f"v_mov_b32 v{50 + i}, 0" for i in range(5)], nxt], [])

    def test_branch_ends_window(self):
        self.check([WMMA, "s_cbranch_scc1 .LBB0_2", "v_mov_b32 v8, 0"], [])

    def test_fail_on_any_exit_code(self):
        _, path = _scan([WMMA, "v_mov_b32 v8, 0"])
        try:
            self.assertEqual(main([path, "--fail-on-any"]), 1)
            self.assertEqual(main([path]), 0)
        finally:
            os.unlink(path)


class SourceGuard(unittest.TestCase):
    def test_inline_asm_only_in_opt_out_converter(self):
        with open(KERNEL, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        users = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)
                 and "inline_asm" in ast.unparse(n)}
        self.assertEqual(users, {"_e4m3x8_to_bf16x4_asm"})


if __name__ == "__main__":
    unittest.main()

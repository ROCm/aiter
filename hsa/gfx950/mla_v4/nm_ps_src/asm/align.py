# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Alignment-preserving pads for script-patched copies of rt.s. Two never-executed s_nop pads (@@PADA@@ before the
loop code, @@PADB@@ after the epilogue; each placed after an unconditional branch or s_endpgm) absorb inserted bytes so
the hot code keeps its original byte offset mod 256: loop copies label_1F54/3D74/5B68, drain label_795C, tail tile
label_8EFC, rescale subroutine label_9DB4. Offsets are read from the symbol table of an assembled pass.
"""

import re
import subprocess
import tempfile

A = 256
PRE = ("label_1F54", "label_3D74", "label_5B68", "label_795C")
POST = ("label_8EFC", "label_9DB4")


def offsets(text, clang, readelf):
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/x.s", "w").write(text)
        subprocess.run(
            [
                clang,
                "-x",
                "assembler",
                "-target",
                "amdgcn-amd-amdhsa",
                "-mcpu=gfx950",
                "-c",
                f"{d}/x.s",
                "-o",
                f"{d}/x.o",
            ],
            check=True,
        )
        r = subprocess.run(
            [readelf, "-s", f"{d}/x.o"], capture_output=True, text=True, check=True
        ).stdout
    return {
        m[2]: int(m[1], 16)
        for m in re.finditer(
            r"^\s*\d+:\s+([0-9a-f]+)\s+0\s+NOTYPE\s+LOCAL\s+DEFAULT\s+\d+\s+(label_\w+)",
            r,
            re.M,
        )
    }


def render(body, pads):
    for k, n in pads.items():
        body = body.replace(k, "\n".join(["\ts_nop 0"] * (n // 4)) if n else "")
    return body


def align(src_text, body, clang, readelf):
    pads = {"@@PADA@@": 0, "@@PADB@@": 0}
    orig = offsets(src_text, clang, readelf)
    o = offsets(render(body, pads), clang, readelf)
    pads["@@PADA@@"] = (-(o["label_1F54"] - orig["label_1F54"])) % A
    o = offsets(render(body, pads), clang, readelf)
    pads["@@PADB@@"] = (-(o["label_8EFC"] - orig["label_8EFC"])) % A
    o = offsets(render(body, pads), clang, readelf)
    for lab in PRE + POST:
        assert (o[lab] - orig[lab]) % A == 0, (lab, o[lab], orig[lab])
    return render(body, pads), pads

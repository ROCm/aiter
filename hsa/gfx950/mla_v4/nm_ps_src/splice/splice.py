# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compile a HIP splice source (splice/*.hip) to gfx950 assembly with `hipcc -S` and extract the body of one kernel
as a text block that asm/patch.py inlines into the patched ASM.

Extraction rules (checked here):
  - the kernel ABI must be: user SGPRs = kernarg pointer only (s[0:1]), workgroup id x only (s2), workitem id x (v0),
    no kernarg preload, no scratch, no LDS (group segment 0) unless allow_lds;
  - register budget: next_free_vgpr <= vmax, next_free_sgpr <= smax (the wrapper keeps its state above these);
  - labels .LBBk_n -> <PFX>_n; every s_endpgm -> s_branch <PFX>_RET (the caller places <PFX>_RET after the block);
  - assembler directives and comments are dropped.
"""

import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
HIP_FLAGS = [
    "--offload-arch=gfx950",
    "-O3",
    "--offload-device-only",
    "-std=c++20",
    "-w",
]


def compile_s(hipcc, src, out, extra=()):
    subprocess.run(
        [hipcc, *HIP_FLAGS, *extra, "-S", src, "-o", out],
        check=True,
        cwd=os.path.dirname(src),
    )
    return out


def body(
    hipcc, src, kernel, prefix, workdir, vmax=240, smax=100, allow_lds=False, extra=()
):
    """-> (block text, kernel descriptor fields {name: int})"""
    out_s = os.path.join(workdir, os.path.basename(src)[:-4] + ".s")
    s = open(compile_s(hipcc, src, out_s, extra)).read().split("\n")
    i0 = next(i for i, ln in enumerate(s) if ln.split(";")[0].strip() == f"{kernel}:")
    i1 = next(i for i in range(i0, len(s)) if s[i].startswith(".Lfunc_end"))
    # kernel descriptor block
    k0 = next(
        i for i in range(i0, len(s)) if s[i].split() == [".amdhsa_kernel", kernel]
    )
    k1 = next(i for i in range(k0, len(s)) if s[i].strip() == ".end_amdhsa_kernel")
    kd = {}
    for ln in s[k0 + 1 : k1]:
        t = ln.split()
        if len(t) == 2 and t[0].startswith(".amdhsa_"):
            kd[t[0][8:]] = int(t[1])
    req = dict(
        user_sgpr_count=2,
        user_sgpr_kernarg_segment_ptr=1,
        user_sgpr_dispatch_ptr=0,
        user_sgpr_queue_ptr=0,
        user_sgpr_kernarg_preload_length=0,
        system_sgpr_workgroup_id_y=0,
        system_sgpr_workgroup_id_z=0,
        system_vgpr_workitem_id=0,
        private_segment_fixed_size=0,
        enable_private_segment=0,
    )
    for k, v in req.items():
        assert kd.get(k, v) == v, (kernel, k, kd.get(k))
    if not allow_lds:
        assert kd["group_segment_fixed_size"] == 0, kd["group_segment_fixed_size"]
    assert kd["next_free_vgpr"] <= vmax and kd["next_free_sgpr"] <= smax, (
        kernel,
        kd["next_free_vgpr"],
        kd["next_free_sgpr"],
    )
    out = []
    for ln in s[i0 + 1 : min(i1, k0)]:
        t = ln.split(";")[0].rstrip()
        if not t.strip() or t.strip().startswith("."):
            m = re.match(r"^\.LBB\d+_(\d+):", t)
            if m:
                out.append(f"{prefix}_{m.group(1)}:")
            continue
        t = re.sub(r"\.LBB\d+_(\d+)", lambda m: f"{prefix}_{m.group(1)}", t)
        if t.strip() == "s_endpgm":
            t = f"\ts_branch {prefix}_RET"
        out.append(t)
    # every register named must stay inside the budget (inline-asm AGPR outputs are allowed)
    for ln in out:
        for m in re.finditer(r"\bv\[?(\d+)(?::(\d+))?", ln):
            assert int(m.group(2) or m.group(1)) < vmax, ln
        for m in re.finditer(r"\bs\[?(\d+)(?::(\d+))?", ln):
            if re.match(r"\s*s_", ln) or "," in ln:
                assert int(m.group(2) or m.group(1)) < smax, ln
    return "\n".join(out), kd

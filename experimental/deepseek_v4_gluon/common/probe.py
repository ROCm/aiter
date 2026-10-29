# SPDX-License-Identifier: MIT
"""GPU-free compile of a Gluon kernel for gfx950: resources and the ISA.

A persistent grid must hold every CTA resident and spill nothing (a spill in a
spinning kernel is a correctness risk, not just a slowdown). This compiles
without a device, so it runs on a box whose GPUs are busy."""

import re

import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource


def compile_probe(fn, signature: dict, constexprs: dict | None = None,
                  num_warps: int = 8, arch: str = "gfx950", **options) -> dict:  # fmt: skip
    """``signature``: arg name -> Triton type ("i64", "i32", "*fp32", "constexpr");
    ``constexprs``: arg name -> value."""
    # by the kernel's own parameter order (a dict in another order would put
    # each constexpr at the wrong position)
    names = list(fn.arg_names)
    missing = set(names) - set(signature)
    assert not missing and set(signature) <= set(names), (missing, set(signature) - set(names))
    signature = {n: signature[n] for n in names}
    consts = {(names.index(k),): v for k, v in (constexprs or {}).items()}
    # addresses (i64) and pointers are 16-byte aligned, as the JIT specializes
    # them at launch: without it the probe shows byte-split loads a real launch
    # does not have
    attrs = {(i,): [["tt.divisibility", 16]] for i, n in enumerate(names)
             if signature[n] == "i64" or signature[n].startswith("*")}
    src = GluonASTSource(fn=fn, signature=signature, constexprs=consts, attrs=attrs)
    k = triton.compile(
        src,
        target=GPUTarget("hip", arch, 64),
        options={"num_warps": num_warps, **options},
    )
    asm = k.asm["amdgcn"]

    def field(name):
        m = re.search(rf"\.{name}:\s*(\d+)", asm)
        return int(m[1]) if m else None

    return {
        "vgpr": field("vgpr_count"),
        "sgpr": field("sgpr_count"),
        "spills": field("vgpr_spill_count"),
        "lds": k.metadata.shared,
        "asm": asm,
        "llir": k.asm["llir"],
    }

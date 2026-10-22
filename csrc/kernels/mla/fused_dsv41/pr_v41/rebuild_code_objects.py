#!/usr/bin/env python3
"""Rebuild PR V4.1 fused kernels with the pinned-register AMDGPU LLVM compiler.

Set HIP_CLANG_PATH to the bin directory of llvm-project commit
583640bc544035fcc87177fee0410e0d87693919 (amdgpu-named-reg-pin-std).
Ordinary ROCm builds only consume the checked-in code objects.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, help="defaults to this repository hsa/gfx950/fused_dsv41"
    )
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[4]
    out = args.output or repo / "hsa/gfx950/fused_dsv41"
    out.mkdir(parents=True, exist_ok=True)
    llvm = Path(os.environ["HIP_CLANG_PATH"])
    assert (
        llvm / "clang++"
    ).is_file(), "HIP_CLANG_PATH must contain the custom clang++"
    metadata = {
        "llvm_commit": "583640bc544035fcc87177fee0410e0d87693919",
        "variants": {},
    }
    for tag in [
        "legacy_check_raw",
        "legacy_check_prepared",
        "paged128_check_raw",
        "paged128_check_prepared",
    ]:
        src = here / "mla_v41_prefill_fused.cu"
        co = out / (tag + ".co")
        flags = [
            "-DCACHE_MODE=" + str(1 if tag.startswith("legacy") else 2),
            "-DRAW_Q=" + str(int(tag.endswith("raw"))),
            "-DFUSE_OUTPUT=1",
            "-std=c++20",
            "-O3",
            "--offload-arch=gfx950",
            "-D__HIPCC_RTC__",
            "-ffast-math",
            "-fgpu-flush-denormals-to-zero",
            "-fno-offload-uniform-block",
            "-mllvm",
            "--amdgpu-kernarg-preload-count=32",
            "-mllvm",
            "--lsr-drop-solution=1",
            "-mllvm",
            "-amdgpu-early-inline-all=false",
            "-mllvm",
            "-amdgpu-function-calls=true",
            "-mllvm",
            "-enable-post-misched=1",
            "-Rpass-analysis=kernel-resource-usage",
        ]
        with (out / (tag + ".build.log")).open("w") as f:
            subprocess.run(
                [
                    "/opt/rocm/bin/hipcc",
                    "-x",
                    "hip",
                    str(src),
                    "-I" + str(here),
                    "-I" + str(repo / "csrc/include"),
                    *flags,
                    "--genco",
                    "-o",
                    str(co),
                ],
                check=True,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        metadata["variants"][tag] = {
            "sha256": hashlib.sha256(co.read_bytes()).hexdigest(),
            "flags": flags,
        }
        print("BUILT", tag, flush=True)
    (out / "build_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

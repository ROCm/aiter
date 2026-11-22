#!/usr/bin/env python3
"""Check gfx950 codegen for the intermediate custom all-reduce barrier.

Run with ROCm hipcc on PATH. This compiles device assembly without using a GPU.
Pass --header to compare the same check against a different AITER revision.
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path


def probe_source(header: str) -> str:
    # Keep the real barrier and signal layouts while avoiding unrelated
    # collective kernels and their build dependencies.
    layouts = [
        re.search(
            r"^struct (?:__align__\(16\) )?" + name + r"\b.*?^};",
            header,
            re.MULTILINE | re.DOTALL,
        ).group()
        for name in ("Signal", "RankSignals")
    ]
    begin = header.index("template <int ngpus>\nDINLINE void start_sync")
    end_begin = header.index("DINLINE void end_sync", begin)
    end = header.index("\ntemplate <", end_begin)
    return (
        "#include <hip/hip_runtime.h>\n"
        "#include <cstdint>\n"
        "#define USE_ROCM\n"
        "#define DINLINE __device__ __forceinline__\n"
        "constexpr int kMaxBlocks = 80;\n"
        + "\n".join(layouts)
        + "\n"
        + header[begin:end]
        + """
extern "C" __global__ void probe(RankSignals peers, Signal* self, int rank) {
    end_sync<4, false>(peers, self, rank);
}
extern "C" __global__ void probe_start(RankSignals peers, Signal* self, int rank) {
    start_sync<4>(peers, self, rank);
}
extern "C" __global__ void probe_final(RankSignals peers, Signal* self, int rank) {
    end_sync<4, true>(peers, self, rank);
}
"""
    )


def check_assembly(assembly: str) -> None:
    # hipcc labels each basic block. Follow the polling branch back to the
    # block that loads the flag, then check for an invalidate in that block.
    middle = kernel_assembly(assembly, "probe")
    blocks = re.split(r"^(\.L[\w.]+):[^\n]*\n", middle, flags=re.MULTILINE)
    for label, block in zip(blocks[1::2], blocks[2::2]):
        back_edge = re.search(
            rf"^\s*s_cbranch_execnz\s+{re.escape(label)}\b", block, re.MULTILINE
        )
        if "global_load" not in block or back_edge is None:
            continue
        loop = block[: back_edge.end()]
        if "buffer_inv" in loop:
            raise SystemExit(
                f"FAIL: cache invalidation in the flag polling loop\n{label}:\n{loop}"
            )
        break
    else:
        raise SystemExit("Could not identify the flag polling loop in hipcc output")

    invalidates = re.findall(r"^\s*buffer_inv\s+([^\n]+)", middle, re.MULTILINE)
    if [instruction.strip() for instruction in invalidates] != ["sc0 sc1"]:
        raise SystemExit("FAIL: expected exactly one system-scope invalidate")
    for kernel in ("probe_start", "probe_final"):
        if "buffer_inv" in kernel_assembly(assembly, kernel):
            raise SystemExit(f"FAIL: unexpected cache invalidation in {kernel}")
    print("PASS: one invalidate after the intermediate wait; none in start or final")


def kernel_assembly(assembly: str, name: str) -> str:
    return assembly.split(f"\n{name}:")[1].split("\n.Lfunc_end", 1)[0]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--header", type=Path, default=root / "csrc/include/custom_all_reduce.cuh"
    )
    parser.add_argument("--hipcc", default="hipcc")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        source, assembly = Path(tmp) / "probe.hip", Path(tmp) / "probe.s"
        source.write_text(probe_source(args.header.read_text()))
        subprocess.run(
            [
                args.hipcc,
                "-x",
                "hip",
                "--offload-arch=gfx950",
                "--cuda-device-only",
                "-std=c++17",
                "-O3",
                "-S",
                str(source),
                "-o",
                str(assembly),
            ],
            check=True,
        )
        check_assembly(assembly.read_text())


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT
"""Build the gfx950 latent MoE backward images from the pinned OPUS grouped kernels."""
import argparse
import hashlib
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

import yaml

HERE = Path(__file__).resolve().parent
PREFIX = "opus_gemm_gfx950_bf16_"
TILE = "256x256x256x64_2x2_16x16x32_0x0x0"
NN = PREFIX + "m_grouped_" + TILE + "_nn"
TN = PREFIX + "k_grouped_" + TILE + "_tn_f32"
HASHES = {
    NN: "f459eeef16515a0d495737d93e357539b3945331c4c5f5af0280000947b97da1",
    TN: "5ae0df53077174254045afe4e385261d26ef3d16b857ba0c734d0abce97f83cb",
}


def at_address(elf, address, size):
    offset = struct.unpack_from("<Q", elf, 40)[0]
    stride, count = struct.unpack_from("<HH", elf, 58)
    for index in range(count):
        flags, base, start, length = struct.unpack_from(
            "<QQQQ", elf, offset + index * stride + 8
        )
        if flags & 2 and base <= address and address + size <= base + length:
            return elf[start + address - base : start + address - base + size]
    raise ValueError(f"no mapped section for {address:#x}")


def assemble(llvm, image, name, suffix, source, out, work):
    elf = image.read_bytes()
    symbols = subprocess.check_output(
        [llvm / "llvm-readelf", "--symbols", image], text=True
    )
    addresses = {
        r[7]: int(r[1], 16)
        for line in symbols.splitlines()
        if len(r := line.split()) == 8 and r[3] in ("OBJECT", "FUNC")
    }
    kd = bytearray(at_address(elf, addresses[name + ".kd"], 64))
    if suffix in ("_descriptor", "_stream"):
        kd[53] |= 1  # Enable workgroup_id_y in COMPUTE_PGM_RSRC2.
    body_size = 50272 if name == NN else 0x5000
    (work / "body.bin").write_bytes(at_address(elf, addresses[name], body_size))
    notes = subprocess.check_output(
        [llvm / "llvm-readelf", "--notes", image], text=True
    )
    metadata = yaml.safe_load(notes[notes.index("---") : notes.index("...") + 3])
    (kernel,) = (item for item in metadata["amdhsa.kernels"] if item[".name"] == name)
    symbol = name + suffix
    kernel.update({".name": symbol, ".symbol": symbol + ".kd"})
    metadata["amdhsa.kernels"] = [kernel]
    signed = lambda seq: ",".join(map(str, seq))
    constants = ""
    if name == NN:
        # The frozen MFMA body reads these mixed tile heights via s_getpc_b64.
        # Keep their displacement from the kernel entry when relinking the image.
        heights = at_address(elf, addresses[name] - 0x1140, 12)
        if struct.unpack("<3I", heights) != (256, 192, 128):
            raise ValueError("unexpected M-grouped tile height table")
        constants = ".zero 192\n.long 256, 192, 128\n.zero 0x1140-12"
    assembly = f""".text
.p2align 8
{constants}
.globl {symbol}
.protected {symbol}
.type {symbol},@function
{symbol}:
.include "{source}"
.size {symbol}, .-{symbol}
.section .rodata,"a",@progbits
.p2align 6
.globl {symbol}.kd
.protected {symbol}.kd
.type {symbol}.kd,@object
{symbol}.kd:
.byte {signed(kd[:16])}
.quad {symbol} - {symbol}.kd
.byte {signed(kd[24:])}
.size {symbol}.kd,64
.amdgpu_metadata
{yaml.safe_dump(metadata,sort_keys=False,explicit_start=True,explicit_end=True)}
.end_amdgpu_metadata
"""
    (work / "kernel.s").write_text(assembly)
    subprocess.run(
        [
            llvm / "llvm-mc",
            "-triple=amdgcn-amd-amdhsa",
            "-mcpu=gfx950",
            "--amdhsa-code-object-version=6",
            "-filetype=obj",
            "kernel.s",
            "-o",
            "kernel.o",
        ],
        cwd=work,
        check=True,
    )
    subprocess.run(
        [llvm / "ld.lld", "-shared", "kernel.o", "-o", out], cwd=work, check=True
    )


def build(csrc, rocm, output):
    llvm = rocm / "llvm/bin"
    base = csrc / "opus_gemm"
    arch = base / "include/gfx950"
    stock = base / "gen_co/gfx950/a16w16_grouped"
    for name, digest in HASHES.items():
        if hashlib.sha256((stock / (name + ".co")).read_bytes()).hexdigest() != digest:
            raise ValueError(f"unexpected upstream code object for {name}")

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name, suffix, asm in [
            (NN, "_compact", "dgrad_compact.s"),
            (NN, "_descriptor", "dgrad_descriptor.s"),
            (TN, "_stream", "wgrad_stream.s"),
        ]:
            assemble(
                llvm,
                stock / (name + ".co"),
                name,
                suffix,
                HERE / asm,
                output / (name + suffix + ".co"),
                work,
            )
        shutil.copyfile(stock / (TN + ".co"), output / (TN + ".co"))
        shutil.copyfile(
            arch / "opus_gemm_pipeline_a16w16_grouped_gfx950.cuh",
            work / "short_pipeline.cuh",
        )
        subprocess.run(
            ["patch", "--batch", "-p1", "-i", HERE / "short_pipeline.patch"],
            cwd=work,
            check=True,
        )
        for symbol, source, flags in [
            (
                TN + "_shortk",
                "shortk.hip",
                [
                    "-D__HIPCC_RTC__",
                    "-DOPUS_ENABLE_RUNTIME_QUERY=0",
                    "-I" + str(csrc / "include"),
                    "-I" + str(base / "include"),
                    "-I" + str(arch),
                    "-I" + str(work),
                ],
            ),
            (PREFIX + "latent_moe_bwd_plan", "plan.hip", ["-DCOMPACT_PLAN"]),
            (NN + "_descriptor_plan", "plan.hip", ["-DDESCRIPTOR_PLAN"]),
            (TN + "_sort_plan", "plan.hip", ["-DSORT_PLAN"]),
            (PREFIX + "latent_moe_bwd_scalar_nn", "scalar.hip", ["-DDGRAD"]),
            (PREFIX + "latent_moe_bwd_scalar_tn_f32", "scalar.hip", []),
        ]:
            bundle = work / "kernel.bundle"
            subprocess.run(
                [
                    rocm / "bin/hipcc",
                    "-x",
                    "hip",
                    "--genco",
                    "--offload-arch=gfx950",
                    "-std=c++20",
                    "-O3",
                    *flags,
                    HERE / source,
                    "-o",
                    bundle,
                ],
                check=True,
            )
            subprocess.run(
                [
                    llvm / "clang-offload-bundler",
                    "--unbundle",
                    "--type=o",
                    "--targets=hipv4-amdgcn-amd-amdhsa--gfx950",
                    "--input=" + str(bundle),
                    "--output=" + str(output / (symbol + ".co")),
                ],
                check=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "upstream_csrc",
        type=Path,
        help="csrc of ROCm/aiter revision 1221392f761b2a3da80b33be7bb4a3714f69655a",
    )
    parser.add_argument("--rocm", type=Path, default=Path("/opt/rocm"))
    parser.add_argument(
        "--out", type=Path, default=HERE.parent / "gfx950/a16w16_grouped"
    )
    args = parser.parse_args()
    build(args.upstream_csrc.resolve(), args.rocm.resolve(), args.out.resolve())
